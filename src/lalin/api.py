"""FastAPI surface over the pipeline: submit a run, poll it, fetch artifacts."""

from __future__ import annotations

import shutil
import sys
import threading
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import aduan, batch, config, models, runner, store
from .langsung import SharedDual, langsung
from .jobs import registry
from .settings import settings

app = FastAPI(title="PP-Vehicle", version="0.1.0",
              description="Wrapper around PaddleDetection PP-Vehicle pipeline")

WEB_DIR = Path(__file__).parent / "web"


@app.on_event("startup")
def _mulai_analisis_berkala() -> None:
    """Nyalakan penjadwal begitu server hidup.

    Penundaan awal disengaja: menarik klip dari lima kamera sekaligus saat
    proses baru start bertabrakan dengan permintaan pertama dari dashboard, dan
    portal publik menutup koneksi berlebih - itulah sumber `Connection timed
    out` yang berulang saat pengujian.
    """
    detik = settings.auto_batch_sec
    if detik <= 0:
        return
    # Mode POC (configs/mode_poc.yml): siklus pelanggaran dijeda.
    if _mode_poc().get("jeda_siklus"):
        return

    def tunda() -> None:
        import time
        time.sleep(20)
        try:
            batch.worker.start(interval_sec=detik)
        except Exception:
            pass

    threading.Thread(target=tunda, daemon=True).start()


# --------------------------------------------------------------------- meta

@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "profile": settings.profile}


@app.get("/api/doctor")
def doctor() -> dict:
    """Everything that commonly blocks a first run, in one payload."""
    try:
        import paddle  # noqa: F401
        paddle_version = paddle.__version__
        try:
            compiled_cuda = bool(paddle.device.is_compiled_with_cuda())
        except Exception:
            compiled_cuda = False
    except Exception as exc:
        paddle_version = f"NOT INSTALLED ({type(exc).__name__})"
        compiled_cuda = False

    return {
        "python": sys.version.split()[0],
        "paddle": paddle_version,
        "paddle_compiled_with_cuda": compiled_cuda,
        "paddledetection": {
            "path": str(settings.paddledet),
            "present": settings.pipeline_script.exists(),
        },
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "default_profile": settings.profile,
        "profiles": config.list_profiles(),
        "models": models.status(),
    }


@app.get("/api/profiles")
def profiles() -> dict:
    return {"default": settings.profile,
            "profiles": {p: config.load_profile(p) for p in config.list_profiles()}}


@app.get("/api/scenarios")
def scenarios() -> list[dict]:
    return config.list_scenarios()


@app.get("/api/models")
def list_models() -> list[dict]:
    return models.status()


@app.post("/api/models/download")
def download_models(keys: list[str] | None = None) -> dict:
    keys = keys or list(models.MODEL_ZOO)
    unknown = [k for k in keys if k not in models.MODEL_ZOO]
    if unknown:
        raise HTTPException(400, f"Unknown model key(s): {', '.join(unknown)}")
    paths = models.ensure_many(keys)
    return {"downloaded": {k: str(v) for k, v in paths.items()}}


@app.get("/api/preview")
def preview(scenario: str, profile: str | None = None) -> dict:
    """Show the exact config + argv a run would use, without running it."""
    spec = runner.RunSpec(scenario=scenario, source="preview.mp4", profile=profile)
    composed = config.compose(scenario, profile)
    return {
        "infer_cfg": composed.infer_cfg,
        "lane_seg_config": composed.lane_seg_cfg,
        "flags": composed.flags,
        "missing_models": runner.preflight(spec),
    }


# --------------------------------------------------------------------- jobs

@app.post("/api/jobs")
async def create_job(
    scenario: str = Form(...),
    profile: str | None = Form(None),
    camera: str | None = Form(None),
    source: str | None = Form(None),
    camera_id: int = Form(0),
    pushurl: str = Form(""),
    region_polygon: str = Form(""),
    illegal_parking_time: int | None = Form(None),
    fence_line: str = Form(""),
    capture_seconds: int = Form(0),
    capture_fps: int = Form(10),
    file: UploadFile | None = File(None),
) -> dict:
    """Start a run.

    Sumbernya salah satu dari: `camera` (key dari configs/cameras.yml), sebuah
    `source` (path di server, URL stream, atau "camera"), atau `file` unggahan.

    Kalau `source` berupa URL stream langsung dan `capture_seconds` > 0, klip
    direkam dulu lalu klip itu yang dianalisis. Tanpa ini, stream HLS tidak
    pernah berakhir dan job akan berjalan selamanya."""
    if camera:
        cams = {c.key: c for c in batch.load_cameras()}
        cam = cams.get(camera)
        if cam is None:
            raise HTTPException(404, f"Kamera tidak dikenal: {camera}")
        source = cam.stream_url
        if not capture_seconds:
            capture_seconds = cam.clip_seconds
        capture_fps = capture_fps or cam.analysis_fps
        if not region_polygon.strip() and cam.region_polygon:
            region_polygon = " ".join(str(v) for v in cam.region_polygon)
        if illegal_parking_time is None:
            illegal_parking_time = cam.illegal_parking_time

    if file is None and not source:
        raise HTTPException(400, "Pilih kamera, isi source, atau unggah berkas.")

    if file is not None:
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        # Take only the basename: a client-supplied name must not escape the dir.
        dest = settings.uploads_dir / Path(file.filename or "upload.mp4").name
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        source = str(dest)

    # Stream langsung harus direkam dulu; kalau tidak, job tidak akan selesai.
    if (source and capture_seconds > 0
            and source.lower().startswith(("http://", "https://", "rtsp://", "rtmp://"))):
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        dest = settings.uploads_dir / f"_adhoc_{datetime.now():%H%M%S}.mp4"
        try:
            batch.capture_stream(source, dest, capture_seconds, capture_fps)
        except Exception as exc:
            raise HTTPException(502, f"Gagal merekam stream: {exc}")
        source = str(dest)

    overrides: dict[str, Any] = {}
    args: dict[str, Any] = {}
    if region_polygon.strip():
        try:
            args["region_polygon"] = [int(v) for v in region_polygon.replace(",", " ").split()]
        except ValueError:
            raise HTTPException(400, "region_polygon must be integers: 'x0 y0 x1 y1 ...'")
        if len(args["region_polygon"]) < 6 or len(args["region_polygon"]) % 2:
            raise HTTPException(400, "region_polygon needs >= 3 (x, y) pairs")
        args["region_type"] = "custom"
    if illegal_parking_time is not None:
        args["illegal_parking_time"] = illegal_parking_time
    if fence_line.strip():
        try:
            line = [int(v) for v in fence_line.replace(",", " ").split()]
        except ValueError:
            raise HTTPException(400, "fence_line must be integers: 'x1 y1 x2 y2'")
        if len(line) != 4:
            raise HTTPException(400, "fence_line needs exactly 4 numbers")
        overrides["VEHICLE_RETROGRADE"] = {"fence_line": line}
    if args:
        overrides["args"] = args

    try:
        config.load_scenario(scenario)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))

    spec = runner.RunSpec(scenario=scenario, source=source or "", profile=profile,
                          camera_id=camera_id, pushurl=pushurl, overrides=overrides)
    job = registry.submit(spec)
    return job.public(log_tail=0)


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [j.public(log_tail=0) for j in registry.all()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, log_tail: int = 80) -> dict:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, f"No such job: {job_id}")
    return job.public(log_tail=log_tail)


@app.get("/api/jobs/{job_id}/artifacts")
def job_artifacts(job_id: str) -> dict:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, f"No such job: {job_id}")
    out = settings.runs_dir / job_id / "output"
    if not out.is_dir():
        return {"artifacts": []}
    return {"artifacts": [str(p.relative_to(out)).replace("\\", "/")
                          for p in sorted(out.rglob("*")) if p.is_file()]}


@app.get("/api/jobs/{job_id}/artifact")
def job_artifact(job_id: str, name: str) -> FileResponse:
    # job_id ikut disusun menjadi path, jadi ia harus divalidasi lebih dulu -
    # tanpa ini "../.." membuat direktori keluaran menunjuk ke luar runs_dir,
    # dan pemeriksaan di bawah hanya menjamin berkasnya ada di dalam direktori
    # yang sudah terlanjur salah itu.
    if registry.get(job_id) is None:
        raise HTTPException(404, f"No such job: {job_id}")
    runs = settings.runs_dir.resolve()
    out = (settings.runs_dir / job_id / "output").resolve()
    target = (out / name).resolve()
    # Reject traversal: the resolved path must stay inside this job's output.
    if runs not in out.parents or not target.is_file() or out not in target.parents:
        raise HTTPException(404, f"No such artifact: {name}")
    return FileResponse(target)


# ---------------------------------------------------------------- batch

@app.get("/api/batch")
def batch_status() -> dict:
    """Worker state plus the most recent analysis of every camera."""
    return batch.worker.status()


# ------------------------------------------------------------- mode POC
_MODE_POC_YML = settings.configs_dir / "mode_poc.yml"


def _mode_poc() -> dict:
    """Baca configs/mode_poc.yml. Berkas tidak ada = mode POC mati."""
    import yaml
    try:
        with open(_MODE_POC_YML, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
    except (OSError, ValueError):
        d = {}
    return {"aktif": bool(d.get("aktif", False)), "jeda_siklus": bool(d.get("jeda_siklus", False))}


@app.get("/api/mode-poc")
def mode_poc_get() -> dict:
    return {**_mode_poc(), "siklus_berjalan": batch.worker.running}


@app.put("/api/mode-poc")
def mode_poc_put(payload: dict) -> dict:
    """Nyalakan/matikan Mode POC. Mengubah jeda_siklus langsung menghentikan
    atau menjalankan lagi siklus pelanggaran otomatis."""
    kini = _mode_poc()
    for k in ("aktif", "jeda_siklus"):
        if k in payload:
            kini[k] = bool(payload[k])
    teks = _MODE_POC_YML.read_text(encoding="utf-8") if _MODE_POC_YML.exists() else ""
    kepala = "".join(l for l in teks.splitlines(keepends=True) if l.lstrip().startswith("#"))
    _MODE_POC_YML.write_text(kepala + f"aktif: {str(kini['aktif']).lower()}\n"
                             f"jeda_siklus: {str(kini['jeda_siklus']).lower()}\n", encoding="utf-8")
    if kini["jeda_siklus"]:
        batch.worker.stop()
    elif not batch.worker.running and settings.auto_batch_sec > 0:
        batch.worker.start(interval_sec=settings.auto_batch_sec)
    return mode_poc_get()


@app.post("/api/batch/start")
def batch_start(interval_sec: int = 600, profile: str | None = None) -> dict:
    batch.worker.start(interval_sec=interval_sec, profile=profile)
    return batch.worker.status()


@app.post("/api/batch/stop")
def batch_stop() -> dict:
    batch.worker.stop()
    return {"running": False}


@app.post("/api/batch/run")
def batch_run(key: str | None = None, background: bool = True) -> dict:
    """Trigger one cycle now, or re-analyse a single camera with `key`."""
    if key:
        cams = {c.key: c for c in batch.load_cameras()}
        if key not in cams:
            raise HTTPException(404, f"Kamera tidak dikenal: {key}")
        if not background:
            if batch.analysis_busy():
                raise HTTPException(
                    409, "Analisis lain sedang berjalan. Coba lagi setelah "
                         "selesai, atau pakai background=true untuk mengantre.")
            return batch.analyse(cams[key], batch.worker.profile)
        threading.Thread(target=batch.analyse,
                         args=(cams[key], batch.worker.profile),
                         daemon=True).start()
        return {"started": key}

    if not background:
        if batch.analysis_busy():
            raise HTTPException(
                409, "Analisis lain sedang berjalan. Coba lagi setelah selesai, "
                     "atau pakai background=true untuk mengantre.")
        return {"records": batch.worker.cycle()}
    threading.Thread(target=_safe_cycle, daemon=True).start()
    return {"started": "cycle"}


def _safe_cycle() -> None:
    try:
        batch.worker.cycle()
    except RuntimeError:
        pass          # a cycle was already in flight; the trigger is a no-op


# ------------------------------------------------------ manajemen kamera

@app.get("/api/kamera")
def kamera_list() -> dict:
    """Daftar kamera berikut asal-usulnya.

    `bawaan` menandai kamera dari configs/cameras.yml - yang punya catatan
    survei. Menghapusnya hanya menyembunyikan, tidak membuang berkasnya."""
    custom = batch.load_camera_custom()
    bawaan = batch._kunci_bawaan()
    cams = batch.load_cameras()
    marka = batch.load_marka()
    tampilan_hitung = batch.load_tampilan_hitung()
    return {
        "cameras": [{
            "key": c.key, "id": c.id, "name": c.name, "area": c.area,
            "region": c.region, "owner": c.owner, "scenario": c.scenario,
            "clip_seconds": c.clip_seconds, "analysis_fps": c.analysis_fps,
            "illegal_parking_time": c.illegal_parking_time,
            "zone_label": c.zone_label,
            "punya_zona": len(c.region_polygon) >= 6,
            "region_polygon": c.region_polygon,
            "rtsp_url": c.rtsp_url,
            "marka": marka.get(c.key, []),
            "garis_hitung": (tampilan_hitung.get(c.key) or [{}])[0].get("garis", []),
            "tampilan_hitung": tampilan_hitung.get(c.key, []),
            "bawaan": c.key in bawaan,
            "disunting": c.key in custom["ubah"],
            "stream_url": c.stream_url,
        } for c in cams],
        "disembunyikan": [k for k in custom["sembunyi"]],
        "field_boleh": sorted(batch.FIELD_BOLEH),
    }


@app.post("/api/kamera")
def kamera_simpan(payload: dict) -> dict:
    key = str(payload.get("key", "")).strip()
    try:
        return batch.simpan_kamera(key, payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.delete("/api/kamera/{key}")
def kamera_hapus(key: str) -> dict:
    return batch.hapus_kamera(key)


@app.post("/api/kamera/{key}/pulihkan")
def kamera_pulihkan(key: str) -> dict:
    return batch.pulihkan_kamera(key)


@app.get("/api/kamera/uji")
def kamera_uji(id: str) -> dict:
    """Uji satu id kamera portal sebelum disimpan.

    Menyimpan id yang salah ketik baru ketahuan saat siklus analisis gagal
    belasan menit kemudian; pemeriksaan ini memindahkan kegagalan itu ke muka.
    """
    import urllib.request
    import urllib.error
    cfg_base = "https://dki-jkt.balitower.co.id:7028/"
    hasil = {}
    for nama, sufiks in (("preview", "preview.jpg"), ("playlist", "index.m3u8")):
        url = f"{cfg_base}{id}/{sufiks}"
        try:
            with urllib.request.urlopen(url, timeout=12) as r:
                isi = r.read(4096)
                hasil[nama] = {"ok": r.status == 200, "status": r.status,
                               "bytes": len(isi)}
                if nama == "playlist":
                    teks = isi.decode("utf-8", "replace")
                    for baris in teks.splitlines():
                        if "RESOLUTION=" in baris:
                            hasil["resolusi"] = baris.split("RESOLUTION=")[1].split(",")[0]
                        if "FRAME-RATE=" in baris:
                            hasil["fps"] = baris.split("FRAME-RATE=")[1].split(",")[0]
        except Exception as exc:
            hasil[nama] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    hasil["ok"] = bool(hasil.get("preview", {}).get("ok"))
    return hasil


PORTAL_HLS = "https://dki-jkt.balitower.co.id:7028/"


def _sumber_dari(key: str | None, id: str | None, rtsp: str | None) -> str:
    """Satu alamat siaran dari salah satu dari tiga cara menyebut kamera.

    Hanya dua bentuk yang diterima: id portal (alamatnya disusun di sini, bukan
    dikirim peramban) dan rtsp://. Endpoint ini membuat SERVER menyambung ke
    alamat yang diberikan, jadi skema lain - http ke jaringan dalam, file:// -
    sengaja ditolak.
    """
    if key:
        cams = {c.key: c for c in batch.load_cameras()}
        if key not in cams:
            raise HTTPException(404, f"Kamera tidak dikenal: {key}")
        return cams[key].stream_url
    if rtsp:
        if not rtsp.lower().startswith(("rtsp://", "rtsps://")):
            raise HTTPException(400, "alamat harus diawali rtsp:// atau rtsps://")
        return rtsp
    if id:
        if not all(ch.isalnum() or ch in "_.-" for ch in id):
            raise HTTPException(400, "id portal berisi karakter tidak sah")
        return f"{PORTAL_HLS}{id}/index.m3u8"
    raise HTTPException(400, "sebutkan key, id, atau rtsp")


@app.get("/api/kamera/snapshot")
def kamera_snapshot(key: str | None = None, id: str | None = None,
                    rtsp: str | None = None):
    """Satu frame 640x360 dari sumber kamera, untuk menggambar zona dan marka
    di formulir Manajemen Kamera - termasuk kamera yang belum disimpan."""
    from fastapi.responses import Response
    import time as _time
    # LC-016: gambar mini kamera tanpa preview.jpg (cctv-jsc) memakai endpoint
    # ini; tiap panggilan menjalankan ffmpeg, jadi hasil per kamera disimpan 60 dtk.
    if key and not id and not rtsp:
        c = _SNAP_CACHE.get(key)
        if c and _time.time() - c[0] < 60:
            return Response(c[1], media_type="image/jpeg", headers={"Cache-Control": "max-age=60"})
    url = _sumber_dari(key, id, rtsp)
    try:
        jpg = batch.ambil_snapshot(url)
    except Exception as exc:
        raise HTTPException(502, f"{exc}")
    if key and not id and not rtsp:
        _SNAP_CACHE[key] = (_time.time(), jpg)
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


_SNAP_CACHE: dict = {}


@app.get("/api/kamera/uji-rtsp")
def kamera_uji_rtsp(rtsp: str) -> dict:
    """Uji alamat RTSP dengan ffprobe: resolusi dan fps sumber."""
    import json as _json
    import subprocess
    url = _sumber_dari(None, None, rtsp)
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-rtsp_transport", "tcp", "-timeout", "12000000",
             "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,codec_name",
             "-of", "json", url],
            capture_output=True, text=True, timeout=25, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "tidak ada jawaban dalam 25 detik"}
    if r.returncode != 0:
        return {"ok": False, "error": (r.stderr or "ffprobe gagal").strip()[:300]}
    st = (_json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    return {"ok": bool(st.get("width")), "resolusi": f"{st.get('width')}x{st.get('height')}",
            "fps": st.get("r_frame_rate"), "codec": st.get("codec_name")}


@app.get("/api/marka/{key}")
def marka_get(key: str) -> dict:
    return {"key": key, "garis": batch.load_marka().get(key, [])}


@app.put("/api/marka/{key}")
def marka_put(key: str, payload: dict) -> dict:
    """Simpan garis marka operator. Daftar kosong menghapusnya."""
    garis = payload.get("garis") or []
    try:
        bersih = [[float(v) for v in g[:4]] for g in garis if len(g) >= 4]
    except (TypeError, ValueError):
        raise HTTPException(400, "garis harus berupa [[x1,y1,x2,y2], ...]")
    for g in bersih:
        if not all(0 <= g[i] <= (640 if i % 2 == 0 else 360) for i in range(4)):
            raise HTTPException(400, "koordinat di luar frame 640x360")
    return batch.simpan_marka(key, bersih)


def _garis_640x360(payload: dict) -> list[list[float]]:
    garis = payload.get("garis") or []
    try:
        bersih = [[float(v) for v in g[:4]] for g in garis if len(g) >= 4]
    except (TypeError, ValueError):
        raise HTTPException(400, "garis harus berupa [[x1,y1,x2,y2], ...]")
    for g in bersih:
        if not all(0 <= g[i] <= (640 if i % 2 == 0 else 360) for i in range(4)):
            raise HTTPException(400, "koordinat di luar frame 640x360")
        if abs(g[2] - g[0]) + abs(g[3] - g[1]) < 20:
            raise HTTPException(400, "ruas terlalu pendek (< 20 piksel)")
    if len(bersih) > 6:
        raise HTTPException(400, "paling banyak 6 ruas per kamera")
    return bersih


@app.get("/api/garis-putih/{key}")
def garis_putih(key: str, segar: bool = False) -> dict:
    """Latar bersih (median +-20 dtk, kendaraan terhapus) dan garis marka putih
    terdeteksi, untuk penyunting garis hitung. Disimpan 10 menit per kamera."""
    from . import marka_putih
    cams = {c.key: c for c in batch.load_cameras()}
    cam = cams.get(key)
    if not cam:
        raise HTTPException(404, "kamera tidak dikenal")
    try:
        return marka_putih.pemandu(key, cam.stream_url, segar)
    except Exception as exc:
        raise HTTPException(502, f"pemandu gagal: {exc}")


@app.get("/api/garis-hitung/{key}")
def garis_hitung_get(key: str) -> dict:
    return {"key": key, "tampilan": batch.load_tampilan_hitung().get(key, [])}


@app.put("/api/garis-hitung/{key}")
def garis_hitung_put(key: str, payload: dict) -> dict:
    """Simpan garis hitung untuk satu tampilan (posisi PTZ) kamera.

    payload: {"garis": [[x1,y1,x2,y2],...], "acuan": base64 80x36 abu-abu,
    "indeks": tampilan yang sedang cocok, atau null untuk tampilan baru}.
    """
    if key not in {c.key for c in batch.load_cameras()}:
        raise HTTPException(404, "kamera tidak dikenal")
    garis = _garis_640x360(payload)
    acuan = payload.get("acuan")
    if acuan and not batch._acuan_sah(acuan):
        raise HTTPException(400, "acuan harus base64 80x36 byte abu-abu")
    indeks = payload.get("indeks")
    if indeks is not None:
        try:
            indeks = int(indeks)
        except (TypeError, ValueError):
            raise HTTPException(400, "indeks harus bilangan bulat")
    if indeks is None and garis and not acuan:
        raise HTTPException(400, "tampilan baru butuh gambar acuan posisi kamera")
    return batch.simpan_garis_hitung(key, garis, acuan, indeks)


# ----------------------------------------------------- aduan masyarakat

def _kamera_ringkas() -> list[dict]:
    return [{"key": c.key, "name": c.name, "area": c.area, "region": c.region}
            for c in batch.load_cameras()]


@app.post("/api/aduan")
def aduan_buat(payload: dict, cocokkan: bool = True) -> dict:
    """Satu aduan dari petugas CRM. Langsung dicocokkan bila cocokkan=true."""
    try:
        aid, baru = aduan.buat(payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    hasil = aduan.cocokkan(aid, _kamera_ringkas()) if cocokkan else None
    return {"id": aid, "baru": baru, "pencocokan": hasil}


@app.post("/api/aduan/impor")
def aduan_impor(payload: list[dict]) -> dict:
    """Impor massal dari ekspor CRM/JAKI. Aman diulang: pasangan
    (sumber, ref_sumber) yang sudah ada tidak digandakan. Baris yang gagal
    validasi dilaporkan per baris, tidak menggagalkan seluruh impor."""
    kam = _kamera_ringkas()
    hasil = {"baru": 0, "sudah_ada": 0, "gagal": []}
    for i, baris in enumerate(payload):
        try:
            aid, baru = aduan.buat(baris, oleh="impor")
            if baru:
                hasil["baru"] += 1
                aduan.cocokkan(aid, kam)
            else:
                hasil["sudah_ada"] += 1
        except ValueError as exc:
            hasil["gagal"].append({"baris": i, "ref_sumber": baris.get("ref_sumber"), "alasan": str(exc)})
    return hasil


@app.get("/api/aduan")
def aduan_daftar(status: str | None = None, page: int = 1, per_page: int = 20) -> dict:
    return aduan.daftar(status or None, page, per_page)


@app.get("/api/aduan/kamera")
def aduan_per_kamera() -> list[dict]:
    """Ukuran per kamera: aduan, ketepatan deteksi, cakupan rekaman."""
    return aduan.statistik_kamera(_kamera_ringkas())


@app.get("/api/aduan/{aduan_id}")
def aduan_detail(aduan_id: int) -> dict:
    try:
        d = aduan.detail(aduan_id)
    except KeyError:
        raise HTTPException(404, f"Aduan tidak ditemukan: {aduan_id}")
    for k in d["kandidat"]:
        k["ada_bukti"] = batch.bukti_path(k["event_id"]).is_file()
    return d


@app.post("/api/aduan/{aduan_id}/cocokkan")
def aduan_cocokkan(aduan_id: int, jendela_menit: int = 30) -> dict:
    if not 5 <= jendela_menit <= 360:
        raise HTTPException(400, "jendela_menit harus 5-360")
    try:
        return aduan.cocokkan(aduan_id, _kamera_ringkas(), jendela_menit)
    except KeyError:
        raise HTTPException(404, f"Aduan tidak ditemukan: {aduan_id}")


@app.post("/api/aduan/{aduan_id}/putus")
def aduan_putus(aduan_id: int, payload: dict) -> dict:
    try:
        return aduan.putus(aduan_id, int(payload.get("event_id")),
                           str(payload.get("keputusan")), payload.get("catatan"))
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc))
    except KeyError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/aduan/{aduan_id}/status")
def aduan_status(aduan_id: int, payload: dict) -> dict:
    try:
        return aduan.ubah_status(aduan_id, str(payload.get("status")), payload.get("catatan"))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except KeyError:
        raise HTTPException(404, f"Aduan tidak ditemukan: {aduan_id}")


@app.get("/api/kamera-lokasi")
def kamera_lokasi_semua() -> dict:
    return aduan.lokasi_semua()


@app.put("/api/kamera-lokasi/{key}")
def kamera_lokasi_simpan(key: str, payload: dict) -> dict:
    try:
        return aduan.lokasi_simpan(key, float(payload["lat"]), float(payload["lon"]),
                                   int(payload.get("radius_m") or 150), payload.get("alamat"))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"lat, lon wajib; {exc}")


# ------------------------------------------------------- hitung per kelas

@app.post("/api/hitung/run")
def hitung_run(key: str, background: bool = True, seconds: int = 10) -> dict:
    """Ukur komposisi lalu lintas per kelas pada satu kamera.

    Mesinnya berbeda dari analisis PP-Vehicle - detektor COCO enam kelas - dan
    satu pengukuran memakan ±30 detik di CPU, jadi bawaannya berjalan di latar.
    """
    cams = {c.key: c for c in batch.load_cameras()}
    cam = cams.get(key)
    if cam is None:
        raise HTTPException(404, f"Kamera tidak dikenal: {key}")
    if batch.analysis_busy():
        raise HTTPException(
            409, "Analisis lain sedang berjalan. Coba lagi setelah selesai.")
    if not background:
        return batch.hitung(cam, seconds=seconds)
    threading.Thread(target=batch.hitung, args=(cam,),
                     kwargs={"seconds": seconds}, daemon=True).start()
    return {"started": key, "seconds": seconds}


@app.get("/api/hitung/{key}")
def hitung_status(key: str) -> dict:
    """Hasil penghitungan terakhir satu kamera."""
    cams = {c.key: c for c in batch.load_cameras()}
    cam = cams.get(key)
    if cam is None:
        raise HTTPException(404, f"Kamera tidak dikenal: {key}")
    rekaman = batch.read_hitung(cam)
    return {"key": key, "ada": rekaman is not None, "record": rekaman}


@app.get("/api/hitung/frame/{key}")
def hitung_frame(key: str) -> FileResponse:
    cams = {c.key: c for c in batch.load_cameras()}
    cam = cams.get(key)
    if cam is None or not cam.hitung_frame_path.is_file():
        raise HTTPException(404, f"Belum ada frame penghitungan untuk {key}")
    return FileResponse(cam.hitung_frame_path, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})


# ------------------------------------------------------ deteksi langsung

@app.post("/api/langsung/mulai")
def langsung_mulai(key: str, model: str = "accurate",
                   interval: float | None = None) -> dict:
    """Mulai PP-YOLOE+ L + ByteTrack menerus pada kamera yang sedang ditonton
    di Live. Satu kamera sekaligus; pantau (siklus pendek) dihentikan karena
    keduanya memakai CPU yang sama di server bersama."""
    try:
        _cpu_poc_stop()
        batch.watcher.stop()
        return langsung.mulai(key, model, sample_interval=interval)
    except KeyError:
        raise HTTPException(404, f"Kamera tidak dikenal: {key}")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/langsung")
def langsung_ambil(key: str, sejak: float | None = None, sejak_p: int | None = None) -> dict:
    """Hasil per frame (dengan PTS) dan peristiwa lintas garis sesudah `sejak`.
    Memanggilnya juga menandakan masih ada yang menonton."""
    return langsung.ambil(key, sejak, sejak_p)


@app.post("/api/langsung/henti")
def langsung_henti() -> dict:
    langsung.henti()
    return langsung.status()


# ---------------------------------------------------------------- pantau

@app.post("/api/watch/start")
def watch_start(key: str) -> dict:
    """Mulai memantau satu kamera dengan siklus pendek berulang.

    Memanggilnya lagi untuk kamera yang sama hanya memperbarui detak penonton,
    tidak merestart siklusnya."""
    try:
        langsung.henti()             # CPU yang sama; lihat /api/langsung/mulai
        return batch.watcher.start(key)
    except KeyError:
        raise HTTPException(404, f"Kamera tidak dikenal: {key}")


@app.post("/api/watch/stop")
def watch_stop() -> dict:
    batch.watcher.stop()
    return batch.watcher.status()


@app.get("/api/watch")
def watch_status() -> dict:
    """Keadaan pantau berikut hasil putaran terakhir.

    Memanggilnya juga menandakan masih ada yang menonton; tanpa panggilan
    selama 90 detik, siklusnya berhenti sendiri supaya tab yang terlupa tidak
    membebani server."""
    batch.watcher.touch()
    return batch.watcher.status()


@app.get("/api/watch/frame/{key}")
def watch_frame(key: str) -> FileResponse:
    """Frame bersih putaran pantau terakhir."""
    cams = {c.key: c for c in batch.load_cameras()}
    cam = cams.get(key)
    if cam is None:
        raise HTTPException(404, f"Kamera {key} tidak dikenal")
    path = batch.Watcher.frame_path(cam)
    if not path.is_file():
        raise HTTPException(404, f"Belum ada frame pantau untuk {key}")
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})


@app.get("/api/stats/cameras")
def stats_cameras() -> list[dict]:
    """Rincian di balik kartu "Kendaraan terpantau" - satu baris per kamera,
    dijumlah dari tabel siklus dengan batas hari WIB yang sama."""
    return store.stats_per_camera()


# ------------------------------------------------------ CPU PoC dua kamera

_CPU_POC_KEYS = ("benhil2", "gerbangpemuda")
_CPU_POC_SHARED = SharedDual(_CPU_POC_KEYS)
_CPU_POC_ENGINES = _CPU_POC_SHARED.engines
_CPU_POC_ACTIVE = False


def _cpu_poc_stop() -> None:
    global _CPU_POC_ACTIVE
    _CPU_POC_SHARED.stop()
    _CPU_POC_ACTIVE = False


@app.post("/api/cpu-poc/start")
def cpu_poc_start(model: str = "compact", threads_each: int = 4,
                  interval: float | None = None) -> dict:
    """Dua kamera, satu predictor bersama, tracker terpisah per kamera."""
    global _CPU_POC_ACTIVE
    if model not in ("compact", "accurate"):
        raise HTTPException(400, "model harus compact atau accurate")
    if not 1 <= threads_each <= 4:
        raise HTTPException(400, "threads_each harus 1..4")
    # Compact terukur sekitar 250 ms/inference di CPU ini. Dua kamera berarti
    # cadence aman ~0,5 dtk/kamera; 0,33 meminta 6 FPS total dan pasti drop.
    interval = interval or (0.5 if model == "compact" else 0.8)
    batch.watcher.stop()
    langsung.henti()
    _cpu_poc_stop()
    try:
        total_threads = min(8, threads_each * 2)
        _CPU_POC_SHARED.start(model, cpu_threads=total_threads,
                              sample_interval=interval)
        _CPU_POC_ACTIVE = True
    except Exception as exc:
        _cpu_poc_stop()
        raise HTTPException(500, f"gagal memulai dua kamera: {type(exc).__name__}: {exc}")
    return {"active": True, "model": model, "cpu_threads_total": total_threads,
            "sample_interval": interval,
            "cameras": {k: e.status() for k, e in _CPU_POC_ENGINES.items()}}


@app.post("/api/cpu-poc/stop")
def cpu_poc_stop() -> dict:
    _cpu_poc_stop()
    return {"active": False, "cameras": {k: e.status() for k, e in _CPU_POC_ENGINES.items()}}


def _umur_iso(teks: str | None) -> float | None:
    if not teks:
        return None
    try:
        dt = datetime.fromisoformat(str(teks).replace("Z", "+00:00"))
        kini = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        return round(max(0.0, (kini - dt).total_seconds()), 1)
    except (TypeError, ValueError):
        return None


def _memori_cgroup() -> dict:
    def baca(path: str) -> str | None:
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    kini = baca("/sys/fs/cgroup/memory.current")
    batas = baca("/sys/fs/cgroup/memory.max")
    return {
        "used_mb": round(int(kini) / 1048576, 1) if kini and kini.isdigit() else None,
        "limit_mb": round(int(batas) / 1048576, 1) if batas and batas.isdigit() else None,
    }


def _traffic_terakhir(key: str) -> dict:
    path = settings.runs_dir / "_latest" / f"{key}_traffic.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    data["age_sec"] = _umur_iso(data.get("at"))
    return data


@app.get("/api/cpu-poc")
def cpu_poc_status() -> dict:
    """Keadaan nyata PoC CPU dua kamera; tidak memulai job/stream baru."""
    kamera = {c.key: c for c in batch.load_cameras()}
    tampilan = batch.load_tampilan_hitung()
    live = langsung.status()
    dual_running = _CPU_POC_SHARED.running
    if dual_running:
        _CPU_POC_SHARED.touch()
    hasil = []
    for key in _CPU_POC_KEYS:
        cam = kamera.get(key)
        if cam is None:
            hasil.append({"key": key, "ada": False, "state": "NOT_CONFIGURED"})
            continue
        raw = batch.read_record(cam) or {}
        hitung = batch.read_hitung(cam) or {}
        traffic_saved = _traffic_terakhir(key)
        hasil_hitung = hitung.get("results") or {}
        per_kelas = hasil_hitung.get("per_class") or []
        visible = sum(int(x.get("per_frame") or 0) for x in per_kelas)
        crossing = sum(int(x.get("masuk") or 0) for x in per_kelas)
        lines = tampilan.get(key) or []
        capture_ok = bool(raw.get("ok"))
        counter_ok = bool(hitung.get("ok"))
        dual = _CPU_POC_ENGINES[key].status()
        if not (dual.get("traffic") or {}).get("samples") and traffic_saved:
            dual["traffic"] = traffic_saved
        live_here = dual if (dual_running or dual.get("key") == key or traffic_saved) else (
            live if live.get("key") == key else None)
        last_live_recent = bool(traffic_saved and (traffic_saved.get("age_sec") or 1e9) < 300)
        if live_here and live_here.get("running"):
            state = live_here.get("kondisi") or "RUNNING"
        elif last_live_recent:
            state = "LAST_LIVE_OK"
        elif not capture_ok:
            state = "SOURCE_DOWN" if raw.get("error") else "NO_DATA"
        elif not counter_ok:
            state = "COUNTER_FAILED"
        else:
            state = "LAST_RUN_OK"
        hasil.append({
            "key": key, "ada": True, "name": cam.name, "area": cam.area,
            "state": state, "stream_kind": "RTSP" if cam.rtsp_url else "HLS",
            "line_configured": bool(lines), "line_views": len(lines),
            "capture": {"ok": capture_ok, "at": raw.get("finished_at"),
                        "age_sec": _umur_iso(raw.get("finished_at")),
                        "elapsed_sec": raw.get("total_sec"), "error": raw.get("error")},
            "counter": {"ok": counter_ok, "at": hitung.get("at"),
                        "age_sec": _umur_iso(hitung.get("at")),
                        "frames": hasil_hitung.get("frames"),
                        "video_fps": hasil_hitung.get("video_fps"),
                        "unique_tracks": hasil_hitung.get("unique_tracks"),
                        "visible_now": visible, "crossing": crossing,
                        "per_class": per_kelas, "error": hitung.get("error")},
            "live": live_here,
            "last_live": traffic_saved,
            "frame_url": f"/api/hitung/frame/{key}",
            "gates": {
                "source_last_run": capture_ok or last_live_recent,
                "counter_last_run": counter_ok,
                "line_configured": bool(lines),
                "class_7_ready": False,
                "congestion_ready": bool(live_here and
                    (live_here.get("traffic") or {}).get("quality_score", 0) >= .5),
                "violation_validated": False,
            },
        })
    load = os.getloadavg()
    return {
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "profile": settings.profile, "target_cameras": list(_CPU_POC_KEYS),
        "runtime": {"cpu_count": os.cpu_count(), "load_1m": round(load[0], 2),
                    "load_5m": round(load[1], 2), "memory": _memori_cgroup()},
        "live_engine": {"dual_active": dual_running,
                        "mode": "shared-predictor / isolated-trackers",
                        "runtime": _CPU_POC_SHARED.status(),
                        "note": "satu model CPU, scheduler round-robin latest-frame"},
        "cameras": hasil,
        "known_gaps": [
            "Model COCO belum memisahkan tujuh kelas Dishub",
            "Engine kemacetan masih provisional dan belum UAT",
            "Rule pelanggaran dua kamera belum tervalidasi",
            "Model compact belum dilatih khusus tujuh kelas Dishub",
        ],
    }


@app.get("/api/batch/frame/{key}")
def batch_frame(key: str, annotated: bool = False) -> FileResponse:
    """Frame terakhir hasil analisis satu kamera - keluaran model sungguhan.

    Bawaannya frame BERSIH: dashboard menggambar sendiri kotak, zona, dan
    lintasannya dari angka di results.json, supaya ukuran huruf dan jumlah
    label bisa dikendalikan. `?annotated=1` memberi gambar beranotasi milik
    pipeline apa adanya, untuk pemeriksaan tim teknis."""
    cams = {c.key: c for c in batch.load_cameras()}
    cam = cams.get(key)
    if cam is None:
        raise HTTPException(404, f"Kamera {key} tidak dikenal")
    path = cam.frame_pipeline_path if annotated else cam.frame_path
    if not path.is_file():
        raise HTTPException(404, f"Belum ada frame hasil analisis untuk {key}")
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})


# ---------------------------------------------------------------- zona

@app.get("/api/cameras")
def cameras() -> list[dict]:
    """Daftar kamera beserta zona yang sedang berlaku."""
    over = batch.load_zone_overrides()
    return [{
        "key": c.key, "id": c.id, "name": c.name, "area": c.area,
        "region": c.region, "owner": c.owner, "scenario": c.scenario,
        "stream": c.stream_url,
        "region_polygon": c.region_polygon,
        "zone_label": c.zone_label,
        "illegal_parking_time": c.illegal_parking_time,
        "clip_seconds": c.clip_seconds,
        "analysis_fps": c.analysis_fps,
        # Supaya operator tahu nilai ini dari mana - tulisan tangan di
        # cameras.yml, atau hasil editor yang menimpanya.
        "zone_source": "editor" if c.key in over else "cameras.yml",
        "zone_updated_at": (over.get(c.key) or {}).get("updated_at"),
    } for c in batch.load_cameras()]


@app.put("/api/cameras/{key}/zone")
def save_zone(key: str, payload: dict) -> dict:
    """Simpan poligon dari editor visual. Berlaku untuk batch berikutnya."""
    cams = {c.key: c for c in batch.load_cameras()}
    if key not in cams:
        raise HTTPException(404, f"Kamera tidak dikenal: {key}")

    poly = payload.get("region_polygon") or []
    try:
        poly = [int(v) for v in poly]
    except (TypeError, ValueError):
        raise HTTPException(400, "region_polygon harus berisi bilangan bulat")
    if len(poly) < 6 or len(poly) % 2:
        raise HTTPException(400, "Zona butuh minimal 3 titik (x, y)")
    # Frame CCTV DKI 640x360; titik di luar itu tidak akan pernah kena.
    for x, y in zip(poly[::2], poly[1::2]):
        if not (0 <= x <= 640 and 0 <= y <= 360):
            raise HTTPException(
                400, f"Titik ({x}, {y}) di luar frame 640x360 - zona tidak akan berfungsi")

    parking = payload.get("illegal_parking_time")
    if parking is not None:
        try:
            parking = int(parking)
        except (TypeError, ValueError):
            raise HTTPException(400, "illegal_parking_time harus bilangan bulat")

    entry = batch.save_zone_override(key, poly, str(payload.get("zone_label") or ""),
                                     parking)
    return {"key": key, **entry}


@app.delete("/api/cameras/{key}/zone")
def reset_zone(key: str) -> dict:
    """Buang timpaan editor, kembali ke nilai di cameras.yml."""
    zones = batch.load_zone_overrides()
    if key not in zones:
        raise HTTPException(404, f"Tidak ada timpaan zona untuk {key}")
    zones.pop(key)
    import yaml as _yaml
    with open(batch.ZONES_FILE, "w", encoding="utf-8") as f:
        f.write("# Ditulis oleh editor zona visual (/zona).\n\n")
        _yaml.safe_dump({"zones": zones}, f, sort_keys=True, allow_unicode=True)
    return {"key": key, "reset": True}


# --------------------------------------------------------------- statistik

@app.get("/api/stats")
def stats() -> dict:
    """Recap figures for the dashboard tiles, for the current WIB day."""
    return store.stats()


@app.get("/api/events")
def events(limit: int = 40, today_only: bool = True) -> list[dict]:
    """Bawaannya hari WIB berjalan, supaya daftar Kejadian di dashboard dan
    kartu rekap selalu menghitung himpunan yang sama. `today_only=false` untuk
    penelusuran riwayat."""
    return store.recent_events(limit=limit, today_only=today_only)


@app.get("/api/events/list")
def events_list(page: int = 1, per_page: int = 20, status: str | None = None,
                camera: str | None = None, kind: str | None = None,
                authority: str | None = None, today_only: bool = False) -> dict:
    """Daftar berhalaman untuk halaman Antrean Verifikasi."""
    if status not in (None, "", "belum", "sudah"):
        raise HTTPException(400, "status harus 'belum' atau 'sudah'")
    hasil = store.list_events(page=page, per_page=per_page, status=status or None,
                              camera=camera or None, kind=kind or None,
                              authority=authority or None, today_only=today_only)
    for it in hasil["items"]:
        it["ada_bukti"] = batch.bukti_path(it["id"]).is_file()
    return hasil


@app.get("/api/events/{event_id}/bukti")
def event_bukti(event_id: int) -> FileResponse:
    """Foto bukti kejadian, diambil dari klip siklus kejadian itu sendiri."""
    path = batch.bukti_path(event_id)
    if not path.is_file():
        raise HTTPException(404, "Foto bukti tidak tersedia untuk kejadian ini")
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=86400"})


@app.post("/api/events/{event_id}/verify")
def verify_event(event_id: int, verdict: str = "benar", catatan: str | None = None) -> dict:
    if verdict not in ("benar", "bukan"):
        raise HTTPException(400, "verdict harus 'benar' atau 'bukan'")
    if not store.verify(event_id, verdict, catatan=catatan):
        raise HTTPException(404, f"Kejadian tidak ditemukan: {event_id}")
    return {"id": event_id, "verified": True, "verdict": verdict}


# ----------------------------------------------------------------------- ui

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>PP-Vehicle</h1><p>UI missing. API at /docs</p>")
    return HTMLResponse(page.read_text(encoding="utf-8"))


# dashboard.html is authored as an artifact-style fragment (no <html>/<head>) so
# the same file can be published to claude.ai for the presentation; serving it
# here just wraps it in a document skeleton.
_SKELETON = """<!doctype html><html lang="id"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<link rel="icon" href="/aset/dishub-dki.png">
<style>:root{{padding-top:env(safe-area-inset-top,0);padding-bottom:env(safe-area-inset-bottom,0)}}
body{{margin:0}}img{{max-width:100%}}[hidden]{{display:none!important}}</style>
</head><body>{body}</body></html>"""


@app.get("/zona", response_class=HTMLResponse)
def zona() -> HTMLResponse:
    """Editor zona visual."""
    page = WEB_DIR / "zona.html"
    if not page.exists():
        raise HTTPException(404, "zona.html not found")
    return HTMLResponse(page.read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/aduan", response_class=HTMLResponse)
def aduan_page() -> HTMLResponse:
    page = WEB_DIR / "aduan.html"
    if not page.exists():
        raise HTTPException(404, "aduan.html tidak ada")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/verifikasi", response_class=HTMLResponse)
def verifikasi_page() -> HTMLResponse:
    page = WEB_DIR / "verifikasi.html"
    if not page.exists():
        raise HTTPException(404, "verifikasi.html tidak ada")
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/kamera", response_class=HTMLResponse)
def kamera_page() -> HTMLResponse:
    """Manajemen Kamera."""
    page = WEB_DIR / "kamera.html"
    if not page.exists():
        raise HTTPException(404, "kamera.html not found")
    return HTMLResponse(page.read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/demo", response_class=HTMLResponse)
def demo() -> HTMLResponse:
    """Presentation dashboard (Ruang Kendali Lalin)."""
    page = WEB_DIR / "dashboard.html"
    if not page.exists():
        raise HTTPException(404, "dashboard.html not found")
    # no-store, bukan sekadar no-cache: halaman ini masih sering diubah, dan
    # tanpa header ini browser menyimpannya secara heuristik - pengguna lalu
    # melihat versi lama dan mengira perubahannya gagal terpasang.
    return HTMLResponse(_SKELETON.format(body=page.read_text(encoding="utf-8")),
                        headers={"Cache-Control": "no-store, must-revalidate"})


# Rekaman siang tetap dipisahkan dari source Live. Daftar eksplisit mencegah
# path traversal dan memastikan dashboard hanya menyajikan klip POC yang sudah
# diperiksa, bukan semua berkas di data/samples.
_DAYLIGHT_REPLAYS = {
    "benhil2": settings.root / "data" / "samples" / "_batch_benhil2.mp4",
    "gerbangpemuda": settings.root / "data" / "samples" / "_batch_gerbangpemuda.mp4",
}
_DAYLIGHT_ANALYSED = {
    "benhil2": settings.runs_dir / "daylight_classified" / "benhil2_classified.mp4",
    "gerbangpemuda": settings.runs_dir / "daylight_classified" / "gerbangpemuda_classified.mp4",
}
_DAYLIGHT_PIPELINE_TRACKS = {"benhil2": 37, "gerbangpemuda": 28}
_DAYLIGHT_POSTERS = {
    key: settings.runs_dir / "daylight_classified" / f"{key}_poster.jpg"
    for key in _DAYLIGHT_REPLAYS
}


@app.get("/api/daylight-replays")
def daylight_replays() -> dict:
    def version(path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0

    def metrics(key: str) -> dict:
        path = settings.runs_dir / "daylight_baseline" / f"{key}_metrics.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        classified_path = (settings.runs_dir / "daylight_classified" /
                           f"{key}_classified.json")
        try:
            classified = json.loads(classified_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            classified = {}
        if classified:
            names = {
                "motorcycle": "Sepeda Motor", "car": "Mobil Penumpang",
                "bus": "Bus (belum terpilah)", "truck": "Truk (belum terpilah)",
                "person": "Pejalan Kaki", "bicycle": "Sepeda",
            }
            median = classified.get("counts_median") or {}
            maximum = classified.get("counts_max") or {}
            data["model"] = "ppyoloe_coco + road_rider_rule_v1"
            data["classes"] = [{
                "detector_label": label, "name": name,
                "visible_median": median.get(label, 0),
                "visible_max": maximum.get(label, 0),
                "unique_tracks": None,
            } for label, name in names.items()]
            data["inference_ms"] = {
                "p50": classified.get("infer_ms_p50"),
                "p95": classified.get("infer_ms_p95"),
            }
            data["classified_replay"] = {
                "fps": classified.get("fps"), "frames": classified.get("frames"),
                "resolution": classified.get("size"),
                "status": "CLASSIFIED_REPLAY",
            }
        data.setdefault("counting", {
            "unique_tracks": _DAYLIGHT_PIPELINE_TRACKS.get(key),
            "crossing": None,
            "status": "TRACKS_NOT_CROSSING",
        })
        return data

    return {key: {"available": path.is_file(),
                  "url": f"/api/daylight-replays/{key}.mp4",
                  "analysed_available": _DAYLIGHT_ANALYSED[key].is_file(),
                  "analysis_url": (f"/api/daylight-analysis/{key}.mp4"
                                   f"?v={version(_DAYLIGHT_ANALYSED[key])}"),
                  "poster_url": (f"/api/daylight-posters/{key}.jpg"
                                 f"?v={version(_DAYLIGHT_POSTERS[key])}"),
                  "metrics": metrics(key)}
            for key, path in _DAYLIGHT_REPLAYS.items()}


@app.get("/api/daylight-replays/{key}.mp4")
def daylight_replay(key: str) -> FileResponse:
    path = _DAYLIGHT_REPLAYS.get(key)
    if path is None or not path.is_file():
        raise HTTPException(404, f"Rekaman siang tidak tersedia: {key}")
    return FileResponse(path, media_type="video/mp4",
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/api/daylight-analysis/{key}.mp4")
def daylight_analysis(key: str) -> FileResponse:
    path = _DAYLIGHT_ANALYSED.get(key)
    if path is None or not path.is_file():
        raise HTTPException(404, f"Hasil baseline siang tidak tersedia: {key}")
    return FileResponse(path, media_type="video/mp4",
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/api/daylight-posters/{key}.jpg")
def daylight_poster(key: str) -> FileResponse:
    path = _DAYLIGHT_POSTERS.get(key)
    if path is None or not path.is_file():
        raise HTTPException(404, f"Poster rekaman siang tidak tersedia: {key}")
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/cpu-poc", include_in_schema=False)
def cpu_poc_page() -> RedirectResponse:
    """Halaman benchmark lama dipensiunkan; flow operator tunggal ada di /demo."""
    return RedirectResponse(url="/demo", status_code=307)


# Lambang instansi dan aset statis lain, dirujuk relatif sebagai aset/<berkas>.
_ASET_DIR = WEB_DIR / "assets"
if _ASET_DIR.is_dir():
    app.mount("/aset", StaticFiles(directory=_ASET_DIR), name="aset")


# Demo stills referenced by dashboard.html as cam/<name>.jpg
_DEMO_DIR = settings.root / "data" / "demo"
if _DEMO_DIR.is_dir():
    app.mount("/cam", StaticFiles(directory=_DEMO_DIR), name="cam")


@app.exception_handler(FileNotFoundError)
def _not_found(_request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=404)
