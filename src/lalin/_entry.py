"""Run PaddleDetection's pipeline in-process and persist structured results.

`pipeline.py` writes an annotated video and nothing else: the illegal-parking
alarms and the in/out counts it computes live and die inside the process. This
entry point imports that module, wraps the two functions where those results
are produced, runs its normal `main()`, and writes `results.json` beside the
video.

Wrapping module-level names is a deliberate seam, not a monkeypatch of
convenience: `update_object_info` and `flow_statistic` are imported into
`pipeline`'s namespace at module scope, so rebinding them there is enough. If
upstream renames or relocates either one, the lookup below raises
AttributeError at startup instead of silently producing empty results.

Invoked by runner.py with LALIN_PIPELINE_DIR and LALIN_RESULTS in the
environment; every other argument is passed straight through to upstream's
own argument parser.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

PIPELINE_DIR = os.environ["LALIN_PIPELINE_DIR"]        # <root>/deploy/pipeline
RESULTS_PATH = os.environ["LALIN_RESULTS"]
DEPLOY_DIR = os.path.dirname(PIPELINE_DIR)             # <root>/deploy

# Upstream's import graph needs both directories on the path, and the module
# must be reached as `pipeline.pipeline`, not `pipeline`:
#   * pipeline.py itself does `from datacollector import ...` - an implicit
#     relative import that only resolves with deploy/pipeline on sys.path;
#   * ppvehicle/vehicle_plate.py does `from pipeline.ppvehicle.vehicle_plateutils
#     import ...`, which needs deploy on sys.path AND the name `pipeline` bound
#     to the package, not to the module.
# Running `python pipeline.py` satisfies both because the script is __main__ and
# never occupies the name `pipeline`. Importing it as plain `pipeline` shadows
# the package and breaks plate recognition.
# Order is load-bearing: DEPLOY_DIR must come first, otherwise `pipeline`
# resolves to deploy/pipeline/pipeline.py (the module) instead of
# deploy/pipeline/ (the package) and we are back to the shadowing bug.
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, DEPLOY_DIR)

import numpy as np  # noqa: E402

# PaddleDetection 2.9 still uses the numpy aliases removed in numpy 1.24 -
# deploy/pipeline/ppvehicle/vehicleplate_postprocess.py calls np.int in the
# OCR box post-processing, so plate recognition crashes the run the first time
# the text detector proposes a box. It is intermittent: a clip where OCR finds
# no candidate never touches that line.
#
# The shim lives here rather than as a patch to vendor/ because setup re-clones
# that tree, which would silently drop the patch. Restoring the aliases is safe:
# they were exact synonyms for the builtins.
for _name, _builtin in (("int", int), ("float", float), ("bool", bool),
                        ("object", object), ("str", str)):
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

import paddle  # noqa: E402
import pipeline.pipeline as pp  # noqa: E402

# track_id -> event. The pipeline re-reports a parked vehicle on every frame
# once it trips, so keep the first sighting and extend its duration.
PARKING: dict[int, dict] = {}
SEEN_IDS: set[int] = set()
COUNTS = {"in": 0, "out": 0, "frames": 0, "fps": 0.0, "mode": ""}

# Atribut dan pelat dihasilkan tiap frame lalu langsung dibuang oleh pipeline.
# Keduanya adalah dua dari empat toolbox PP-Vehicle, jadi tanpa ditangkap di
# sini dashboard tidak punya apa pun untuk ditampilkan selain hasil tracking.
ATTRS: list[str] = []
PLATES: list[str] = []

# Kotak frame terakhir dan jejak per track. Pipeline sudah menggambar keduanya
# ke dalam video, tetapi dengan ukuran huruf tetap yang di 640x360 saling
# menimpa sampai tak terbaca. Menyimpan angkanya membuat dashboard bisa
# menggambar sendiri seperlunya - dan membuat tombol lapisan Objek/Lintasan
# berfungsi pada mode Analisis, bukan hanya pada cuplikan.
LAST_BOXES: list[dict] = []
TRAILS: dict[int, list[tuple[float, float]]] = {}
TRAIL_MAX = 60          # +-3 detik pada 20 fps; cukup untuk membaca arah
# Riwayat titik roda LENGKAP per track, khusus untuk menguji apakah kendaraan
# benar-benar melintasi marka. TRAILS di atas dipangkas ke 60 titik untuk
# tampilan; itu terlalu pendek untuk klip 20 detik.
JEJAK_PENUH: dict[int, list[tuple[float, float]]] = {}
JEJAK_MAKS = 800

# Pelanggaran marka. VehiclePressingRecognizer.mot_run menguji perpotongan sisi
# BAWAH kotak dengan tiap ruas garis marka, lalu pipeline membuang hasilnya
# setelah menggambarnya ke video. Tanpa ditangkap di sini results.json selalu
# melaporkan nol pelanggaran walaupun modulnya menyala - itulah keadaan sebelum
# blok ini ada.
#
# Satu frame menyentuh garis BUKAN pelanggaran: kotak deteksi bergetar, dan di
# 640x360 getaran beberapa piksel sudah cukup membuat sisi bawah menyerempet
# marka. Karena itu sebuah track baru dicatat setelah tersentuh pada sekurangnya
# PRESS_MIN_FRAMES frame.
PRESSING: dict[int, dict] = {}
PRESS_MIN_FRAMES = 3

# Lawan arah. VehicleRetrogradeRecognizer.mot_run mengembalikan daftar track ID
# yang arahnya berlawanan dengan arus, ditambah fence_line - garis tengah lajur
# yang ditebak dari arah arus bila konfigurasi tidak memberinya. Pipeline
# menumpuk ID-nya di pipeline_res['vehicle_retrograde'] lalu hanya
# menggambarnya; tanpa blok ini hasilnya hilang sama seperti pressing dulu.
RETROGRADE: dict[int, dict] = {}
FENCE: list[float] = []

# Garis marka hasil segmentasi, disimpan agar dashboard dapat menggambarnya di
# atas frame. Tanpa ini pelanggaran hanya berupa angka yang tak bisa diperiksa
# mata - petugas tidak punya cara menilai apakah garisnya memang benar.
LANES: list[list[list[float]]] = []

# Marka yang digambar operator, menggantikan hasil segmentasi.
#
# Alasannya diuji, bukan diasumsikan: pada feed publik 640x360 (JPO Gatot
# Subroto 7) PP-LiteSeg tidak menemukan marka jalan sama sekali - ia mengikuti
# pagar pembatas/median, lalu setiap kendaraan yang lewat di dekat pagar itu
# dilaporkan melanggar. Bus yang sedang berada di jalurnya sendiri pun ikut
# tertandai. Selama resolusi feed belum naik, satu-satunya geometri yang bisa
# dipertanggungjawabkan adalah garis yang ditetapkan manusia.
#
# Format: JSON [[x1,y1,x2,y2], ...] pada koordinat frame 640x360.
def _muat_marka() -> list[list[float]]:
    mentah = os.environ.get("LALIN_MARKA") or ""
    if not mentah.strip():
        return []
    try:
        garis = json.loads(mentah)
        bersih = []
        for g in garis:
            if len(g) >= 4:
                bersih.append([float(v) for v in g[:4]])
        return bersih
    except Exception:
        return []


MARKA_MANUAL = _muat_marka()

_orig_update_object_info = pp.update_object_info
_orig_flow_statistic = pp.flow_statistic
_orig_result_update = pp.Result.update


def _result_update(self, res, name):
    """Seam ketiga: Result.update adalah satu-satunya titik yang dilewati SEMUA
    hasil modul sebelum menghilang. Membungkus di sini menangkap atribut dan
    pelat tanpa menyentuh alur pipeline."""
    try:
        if name == "vehicle_attr":
            for item in (res or {}).get("output") or []:
                if isinstance(item, (list, tuple)):
                    ATTRS.extend(str(x) for x in item if x)
                elif item:
                    ATTRS.append(str(item))
        elif name == "lanes":
            if MARKA_MANUAL:
                # Ditukar SEBELUM diteruskan, sehingga VehiclePressingRecognizer
                # menguji perpotongan terhadap garis operator - bukan terhadap
                # pagar yang salah dikenali sebagai marka.
                res = dict(res or {})
                res["output"] = [MARKA_MANUAL]
            LANES.clear()
            for lane in (res or {}).get("output") or []:
                ruas = []
                for seg in lane:
                    try:
                        ruas.append([round(float(v), 1) for v in seg[:4]])
                    except Exception:
                        pass
                if ruas:
                    LANES.append(ruas)
        elif name == "vehicle_retrograde":
            frame = COUNTS["frames"]
            for tid in (res or {}).get("output") or []:
                try:
                    tid = int(tid)
                except (TypeError, ValueError):
                    continue
                if tid not in RETROGRADE:
                    RETROGRADE[tid] = {"track_id": tid, "first_frame": frame,
                                       "last_frame": frame}
                else:
                    RETROGRADE[tid]["last_frame"] = frame
            garis = (res or {}).get("fence_line")
            if garis is not None and len(garis) >= 4:
                FENCE[:] = [round(float(v), 1) for v in list(garis)[:4]]
        elif name == "vehicle_press":
            for bbox in (res or {}).get("output") or []:
                # [track_id, cls_id, score, x1, y1, x2, y2]
                if len(bbox) < 7:
                    continue
                tid = int(bbox[0])
                rec = PRESSING.get(tid)
                frame = COUNTS["frames"]
                if rec is None:
                    PRESSING[tid] = {
                        "track_id": tid, "first_frame": frame,
                        "last_frame": frame, "frames": 1,
                        "score": round(float(bbox[2]), 3),
                        "bbox": [round(float(v), 1) for v in bbox[3:7]],
                    }
                else:
                    rec["last_frame"] = frame
                    rec["frames"] += 1
                    rec["bbox"] = [round(float(v), 1) for v in bbox[3:7]]
        elif name == "vehicleplate":
            for key in ("plate", "vehicleplate"):
                for teks in (res or {}).get(key) or []:
                    teks = str(teks).strip()
                    if teks:
                        PLATES.append(teks)
    except Exception:
        pass          # penangkapan data sekunder tidak boleh menjatuhkan run
    return _orig_result_update(self, res, name)


def _update_object_info(object_in_region_info, result, region_type, entrance,
                        fps, illegal_parking_time, *args, **kwargs):
    info, illegal = _orig_update_object_info(
        object_in_region_info, result, region_type, entrance, fps,
        illegal_parking_time, *args, **kwargs)
    frame_id = result[0]
    COUNTS["fps"] = float(fps or 0)
    for track_id, value in (illegal or {}).items():
        key = int(track_id)
        bbox = value.get("bbox") if isinstance(value, dict) else None
        rec = PARKING.get(key)
        if rec is None:
            PARKING[key] = {
                "track_id": key,
                "first_frame": int(frame_id),
                "last_frame": int(frame_id),
                "bbox": [float(v) for v in bbox] if bbox else None,
                "threshold_sec": int(illegal_parking_time),
            }
        else:
            rec["last_frame"] = int(frame_id)
            if bbox:
                rec["bbox"] = [float(v) for v in bbox]
    return info, illegal


def _flow_statistic(result, secs_interval=None, do_entrance_counting=False,
                    do_break_in_counting=False, *args, **kwargs):
    statistic = _orig_flow_statistic(result, secs_interval, do_entrance_counting,
                                     do_break_in_counting, *args, **kwargs)
    try:
        # Mode menentukan arti angkanya: garis lintas memberi masuk DAN keluar,
        # sedangkan zona hanya memberi "masuk zona". Keduanya toolbox yang sama.
        COUNTS["mode"] = ("garis" if do_entrance_counting
                          else "zona" if do_break_in_counting else "")
        COUNTS["frames"] = int(result[0])
        for track_id in result[3]:
            if int(track_id) >= 0:
                SEEN_IDS.add(int(track_id))
        COUNTS["in"] = len(statistic.get("in_id_list", []) or [])
        COUNTS["out"] = len(statistic.get("out_id_list", []) or [])

        # result = (frame_id, tlwhs, scores, track_ids). Ditulis ulang tiap
        # frame, jadi yang tersisa saat run selesai adalah frame terakhir -
        # frame yang sama dengan gambar diam yang ditampilkan dashboard.
        tlwhs, scores, ids = result[1], result[2], result[3]
        LAST_BOXES.clear()
        for tlwh, score, track_id in zip(tlwhs, scores, ids):
            tid = int(track_id)
            if tid < 0:
                continue
            x, y, w, h = (float(v) for v in tlwh[:4])
            LAST_BOXES.append({"id": tid, "x": round(x, 1), "y": round(y, 1),
                               "w": round(w, 1), "h": round(h, 1),
                               "score": round(float(score), 3)})
            jejak = TRAILS.setdefault(tid, [])
            jejak.append((round(x + w / 2, 1), round(y + h, 1)))   # titik roda
            if len(jejak) > TRAIL_MAX:
                del jejak[:-TRAIL_MAX]
            penuh = JEJAK_PENUH.setdefault(tid, [])
            penuh.append((x + w / 2, y + h))
            if len(penuh) > JEJAK_MAKS:
                del penuh[:-JEJAK_MAKS]
    except Exception:
        # Counting is a nice-to-have; never let it kill the run.
        pass
    return statistic


pp.update_object_info = _update_object_info
pp.flow_statistic = _flow_statistic
pp.Result.update = _result_update


# Batas getaran: titik roda harus berada lebih dari sekian piksel di masing-
# masing sisi garis. Kotak deteksi bergoyang beberapa piksel antar-frame, dan
# tanpa batas ini kendaraan yang melaju sejajar tepat di atas garis akan
# terbaca "menyeberang" bolak-balik.
LINTAS_TOLERANSI_PX = 3.0


# Syarat tambahan setelah uji lapangan: perpindahan sisi harus BERKELANJUTAN
# dan MULUS. Di Gerbang Pemuda, kotak satu track membesar menelan gerombolan
# motor di sekitarnya saat antre, titik tengah bawahnya melompat +-35 px
# melewati garis dalam beberapa frame, dan tercatat "melintas" padahal tidak
# satu roda pun berpindah lajur. Kendaraan sungguhan menyeberang dengan
# langkah kecil antar-frame; lompatan besar adalah artefak kotak.
LINTAS_MIN_TITIK = 3          # titik jelas di MASING-MASING sisi
# Diukur TEGAK LURUS garis, bukan jarak total: motor yang melaju bisa bergeser
# 20-40 px sejajar garis antar-frame, sedangkan lompatan kotak yang menelan
# gerombolan berupa lonjakan tegak lurus.
LINTAS_LANGKAH_MAKS_PX = 15.0


def _melintas(titik: list[tuple[float, float]], garis: list[list[float]]) -> bool:
    """Apakah titik roda sebuah track berpindah sisi pada salah satu ruas marka.

    Inilah arti "melintasi" pada Permenhub PM 67/2018 untuk marka membujur
    garis utuh. Aturan upstream PP-Vehicle lebih longgar - cukup sisi bawah
    KOTAK menyentuh garis - dan karena kotak selalu lebih lebar dari jejak
    rodanya akibat perspektif, antrean yang berhenti di lajurnya sendiri pun
    ikut tercatat.

    Tiga syarat, semuanya wajib:
      1. ada >= LINTAS_MIN_TITIK titik di tiap sisi, masing-masing lebih jauh
         dari LINTAS_TOLERANSI_PX (getaran kotak tidak dihitung);
      2. titik-titik itu berada dalam rentang panjang ruas garisnya;
      3. perpindahan sisi terjadi lewat langkah kecil TEGAK LURUS garis -
         bukan lompatan.
    """
    for g in garis:
        if len(g) < 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in g[:4])
        dx, dy = x2 - x1, y2 - y1
        panjang = (dx * dx + dy * dy) ** 0.5
        if panjang < 1:
            continue
        sisi: list[tuple[int, float]] = []             # (tanda, jarak bertanda)
        for (px, py) in titik:
            t = ((px - x1) * dx + (py - y1) * dy) / (panjang * panjang)
            if t < -0.02 or t > 1.02:
                continue
            jarak = ((px - x1) * dy - (py - y1) * dx) / panjang
            if abs(jarak) > LINTAS_TOLERANSI_PX:
                sisi.append((1 if jarak > 0 else -1, jarak))
        if (sum(1 for s_ in sisi if s_[0] > 0) < LINTAS_MIN_TITIK
                or sum(1 for s_ in sisi if s_[0] < 0) < LINTAS_MIN_TITIK):
            continue
        # Cari titik pergantian sisi; langkah di sekitarnya harus kecil.
        for (a, ja), (b, jb) in zip(sisi, sisi[1:]):
            if a != b and abs(jb - ja) <= LINTAS_LANGKAH_MAKS_PX:
                return True
    return False


def _fps_sumber() -> float:
    """fps sebenarnya dari berkas sumber.

    COUNTS["fps"] hanya terisi lewat update_object_info, yang cuma dipanggil
    bila penghitungan region atau parkir liar menyala. Pada skenario seperti
    press_line keduanya mati, sehingga fps tinggal 0 dan setiap durasi dihitung
    dengan asumsi 1 fps - sebuah klip 10 detik dilaporkan 28 detik. Membaca fps
    langsung dari berkas menutup lubang itu untuk semua skenario.
    """
    src = os.environ.get("LALIN_SOURCE") or ""
    if not src:
        # Jalur `lalin run` tidak memakai LALIN_SOURCE - sumbernya diteruskan
        # sebagai flag ke pipeline upstream.
        for bendera in ("--video_file", "--rtsp", "--image_file"):
            if bendera in sys.argv:
                src = sys.argv[sys.argv.index(bendera) + 1]
                break
    if src and os.path.exists(src):
        try:
            import cv2
            cap = cv2.VideoCapture(src)
            nilai = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            cap.release()
            if 1.0 < nilai < 121.0:
                return nilai
        except Exception:
            pass
    return 0.0


def _write(status: str, error: str | None = None) -> None:
    fps = COUNTS["fps"] or _fps_sumber() or 1.0
    events = []
    for rec in PARKING.values():
        duration = (rec["last_frame"] - rec["first_frame"]) / fps
        events.append({
            **rec,
            "kind": "parkir_liar",
            "duration_sec": round(duration + rec["threshold_sec"], 1),
            "first_sec": round(rec["first_frame"] / fps, 1),
        })
    garis_uji = MARKA_MANUAL or [g for lajur in LANES for g in lajur]
    ditolak = 0
    ditolak_rinci: list[dict] = []
    for rec in PRESSING.values():
        if rec["frames"] < PRESS_MIN_FRAMES:
            continue
        jejak = JEJAK_PENUH.get(rec["track_id"], [])
        if not _melintas(jejak, garis_uji):
            ditolak += 1                 # menyentuh, tidak melintas
            langkah = max(1, len(jejak) // 40)
            ditolak_rinci.append({
                "track_id": rec["track_id"], "first_frame": rec["first_frame"],
                "last_frame": rec["last_frame"], "bbox": rec.get("bbox"),
                "jejak": [[round(x, 1), round(y, 1)] for x, y in jejak[::langkah]]})
            continue
        # Jejak roda ikut disimpan (dijarangkan ke <= 40 titik) supaya foto
        # bukti bisa memperlihatkan DASAR keputusannya - lintasan roda yang
        # berpindah sisi - bukan sekadar kotak di satu frame.
        langkah = max(1, len(jejak) // 40)
        events.append({
            "jejak": [[round(x, 1), round(y, 1)] for x, y in jejak[::langkah]],
            "marka": garis_uji,
            **rec,
            "kind": "langgar_marka",
            "duration_sec": round((rec["last_frame"] - rec["first_frame"]) / fps, 1),
            "first_sec": round(rec["first_frame"] / fps, 1),
        })
    for rec in RETROGRADE.values():
        events.append({
            **rec,
            "kind": "lawan_arah",
            "duration_sec": round((rec["last_frame"] - rec["first_frame"]) / fps, 1),
            "first_sec": round(rec["first_frame"] / fps, 1),
        })
    events.sort(key=lambda e: e["first_frame"])

    # Tandai kotak milik kendaraan yang melanggar supaya dashboard bisa
    # membedakannya tanpa mencocokkan ulang.
    melanggar = {e["track_id"] for e in events}
    boxes = [{**b, "bad": b["id"] in melanggar} for b in LAST_BOXES]
    trails = [{"id": b["id"], "pts": TRAILS.get(b["id"], [])}
              for b in LAST_BOXES if len(TRAILS.get(b["id"], [])) > 3]

    payload = {
        "status": status,
        "error": error,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frames": COUNTS["frames"],
        "video_fps": fps,
        "unique_tracks": len(SEEN_IDS),
        "entrance_in": COUNTS["in"],
        "entrance_out": COUNTS["out"],
        "count_mode": COUNTS["mode"],
        # Frekuensi label atribut, terbanyak dulu. Daftar mentahnya panjang
        # (satu entri per kendaraan per frame), yang berguna adalah sebarannya.
        "attributes": [{"label": k, "n": v} for k, v in
                       sorted(Counter(ATTRS).items(), key=lambda x: -x[1])[:12]],
        "attr_samples": len(ATTRS),
        "plates": [{"text": k, "n": v} for k, v in
                   sorted(Counter(PLATES).items(), key=lambda x: -x[1])[:10]],
        "plate_reads": len(PLATES),
        "events": events,
        "lanes": LANES,
        "lanes_manual": bool(MARKA_MANUAL),
        # Kandidat yang kotaknya menyentuh marka tetapi rodanya tidak pernah
        # berpindah sisi - dicatat supaya penyaringan ini bisa diaudit.
        "langgar_marka_ditolak": ditolak,
        "langgar_marka_ditolak_rinci": ditolak_rinci[:20],
        "fence_line": FENCE,
        "boxes": boxes,
        "trails": trails,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> int:
    paddle.enable_static()
    parser = pp.argsparser()
    flags = parser.parse_args()
    flags.device = flags.device.upper()
    assert flags.device in ["CPU", "GPU", "XPU", "NPU", "GCU"], \
        "device should be CPU, GPU, XPU, NPU or GCU"
    pp.FLAGS = flags          # pipeline.main() reads this module global

    # Keluar PAKSA lewat os._exit, baik gagal maupun berhasil. Pipeline
    # upstream menyalakan thread pembaca video yang bukan daemon; bila ia
    # berhenti di tengah - misalnya karena AssertionError konfigurasi -
    # thread itu tetap hidup, Python menunggunya saat keluar, dan proses ini
    # menggantung selamanya dengan CPU 0%. Tercatat 24-25 Sep 2026: satu run
    # CCTV-01 menggantung 19 jam dan membekukan seluruh siklus otomatis.
    import traceback
    try:
        pp.main()
    except BaseException as exc:
        _write("failed", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    _write("ok")
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
