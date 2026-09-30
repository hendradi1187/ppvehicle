"""Periodic batch analysis of the configured cameras.

Continuous per-camera inference needs a GPU. On CPU the pipeline runs at
roughly 1.4 fps, so a cycle over four cameras takes minutes — this worker
therefore treats each camera as a short clip captured on a schedule, analysed,
and published as "the most recent look at this camera" rather than pretending
to be a live stream. The dashboard shows the capture time so the lag is
visible instead of implied.

Each cycle, per camera:
  1. ffmpeg records `clip_seconds` from the HLS stream at `analysis_fps`
  2. the configured scenario runs over that clip
  3. the last annotated frame is extracted as the camera's current still
  4. the run's results.json plus that frame are stored under runs/_latest/
"""

from __future__ import annotations

import base64
import json
import subprocess
import threading
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import config, runner, store
from .settings import settings

LATEST_DIR = settings.runs_dir / "_latest"
CAMERAS_FILE = settings.configs_dir / "cameras.yml"

# Zona yang digambar operator lewat editor visual disimpan TERPISAH dan
# menimpa cameras.yml saat dimuat. Menulis balik ke cameras.yml akan
# menghancurkan komentarnya - dan komentar itu justru isinya paling berharga:
# hasil survei kamera, kalibrasi fps, alasan tiap ambang berbeda. Pemisahan
# ini juga membuat asal-usul tiap nilai jelas: yang ditulis tangan tetap di
# cameras.yml, yang disetel operator ada di sini.
ZONES_FILE = settings.configs_dir / "zones.yml"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Camera:
    key: str
    id: str
    name: str
    area: str = ""
    region: str = ""          # wilayah DKI, untuk pengelompokan di dashboard
    owner: str = ""
    scenario: str = "tracking"
    clip_seconds: int = 12
    analysis_fps: int = 5
    illegal_parking_time: int | None = None
    region_polygon: list[int] = field(default_factory=list)
    zone_label: str = ""
    base_url: str = ""
    # RTSP langsung dari kamera (tab "RTSP langsung" di Manajemen Kamera).
    # Sengaja field BARU, bukan menghidupkan `source_url` lama di cameras.yml:
    # field itu selama ini dibuang pemuat karena bukan field dataclass, dan
    # menghidupkannya akan diam-diam memindahkan CCTV-01 ke relay RTSP proyek
    # lain yang sempat mati 24 Sep 2026.
    rtsp_url: str = ""

    @property
    def stream_url(self) -> str:
        if self.rtsp_url:
            return self.rtsp_url
        return f"{self.base_url}{self.id}/index.m3u8"

    @property
    def clip_path(self) -> Path:
        return settings.uploads_dir / f"_batch_{self.key}.mp4"

    @property
    def frame_path(self) -> Path:
        """Frame bersih tanpa anotasi - yang ditampilkan dashboard."""
        return LATEST_DIR / f"{self.key}.jpg"

    @property
    def frame_pipeline_path(self) -> Path:
        """Frame beranotasi milik pipeline, apa adanya. Tidak dipakai dashboard
        karena tulisannya berukuran tetap dan di 640x360 saling menimpa, tetapi
        disimpan supaya keluaran mentah tetap bisa diperiksa tim teknis."""
        return LATEST_DIR / f"{self.key}_pipeline.jpg"

    @property
    def record_path(self) -> Path:
        return LATEST_DIR / f"{self.key}.json"

    @property
    def hitung_path(self) -> Path:
        """Hasil penghitungan per kelas. Terpisah dari record biasa karena
        mesinnya lain - detektor COCO, bukan PP-Vehicle - dan angkanya tidak
        sebanding dengan angka toolbox."""
        return LATEST_DIR / f"{self.key}_hitung.json"

    @property
    def hitung_frame_path(self) -> Path:
        return LATEST_DIR / f"{self.key}_hitung.jpg"


MARKA_FILE = settings.configs_dir / "marka.yml"


def load_marka() -> dict[str, list[list[float]]]:
    """Garis marka yang ditetapkan operator, per kamera.

    Dipisahkan dari zones.yml karena maknanya berbeda: zona adalah AREA untuk
    penghitungan dan parkir liar, marka adalah GARIS yang tidak boleh dilintasi.
    Menyatukan keduanya akan membuat editor zona diam-diam mengubah hasil
    pelanggaran marka.
    """
    if not MARKA_FILE.exists():
        return {}
    try:
        with open(MARKA_FILE, encoding="utf-8") as f:
            data = (yaml.safe_load(f) or {}).get("marka") or {}
    except (OSError, yaml.YAMLError):
        return {}
    bersih: dict[str, list[list[float]]] = {}
    for key, entri in data.items():
        ruas = (entri or {}).get("garis") or []
        valid = [[float(v) for v in g[:4]] for g in ruas if len(g) >= 4]
        if valid:
            bersih[key] = valid
    return bersih


KEPALA_MARKA = """# Garis marka yang ditetapkan operator, koordinat frame 640x360.
# Dipakai skenario press_line untuk menguji pelanggaran marka.
#
# Ada sebabnya garis ini manual: pada feed publik 640x360, PP-LiteSeg tidak
# menemukan marka jalan - ia mengikuti pagar pembatas, sehingga kendaraan yang
# berjalan normal di dekat pagar dilaporkan melanggar. Selama resolusi feed
# belum naik, geometri yang bisa dipertanggungjawabkan hanya yang ditetapkan
# manusia.

"""


def simpan_marka(key: str, garis: list[list[float]]) -> dict:
    """Simpan garis marka satu kamera. Daftar kosong menghapus entrinya."""
    if not MARKA_FILE.exists():
        data: dict[str, Any] = {}
    else:
        try:
            with open(MARKA_FILE, encoding="utf-8") as f:
                data = (yaml.safe_load(f) or {}).get("marka") or {}
        except (OSError, yaml.YAMLError):
            data = {}
    valid = [[float(v) for v in g[:4]] for g in (garis or []) if len(g) >= 4]
    if valid:
        data[key] = {"garis": valid, "updated_at": _now()}
    else:
        data.pop(key, None)
    MARKA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MARKA_FILE, "w", encoding="utf-8") as f:
        f.write(KEPALA_MARKA)
        yaml.safe_dump({"marka": data}, f, sort_keys=True, allow_unicode=True)
    return {"key": key, "garis": valid}


GARIS_HITUNG_FILE = settings.configs_dir / "garis_hitung.yml"

KEPALA_GARIS_HITUNG = """# Garis hitung lalu lintas per kamera, koordinat frame 640x360.
# Dipakai penghitung real-time di Live (bbox peramban): kendaraan dihitung
# saat titik roda bawahnya MELINTASI salah satu ruas di sini.
#
# Tiap ruas ditarik melintang badan jalan, dari tepi ke tepi, dan berhenti di
# kerb - bukan garis datar selebar frame. Garis selebar frame ikut memotong
# trotoar, lapak, dan motor parkir, sehingga objek diam yang bergoyang di
# dekatnya ikut terhitung. Kamera tanpa entri di sini tidak menghitung sama
# sekali (lebih jujur daripada garis tebakan).

"""


# Kamera DKI banyak yang PTZ: posisinya bisa digeser operator Diskominfotik
# kapan saja. Garis yang ditarik untuk satu posisi tidak berlaku di posisi
# lain, jadi garis disimpan per TAMPILAN, masing-masing dengan gambar acuan
# kecil (80x36 abu-abu, tepi atas/bawah berisi teks OSD dibuang). Peramban
# mencocokkan video dengan acuan itu dan hanya menghitung bila cocok.
ACUAN_W, ACUAN_H = 80, 36
TAMPILAN_MAKS = 4


def _acuan_sah(acuan: Any) -> str | None:
    if not acuan:
        return None
    try:
        mentah = base64.b64decode(str(acuan), validate=True)
    except (ValueError, TypeError):
        return None
    return str(acuan) if len(mentah) == ACUAN_W * ACUAN_H else None


def _baca_garis_hitung_mentah() -> dict[str, Any]:
    if not GARIS_HITUNG_FILE.exists():
        return {}
    try:
        with open(GARIS_HITUNG_FILE, encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("garis_hitung") or {}
    except (OSError, yaml.YAMLError):
        return {}


def _ruas_sah(garis: Any) -> list[list[float]]:
    return [[round(float(v), 1) for v in g[:4]] for g in (garis or []) if len(g) >= 4]


def _poli_sah(poli: Any) -> list[list[float]] | None:
    """LC-015: poligon [[x, y], ...] (>= 3 titik) pada frame 640x360, atau None."""
    try:
        titik = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in (poli or [])]
    except (TypeError, ValueError, IndexError):
        return None
    return titik if len(titik) >= 3 else None


def load_tampilan_hitung() -> dict[str, list[dict]]:
    """{key: [{"garis": [[x1,y1,x2,y2], ...], "acuan": base64|None}, ...]}.

    Format lama (satu daftar `garis` langsung di bawah kamera) dibaca sebagai
    satu tampilan tanpa acuan.
    """
    hasil: dict[str, list[dict]] = {}
    for key, entri in _baca_garis_hitung_mentah().items():
        entri = entri or {}
        daftar = entri.get("tampilan")
        if daftar is None and entri.get("garis"):
            daftar = [{"garis": entri.get("garis")}]
        bersih = []
        for t in daftar or []:
            ruas = _ruas_sah((t or {}).get("garis"))
            if ruas:
                bersih.append({"garis": ruas, "acuan": _acuan_sah(t.get("acuan")),
                               "acuan_2": _acuan_sah(t.get("acuan_2")),
                               # LC-015: garis khusus pejalan kaki & area badan jalan
                               "garis_pejalan": _ruas_sah(t.get("garis_pejalan")) or None,
                               "badan_jalan": _poli_sah(t.get("badan_jalan")),
                               "oleh": t.get("oleh"),
                               "updated_at": t.get("updated_at")})
        if bersih:
            hasil[key] = bersih
    return hasil


def load_garis_hitung() -> dict[str, list[list[float]]]:
    """Garis tampilan pertama per kamera (untuk pembaca lama)."""
    return {k: v[0]["garis"] for k, v in load_tampilan_hitung().items()}


def simpan_garis_hitung(key: str, garis: list[list[float]], acuan: str | None = None,
                        indeks: int | None = None, oleh: str = "operator") -> dict:
    """Simpan garis untuk satu tampilan kamera.

    `indeks` = tampilan yang sedang cocok di peramban: garisnya diganti (atau
    tampilan dihapus bila `garis` kosong). Tanpa `indeks`, posisi kamera ini
    belum pernah digaris, jadi ia ditambahkan sebagai tampilan baru.
    """
    semua = load_tampilan_hitung()
    daftar = [dict(t) for t in semua.get(key, [])]
    ruas = _ruas_sah(garis)
    acuan = _acuan_sah(acuan)
    baru = {"garis": ruas, "acuan": acuan, "updated_at": _now(), "oleh": oleh}
    if indeks is not None and 0 <= indeks < len(daftar):
        if ruas:
            lama = daftar[indeks].get("acuan")
            # LC-014: acuan kedua (mis. malam) dipertahankan. Sebelumnya ikut
            # terbuang setiap kali garis diganti tanpa acuan baru.
            baru["acuan_2"] = daftar[indeks].get("acuan_2")
            # LC-015: garis pejalan kaki & badan jalan (ditetapkan di berkas)
            # tidak ikut terbuang saat operator menyimpan garis kendaraan.
            for k in ("garis_pejalan", "badan_jalan"):
                baru[k] = daftar[indeks].get(k)
            if not acuan:
                baru["acuan"] = lama
            elif lama and lama != acuan:
                # Acuan lama disimpan sebagai acuan kedua: posisi PTZ yang sama
                # terlihat sangat berbeda siang vs malam, dan satu acuan saja
                # membuat hitungan terjeda saat terang berganti.
                baru["acuan_2"] = lama
            daftar[indeks] = baru
        else:
            daftar.pop(indeks)
    elif ruas:
        daftar.append(baru)
        daftar = daftar[-TAMPILAN_MAKS:]
    data = _baca_garis_hitung_mentah()
    if daftar:
        data[key] = {"tampilan": [{k: v for k, v in t.items() if v is not None}
                                  for t in daftar]}
    else:
        data.pop(key, None)
    GARIS_HITUNG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GARIS_HITUNG_FILE, "w", encoding="utf-8") as f:
        f.write(KEPALA_GARIS_HITUNG)
        yaml.safe_dump({"garis_hitung": data}, f, sort_keys=True, allow_unicode=True,
                       width=4096)
    return {"key": key, "tampilan": load_tampilan_hitung().get(key, [])}


def load_zone_overrides() -> dict[str, dict]:
    if not ZONES_FILE.exists():
        return {}
    try:
        with open(ZONES_FILE, encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("zones") or {}
    except (OSError, yaml.YAMLError):
        return {}


def save_zone_override(key: str, polygon: list[int], label: str = "",
                       parking_sec: int | None = None) -> dict:
    """Simpan satu zona hasil editor. Menimpa entri lama untuk kamera itu."""
    zones = load_zone_overrides()
    entry: dict[str, Any] = {
        "region_polygon": [int(v) for v in polygon],
        "updated_at": _now(),
    }
    if label:
        entry["zone_label"] = label
    if parking_sec is not None:
        entry["illegal_parking_time"] = int(parking_sec)
    zones[key] = entry

    ZONES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ZONES_FILE, "w", encoding="utf-8") as f:
        f.write(
            "# Ditulis oleh editor zona visual (/zona) - JANGAN disunting\n"
            "# tangan kecuali paham: berkas ini MENIMPA region_polygon,\n"
            "# zone_label dan illegal_parking_time dari cameras.yml.\n"
            "# Hapus entri di sini untuk kembali ke nilai di cameras.yml.\n\n")
        yaml.safe_dump({"zones": zones}, f, sort_keys=True, allow_unicode=True)
    return entry


CUSTOM_FILE = settings.configs_dir / "cameras_custom.yml"

# Field yang boleh disunting lewat UI. Sengaja daftar putih, bukan daftar
# hitam: field baru pada Camera tidak otomatis menjadi bisa disunting orang
# lewat jaringan.
FIELD_BOLEH = {
    "id", "name", "area", "region", "owner", "scenario",
    "clip_seconds", "analysis_fps", "illegal_parking_time", "zone_label",
    "rtsp_url",
}


def load_camera_custom() -> dict:
    """Suntingan kamera dari UI. Kosong kalau berkasnya belum ada."""
    if not CUSTOM_FILE.exists():
        return {"ubah": {}, "tambahan": [], "sembunyi": []}
    try:
        with open(CUSTOM_FILE, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {"ubah": {}, "tambahan": [], "sembunyi": []}
    return {
        "ubah": d.get("ubah") or {},
        "tambahan": d.get("tambahan") or [],
        "sembunyi": d.get("sembunyi") or [],
    }


def _tulis_custom(data: dict) -> None:
    CUSTOM_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CUSTOM_FILE, "w", encoding="utf-8") as f:
        f.write("# Ditulis oleh halaman Manajemen Kamera - JANGAN disunting tangan.\n"
                "# Berkas ini MENIMPA configs/cameras.yml; catatan survei di sana\n"
                "# sengaja tidak pernah ditulis ulang supaya tidak hilang.\n"
                f"# Terakhir diubah: {_now()}\n\n")
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _kunci_bawaan() -> set[str]:
    with open(CAMERAS_FILE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return {e.get("key") for e in (cfg.get("cameras") or [])}


def simpan_kamera(key: str, data: dict) -> dict:
    """Simpan satu kamera. Menimpa kalau ia berasal dari cameras.yml,
    menambah/memperbarui daftar tambahan kalau tidak."""
    key = (key or "").strip()
    if not key or not key.replace("_", "").replace("-", "").isalnum():
        raise ValueError("key hanya boleh huruf, angka, tanda hubung, garis bawah")
    bersih = {k: v for k, v in data.items() if k in FIELD_BOLEH and v not in (None, "")}
    if not bersih.get("name"):
        raise ValueError("nama kamera wajib diisi")
    rtsp = str(bersih.get("rtsp_url") or "")
    if rtsp and not rtsp.lower().startswith(("rtsp://", "rtsps://")):
        raise ValueError("alamat RTSP harus diawali rtsp:// atau rtsps://")
    if not bersih.get("id") and not rtsp:
        raise ValueError("isi ID kamera portal atau alamat RTSP")

    custom = load_camera_custom()
    custom["sembunyi"] = [k for k in custom["sembunyi"] if k != key]
    if key in _kunci_bawaan():
        custom["ubah"][key] = bersih
    else:
        tambahan = [t for t in custom["tambahan"] if t.get("key") != key]
        tambahan.append({"key": key, **bersih})
        custom["tambahan"] = tambahan
    _tulis_custom(custom)
    return {"key": key, "bawaan": key in _kunci_bawaan()}


def hapus_kamera(key: str) -> dict:
    """Kamera bawaan tidak dihapus dari cameras.yml - ia disembunyikan, supaya
    catatan surveinya tetap utuh dan keputusannya bisa dibatalkan."""
    custom = load_camera_custom()
    bawaan = key in _kunci_bawaan()
    if bawaan:
        if key not in custom["sembunyi"]:
            custom["sembunyi"].append(key)
        custom["ubah"].pop(key, None)
    else:
        custom["tambahan"] = [t for t in custom["tambahan"] if t.get("key") != key]
    _tulis_custom(custom)
    return {"key": key, "disembunyikan": bawaan, "dihapus": not bawaan}


def pulihkan_kamera(key: str) -> dict:
    """Kembalikan kamera bawaan yang disembunyikan, dan buang suntingannya."""
    custom = load_camera_custom()
    custom["sembunyi"] = [k for k in custom["sembunyi"] if k != key]
    custom["ubah"].pop(key, None)
    _tulis_custom(custom)
    return {"key": key, "dipulihkan": True}


# LC-016: server portal yang boleh dipakai per kamera. Hanya dibaca dari
# configs/cameras.yml (bukan dari suntingan UI): server menyambung ke alamat
# ini, jadi alamatnya tidak boleh bisa diisi sembarang dari peramban.
BASE_URL_SAH = (
    "https://dki-jkt.balitower.co.id:7028/",     # portal jakcctv, 640x360
    "https://cctv-jsc.balitower.co.id:8011/",    # portal jakcctv + CCTV kelurahan, HD
)


def load_cameras() -> list[Camera]:
    if not CAMERAS_FILE.exists():
        raise FileNotFoundError(f"Missing {CAMERAS_FILE}")
    with open(CAMERAS_FILE, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.get("base_url", "")
    defaults = cfg.get("defaults") or {}
    overrides = load_zone_overrides()
    custom = load_camera_custom()
    known = {f for f in Camera.__dataclass_fields__}

    # Urutan penimpaan, dari lemah ke kuat:
    #   defaults -> entri cameras.yml -> suntingan UI -> zona dari editor
    # Zona ditaruh paling akhir dengan sengaja: ia disunting lewat kanvas dan
    # tidak boleh terhapus hanya karena seseorang menyimpan nama kamera.
    daftar = [(e, True) for e in (cfg.get("cameras") or [])
              if e.get("key") not in custom["sembunyi"]]
    daftar += [(dict(t), False) for t in custom["tambahan"]]

    out = []
    for entry, dari_cfg in daftar:
        key = entry.get("key")
        merged = {**defaults, **entry}
        merged.update(custom["ubah"].get(key, {}))
        merged.update(overrides.get(key, {}))
        pilihan = entry.get("base_url") if dari_cfg else None
        merged["base_url"] = pilihan if pilihan in BASE_URL_SAH else base
        out.append(Camera(**{k: v for k, v in merged.items() if k in known}))
    return out


def _opsi_masukan(url: str) -> list[str]:
    """RTSP lewat TCP: di atas UDP, paket hilang di jaringan kota membuat frame
    rusak berkotak-kotak, dan detektor membaca kotak-kotak itu sebagai objek."""
    return ["-rtsp_transport", "tcp", "-timeout", "15000000"] if url.lower().startswith("rtsp") else []


def _opsi_skala(url: str) -> list[str]:
    """Kamera RTSP diperkecil ke 640x360 - resolusi feed portal.

    Seluruh aplikasi - overlay dashboard, zona, garis marka, HUD Live - bekerja
    dalam koordinat frame 640x360. Tanpa penyamaan ini, zona yang digambar di
    snapshot 1080p akan jatuh di tempat yang salah pada frame analisis.
    Harganya: OCR pelat tidak mendapat manfaat dari resolusi asli kamera.
    Menaikkan aplikasi ke resolusi asli adalah pekerjaan tersendiri.
    """
    return ["-vf", "scale=640:360"] if url.lower().startswith("rtsp") else []


def ambil_snapshot(url: str, timeout: int = 30) -> bytes:
    """Satu frame JPEG 640x360 dari sumber - untuk menggambar zona dan marka."""
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    argv += _opsi_masukan(url)
    argv += ["-i", url, "-frames:v", "1", "-vf", "scale=640:360",
             "-f", "image2", "-c:v", "mjpeg", "-q:v", "3", "pipe:1"]
    proc = subprocess.run(argv, capture_output=True, stdin=subprocess.DEVNULL,
                          timeout=timeout)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("snapshot gagal: "
                           + proc.stderr.decode("utf-8", "replace").strip()[:300])
    return proc.stdout


def capture_stream(url: str, dest: Path, seconds: int, fps: int,
                   timeout: int = 300) -> Path:
    """Record `seconds` from a live stream into `dest`.

    A live HLS stream never ends, so it cannot be fed to the pipeline directly:
    the job would run forever. Everything that analyses a camera goes through
    here first, which is also why the wall-clock floor per camera is the clip
    length itself.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # -nostdin matters: ffmpeg inherits this process's stdin, and on a broken
    # or unexpected input it blocks waiting for interactive keys instead of
    # exiting - which hangs the whole cycle.
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    argv += _opsi_masukan(url)
    argv += ["-i", url, "-t", str(int(seconds)), "-r", str(int(fps))]
    argv += _opsi_skala(url)
    argv += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-y", str(dest)]
    proc = subprocess.run(argv, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=timeout)
    if proc.returncode != 0 or not dest.exists():
        raise RuntimeError(f"ffmpeg gagal: {proc.stderr.strip()[:300]}")
    return dest


def capture(cam: Camera, timeout: int = 180) -> Path:
    """Record a short clip from the camera's HLS stream."""
    return capture_stream(cam.stream_url, cam.clip_path,
                          cam.clip_seconds, cam.analysis_fps, timeout)


def extract_last_frame(video: Path, dest: Path) -> bool:
    """Grab the final frame of a video as a still."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # -sseof seeks from the end, so this does not depend on knowing the length.
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
         "-sseof", "-1", "-i", str(video), "-update", "1", "-q:v", "3",
         "-y", str(dest)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    return proc.returncode == 0 and dest.exists()


# Satu analisis pada satu waktu untuk seluruh proses. Dua pemanggilan untuk
# kamera yang sama - misalnya tombol "Analisis kamera ini" ditekan saat siklus
# batch kebetulan sampai di kamera itu - akan menulis berkas klip, record, dan
# frame yang sama secara bersamaan, sehingga hasilnya campur aduk. Serialisasi
# juga sejalan dengan alasan Worker memproses satu kamera pada satu waktu: pada
# CPU dua pipeline paralel lebih lambat daripada dua yang berurutan.
_analysis_lock = threading.Lock()


def analysis_busy() -> bool:
    """Benar bila ada analisis sedang berjalan. Dipakai API untuk menolak
    permintaan sinkron alih-alih menggantungnya selama beberapa menit."""
    locked = _analysis_lock.acquire(blocking=False)
    if locked:
        _analysis_lock.release()
    return not locked


def analyse(cam: Camera, profile: str | None = None) -> dict:
    """Capture, run the scenario, and store the result. Returns the record."""
    with _analysis_lock:
        return _analyse_locked(cam, profile)


def _analyse_locked(cam: Camera, profile: str | None = None) -> dict:
    started = time.time()
    record: dict[str, Any] = {
        "key": cam.key, "name": cam.name, "area": cam.area,
        "region": cam.region, "owner": cam.owner, "zone_label": cam.zone_label,
        "scenario": cam.scenario, "started_at": _now(),
    }
    try:
        # Dijaga SEBELUM menarik klip, bukan sesudah. Skenario seperti
        # empat_toolbox dan illegal_parking memasang `region_type: custom` di
        # berkas skenarionya; tanpa poligon, upstream berhenti dengan
        # AssertionError di dalam SDE_Detector - pesan yang tidak memberi tahu
        # operator bahwa yang kurang adalah zona. Kamera cctv01 jatuh persis di
        # lubang ini saat pertama kali ditambahkan.
        if not cam.region_polygon:
            try:
                butuh_zona = (config.load_scenario(cam.scenario)
                              .get("args", {}).get("region_type") == "custom")
            except Exception:
                butuh_zona = False
            if butuh_zona:
                raise ValueError(
                    f"Skenario '{cam.scenario}' memerlukan zona, tetapi kamera "
                    f"'{cam.key}' belum punya. Gambar zonanya di halaman Zona, "
                    f"atau pilih skenario yang tidak memerlukan zona.")

        capture(cam)

        overrides: dict[str, Any] = {}
        args: dict[str, Any] = {}
        if cam.region_polygon:
            args["region_polygon"] = cam.region_polygon
            args["region_type"] = "custom"
            # Penghitungan garis-lintas menuntut region_type horizontal/vertical,
            # zona menuntut custom; upstream menolak kombinasinya dengan
            # AssertionError. Begitu sebuah kamera diberi zona, penghitungannya
            # dialihkan ke masuk-zona - mode yang memang sah bersama zona - dan
            # pengalihan itu dicatat di rekaman. Tercatat 24 Sep 2026: zona
            # CCTV-01 digambar di Editor Zona sementara skenarionya `full`.
            try:
                arg_skenario = config.load_scenario(cam.scenario).get("args") or {}
            except Exception:
                arg_skenario = {}
            if arg_skenario.get("do_entrance_counting"):
                args["do_entrance_counting"] = False
                args["do_break_in_counting"] = True
                record["catatan"] = ("penghitungan garis-lintas dialihkan ke masuk-zona "
                                     "karena kamera ini punya zona")
        if cam.illegal_parking_time is not None:
            args["illegal_parking_time"] = cam.illegal_parking_time
        if args:
            overrides["args"] = args

        spec = runner.RunSpec(scenario=cam.scenario, source=str(cam.clip_path),
                              profile=profile, overrides=overrides,
                              marka=load_marka().get(cam.key) or [],
                              run_id=f"batch-{cam.key}-{datetime.now():%H%M%S}")
        result = runner.run(spec, download_missing=True)

        record["returncode"] = result["returncode"]
        record["elapsed_sec"] = result["elapsed_sec"]
        record["results"] = result.get("results")
        record["ok"] = result["returncode"] == 0

        videos = [Path(p) for p in result["artifacts"] if p.endswith(".mp4")]
        if videos:
            extract_last_frame(videos[0], cam.frame_pipeline_path)
        # Video beranotasi dibuang begitu frame-nya diambil. Tidak ada yang
        # memakainya sesudah ini - dashboard memakai frame bersih, bukti memakai
        # foto per kejadian - dan tiap video +-5 MB: diukur 25 Sep 2026, 128 run
        # memakan 490 MB dan tumbuh +-2 GB per hari.
        for v in videos:
            try:
                v.unlink()
            except OSError:
                pass
        # Gambar yang dilihat petugas diambil dari klip MENTAH, bukan dari video
        # beranotasi. Pipeline menulis satu frame keluaran per frame masukan,
        # jadi frame terakhir keduanya adalah detik yang sama - hanya saja yang
        # ini masih bersih, sehingga kotak dan label bisa digambar ulang
        # seperlunya oleh dashboard.
        record["frame"] = extract_last_frame(cam.clip_path, cam.frame_path)
    except Exception as exc:
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["finished_at"] = _now()
        record["total_sec"] = round(time.time() - started, 1)
        LATEST_DIR.mkdir(parents=True, exist_ok=True)
        try:
            record["cycle_id"] = store.record_cycle(record)
            simpan_bukti(cam, record)
        except Exception as exc:
            # The recap is secondary; never lose an analysis over a store error.
            record["store_error"] = f"{type(exc).__name__}: {exc}"
        # Ditulis SETELAH siklus tercatat, supaya cycle_id ikut tersimpan -
        # dashboard memakainya untuk tahu apakah frame yang ditampilkan berasal
        # dari siklus yang sama dengan temuan yang sedang dibuka.
        with open(cam.record_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    return record


BUKTI_DIR = settings.runs_dir / "_bukti"
LABEL_BUKTI = {"parkir_liar": "PARKIR LIAR", "langgar_marka": "LANGGAR MARKA",
               "lawan_arah": "LAWAN ARAH"}


def bukti_path(event_id: int) -> Path:
    return BUKTI_DIR / f"{int(event_id)}.jpg"


def simpan_bukti(cam: Camera, record: dict) -> None:
    """Satu foto bukti per kejadian, diambil dari klip siklus kejadian itu.

    Sebelumnya dialog verifikasi memperlihatkan frame pemeriksaan TERAKHIR
    kamera - untuk kejadian dari siklus sebelumnya gambar itu bukan momen
    kejadiannya, sehingga petugas memverifikasi tanpa bukti. Klip siklus
    ditimpa pada siklus berikutnya, jadi foto harus diambil sekarang.

    Frame yang diambil adalah last_frame bila kejadian membawa bbox, karena
    bbox yang tersimpan adalah posisi terakhir kendaraan; tanpa bbox (lawan
    arah) dipakai first_frame.
    """
    evs = [e for e in (record.get("results") or {}).get("events") or []
           if e.get("event_id")]
    if not evs or not cam.clip_path.is_file():
        return
    try:
        import cv2
    except Exception:
        return
    BUKTI_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(cam.clip_path))
    try:
        for e in evs:
            bbox = e.get("bbox")
            idx = e.get("last_frame") if bbox else e.get("first_frame")
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(idx or 0) - 1))
            ok, frame = cap.read()
            if not ok:
                continue
            # Marka acuan (kuning) dan jejak roda (sian) digambar lebih dulu,
            # di bawah kotak: petugas harus bisa melihat apakah roda benar-benar
            # berpindah sisi garis, bukan hanya bahwa kotaknya menyentuhnya.
            for g in e.get("marka") or []:
                cv2.line(frame, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])),
                         (0, 210, 255), 2, cv2.LINE_AA)
            titik = [(int(x), int(y)) for x, y in (e.get("jejak") or [])]
            for a, b in zip(titik, titik[1:]):
                cv2.line(frame, a, b, (230, 220, 40), 2, cv2.LINE_AA)
            for t in titik[::4]:
                cv2.circle(frame, t, 2, (230, 220, 40), -1)
            teks = f"{LABEL_BUKTI.get(e.get('kind'), str(e.get('kind')).upper())} #{e.get('track_id')}"
            if bbox and len(bbox) >= 4:
                x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 230), 2)
                ty = y1 - 6 if y1 > 18 else y2 + 14
                (tw, th), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                tx = max(0, min(x1, frame.shape[1] - tw - 6))
                cv2.rectangle(frame, (tx, ty - th - 4), (tx + tw + 6, ty + 3), (40, 40, 230), -1)
                cv2.putText(frame, teks, (tx + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(bukti_path(e["event_id"])), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
    except Exception:
        pass            # bukti adalah pelengkap; siklus tidak boleh gagal karenanya
    finally:
        cap.release()


# Direktori run yang boleh dihapus, beserta umur simpannya. Pola dicocokkan
# pada NAMA direktori langsung di bawah runs/ - apa pun di luar pola ini
# (lalin.db dan cadangannya, _latest, _bukti) tidak pernah disentuh oleh
# penghapus direktori. Foto bukti punya aturannya sendiri di bawah.
_POLA_RUN = (
    (re.compile(r"^pantau-"), "pantau"),
    (re.compile(r"^(batch|hitung)-"), "run"),
    (re.compile(r"^\d{8}-\d{6}-"), "run"),       # run manual Konsol Operator / CLI
)


def bersihkan(kini: float | None = None, coba: bool = False) -> dict:
    """Terapkan retensi berkas. Mengembalikan jumlah yang dihapus.
    coba=True hanya menghitung, tidak menghapus apa pun."""
    kini = kini or time.time()
    batas = {"run": settings.retensi_run_hari * 86400,
             "pantau": settings.retensi_pantau_jam * 3600}
    hasil = {"run": 0, "pantau": 0, "bukti": 0, "byte": 0}
    akar = settings.runs_dir
    if not akar.is_dir():
        return hasil
    for d in akar.iterdir():
        if not d.is_dir() or d.is_symlink():
            continue
        jenis = next((j for pola, j in _POLA_RUN if pola.match(d.name)), None)
        if jenis is None:
            continue
        try:
            umur = kini - d.stat().st_mtime
        except OSError:
            continue
        if umur <= batas[jenis]:
            continue
        ukuran = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        if not coba:
            shutil.rmtree(d, ignore_errors=True)
        if coba or not d.exists():
            hasil[jenis] += 1
            hasil["byte"] += ukuran
    if BUKTI_DIR.is_dir():
        batas_bukti = settings.retensi_bukti_hari * 86400
        for f in BUKTI_DIR.glob("*.jpg"):
            try:
                if kini - f.stat().st_mtime > batas_bukti:
                    hasil["byte"] += f.stat().st_size
                    if not coba:
                        f.unlink()
                    hasil["bukti"] += 1
            except OSError:
                pass
    return hasil


def read_record(cam: Camera) -> dict | None:
    if not cam.record_path.is_file():
        return None
    try:
        with open(cam.record_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------- pantau
# 2 detik @ 3 fps = 6 frame. Live hanya menampilkan MEDIAN objek per frame,
# yang tidak butuh identitas antar-frame, dan median enam frame sudah stabil.
# Dengan PP-YOLOE+ L di 3 core (+-1,5-2 dtk/frame), 20 frame membuat satu
# putaran 56-67 dtk - diukur 24 Sep 2026 dengan siklus otomatis berjalan.
WATCH_SECONDS = 2          # panjang klip tiap putaran
WATCH_FPS = 3
# Pantau kini memakai mesin penghitung tujuh kelas (detektor COCO), bukan
# skenario tracking PP-Vehicle. Model PP-Vehicle dilatih dengan car, truck, bus
# dan van digabung menjadi satu kelas `vehicle` - SEPEDA MOTOR TIDAK TERMASUK -
# sehingga di mode Live deretan motor terparkir di trotoar tidak pernah terkotak
# sama sekali. Mesin tujuh kelas melihatnya, dan menamai tiap kotak.
WATCH_SCENARIO = "hitung7"
WATCH_IDLE_STOP = 90       # berhenti sendiri bila tidak ada yang menonton


HITUNG_SECONDS = 10        # klip untuk satu pengukuran hitung
# 5 fps. Sempat dinaikkan ke 10 dengan dugaan fps rendah memecah ID track -
# dugaan itu SALAH: pada 10 fps angka justru memburuk (mobil 99, bus 34 dalam
# 10 detik), karena jumlah ID unik tumbuh bersama jumlah frame. Masalahnya ada
# pada metrik, bukan fps; panel kini memakai hitungan per frame, yang tidak
# terpengaruh ID pecah. Pada metrik itu 5 fps sudah cukup dan separuh biayanya.
HITUNG_FPS = 5


def hitung(cam: Camera, seconds: int = HITUNG_SECONDS,
           fps: int = HITUNG_FPS) -> dict:
    """Ukur komposisi lalu lintas per kelas pada satu kamera.

    Memakai lock yang sama dengan analisis lain: satu pipeline pada satu waktu.
    Hasilnya TIDAK dicatat ke basis data - ia pengukuran komposisi, bukan
    temuan pelanggaran, dan mencampurnya ke rekap harian akan membuat angka
    kendaraan terhitung dua kali oleh dua mesin berbeda.
    """
    mulai = time.time()
    rekaman: dict[str, Any] = {"key": cam.key, "name": cam.name, "at": _now()}
    try:
        klip = settings.uploads_dir / f"_hitung_{cam.key}.mp4"
        capture_stream(cam.stream_url, klip, seconds, fps, timeout=120)

        poligon = (" ".join(str(v) for v in cam.region_polygon)
                   if cam.region_polygon else "")
        with _analysis_lock:
            hasil = runner.run_hitung(
                source=str(klip),
                run_id=f"hitung-{cam.key}-{datetime.now():%H%M%S}",
                polygon=poligon)
    except Exception as exc:
        rekaman.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                        "total_sec": round(time.time() - mulai, 1)})
        LATEST_DIR.mkdir(parents=True, exist_ok=True)
        with open(cam.hitung_path, "w", encoding="utf-8") as f:
            json.dump(rekaman, f, indent=2, ensure_ascii=False)
        return rekaman

    res = hasil.get("results") or {}
    rekaman = {
        "key": cam.key, "name": cam.name, "area": cam.area,
        "region": cam.region, "zone_label": cam.zone_label,
        "at": _now(), "seconds": seconds, "fps": fps,
        "zona_dipakai": bool(poligon),
        "ok": hasil["returncode"] == 0 and bool(res),
        "elapsed_sec": hasil["elapsed_sec"],
        "total_sec": round(time.time() - mulai, 1),
        "results": res,
    }
    extract_last_frame(klip, cam.hitung_frame_path)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(cam.hitung_path, "w", encoding="utf-8") as f:
        json.dump(rekaman, f, indent=2, ensure_ascii=False)
    return rekaman


def read_hitung(cam: Camera) -> dict | None:
    if not cam.hitung_path.is_file():
        return None
    try:
        with open(cam.hitung_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


class Watcher:
    """Memantau SATU kamera dengan siklus pendek berulang.

    Bukan live: tiap putaran tetap merekam klip lalu menganalisisnya, jadi
    kotaknya tertinggal satu putaran dari kenyataan. Yang membuatnya terasa
    hidup adalah putarannya pendek, bukan karena ada aliran menerus.

    Profilnya sengaja ringan - tracking saja, tanpa atribut, pelat, dan
    segmentasi lajur. Toolbox penuh 319 ms/frame; tracking saja 117 ms/frame.
    """

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.camera: str | None = None
        self.since: str | None = None
        self.rounds = 0
        self.last_at: str | None = None
        self.last_sec: float | None = None
        self.last_error: str | None = None
        self.last_poll = 0.0

    # ---- berkas hasil, terpisah dari siklus batch supaya tidak saling timpa
    @staticmethod
    def frame_path(cam: Camera) -> Path:
        return LATEST_DIR / f"{cam.key}_pantau.jpg"

    @staticmethod
    def record_path(cam: Camera) -> Path:
        return LATEST_DIR / f"{cam.key}_pantau.json"

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self, key: str) -> dict:
        cams = {c.key: c for c in load_cameras()}
        if key not in cams:
            raise KeyError(key)
        if self.running and self.camera == key:
            self.last_poll = time.time()
            return self.status()
        # Menunggu di sini disengaja: siklus lama harus benar-benar keluar
        # sebelum yang baru dimulai, kalau tidak keduanya menulis berkas hasil
        # kamera yang berbeda secara bergantian.
        self.stop(wait=True)
        self._stop.clear()
        self.camera = key
        self.since = _now()
        self.rounds = 0
        self.last_error = None
        self.last_poll = time.time()
        self._thread = threading.Thread(target=self._loop, args=(cams[key],),
                                        daemon=True)
        self._thread.start()
        return self.status()

    def stop(self, wait: bool = False) -> None:
        """Memberi aba-aba berhenti. Bawaannya TIDAK menunggu: putaran yang
        sedang berjalan masih perlu belasan detik, dan menahan permintaan HTTP
        selama itu membuat antarmuka terasa menggantung. Siklusnya keluar pada
        pemeriksaan berikutnya."""
        self._stop.set()
        t = self._thread
        if wait and t and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=40)

    def touch(self) -> None:
        """Dipanggil tiap kali status ditanyakan - penanda masih ada penonton."""
        self.last_poll = time.time()

    def _loop(self, cam: Camera) -> None:
        while not self._stop.is_set():
            # Berhenti sendiri kalau tabnya ditutup. Tanpa ini, satu tab yang
            # terlupa akan membebani server bersama tanpa batas waktu.
            if time.time() - self.last_poll > WATCH_IDLE_STOP:
                self.last_error = "berhenti sendiri: tidak ada yang menonton"
                break
            try:
                self._once(cam)
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._stop.wait(5)
        # Hanya bersihkan kalau rujukannya memang masih menunjuk thread ini.
        # Tanpa penjagaan ini, siklus lama yang telat keluar akan menghapus
        # rujukan ke siklus baru.
        if self._thread is threading.current_thread():
            self._thread = None
            self.camera = None

    def _once(self, cam: Camera) -> None:
        mulai = time.time()
        klip = settings.uploads_dir / f"_pantau_{cam.key}.mp4"
        capture_stream(cam.stream_url, klip, WATCH_SECONDS, WATCH_FPS,
                       timeout=90)
        if self._stop.is_set():
            return
        # TANPA _analysis_lock. Putaran ini menulis klip, run_dir, dan berkas
        # hasilnya sendiri (_pantau_*), tidak menyentuh berkas siklus batch.
        # Memegang lock yang sama membuat tampilan Live membeku sampai ±100
        # detik tiap kali siklus otomatis sedang menganalisis sebuah kamera.
        poligon = (" ".join(str(v) for v in cam.region_polygon)
                   if cam.region_polygon else "")
        hasil = runner.run_hitung(
            source=str(klip),
            run_id=f"pantau-{cam.key}-{datetime.now():%H%M%S}",
            polygon=poligon)

        res = hasil.get("results") or {}
        rekaman = {
            "key": cam.key, "name": cam.name, "area": cam.area,
            "at": _now(), "seconds": WATCH_SECONDS, "fps": WATCH_FPS,
            "scenario": WATCH_SCENARIO,
            "ok": hasil["returncode"] == 0,
            "elapsed_sec": hasil["elapsed_sec"],
            "round_sec": round(time.time() - mulai, 1),
            "results": res,
        }
        extract_last_frame(klip, self.frame_path(cam))
        with open(self.record_path(cam), "w", encoding="utf-8") as f:
            json.dump(rekaman, f, ensure_ascii=False)
        self.rounds += 1
        self.last_at = rekaman["at"]
        self.last_sec = rekaman["round_sec"]

    def status(self) -> dict:
        rekaman = None
        # Berkas hasil sesi pantau sebelumnya masih ada di disk. Menyajikannya
        # sebelum putaran pertama selesai akan menampilkan gambar lama berlabel
        # "putaran 0" - terlihat seperti pantau padahal bukan.
        if self.camera and self.rounds > 0:
            cams = {c.key: c for c in load_cameras()}
            cam = cams.get(self.camera)
            if cam and self.record_path(cam).is_file():
                try:
                    with open(self.record_path(cam), encoding="utf-8") as f:
                        rekaman = json.load(f)
                except (OSError, json.JSONDecodeError):
                    pass
        return {
            "running": self.running,
            "stopping": self._stop.is_set() and self.running,
            "camera": self.camera,
            "since": self.since,
            "rounds": self.rounds,
            "last_at": self.last_at,
            "last_round_sec": self.last_sec,
            "error": self.last_error,
            "seconds": WATCH_SECONDS,
            "fps": WATCH_FPS,
            "scenario": WATCH_SCENARIO,
            "record": rekaman,
        }


watcher = Watcher()


class Worker:
    """Runs cycles on an interval. One camera at a time: on CPU two concurrent
    pipelines are slower than two sequential ones, and on GPU the limit is
    VRAM per worker rather than wall clock."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.interval_sec = 600
        self.profile: str | None = None
        self.running = False
        self.current: str | None = None
        self.cycles = 0
        self.last_cycle_at: str | None = None
        self.last_cleanup: dict | None = None
        self.last_error: str | None = None

    def cycle(self) -> list[dict]:
        """One pass over every camera. Serialised against concurrent triggers."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Siklus batch sedang berjalan")
        try:
            records = []
            for cam in load_cameras():
                if self._stop.is_set():
                    break
                self.current = cam.key
                records.append(analyse(cam, self.profile))

                # Penghitungan tujuh kelas ikut tiap siklus. Sebelumnya ia hanya
                # berjalan bila operator menekan "Ukur sekarang", sehingga kamera
                # yang tak pernah ditekan tampil tanpa angka sama sekali - dan
                # kamera yang pernah ditekan menampilkan angka berjam-jam lalu
                # tanpa menyatakan bahwa itu basi.
                #
                # Mesinnya memang terpisah dari pipeline PP-Vehicle (detektor
                # COCO, bukan ppvehicle9cls), jadi ia harus dipanggil sendiri -
                # skenario apa pun yang dipakai kamera tidak menghasilkannya.
                if self._stop.is_set():
                    break
                try:
                    hitung(cam)
                except Exception as exc:          # satu kamera gagal != siklus gagal
                    self.last_error = f"hitung {cam.key}: {type(exc).__name__}: {exc}"
            self.cycles += 1
            self.last_cycle_at = _now()
            try:
                self.last_cleanup = {**bersihkan(), "at": _now()}
            except Exception as exc:          # retensi tidak boleh menggagalkan siklus
                self.last_cleanup = {"error": f"{type(exc).__name__}: {exc}", "at": _now()}
            return records
        finally:
            self.current = None
            self._lock.release()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.cycle()
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.interval_sec)

    def start(self, interval_sec: int | None = None,
              profile: str | None = None) -> None:
        if self.running:
            return
        if interval_sec:
            self.interval_sec = max(60, int(interval_sec))
        self.profile = profile
        self._stop.clear()
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.running = False

    def status(self) -> dict:
        cams = load_cameras()
        garis_hitung = load_tampilan_hitung()
        over = load_zone_overrides()
        return {
            "running": self.running,
            "interval_sec": self.interval_sec,
            "profile": self.profile or settings.profile,
            "current": self.current,
            "cycles": self.cycles,
            "last_cycle_at": self.last_cycle_at,
            "last_cleanup": self.last_cleanup,
            "last_error": self.last_error,
            "cameras": [
                {
                    "key": c.key, "id": c.id, "name": c.name, "area": c.area,
                    "region": c.region, "owner": c.owner, "scenario": c.scenario,
                    "zone_label": c.zone_label,
                    "illegal_parking_time": c.illegal_parking_time,
                    "zone_updated_at": (over.get(c.key) or {}).get("updated_at"),
                    "stream": c.stream_url,
                    "region_polygon": c.region_polygon,
                    "garis_hitung": (garis_hitung.get(c.key) or [{}])[0].get("garis", []),
                    "tampilan_hitung": garis_hitung.get(c.key, []),
                    "has_frame": c.frame_path.is_file(),
                    "record": read_record(c),
                }
                for c in cams
            ],
        }


worker = Worker()
