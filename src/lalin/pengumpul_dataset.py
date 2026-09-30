"""Pengumpul frame mentah untuk dataset klasifikasi kendaraan DKI.

Mengambil frame resolusi asli dari kamera HD (Bali Tower cctv-jsc) dengan
ffmpeg yang HANYA mendekode keyframe (-skip_frame nokey) - beban CPU kecil,
tidak mengganggu deteksi Live di server bersama.

Keluaran: /app/runs/dataset/mentah/<kamera>/<YYYYmmdd-HHMMSS>.jpg (WIB)
Pengaman: berhenti setelah --jam jam, atau bila sisa disk < --sisa-gb.

DATA PRIBADI: frame resolusi tinggi bisa memuat pelat nomor dan wajah.
Berkas mentah hanya disimpan di server (izin 700), tidak disalin ke laptop,
tidak dipublikasikan; hanya potongan kendaraan yang diperkecil yang dipakai
untuk pelabelan (UU No. 27/2022).

Jalankan di kontainer:  docker exec -d docker-ppvehicle-1 nice -n 19 python -m lalin.pengumpul_dataset
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import batch

AKAR = Path("/app/runs/dataset/mentah")
LOG = Path("/app/runs/dataset/pengumpul.log")
# kamera -> interval detik antar-frame (disesuaikan dengan jarak keyframe)
KAMERA = {"benhil2": 5.0, "benhil4": 15.0, "kebonmelati": 15.0,
          "senayan001": 15.0, "senayan018": 15.0, "kuninganbarat": 15.0}


def catat(pesan: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + pesan + "\n")


def jalan_kamera(key: str, url: str, interval: float, batas: float, sisa_gb: float) -> None:
    tujuan = AKAR / key
    tujuan.mkdir(parents=True, exist_ok=True)
    os.chmod(tujuan, 0o700)
    pilih = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval - 0.5})'"
    argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1",
            "-skip_frame", "nokey", "-rw_timeout", "20000000", "-i", url, "-an",
            "-vf", pilih, "-fps_mode", "vfr", "-q:v", "3",
            "-f", "image2", "-strftime", "1", str(tujuan / "%Y%m%d-%H%M%S.jpg")]
    jeda = 5
    while time.time() < batas:
        if shutil.disk_usage("/app/runs").free / 1e9 < sisa_gb:
            catat(f"{key}: berhenti - sisa disk < {sisa_gb} GB")
            return
        mulai = time.time()
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE)
        while p.poll() is None:
            time.sleep(10)
            if time.time() >= batas or shutil.disk_usage("/app/runs").free / 1e9 < sisa_gb:
                p.terminate()
                break
        galat = (p.stderr.read() or b"").decode("utf-8", "replace").strip()[-200:]
        catat(f"{key}: ffmpeg keluar (kode {p.returncode}, {time.time() - mulai:.0f} dtk) {galat}")
        # sambung ulang; jeda bertambah bila langsung gagal
        jeda = 5 if time.time() - mulai > 60 else min(jeda * 2, 300)
        time.sleep(jeda)
    catat(f"{key}: selesai (batas waktu)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jam", type=float, default=36.0)
    ap.add_argument("--sisa-gb", type=float, default=100.0)
    # 27-09-2026: POC memakai SATU kamera HD (Bendungan Hilir 2) - keputusan pemilik proyek.
    ap.add_argument("--kamera", default="benhil2", help="daftar kunci dipisah koma")
    a = ap.parse_args()
    pilihan = [k.strip() for k in a.kamera.split(",") if k.strip()]
    AKAR.mkdir(parents=True, exist_ok=True)
    os.chmod(AKAR.parent, 0o700)
    os.environ["TZ"] = "WIB-7"          # POSIX: UTC+7, tanpa butuh tzdata
    time.tzset()
    cams = {c.key: c for c in batch.load_cameras()}
    batas = time.time() + a.jam * 3600
    utas = []
    for key in pilihan:
        iv = KAMERA.get(key, 15.0)
        if key not in cams:
            catat(f"{key}: tidak ada di cameras.yml, dilewati")
            continue
        t = threading.Thread(target=jalan_kamera, args=(key, cams[key].stream_url, iv, batas, a.sisa_gb), daemon=True)
        t.start()
        utas.append(t)
        time.sleep(2)
    catat(f"mulai: {len(utas)} kamera, {a.jam} jam, berhenti bila sisa disk < {a.sisa_gb} GB")
    for t in utas:
        t.join()
    catat("semua selesai")
    return 0


if __name__ == "__main__":
    sys.exit(main())
