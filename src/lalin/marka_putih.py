"""Pemandu untuk menggambar garis hitung: latar bersih + garis marka putih.

Operator menggambar garis hitung di atas video yang bergerak - kendaraan
menutupi marka, dan malam hari lampu kendaraan memantul. Di sini diambil
+-10 frame selama +-20 dtk dari aliran yang sama dengan Live, lalu dihitung
MEDIAN per piksel: kendaraan yang lewat terhapus, yang tersisa jalan dan
markanya. Pada latar itu garis putih dicari dengan Hough (OpenCV), dan
hasilnya dipakai penyunting di dasbor sebagai acuan "tegak lurus garis putih"
(aturan garis melintang PM 67/2018: garis henti tegak lurus sumbu lajur).
"""

from __future__ import annotations

import base64
import math
import subprocess
import threading
import time

import cv2
import numpy as np

W, H = 640, 360
_cache: dict[str, tuple[float, dict]] = {}
_kunci = threading.Lock()
SIMPAN_DTK = 600


def _latar(url: str, detik: int = 20) -> np.ndarray | None:
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-rw_timeout", "15000000",
         "-live_start_index", "-3", "-i", url, "-t", str(detik),
         "-vf", f"fps=0.5,scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        capture_output=True, timeout=detik + 60)
    n = len(r.stdout) // (W * H * 3)
    if not n:
        return None
    fr = np.frombuffer(r.stdout[:n * W * H * 3], np.uint8).reshape(n, H, W, 3)
    return np.median(fr, axis=0).astype(np.uint8)


def _garis_putih(im: np.ndarray) -> list[list[float]]:
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # putih/abu terang, jenuh rendah; ambang V adaptif supaya malam tetap dapat
    v = hsv[..., 2]
    ambang_v = max(120, int(np.percentile(v[30:318], 92)))
    putih = cv2.inRange(hsv, (0, 0, ambang_v), (180, 70, 255))
    putih[:30] = 0
    putih[318:] = 0                       # teks OSD
    tepi = cv2.Canny(putih, 50, 150)
    seg = cv2.HoughLinesP(tepi, 1, np.pi / 360, threshold=30, minLineLength=30, maxLineGap=14)
    daftar = []
    for s in (seg if seg is not None else []):
        x1, y1, x2, y2 = (float(v) for v in s[0])
        L = math.hypot(x2 - x1, y2 - y1)
        sudut = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        daftar.append([x1, y1, x2, y2, L, sudut])
    daftar.sort(key=lambda d: -d[4])
    # Buang duplikat: dua tepi satu garis cat (kiri/kanan) hampir sejajar dan
    # berdekatan - cukup satu.
    hasil: list[list[float]] = []
    for d in daftar:
        cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
        ganda = False
        for h in hasil:
            beda = abs((d[5] - h[5] + 90) % 180 - 90)
            if beda < 4:
                # jarak titik tengah d ke garis h
                ax, ay, bx, by = h[:4]
                L = math.hypot(bx - ax, by - ay) or 1
                jarak = abs((bx - ax) * (ay - cy) - (ax - cx) * (by - ay)) / L
                if jarak < 6:
                    ganda = True
                    break
        if not ganda:
            hasil.append([round(v, 1) for v in d])
        if len(hasil) >= 30:
            break
    return hasil


def pemandu(key: str, url: str, segar: bool = False) -> dict:
    with _kunci:
        c = _cache.get(key)
        if c and not segar and time.time() - c[0] < SIMPAN_DTK:
            return c[1]
    im = _latar(url)
    if im is None:
        raise RuntimeError("aliran kamera tidak bisa dibaca")
    ok, jpg = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
    hasil = {"key": key, "garis_putih": _garis_putih(im),
             "latar": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode(),
             "dibuat": time.time()}
    with _kunci:
        _cache[key] = (time.time(), hasil)
    return hasil
