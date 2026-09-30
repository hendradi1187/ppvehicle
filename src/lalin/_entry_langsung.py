"""Mesin deteksi LANGSUNG: PP-YOLOE+ L + ByteTrack pada aliran HLS menerus.

Kenapa ada mesin ketiga. Live di dasbor memutar HLS yang sama dengan yang
dibaca server, tetapi peramban memutarnya +-4 segmen di belakang tepi siaran
(segmen Bali Tower +-7,5 dtk). Mesin ini membaca dari tepi siaran, jadi setiap
frame sudah dianalisis +-20 dtk SEBELUM frame itu tampil di layar operator.
Tiap hasil membawa PTS asli frame-nya (ffmpeg -copyts), dan peramban
menampilkannya tepat saat frame dengan PTS itu diputar. Hasilnya: video tetap
mulus 20 fps, kotak tetap menempel pada kendaraannya, dan labelnya dari model
yang sama dengan siklus resmi - bukan dari model mini di peramban.

Kenapa bukan model di peramban. Diuji 25-09-2026 malam pada frame S. Parman
C02 yang padat: PP-YOLOE+ L menemukan 12 mobil dan 16 sepeda motor; YOLOv10n
(ONNX, peramban) 0 sepeda motor - pengendaranya dibaca "orang"; COCO-SSD lite
setali tiga uang. Pesepeda dibaca pejalan kaki karena sepedanya tidak
terlihat oleh model mini. WebGPU dan WASM multi-thread juga tertutup: dasbor
dilayani lewat http, bukan konteks aman.

Laju: +-510 ms/frame di 6 thread (diukur), jadi frame diambil tiap
LALIN_INTERVAL detik (bawaan 0,6). Bila server sesaat lebih lambat, frame
tertua dibuang - tracker tetap berjalan, hanya jaraknya lebih renggang.

Keluaran: satu baris JSON per frame di stdout.
"""

from __future__ import annotations

import base64
import collections
import copy
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

PD_DIR = os.environ["LALIN_PD_DIR"]
MODEL_DIR = os.environ["LALIN_MODEL_DIR"]
SOURCE = os.environ.get("LALIN_SOURCE", "")
SOURCES = json.loads(os.environ.get("LALIN_SOURCES_JSON", "{}")) or {"default": SOURCE}
if not all(isinstance(k, str) and isinstance(v, str) and v for k, v in SOURCES.items()):
    raise ValueError("LALIN_SOURCES_JSON harus berupa object key -> URL")
# Modul aturan lama masih membaca LALIN_SOURCE pada saat import. Dalam mode
# multi-camera nilainya hanya dipakai untuk inisialisasi konstanta, jadi pakai
# URL pertama sebagai compatibility shim; tiap FFmpeg tetap memakai SOURCES.
if not SOURCE:
    SOURCE = next(iter(SOURCES.values()))
    os.environ["LALIN_SOURCE"] = SOURCE
TRACKER_CFG = os.environ["LALIN_TRACKER_CFG"]
THRESHOLD = float(os.environ.get("LALIN_THRESHOLD", "0.30"))
INTERVAL = float(os.environ.get("LALIN_INTERVAL", "0.6"))
DEVICE = os.environ.get("LALIN_DEVICE", "CPU").upper()
W, H = 640, 360

sys.path.insert(0, PD_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(PD_DIR)))

import numpy as np  # noqa: E402

for _name, _builtin in (("int", int), ("float", float), ("bool", bool),
                        ("object", object), ("str", str)):
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

from mot_sde_infer import SDE_Detector  # noqa: E402

# Aturan penyaringan yang sama persis dengan mesin hitung - satu sumber.
# _entry_hitung membaca beberapa LALIN_* saat diimpor; yang tidak dipakai di
# sini diisi nilai kosong supaya impor tidak gagal.
os.environ.setdefault("LALIN_RESULTS", os.devnull)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _entry_hitung import (  # noqa: E402
    AMBANG_MALAM, AMBANG_NMS_ANTARKELAS, AMBANG_PENGENDARA, KELAS_LALIN,
    KENDARAAN_BERPENGENDARA, KENDARAAN_NMS, _iou, _malam, tumpang_relatif, wajar)

# Acuan posisi kamera - HARUS sama dengan ambilAbu() di web/assets/bbox.js.
AW, AH, POTONG_ATAS, POTONG_BAWAH = 80, 36, 30, 318
ABU_TIAP_DTK = 2.0


# Letak teks OSD, diukur 26-09-2026 pada latar median 7 kamera: nama kamera +
# jam di baris y 2-11 selebar frame, "CCTV-DISKOMINFOTIK #JagaJakarta" di
# kiri bawah y 322-331, x 0-245 (sama di ketujuh kamera). Diberi margin.
OSD_KOTAK = ((0, 0, 640, 14), (0, 318, 252, 335))
OSD_TINGGI_MAKS = 22


def di_osd(x1: float, y1: float, x2: float, y2: float) -> bool:
    """Kotak yang SEBENARNYA teks OSD (nama kamera "CCTV-02", jam) - teks itu
    terdeteksi sebagai car/truck/bus 0,31-0,54.

    LC-006: aturan LC-005 membuang seluruh pita y<=34 dan kiri-bawah sampai
    x 340. Pada 821 deteksi uji 26-09 pagi, dari 55 kotak yang dibuang hanya
    11 teks; 40 kendaraan nyata ikut hilang, dan pita bawah memotong 24%
    garis hitung Gerbang Pemuda. Sekarang kotak dibuang hanya bila pendek
    (tinggi <= 22 px), melebar seperti teks (lebar >= 1,2 x tinggi), dan
    >= 60% luasnya di dalam kotak teks. Pada data yang sama: 12 dibuang,
    semuanya teks; pejalan kaki kecil di tepi atas tidak ikut terbuang.
    """
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0 or h > OSD_TINGGI_MAKS or w < 1.2 * h:
        return False
    for a, b, c, d in OSD_KOTAK:
        iw = max(0.0, min(x2, c) - max(x1, a))
        ih = max(0.0, min(y2, d) - max(y1, b))
        if iw * ih >= 0.6 * w * h:
            return True
    return False


def wajar_langsung(x1: float, y1: float, x2: float, y2: float, malam: bool) -> bool:
    """Batas ukuran kotak. Malam: aturan ketat mesin hitung (<=20% frame) -
    di sanalah kotak raksasa halusinasi muncul. Siang: bus Transjakarta yang
    melintas tepat di bawah kamera Gerbang Pemuda menutup +-50% frame dan
    dengan batas 20% justru dibuang (terlihat 26-09-2026 08.09 WIB), jadi
    batasnya dilonggarkan."""
    if malam:
        return wajar(x1, y1, x2, y2, W, H)
    w, h = x2 - x1, y2 - y1
    return w > 0 and h > 0 and w * h <= 0.60 * W * H and w <= 0.95 * W and h <= 0.95 * H
TERTINGGAL_MAKS = 6.0


def keluar(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def abu_kecil(bgr: np.ndarray) -> str:
    g = (0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2])
    g = g[POTONG_ATAS:POTONG_BAWAH].reshape(AH, (POTONG_BAWAH - POTONG_ATAS) // AH,
                                            AW, W // AW).mean(axis=(1, 3))
    return base64.b64encode(np.clip(np.rint(g), 0, 255).astype(np.uint8).tobytes()).decode()


def main() -> int:
    # Siang/malam dievaluasi ulang tiap menit (lihat loop di bawah): Live bisa
    # ditonton melewati magrib, dan ambang malam harus ikut berlaku.
    keadaan = {"malam": _malam(), "cek": time.time()}
    keadaan["ambang"] = max(THRESHOLD, AMBANG_MALAM) if keadaan["malam"] else THRESHOLD
    malam, ambang = keadaan["malam"], keadaan["ambang"]
    detector = SDE_Detector(
        model_dir=MODEL_DIR, tracker_config=TRACKER_CFG, device=DEVICE,
        threshold=min(THRESHOLD, ambang),
        cpu_threads=int(os.environ.get("LALIN_CPU_THREADS", "6")),
        enable_mkldnn=DEVICE == "CPU")
    labels = list(detector.pred_config.labels)
    n_kelas = len(labels)
    # LC-014: bus dan truck dilacak sebagai SATU kelas. Pelacak MCMOT memberi
    # ID per kelas, sehingga kendaraan besar yang labelnya berganti bus <-> truck
    # antar-frame (sering, terutama malam) terpecah menjadi banyak track pendek
    # yang tidak pernah terhitung melintas. Label & skor asli tiap deteksi
    # disimpan, lalu dipulihkan per kotak keluaran lewat IoU (lihat bawah).
    CI_BUS = labels.index("bus") if "bus" in labels else -1
    CI_TRUK = labels.index("truck") if "truck" in labels else -1
    deteksi_besar: list = []       # [(x1, y1, x2, y2, label_asli)] frame ini

    _post_asli = detector.postprocess

    def _post(inputs, result):
        out = _post_asli(inputs, result)
        b = out.get("boxes")
        if b is not None and len(b):
            simpan = [i for i, r in enumerate(b)
                      if float(r[1]) >= keadaan["ambang"]
                      and 0 <= int(r[0]) < n_kelas and labels[int(r[0])] in KELAS_LALIN
                      and wajar_langsung(float(r[2]), float(r[3]), float(r[4]), float(r[5]), keadaan["malam"])
                      and not di_osd(float(r[2]), float(r[3]), float(r[4]), float(r[5]))]
            simpan.sort(key=lambda i: -float(b[i][1]))
            akhir: list[int] = []
            for i in simpan:
                ci = int(b[i][0])
                if labels[ci] in KENDARAAN_NMS and any(
                        labels[int(b[j][0])] in KENDARAAN_NMS and int(b[j][0]) != ci
                        and _iou(b[i][2:6], b[j][2:6]) > AMBANG_NMS_ANTARKELAS
                        for j in akhir):
                    continue
                akhir.append(i)
            simpan = sorted(akhir)
            b = b[simpan] if simpan else np.zeros([0, 6], dtype=b.dtype)
            deteksi_besar.clear()
            if CI_BUS >= 0 and CI_TRUK >= 0 and len(b):
                b = b.copy()
                for r in b:
                    ci = int(r[0])
                    if ci in (CI_BUS, CI_TRUK):
                        deteksi_besar.append((float(r[2]), float(r[3]), float(r[4]), float(r[5]), labels[ci]))
                        r[0] = CI_BUS
            out["boxes"] = b
            out["boxes_num"] = [len(b)]
        return out

    detector.postprocess = _post

    # Satu predictor dibagi semua kamera; tracker wajib terpisah agar ID dari
    # Benhil tidak pernah bercampur dengan Gerbang Pemuda.
    trackers = {key: copy.deepcopy(detector.tracker) for key in SOURCES}

    # ffmpeg: -copyts mempertahankan PTS asli MPEG-TS, yang juga dipakai hls.js
    # di peramban. select mengambil satu frame tiap INTERVAL detik TANPA
    # mengubah PTS-nya (filter fps akan membulatkannya). showinfo menulis
    # pts_time tiap frame yang keluar ke stderr.
    pilih = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{INTERVAL})'"
    def argv_untuk(source: str) -> list[str]:
        return ["ffmpeg", "-hide_banner", "-loglevel", "info", "-nostdin",
            # TANPA -re: dengan -re ffmpeg tertahan 2 segmen (+-15 dtk) di
            # belakang tepi siaran dan hasil hanya +-5 dtk di depan video
            # peramban - terlalu tipis. Tanpa -re segmen datang sekaligus;
            # antrean di bawah menampungnya dan pengolah melompat ke dekat
            # tepi bila tertinggal.
            # -2, bukan -1: segmen paling akhir Bali Tower kadang belum siap
            # diunduh dan ffmpeg langsung keluar ("Error when loading first
            # segment").
            # LC-017: dekoder dibatasi 2 thread. Bawaan ffmpeg = semua core, yang
            # berebut dengan detektor; pada kamera HD 3200x1800 tunda median
            # turun dari 7,7 ke 1,0 dtk (uji 27-09-2026, Bendungan Hilir 2).
            "-threads", "2",
            "-live_start_index", "-2", "-copyts",
            "-rw_timeout", "15000000", "-i", source, "-an", "-sn",
            "-vf", f"{pilih},scale={W}:{H},showinfo",
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    pola = re.compile(rb"pts_time:\s*(-?[0-9.]+)")
    ukuran = W * H * 3
    # HLS tidak selalu keluar rata: stream resolusi rendah dapat mengirim satu
    # segmen sekaligus, sedangkan stream HD terdepak lebih pelan oleh decode.
    # Buffer 2 frame membuat kamera burst hanya kebagian 1/4 scheduler. Simpan
    # jendela bounded <= TERTINGGAL_MAKS; deque otomatis membuang yang tertua.
    # Ini tetap latest-window (bukan backlog tak terbatas) dan menyetarakan
    # round-robin dua kamera tanpa menambah RAM berarti (< 30 MB total).
    panjang_antre = max(2, int(TERTINGGAL_MAKS / INTERVAL) + 1)
    antre = {key: collections.deque(maxlen=panjang_antre) for key in SOURCES}
    meter = {key: {"diterima": 0, "dibuang_penuh": 0, "dibuang_pts": 0,
                   "dibuang_lag": 0, "putus": 0} for key in SOURCES}
    ff_kini: dict[str, subprocess.Popen] = {}

    def jalankan_ffmpeg(key: str, source: str):
        """Satu proses ffmpeg; kembali bila alirannya putus."""
        ff = subprocess.Popen(argv_untuk(source), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, bufsize=0)
        ff_kini[key] = ff
        pts_antre: collections.deque = collections.deque()

        def baca_stderr():
            for baris in ff.stderr:
                if b"Parsed_showinfo" in baris and b"pts_time" in baris:
                    m = pola.search(baris)
                    if m:
                        pts_antre.append(float(m.group(1)))

        threading.Thread(target=baca_stderr, daemon=True).start()
        while True:
            buf = b""
            while len(buf) < ukuran:
                potong = ff.stdout.read(ukuran - len(buf))
                if not potong:
                    ff.wait()
                    return ff.returncode
                buf += potong
            for _ in range(50):
                if pts_antre:
                    break
                time.sleep(0.01)
            pts = pts_antre.popleft() if pts_antre else None
            meter[key]["diterima"] += 1
            if len(antre[key]) == antre[key].maxlen:
                meter[key]["dibuang_penuh"] += 1
            antre[key].append((pts, time.time(), buf))

    def pemasok(key: str, source: str):
        # Jaringan ke Bali Tower sesekali timeout. ffmpeg dijalankan ulang;
        # tracker di proses ini tetap hidup, jadi ID kendaraan tidak hilang.
        putus = 0
        while True:
            kode = jalankan_ffmpeg(key, source)
            putus += 1
            meter[key]["putus"] = putus
            keluar({"jenis": "putus", "key": key, "kode": kode, "ke": putus,
                    "capture": dict(meter[key])})
            time.sleep(min(2 * putus, 10))

    for key, source in SOURCES.items():
        threading.Thread(target=pemasok, args=(key, source), daemon=True).start()

    keluar({"jenis": "mulai", "labels": labels, "malam": malam, "ambang": ambang,
            "interval": INTERVAL, "model": os.path.basename(MODEL_DIR.rstrip("/")),
            "keys": list(SOURCES),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    frame_id = {key: 0 for key in SOURCES}
    pts_terakhir = {key: None for key in SOURCES}
    abu_terakhir = {key: 0.0 for key in SOURCES}
    id_pengendara = {key: set() for key in SOURCES}
    keys = list(SOURCES)
    cursor = 0
    while True:
        key = next((keys[(cursor + i) % len(keys)] for i in range(len(keys))
                    if antre[keys[(cursor + i) % len(keys)]]), None)
        if key is None:
            time.sleep(0.02)
            continue
        cursor = (keys.index(key) + 1) % len(keys)
        q = antre[key]
        mtr = meter[key]
        pts, diterima, buf = q.popleft()
        # LC-006 (D2): ffmpeg yang tersambung ulang mulai lagi 2 segmen ke
        # belakang (-live_start_index -2), jadi +-15 dtk yang sudah dianalisis
        # datang lagi. Tracker tetap hidup: kendaraan yang sama mendapat ID baru
        # dan bisa terhitung dua kali. Frame yang PTS-nya tidak maju dibuang.
        # Mundur > 600 dtk dianggap putaran PTS / diskontinuitas: diterima.
        if pts is not None and pts_terakhir[key] is not None \
                and pts <= pts_terakhir[key] and pts_terakhir[key] - pts < 600:
            mtr["dibuang_pts"] += 1
            continue
        if pts is not None:
            pts_terakhir[key] = pts
        if time.time() - keadaan["cek"] > 60:
            keadaan["cek"] = time.time()
            m = _malam()
            if m != keadaan["malam"]:
                keadaan["malam"] = m
                keadaan["ambang"] = max(THRESHOLD, AMBANG_MALAM) if m else THRESHOLD
                keluar({"jenis": "mulai", "labels": labels, "malam": m, "ambang": keadaan["ambang"],
                        "interval": INTERVAL, "model": os.path.basename(MODEL_DIR.rstrip("/")),
                        "keys": list(SOURCES),
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        # Tertinggal lebih dari TERTINGGAL_MAKS dtk dari frame terbaru: lompati.
        # Frame yang dilewati memang tidak dianalisis, tetapi hasil yang datang
        # setelah frame itu tampil di layar tidak ada gunanya sama sekali.
        if pts is not None and q and q[-1][0] is not None \
                and q[-1][0] - pts > TERTINGGAL_MAKS:
            mtr["dibuang_lag"] += 1
            continue
        bgr = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
        t0 = time.time()
        detector.tracker = trackers[key]
        hasil = detector.predict_image([bgr[..., ::-1].copy()], visual=False,
                                       seq_name=f"langsung-{key}", frame_count=frame_id[key])
        tlwhs, skor, ids = hasil[0]
        ms = round((time.time() - t0) * 1000)
        kunci = list(tlwhs.keys()) if hasattr(tlwhs, "keys") else range(len(tlwhs))
        tunggangan = [tuple(float(v) for v in tl[:4])
                      for cid in kunci if 0 <= int(cid) < n_kelas
                      and labels[int(cid)] in KENDARAAN_BERPENGENDARA
                      for tl in tlwhs[int(cid)]]
        kotak = []
        for cid in kunci:
            cid = int(cid)
            if not 0 <= cid < n_kelas or labels[cid] not in KELAS_LALIN:
                continue
            for tlwh, sc, tid in zip(tlwhs[cid], skor[cid], ids[cid]):
                tid = int(tid)
                if tid < 0:
                    continue
                x, y, w, h = (float(v) for v in tlwh[:4])
                # ID tracker unik PER KELAS pada MCMOT: gabungkan dengan kelas.
                uid = cid * 100000 + tid
                label = labels[cid]
                if cid == CI_BUS and deteksi_besar:
                    # LC-014: label asli (bus/truck) dari deteksi frame ini yang
                    # paling bertumpuk; tanpa pasangan, tetap "bus".
                    iou, asli = max((_iou((x, y, x + w, y + h), d[:4]), d[4]) for d in deteksi_besar)
                    if iou >= 0.3:
                        label = asli
                if labels[cid] == "person" and any(
                        tumpang_relatif((x, y, w, h), t) >= AMBANG_PENGENDARA for t in tunggangan):
                    id_pengendara[key].add(uid)
                # Detector/tracker boleh mengeluarkan kotak yang sedikit
                # melewati tepi frame. Bbox publik harus selalu berada di
                # dalam kanvas 640x360 agar overlay dan metrik luas tidak
                # memakai koordinat negatif atau melewati ukuran frame.
                x1 = max(0.0, min(W, x))
                y1 = max(0.0, min(H, y))
                x2 = max(0.0, min(W, x + w))
                y2 = max(0.0, min(H, y + h))
                if x2 <= x1 or y2 <= y1:
                    continue
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                kotak.append([uid, label, round(x, 1), round(y, 1), round(w, 1),
                              round(h, 1), round(float(sc), 3), uid in id_pengendara[key]])
        baris = {"jenis": "frame", "key": key, "pts": pts, "n": frame_id[key], "ms": ms,
                 "tunda": round(time.time() - diterima, 2), "kotak": kotak,
                 "capture": {**mtr, "antre": len(q),
                             "dibuang_total": mtr["dibuang_penuh"]
                             + mtr["dibuang_pts"] + mtr["dibuang_lag"]}}
        if time.time() - abu_terakhir[key] >= ABU_TIAP_DTK:
            baris["abu"] = abu_kecil(bgr)
            abu_terakhir[key] = time.time()
        keluar(baris)
        frame_id[key] += 1
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:  # noqa: BLE001
        keluar({"jenis": "galat", "pesan": f"{type(exc).__name__}: {exc}"})
        code = 1
    sys.stdout.flush()
    os._exit(code)
