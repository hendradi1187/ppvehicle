"""Mesin kedua: penghitungan lalu lintas per kelas kendaraan.

Kenapa terpisah dari `_entry.py`. Pipeline PP-Vehicle dikunci satu kelas -
kode sumbernya sendiri menulis `num_classes = 1` dan
`flow_statistic only support single class MOT`. Semua yang ia keluarkan
bernama "vehicle". Membedakan sepeda motor dari truk, dari pejalan kaki, dari
sepeda, menuntut detektor lain.

Detektornya PP-YOLOE terlatih COCO, BUKAN `ppvehicle9cls` seperti percobaan
pertama. Model sembilan kelas itu diuji lebih dulu dan gagal pada lalu lintas
Jakarta: pada satu frame Pasar Tanah Abang yang padat sepeda motor ia hanya
menemukan 3 objek dan melabeli sepeda motor sebagai `car`. Model COCO pada
frame yang sama persis menemukan 76 objek - 24 sepeda motor, 7 pejalan kaki,
1 sepeda - dan benar secara visual. Dugaan awal bahwa penyebabnya resolusi
terbantah oleh uji berubin: memperbesar piksel per objek dua kali lipat
menaikkan deteksi mobil 79% tetapi sepeda motor tetap nol.

Modul ini memakai SDE_Detector dari pptracking, yang membaca jumlah kelas dari
label model dan otomatis masuk mode MCMOT. Penggambar bawaannya tidak dipakai
(`plot_tracking only supports single classes`) - dashboard menggambar sendiri
dari koordinat, seperti yang sudah berlaku untuk mesin pertama.

Dua angka dihasilkan, dan keduanya berbeda arti:

  terlacak  jumlah objek berbeda yang terlihat sepanjang klip. Menggambarkan
            kepadatan, bukan arus: kendaraan yang diam ikut terhitung.
  masuk     jumlah objek yang MELINTAS masuk ke zona. Inilah penghitungan lalu
            lintas yang sebenarnya - satu objek dihitung sekali, saat titik
            rodanya berpindah dari luar ke dalam zona.

Dipanggil runner dengan LALIN_* di environment.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

PD_DIR = os.environ["LALIN_PD_DIR"]                 # <root>/deploy/pptracking/python
MODEL_DIR = os.environ["LALIN_MODEL_DIR"]
SOURCE = os.environ["LALIN_SOURCE"]
RESULTS_PATH = os.environ["LALIN_RESULTS"]
TRACKER_CFG = os.environ["LALIN_TRACKER_CFG"]   # wajib JDETracker; lihat berkasnya
POLYGON = os.environ.get("LALIN_POLYGON", "").strip()
THRESHOLD = float(os.environ.get("LALIN_THRESHOLD", "0.35"))
DEVICE = os.environ.get("LALIN_DEVICE", "CPU").upper()
TRAIL_MAX = 60

sys.path.insert(0, PD_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(PD_DIR)))   # <root>/deploy

import numpy as np  # noqa: E402

# Alias numpy lama yang dibuang di numpy 1.24, sama seperti di _entry.py:
# PaddleDetection 2.9 masih memakainya di beberapa jalur.
for _name, _builtin in (("int", int), ("float", float), ("bool", bool),
                        ("object", object), ("str", str)):
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

import cv2  # noqa: E402
from mot_sde_infer import SDE_Detector  # noqa: E402

# Enam kelas COCO yang relevan bagi lalu lintas, dipetakan ke nama Indonesia.
# Detektor mengeluarkan 80 kelas; sisanya (payung pasar, tas, kursi) memang
# terdeteksi dengan benar tetapi bukan urusan Dishub, jadi disaring di sini.
#
# Nama TIDAK diberi kualifikasi "besar"/"berat": model mengenali bus dan truk,
# bukan jumlah sumbunya. Menambahkan kualifikasi itu pada label akan
# menjanjikan klasifikasi MKJI yang tidak dihasilkan model mana pun.
KELAS_LALIN = {
    "motorcycle": "Sepeda motor",
    "car": "Mobil penumpang",
    "bus": "Bus",
    "truck": "Truk",
    "person": "Pejalan kaki",
    "bicycle": "Sepeda",
}

# Di atas nilai ini, kotak orang dianggap PENGENDARA kendaraan yang
# ditumpanginya, bukan pejalan kaki. Dipilih longgar dengan sengaja: menghitung
# seorang pengendara sebagai pejalan kaki lebih menyesatkan daripada kehilangan
# satu pejalan kaki yang kebetulan berdiri rapat dengan motor - angka pejalan
# kaki dipakai untuk menilai kebutuhan fasilitas penyeberangan.
AMBANG_PENGENDARA = 0.40

# ------------------------------------------------------------------ malam
# Diukur 24 Sep 2026 pukul 18.42 WIB pada enam frame dari tiga kamera.
# Masalah utama malam BUKAN gelap - kecerahan rata-rata frame masih 95-107 -
# melainkan motion blur: kamera memakai pencahayaan panjang dan kendaraan yang
# bergerak tercoreng. Pada ambang siang 0.20 detektor berhalusinasi:
#   Gerbang Pemuda  42 "mobil" (terlihat +-8; motor tercoreng dilabeli mobil)
#   CCTV-01          1 kotak "bus" menutupi hampir seluruh frame
# CLAHE, gamma, penajaman, dan deteksi berubin diuji; tidak satu pun
# memperbaikinya, CLAHE malah memperburuk beberapa frame.
#
# Yang dipasang di sini hanya MEMBUANG yang palsu, bukan menemukan yang hilang:
# ambang 0.35 menurunkan 42 -> 7 mobil di Gerbang Pemuda dengan kotak yang
# memang di atas mobil. Harga yang dibayar: sepeda motor yang tercoreng blur
# tidak terdeteksi sama sekali, jadi hasil malam ditandai `malam: true` dan
# tidak boleh dikutip sebagai hitungan.
AMBANG_MALAM = 0.35


def _malam() -> bool:
    paksa = os.environ.get("LALIN_MALAM", "").strip()
    if paksa in ("0", "1"):
        return paksa == "1"
    jam_wib = (datetime.now(timezone.utc).hour + 7) % 24
    return jam_wib >= 18 or jam_wib < 6


MALAM = _malam()


# Penyaringan kotak ganda ANTARKELAS untuk kendaraan. NMS bawaan model hanya
# bekerja per kelas, sehingga satu kendaraan bisa keluar dua kali pada kotak
# yang nyaris sama dengan label berbeda. Diukur 25 Sep 2026 pada 45 frame
# siang CCTV-01: 81% "truk" dan 93% "sepeda" adalah kembaran semacam itu
# (bus juga dilabeli truk, motor juga dilabeli sepeda). `person` sengaja tidak
# ikut: pengendara berimpit dengan motornya dan harus tetap terdeteksi supaya
# penyaring pengendara bekerja. IoU 0,7 dan 0,5 memberi hasil hampir sama -
# kembarannya memang berimpit - jadi dipakai yang lebih hati-hati.
KENDARAAN_NMS = ("car", "bus", "truck", "motorcycle", "bicycle")
AMBANG_NMS_ANTARKELAS = 0.7


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def wajar(x1: float, y1: float, x2: float, y2: float, W: float, H: float) -> bool:
    """Kotak yang secara fisik mustahil untuk satu kendaraan atau orang.

    Pada frame malam CCTV-01 detektor mengeluarkan kotak "bus" yang menutupi
    hampir seluruh gambar. Tidak ada satu kendaraan pun di kamera-kamera ini
    yang mengisi lebih dari seperlima frame, jadi batas ini tidak membuang
    objek nyata - diuji pada frame siang dan malam.
    """
    w, h = x2 - x1, y2 - y1
    return w > 0 and h > 0 and w * h <= 0.20 * W * H and w <= 0.6 * W and h <= 0.75 * H
KENDARAAN_BERPENGENDARA = ("motorcycle", "bicycle")

# Tujuh kelas yang diminta untuk POC, berikut padanannya. Yang tidak punya
# padanan tetap ikut dilaporkan dengan tersedia=False, supaya layar menyatakan
# ketiadaannya secara terbuka - bukan menghilangkannya diam-diam sehingga
# pembaca mengira kelas itu memang nihil di jalan.
DIMINTA = [
    ("Sepeda motor", "motorcycle"),
    ("Mobil penumpang", "car"),
    ("Kendaraan sedang", None),      # tidak ada padanan: COCO tak punya kelas van
    ("Bus", "bus"),
    ("Truk", "truck"),
    ("Pejalan kaki", "person"),
    ("Sepeda", "bicycle"),
]


def tumpang_relatif(a: tuple[float, float, float, float],
                    b: tuple[float, float, float, float]) -> float:
    """Luas irisan dibagi luas kotak TERKECIL.

    Bukan IoU. Kotak pengendara dan kotak motornya berukuran jauh berbeda, dan
    IoU menghukum perbedaan ukuran itu - dua kotak yang benar-benar bertindih
    bisa ber-IoU 0,2 saja. Yang ingin diukur di sini bukan kemiripan bentuk,
    melainkan "apakah yang satu berada di atas yang lain".
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    irisan = (x2 - x1) * (y2 - y1)
    terkecil = min(aw * ah, bw * bh) or 1e-9
    return irisan / terkecil


def titik_dalam(poly: list[tuple[float, float]], x: float, y: float) -> bool:
    """Ray casting. Poligon zona digambar operator di editor visual."""
    di_dalam = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            potong = x1 + (y - y1) * (x2 - x1) / ((y2 - y1) or 1e-9)
            if x < potong:
                di_dalam = not di_dalam
    return di_dalam


def baca_poligon() -> list[tuple[float, float]]:
    if not POLYGON:
        return []
    angka = [float(v) for v in POLYGON.replace(",", " ").split()]
    if len(angka) < 6 or len(angka) % 2:
        return []
    return [(angka[i], angka[i + 1]) for i in range(0, len(angka), 2)]


def main() -> int:
    poly = baca_poligon()

    # cpu_threads dan MKLDNN WAJIB disetel. Bawaan SDE_Detector adalah 1 thread
    # tanpa MKLDNN, dan selama itu mesin ini diam-diam berjalan di satu core
    # dari enam belas: dengan PP-YOLOE+ L satu putaran Pantau memakan 58,9 dtk.
    # Tiga: kontainer dibatasi 8 CPU dan pipeline siklus otomatis memakai lima
    # (configs/profiles/server.yml). Lebih dari itu, kedua mesin saling
    # menunggu dan sama-sama melambat.
    detector = SDE_Detector(
        model_dir=MODEL_DIR,
        tracker_config=TRACKER_CFG,
        device=DEVICE,
        threshold=THRESHOLD,
        cpu_threads=int(os.environ.get("LALIN_CPU_THREADS", "3")),
        enable_mkldnn=DEVICE == "CPU" and os.environ.get("LALIN_MKLDNN", "1") == "1",
    )
    labels = list(detector.pred_config.labels)
    n_kelas = len(labels)

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"tidak bisa membuka {SOURCE}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360
    ambang_efektif = max(THRESHOLD, AMBANG_MALAM) if MALAM else THRESHOLD

    # Deteksi mentah frame terakhir, sesudah disaring. Hitungan per frame
    # diambil dari sini, BUKAN dari keluaran tracker: tracker hanya mengeluarkan
    # track yang sudah terkonfirmasi di frame berikutnya, dan pada blur malam
    # konfirmasi itu sering gagal - HUD Live sempat menulis 4 mobil di jalan
    # yang jelas berisi lebih banyak. Kepadatan tidak butuh identitas.
    det_terakhir: list[list[float]] = []
    _post_asli = detector.postprocess

    def _post(inputs, result):
        out = _post_asli(inputs, result)
        b = out.get("boxes")
        if b is not None and len(b):
            simpan = [i for i, r in enumerate(b)
                      if float(r[1]) >= ambang_efektif
                      and wajar(float(r[2]), float(r[3]), float(r[4]), float(r[5]), W, H)]
            # Antarkelas: urut skor turun, buang kendaraan yang berimpit dengan
            # kendaraan BERLABEL LAIN yang skornya lebih tinggi.
            simpan.sort(key=lambda i: -float(b[i][1]))
            akhir: list[int] = []
            for i in simpan:
                ci = int(b[i][0])
                nm = labels[ci] if 0 <= ci < n_kelas else ""
                if nm in KENDARAAN_NMS and any(
                        (labels[int(b[j][0])] in KENDARAAN_NMS)
                        and int(b[j][0]) != ci
                        and _iou(b[i][2:6], b[j][2:6]) > AMBANG_NMS_ANTARKELAS
                        for j in akhir):
                    continue
                akhir.append(i)
            simpan = sorted(akhir)
            b = b[simpan] if simpan else np.zeros([0, 6], dtype=b.dtype)
            out["boxes"] = b
            out["boxes_num"] = [len(b)]
        det_terakhir[:] = [] if b is None else [list(map(float, r[:6])) for r in b]
        return out

    # Dipasang di instance, jadi SDE_Detector.predict_image memakainya untuk
    # tracker juga: kotak halusinasi tidak pernah sempat menjadi track.
    detector.postprocess = _post

    terlacak: dict[int, set[int]] = defaultdict(set)
    # Jumlah objek per kelas PADA TIAP FRAME. Inilah angka yang dipakai panel.
    #
    # Jumlah track ID unik ("terlacak") ternyata tidak bisa dipercaya: diukur
    # pada klip 10 detik JPO Gatot Subroto 7, menurunkan ambang agar sepeda
    # motor terlihat membuatnya melonjak ke 99 mobil dan 34 bus - mustahil.
    # Setiap kedipan deteksi melahirkan ID baru, dan makin banyak frame makin
    # banyak ID. Hitungan per frame tidak punya kelemahan itu: satu kendaraan di
    # satu frame adalah satu kendaraan, berapa kali pun ID-nya berganti.
    per_frame: dict[int, list[int]] = defaultdict(list)
    masuk: dict[int, set[int]] = defaultdict(set)
    di_dalam_sebelumnya: dict[tuple[int, int], bool] = {}
    jejak: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    kotak_terakhir: list[dict] = []
    # Track orang yang pernah terlihat menumpang kendaraan. Ditandai SEKALI
    # seumur track: pengendara yang sesaat terpisah dari motornya karena
    # terhalang tidak lalu berubah menjadi pejalan kaki.
    id_pengendara: set[int] = set()
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hasil = detector.predict_image([rgb], visual=False, seq_name="hitung",
                                       frame_count=frame_id)
        tlwhs, skor, ids = hasil[0]

        kotak_terakhir = []

        # COCO mendeteksi seorang pengendara DUA KALI: sebagai `person` dan
        # sebagai `motorcycle`. Tanpa penyaringan ini, satu orang di atas motor
        # menambah satu ke sepeda motor DAN satu ke pejalan kaki - dan di jalan
        # Jakarta itu menggelembungkan angka pejalan kaki sampai berlipat.
        tunggangan = []
        for cid in (list(tlwhs.keys()) if hasattr(tlwhs, "keys") else range(len(tlwhs))):
            cid = int(cid)
            if 0 <= cid < n_kelas and labels[cid] in KENDARAAN_BERPENGENDARA:
                for tl in tlwhs[cid]:
                    tunggangan.append(tuple(float(v) for v in tl[:4]))

        # JDETracker mengembalikan defaultdict yang HANYA berisi kelas yang
        # muncul pada frame itu. Jadi yang ditelusuri adalah kuncinya, bukan
        # range(n_kelas): mengindeks defaultdict dengan kelas yang tidak ada
        # justru menciptakan entri kosong, dan `len()`-nya menghitung kunci -
        # bukan jumlah kelas. Salah membacanya membuat seluruh objek runtuh ke
        # kelas 0 dan terlaporkan sebagai pejalan kaki.
        kunci_kelas = list(tlwhs.keys()) if hasattr(tlwhs, "keys") else range(len(tlwhs))
        for cls_id in kunci_kelas:
            cls_id = int(cls_id)
            if not 0 <= cls_id < n_kelas:
                continue
            # Saring di sini, bukan saat melapor: payung dan tas ikut terlacak
            # oleh tracker, dan membiarkannya masuk ke jejak hanya membebani
            # muatan JSON tanpa ada yang membacanya.
            if labels[cls_id] not in KELAS_LALIN:
                continue
            b, s, t = tlwhs[cls_id], skor[cls_id], ids[cls_id]
            for tlwh, sc, tid in zip(b, s, t):
                tid = int(tid)
                if tid < 0:
                    continue
                x, y, w, h = (float(v) for v in tlwh[:4])
                if labels[cls_id] == "person" and any(
                        tumpang_relatif((x, y, w, h), t) >= AMBANG_PENGENDARA
                        for t in tunggangan):
                    id_pengendara.add(tid)
                terlacak[cls_id].add(tid)
                kunci = (cls_id, tid)

                # Titik roda, bukan titik tengah: yang menentukan sebuah
                # kendaraan berada di dalam zona adalah tempat ia menapak.
                cx, cy = x + w / 2, y + h
                jr = jejak[kunci]
                jr.append((round(cx, 1), round(cy, 1)))
                if len(jr) > TRAIL_MAX:
                    del jr[:-TRAIL_MAX]

                if poly:
                    kini = titik_dalam(poly, cx, cy)
                    tadi = di_dalam_sebelumnya.get(kunci)
                    # Dihitung hanya pada PERPINDAHAN luar -> dalam, sekali per
                    # objek. Tanpa syarat ini, satu kendaraan yang berhenti di
                    # dalam zona akan terhitung tiap frame.
                    if kini and tadi is False:
                        masuk[cls_id].add(tid)
                    di_dalam_sebelumnya[kunci] = kini

                kotak_terakhir.append({
                    "id": tid, "cls": cls_id, "kelas": labels[cls_id],
                    "nama": KELAS_LALIN.get(labels[cls_id], labels[cls_id]),
                    "x": round(x, 1), "y": round(y, 1),
                    "w": round(w, 1), "h": round(h, 1),
                    "score": round(float(sc), 3),
                    "pengendara": tid in id_pengendara,
                })

        # Isi hitungan per frame untuk SEMUA kelas yang diminta, termasuk yang
        # nol di frame ini - median atas deret yang bolong-bolong akan bias ke
        # atas. Pengendara tidak dihitung sebagai pejalan kaki.
        ada = defaultdict(int)
        motor_mentah = [(r[2], r[3], r[4] - r[2], r[5] - r[3]) for r in det_terakhir
                        if 0 <= int(r[0]) < n_kelas
                        and labels[int(r[0])] in KENDARAAN_BERPENGENDARA]
        for r in det_terakhir:
            ci = int(r[0])
            if not 0 <= ci < n_kelas or labels[ci] not in KELAS_LALIN:
                continue
            if labels[ci] == "person":
                kotak = (r[2], r[3], r[4] - r[2], r[5] - r[3])
                if any(tumpang_relatif(kotak, m) >= AMBANG_PENGENDARA for m in motor_mentah):
                    continue
            ada[ci] += 1
        for _, lab in DIMINTA:
            if lab is not None and lab in labels:
                ci = labels.index(lab)
                per_frame[ci].append(ada.get(ci, 0))
        frame_id += 1

    cap.release()

    # Indeks label -> cls_id, untuk menjawab "berapa banyak kelas X".
    idx = {label: i for i, label in enumerate(labels)}

    # Urutannya mengikuti daftar yang diminta, BUKAN diurutkan dari terbanyak:
    # petugas membaca baris yang sama di posisi yang sama tiap kali, dan kelas
    # yang nihil tetap terlihat sebagai nol - bukan lenyap dari tabel.
    per_kelas = []
    for nama, label in DIMINTA:
        if label is None or label not in idx:
            per_kelas.append({
                "nama": nama, "kelas": label, "cls": None,
                "terlacak": None, "masuk": None, "tersedia": False,
                "catatan": "Tidak ada padanan kelas pada model; pembedaan "
                           "ukuran/sumbu menuntut model terlatih khusus.",
            })
            continue
        cls_id = idx[label]
        mentah = terlacak[cls_id]
        deret = sorted(per_frame.get(cls_id, []))
        median = deret[len(deret) // 2] if deret else 0
        baris = {
            "nama": nama, "kelas": label, "cls": cls_id,
            # Angka utama: berapa objek kelas ini terlihat pada satu frame yang
            # khas (median), dan paling banyak (maks). Keduanya tahan terhadap
            # ID yang pecah.
            "per_frame": median,
            "per_frame_maks": deret[-1] if deret else 0,
            # Dipertahankan hanya untuk diagnosis. JANGAN ditampilkan sebagai
            # jumlah kendaraan - lihat catatan pada deklarasi per_frame.
            "terlacak": len(mentah),
            "masuk": len(masuk[cls_id]),
            "tersedia": True,
        }
        if label == "person":
            pengendara = mentah & id_pengendara
            baris["terlacak"] = len(mentah) - len(pengendara)
            baris["mentah"] = len(mentah)
            baris["pengendara_dikecualikan"] = len(pengendara)
            baris["masuk"] = len(masuk[cls_id] - id_pengendara)
        per_kelas.append(baris)

    payload = {
        "status": "ok",
        "engine": "hitung9cls",
        "model": os.path.basename(MODEL_DIR.rstrip("/")),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frames": frame_id,
        "video_fps": fps,
        "labels": labels,
        "zona_aktif": bool(poly),
        "pengendara_dikecualikan": len(id_pengendara),
        "ambang_pengendara": AMBANG_PENGENDARA,
        "malam": MALAM,
        "ambang_efektif": max(THRESHOLD, AMBANG_MALAM) if MALAM else THRESHOLD,
        "unique_tracks": sum(len(v) for v in terlacak.values()),
        "per_class": per_kelas,
        "boxes": kotak_terakhir,
        "trails": [{"id": k[1], "cls": k[0], "pts": v}
                   for k, v in jejak.items() if len(v) > 3][:80],
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
