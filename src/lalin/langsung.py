"""Pengelola deteksi langsung untuk Live: satu kamera, selama ada yang menonton.

Menjalankan `_entry_langsung.py` (PP-YOLOE+ L + ByteTrack pada aliran HLS)
dan, dari keluarannya, mengerjakan tiga hal yang butuh ingatan lintas frame:

1. Tujuh kelas POC per track, dengan suara terbanyak sepanjang umur track,
   supaya label satu kendaraan tidak berkedip antar-frame.
2. Pencocokan posisi kamera PTZ dengan tampilan yang pernah digaris
   (configs/garis_hitung.yml). Hitungan hanya berjalan bila posisi cocok.
3. Penghitungan lintas garis: satu track dihitung sekali, saat titik rodanya
   melintasi salah satu ruas garis tampilan yang aktif.

Peramban mengambil hasil per frame berikut PTS-nya dan menampilkannya tepat
saat frame itu diputar (lihat web/assets/bbox.js).
"""

from __future__ import annotations

import base64
import collections
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import batch, klasifikasi, simpan_lintas
from .settings import settings

INTERVAL_DTK = 0.6
CPU_THREADS = 6
DIAM_BERHENTI = 60          # dtk tanpa polling -> berhenti (server bersama)
SIMPAN_FRAME = 600          # +-6 menit hasil pada 1,6 fps
JEJAK_MAKS = 80             # +-48 dtk di interval Live 0,6 dtk

# Sama dengan bbox.js / _entry_langsung.py
AW, AH = 80, 36
NCC_MIN = 0.15
GERAK_MIN_PX = 14.0
# Kelas, ambang yakin, dan status (CLASSIFIED/AMBIGUOUS/UNKNOWN) TIDAK lagi
# ditanam di sini: semuanya dari configs/klasifikasi.yml lewat
# src/lalin/klasifikasi.py (LC-007). Isi kelas mengikuti PKJI 2023; COCO
# `bus` dan `truck` masing-masing mencakup lebih dari satu kelas PKJI sehingga
# berstatus AMBIGUOUS sampai ada classifier jenis rinci.
#
# Latar ambang yakin (UAT 25-26 Sep 2026): gerobak pedagang dibaca "Truk
# Berat 55%"; dari 104 kotak truk/bus yang diperiksa, yang salah hampir
# semuanya berskor 0,30-0,55, yang benar umumnya >= 0,65.
KLASIFIKASI_YML = settings.configs_dir / "klasifikasi.yml"


# Peristiwa lintas ditahan sampai kelas track-nya pasti (track hilang dari
# layar, atau beberapa detik sesudah melintas). Server +-20 dtk di depan
# video peramban, jadi penahanan ini tidak terlihat operator.
TAHAN_DTK = 6.0


# ------------------------------------------------ posisi kamera (numpy)
def _tepi(g: np.ndarray) -> np.ndarray:
    e = np.zeros_like(g)
    e[1:-1, 1:-1] = (np.abs(g[1:-1, 2:] - g[1:-1, :-2]) + np.abs(g[2:, 1:-1] - g[:-2, 1:-1]))
    return e


def _abu(b64: str | None) -> np.ndarray | None:
    if not b64:
        return None
    try:
        t = base64.b64decode(b64)
    except ValueError:
        return None
    if len(t) != AW * AH:
        return None
    return np.frombuffer(t, dtype=np.uint8).astype(np.float32).reshape(AH, AW)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / math.sqrt(float((a * a).sum() * (b * b).sum()) + 1e-9))


def cocokkan(acuan_tepi: np.ndarray, kini_tepi: np.ndarray) -> tuple[float, int, int, float]:
    """(ncc di 0,0, dx, dy, ncc terbaik) - pencarian +-10 x +-6 sel."""
    terbaik, bx, by = -2.0, 0, 0
    nol = _ncc(acuan_tepi, kini_tepi)
    for dy in range(-6, 7):
        for dx in range(-10, 11):
            if dx == 0 and dy == 0:
                v = nol
            else:
                ya = slice(max(0, -dy), AH - max(0, dy))
                yb = slice(max(0, dy), AH + min(0, dy))
                xa = slice(max(0, -dx), AW - max(0, dx))
                xb = slice(max(0, dx), AW + min(0, dx))
                v = _ncc(acuan_tepi[ya, xa], kini_tepi[yb, xb])
            if v > terbaik:
                terbaik, bx, by = v, dx, dy
    return nol, bx, by, terbaik


def _sisi(ax, ay, bx, by, px, py) -> float:
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def dalam_poligon(p: tuple[float, float], poli: list) -> bool:
    """LC-015: titik di dalam poligon (ray casting)."""
    x, y = p
    ada = False
    for i in range(len(poli)):
        (x1, y1), (x2, y2) = poli[i][:2], poli[(i + 1) % len(poli)][:2]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            ada = not ada
    return ada


def potong(g, x1, y1, x2, y2) -> int:
    ax, ay, bx, by = g
    s1, s2 = _sisi(ax, ay, bx, by, x1, y1), _sisi(ax, ay, bx, by, x2, y2)
    if s1 == 0 or (s1 > 0) == (s2 > 0):
        return -1
    s3, s4 = _sisi(x1, y1, x2, y2, ax, ay), _sisi(x1, y1, x2, y2, bx, by)
    if (s3 > 0) == (s4 > 0):
        return -1
    return 0 if s1 > 0 else 1


class Langsung:
    def __init__(self) -> None:
        self._kunci = threading.Lock()
        # LC-017: satu "mulai" pada satu waktu. Dua permintaan yang hampir
        # bersamaan (mis. dua tab / tarik() yang mengulang) dulu masing-masing
        # membuat proses deteksi; hanya yang terakhir dilacak, yang lain
        # tertinggal yatim memakan CPU & memori (terukur 26-09 malam: 2 proses
        # HD yatim, +-1,2 GB masing-masing).
        self._kunci_mulai = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self.key: str | None = None
        self.mulai_pada = 0.0
        self.last_poll = 0.0
        self.info: dict = {}
        self.galat: str | None = None
        self._reset()

    def _reset(self) -> None:
        self.frames: collections.deque = collections.deque(maxlen=SIMPAN_FRAME)
        self.peristiwa: collections.deque = collections.deque(maxlen=5000)
        self._tertahan: list = []
        self._seq = 0
        self.track: dict[int, dict] = {}
        self.tampilan: list[dict] = []
        self._tampilan_dibaca = 0.0
        self.aktif = -1
        self.skor_posisi: float | None = None
        self._calon, self._calon_n = -2, 0
        self._abu_riwayat: collections.deque = collections.deque(maxlen=4)
        self.ms_terakhir: int | None = None
        self.tunda_terakhir: float | None = None
        self.diterima_terakhir = 0.0
        self._ms_hist: collections.deque = collections.deque(maxlen=120)
        self._tunda_hist: collections.deque = collections.deque(maxlen=120)
        self._waktu_frame: collections.deque = collections.deque(maxlen=120)
        self.capture: dict = {"diterima": 0, "dibuang_total": 0, "putus": 0,
                              "antre": 0}
        self._traffic_window: collections.deque = collections.deque(maxlen=60)
        self.traffic: dict = {"state": "UNKNOWN", "status": "PROVISIONAL",
                              "reason": "belum cukup frame", "samples": 0}
        self._traffic_write_at = 0.0
        # LC-012: ringkasan sesi untuk lalin.sesi_hitung
        self._n_frame = 0
        self._detik_hitung = 0.0
        self._pts_lalu: float | None = None
        self._sesi_catat_t = 0.0
        self._sesi_baris: dict | None = None

    @property
    def running(self) -> bool:
        return bool(self._proc and self._proc.poll() is None)

    # ------------------------------------------------------------ kendali
    def mulai(self, key: str, model_mode: str = "accurate",
              cpu_threads: int | None = None,
              sample_interval: float | None = None) -> dict:
        with self._kunci_mulai:
            return self._mulai(key, model_mode, cpu_threads, sample_interval)

    def _mulai(self, key: str, model_mode: str = "accurate",
               cpu_threads: int | None = None,
               sample_interval: float | None = None) -> dict:
        if model_mode not in ("accurate", "compact"):
            raise ValueError("model harus 'accurate' atau 'compact'")
        threads = int(cpu_threads or CPU_THREADS)
        interval = float(sample_interval or (0.25 if model_mode == "compact" else INTERVAL_DTK))
        if not 1 <= threads <= 8:
            raise ValueError("cpu_threads harus 1..8")
        if not 0.15 <= interval <= 2.0:
            raise ValueError("sample_interval harus 0.15..2.0 detik")
        cams = {c.key: c for c in batch.load_cameras()}
        if key not in cams:
            raise KeyError(key)
        cam = cams[key]
        # RTSP cameras are captured server-side by the live worker. Browser
        # playback is handled separately through the MediaMTX HLS relay.
        self.last_poll = time.time()
        if (self.running and self.key == key
                and getattr(self, "model_mode", None) == model_mode
                and getattr(self, "cpu_threads", None) == threads
                and getattr(self, "sample_interval", None) == interval):
            return self.status()
        self.henti()
        with self._kunci:
            self._reset()
            self.key = key
            self.model_mode = model_mode
            self.cpu_threads = threads
            self.sample_interval = interval
            self.mulai_pada = time.time()
            # LC-006: penanda sesi. Nomor urut peristiwa (seq) mulai lagi dari
            # 0 setiap sesi; peramban yang masih memegang seq lama harus tahu
            # sesinya berganti, kalau tidak peristiwa baru tersaring diam-diam.
            self.sesi_id = f"{int(self.mulai_pada * 1000)}"
            # Aturan kelas dibaca ulang tiap sesi: perubahan klasifikasi.yml
            # berlaku pada sesi Live berikutnya tanpa restart kontainer.
            self.aturan = klasifikasi.muat(KLASIFIKASI_YML)
            self.galat = None
            self.info = {}
            # LC-012: cakupan penghitungan dicatat sejak sesi mulai.
            self._sesi_baris = {
                "sesi_id": self.sesi_id, "camera": key, "mulai_pada": simpan_lintas.iso(self.mulai_pada),
                "akhir_pada": None, "n_frame": 0, "detik_terhitung": 0.0, "n_lintas": 0,
                "garis": simpan_lintas.js([t.get("garis") for t in batch.load_tampilan_hitung().get(key, [])]),
                "versi_aturan": self.aturan.versi,
                "model": "ppyoloe_coco" if model_mode == "compact" else settings.hitung_model.name,
                "catatan": None}
            simpan_lintas.pencatat.sesi(**self._sesi_baris)
        entry = Path(__file__).parent / "_entry_langsung.py"
        env = dict(os.environ)
        env["LALIN_PD_DIR"] = str(settings.paddledet / "deploy" / "pptracking" / "python")
        model_dir = (settings.models_dir / "ppyoloe_coco"
                     if model_mode == "compact" else settings.hitung_model)
        if not (model_dir / "model.pdmodel").is_file():
            raise FileNotFoundError(f"Model Live tidak tersedia: {model_dir}")
        env["LALIN_MODEL_DIR"] = str(model_dir)
        env["LALIN_TRACKER_CFG"] = str(settings.hitung_tracker_cfg)
        env["LALIN_SOURCE"] = cam.stream_url
        env["LALIN_INTERVAL"] = str(interval)
        env["LALIN_CPU_THREADS"] = str(threads)
        # GPU dipilih oleh deployment target; server CPU yang ada tetap
        # mengirim CPU. Menggunakan env eksplisit mencegah image GPU diam-diam
        # tetap menjalankan SDE_Detector di CPU.
        env["LALIN_DEVICE"] = os.environ.get(
            "LALIN_DEVICE", "GPU" if settings.profile == "gpu" else "CPU").upper()
        # LC-018: recall trial for the single-camera HD POC. Admit weaker
        # detector candidates; the separate motorcycle class gate (0.45 and
        # >= 50% of track observations) is unchanged, so weak candidates stay
        # UNKNOWN for evaluation rather than being forced into a class.
        env["LALIN_THRESHOLD"] = "0.20"
        self._proc = subprocess.Popen(
            [sys.executable, str(entry)], cwd=str(settings.paddledet), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            encoding="utf-8", errors="replace", bufsize=1)
        self._thread = threading.Thread(target=self._baca, args=(self._proc, key), daemon=True)
        self._thread.start()
        threading.Thread(target=self._jaga, args=(self._proc,), daemon=True).start()
        return self.status()

    def henti(self) -> None:
        p = self._proc
        self._proc = None
        if p:
            self._catat_sesi(catatan=self.galat)
        if p and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    def _jaga(self, proc: subprocess.Popen) -> None:
        # Berhenti sendiri bila tidak ada yang menonton lagi.
        while proc.poll() is None:
            time.sleep(5)
            if time.time() - self.last_poll > DIAM_BERHENTI and self._proc is proc:
                self.galat = "berhenti sendiri: tidak ada yang menonton"
                self.henti()
                return

    # ------------------------------------------------------------ baca hasil
    def _baca(self, proc: subprocess.Popen, key: str) -> None:
        assert proc.stdout is not None
        for baris in proc.stdout:
            if self._proc is not proc:
                break
            try:
                d = json.loads(baris)
            except json.JSONDecodeError:
                continue
            jenis = d.get("jenis")
            if jenis == "mulai":
                self.info = {k: d.get(k) for k in ("malam", "ambang", "interval", "model", "at")}
            elif jenis == "galat":
                self.galat = d.get("pesan")
            elif jenis == "putus":
                self.capture.update(d.get("capture") or {})
                self.capture["putus"] = d.get("ke", self.capture.get("putus", 0))
                self.galat = f"sumber putus; reconnect #{self.capture['putus']}"
            elif jenis == "frame" and d.get("pts") is not None:
                try:
                    self._olah(key, d)
                except Exception as exc:  # jangan sampai satu frame mematikan pembaca
                    self.galat = f"olah: {type(exc).__name__}: {exc}"

    def _muat_tampilan(self, key: str) -> None:
        if time.time() - self._tampilan_dibaca < 5:
            return
        self._tampilan_dibaca = time.time()
        daftar = batch.load_tampilan_hitung().get(key, [])
        baru = []
        lama = {json.dumps(t["garis"]): t for t in self.tampilan}
        for t in daftar:
            acuan = [a for a in (_abu(t.get("acuan")), _abu(t.get("acuan_2"))) if a is not None]
            entri = {"garis": t["garis"], "tepi": [_tepi(a) for a in acuan] or None,
                     "dinamis": None, "dinamis_t": 0.0,
                     "pejalan": t.get("garis_pejalan") or [], "jalan": t.get("badan_jalan")}
            sebelum = lama.get(json.dumps(t["garis"]))
            if sebelum:
                entri["dinamis"], entri["dinamis_t"] = sebelum.get("dinamis"), sebelum.get("dinamis_t", 0.0)
            baru.append(entri)
        # Garis berubah (disunting operator): pencocokan dimulai lagi.
        if [t["garis"] for t in baru] != [t["garis"] for t in self.tampilan]:
            self._calon, self._calon_n = -2, 0
            if self.aktif >= len(baru):
                self.aktif = -1
        self.tampilan = baru

    def _periksa_posisi(self, abu_b64: str) -> None:
        abu = _abu(abu_b64)
        if abu is None:
            return
        self._abu_riwayat.append(abu)
        kini = _tepi(np.mean(np.stack(self._abu_riwayat), axis=0))
        pilih, skor = -1, None
        for i, t in enumerate(self.tampilan):
            if t["tepi"] is None:
                continue
            # Acuan tersimpan (siang/malam) + acuan dinamis: salinan tampilan
            # yang terakhir kali cocok, diperbarui tiap menit. Terang yang
            # berubah perlahan (fajar, senja, mendung) ikut terbawa; kamera
            # yang digeser PTZ berubah mendadak dan tetap tertangkap.
            for ref in t["tepi"] + ([t["dinamis"]] if t["dinamis"] is not None else []):
                nol, dx, dy, _ = cocokkan(ref, kini)
                if abs(dx) <= 1 and abs(dy) <= 1 and nol >= NCC_MIN and (skor is None or nol > skor):
                    pilih, skor = i, nol
        if pilih >= 0 and skor is not None and skor >= 0.25                 and time.time() - self.tampilan[pilih]["dinamis_t"] > 60:
            self.tampilan[pilih]["dinamis"] = kini
            self.tampilan[pilih]["dinamis_t"] = time.time()
        if pilih < 0 and len(self.tampilan) == 1 and self.tampilan[0]["tepi"] is None:
            pilih = 0
        if pilih == self._calon:
            self._calon_n += 1
        else:
            self._calon, self._calon_n = pilih, 1
        if self._calon_n >= (3 if pilih < 0 else 2) and pilih != self.aktif:
            self.aktif = pilih
            for tr in self.track.values():      # jangan hitung ulang objek yang sedang tampak
                tr["terhitung"] = True
        self.skor_posisi = round(skor, 3) if skor is not None else None

    def _olah(self, key: str, d: dict) -> None:
        with self._kunci:
            self._muat_tampilan(key)
            if d.get("abu"):
                self._periksa_posisi(d["abu"])
            pts = float(d["pts"])
            tpl = self.tampilan[self.aktif] if 0 <= self.aktif < len(self.tampilan) else None
            garis = tpl["garis"] if tpl else []
            pejalan = (tpl.get("pejalan") or []) if tpl else []
            jalan = tpl.get("jalan") if tpl else None
            keluar = []
            traffic_area: list[float] = []
            traffic_speed: list[float] = []
            for uid, label, x, y, w, h, sc, pengendara in d.get("kotak", []):
                if label == "person" and pengendara:
                    continue        # pengendara: sudah terwakili motor/sepedanya
                if not self.aturan.dikenal(label):
                    continue
                roda = (x + w / 2, y + h)
                tr = self.track.get(uid)
                if tr is None:
                    # ID tracker unik per kelas COCO, jadi satu track = satu label
                    tr = self.track[uid] = {"label": label, "yakin": 0, "roda": roda,
                                            "awal": roda, "terhitung": False, "lihat": 0,
                                            "peristiwa": None, "jejak": collections.deque(maxlen=JEJAK_MAKS)}
                tr["lihat"] += 1
                # LC-014: label track = suara terbanyak pengamatannya (bus/truck
                # kini satu track); seri -> label yang sudah ada.
                suara = tr.setdefault("suara", {})
                suara[label] = suara.get(label, 0) + 1
                if suara[label] > suara.get(tr["label"], 0):
                    tr["label"] = label
                if sc >= self.aturan.ambang(label):
                    tr["yakin"] += 1
                if label in {"motorcycle", "car", "bus", "truck"}:
                    traffic_area.append(max(0.0, float(w) * float(h)))
                    dt = pts - float(tr.get("waktu", pts))
                    if dt > 0:
                        traffic_speed.append(math.dist(roda, tr["roda"]) / dt)
                # LC-015: orang yang berada di badan jalan / lajur sepeda
                if label == "person" and jalan:
                    tr["cek_jalan"] = tr.get("cek_jalan", 0) + 1
                    if dalam_poligon(roda, jalan):
                        tr["di_jalan"] = tr.get("di_jalan", 0) + 1
                tr["skor_maks"] = max(tr.get("skor_maks", 0.0), float(sc))
                tr["skor_jml"] = tr.get("skor_jml", 0.0) + float(sc)
                # Rekam metadata lintasan, tanpa frame, pelat, atau identitas.
                # Salinan ini dipasang ke baris DB ketika kendaraan melintas,
                # sehingga anotator dapat mengaitkan seq -> track -> bbox pada
                # PTS yang benar tanpa menebak dari keyframe HD terdekat.
                tr["jejak"].append({"pts": round(pts, 3), "bbox": [round(float(v), 1) for v in (x, y, w, h)],
                                    "skor": round(float(sc), 4), "label": label})
                kelas = self._putuskan(tr).kunci
                lintas = False
                # LC-015: garis pejalan kaki hanya menghitung orang; nomor ruasnya
                # disambung setelah ruas garis kendaraan.
                ruas_hitung = list(enumerate(garis))
                if label == "person":
                    ruas_hitung += [(len(garis) + i, g) for i, g in enumerate(pejalan)]
                if (ruas_hitung and not tr["terhitung"] and tr["lihat"] >= 2
                        and math.dist(roda, tr["awal"]) >= GERAK_MIN_PX):
                    for r, g in ruas_hitung:
                        arah = potong(g, *tr["roda"], *roda)
                        if arah >= 0:
                            tr["terhitung"] = lintas = True
                            tr["peristiwa"] = {"pts": pts, "id": uid, "kelas": kelas,
                                               "ruas": r, "arah": arah, "tampilan": self.aktif,
                                               "t_server": time.time(),
                                               "jenis_garis": "pejalan" if r >= len(garis) else "kendaraan",
                                               "bbox": [round(float(v), 1) for v in (x, y, w, h)]}
                            self._tertahan.append((uid, tr))
                            break
                tr["roda"] = roda
                tr["waktu"] = pts
                keluar.append([uid, kelas, x, y, w, h, sc, 1 if tr["terhitung"] else 0])
            self._olah_traffic(key, pts, traffic_area, traffic_speed)
            # Peristiwa tertahan dilepas dengan kelas akhir track-nya.
            sisa = []
            for uid, tr in self._tertahan:
                ev = tr["peristiwa"]
                if pts - tr.get("waktu", pts) >= 2.0 or pts - ev["pts"] >= TAHAN_DTK:
                    kp = self._putuskan(tr)
                    # kelas = kunci tampil lama (kontrak API); field baru bersifat
                    # tambahan: status, kelas akhir (null bila AMBIGUOUS/UNKNOWN),
                    # kandidat, label detektor mentah, versi aturan.
                    ev.update(kelas=kp.kunci, status=kp.status, final_class=kp.final_class,
                              kandidat=list(kp.kandidat), label_detektor=tr["label"],
                              n_lihat=tr["lihat"], n_yakin=tr["yakin"],
                              versi_aturan=self.aturan.versi, alasan_status=self._alasan(tr, kp))
                    self._seq += 1
                    ev["seq"] = self._seq
                    self.peristiwa.append(ev)
                    simpan_lintas.pencatat.lintas(self._baris_lintas(key, ev, tr))
                else:
                    sisa.append((uid, tr))
            self._tertahan = sisa
            # buang track basi
            for uid in [u for u, t in self.track.items() if pts - t.get("waktu", pts) > 30]:
                del self.track[uid]
            self.frames.append({"pts": pts, "kotak": keluar, "tampilan": self.aktif,
                                "traffic": dict(self.traffic)})
            self.ms_terakhir = d.get("ms")
            self.tunda_terakhir = d.get("tunda")
            self.diterima_terakhir = time.time()
            if self.ms_terakhir is not None:
                self._ms_hist.append(float(self.ms_terakhir))
            if self.tunda_terakhir is not None:
                self._tunda_hist.append(float(self.tunda_terakhir))
            self._waktu_frame.append(self.diterima_terakhir)
            self.capture.update(d.get("capture") or {})
            if self.galat and self.galat.startswith("sumber putus"):
                self.galat = None
            # LC-012: cakupan = detik dengan garis aktif (posisi kamera cocok)
            self._n_frame += 1
            if self._pts_lalu is not None and self.aktif >= 0 and 0 < pts - self._pts_lalu <= 5:
                self._detik_hitung += pts - self._pts_lalu
            self._pts_lalu = pts
            if time.time() - self._sesi_catat_t > 30:
                self._catat_sesi()

    def _catat_sesi(self, catatan: str | None = None) -> None:
        if not self._sesi_baris:
            return
        self._sesi_catat_t = time.time()
        self._sesi_baris.update(
            akhir_pada=simpan_lintas.iso(self.diterima_terakhir or None), n_frame=self._n_frame,
            detik_terhitung=round(self._detik_hitung, 1), n_lintas=self._seq,
            catatan=catatan or self._sesi_baris.get("catatan"))
        simpan_lintas.pencatat.sesi(**self._sesi_baris)

    def _olah_traffic(self, key: str, pts: float, area: list[float], speed: list[float]) -> None:
        """Metrik kemacetan transparan dari track; rule awal, belum UAT."""
        occupancy = min(1.0, sum(area) / (640.0 * 360.0))
        median_speed = float(np.median(speed)) if speed else 0.0
        stopped = (sum(1 for v in speed if v < 5.0) / len(speed)) if speed else 0.0
        self._traffic_window.append({"pts": pts, "visible": len(area),
                                     "occupancy": occupancy, "speed": median_speed,
                                     "stopped": stopped, "speed_n": len(speed)})
        window = list(self._traffic_window)
        occ = float(np.median([x["occupancy"] for x in window]))
        spd = float(np.median([x["speed"] for x in window]))
        stop = float(np.median([x["stopped"] for x in window]))
        vis = int(round(float(np.median([x["visible"] for x in window]))))
        speed_n = sum(int(x["speed_n"]) for x in window)
        quality = min(1.0, len(window) / 20.0) * min(1.0, speed_n / 20.0) \
            * min(1.0, vis / 5.0)
        if len(window) < 10:
            state, reason = "UNKNOWN", "minimal 10 frame"
        elif quality < .5:
            state, reason = "UNKNOWN", "track bergerak belum cukup untuk verdict"
        elif occ >= .18 and stop >= .55 and spd < 8:
            state, reason = "MACET", "occupancy tinggi dan mayoritas berhenti"
        elif occ >= .14 or (vis >= 8 and spd < 15):
            state, reason = "PADAT", "occupancy tinggi atau gerak lambat"
        elif occ >= .07 or vis >= 5:
            state, reason = "RAMAI", "volume kendaraan meningkat"
        else:
            state, reason = "LANCAR", "occupancy rendah"
        self.traffic = {"state": state, "status": "PROVISIONAL", "reason": reason,
                        "samples": len(window), "visible": vis,
                        "speed_observations": speed_n,
                        "quality_score": round(quality, 3),
                        "occupancy_ratio": round(occ, 4),
                        "median_speed_px_s": round(spd, 2),
                        "stopped_ratio": round(stop, 4)}
        # Snapshot kecil untuk audit dan agar hasil terakhir tetap terlihat
        # setelah proses Live dihentikan. Maksimal satu tulis tiap lima detik.
        if time.time() - self._traffic_write_at >= 5:
            self._traffic_write_at = time.time()
            path = settings.runs_dir / "_latest" / f"{key}_traffic.json"
            tmp = path.with_suffix(".tmp")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps({"key": key, "at": simpan_lintas.iso(time.time()),
                                           **self.traffic}, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                tmp.replace(path)
            except OSError:
                pass

    def _baris_lintas(self, key: str, ev: dict, tr: dict) -> dict:
        """Satu baris lalin.lintas_kendaraan: data mentah + versi aturan."""
        return {
            "sesi_id": self.sesi_id, "seq": ev["seq"], "camera": key,
            "waktu": simpan_lintas.iso(ev.get("t_server") or time.time()), "pts": ev.get("pts"),
            "track_id": str(ev.get("id")), "tampilan": ev.get("tampilan"), "ruas": ev.get("ruas"),
            "arah": ev.get("arah"), "label_detektor": ev.get("label_detektor"), "kelas_tampil": ev["kelas"],
            "status": ev["status"], "final_class": ev.get("final_class"),
            "kandidat": simpan_lintas.js(ev.get("kandidat")), "n_lihat": ev.get("n_lihat"),
            "n_yakin": ev.get("n_yakin"), "skor_maks": round(tr.get("skor_maks", 0.0), 4),
            "skor_rata": round(tr.get("skor_jml", 0.0) / max(tr.get("lihat", 1), 1), 4),
            "malam": bool(self.info.get("malam")) if self.info else None,
            "bbox": simpan_lintas.js(ev.get("bbox")),
            "jejak": simpan_lintas.js(list(tr.get("jejak") or [])),
            "versi_aturan": ev.get("versi_aturan") or self.aturan.versi,
            "alasan_status": ev.get("alasan_status"), "jenis_garis": ev.get("jenis_garis")}

    @staticmethod
    def _orang_di_jalan(tr: dict) -> bool:
        return (tr["label"] == "person" and tr.get("cek_jalan", 0) > 0
                and tr.get("di_jalan", 0) * 2 >= tr["cek_jalan"])

    def _putuskan(self, tr: dict) -> "klasifikasi.Keputusan":
        # Orang harus tetap dapat dibaca sebagai Pejalan Kaki bila modelnya
        # benar-benar mendeteksi person, bahkan ketika sebagian observasi
        # terlihat di badan jalan. Kebijakan UNKNOWN hanya dipakai untuk objek
        # yang tidak cocok dengan taksonomi kendaraan dan tidak punya klasifikasi
        # yang jelas; bukan untuk person yang sudah valid dikenali.
        # AturanKlasifikasi saat ini mengambil tiga argumen. `final_class`
        # belum menjadi bagian kontraknya; mengirim keyword itu membuat setiap
        # frame live gagal diolah sebelum masuk telemetry/counting.
        return self.aturan.putuskan(tr["label"], tr["lihat"], tr["yakin"])

    def _alasan(self, tr: dict, kp) -> str | None:
        if kp.status == "UNKNOWN":
            return "yakin_rendah"
        if kp.status == "AMBIGUOUS":
            return "label_detektor_ambigu"
        return None

    # ------------------------------------------------------------ baca API
    def ambil(self, key: str, sejak: float | None, sejak_p: int | None = None) -> dict:
        self.last_poll = time.time()
        st = self.status()
        if key != self.key:
            st["frames"], st["peristiwa"] = [], []
            return st
        with self._kunci:
            if sejak is None:
                st["frames"] = list(self.frames)
                st["peristiwa"] = list(self.peristiwa)
            else:
                st["frames"] = [f for f in self.frames if f["pts"] > sejak]
                st["peristiwa"] = [p for p in self.peristiwa
                                   if p.get("seq", 0) > (sejak_p or 0)]
        return st

    def status(self) -> dict:
        def persentil(data, p: float) -> float | None:
            if not data:
                return None
            nilai = sorted(float(x) for x in data)
            i = min(len(nilai) - 1, max(0, round((len(nilai) - 1) * p)))
            return round(nilai[i], 2)

        fps = None
        if len(self._waktu_frame) >= 2:
            rentang = self._waktu_frame[-1] - self._waktu_frame[0]
            if rentang > 0:
                fps = round((len(self._waktu_frame) - 1) / rentang, 2)
        umur = (round(time.time() - self.diterima_terakhir, 1)
                if self.diterima_terakhir else None)
        p95_tunda = persentil(self._tunda_hist, .95)
        if not self.running:
            kondisi = "STOPPED"
        elif umur is None or umur > 15:
            kondisi = "SOURCE_DOWN"
        elif (self.capture.get("dibuang_total") or 0) > 0 or (p95_tunda or 0) > 2:
            kondisi = "DEGRADED"
        else:
            kondisi = "HEALTHY"
        return {
            "running": self.running, "key": self.key, "galat": self.galat,
            "model_mode": getattr(self, "model_mode", None),
            "cpu_threads": getattr(self, "cpu_threads", None),
            "sample_interval": getattr(self, "sample_interval", None),
            "sesi": getattr(self, "sesi_id", None),
            "klasifikasi": self.aturan.ringkas() if getattr(self, "aturan", None) else None,
            "info": self.info, "aktif": self.aktif, "skor_posisi": self.skor_posisi,
            "jumlah_tampilan": len(self.tampilan),
            "ms": self.ms_terakhir, "tunda": self.tunda_terakhir,
            "detik_sejak_frame": umur,
            "kondisi": kondisi,
            "telemetry": {
                "effective_fps": fps,
                "inference_ms_p50": persentil(self._ms_hist, .50),
                "inference_ms_p95": persentil(self._ms_hist, .95),
                "queue_delay_sec_p50": persentil(self._tunda_hist, .50),
                "queue_delay_sec_p95": p95_tunda,
                "samples": len(self._ms_hist),
                "capture": dict(self.capture),
            },
            "traffic": dict(self.traffic),
            "pts_terakhir": self.frames[-1]["pts"] if self.frames else None,
            "pts_awal": self.frames[0]["pts"] if self.frames else None,
            "db": simpan_lintas.pencatat.status(),
        }

    def touch(self) -> None:
        """Pertahankan sesi ketika halaman monitor dua kamera masih dibuka."""
        self.last_poll = time.time()


class SharedDual:
    """Satu predictor subprocess, beberapa state tracker/olah per kamera."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        self.keys = keys
        self.engines = {key: Langsung() for key in keys}
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.last_touch = 0.0
        self.model_mode: str | None = None
        self.cpu_threads: int | None = None
        self.sample_interval: float | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=30)

    @property
    def running(self) -> bool:
        return bool(self._proc and self._proc.poll() is None)

    def start(self, model_mode: str = "compact", cpu_threads: int = 8,
              sample_interval: float = .5) -> dict:
        if model_mode not in ("compact", "accurate"):
            raise ValueError("model harus compact atau accurate")
        if not 1 <= int(cpu_threads) <= 8:
            raise ValueError("cpu_threads harus 1..8")
        if not .15 <= float(sample_interval) <= 2:
            raise ValueError("sample_interval harus 0.15..2.0")
        with self._lock:
            self.stop()
            cams = {c.key: c for c in batch.load_cameras()}
            hilang = [key for key in self.keys if key not in cams]
            if hilang:
                raise KeyError(", ".join(hilang))
            model_dir = (settings.models_dir / "ppyoloe_coco"
                         if model_mode == "compact" else settings.hitung_model)
            if not (model_dir / "model.pdmodel").is_file():
                raise FileNotFoundError(f"Model shared tidak tersedia: {model_dir}")
            now = time.time()
            for key, engine in self.engines.items():
                with engine._kunci:
                    engine._reset()
                    engine.key = key
                    engine.model_mode = model_mode
                    engine.cpu_threads = int(cpu_threads)
                    engine.sample_interval = float(sample_interval)
                    engine.mulai_pada = now
                    engine.last_poll = now
                    engine.sesi_id = f"{int(now * 1000)}-{key}"
                    engine.aturan = klasifikasi.muat(KLASIFIKASI_YML)
                    engine.galat = None
                    engine.info = {}
                    engine._sesi_baris = {
                        "sesi_id": engine.sesi_id, "camera": key,
                        "mulai_pada": simpan_lintas.iso(now), "akhir_pada": None,
                        "n_frame": 0, "detik_terhitung": 0.0, "n_lintas": 0,
                        "garis": simpan_lintas.js([t.get("garis") for t in
                                                   batch.load_tampilan_hitung().get(key, [])]),
                        "versi_aturan": engine.aturan.versi, "model": model_dir.name,
                        "catatan": "shared predictor"}
                    simpan_lintas.pencatat.sesi(**engine._sesi_baris)
            env = dict(os.environ)
            env["LALIN_PD_DIR"] = str(settings.paddledet / "deploy" / "pptracking" / "python")
            env["LALIN_MODEL_DIR"] = str(model_dir)
            env["LALIN_TRACKER_CFG"] = str(settings.hitung_tracker_cfg)
            env["LALIN_SOURCES_JSON"] = json.dumps(
                {key: cams[key].stream_url for key in self.keys})
            # Compatibility untuk _entry_hitung yang membaca variabel ini saat
            # import. Worker shared tetap memakai LALIN_SOURCES_JSON.
            env["LALIN_SOURCE"] = cams[self.keys[0]].stream_url
            env["LALIN_INTERVAL"] = str(float(sample_interval))
            env["LALIN_CPU_THREADS"] = str(int(cpu_threads))
            env["LALIN_DEVICE"] = "CPU"
            env["LALIN_THRESHOLD"] = "0.20"
            entry = Path(__file__).parent / "_entry_langsung.py"
            proc = subprocess.Popen(
                [sys.executable, str(entry)], cwd=str(settings.paddledet), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding="utf-8", errors="replace", bufsize=1,
                start_new_session=True)
            self._proc = proc
            for engine in self.engines.values():
                engine._proc = proc
            self.model_mode = model_mode
            self.cpu_threads = int(cpu_threads)
            self.sample_interval = float(sample_interval)
            self.last_touch = now
            self._stderr_tail.clear()
            threading.Thread(target=self._baca, args=(proc,), daemon=True).start()
            threading.Thread(target=self._baca_stderr, args=(proc,), daemon=True).start()
            threading.Thread(target=self._jaga, args=(proc,), daemon=True).start()
            # Gagal cepat harus kembali sebagai HTTP 500, bukan active palsu.
            batas = time.time() + 5
            while time.time() < batas and proc.poll() is None:
                time.sleep(.1)
            if proc.poll() is not None:
                pesan = self._pesan_exit(proc)
                self.stop()
                raise RuntimeError(pesan)
        return self.status()

    def _baca(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for baris in proc.stdout:
            if self._proc is not proc:
                break
            try:
                d = json.loads(baris)
            except json.JSONDecodeError:
                continue
            jenis, key = d.get("jenis"), d.get("key")
            tujuan = ([self.engines[key]] if key in self.engines
                      else list(self.engines.values()))
            if jenis == "mulai":
                for engine in tujuan:
                    engine.info = {k: d.get(k) for k in
                                   ("malam", "ambang", "interval", "model", "at")}
            elif jenis == "galat":
                for engine in tujuan:
                    engine.galat = d.get("pesan")
            elif jenis == "putus":
                for engine in tujuan:
                    engine.capture.update(d.get("capture") or {})
                    engine.galat = f"sumber putus; reconnect #{d.get('ke', 0)}"
            elif jenis == "frame" and key in self.engines and d.get("pts") is not None:
                try:
                    self.engines[key]._olah(key, d)
                except Exception as exc:
                    self.engines[key].galat = f"olah shared: {type(exc).__name__}: {exc}"
        if self._proc is proc and proc.poll() is not None:
            pesan = self._pesan_exit(proc)
            for engine in self.engines.values():
                engine.galat = pesan
                if engine._proc is proc:
                    engine._proc = None

    def _baca_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for baris in proc.stderr:
            teks = baris.strip()
            if teks:
                self._stderr_tail.append(teks)

    def _pesan_exit(self, proc: subprocess.Popen) -> str:
        kode = proc.poll()
        rincian = self._stderr_tail[-1] if self._stderr_tail else "tanpa stderr"
        return f"shared worker exit {kode}: {rincian}"

    def touch(self) -> None:
        self.last_touch = time.time()
        for engine in self.engines.values():
            engine.touch()

    def _jaga(self, proc: subprocess.Popen) -> None:
        while proc.poll() is None:
            time.sleep(5)
            if time.time() - self.last_touch > 90 and self._proc is proc:
                self.stop()
                return

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        for engine in self.engines.values():
            if engine._proc is proc:
                engine._proc = None
                engine._catat_sesi(catatan=engine.galat or "shared predictor berhenti")
        if proc and proc.poll() is None:
            try:
                # Worker membuat dua FFmpeg. Akhiri seluruh process group agar
                # tidak ada decoder orphan yang tetap memakan CPU setelah Stop.
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=8)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    def status(self) -> dict:
        return {"running": self.running, "mode": "shared-predictor",
                "model": self.model_mode, "cpu_threads": self.cpu_threads,
                "sample_interval": self.sample_interval,
                "stderr_tail": list(self._stderr_tail)[-5:],
                "cameras": {key: engine.status() for key, engine in self.engines.items()}}


langsung = Langsung()


import atexit  # noqa: E402
atexit.register(langsung.henti)
