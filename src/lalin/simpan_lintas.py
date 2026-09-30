"""LC-012: catat penghitungan Live ke PostgreSQL.

Tabel `lalin.sesi_hitung` (cakupan) dan `lalin.lintas_kendaraan` (satu baris
per kendaraan yang melintas), DDL di docs/ddl/lintas_kendaraan.sql.

Penulisan berjalan di utasnya sendiri dengan antrean, supaya basis data yang
lambat atau mati tidak pernah menghambat deteksi Live. Bila tabel belum ada
atau DB tidak terjangkau, lintasan ditahan di memori (paling banyak
MAKS_ANTRE) dan dicoba lagi tiap JEDA_ULANG detik; keadaannya terbaca di
/api/langsung -> "db". Pada SQLite (laptop lokal) pencatatan tidak aktif.
"""

from __future__ import annotations

import collections
import json
import threading
import time
from datetime import datetime, timezone

from . import store

MAKS_ANTRE = 5000
JEDA_TULIS = 2.0
JEDA_ULANG = 30.0

SQL_SESI = (
    "INSERT INTO sesi_hitung (sesi_id, camera, mulai_pada, akhir_pada, n_frame, detik_terhitung,"
    " n_lintas, garis, versi_aturan, model, catatan) VALUES (?,?,?,?,?,?,?,?,?,?,?)"
    " ON CONFLICT (sesi_id) DO UPDATE SET akhir_pada = EXCLUDED.akhir_pada,"
    " n_frame = EXCLUDED.n_frame, detik_terhitung = EXCLUDED.detik_terhitung,"
    " n_lintas = EXCLUDED.n_lintas, catatan = COALESCE(EXCLUDED.catatan, sesi_hitung.catatan)")
KOLOM_SESI = ("sesi_id", "camera", "mulai_pada", "akhir_pada", "n_frame", "detik_terhitung",
              "n_lintas", "garis", "versi_aturan", "model", "catatan")

KOLOM_LINTAS = ("sesi_id", "seq", "camera", "waktu", "pts", "track_id", "tampilan", "ruas", "arah",
                "label_detektor", "kelas_tampil", "status", "final_class", "kandidat", "n_lihat",
                "n_yakin", "skor_maks", "skor_rata", "malam", "bbox", "versi_aturan",
                "alasan_status", "jenis_garis", "jejak")
SQL_LINTAS = (f"INSERT INTO lintas_kendaraan ({', '.join(KOLOM_LINTAS)})"
              f" VALUES ({','.join('?' * len(KOLOM_LINTAS))}) ON CONFLICT (sesi_id, seq) DO NOTHING")


def iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t, timezone.utc).isoformat()


def js(x) -> str | None:
    return None if x is None else json.dumps(x)


class Pencatat:
    def __init__(self, hubung=None):
        self._hubung = hubung or store._connect
        self._antre: collections.deque = collections.deque(maxlen=MAKS_ANTRE)
        self._sesi: dict[str, dict] = {}
        self._kunci = threading.Lock()
        self._bangun = threading.Event()
        self._utas: threading.Thread | None = None
        self.aktif = hubung is not None or store.DIALEK == "pg"
        self.galat: str | None = None
        self.tertulis = 0
        self.dibuang = 0
        self.terakhir: float | None = None

    def status(self) -> dict:
        return {"aktif": self.aktif, "tertulis": self.tertulis, "antre": len(self._antre),
                "dibuang": self.dibuang, "galat": self.galat, "terakhir": iso(self.terakhir)}

    # ------------------------------------------------------------ masukan
    def sesi(self, **baris) -> None:
        """Daftarkan / perbarui ringkasan sesi (upsert pada tulis berikutnya)."""
        if not self.aktif:
            return
        with self._kunci:
            self._sesi[baris["sesi_id"]] = {**self._sesi.get(baris["sesi_id"], {}), **baris}
        self._pastikan_utas()

    def lintas(self, baris: dict) -> None:
        if not self.aktif:
            return
        with self._kunci:
            if len(self._antre) == self._antre.maxlen:
                self.dibuang += 1
            self._antre.append(baris)
        self._pastikan_utas()
        self._bangun.set()

    # ------------------------------------------------------------ tulis
    def _pastikan_utas(self) -> None:
        if self._utas is None or not self._utas.is_alive():
            self._utas = threading.Thread(target=self._jalan, daemon=True, name="pencatat-lintas")
            self._utas.start()

    def _jalan(self) -> None:
        while True:
            self._bangun.wait(JEDA_TULIS)
            self._bangun.clear()
            if not self.tulis_sekarang():
                time.sleep(JEDA_ULANG)

    def tulis_sekarang(self) -> bool:
        with self._kunci:
            sesi = list(self._sesi.values())
            lintas = list(self._antre)
        if not sesi and not lintas:
            return True
        try:
            with self._hubung() as conn:
                # Sesi dulu: lintas_kendaraan.sesi_id merujuk sesi_hitung.
                for s in sesi:
                    conn.execute(SQL_SESI, tuple(s.get(k) for k in KOLOM_SESI))
                for b in lintas:
                    conn.execute(SQL_LINTAS, tuple(b.get(k) for k in KOLOM_LINTAS))
        except Exception as exc:
            self.galat = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
            return False
        with self._kunci:
            for s in sesi:
                if self._sesi.get(s["sesi_id"]) is s:
                    del self._sesi[s["sesi_id"]]
            # Buang yang sudah tertulis saja: selama penulisan, antrean penuh bisa
            # sudah membuang baris terlama sendiri.
            tertulis = {id(b) for b in lintas}
            while self._antre and id(self._antre[0]) in tertulis:
                self._antre.popleft()
        self.tertulis += len(lintas)
        self.terakhir = time.time()
        self.galat = None
        return True


pencatat = Pencatat()
