"""Lapisan pemetaan kelas pusat, dibaca dari configs/klasifikasi.yml.

Sebelumnya pemetaan COCO -> kelas dan ambang yakin ditanam di langsung.py,
_entry_hitung.py, dan bbox.js dengan isi yang berbeda-beda (audit 01, R4).
Modul ini menjadi satu-satunya tempat keputusan kelas untuk Live:

    putuskan(label_detektor, n_lihat, n_yakin) -> Keputusan

Status yang mungkin: CLASSIFIED (satu kelas) atau UNKNOWN (detektor belum
memberi dasar untuk satu kelas Dishub). Label yang perlu dibedakan dengan
ukuran bbox memakai aturan sementara dari konfigurasi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STATUS_SUMBER = {"OFFICIAL", "PROJECT_BASELINE", "PROVISIONAL", "PENDING_VALIDATION"}
LABEL_MENTAH_TAMBAHAN = {"car", "bus", "truck"}


@dataclass(frozen=True)
class Keputusan:
    kunci: str                 # kunci tampil API (sepeda_motor, bus, truk, tidak_dikenal, ...)
    status: str                # CLASSIFIED | UNKNOWN
    final_class: str | None    # salah satu dari 7 kelas, atau None
    kandidat: tuple[str, ...] = field(default_factory=tuple)


class AturanKlasifikasi:
    def __init__(self, data: dict[str, Any]):
        self.versi = str(data.get("versi", ""))
        self.kelas = [dict(k) for k in data.get("kelas", [])]
        self.kunci_kelas = [k["kunci"] for k in self.kelas]
        if len(self.kunci_kelas) != 7 or len(set(self.kunci_kelas)) != 7:
            raise ValueError("klasifikasi.yml: harus tepat 7 kelas dengan kunci unik")
        self.jenis_rinci = dict(data.get("jenis_rinci") or {})
        self.label = dict(data.get("label_detektor") or {})
        g = data.get("gerbang") or {}
        self.porsi_yakin = float(g.get("porsi_yakin", 0.5))
        self.kunci_tak = str(g.get("kunci_tidak_dikenal", "tidak_dikenal"))
        self.nama_tak = str(g.get("nama_tidak_dikenal", "Tidak Dikenal"))
        self._periksa()

    def _periksa(self) -> None:
        sah = set(self.kunci_kelas)
        for nama, j in self.jenis_rinci.items():
            if j.get("kelas") is not None and j["kelas"] not in sah:
                raise ValueError(f"jenis_rinci.{nama}: kelas tidak dikenal {j['kelas']!r}")
            if j.get("status") not in STATUS_SUMBER:
                raise ValueError(f"jenis_rinci.{nama}: status tidak sah {j.get('status')!r}")
        for lab, a in self.label.items():
            kandidat = a.get("kandidat")
            aturan_ukuran = a.get("aturan_ukuran")
            kelas_raw = a.get("kelas_raw")
            if kandidat:
                if not set(kandidat) <= sah or len(kandidat) < 2:
                    raise ValueError(f"label_detektor.{lab}: kandidat tidak sah")
                if not a.get("kunci_tampil") or a["kunci_tampil"] in sah:
                    raise ValueError(f"label_detektor.{lab}: kunci_tampil wajib dan bukan kunci kelas")
            elif kelas_raw:
                if kelas_raw not in LABEL_MENTAH_TAMBAHAN or kelas_raw != lab:
                    raise ValueError(f"label_detektor.{lab}: kelas_raw tidak sah")
            elif aturan_ukuran:
                kelas_ukuran = {aturan_ukuran.get("di_bawah"), aturan_ukuran.get("di_atas_atau_sama")}
                ambang = float(aturan_ukuran.get("ambang_tinggi_rel_frame", 0))
                if not kelas_ukuran <= sah or len(kelas_ukuran) != 2:
                    raise ValueError(f"label_detektor.{lab}: aturan_ukuran memakai kelas tidak sah")
                if not 0.0 < ambang < 1.0:
                    raise ValueError(f"label_detektor.{lab}: ambang tinggi bbox di luar (0,1)")
            elif a.get("kelas") not in sah:
                raise ValueError(f"label_detektor.{lab}: kelas tidak dikenal")
            if not 0.0 < float(a.get("ambang_yakin", 0)) <= 1.0:
                raise ValueError(f"label_detektor.{lab}: ambang_yakin di luar (0,1]")

    # ------------------------------------------------------------ query
    def dikenal(self, label: str) -> bool:
        return label in self.label

    def ambang(self, label: str) -> float:
        return float(self.label[label]["ambang_yakin"])

    def putuskan(self, label: str, n_lihat: int, n_yakin: int,
                 final_class: str | None = None,
                 tinggi_rel_frame: float | None = None) -> Keputusan:
        """Keputusan track; final_class hanya berasal dari verifikasi eksplisit."""
        if final_class is not None:
            final = str(final_class)
            if final in self.kunci_kelas:
                return Keputusan(final, "CLASSIFIED", final, (final,))
            if final == self.kunci_tak:
                return Keputusan(final, "UNKNOWN", None, ())
        a = self.label.get(label)
        if a is None:
            return Keputusan(self.kunci_tak, "UNKNOWN", None, ())
        if a.get("kelas_raw"):
            # Untuk POC, label operasional COCO dipakai apa adanya. Dengan
            # demikian kendaraan_sedang/car, bus_besar/bus, dan truk_berat/truck
            # tidak bercampur antar endpoint atau antar frame.
            return Keputusan(a["kelas_raw"], "CLASSIFIED", None, (a["kelas_raw"],))
        if n_yakin < 1 or n_yakin < self.porsi_yakin * max(n_lihat, 1):
            return Keputusan(self.kunci_tak, "UNKNOWN", None, ())
        if a.get("kandidat"):
            return Keputusan(self.kunci_tak, "UNKNOWN", None, tuple(a["kandidat"]))
        aturan_ukuran = a.get("aturan_ukuran")
        if aturan_ukuran:
            if tinggi_rel_frame is None:
                return Keputusan(self.kunci_tak, "UNKNOWN", None,
                                 (aturan_ukuran["di_bawah"], aturan_ukuran["di_atas_atau_sama"]))
            kelas = (aturan_ukuran["di_atas_atau_sama"]
                     if tinggi_rel_frame >= float(aturan_ukuran["ambang_tinggi_rel_frame"])
                     else aturan_ukuran["di_bawah"])
            return Keputusan(kelas, "CLASSIFIED", kelas, (kelas,))
        return Keputusan(a["kelas"], "CLASSIFIED", a["kelas"], (a["kelas"],))

    def ringkas(self) -> dict:
        """Untuk API/UI: nama & warna kunci tampil."""
        tampil = {k["kunci"]: {"nama": k["nama"], "warna": k.get("warna"), "no": k.get("no")}
                  for k in self.kelas}
        for lab, a in self.label.items():
            if a.get("kandidat"):
                tampil[a["kunci_tampil"]] = {"nama": a.get("nama_tampil", a["kunci_tampil"]),
                                            "warna": a.get("warna"), "kandidat": list(a["kandidat"])}
            elif a.get("kelas_raw"):
                # Label operasional COCO harus tersedia di API/UI meskipun
                # bukan salah satu dari tujuh kelas resmi Dishub.
                raw = str(a["kelas_raw"])
                tampil[raw] = {"nama": a.get("nama_tampil", raw.title()),
                               "warna": a.get("warna")}
        tampil[self.kunci_tak] = {"nama": self.nama_tak, "warna": "#9ca3af"}
        return {"versi": self.versi, "kelas": self.kunci_kelas, "tampil": tampil}


def muat(path: Path | str) -> AturanKlasifikasi:
    with open(path, encoding="utf-8") as f:
        return AturanKlasifikasi(yaml.safe_load(f) or {})
