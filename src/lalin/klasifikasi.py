"""Lapisan pemetaan kelas pusat, dibaca dari configs/klasifikasi.yml.

Sebelumnya pemetaan COCO -> kelas dan ambang yakin ditanam di langsung.py,
_entry_hitung.py, dan bbox.js dengan isi yang berbeda-beda (audit 01, R4).
Modul ini menjadi satu-satunya tempat keputusan kelas untuk Live:

    putuskan(label_detektor, n_lihat, n_yakin) -> Keputusan

Status yang mungkin: CLASSIFIED (satu kelas), AMBIGUOUS (beberapa kelas
mungkin, final_class kosong), UNKNOWN (model tidak yakin). UNKNOWN adalah
status, bukan kelas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

STATUS_SUMBER = {"OFFICIAL", "PROJECT_BASELINE", "PROVISIONAL", "PENDING_VALIDATION"}


@dataclass(frozen=True)
class Keputusan:
    kunci: str                 # kunci tampil API (sepeda_motor, bus, truk, tidak_dikenal, ...)
    status: str                # CLASSIFIED | AMBIGUOUS | UNKNOWN
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
            if kandidat:
                if not set(kandidat) <= sah or len(kandidat) < 2:
                    raise ValueError(f"label_detektor.{lab}: kandidat tidak sah")
                if not a.get("kunci_tampil") or a["kunci_tampil"] in sah:
                    raise ValueError(f"label_detektor.{lab}: kunci_tampil wajib dan bukan kunci kelas")
            elif a.get("kelas") not in sah:
                raise ValueError(f"label_detektor.{lab}: kelas tidak dikenal")
            if not 0.0 < float(a.get("ambang_yakin", 0)) <= 1.0:
                raise ValueError(f"label_detektor.{lab}: ambang_yakin di luar (0,1]")

    # ------------------------------------------------------------ query
    def dikenal(self, label: str) -> bool:
        return label in self.label

    def ambang(self, label: str) -> float:
        return float(self.label[label]["ambang_yakin"])

    def putuskan(self, label: str, n_lihat: int, n_yakin: int) -> Keputusan:
        """Keputusan untuk satu track berlabel detektor `label`."""
        a = self.label.get(label)
        if a is None or n_yakin < 1 or n_yakin < self.porsi_yakin * max(n_lihat, 1):
            return Keputusan(self.kunci_tak, "UNKNOWN", None, ())
        if a.get("kandidat"):
            return Keputusan(a["kunci_tampil"], "AMBIGUOUS", None, tuple(a["kandidat"]))
        return Keputusan(a["kelas"], "CLASSIFIED", a["kelas"], (a["kelas"],))

    def ringkas(self) -> dict:
        """Untuk API/UI: nama & warna kunci tampil."""
        tampil = {k["kunci"]: {"nama": k["nama"], "warna": k.get("warna"), "no": k.get("no")}
                  for k in self.kelas}
        for lab, a in self.label.items():
            if a.get("kandidat"):
                tampil[a["kunci_tampil"]] = {"nama": a.get("nama_tampil", a["kunci_tampil"]),
                                            "warna": a.get("warna"), "kandidat": list(a["kandidat"])}
        tampil[self.kunci_tak] = {"nama": self.nama_tak, "warna": "#9ca3af"}
        return {"versi": self.versi, "kelas": self.kunci_kelas, "tampil": tampil}


def muat(path: Path | str) -> AturanKlasifikasi:
    with open(path, encoding="utf-8") as f:
        return AturanKlasifikasi(yaml.safe_load(f) or {})
