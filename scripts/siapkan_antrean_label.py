"""Buat antrean label metadata-only dari lintasan Live yang sudah berjejak.

Jalankan di server/container setelah Live Benhil 2 mengumpulkan beberapa
lintasan. Keluaran tidak memuat gambar, pelat, atau wajah. Peninjau membuka
frame yang relevan hanya dari server terbatas, memakai session, seq, PTS, dan
bbox/jejak sebagai pengait.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lalin import store
from lalin.settings import settings


KELAS_MUNGKIN = {
    "motorcycle": ["Sepeda Motor"],
    "car": ["Mobil Penumpang"],
    "bus": ["Kendaraan Sedang", "Bus Besar"],
    "truck": ["Mobil Penumpang", "Kendaraan Sedang", "Truk Berat"],
    "person": ["Pejalan Kaki"],
    "bicycle": ["Sepeda"],
}


def waktu_wib(value) -> str:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M:%S WIB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kamera", default="benhil2")
    ap.add_argument("--per-label", type=int, default=8)
    ap.add_argument("--output", type=Path,
                    default=settings.runs_dir / "dataset/labeling/benhil2-antrean-label.json")
    args = ap.parse_args()
    if args.per_label < 1:
        raise ValueError("--per-label harus minimal 1")

    # Data baru saja: jejak tidak null membuktikan pencatatan track sudah aktif.
    sql = """
        SELECT sesi_id, seq, camera, waktu, malam, pts, track_id, label_detektor,
               status, n_lihat, n_yakin, skor_maks, skor_rata, bbox, jejak
          FROM lintas_kendaraan
         WHERE camera = ? AND jejak IS NOT NULL
           AND label_detektor IN ('motorcycle', 'car', 'bus', 'truck', 'person', 'bicycle')
         ORDER BY waktu DESC
    """
    with store._connect() as conn:
        rows = [dict(r) for r in conn.execute(sql, (args.kamera,)).fetchall()]

    # Ambil bergiliran per label agar satu jenis yang sangat ramai tidak
    # menutup peluang kelas lain. Setiap sampel tetap membutuhkan keputusan manusia.
    per_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        label = row["label_detektor"]
        if len(per_label[label]) >= args.per_label:
            continue
        jejak = row["jejak"]
        if isinstance(jejak, str):
            jejak = json.loads(jejak)
        if not isinstance(jejak, list) or not jejak:
            continue
        per_label[label].append({
            "sesi_id": row["sesi_id"], "seq": row["seq"], "kamera": row["camera"],
            "waktu_server": waktu_wib(row["waktu"]),
            "malam": bool(row["malam"]) if row["malam"] is not None else None,
            "pts_lintas": row["pts"],
            "track_id": row["track_id"], "label_detektor": label,
            "kelas_mungkin": KELAS_MUNGKIN[label], "status_internal": row["status"],
            "n_lihat": row["n_lihat"], "n_yakin": row["n_yakin"],
            "skor_maks": row["skor_maks"], "skor_rata": row["skor_rata"],
            "bbox_lintas": json.loads(row["bbox"]) if isinstance(row["bbox"], str) else row["bbox"],
            "jejak": jejak,
            "label_acuan": None, "kepastian": None, "catatan_reviewer": None,
        })

    antrean = [item for label in KELAS_MUNGKIN for item in per_label[label]]
    result = {
        "kamera": args.kamera,
        "dibuat_pada": datetime.now(timezone.utc).isoformat(),
        "privasi": "Metadata-only. Frame HD dan crop tetap di server terbatas.",
        "petunjuk": "Pilih label acuan dari bukti visual server. Jangan menyalin gambar ke antrean.",
        "jumlah": len(antrean),
        "per_label_detektor": {label: len(per_label[label]) for label in KELAS_MUNGKIN},
        "sampel": antrean,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output.parent, 0o700)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(args.output, 0o600)
    print(json.dumps({"output": str(args.output), "jumlah": len(antrean),
                      "per_label_detektor": result["per_label_detektor"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
