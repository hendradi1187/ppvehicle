"""Buat kandidat pelabelan metadata-only dari frame Benhil 2 di server.

Gambar asli dan setiap crop tetap di server. Berkas keluaran hanya menyimpan
nama frame server, bbox, skor, dan label detektor supaya peninjauan dilakukan
di lingkungan terbatas.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

ROOT = Path("/app")
DATA = ROOT / "runs/dataset/mentah/benhil2"
OUT = ROOT / "runs/dataset/labeling/benhil2-kandidat-dataset.json"
MODEL = Path("/models/ppyoloe_plus_l_coco")
KELAS_MUNGKIN = {
    "motorcycle": ["Sepeda Motor"],
    "car": ["Mobil Penumpang"],
    "bus": ["Kendaraan Sedang", "Bus Besar"],
    "truck": ["Mobil Penumpang", "Kendaraan Sedang", "Truk Berat"],
    "person": ["Pejalan Kaki"],
    "bicycle": ["Sepeda"],
}
WIB = timezone(timedelta(hours=7))


def waktu_wib(path: Path) -> str:
    parsed = datetime.strptime(path.stem, "%Y%m%d-%H%M%S").replace(tzinfo=WIB)
    return parsed.strftime("%Y-%m-%d %H:%M:%S WIB")


def tersebar(files: list[Path], jumlah: int) -> list[Path]:
    if not files:
        return []
    if len(files) <= jumlah:
        return files
    return [files[round(i * (len(files) - 1) / (jumlah - 1))] for i in range(jumlah)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=120, help="jumlah frame tersebar waktu")
    ap.add_argument("--per-label", type=int, default=8)
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    if args.frame < 1 or args.per_label < 1:
        raise ValueError("--frame dan --per-label harus minimal 1")

    files = sorted(DATA.glob("*.jpg"))
    if not files:
        raise RuntimeError(f"tidak ada frame di {DATA}")
    sampled = tersebar(files, args.frame)

    sys.path.insert(0, str(ROOT / "vendor/PaddleDetection/deploy/python"))
    from infer import Detector

    # Seluruh inferensi dibuat kecil dan satu thread agar tidak mengganggu Live.
    os.nice(19)
    selected: dict[str, list[dict]] = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix="benhil2-label-", dir="/tmp") as td:
        tmp = Path(td)
        paths: list[Path] = []
        source: list[Path] = []
        dimensions: list[tuple[int, int]] = []
        for idx, frame in enumerate(sampled):
            image = cv2.imread(str(frame))
            if image is None:
                continue
            h, w = image.shape[:2]
            small = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
            target = tmp / f"{idx:04d}.jpg"
            cv2.imwrite(str(target), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
            paths.append(target); source.append(frame); dimensions.append((w, h))

        detector = Detector(model_dir=str(MODEL), device="CPU", threshold=0.30,
                            cpu_threads=1, enable_mkldnn=True,
                            output_dir=str(tmp / "detector-output"))
        result = detector.predict_image([str(p) for p in paths], visual=False)
        labels = list(detector.pred_config.labels)
        offset = 0
        for index, count in enumerate(result["boxes_num"]):
            width, height = dimensions[index]
            for box in result["boxes"][offset:offset + int(count)].tolist():
                class_id, score, x1, y1, x2, y2 = box[:6]
                label = labels[int(class_id)] if 0 <= int(class_id) < len(labels) else ""
                if label not in KELAS_MUNGKIN or len(selected[label]) >= args.per_label:
                    continue
                scale_x, scale_y = width / 640.0, height / 360.0
                bbox = [x1 * scale_x, y1 * scale_y, (x2 - x1) * scale_x, (y2 - y1) * scale_y]
                selected[label].append({
                    "frame_server": source[index].name,
                    "waktu_server": waktu_wib(source[index]),
                    "malam": not (6 <= int(source[index].stem[9:11]) < 18),
                    "label_detektor": label,
                    "skor_detektor": round(float(score), 4),
                    "bbox": [round(float(value), 1) for value in bbox],
                    "kelas_mungkin": KELAS_MUNGKIN[label],
                    "label_acuan": None,
                    "kepastian": None,
                    "catatan_reviewer": None,
                })
            offset += int(count)

    candidates = [item for label in KELAS_MUNGKIN for item in selected[label]]
    payload = {
        "kamera": "benhil2",
        "dibuat_pada": datetime.now(timezone.utc).isoformat(),
        "privasi": "Metadata-only. Frame HD dan crop tetap di server terbatas.",
        "petunjuk": "Tinjau frame hanya di server dan isi label_acuan dari tujuh kelas Dishub.",
        "frame_diperiksa": len(sampled),
        "jumlah": len(candidates),
        "per_label_detektor": {label: len(selected[label]) for label in KELAS_MUNGKIN},
        "sampel": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(args.output.parent, 0o700)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(args.output, 0o600)
    print(json.dumps({"output": str(args.output), "jumlah": len(candidates),
                      "per_label_detektor": payload["per_label_detektor"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
