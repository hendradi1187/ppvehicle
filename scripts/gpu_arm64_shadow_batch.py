"""Jalankan deteksi GPU ARM64 dalam mode bayangan pada frame server.

Keluaran hanya metadata bbox/kelas/skor. Tidak ada gambar, crop, pelat, atau
wajah yang ditulis ulang. Script ini tidak mengubah database maupun Live.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default="yolo11n.pt")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia")
    files = sorted(args.input.glob("*.jpg"))
    if not files:
        raise RuntimeError("tidak ada frame")
    model = YOLO(args.model)
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), device=0, imgsz=640, verbose=False)
    started = time.perf_counter()
    records: list[dict] = []
    for offset in range(0, len(files), args.batch):
        paths = files[offset:offset + args.batch]
        for path, result in zip(paths, model.predict([str(p) for p in paths], device=0, imgsz=640, verbose=False)):
            boxes = []
            for xyxy, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
                x1, y1, x2, y2 = map(float, xyxy)
                boxes.append({"label_detektor": result.names[int(cls)], "skor": round(float(score), 4),
                              "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]})
            records.append({"frame_server": path.name, "deteksi": boxes})
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    payload = {"runtime": "torch-cuda-arm64", "gpu": torch.cuda.get_device_name(0),
               "model": args.model, "jumlah_frame": len(files),
               "latensi_total_ms": elapsed, "latensi_rata2_ms": round(elapsed / len(files), 1),
               "frame": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("runtime", "gpu", "model", "jumlah_frame", "latensi_total_ms", "latensi_rata2_ms")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
