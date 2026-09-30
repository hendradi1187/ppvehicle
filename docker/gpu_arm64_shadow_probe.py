"""Buktikan jalur deteksi CUDA ARM64 tanpa mengakses data CCTV.

Model COCO digunakan hanya untuk mengukur kompatibilitas runtime. Hasilnya
tidak mengubah keputusan Live, tracker, garis hitung, database, atau UI.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from ultralytics import YOLO


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia di container")
    device = torch.cuda.get_device_name(0)
    model = YOLO("yolo11n.pt")
    # Citra sintetis: memastikan inferensi GPU tanpa membawa frame CCTV.
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    started = time.perf_counter()
    result = model.predict(image, device=0, imgsz=640, verbose=False)
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    print(json.dumps({
        "runtime": "torch-cuda-arm64",
        "gpu": device,
        "cuda": torch.version.cuda,
        "model": "yolo11n.pt",
        "latensi_ms_sintetis": elapsed,
        "jumlah_deteksi_sintetis": len(result[0].boxes),
    }))


if __name__ == "__main__":
    main()
