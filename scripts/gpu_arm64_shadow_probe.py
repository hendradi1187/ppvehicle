"""Buktikan jalur deteksi CUDA ARM64 tanpa mengakses data CCTV.

Model COCO digunakan hanya untuk mengukur kompatibilitas runtime. Hasilnya
tidak mengubah keputusan Live, tracker, garis hitung, database, atau UI.
"""

from __future__ import annotations

import json
import time
import argparse

import numpy as np
import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None,
                        help="path frame server yang dipasang read-only")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia di container")
    device = torch.cuda.get_device_name(0)
    model = YOLO("yolo11n.pt")
    # Warmup terpisah agar latensi berikutnya mengukur inferensi, bukan
    # inisialisasi CUDA atau pemuatan kernel pertama.
    model.predict(np.zeros((640, 640, 3), dtype=np.uint8), device=0, imgsz=640, verbose=False)
    # Citra sintetis tetap menjadi fallback: tidak ada frame CCTV dipakai
    # sebelum caller sengaja memasang satu path read-only.
    image = args.image or np.zeros((640, 640, 3), dtype=np.uint8)
    started = time.perf_counter()
    result = model.predict(image, device=0, imgsz=640, verbose=False)
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    print(json.dumps({
        "runtime": "torch-cuda-arm64",
        "gpu": device,
        "cuda": torch.version.cuda,
        "model": "yolo11n.pt",
        "latensi_inferensi_ms": elapsed,
        "mode": "frame_server" if args.image else "sintetis",
        "jumlah_deteksi": len(result[0].boxes),
        "kelas_coco": sorted({result[0].names[int(i)] for i in result[0].boxes.cls.tolist()}),
    }))


if __name__ == "__main__":
    main()
