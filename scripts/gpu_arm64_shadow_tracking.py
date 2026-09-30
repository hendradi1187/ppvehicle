"""Tracking dan garis hitung GPU ARM64 dalam mode bayangan Benhil 2.

Input merupakan keyframe dataset. Karena intervalnya sekitar lima detik,
event yang dihasilkan bersifat diagnostik, bukan hitungan resmi Live. Keluaran
hanya menyimpan metadata track/bbox/event; frame selalu dibaca read-only.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

GARIS_640 = (264.0, 280.0, 525.0, 115.0)  # sesi CPU Benhil 2 terbaru
KELAS_LALIN = {
    "motorcycle": "sepeda_motor", "car": "mobil_penumpang",
    "bus": "bus_ambigu", "truck": "truk_ambigu",
    "person": "pejalan_kaki", "bicycle": "sepeda",
}


def timestamp(path: Path) -> datetime:
    return datetime.strptime(path.stem, "%Y%m%d-%H%M%S")


def sisi(garis: tuple[float, float, float, float], x: float, y: float) -> float:
    x1, y1, x2, y2 = garis
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia")
    files = sorted(args.input.glob("*.jpg"))
    if len(files) < 2:
        raise RuntimeError("minimal dua frame diperlukan")
    intervals = [(timestamp(b) - timestamp(a)).total_seconds() for a, b in zip(files, files[1:])]
    model = YOLO(args.model)
    model.track(np.zeros((640, 640, 3), dtype=np.uint8), persist=True, device=0, imgsz=640, verbose=False)
    previous: dict[int, float] = {}
    events: list[dict] = []
    frames: list[dict] = []
    started = time.perf_counter()
    for path in files:
        result = model.track(str(path), persist=True, device=0, imgsz=640, conf=0.25,
                             tracker="bytetrack.yaml", verbose=False)[0]
        h, w = result.orig_shape
        line = (GARIS_640[0] * w / 640, GARIS_640[1] * h / 360,
                GARIS_640[2] * w / 640, GARIS_640[3] * h / 360)
        boxes = []
        ids = result.boxes.id.tolist() if result.boxes.id is not None else [None] * len(result.boxes)
        for xyxy, score, cls, track_id in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist(), ids):
            label = result.names[int(cls)]
            if label not in KELAS_LALIN or track_id is None:
                continue
            x1, y1, x2, y2 = map(float, xyxy)
            roda = ((x1 + x2) / 2, y2)
            track = int(track_id)
            now = sisi(line, *roda)
            before = previous.get(track)
            if before is not None and before * now < 0:
                events.append({"frame_server": path.name, "track_id": track,
                               "kelas_adaptor": KELAS_LALIN[label], "label_detektor": label,
                               "arah": "a_ke_b" if before < 0 else "b_ke_a"})
            previous[track] = now
            boxes.append({"track_id": track, "label_detektor": label,
                          "kelas_adaptor": KELAS_LALIN[label], "skor": round(float(score), 4),
                          "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]})
        frames.append({"frame_server": path.name, "deteksi": boxes})
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    payload = {"mode": "shadow_keyframe_not_official", "kamera": "benhil2",
               "runtime": "torch-cuda-arm64", "gpu": torch.cuda.get_device_name(0),
               "model": args.model, "garis_640": GARIS_640,
               "jumlah_frame": len(files), "interval_median_detik": float(np.median(intervals)),
               "latensi_rata2_ms": round(elapsed / len(files), 1),
               "peringatan": "Keyframe sekitar lima detik tidak cukup rapat untuk memvalidasi track atau hitungan garis Live.",
               "jumlah_event_diagnostik": len(events), "per_kelas_event": dict(Counter(e["kelas_adaptor"] for e in events)),
               "event": events, "frame": frames}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("mode", "jumlah_frame", "interval_median_detik", "latensi_rata2_ms", "jumlah_event_diagnostik", "per_kelas_event")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
