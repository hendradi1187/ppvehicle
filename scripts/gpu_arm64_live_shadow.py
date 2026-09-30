"""Live shadow GPU Benhil 2 tanpa menyimpan frame atau mengubah layanan CPU."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

GARIS_640 = (264.0, 280.0, 525.0, 115.0)
KELAS = {"motorcycle": "sepeda_motor", "car": "mobil_penumpang", "bus": "bus_ambigu",
         "truck": "truk_ambigu", "person": "pejalan_kaki", "bicycle": "sepeda"}


def sisi(line: tuple[float, float, float, float], x: float, y: float) -> float:
    x1, y1, x2, y2 = line
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--interval", type=float, default=0.6)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia")
    model = YOLO(args.model)
    model.track(np.zeros((640, 640, 3), dtype=np.uint8), persist=True, device=0, imgsz=640, verbose=False)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError("stream tidak dapat dibuka")
    started, next_frame = time.monotonic(), 0.0
    previous: dict[int, float] = {}
    events: list[dict] = []
    n_frame = n_deteksi = 0
    while time.monotonic() - started < args.seconds:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        now_time = time.monotonic()
        if now_time < next_frame:
            continue
        next_frame = now_time + args.interval
        result = model.track(frame, persist=True, device=0, imgsz=640, conf=0.25,
                             tracker="bytetrack.yaml", verbose=False)[0]
        h, w = result.orig_shape
        line = (GARIS_640[0] * w / 640, GARIS_640[1] * h / 360,
                GARIS_640[2] * w / 640, GARIS_640[3] * h / 360)
        ids = result.boxes.id.tolist() if result.boxes.id is not None else [None] * len(result.boxes)
        for xyxy, cls, track_id in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), ids):
            label = result.names[int(cls)]
            if label not in KELAS or track_id is None:
                continue
            x1, _, x2, y2 = map(float, xyxy)
            track = int(track_id); current = sisi(line, (x1 + x2) / 2, y2)
            before = previous.get(track)
            if before is not None and before * current < 0:
                events.append({"track_id": track, "kelas_adaptor": KELAS[label],
                               "label_detektor": label,
                               "arah": "a_ke_b" if before < 0 else "b_ke_a"})
            previous[track] = current
            n_deteksi += 1
        n_frame += 1
    cap.release()
    payload = {"mode": "shadow_live_not_official", "runtime": "torch-cuda-arm64",
               "gpu": torch.cuda.get_device_name(0), "model": args.model, "garis_640": GARIS_640,
               "durasi_target_detik": args.seconds, "interval_target_detik": args.interval,
               "n_frame_dianalisis": n_frame, "n_deteksi_lalu_lintas": n_deteksi,
               "n_event": len(events), "per_kelas_event": dict(Counter(e["kelas_adaptor"] for e in events)),
               "event": events}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("n_frame_dianalisis", "n_deteksi_lalu_lintas", "n_event", "per_kelas_event")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
