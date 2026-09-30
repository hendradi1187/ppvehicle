#!/usr/bin/env python3
"""Build deterministic replay metrics from the compact CPU detector/tracker."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

for name, builtin in (("int", int), ("float", float), ("bool", bool),
                      ("object", object), ("str", str)):
    if not hasattr(np, name):
        setattr(np, name, builtin)

PD_DIR = Path("/app/vendor/PaddleDetection/deploy/pptracking/python")
sys.path.insert(0, str(PD_DIR))
sys.path.insert(0, str(PD_DIR.parent.parent))
from mot_sde_infer import SDE_Detector  # noqa: E402


SOURCES = {
    "benhil2": Path("/app/data/samples/_batch_benhil2.mp4"),
    "gerbangpemuda": Path("/app/data/samples/_batch_gerbangpemuda.mp4"),
}
PIPELINE_UNIQUE_TRACKS = {"benhil2": 37, "gerbangpemuda": 28}
NAMA = {
    "motorcycle": "Sepeda Motor",
    "car": "Mobil Penumpang",
    "bus": "Bus (belum terpilah)",
    "truck": "Truk (belum terpilah)",
    "person": "Pejalan Kaki",
    "bicycle": "Sepeda",
}
KENDARAAN = {"motorcycle", "car", "bus", "truck", "bicycle"}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values), p)), 2)


def analyze(detector, tracker_template, labels: list[str], key: str, source: Path,
            model_name: str, frame_width: int, frame_height: int) -> dict:
    detector.tracker = copy.deepcopy(tracker_template)
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    sample_every = max(1, round(fps / 5.0))
    frame_index = 0
    analyzed = 0
    times: list[float] = []
    per_frame: dict[str, list[int]] = {label: [] for label in NAMA}
    unique: dict[str, set[int]] = {label: set() for label in NAMA}
    previous: dict[tuple[int, int], tuple[float, float, float]] = {}
    speeds: list[float] = []
    occupancies: list[float] = []
    visible_vehicle: list[int] = []
    stopped: list[float] = []

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame_index += 1
        if (frame_index - 1) % sample_every:
            continue
        bgr = cv2.resize(bgr, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        result = detector.predict_image([bgr[..., ::-1].copy()], visual=False,
                                        seq_name=f"daylight-{key}", frame_count=analyzed)
        times.append((time.perf_counter() - t0) * 1000)
        tlwhs, scores, ids = result[0]
        keys = list(tlwhs.keys()) if hasattr(tlwhs, "keys") else range(len(tlwhs))
        now = frame_index / fps
        frame_counts = defaultdict(int)
        frame_area = 0.0
        frame_speeds: list[float] = []
        for cid_raw in keys:
            cid = int(cid_raw)
            if not 0 <= cid < len(labels) or labels[cid] not in NAMA:
                continue
            label = labels[cid]
            for tlwh, score, tid_raw in zip(tlwhs[cid], scores[cid], ids[cid]):
                if float(score) < .25:
                    continue
                tid = int(tid_raw)
                if tid < 0:
                    continue
                x, y, w, h = (float(v) for v in tlwh[:4])
                frame_counts[label] += 1
                unique[label].add(tid)
                if label in KENDARAAN:
                    frame_area += max(0.0, w * h)
                    track_key = (cid, tid)
                    point = (x + w / 2.0, y + h)
                    old = previous.get(track_key)
                    if old and now > old[2]:
                        speed = float(np.hypot(point[0] - old[0], point[1] - old[1])
                                      / (now - old[2]))
                        frame_speeds.append(speed)
                        speeds.append(speed)
                    previous[track_key] = (point[0], point[1], now)
        for label in NAMA:
            per_frame[label].append(frame_counts[label])
        n_vehicle = sum(frame_counts[label] for label in KENDARAAN)
        visible_vehicle.append(n_vehicle)
        occupancies.append(min(1.0, frame_area / float(frame_width * frame_height)))
        if frame_speeds:
            stopped.append(sum(v < 5.0 for v in frame_speeds) / len(frame_speeds))
        analyzed += 1
    cap.release()

    occ = float(np.median(occupancies)) if occupancies else 0.0
    speed = float(np.median(speeds)) if speeds else 0.0
    stop = float(np.median(stopped)) if stopped else 0.0
    visible = int(round(float(np.median(visible_vehicle)))) if visible_vehicle else 0
    quality = min(1.0, analyzed / 20.0) * min(1.0, len(speeds) / 20.0) \
        * min(1.0, visible / 5.0)
    if analyzed < 10 or quality < .5:
        traffic, reason = "UNKNOWN", "observasi gerak belum cukup"
    elif occ >= .18 and stop >= .55 and speed < 8:
        traffic, reason = "MACET", "occupancy tinggi dan mayoritas berhenti"
    elif occ >= .14 or (visible >= 8 and speed < 15):
        traffic, reason = "PADAT", "occupancy tinggi atau gerak lambat"
    elif occ >= .07 or visible >= 5:
        traffic, reason = "RAMAI", "volume kendaraan meningkat"
    else:
        traffic, reason = "LANCAR", "occupancy rendah"

    classes = []
    for label, name in NAMA.items():
        values = per_frame[label]
        classes.append({
            "detector_label": label,
            "name": name,
            "visible_median": round(float(np.median(values)), 1) if values else 0,
            "visible_max": max(values, default=0),
            "unique_tracks": len(unique[label]),
        })
    return {
        "key": key, "source": source.name, "status": "BASELINE_NOT_UAT",
        "model": model_name, "threshold": .25,
        "analysis_resolution": [frame_width, frame_height],
        "source_frames": frame_index, "analyzed_frames": analyzed,
        "source_fps": fps, "sampling_fps": round(fps / sample_every, 2),
        "unique_tracks": sum(len(v) for k, v in unique.items() if k in KENDARAAN),
        "counting": {"unique_tracks": PIPELINE_UNIQUE_TRACKS.get(key),
                     "crossing": None, "status": "TRACKS_NOT_CROSSING"},
        "classes": classes,
        "inference_ms": {"p50": percentile(times, 50), "p95": percentile(times, 95)},
        "traffic": {
            "state": traffic, "status": "PROVISIONAL", "reason": reason,
            "quality_score": round(quality, 3), "visible_median": visible,
            "occupancy_ratio": round(occ, 4),
            "median_speed_px_s": round(speed, 2), "stopped_ratio": round(stop, 4),
            "speed_observations": len(speeds),
        },
        "limitations": [
            "COCO tidak memisahkan kendaraan_sedang dan bus_besar secara deterministik",
            "unique_tracks bukan crossing garis",
            "traffic rule belum UAT per sudut kamera",
        ],
    }


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    model_dir = os.environ.get("LALIN_AB_MODEL", "/models/ppyoloe_coco")
    model_name = Path(model_dir).name
    frame_width = int(os.environ.get("LALIN_AB_WIDTH", "640"))
    frame_height = int(os.environ.get("LALIN_AB_HEIGHT", "360"))
    detector = SDE_Detector(
        model_dir=model_dir,
        tracker_config="/app/configs/tracker_hitung.yml",
        device="CPU", threshold=.25, cpu_threads=8, enable_mkldnn=True)
    labels = list(detector.pred_config.labels)
    template = copy.deepcopy(detector.tracker)
    output = Path(os.environ.get("LALIN_AB_OUTPUT", "/app/runs/daylight_baseline"))
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key, source in SOURCES.items():
        metrics = analyze(detector, template, labels, key, source,
                          model_name, frame_width, frame_height)
        (output / f"{key}_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[key] = metrics
        print(json.dumps({"key": key, "frames": metrics["analyzed_frames"],
                          "tracks": metrics["unique_tracks"],
                          "traffic": metrics["traffic"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
