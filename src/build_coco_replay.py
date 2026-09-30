#!/usr/bin/env python3
"""Build a browser-safe classified replay from the raw CPU PoC capture."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

for name, builtin in (("int", int), ("float", float), ("bool", bool),
                      ("object", object), ("str", str)):
    if not hasattr(np, name):
        setattr(np, name, builtin)

pd_dir = Path("/app/vendor/PaddleDetection/deploy/pptracking/python")
sys.path.insert(0, str(pd_dir))
sys.path.insert(0, str(pd_dir.parent.parent))
from mot_sde_infer import SDE_Detector  # noqa: E402

SOURCES = {
    "benhil2": Path("/app/data/samples/_batch_benhil2.mp4"),
    "gerbangpemuda": Path("/app/data/samples/_batch_gerbangpemuda.mp4"),
}
ALLOWED = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
NAMES = {
    "person": "person", "bicycle": "bicycle", "car": "car",
    "motorcycle": "motorcycle", "bus": "bus", "truck": "truck",
}
COLORS = {
    "person": (35, 190, 90), "bicycle": (60, 200, 200),
    "car": (235, 115, 35), "motorcycle": (35, 55, 235),
    "bus": (30, 165, 245), "truck": (170, 80, 220),
}

# Kamera Benhil melihat pengendara dari atas. Pada jarak menengah COCO kerap
# hanya mempertahankan kotak `person` dan kehilangan kotak motornya. Titik kaki
# di dalam badan jalan utama adalah aturan kamera-spesifik yang deterministik;
# orang di trotoar kiri tidak masuk poligon ini.
ROAD_POLYGONS = {
    "benhil2": np.asarray([
        [.43, .22], [.50, .20], [.78, 1.00], [.47, 1.00]
    ], dtype=np.float32),
}


def iou(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def global_nms(items: list[dict], threshold: float = .58) -> list[dict]:
    kept = []
    for item in sorted(items, key=lambda x: x["score"], reverse=True):
        if all(iou(item["box"], old["box"]) < threshold for old in kept):
            kept.append(item)
    return kept


def detections(detector, labels, frame, frame_no: int, key: str) -> list[dict]:
    result = detector.predict_image(
        [frame[..., ::-1].copy()], visual=False,
        seq_name=f"classified-{key}", frame_count=frame_no)
    tlwhs, scores, ids = result[0]
    found = []
    keys = list(tlwhs.keys()) if hasattr(tlwhs, "keys") else range(len(tlwhs))
    for raw_cid in keys:
        cid = int(raw_cid)
        if not 0 <= cid < len(labels) or labels[cid] not in ALLOWED:
            continue
        label = labels[cid]
        for tlwh, score, tid in zip(tlwhs[cid], scores[cid], ids[cid]):
            score = float(score)
            if score < .25:
                continue
            found.append({"label": label, "score": score,
                          "track_id": int(tid),
                          "box": tuple(float(v) for v in tlwh[:4])})
    found = global_nms(found)
    polygon = ROAD_POLYGONS.get(key)
    if polygon is not None:
        h, w = frame.shape[:2]
        scaled = polygon * np.asarray([w, h], dtype=np.float32)
        for item in found:
            if item["label"] != "person":
                continue
            x, y, bw, bh = item["box"]
            foot = (float(x + bw / 2.0), float(y + bh))
            if cv2.pointPolygonTest(scaled, foot, False) >= 0:
                item["label"] = "motorcycle"
                item["method"] = "road_rider_rule_v1"
    # Reclassification can turn an overlapping person+rider pair into two
    # motorcycle boxes. Suppress again after the rule so one physical rider is
    # not drawn/counted twice.
    return global_nms(found, threshold=.38)


def draw(frame, found: list[dict], frame_no: int, fps: float):
    h, w = frame.shape[:2]
    scale = min(w / 1280.0, h / 720.0)
    line = max(2, round(3 * scale))
    font_scale = max(.55, .68 * scale)
    counts = Counter(item["label"] for item in found)
    for item in found:
        x, y, bw, bh = item["box"]
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w - 1, int(x + bw)), min(h - 1, int(y + bh))
        color = COLORS[item["label"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line)
        # Keep every accepted box, but avoid unreadable walls of labels for
        # tiny/distant low-confidence objects.
        if item["score"] < .38:
            continue
        text = f'{NAMES[item["label"]]} {item["score"]:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line)
        ty = max(th + 5, y1)
        cv2.rectangle(frame, (x1, ty - th - 5), (min(w - 1, x1 + tw + 6), ty + 3), color, -1)
        cv2.putText(frame, text, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), line, cv2.LINE_AA)
    title = (f'CPU classified replay  t={frame_no / max(fps, .1):.1f}s  ' +
             '  '.join(f'{k}:{counts.get(k, 0)}' for k in
                      ("motorcycle", "car", "bus", "truck", "person", "bicycle")))
    cv2.rectangle(frame, (0, 0), (w, max(34, round(45 * scale))), (8, 17, 30), -1)
    cv2.putText(frame, title, (12, max(25, round(30 * scale))),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 250, 255), line,
                cv2.LINE_AA)
    return counts


def build(detector, tracker_template, labels, key: str, source: Path, output: Path):
    detector.tracker = copy.deepcopy(tracker_template)
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 10.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_width = 1280
    ratio = min(1.0, max_width / max(width, 1))
    out_size = (int(width * ratio), int(height * ratio))
    temp = output / f"{key}_classified_mp4v.mp4"
    final = output / f"{key}_classified.mp4"
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    frame_no, infer_ms = 0, []
    max_counts = Counter()
    count_samples = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        found = detections(detector, labels, frame, frame_no, key)
        infer_ms.append((time.perf_counter() - t0) * 1000)
        counts = draw(frame, found, frame_no, fps)
        max_counts |= counts
        count_samples.append(dict(counts))
        if (width, height) != out_size:
            frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
        writer.write(frame)
        frame_no += 1
        if frame_no % 20 == 0:
            print(json.dumps({"key": key, "frame": frame_no,
                              "motorcycle_max": max_counts["motorcycle"]}), flush=True)
    cap.release()
    writer.release()
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(temp),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)], check=True)
    temp.unlink(missing_ok=True)
    medians = {}
    for label in ALLOWED:
        values = [sample.get(label, 0) for sample in count_samples]
        medians[label] = round(float(np.median(values)), 1) if values else 0
    result = {"key": key, "frames": frame_no, "fps": fps, "size": out_size,
              "model": "ppyoloe_coco", "threshold": .25,
              "counts_median": medians, "counts_max": dict(max_counts),
              "infer_ms_p50": round(float(np.percentile(infer_ms, 50)), 2),
              "infer_ms_p95": round(float(np.percentile(infer_ms, 95)), 2),
              "video": str(final)}
    (output / f"{key}_classified.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    output = Path("/app/runs/daylight_classified")
    output.mkdir(parents=True, exist_ok=True)
    detector = SDE_Detector(
        model_dir="/models/ppyoloe_coco",
        tracker_config="/app/configs/tracker_hitung.yml",
        device="CPU", threshold=.25, cpu_threads=8, enable_mkldnn=True)
    labels = list(detector.pred_config.labels)
    tracker_template = copy.deepcopy(detector.tracker)
    requested = os.environ.get("LALIN_KEYS", "benhil2,gerbangpemuda").split(",")
    for key in requested:
        key = key.strip()
        if key:
            build(detector, tracker_template, labels, key, SOURCES[key], output)


if __name__ == "__main__":
    main()
