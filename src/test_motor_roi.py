#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
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


def run(model: str, image, tag: str):
    detector = SDE_Detector(
        model_dir=model,
        tracker_config="/app/configs/tracker_hitung.yml",
        device="CPU", threshold=.18, cpu_threads=8, enable_mkldnn=True)
    labels = list(detector.pred_config.labels)
    detector.tracker = copy.deepcopy(detector.tracker)
    result = detector.predict_image(
        [image[..., ::-1].copy()], visual=False, seq_name=tag, frame_count=0)
    tlwhs, scores, ids = result[0]
    found = []
    keys = list(tlwhs.keys()) if hasattr(tlwhs, "keys") else range(len(tlwhs))
    for raw_cid in keys:
        cid = int(raw_cid)
        if not 0 <= cid < len(labels):
            continue
        for tlwh, score, tid in zip(tlwhs[cid], scores[cid], ids[cid]):
            if float(score) < .18:
                continue
            found.append({"label": labels[cid], "score": round(float(score), 3),
                          "tlwh": [round(float(v), 1) for v in tlwh[:4]]})
    return found


def main():
    source = cv2.imread("/app/runs/daylight_baseline/benhil2_poster.jpg")
    h, w = source.shape[:2]
    x1, y1, x2, y2 = (int(w * .29), int(h * .24), int(w * .88), int(h * .94))
    roi = source[y1:y2, x1:x2].copy()
    results = {
        "shape": [w, h], "roi": [x1, y1, x2, y2],
        "compact_full": run("/models/ppyoloe_coco", source, "compact-full"),
        "compact_roi": run("/models/ppyoloe_coco", roi, "compact-roi"),
        "accurate_roi": run("/models/ppyoloe_plus_l_coco", roi, "accurate-roi"),
    }
    out = Path("/app/runs/daylight_ab/motor_roi.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k not in {"shape", "roi"}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
