#!/usr/bin/env python3
"""Generate review-only detector suggestions for the daylight COCO dataset.

Run inside the ppvehicle container. Output is deliberately not an evaluation
prediction set until a reviewer confirms every box and resolves ambiguous
COCO ``bus``/``truck`` labels into the Dishub taxonomy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mot_sde_infer import SDE_Detector


RAW_TO_DISHUB = {
    "motorcycle": 1,
    "car": 2,
    "person": 6,
    "bicycle": 7,
}
ALLOWED = set(RAW_TO_DISHUB) | {"bus", "truck"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/app/runs/training/daylight_v1"))
    parser.add_argument("--output", type=Path,
                        default=Path("/app/runs/training/daylight_v1/predictions_suggestions.json"))
    parser.add_argument("--threshold", type=float, default=0.20)
    args = parser.parse_args()
    records = []
    for split in ("train", "val"):
        ann_path = args.dataset / "annotations" / f"instances_{split}.json"
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        image_root = args.dataset / "images" / split
        records.extend((split, image_root / image["file_name"], image) for image in payload["images"])

    detector = SDE_Detector(model_dir="/models/ppyoloe_plus_l_coco",
                            tracker_config="/app/configs/tracker_hitung.yml",
                            device="CPU", threshold=args.threshold,
                            cpu_threads=2, enable_mkldnn=True)
    labels = list(detector.pred_config.labels)
    predictions = []
    for split, path, image in records:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"cannot read {path}")
        result = detector.predict_image([frame[..., ::-1].copy()], visual=False)
        boxes, scores, _ = result[0]
        keys = list(boxes.keys()) if hasattr(boxes, "keys") else range(len(boxes))
        for raw_cid in keys:
            cid = int(raw_cid)
            if not 0 <= cid < len(labels) or labels[cid] not in ALLOWED:
                continue
            label = labels[cid]
            for box, score in zip(boxes[cid], scores[cid]):
                score = float(score)
                if score < args.threshold:
                    continue
                x, y, w, h = [float(value) for value in box[:4]]
                x1, y1 = max(0.0, x), max(0.0, y)
                x2 = min(float(image["width"]), x + w)
                y2 = min(float(image["height"]), y + h)
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append({
                    "image_id": int(image["id"]),
                    "file_name": image["file_name"],
                    "split": split,
                    "detector_label": label,
                    "category_id": RAW_TO_DISHUB.get(label),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(score, 4),
                    "review_status": "SUGGESTION_NOT_GROUND_TRUTH",
                    "review_note": "bus/truck require human Dishub taxonomy mapping" if label in {"bus", "truck"} else "confirm box and class",
                })
    output = {"status": "SUGGESTIONS_ONLY_NOT_GROUND_TRUTH", "model": "ppyoloe_plus_l_coco",
              "threshold": args.threshold, "categories": [
                  {"id": 1, "name": "sepeda_motor"}, {"id": 2, "name": "mobil_penumpang"},
                  {"id": 3, "name": "kendaraan_sedang"}, {"id": 4, "name": "bus_besar"},
                  {"id": 5, "name": "truk_berat"}, {"id": 6, "name": "pejalan_kaki"},
                  {"id": 7, "name": "sepeda"}], "predictions": predictions}
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "predictions": len(predictions)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
