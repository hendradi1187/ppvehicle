#!/usr/bin/env python3
"""Evaluate vehicle semantics against reviewed COCO ground truth.

The prediction file must contain COCO-style records with ``image_id``,
``category_id``, ``bbox`` (xywh), and ``score``.  This evaluator intentionally
does not infer labels from the detector: predictions with ``category_id`` null
are ignored and the report states that they were unresolved.  That keeps
model suggestions separate from human ground truth.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def load_predictions(path: Path, threshold: float, image_ids: set[int] | None = None) -> tuple[list[dict], int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("predictions", raw.get("annotations", []))
    out = []
    unresolved = 0
    ignored_out_of_split = 0
    for item in raw:
        if image_ids is not None and int(item.get("image_id", -1)) not in image_ids:
            ignored_out_of_split += 1
            continue
        score = float(item.get("score", 1.0))
        category = item.get("category_id")
        if category is None:
            unresolved += 1
            continue
        if score >= threshold:
            out.append({**item, "category_id": int(category), "score": score,
                        "bbox": [float(x) for x in item["bbox"]]})
    return out, unresolved, ignored_out_of_split


def evaluate(gt: dict, predictions: list[dict], iou_threshold: float) -> dict:
    by_image_class: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for pred in predictions:
        by_image_class[(int(pred["image_id"]), int(pred["category_id"]))].append(pred)
    for values in by_image_class.values():
        values.sort(key=lambda x: x["score"], reverse=True)

    gt_by_image_class: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ann in gt.get("annotations", []):
        gt_by_image_class[(int(ann["image_id"]), int(ann["category_id"]))].append(ann)

    category_names = {int(x["id"]): x["name"] for x in gt.get("categories", [])}
    classes = sorted(set(category_names) | {int(x["category_id"]) for x in predictions})
    rows = []
    total_tp = total_fp = total_fn = 0
    for category in classes:
        tp = fp = fn = 0
        image_ids = {key[0] for key in gt_by_image_class if key[1] == category}
        image_ids |= {key[0] for key in by_image_class if key[1] == category}
        for image_id in image_ids:
            truth = gt_by_image_class[(image_id, category)]
            preds = by_image_class[(image_id, category)]
            used = set()
            for pred in preds:
                matches = [
                    (iou(pred["bbox"], ann["bbox"]), index)
                    for index, ann in enumerate(truth) if index not in used
                ]
                if matches:
                    best_iou, best_index = max(matches)
                else:
                    best_iou, best_index = 0.0, None
                if best_index is not None and best_iou >= iou_threshold:
                    used.add(best_index)
                    tp += 1
                else:
                    fp += 1
            fn += len(truth) - len(used)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"category_id": category, "class": category_names.get(category, str(category)),
                     "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(precision, 4), "recall": round(recall, 4),
                     "f1": round(f1, 4)})
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    return {"iou_threshold": iou_threshold, "classes": rows,
            "micro": {"tp": total_tp, "fp": total_fp, "fn": total_fn,
                      "precision": round(precision, 4), "recall": round(recall, 4)}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    gt = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    if not gt.get("annotations"):
        raise SystemExit("GROUND_TRUTH_REQUIRED: annotations is empty")
    gt_image_ids = {int(x["id"]) for x in gt.get("images", [])}
    predictions, unresolved, ignored_out_of_split = load_predictions(
        args.predictions, args.score_threshold, gt_image_ids)
    report = evaluate(gt, predictions, args.iou_threshold)
    report.update({"ground_truth": str(args.ground_truth), "predictions": str(args.predictions),
                   "score_threshold": args.score_threshold, "evaluated_predictions": len(predictions),
                   "unresolved_suggestions": unresolved,
                   "ignored_predictions_out_of_split": ignored_out_of_split,
                   "status": "VALIDATED_AGAINST_GROUND_TRUTH"})
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
