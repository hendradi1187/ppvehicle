"""Score every public camera by how well the detector actually sees it.

Picking demo cameras by eye does not scale to 46 feeds, and the thing that
decides whether a camera is usable is not the view but the number of pixels a
vehicle occupies in it. This runs the real detector over one snapshot per
camera and reports, per camera: how many vehicles it finds, how large they are
relative to the frame, and their mean confidence.

    python scripts/survey_cameras.py --list kamera.txt --out survey.json

A camera is good for a demo when it finds several vehicles AND their median
box covers a meaningful fraction of the frame. A camera that finds two tiny
boxes is a camera the detector is guessing on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

PIPELINE_DIR = os.environ.get(
    "LALIN_PIPELINE_DIR", "/app/vendor/PaddleDetection/deploy/pipeline")
DEPLOY_DIR = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, DEPLOY_DIR)

import numpy as np  # noqa: E402

for _n, _b in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _n):
        setattr(np, _n, _b)

import cv2  # noqa: E402
from python.infer import Detector  # noqa: E402

BASE = "https://dki-jkt.balitower.co.id:7028/"


def fetch_previews(cam_ids: list[str], dest: Path, timeout: int = 20) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    ok = {}
    for cam in cam_ids:
        path = dest / f"{cam}.jpg"
        if path.exists() and path.stat().st_size > 5000:
            ok[cam] = path
            continue
        try:
            with urllib.request.urlopen(BASE + cam + "/preview.jpg",
                                        timeout=timeout) as r:
                data = r.read()
            if len(data) > 5000:
                path.write_bytes(data)
                ok[cam] = path
        except Exception as exc:
            print(f"  lewat {cam[:44]}: {type(exc).__name__}", file=sys.stderr)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True, help="file berisi satu id kamera per baris")
    ap.add_argument("--model", default="/models/mot_ppyoloe_s_36e_ppvehicle")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--cache", default="/app/runs/_survey")
    ap.add_argument("--out", default="/app/runs/_survey/survey.json")
    args = ap.parse_args()

    cam_ids = [l.strip() for l in Path(args.list).read_text().splitlines() if l.strip()]
    print(f"kamera dalam daftar : {len(cam_ids)}")

    cache = Path(args.cache)
    shots = fetch_previews(cam_ids, cache / "shots")
    print(f"snapshot terambil   : {len(shots)}")

    det = Detector(model_dir=args.model, device="CPU", run_mode="paddle",
                   batch_size=1, threshold=args.threshold, output_dir=str(cache))

    rows = []
    for i, (cam, path) in enumerate(sorted(shots.items()), 1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        res = det.predict_image([str(path)], visual=False)
        boxes = res.get("boxes", np.empty((0, 6)))
        keep = boxes[boxes[:, 1] >= args.threshold] if len(boxes) else boxes
        areas = [((b[4] - b[2]) * (b[5] - b[3])) / (w * h) * 100 for b in keep]
        rows.append({
            "camera": cam,
            "region": cam.split("_")[1] if "_" in cam else "?",
            "n": len(keep),
            "area_median_pct": round(float(np.median(areas)), 3) if areas else 0.0,
            "area_max_pct": round(float(max(areas)), 3) if areas else 0.0,
            "score_mean": round(float(np.mean(keep[:, 1])), 3) if len(keep) else 0.0,
            "frame": f"{w}x{h}",
        })
        if i % 10 == 0:
            print(f"  ... {i}/{len(shots)}")

    # A usable demo camera needs both: several vehicles, and vehicles big
    # enough that the detector is not guessing. Multiplying the two punishes
    # "many tiny boxes" and "one big box" alike.
    for r in rows:
        r["skor"] = round(r["n"] * r["area_median_pct"], 2)
    rows.sort(key=lambda r: r["skor"], reverse=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print()
    print(f"{'skor':>7}  {'n':>3}  {'med%':>6}  {'maks%':>6}  {'conf':>5}  wil  kamera")
    for r in rows[:20]:
        print(f"{r['skor']:>7.2f}  {r['n']:>3}  {r['area_median_pct']:>6.2f}  "
              f"{r['area_max_pct']:>6.2f}  {r['score_mean']:>5.2f}  "
              f"{r['region']:<4} {r['camera'][:52]}")
    print(f"\nhasil lengkap -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
