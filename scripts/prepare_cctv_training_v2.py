"""Sample local CCTV videos into a source-disjoint COCO annotation workspace.

Example:
  python scripts/prepare_cctv_training_v2.py \
    --samples data/samples --output runs/training/daylight_v2 --fps 1

Frames from one source video are kept in one split to prevent temporal leakage.
The generated COCO files intentionally contain no annotations until a reviewer
has completed manual bbox annotation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

CATEGORIES = [
    {"id": 1, "name": "sepeda_motor"},
    {"id": 2, "name": "mobil_penumpang"},
    {"id": 3, "name": "kendaraan_sedang"},
    {"id": 4, "name": "bus_besar"},
    {"id": 5, "name": "truk_berat"},
    {"id": 6, "name": "pejalan_kaki"},
    {"id": 7, "name": "sepeda"},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--val-mod", type=int, default=5)
    args = ap.parse_args()
    if args.fps <= 0 or args.val_mod < 2:
        ap.error("fps harus > 0 dan val-mod >= 2")

    out = args.output.resolve()
    all_dir = out / "images" / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted((*args.samples.glob("_batch_*.mp4"), *args.samples.glob("_adhoc_*.mp4")))
    manifest: list[dict] = []
    for video in videos:
        split = "val" if int(hashlib.sha1(video.name.encode()).hexdigest()[:2], 16) % args.val_mod == 0 else "train"
        prefix = video.stem.replace(" ", "_") + "__"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
            "-vf", f"fps={args.fps:g},scale=1280:-2", "-q:v", "2",
            str(all_dir / f"{prefix}%05d.jpg"),
        ], check=True)
        for index, image in enumerate(sorted(all_dir.glob(f"{prefix}*.jpg"))):
            manifest.append({
                "file_name": str(image.relative_to(out)).replace("\\", "/"),
                "source": video.name,
                "source_seconds": round(index / args.fps, 2),
                "split": split,
            })

    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8"
    )
    annotations = out / "annotations"
    annotations.mkdir(exist_ok=True)
    for split in ("train", "val"):
        split_dir = out / "images" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        rows = [row for row in manifest if row["split"] == split]
        images = []
        for image_id, row in enumerate(rows, 1):
            source = out / row["file_name"]
            target = split_dir / source.name
            if not target.exists():
                os.link(source, target)
            images.append({
                "id": image_id,
                "file_name": f"images/{split}/{source.name}",
                "width": 1280,
                "height": 720,
                "source_video": row["source"],
                "source_seconds": row["source_seconds"],
                "review_status": "PENDING",
            })
        payload = {
            "info": {"description": "PPVehicle local CCTV v2 pending manual annotation"},
            "licenses": [], "images": images, "annotations": [],
            "categories": CATEGORIES, "annotation_status": "PENDING_MANUAL_ANNOTATION",
        }
        (annotations / f"instances_{split}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(split, len(images))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
