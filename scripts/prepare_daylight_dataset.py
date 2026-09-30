#!/usr/bin/env python3
"""Prepare an annotation-ready daytime dataset from the two PoC recordings.

This deliberately creates COCO files with zero annotations.  Bounding boxes
and the seven Dishub classes must be reviewed by a human before fine-tuning;
predictions from the current model may be imported later only as suggestions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


SOURCES = {
    "benhil2": Path("/app/data/samples/_batch_benhil2.mp4"),
    "gerbangpemuda": Path("/app/data/samples/_batch_gerbangpemuda.mp4"),
}

CATEGORIES = [
    (1, "sepeda_motor"),
    (2, "mobil_penumpang"),
    (3, "kendaraan_sedang"),
    (4, "bus_besar"),
    (5, "truk_berat"),
    (6, "pejalan_kaki"),
    (7, "sepeda"),
]


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def dimensions(path: Path) -> tuple[int, int]:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(path),
    ], text=True)
    stream = json.loads(raw)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def main() -> int:
    parser = argparse.ArgumentParser()
    # /app/runs is a persistent bind mount in the current CPU deployment.
    parser.add_argument("--output", default="/app/runs/training/daylight_v1")
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty dataset: {root}")

    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "annotations").mkdir(parents=True, exist_ok=True)

    records: dict[str, list[dict]] = {"train": [], "val": []}
    next_id = 1
    for camera, source in SOURCES.items():
        if not source.is_file():
            raise SystemExit(f"Missing source: {source}")
        scratch = root / ".extract" / camera
        scratch.mkdir(parents=True, exist_ok=True)
        run("ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vf", f"fps={args.fps},scale='min(1280,iw)':-2",
            "-q:v", "2", str(scratch / f"{camera}_%05d.jpg"))
        for index, frame in enumerate(sorted(scratch.glob("*.jpg")), start=1):
            # Deterministic 80/20 split, preserving both cameras in validation.
            split = "val" if index % 5 == 0 else "train"
            target = root / "images" / split / frame.name
            shutil.move(str(frame), target)
            width, height = dimensions(target)
            records[split].append({
                "id": next_id,
                "file_name": frame.name,
                "width": width,
                "height": height,
                "camera_key": camera,
                "source_video": source.name,
                "source_time_sec": round((index - 1) / args.fps, 3),
                "review_status": "UNLABELLED",
            })
            next_id += 1
    shutil.rmtree(root / ".extract", ignore_errors=True)

    categories = [{"id": key, "name": name, "supercategory": "road_object"}
                  for key, name in CATEGORIES]
    for split, images in records.items():
        payload = {
            "info": {
                "description": "PPVehicle PoC daytime bootstrap; annotations require human review",
                "version": "daylight_v1",
                "annotation_status": "UNLABELLED_DO_NOT_TRAIN",
                "sampling_fps": args.fps,
            },
            "licenses": [], "images": images, "annotations": [],
            "categories": categories,
        }
        (root / "annotations" / f"instances_{split}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "dataset": "daylight_v1",
        "status": "ANNOTATION_REQUIRED",
        "sources": {key: str(path) for key, path in SOURCES.items()},
        "sampling_fps": args.fps,
        "train_images": len(records["train"]),
        "val_images": len(records["val"]),
        "categories": [name for _, name in CATEGORIES],
        "next_step": "Review bounding boxes/classes, then change status to APPROVED_FOR_TRAINING",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
