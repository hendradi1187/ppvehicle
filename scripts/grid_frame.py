"""Overlay a coordinate grid on a video frame, for picking zone polygons
where there is no display (Docker, SSH) and pick_zone.py cannot open a window.

    python scripts/grid_frame.py data/samples/jatibaru.mp4 -o runs/grid.png
    python scripts/grid_frame.py data/samples/jatibaru.mp4 --frame 40 --step 40

Read the coordinates off the grid, then pass them straight through:

    python -m lalin run --scenario illegal_parking \
        --source data/samples/jatibaru.mp4 --region-polygon 8 24 132 18 ...

Coordinates are in the video's native pixels, which is what --region_polygon
expects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="video or image to sample")
    ap.add_argument("--frame", type=int, default=0, help="frame index (default 0)")
    ap.add_argument("--step", type=int, default=50, help="grid spacing in px")
    ap.add_argument("-o", "--out", default="runs/grid.png", help="output PNG")
    args = ap.parse_args()

    src = Path(args.video)
    if not src.exists():
        print(f"not found: {src}")
        return 2

    cap = cv2.VideoCapture(str(src))
    if args.frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"could not read frame {args.frame} from {src}")
        return 2

    h, w = frame.shape[:2]
    # Scale up so the labels stay readable on a 640x360 CCTV frame.
    scale = 2 if w <= 800 else 1
    if scale > 1:
        frame = cv2.resize(frame, (w * scale, h * scale),
                           interpolation=cv2.INTER_NEAREST)

    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.38
    for x in range(0, w + 1, args.step):
        px = x * scale
        major = x % (args.step * 2) == 0
        cv2.line(frame, (px, 0), (px, h * scale),
                 (0, 255, 255) if major else (90, 90, 90), 1)
        if major:
            cv2.putText(frame, str(x), (px + 3, 14), font, fs, (0, 0, 0), 3)
            cv2.putText(frame, str(x), (px + 3, 14), font, fs, (0, 255, 255), 1)
    for y in range(0, h + 1, args.step):
        py = y * scale
        major = y % (args.step * 2) == 0
        cv2.line(frame, (0, py), (w * scale, py),
                 (0, 255, 255) if major else (90, 90, 90), 1)
        if major:
            cv2.putText(frame, str(y), (3, py - 4), font, fs, (0, 0, 0), 3)
            cv2.putText(frame, str(y), (3, py - 4), font, fs, (0, 255, 255), 1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"{src.name}: {w}x{h}, grid tiap {args.step} px -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
