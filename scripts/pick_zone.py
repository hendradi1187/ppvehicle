"""Click a region polygon on the first frame of a video.

    python scripts/pick_zone.py data/samples/jalan.mp4
    python scripts/pick_zone.py data/samples/jalan.mp4 --frame 120 --save my_zone

Keys: left click = add point, u = undo, s = save JSON, q/Esc = quit.
Coordinates are printed in the video's native resolution even when the preview
window is scaled down, which is what --region_polygon expects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_PREVIEW_W = 1280


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="video file (or image) to pick the zone on")
    ap.add_argument("--frame", type=int, default=0, help="frame index to show")
    ap.add_argument("--save", default="", help="name to save under configs/zones/")
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
    scale = min(1.0, MAX_PREVIEW_W / w)
    points: list[tuple[int, int]] = []

    def redraw() -> None:
        canvas = frame.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(canvas, (x, y), 6, (0, 200, 255), -1)
            cv2.putText(canvas, str(i), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if len(points) > 1:
            cv2.polylines(canvas, [np.array(points)],
                          len(points) > 2, (0, 200, 255), 2)
        cv2.putText(canvas, f"{len(points)} titik | klik=tambah  u=undo  s=simpan  q=keluar",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if scale < 1.0:
            canvas = cv2.resize(canvas, (int(w * scale), int(h * scale)))
        cv2.imshow("pick zone", canvas)

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x / scale), int(y / scale)))
            redraw()

    cv2.namedWindow("pick zone")
    cv2.setMouseCallback("pick zone", on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u") and points:
            points.pop()
            redraw()
        if key == ord("s"):
            if len(points) < 3:
                print("butuh minimal 3 titik")
                continue
            flat = [c for p in points for c in p]
            print("\nregion_polygon:", flat)
            print("CLI  : --region-polygon " + " ".join(str(v) for v in flat))
            print("config: region_polygon: " + json.dumps(flat))
            if args.save:
                dest = ROOT / "configs" / "zones" / f"{args.save}.json"
                dest.write_text(json.dumps({
                    "name": args.save,
                    "video": str(src),
                    "frame_size": [w, h],
                    "region_polygon": flat,
                }, indent=2) + "\n", encoding="utf-8")
                print("tersimpan:", dest)
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
