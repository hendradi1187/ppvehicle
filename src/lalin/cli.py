"""Command line entry point: `python -m lalin <command>`."""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from . import batch, config, models, runner
from .settings import settings


def cmd_doctor(_args: argparse.Namespace) -> int:
    from .api import doctor
    info = doctor()
    print(f"python                 : {info['python']}")
    print(f"paddlepaddle           : {info['paddle']}")
    print(f"paddle CUDA build      : {info['paddle_compiled_with_cuda']}")
    print(f"PaddleDetection        : {info['paddledetection']['path']} "
          f"({'OK' if info['paddledetection']['present'] else 'MISSING'})")
    print(f"ffmpeg on PATH         : {info['ffmpeg']}")
    print(f"default profile        : {info['default_profile']} "
          f"(available: {', '.join(info['profiles'])})")
    print("\nmodels:")
    for m in info["models"]:
        mark = "OK " if m["downloaded"] else "-- "
        print(f"  {mark} {m['key']:<14} {m['size_mb']:>6.1f} MB  {m['task']}")
    if not info["paddledetection"]["present"]:
        print("\nPaddleDetection is missing. Run:")
        print("  powershell -ExecutionPolicy Bypass -File scripts/setup.ps1")
        return 1
    return 0


def cmd_scenarios(_args: argparse.Namespace) -> int:
    for s in config.list_scenarios():
        mods = ", ".join(s["modules"]) or "DET only"
        print(f"{s['name']:<18} {s['title']}")
        print(f"{'':<18} modules: {mods}")
        if s["description"]:
            for line in s["description"].splitlines():
                print(f"{'':<18} {line.strip()}")
        print()
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    if args.download:
        keys = args.keys or list(models.MODEL_ZOO)
        unknown = [k for k in keys if k not in models.MODEL_ZOO]
        if unknown:
            print(f"Unknown model key(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        total = sum(models.MODEL_ZOO[k].size_mb for k in keys
                    if not models.is_present(k))
        print(f"Downloading into {settings.models_dir} (~{total:.0f} MB to fetch)")
        models.ensure_many(keys, force=args.force)
        return 0
    for m in models.status():
        mark = "OK" if m["downloaded"] else "--"
        print(f"[{mark}] {m['key']:<14} {m['size_mb']:>6.1f} MB  {m['task']:<22} {m['note']}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    composed = config.compose(args.scenario, args.profile)
    print("# infer_cfg.yml")
    print(yaml.safe_dump(composed.infer_cfg, sort_keys=False, allow_unicode=True))
    print("# extra CLI flags")
    print(" ".join(composed.flags))
    spec = runner.RunSpec(scenario=args.scenario, source="preview.mp4",
                          profile=args.profile)
    missing = runner.preflight(spec)
    if missing:
        print(f"\n# models not downloaded yet: {', '.join(missing)}")
        print(f"#   python -m lalin models --download {' '.join(missing)}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    overrides: dict = {}
    run_args: dict = {}
    if args.region_polygon:
        run_args["region_polygon"] = args.region_polygon
        run_args["region_type"] = "custom"
    if args.region_type:
        run_args["region_type"] = args.region_type
    if args.illegal_parking_time is not None:
        run_args["illegal_parking_time"] = args.illegal_parking_time
    if args.fence_line:
        overrides["VEHICLE_RETROGRADE"] = {"fence_line": args.fence_line}
    if run_args:
        overrides["args"] = run_args

    spec = runner.RunSpec(
        scenario=args.scenario,
        source=args.source,
        profile=args.profile,
        source_kind=args.source_kind,
        camera_id=args.camera_id,
        pushurl=args.pushurl,
        overrides=overrides,
    )

    if args.dry_run:
        argv, run_dir, out_dir = runner.build_command(spec)
        print("cwd:", settings.paddledet)
        print("cmd:", " ".join(argv))
        print("run dir:", run_dir)
        print("output :", out_dir)
        return 0

    result = runner.run(spec, on_line=lambda line: print(line, flush=True))
    print("\n--- result ---")
    print(json.dumps(result, indent=2))
    return 0 if result.get("returncode") == 0 else 1


def cmd_batch(args: argparse.Namespace) -> int:
    cams = {c.key: c for c in batch.load_cameras()}
    if args.list:
        for c in cams.values():
            poly = " ".join(str(v) for v in c.region_polygon) or "-"
            print(f"{c.key:<12} {c.name:<26} {c.scenario:<16} zona: {poly}")
        return 0

    if args.camera:
        if args.camera not in cams:
            print(f"Kamera tidak dikenal: {args.camera}. "
                  f"Tersedia: {', '.join(cams)}", file=sys.stderr)
            return 2
        rec = batch.analyse(cams[args.camera], args.profile)
        print(json.dumps(rec, indent=2, ensure_ascii=False))
        return 0 if rec.get("ok") else 1

    batch.worker.profile = args.profile
    records = batch.worker.cycle()
    print(json.dumps(records, indent=2, ensure_ascii=False))
    return 0 if all(r.get("ok") for r in records) else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    uvicorn.run("lalin.api:app", host=args.host, port=args.port,
                reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("lalin", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check environment, models, PaddleDetection")
    sub.add_parser("scenarios", help="list available scenarios")

    m = sub.add_parser("models", help="list or download model weights")
    m.add_argument("keys", nargs="*", help="zoo keys; default = all")
    m.add_argument("--download", action="store_true")
    m.add_argument("--force", action="store_true", help="re-extract even if present")

    pv = sub.add_parser("preview", help="print the config a run would use")
    pv.add_argument("--scenario", required=True)
    pv.add_argument("--profile", default=None)

    r = sub.add_parser("run", help="run a scenario on a source")
    r.add_argument("--scenario", required=True)
    r.add_argument("--source", required=True,
                   help="video/image path, directory, rtsp:// URL, or 'camera'")
    r.add_argument("--profile", default=None, help="cpu | gpu (default: $PPVEHICLE_PROFILE)")
    r.add_argument("--source-kind", dest="source_kind", default="auto",
                   choices=["auto", "video", "image", "image_dir", "video_dir",
                            "rtsp", "camera"])
    r.add_argument("--camera-id", dest="camera_id", type=int, default=0)
    r.add_argument("--pushurl", default="", help="rtsp://host:8554 to push results")
    r.add_argument("--region-polygon", dest="region_polygon", nargs="+", type=int,
                   help="x0 y0 x1 y1 ... clockwise")
    r.add_argument("--region-type", dest="region_type",
                   choices=["horizontal", "vertical", "custom"])
    r.add_argument("--illegal-parking-time", dest="illegal_parking_time", type=int)
    r.add_argument("--fence-line", dest="fence_line", nargs=4, type=int,
                   metavar=("X1", "Y1", "X2", "Y2"))
    r.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="print the command instead of running it")

    b = sub.add_parser("batch", help="analisis batch berkala atas kamera")
    b.add_argument("--list", action="store_true", help="daftar kamera saja")
    b.add_argument("--camera", help="analisis satu kamera (key)")
    b.add_argument("--profile", default=None, help="cpu | gpu")

    s = sub.add_parser("serve", help="start the API + web UI")
    s.add_argument("--host", default=settings.host)
    s.add_argument("--port", type=int, default=settings.port)
    s.add_argument("--reload", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {
        "doctor": cmd_doctor,
        "scenarios": cmd_scenarios,
        "models": cmd_models,
        "preview": cmd_preview,
        "run": cmd_run,
        "batch": cmd_batch,
        "serve": cmd_serve,
    }[args.command]
    try:
        return handler(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
