"""Launch PaddleDetection's pipeline as a subprocess.

Running it in-process would mean importing a script that does its own sys.path
surgery and calls sys.exit; a subprocess keeps our API alive when the pipeline
dies, and lets us stream logs. The working directory must be the
PaddleDetection root because parts of the upstream code resolve
`deploy/...` paths relative to cwd.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import config, models
from .settings import settings


@dataclass
class RunSpec:
    """One pipeline invocation."""
    scenario: str
    source: str                      # file path, directory, rtsp:// url, or "camera"
    profile: str | None = None
    source_kind: str = "auto"        # auto|video|image|image_dir|video_dir|rtsp|camera
    camera_id: int = 0
    pushurl: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    # Garis marka operator, koordinat frame. Diteruskan lewat environment dan
    # bukan lewat infer_cfg karena upstream tidak punya kunci untuknya - ia
    # selalu mengambil garis dari keluaran LANE_SEG.
    marka: list[list[float]] = field(default_factory=list)

    def resolved_kind(self) -> str:
        if self.source_kind != "auto":
            return self.source_kind
        src = self.source.strip()
        if src.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
            return "rtsp"
        if src == "camera":
            return "camera"
        p = Path(src)
        if p.is_dir():
            exts = {q.suffix.lower() for q in p.iterdir() if q.is_file()}
            return "video_dir" if exts & {".mp4", ".avi", ".mov", ".mkv"} else "image_dir"
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return "image"
        return "video"


def new_run_id(scenario: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{scenario}"


def _source_flags(spec: RunSpec) -> list[str]:
    kind = spec.resolved_kind()
    if kind == "rtsp":
        return ["--rtsp", spec.source]
    if kind == "camera":
        return ["--camera_id", str(spec.camera_id)]
    path = str(Path(spec.source).resolve())
    return {
        "image": ["--image_file", path],
        "image_dir": ["--image_dir", path],
        "video": ["--video_file", path],
        "video_dir": ["--video_dir", path],
    }[kind]



def _prioritas_rendah() -> None:
    """Siklus otomatis dan pantau berjalan pada nice 10.

    Deteksi langsung (Live yang sedang ditonton) berbagi 8 CPU kontainer
    dengan siklus ini. Tanpa prioritas, keduanya melambat bersama - terukur
    1215 ms/frame alih-alih 510 - dan kotak Live tertinggal dari videonya.
    Dengan nice, siklus tetap jalan, hanya mengalah saat ada yang menonton.
    """
    try:
        os.nice(10)
    except OSError:
        pass


def build_command(spec: RunSpec) -> tuple[list[str], Path, Path]:
    """Returns (argv, run_dir, output_dir). Also writes the generated configs."""
    settings.check_paddledet()

    composed = config.compose(spec.scenario, spec.profile, spec.overrides)
    run_id = spec.run_id or new_run_id(spec.scenario)
    run_dir = settings.runs_dir / run_id
    output_dir = run_dir / "output"
    cfg_path = composed.write(run_dir)

    # _entry.py wraps pipeline.py so the run also leaves a results.json behind;
    # see its docstring. Falls back to calling pipeline.py directly if the
    # entry module is somehow missing.
    entry = Path(__file__).parent / "_entry.py"
    script = entry if entry.exists() else settings.pipeline_script

    argv = [
        sys.executable,
        str(script),
        "--config", str(cfg_path),
        "--output_dir", str(output_dir),
        *_source_flags(spec),
        *composed.flags,
    ]
    if spec.pushurl:
        argv += ["--pushurl", spec.pushurl]

    kind = spec.resolved_kind()
    if kind == "image" or kind == "image_dir":
        # MOT is meaningless on stills and upstream errors out if it is on.
        if composed.infer_cfg["MOT"]["enable"]:
            argv += ["-o", "MOT.enable=False"]

    return argv, run_dir, output_dir


def _child_env() -> dict:
    """Environment for the pipeline subprocess."""
    env = dict(os.environ)
    env["LALIN_PIPELINE_DIR"] = str(settings.paddledet / "deploy" / "pipeline")
    return env


def results_path(run_dir: Path) -> Path:
    return run_dir / "results.json"


def read_results(run_dir: Path) -> dict | None:
    path = results_path(run_dir)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def preflight(spec: RunSpec) -> list[str]:
    """Zoo keys this run needs that are not downloaded yet."""
    profile = config.load_profile(spec.profile or settings.profile)
    scenario = config.load_scenario(spec.scenario)
    return [k for k in config.required_models(scenario, profile)
            if not models.is_present(k)]


def run(spec: RunSpec,
        on_line: Callable[[str], None] | None = None,
        download_missing: bool = True) -> dict:
    """Run to completion, streaming stdout. Returns a result summary."""
    if download_missing:
        missing = preflight(spec)
        if missing:
            if on_line:
                on_line(f"[ppvehicle] downloading {len(missing)} model(s): "
                        f"{', '.join(missing)}")
            models.ensure_many(missing)

    argv, run_dir, output_dir = build_command(spec)
    log_path = run_dir / "run.log"
    started = time.time()

    if on_line:
        on_line("[ppvehicle] cwd  = " + str(settings.paddledet))
        on_line("[ppvehicle] argv = " + " ".join(argv))

    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write(" ".join(argv) + "\n\n")
        log.flush()
        env = _child_env()
        env["LALIN_RESULTS"] = str(results_path(run_dir))
        if spec.marka:
            env["LALIN_MARKA"] = json.dumps(spec.marka)
        proc = subprocess.Popen(
            argv,
            cwd=str(settings.paddledet),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=_prioritas_rendah,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        # Batas waktu keras. Tanpa ini satu pipeline yang menggantung membekukan
        # seluruh siklus otomatis tanpa tanda apa pun di dasbor - tercatat 19 jam
        # pada 24-25 Sep 2026. Batasnya longgar: klip 20 dtk di CPU normalnya
        # selesai dalam 2-4 menit.
        habis = threading.Event()
        def _bunuh(p=proc):
            habis.set()
            try:
                p.kill()
            except Exception:
                pass
        penjaga = threading.Timer(settings.run_timeout_sec, _bunuh)
        penjaga.daemon = True
        penjaga.start()
        for line in proc.stdout:
            log.write(line)
            if on_line:
                on_line(line.rstrip())
        code = proc.wait()
        penjaga.cancel()
        if habis.is_set():
            log.write(f"\n[lalin] dihentikan paksa: melewati batas {settings.run_timeout_sec} dtk\n")
            code = code if code not in (0, None) else -9

    return {
        "returncode": code,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "log": str(log_path),
        "artifacts": [str(p) for p in sorted(output_dir.rglob("*")) if p.is_file()],
        "elapsed_sec": round(time.time() - started, 1),
        "results": read_results(run_dir),
    }


def run_hitung(source: str, run_id: str, polygon: str = "",
               threshold: float | None = None,
               on_line: Callable[[str], None] | None = None) -> dict:
    """Jalankan mesin penghitung multi-kelas atas satu klip.

    Jalurnya sengaja tidak lewat `build_command`: mesin ini bukan pipeline
    PP-Vehicle. Ia detektor lain, konfigurasi lain, dan skenario PP-Vehicle
    tidak berlaku baginya. Yang dipakai bersama hanyalah tata letak run_dir,
    berkas log, dan results.json - supaya hasilnya bisa dibaca dengan cara yang
    sama oleh sisa aplikasi.
    """
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    entry = Path(__file__).parent / "_entry_hitung.py"
    if not entry.exists():
        raise FileNotFoundError(f"modul penghitung tidak ada: {entry}")
    if not (settings.hitung_model / "model.pdmodel").is_file():
        raise FileNotFoundError(
            f"model penghitung belum terpasang di {settings.hitung_model}. "
            "Ia diekspor sendiri dari bobot COCO; lihat docs/.")

    argv = [sys.executable, str(entry)]
    env = dict(os.environ)
    env["LALIN_PD_DIR"] = str(settings.paddledet / "deploy" / "pptracking" / "python")
    env["LALIN_MODEL_DIR"] = str(settings.hitung_model)
    env["LALIN_TRACKER_CFG"] = str(settings.hitung_tracker_cfg)
    env["LALIN_SOURCE"] = str(source)
    env["LALIN_RESULTS"] = str(results_path(run_dir))
    env["LALIN_POLYGON"] = polygon or ""
    env["LALIN_DEVICE"] = os.environ.get(
        "LALIN_DEVICE", "GPU" if settings.profile == "gpu" else "CPU").upper()
    env["LALIN_THRESHOLD"] = str(
        settings.hitung_threshold if threshold is None else threshold)

    started = time.time()
    log_path = run_dir / "run.log"
    if on_line:
        on_line("[hitung] model = " + str(settings.hitung_model))
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write(" ".join(argv) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            argv, cwd=str(settings.paddledet), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, preexec_fn=_prioritas_rendah,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout is not None
        # Batas waktu keras. Tanpa ini satu pipeline yang menggantung membekukan
        # seluruh siklus otomatis tanpa tanda apa pun di dasbor - tercatat 19 jam
        # pada 24-25 Sep 2026. Batasnya longgar: klip 20 dtk di CPU normalnya
        # selesai dalam 2-4 menit.
        habis = threading.Event()
        def _bunuh(p=proc):
            habis.set()
            try:
                p.kill()
            except Exception:
                pass
        penjaga = threading.Timer(settings.run_timeout_sec, _bunuh)
        penjaga.daemon = True
        penjaga.start()
        for line in proc.stdout:
            log.write(line)
            if on_line:
                on_line(line.rstrip())
        code = proc.wait()
        penjaga.cancel()
        if habis.is_set():
            log.write(f"\n[lalin] dihentikan paksa: melewati batas {settings.run_timeout_sec} dtk\n")
            code = code if code not in (0, None) else -9

    return {
        "returncode": code,
        "run_dir": str(run_dir),
        "log": str(log_path),
        "elapsed_sec": round(time.time() - started, 1),
        "results": read_results(run_dir),
    }


def run_async(spec: RunSpec, on_line: Callable[[str], None] | None = None,
              on_done: Callable[[dict], None] | None = None) -> threading.Thread:
    def target():
        try:
            result = run(spec, on_line=on_line)
        except Exception as exc:  # surfaced to the caller as a failed job
            result = {"returncode": -1, "error": f"{type(exc).__name__}: {exc}"}
            if on_line:
                on_line(f"[ppvehicle] ERROR {result['error']}")
        if on_done:
            on_done(result)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t
