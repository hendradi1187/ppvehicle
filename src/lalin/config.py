"""Compose a runnable PaddleDetection pipeline config from our own
profile + scenario files.

Our configs are deliberately NOT PaddleDetection's format: a *profile* says
where and how fast to run (device, precision, which detector size), a
*scenario* says what to compute (which modules, which CLI flags). This module
merges the two into the real `infer_cfg_ppvehicle.yml` schema plus an argv
list for `deploy/pipeline/pipeline.py`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import models
from .settings import settings

# Modules that carry an `enable` flag in the upstream schema.
TOGGLEABLE = ("MOT", "VEHICLE_ATTR", "VEHICLE_PLATE", "VEHICLE_PRESSING",
              "VEHICLE_RETROGRADE")

# CLI flags that are booleans (passed as a bare --flag, omitted when false).
BOOL_FLAGS = ("do_entrance_counting", "do_break_in_counting", "draw_center_traj")

# CLI flags taking a value.
VALUE_FLAGS = ("region_type", "illegal_parking_time", "secs_interval")


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def list_profiles() -> list[str]:
    return sorted(p.stem for p in (settings.configs_dir / "profiles").glob("*.yml"))


def list_scenarios() -> list[dict]:
    out = []
    for p in sorted((settings.configs_dir / "scenarios").glob("*.yml")):
        cfg = _load(p)
        out.append({
            "name": cfg.get("name", p.stem),
            "title": cfg.get("title", p.stem),
            "description": (cfg.get("description") or "").strip(),
            "modules": [k for k, v in (cfg.get("modules") or {}).items() if v],
        })
    return out


def load_profile(name: str) -> dict:
    path = settings.configs_dir / "profiles" / f"{name}.yml"
    if not path.exists():
        raise FileNotFoundError(
            f"Unknown profile {name!r}. Available: {', '.join(list_profiles())}")
    return _load(path)


def load_scenario(name: str) -> dict:
    path = settings.configs_dir / "scenarios" / f"{name}.yml"
    if not path.exists():
        avail = ", ".join(s["name"] for s in list_scenarios())
        raise FileNotFoundError(f"Unknown scenario {name!r}. Available: {avail}")
    return _load(path)


def required_models(scenario: dict, profile: dict) -> list[str]:
    """Zoo keys this combination needs, detector first."""
    keys = [profile.get("detector", "ppyoloe_l")]
    enabled = {k for k, v in (scenario.get("modules") or {}).items() if v}
    for module, needed in models.MODULE_MODELS.items():
        if module in enabled:
            keys.extend(needed)
    return list(dict.fromkeys(keys))


@dataclass
class Composed:
    """Everything needed to launch one pipeline run."""
    infer_cfg: dict
    lane_seg_cfg: dict
    flags: list[str] = field(default_factory=list)
    profile_name: str = "cpu"
    scenario_name: str = "tracking"

    def write(self, run_dir: Path) -> Path:
        """Materialise the config files into run_dir, return the infer cfg path."""
        run_dir.mkdir(parents=True, exist_ok=True)
        lane_path = run_dir / "lane_seg_config.yml"
        with open(lane_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.lane_seg_cfg, f, sort_keys=False,
                           allow_unicode=True)

        cfg = copy.deepcopy(self.infer_cfg)
        cfg["LANE_SEG"]["lane_seg_config"] = str(lane_path)

        infer_path = run_dir / "infer_cfg.yml"
        with open(infer_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        return infer_path


def compose(scenario_name: str, profile_name: str | None = None,
            overrides: dict[str, Any] | None = None) -> Composed:
    """Build the PaddleDetection config for one scenario on one profile.

    `overrides` is a shallow per-section dict, e.g.
    {"VEHICLE_RETROGRADE": {"fence_line": [570, 163, 1030, 752]},
     "args": {"region_polygon": [600, 300, 1300, 300, 1300, 800, 600, 800]}}
    """
    profile_name = profile_name or settings.profile
    profile = load_profile(profile_name)
    scenario = load_scenario(scenario_name)

    device = str(profile.get("device", "cpu")).lower()
    detector = profile.get("detector", "ppyoloe_l")
    det_dir = models.resolve(detector)

    pipeline_root = settings.paddledet / "deploy" / "pipeline"

    infer: dict[str, Any] = {
        "crop_thresh": 0.5,
        "visual": bool(profile.get("visual", True)),
        "warmup_frame": 50 if device == "gpu" else 0,
        "DET": {"model_dir": det_dir, "batch_size": 1},
        "MOT": {
            "model_dir": det_dir,
            # OC-SORT sesuai dokumen PP-Vehicle - lihat kepala berkasnya untuk
            # mengapa bukan tracker_config.yml bawaan vendor (BOTSORT).
            "tracker_config": str(settings.configs_dir / "tracker_ppvehicle.yml"),
            "batch_size": 1,
            "skip_frame_num": int(profile.get("skip_frame_num", -1)),
            "enable": False,
        },
        "VEHICLE_PLATE": {
            "det_model_dir": models.resolve("plate_det"),
            "det_limit_side_len": 736,
            "det_limit_type": "min",
            "rec_model_dir": models.resolve("plate_rec"),
            "rec_image_shape": [3, 48, 320],
            "rec_batch_num": 6,
            "word_dict_path": str(pipeline_root / "ppvehicle" / "rec_word_dict.txt"),
            "enable": False,
        },
        "VEHICLE_ATTR": {
            "model_dir": models.resolve("vehicle_attr"),
            "batch_size": 8,
            "color_threshold": 0.5,
            "type_threshold": 0.5,
            "enable": False,
        },
        "LANE_SEG": {
            "lane_seg_config": "",           # filled in by Composed.write()
            "model_dir": models.resolve("lane_seg"),
        },
        "VEHICLE_PRESSING": {"enable": False},
        "VEHICLE_RETROGRADE": {
            "frame_len": 8,
            "sample_freq": 7,
            "enable": False,
            "filter_horizontal_flag": True,
            "keep_right_flag": True,
            "deviation": 23,
            "move_scale": 0.01,
            "fence_line": [],
        },
    }

    # 1. scenario module toggles
    for module, on in (scenario.get("modules") or {}).items():
        if module not in TOGGLEABLE:
            raise ValueError(f"{scenario_name}: {module!r} has no enable flag")
        infer[module]["enable"] = bool(on)

    # 2. scenario section overrides, then caller overrides
    for source in (scenario.get("overrides") or {}, (overrides or {})):
        for section, values in source.items():
            if section == "args":
                continue
            if section not in infer:
                raise ValueError(f"Unknown config section {section!r}")
            infer[section].update(values)

    # Upstream's lane_seg_config.yml hardcodes device: gpu, so always regenerate.
    lane_seg = {
        "type": "PLSLaneseg",
        "PLSLaneseg": {
            "run_mode": profile.get("run_mode", "paddle"),
            "batch_size": 1,
            "device": device,
            "min_subgraph_size": 3,
            "use_dynamic_shape": False,
            "trt_min_shape": [100, 100],
            "trt_max_shape": [2000, 3000],
            "trt_opt_shape": [512, 1024],
            "trt_calib_mode": False,
            "cpu_threads": int(profile.get("cpu_threads", 1)),
            "enable_mkldnn": bool(profile.get("enable_mkldnn", False)),
            "filter_horizontal_flag":
                infer["VEHICLE_RETROGRADE"]["filter_horizontal_flag"],
            "horizontal_filtration_degree": 23,
            "horizontal_filtering_threshold": 0.25,
        },
    }

    # 3. CLI flags: profile runtime, then scenario args, then caller args
    run_mode = str(profile.get("run_mode", "paddle"))
    flags = [
        "--device", device,
        "--run_mode", run_mode,
        "--cpu_threads", str(profile.get("cpu_threads", 1)),
        "--enable_mkldnn", str(bool(profile.get("enable_mkldnn", False))),
    ]
    if run_mode.startswith("trt"):
        for key in ("trt_min_shape", "trt_max_shape", "trt_opt_shape"):
            if key in profile:
                flags += [f"--{key}", str(profile[key])]

    args: dict[str, Any] = dict(scenario.get("args") or {})
    args.update((overrides or {}).get("args") or {})

    for key in BOOL_FLAGS:
        if args.get(key):
            flags.append(f"--{key}")
    for key in VALUE_FLAGS:
        if key in args:
            flags += [f"--{key}", str(args[key])]
    if args.get("region_polygon"):
        flags.append("--region_polygon")
        flags += [str(v) for v in args["region_polygon"]]

    return Composed(infer_cfg=infer, lane_seg_cfg=lane_seg, flags=flags,
                    profile_name=profile_name, scenario_name=scenario_name)
