"""Model zoo registry and offline downloader.

PaddleDetection happily takes a bcebos URL as `model_dir` and caches it itself,
but a GPU server behind a firewall (and repeatable builds) want the weights on
disk. `ensure(...)` downloads + extracts into `models/` and returns a local
path; `resolve(...)` falls back to the URL when the local copy is missing.
"""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from .settings import settings

BCE = "https://bj.bcebos.com/v1/paddledet/models/pipeline"


@dataclass(frozen=True)
class Model:
    key: str
    url: str
    size_mb: float
    task: str
    note: str = ""

    @property
    def archive_name(self) -> str:
        return self.url.rsplit("/", 1)[-1]

    @property
    def dir_name(self) -> str:
        """Directory the archive extracts to (archive name minus extension)."""
        name = self.archive_name
        for suffix in (".tar.gz", ".tgz", ".tar", ".zip"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name


MODEL_ZOO: dict[str, Model] = {
    "ppyoloe_l": Model(
        "ppyoloe_l", f"{BCE}/mot_ppyoloe_l_36e_ppvehicle.zip", 182.0,
        "detection + MOT", "high precision, 25.7ms on T4+TRT",
    ),
    "ppyoloe_s": Model(
        "ppyoloe_s", f"{BCE}/mot_ppyoloe_s_36e_ppvehicle.zip", 26.6,
        "detection + MOT", "lightweight, 13.2ms on T4+TRT",
    ),
    "vehicle_attr": Model(
        "vehicle_attr", f"{BCE}/vehicle_attribute_model.zip", 6.1,
        "attribute", "PP-LCNet, 10 colors + 9 body types (VeRi dataset)",
    ),
    "plate_det": Model(
        "plate_det", f"{BCE}/ch_PP-OCRv3_det_infer.tar.gz", 2.2,
        "plate detection", "PP-OCRv3 text detector",
    ),
    "plate_rec": Model(
        "plate_rec", f"{BCE}/ch_PP-OCRv3_rec_infer.tar.gz", 9.0,
        "plate recognition", "PP-OCRv3 recognizer, trained on Chinese plates",
    ),
    "lane_seg": Model(
        "lane_seg", f"{BCE}/pp_lite_stdc2_bdd100k.zip", 43.5,
        "lane segmentation", "PP-LiteSeg on BDD100K, needed for pressing/retrograde",
    ),
}

# Which zoo entries a scenario's enabled modules require.
MODULE_MODELS: dict[str, tuple[str, ...]] = {
    "VEHICLE_ATTR": ("vehicle_attr",),
    "VEHICLE_PLATE": ("plate_det", "plate_rec"),
    "VEHICLE_PRESSING": ("lane_seg",),
    "VEHICLE_RETROGRADE": ("lane_seg",),
}


def local_dir(key: str) -> Path:
    return settings.models_dir / MODEL_ZOO[key].dir_name


def is_present(key: str) -> bool:
    d = local_dir(key)
    return d.is_dir() and any(d.iterdir())


def resolve(key: str) -> str:
    """Local path if downloaded, else the upstream URL (auto-download)."""
    return str(local_dir(key)) if is_present(key) else MODEL_ZOO[key].url


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
                done += len(block)
                if total:
                    pct = done * 100 // total
                    print(f"\r  {dest.name}: {pct:3d}%  "
                          f"({done/1e6:.1f}/{total/1e6:.1f} MB)", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


def _extract(archive: Path, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    else:
        with tarfile.open(archive) as t:
            # filter="data" guards against path traversal in the archive
            t.extractall(into, filter="data")


def ensure(key: str, force: bool = False) -> Path:
    """Download + extract one model. Returns its local directory."""
    model = MODEL_ZOO[key]
    target = local_dir(key)
    if target.is_dir() and any(target.iterdir()) and not force:
        print(f"  {key}: already present ({target.name})")
        return target
    if force and target.is_dir():
        shutil.rmtree(target)

    archives = settings.models_dir / "_archives"
    archive = archives / model.archive_name
    if not archive.exists():
        print(f"  {key}: downloading {model.size_mb:.0f} MB ...")
        _download(model.url, archive)
    print(f"  {key}: extracting ...")
    _extract(archive, settings.models_dir)
    if not target.is_dir():
        raise RuntimeError(
            f"{model.archive_name} did not extract to expected dir {target}. "
            f"Contents: {sorted(p.name for p in settings.models_dir.iterdir())}"
        )
    return target


def ensure_many(keys: list[str], force: bool = False) -> dict[str, Path]:
    return {k: ensure(k, force=force) for k in dict.fromkeys(keys)}


def status() -> list[dict]:
    return [
        {
            "key": m.key,
            "task": m.task,
            "size_mb": m.size_mb,
            "note": m.note,
            "downloaded": is_present(m.key),
            "path": str(local_dir(m.key)),
        }
        for m in MODEL_ZOO.values()
    ]
