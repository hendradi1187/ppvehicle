"""Project paths and environment-driven settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# src/ppvehicle/settings.py -> project root is three levels up
ROOT = Path(__file__).resolve().parents[2]


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _path(key: str, default: str) -> Path:
    raw = Path(_env(key, default))
    return raw if raw.is_absolute() else (ROOT / raw)


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    paddledet: Path = _path("PPVEHICLE_PADDLEDET", "vendor/PaddleDetection")
    models_dir: Path = _path("PPVEHICLE_MODELS_DIR", "models")
    configs_dir: Path = ROOT / "configs"
    uploads_dir: Path = _path("PPVEHICLE_UPLOADS_DIR", "data/samples")
    runs_dir: Path = _path("PPVEHICLE_RUNS_DIR", "runs")

    # Mesin penghitung multi-kelas memakai detektor TERPISAH dari PP-Vehicle:
    # PP-YOLOE terlatih COCO, diekspor sendiri karena PaddleDetection tidak
    # menerbitkan paket inferensinya. Lihat _entry_hitung.py untuk alasan
    # pemilihan modelnya.
    # 0.20, bukan 0.30: pada 0.30 seluruh sepeda motor di feed 640x360 hilang.
    # Nilainya harus selaras dengan det_thresh/conf_thres di
    # configs/tracker_hitung.yml - alasannya ditulis lengkap di berkas itu.
    hitung_threshold: float = float(_env("PPVEHICLE_HITUNG_THRESHOLD", "0.20"))

    profile: str = _env("PPVEHICLE_PROFILE", "cpu")
    host: str = _env("PPVEHICLE_HOST", "127.0.0.1")
    port: int = int(_env("PPVEHICLE_PORT", "8000"))
    max_jobs: int = int(_env("PPVEHICLE_MAX_JOBS", "1"))

    # Analisis berkala menyala sendiri saat server start. Sebelumnya penjadwal
    # ada tetapi harus dinyalakan manual, sehingga dashboard hanya berisi data
    # bila seseorang menekan tombol analisis per kamera - bukan perilaku yang
    # bisa dipakai di ruang kendali. Setel 0 untuk mematikan.
    auto_batch_sec: int = int(_env("PPVEHICLE_AUTO_BATCH_SEC", "900"))
    # Basis data. Bila PPVEHICLE_DB_HOST diisi, aplikasi memakai PostgreSQL;
    # bila kosong, SQLite di runs/lalin.db (laptop pengembangan). Kata sandi
    # dibaca dari LALIN_APP_PASSWORD - di server berasal dari docker/.env.app,
    # berkas izin 600 yang HANYA memuat kata sandi akun aplikasi, bukan akun
    # admin atau akun baca tim DB.
    db_host: str = _env("PPVEHICLE_DB_HOST", "")
    db_port: int = int(_env("PPVEHICLE_DB_PORT", "5432"))
    db_name: str = _env("PPVEHICLE_DB_NAME", "lalin")
    db_user: str = _env("PPVEHICLE_DB_USER", "lalin_app")

    # Retensi berkas (batch.bersihkan). Disetujui pengguna 25 Sep 2026.
    # Basis data TIDAK termasuk: kejadian, aduan, dan keputusan verifikasi
    # disimpan penuh. Yang dibersihkan hanya berkas sampingan analisis.
    retensi_run_hari: float = float(_env("PPVEHICLE_RETENSI_RUN_HARI", "2"))
    retensi_pantau_jam: float = float(_env("PPVEHICLE_RETENSI_PANTAU_JAM", "1"))
    retensi_bukti_hari: float = float(_env("PPVEHICLE_RETENSI_BUKTI_HARI", "90"))

    # Batas waktu satu proses pipeline atau penghitung. Lihat runner.run.
    run_timeout_sec: int = int(_env("PPVEHICLE_RUN_TIMEOUT_SEC", "900"))

    @property
    def hitung_model(self) -> Path:
        """Harus mengikuti models_dir, bukan ROOT: di kontainer model berada di
        volume /models sedangkan ROOT adalah /app. Sebagai field biasa,
        defaultnya diselesaikan relatif ROOT dan menunjuk direktori yang tidak
        pernah ada."""
        raw = os.environ.get("PPVEHICLE_HITUNG_MODEL")
        if raw:
            p = Path(raw)
            return p if p.is_absolute() else (self.root / p)
        # PP-YOLOE+ CRN L (COCO) bila terpasang, CRN S sebagai cadangan.
        # Diuji 24 Sep 2026 pada enam frame malam dan dua frame siang dari
        # kamera yang sama, aturan ambang identik:
        #   - malam S. Parman 07: S tidak menemukan satu pun sepeda motor dan
        #     menandai papan informasi; L menemukan motor tepat di bawah
        #     pengendaranya, sehingga penyaring pengendara bisa bekerja.
        #   - siang CCTV-01: L menutup deretan motor terparkir lebih lengkap
        #     dan tidak lagi melabeli mobil atau atap halte sebagai bus.
        # Harganya 2,5x lebih lambat (449 vs 183 ms/frame di CPU ini).
        besar = self.models_dir / "ppyoloe_plus_l_coco"
        if (besar / "model.pdmodel").is_file():
            return besar
        return self.models_dir / "ppyoloe_coco"

    @property
    def hitung_tracker_cfg(self) -> Path:
        """Wajib JDETracker; tracker bawaan pptracking hanya satu kelas."""
        return self.configs_dir / "tracker_hitung.yml"

    @property
    def pipeline_script(self) -> Path:
        return self.paddledet / "deploy" / "pipeline" / "pipeline.py"

    def check_paddledet(self) -> None:
        """Fail early with an actionable message instead of a subprocess error."""
        if not self.pipeline_script.exists():
            raise FileNotFoundError(
                f"PaddleDetection not found at {self.paddledet}.\n"
                "Run: powershell -ExecutionPolicy Bypass -File scripts/setup.ps1"
            )


settings = Settings()
