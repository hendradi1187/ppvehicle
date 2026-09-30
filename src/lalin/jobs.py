"""In-memory job registry for pipeline runs.

Deliberately not a real queue: PP-Vehicle runs are minutes-long GPU-bound
subprocesses, and one box handles a handful at a time. If this ever needs to
survive a restart or fan out across workers, swap this module for Celery/RQ —
the API surface (`submit`, `get`, `all`, `cancel`) is what the rest of the code
depends on.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import runner
from .settings import settings

MAX_LOG_LINES = 2000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    spec: runner.RunSpec
    status: str = "queued"           # queued | running | done | failed
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))

    def public(self, log_tail: int = 50) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "scenario": self.spec.scenario,
            "profile": self.spec.profile or settings.profile,
            "source": self.spec.source,
            "source_kind": self.spec.resolved_kind(),
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "log_tail": list(self.lines)[-log_tail:] if log_tail else [],
        }


class Registry:
    def __init__(self, max_concurrent: int | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(max_concurrent or settings.max_jobs)

    def submit(self, spec: runner.RunSpec) -> Job:
        with self._lock:
            base_id = spec.run_id or runner.new_run_id(spec.scenario)
            job_id = base_id
            urutan = 2
            while job_id in self._jobs:
                job_id = f"{base_id}-{urutan}"
                urutan += 1
            spec.run_id = job_id
            job = Job(id=job_id, spec=spec)
            self._jobs[job_id] = job

        def worker() -> None:
            self._slots.acquire()
            job.status = "running"
            try:
                result = runner.run(spec, on_line=job.lines.append)
                job.result = result
                job.status = "done" if result.get("returncode") == 0 else "failed"
            except Exception as exc:
                job.result = {"returncode": -1,
                              "error": f"{type(exc).__name__}: {exc}"}
                job.status = "failed"
                job.lines.append(f"[ppvehicle] ERROR {job.result['error']}")
            finally:
                job.finished_at = _now()
                self._slots.release()

        threading.Thread(target=worker, daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)


registry = Registry()
