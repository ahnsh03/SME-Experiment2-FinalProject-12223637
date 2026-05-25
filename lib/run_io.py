"""Run logging and artifact manifest for final_project."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .io_mat import project_root


class RunSession:
    def __init__(self, script_stem: str) -> None:
        self.root = project_root()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{script_stem}_{ts}"
        self.run_dir = self.root / "outputs" / "run_reports" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "run.log"
        self._t0 = time.perf_counter()
        self._log = self.log_path.open("w", encoding="utf-8", newline="\n")
        self._orig_out, self._orig_err = sys.stdout, sys.stderr

    def __enter__(self) -> "RunSession":
        sys.stdout = _Tee(sys.stdout, self._log)
        sys.stderr = _Tee(sys.stderr, self._log)
        print(f"[run] id={self.run_id} cwd={Path.cwd()}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        sys.stdout, sys.stderr = self._orig_out, self._orig_err
        elapsed = time.perf_counter() - self._t0
        manifest = {
            "run_id": self.run_id,
            "elapsed_s": round(elapsed, 3),
            "exit": "error" if exc else "ok",
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        self._log.close()


class _Tee:
    def __init__(self, primary: Any, log_fp: Any) -> None:
        self._p = primary
        self._log = log_fp

    def write(self, data: str) -> int:
        self._log.write(data)
        self._log.flush()
        return self._p.write(data)

    def flush(self) -> None:
        self._log.flush()
        self._p.flush()

    def isatty(self) -> bool:
        return bool(self._p.isatty())
