"""Shared run-artifact conventions for the evolve/ pipeline.

Every run writer (inner / outer / co-evolution / metrics) stamps a `manifest.json`
and a `schema_version` so a saved run is self-describing and reproducible:
config + seed + detector identity + axes-config hash + git rev + argv + timestamps.
This is the provenance principle behind W&B / MLflow run metadata and the Croissant
dataset standard, kept dependency-free. `SCHEMA_VERSION` lets a future reader detect
format drift instead of silently mis-parsing an old run.

Convention across the pipeline:
  <run_dir>/manifest.json     provenance (this module)
  <run_dir>/iterations.jsonl  inner per-iteration stream (JSON Lines)
  <run_dir>/archive.json      inner final MAP-Elites archive (schema_version'd)
  <run_dir>/metrics.json      inner derived metrics (schema_version'd)
  <run_dir>/outer_log.jsonl   outer per-epoch stream (JSON Lines)
  <run_dir>/scenarios.json    outer final scenario population (schema_version'd)
  <run_dir>/coevolution.json  co-evolution per-round log (schema_version'd)
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_sha1(path: Optional[Any]) -> Optional[str]:
    if not path:
        return None
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]
    except Exception:
        return None


def git_rev(cwd: Optional[Any] = None) -> Optional[str]:
    """Short HEAD rev, or None if not a git repo / git unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(cwd) if cwd else None,
                             capture_output=True, text=True, timeout=3)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def build_manifest(layer: str, seed: Optional[int], *,
                   detector_signature: Optional[str] = None,
                   axes_path: Optional[Any] = None,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": layer,
        "created_utc": _utc_now(),
        "seed": seed,
        "detector_signature": detector_signature,
        "axes_path": str(axes_path) if axes_path else None,
        "axes_sha1": file_sha1(axes_path),
        "git_rev": git_rev(Path(axes_path).parent if axes_path else None),
        "argv": list(sys.argv),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "extra": extra or {},
    }


def write_manifest(out_dir: Any, layer: str, seed: Optional[int], **kw) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = build_manifest(layer, seed, **kw)
    (out_dir / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    return m
