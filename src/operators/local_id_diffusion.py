"""Diffusion-based ID / synthesis / edit operators (image-only forgery family).

These wrap heavy diffusion pipelines (InstructPix2Pix, SDXL T2I, IP-Adapter-face,
InstantID) whose deps live in the ISOLATED `forgery_img` conda env — NOT in the
orchestrator's `fakevlm` env. Each operator therefore shells out to a one-shot
subprocess running `forgery_img/bin/python id_diffusion_worker.py`, keeping
`fakevlm` (and every other env) untouched.

The wrapper imports nothing heavy, so it loads fine under any interpreter and the
registry registration in operators/__init__.py never ImportErrors.
"""
from __future__ import annotations
import os
import time
import uuid
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional

try:
    from operators.api_image import OperatorResult, ApiImageOperator
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from operators.api_image import OperatorResult, ApiImageOperator

_log = logging.getLogger(__name__)

FORGERY_IMG_PY = "/data/disk4/lyx_ICML/conda_envs/forgery_img/bin/python"
WORKER = str(Path(__file__).parent / "id_diffusion_worker.py")
_GEN_TIMEOUT = 600  # seconds per generation (model load + inference)


class _SubprocDiffusionOperator(ApiImageOperator):
    """Base: run a diffusion `method` in the isolated env via subprocess."""
    method: str = ""
    family: str = "id_diff"
    cost_per_call: float = 0.0
    default_size: str = "1024x1024"
    needs_src: bool = True  # T2I synthesis sets this False

    def __init__(self, client=None, out_dir="/tmp/face_attack_outputs", **_):
        # client unused (local op); kept for registry-uniform instantiation
        self.client = client
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def run(self, src_face_path=None, tgt_face_path=None, params=None, size=None):
        params = params or {}
        t0 = time.time()
        if self.needs_src and (not src_face_path or not Path(src_face_path).exists()):
            return OperatorResult(success=False, error="no src face",
                                  duration_sec=time.time() - t0, model_used=self.method)
        if not Path(FORGERY_IMG_PY).exists():
            return OperatorResult(success=False,
                                  error=f"forgery_img env missing: {FORGERY_IMG_PY}",
                                  duration_sec=time.time() - t0, model_used=self.method)

        out_path = self.out_dir / f"{self.name}_{uuid.uuid4().hex[:8]}.png"
        prompt = params.get("instruction") or params.get("prompt")
        cmd = [FORGERY_IMG_PY, WORKER, "--method", self.method,
               "--out", str(out_path), "--seed", str(int(params.get("seed", 0)))]
        if self.needs_src:
            cmd += ["--src", str(src_face_path)]
        if prompt:
            cmd += ["--prompt", str(prompt)[:400]]
        if params.get("steps"):
            cmd += ["--steps", str(int(params["steps"]))]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=_GEN_TIMEOUT)
        except subprocess.TimeoutExpired:
            return OperatorResult(success=False, error=f"timeout >{_GEN_TIMEOUT}s",
                                  duration_sec=time.time() - t0, model_used=self.method)
        if proc.returncode == 0 and out_path.exists():
            return OperatorResult(
                success=True, output_path=str(out_path), cost_usd=0.0,
                raw_response=(proc.stdout or "").strip()[:200],
                duration_sec=time.time() - t0, model_used=self.method)
        err = (proc.stderr or proc.stdout or "unknown").strip()
        # surface the last meaningful traceback line
        tail = err.splitlines()[-1] if err else "no stderr"
        return OperatorResult(success=False, error=tail[:300],
                              duration_sec=time.time() - t0, model_used=self.method)


class InstructPix2PixOperator(_SubprocDiffusionOperator):
    """Instruction-guided edit of the source face (self-contained SD1.5 fine-tune)."""
    method = "instructpix2pix"
    family = "attribute_edit"


class IPAdapterFaceOperator(_SubprocDiffusionOperator):
    """SD1.5 + IP-Adapter plus-face: identity-conditioned re-generation."""
    method = "ipadapter_face"
    family = "id_diff"


class SDXLSynthOperator(_SubprocDiffusionOperator):
    """SDXL text-to-image entire-face synthesis (no identity input)."""
    method = "sdxl_t2i"
    family = "entire_synthesis"
    needs_src = False


class InstantIDOperator(_SubprocDiffusionOperator):
    """SDXL + InstantID: strong tuning-free identity preservation (needs vendored pipeline)."""
    method = "instantid"
    family = "id_diff"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="instructpix2pix")
    ap.add_argument("--src", default="/data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces/0_row0_real.png")
    a = ap.parse_args()
    cls = {"instructpix2pix": InstructPix2PixOperator,
           "ipadapter_face": IPAdapterFaceOperator,
           "sdxl_t2i": SDXLSynthOperator,
           "instantid": InstantIDOperator}[a.method]
    op = cls(out_dir="/tmp/face_attack_outputs")
    print(f"smoke test: {a.method}  src={a.src}")
    r = op.run(src_face_path=a.src, params={"prompt": "a natural photo of a person, warm window light"})
    print(f"  success={r.success}  out={r.output_path}  dur={r.duration_sec:.1f}s")
    if not r.success:
        print(f"  error={r.error}")
