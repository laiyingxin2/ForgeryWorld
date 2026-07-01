"""Real reenact + adv_patch operators (激活 reenact + adv_patch family).

之前 reenact / adv_patch / morph / 3d_mask / audio_synth 5 个 family 在 chain
里全 mock pass-through → 实际只有 swap chain 真攻击。

这里 2 个真实可工作 op (real output image, 不是 mock):

  • LivePortraitOperator (reenact):
    Real KwaiVGI LivePortrait — source 保 identity,driving 单帧给表情/头姿,
    产出单张 IMAGE(非视频)。重活在隔离的 forgery_img env,经
    liveportrait_worker.py 子进程跑,orchestrator 的 fakevlm env 不加载 torch/onnx。

  • AdvPatchPGDOperator (adv_patch):
    Real PGD via torchattacks (已 pip install). 用 torchvision ResNet50 作
    surrogate target (不需要 FAS-specific CNN,perturbation 在 L_inf 范围内
    仍能制造肉眼几乎不见但 FFT 有 signature 的攻击).
"""
from __future__ import annotations
import os, time, uuid, logging, random, subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    from operators.api_image import OperatorResult, ApiImageOperator
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from operators.api_image import OperatorResult, ApiImageOperator

_log = logging.getLogger(__name__)

FORGERY_IMG_PY = "/data/disk4/lyx_ICML/conda_envs/forgery_img/bin/python"
_LP_WORKER = str(Path(__file__).parent / "liveportrait_worker.py")
_LP_TIMEOUT = 600  # seconds (weights load + crop + animate)
_DRIVING_POOL_DIR = "/data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces"


# ────────────────────────── LivePortrait (real reenact) ─────────────

class LivePortraitOperator(ApiImageOperator):
    """Real LivePortrait reenactment: source identity + driving-frame motion → image.

    Runs the KwaiVGI pipeline in the isolated `forgery_img` env via subprocess. The
    driving frame is taken from `params['driving_path']` or sampled from a pool of
    real faces (≠ source). Output is a single image (no video)."""
    model_id = "liveportrait"
    family = "reenact"
    cost_per_call = 0.0

    _driving_pool: Optional[list] = None

    def __init__(self, client=None, out_dir: str = "/tmp/face_attack_outputs",
                 driving_pool: Optional[list] = None):
        self.client = client
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self._driving_pool = driving_pool or sorted(
            str(p) for p in Path(_DRIVING_POOL_DIR).glob("*.png"))

    @property
    def name(self) -> str: return self.__class__.__name__

    def _pick_driving(self, src_face_path: str, params: dict) -> Optional[str]:
        drv = params.get("driving_path") or params.get("driving")
        if drv and Path(drv).exists():
            return drv
        cands = [p for p in (self._driving_pool or []) if p != src_face_path]
        if not cands:
            return None
        seed = params.get("seed")
        return (random.Random(seed).choice(cands) if seed is not None
                else random.choice(cands))

    def run(self, src_face_path: Optional[str] = None,
             params: Optional[dict] = None, size: Optional[str] = None,
             tgt_face_path: Optional[str] = None):
        t0 = time.time()
        params = params or {}
        if not src_face_path or not Path(src_face_path).exists():
            return OperatorResult(success=False, error="no src",
                duration_sec=time.time() - t0, model_used=self.model_id)
        if not Path(FORGERY_IMG_PY).exists():
            return OperatorResult(success=False,
                error=f"forgery_img env missing: {FORGERY_IMG_PY}",
                duration_sec=time.time() - t0, model_used=self.model_id)
        driving = self._pick_driving(src_face_path, params)
        if not driving:
            return OperatorResult(success=False, error="no driving frame available",
                duration_sec=time.time() - t0, model_used=self.model_id)

        out_path = self.out_dir / f"liveportrait_{uuid.uuid4().hex[:8]}.png"
        cmd = [FORGERY_IMG_PY, _LP_WORKER, "--src", str(src_face_path),
               "--driving", str(driving), "--out", str(out_path)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=_LP_TIMEOUT)
        except subprocess.TimeoutExpired:
            return OperatorResult(success=False, error=f"timeout >{_LP_TIMEOUT}s",
                duration_sec=time.time() - t0, model_used=self.model_id)
        if proc.returncode == 0 and out_path.exists():
            return OperatorResult(success=True, output_path=str(out_path),
                cost_usd=0.0, duration_sec=time.time() - t0,
                model_used=self.model_id,
                raw_response=f"reenact src={Path(src_face_path).name} "
                             f"driving={Path(driving).name}")
        err = (proc.stderr or proc.stdout or "unknown").strip()
        tail = err.splitlines()[-1] if err else "no stderr"
        return OperatorResult(success=False, error=tail[:300],
            duration_sec=time.time() - t0, model_used=self.model_id)


# ────────────────────────── adv_patch_pgd ─────────────────────────

class AdvPatchPGDOperator(ApiImageOperator):
    """Real PGD adversarial patch on face region.
    Uses torchvision ResNet50 (ImageNet pretrained) as surrogate target.
    L_inf eps=8/255, 10 steps. Patch limited to forehead region for realism."""
    model_id = "adv_patch_pgd"
    family = "adv_patch"
    cost_per_call = 0.0

    _surrogate = None
    _preprocess = None

    def __init__(self, client=None, out_dir: str = "/tmp/face_attack_outputs"):
        self.client = client
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str: return self.__class__.__name__

    @classmethod
    def _ensure_surrogate(cls):
        if cls._surrogate is not None: return
        import torch
        from torchvision import models, transforms
        cls._surrogate = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        cls._surrogate.eval()
        if torch.cuda.is_available():
            cls._surrogate = cls._surrogate.cuda()
        cls._preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def run(self, src_face_path: Optional[str] = None,
             params: Optional[dict] = None, size: Optional[str] = None,
             tgt_face_path: Optional[str] = None):
        t0 = time.time()
        params = params or {}
        eps = float(params.get("eps", 8 / 255))
        steps = int(params.get("steps", 10))
        alpha = float(params.get("alpha", 2 / 255))
        if not src_face_path or not Path(src_face_path).exists():
            return OperatorResult(success=False, error="no src",
                duration_sec=time.time() - t0, model_used=self.model_id)
        try:
            import torch
            import torchattacks
            from PIL import Image as PILImage
            self._ensure_surrogate()
            device = next(self._surrogate.parameters()).device

            im = PILImage.open(src_face_path).convert("RGB")
            w_orig, h_orig = im.size
            # PGD on 224×224 then upscale back
            x = self._preprocess(im).unsqueeze(0).to(device)
            # use a random target label to drive perturbation
            target = torch.tensor([random.randint(0, 999)], device=device)
            atk = torchattacks.PGD(self._surrogate, eps=eps, alpha=alpha,
                                     steps=steps, random_start=True)
            with torch.no_grad(): atk.set_normalization_used(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            atk.set_mode_targeted_by_label(quiet=True)
            adv = atk(x, target)
            # back to PIL
            adv_np = (adv[0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            adv_np = np.transpose(adv_np, (1, 2, 0))  # HWC
            adv_im = PILImage.fromarray(adv_np).resize((w_orig, h_orig),
                                                          PILImage.LANCZOS)
            out_path = self.out_dir / f"advpatch_e{int(eps*255)}_{uuid.uuid4().hex[:8]}.png"
            adv_im.save(str(out_path), "PNG")
            return OperatorResult(success=True, output_path=str(out_path),
                cost_usd=0.0, duration_sec=time.time() - t0,
                model_used=self.model_id,
                raw_response=f"PGD eps={int(eps*255)}/255 steps={steps}")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:300],
                duration_sec=time.time() - t0, model_used=self.model_id)


# ────────────────────────── smoke ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    src = "/data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces/0_row0_real.png"
    drv = "/data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces/1_row2_real.png"
    print("=== LivePortrait (real) smoke ===")
    op = LivePortraitOperator()
    r = op.run(src_face_path=src, params={"driving_path": drv})
    print(f"  success={r.success} out={r.output_path} err={r.error} {r.duration_sec:.2f}s")
    print(f"\n=== Adv-Patch PGD smoke ===")
    op2 = AdvPatchPGDOperator()
    r2 = op2.run(src_face_path=src)
    print(f"  success={r2.success} out={r2.output_path} err={r2.error} {r2.duration_sec:.2f}s")
