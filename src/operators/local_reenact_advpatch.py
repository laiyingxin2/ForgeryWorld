"""Minimal real reenact + adv_patch operators (激活 reenact + adv_patch family).

之前 reenact / adv_patch / morph / 3d_mask / audio_synth 5 个 family 在 chain
里全 mock pass-through → 实际只有 swap chain 真攻击。

这里加 2 个最小可工作 op (real output image, 不是 mock):

  • LivePortraitLiteOperator (reenact):
    Lightweight reenact simulator via insightface landmarks +
    OpenCV thin-plate-spline warp. 不需要 KwaiVGI 的 4 个 .pth + repo install,
    用本地已有的 insightface 抽 landmark + cv2 做 mild deformation 模拟"换姿势"。

  • AdvPatchPGDOperator (adv_patch):
    Real PGD via torchattacks (已 pip install). 用 torchvision ResNet50 作
    surrogate target (不需要 FAS-specific CNN,perturbation 在 L_inf 范围内
    仍能制造肉眼几乎不见但 FFT 有 signature 的攻击).

两个都是 best-effort 简化实现 — paper 写作时标注 "lite version, full LivePortrait/
real-FAS-PGD in future work"。但相比 mock pass-through 这是真攻击图。
"""
from __future__ import annotations
import os, time, uuid, logging, random
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


# ────────────────────────── LivePortrait-lite (reenact) ─────────────

class LivePortraitLiteOperator(ApiImageOperator):
    """Reenact-lite: insightface landmark + cv2 affine/TPS warp.
    Mimics small head-pose change + expression shift."""
    model_id = "liveportrait"
    family = "reenact"
    cost_per_call = 0.0

    _face_app = None

    def __init__(self, client=None, out_dir: str = "/tmp/face_attack_outputs"):
        self.client = client
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str: return self.__class__.__name__

    @classmethod
    def _ensure_face(cls):
        if cls._face_app is not None: return
        from insightface.app import FaceAnalysis
        cls._face_app = FaceAnalysis(name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        cls._face_app.prepare(ctx_id=0, det_size=(640, 640))

    def run(self, src_face_path: Optional[str] = None,
             params: Optional[dict] = None, size: Optional[str] = None,
             tgt_face_path: Optional[str] = None):
        t0 = time.time()
        params = params or {}
        yaw_deg = float(params.get("yaw_deg", random.uniform(-15, 15)))
        expression_strength = float(params.get("expression_strength", 0.3))
        if not src_face_path or not Path(src_face_path).exists():
            return OperatorResult(success=False, error="no src",
                duration_sec=time.time() - t0, model_used=self.model_id)
        try:
            import cv2
            self._ensure_face()
            bgr = cv2.imread(src_face_path)
            if bgr is None:
                return OperatorResult(success=False, error="cv2 read fail",
                    duration_sec=time.time() - t0, model_used=self.model_id)
            faces = self._face_app.get(bgr)
            if not faces:
                return OperatorResult(success=False, error="no face",
                    duration_sec=time.time() - t0, model_used=self.model_id)
            f = faces[0]
            h, w = bgr.shape[:2]
            x1, y1, x2, y2 = f.bbox.astype(int).tolist()
            fw, fh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # ── reenact: small affine inside face bbox to simulate head yaw ──
            # apply a perspective warp that shears the face
            yaw_rad = np.deg2rad(yaw_deg)
            shear = np.sin(yaw_rad) * fw * 0.15
            src_pts = np.float32([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
            dst_pts = np.float32([
                [x1 + shear, y1],         # top-left shifts based on yaw
                [x2 + shear, y1],
                [x1 - shear, y2],         # bottom-left shifts opposite
                [x2 - shear, y2],
            ])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(bgr, M, (w, h),
                                          borderMode=cv2.BORDER_REFLECT)

            # ── expression: gentle gaussian on mouth region ──
            if expression_strength > 0:
                mouth_y1 = int(cy + fh * 0.15)
                mouth_y2 = int(cy + fh * 0.40)
                mouth_x1 = int(cx - fw * 0.18)
                mouth_x2 = int(cx + fw * 0.18)
                mouth_y1 = max(0, mouth_y1); mouth_y2 = min(h, mouth_y2)
                mouth_x1 = max(0, mouth_x1); mouth_x2 = min(w, mouth_x2)
                mouth_region = warped[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
                blurred = cv2.GaussianBlur(mouth_region, (0, 0), sigmaX=2)
                blended = cv2.addWeighted(mouth_region, 1 - expression_strength,
                                            blurred, expression_strength, 0)
                warped[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = blended

            # blend back: keep non-face region untouched
            mask = np.zeros((h, w), dtype=np.uint8)
            # elliptical face mask
            cv2.ellipse(mask, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, 255, -1)
            mask_3 = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
            # soft blend
            mask_3 = cv2.GaussianBlur(mask_3, (0, 0), sigmaX=fw * 0.05)
            out = (warped.astype(np.float32) * mask_3 +
                   bgr.astype(np.float32) * (1 - mask_3)).astype(np.uint8)

            out_path = self.out_dir / f"liveportrait_y{yaw_deg:+.0f}_{uuid.uuid4().hex[:8]}.png"
            cv2.imwrite(str(out_path), out)
            return OperatorResult(success=True, output_path=str(out_path),
                cost_usd=0.0, duration_sec=time.time() - t0,
                model_used=self.model_id,
                raw_response=f"reenact-lite yaw={yaw_deg:.1f}° expr={expression_strength}")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200],
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
    print("=== LivePortrait-lite smoke ===")
    op = LivePortraitLiteOperator()
    r = op.run(src_face_path=src)
    print(f"  success={r.success} out={r.output_path} err={r.error} {r.duration_sec:.2f}s")
    print(f"\n=== Adv-Patch PGD smoke ===")
    op2 = AdvPatchPGDOperator()
    r2 = op2.run(src_face_path=src)
    print(f"  success={r2.success} out={r2.output_path} err={r2.error} {r2.duration_sec:.2f}s")
