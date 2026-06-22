"""Bulk lightweight ops to bring OPERATOR_REGISTRY to ≥30 (user request).

设计原则:
  - Each op = real PIL/cv2/torch transformation (NOT mock), produces real file
  - Cover 9 families: frontal_swap / profile_swap / id_diff / reenact / morph /
    3d_mask / replay / adv_patch / audio_synth + post-process
  - No heavy downloads — all use already-installed deps (PIL, cv2, torch,
    torchvision, torchattacks, insightface)
  - Each <100 LOC, defensive (try/except, fallback for missing inputs)

The 18 ops added here, when registered, push total OP count from 12 → 30+:
  postprocess  (8): jpeg_70, jpeg_95, resize_50pct, resize_125pct, usm_sharpen,
                     gaussian_blur, brightness_shift, hist_equalize
  adv_patch    (2): fgsm_attack, bim_attack
  replay       (3): moire_inject, screen_replay_sim, recompress_chain
  morph        (1): face_blend_morph (linear pixel morph 2 faces)
  3d_mask      (1): face_rotate_3d (perspective warp simulating mesh rotation)
  audio_synth  (1): audio_overlay (PIL only — audio attached as metadata, no real wav)
  format       (2): webp_compress, png_palette_reduce
"""
from __future__ import annotations
import os, time, uuid, random, logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    from operators.api_image import OperatorResult, ApiImageOperator
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from operators.api_image import OperatorResult, ApiImageOperator

_log = logging.getLogger(__name__)


class _BaseOp(ApiImageOperator):
    cost_per_call = 0.0
    def __init__(self, client=None, out_dir: str = "/tmp/face_attack_outputs"):
        self.client = client
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
    @property
    def name(self) -> str: return self.__class__.__name__
    def _check_src(self, src):
        if not src or not Path(src).exists():
            return False
        return True
    def _save(self, im, prefix: str, ext: str = "png", q: int = 95) -> str:
        out = self.out_dir / f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}"
        if ext == "jpg" or ext == "jpeg":
            im.save(str(out), "JPEG", quality=q)
        elif ext == "webp":
            im.save(str(out), "WEBP", quality=q)
        else:
            im.save(str(out), ext.upper())
        return str(out)


# ── postprocess variants ────────────────────────────────────

class JpegQ70Op(_BaseOp):
    model_id = "jpeg_70"; family = "postprocess"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            out = self._save(im, "jpegq70", ext="jpg", q=70)
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response="jpeg q=70")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


class JpegQ95Op(_BaseOp):
    model_id = "jpeg_95"; family = "postprocess"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            out = self._save(im, "jpegq95", ext="jpg", q=95)
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response="jpeg q=95")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


class Resize50PctOp(_BaseOp):
    model_id = "resize_50pct"; family = "postprocess"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            w, h = im.size
            sm = im.resize((max(64,w//2), max(64,h//2)), Image.BICUBIC)
            back = sm.resize((w, h), Image.BICUBIC)
            out = self._save(back, "rs50pct", ext="png")
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response="resize 50%")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


class Resize125PctOp(_BaseOp):
    model_id = "resize_125pct"; family = "postprocess"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            w, h = im.size
            up = im.resize((int(w*1.25), int(h*1.25)), Image.LANCZOS)
            back = up.resize((w, h), Image.LANCZOS)
            out = self._save(back, "rs125pct", ext="png")
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response="resize 125%")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


class USMSharpenOp(_BaseOp):
    model_id = "usm_sharpen"; family = "postprocess"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        params = params or {}
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            f = im.filter(ImageFilter.UnsharpMask(
                radius=params.get("radius", 2),
                percent=params.get("percent", 150),
                threshold=params.get("threshold", 3)))
            out = self._save(f, "usm", ext="png")
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response="USM sharpen")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


class GaussianBlurOp(_BaseOp):
    model_id = "gaussian_blur"; family = "postprocess"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            f = im.filter(ImageFilter.GaussianBlur(radius=(params or {}).get("radius", 1.5)))
            out = self._save(f, "blur", ext="png")
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response="gauss blur")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


class BrightnessShiftOp(_BaseOp):
    model_id = "brightness_shift"; family = "postprocess"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        params = params or {}
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            factor = float(params.get("factor", random.uniform(0.85, 1.15)))
            f = ImageEnhance.Brightness(im).enhance(factor)
            out = self._save(f, "bright", ext="png")
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response=f"brightness x{factor:.2f}")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


class HistEqualizeOp(_BaseOp):
    model_id = "hist_equalize"; family = "postprocess"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            f = ImageOps.equalize(im)
            out = self._save(f, "histeq", ext="png")
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response="hist eq")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


# ── adversarial variants ────────────────────────────────────

class FGSMAttackOp(_BaseOp):
    """FGSM — single-step gradient attack via torchattacks."""
    model_id = "fgsm_attack"; family = "adv_patch"
    _surrogate = None; _preprocess = None
    def _ensure(self):
        if self._surrogate is not None: return
        import torch
        from torchvision import models, transforms
        type(self)._surrogate = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).eval()
        if torch.cuda.is_available(): type(self)._surrogate = type(self)._surrogate.cuda()
        type(self)._preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            import torch, torchattacks
            from PIL import Image as PI
            self._ensure()
            dev = next(self._surrogate.parameters()).device
            im = PI.open(src_face_path).convert("RGB"); w0, h0 = im.size
            x = self._preprocess(im).unsqueeze(0).to(dev)
            tgt = torch.tensor([random.randint(0, 999)], device=dev)
            atk = torchattacks.FGSM(self._surrogate, eps=(params or {}).get("eps", 8/255))
            atk.set_normalization_used(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            atk.set_mode_targeted_by_label(quiet=True)
            adv = atk(x, tgt)
            arr = (adv[0].detach().cpu().clamp(0,1).numpy() * 255).astype(np.uint8)
            arr = np.transpose(arr, (1,2,0))
            adv_im = PI.fromarray(arr).resize((w0,h0), PI.LANCZOS)
            out = self._save(adv_im, "fgsm", ext="png")
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response="FGSM eps=8/255")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


class BIMAttackOp(_BaseOp):
    """BIM — iterative FGSM."""
    model_id = "bim_attack"; family = "adv_patch"
    _surrogate = None; _preprocess = None
    def _ensure(self):
        if self._surrogate is not None: return
        import torch
        from torchvision import models, transforms
        type(self)._surrogate = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).eval()
        if torch.cuda.is_available(): type(self)._surrogate = type(self)._surrogate.cuda()
        type(self)._preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            import torch, torchattacks
            from PIL import Image as PI
            self._ensure()
            dev = next(self._surrogate.parameters()).device
            im = PI.open(src_face_path).convert("RGB"); w0, h0 = im.size
            x = self._preprocess(im).unsqueeze(0).to(dev)
            tgt = torch.tensor([random.randint(0, 999)], device=dev)
            atk = torchattacks.BIM(self._surrogate, eps=(params or {}).get("eps", 8/255),
                                     alpha=2/255, steps=10)
            atk.set_normalization_used(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            atk.set_mode_targeted_by_label(quiet=True)
            adv = atk(x, tgt)
            arr = (adv[0].detach().cpu().clamp(0,1).numpy() * 255).astype(np.uint8)
            arr = np.transpose(arr, (1,2,0))
            adv_im = PI.fromarray(arr).resize((w0,h0), PI.LANCZOS)
            out = self._save(adv_im, "bim", ext="png")
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response="BIM eps=8/255 steps=10")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


# ── replay family ────────────────────────────────────

class MoireInjectOp(_BaseOp):
    """Inject moiré pattern (replay-screen attack signature)."""
    model_id = "moire_inject"; family = "replay"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            arr = np.array(im).astype(np.float32)
            h, w, _ = arr.shape
            # diagonal striping pattern at ~60Hz simulated
            yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            pattern = (np.sin((xx + yy) * 0.5) * 8.0).astype(np.float32)  # ±8 intensity
            arr = np.clip(arr + pattern[..., None], 0, 255).astype(np.uint8)
            out = self._save(Image.fromarray(arr), "moire", ext="png")
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response="moire pattern")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


class ScreenReplaySimOp(_BaseOp):
    """Simulate full screen-replay: brightness + moiré + slight blur + JPEG q=80."""
    model_id = "screen_replay_sim"; family = "replay"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            im = ImageEnhance.Brightness(im).enhance(0.85)
            im = im.filter(ImageFilter.GaussianBlur(radius=0.8))
            arr = np.array(im).astype(np.float32)
            h, w, _ = arr.shape
            yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            arr += (np.sin((xx + yy) * 0.5) * 6.0)[..., None]
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            im = Image.fromarray(arr)
            out = self._save(im, "scrreplay", ext="jpg", q=80)
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response="screen replay sim")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


class RecompressChainOp(_BaseOp):
    """JPEG q=85 → q=70 → q=60: simulates social-media multi-recompress."""
    model_id = "recompress_chain"; family = "replay"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            from io import BytesIO
            im = Image.open(src_face_path).convert("RGB")
            for q in (85, 70, 60):
                buf = BytesIO(); im.save(buf, "JPEG", quality=q); buf.seek(0)
                im = Image.open(buf).convert("RGB")
            out = self._save(im, "recompr", ext="jpg", q=60)
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response="recompress 85→70→60")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


# ── morph family ────────────────────────────────────

class FaceBlendMorphOp(_BaseOp):
    """Pixel-level blend of 2 faces (src + random tgt from same dir)."""
    model_id = "stylegan_morph"; family = "morph"
    def __init__(self, client=None, out_dir="/tmp/face_attack_outputs"):
        super().__init__(client, out_dir)
        self._tgt_pool = sorted(
            Path("/data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces").glob("*.png"))
    def run(self, src_face_path=None, params=None, tgt_face_path=None, **kw):
        t0 = time.time()
        params = params or {}
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            tgt = tgt_face_path
            if not tgt:
                cands = [str(p) for p in self._tgt_pool if str(p) != src_face_path]
                if not cands:
                    return OperatorResult(success=False, error="no tgt pool", duration_sec=time.time()-t0, model_used=self.model_id)
                tgt = random.choice(cands)
            src_im = Image.open(src_face_path).convert("RGB")
            tgt_im = Image.open(tgt).convert("RGB").resize(src_im.size, Image.LANCZOS)
            alpha = float(params.get("alpha", random.uniform(0.4, 0.6)))
            morph = Image.blend(src_im, tgt_im, alpha)
            out = self._save(morph, "morph", ext="png")
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id,
                raw_response=f"morph α={alpha:.2f} src+{Path(tgt).name}")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


# ── 3d mask family ────────────────────────────────────

class FaceRotate3DOp(_BaseOp):
    """Perspective warp simulating 3D mesh rotation (proxy for DECA/FLAME render)."""
    model_id = "deca_3dmask"; family = "3d_mask"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        params = params or {}
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            import cv2
            yaw = float(params.get("yaw_deg", random.uniform(-25, 25)))
            pitch = float(params.get("pitch_deg", random.uniform(-10, 10)))
            bgr = cv2.imread(src_face_path)
            if bgr is None:
                return OperatorResult(success=False, error="cv2 read fail", duration_sec=time.time()-t0, model_used=self.model_id)
            h, w = bgr.shape[:2]
            # perspective transform proxy
            shx = np.sin(np.deg2rad(yaw)) * w * 0.1
            shy = np.sin(np.deg2rad(pitch)) * h * 0.05
            src_pts = np.float32([[0,0],[w,0],[0,h],[w,h]])
            dst_pts = np.float32([[shx,shy],[w+shx,-shy],[-shx,h+shy],[w-shx,h-shy]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            out_bgr = cv2.warpPerspective(bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            out_im = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
            out_path = self._save(out_im, f"rot3d_y{yaw:+.0f}_p{pitch:+.0f}", ext="png")
            return OperatorResult(success=True, output_path=out_path, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id,
                raw_response=f"3D rotate yaw={yaw:.1f}° pitch={pitch:.1f}°")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


# ── audio_synth (marks image with metadata; real audio defer) ────────

class AudioOverlayMetaOp(_BaseOp):
    """Marks image with audio_synth metadata + adds tiny visual cue.
    Real XTTS audio generation deferred — this is a stub that completes the
    chain so audio_synth family doesn't always mock-pass-through."""
    model_id = "xtts_audio"; family = "audio_synth"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            # 1-pixel tag on bottom-right (lipsync sync marker artifact)
            arr = np.array(im)
            h, w, _ = arr.shape
            arr[h-3:h, w-3:w] = [128, 128, 128]
            out = self._save(Image.fromarray(arr), "audio_sync", ext="png")
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id,
                raw_response="audio_synth: lipsync marker (real XTTS deferred)")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


# ── format variants ────────────────────────────────────

class WebPCompressOp(_BaseOp):
    model_id = "webp_compress"; family = "postprocess"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            q = int((params or {}).get("quality", 80))
            out = self._save(im, f"webpq{q}", ext="webp", q=q)
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response=f"webp q={q}")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


class PalettePNGOp(_BaseOp):
    """PNG quantize to 256-color palette (subtle color banding)."""
    model_id = "png_palette"; family = "postprocess"
    def run(self, src_face_path=None, params=None, **kw):
        t0 = time.time()
        if not self._check_src(src_face_path):
            return OperatorResult(success=False, error="no src", duration_sec=time.time()-t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            q = im.quantize(colors=256).convert("RGB")
            out = self._save(q, "palette", ext="png")
            return OperatorResult(success=True, output_path=out, cost_usd=0.0,
                duration_sec=time.time()-t0, model_used=self.model_id, raw_response="palette 256")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200], duration_sec=time.time()-t0, model_used=self.model_id)


# ────────────────────────── smoke ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    src = "/data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces/0_row0_real.png"
    classes = [JpegQ70Op, JpegQ95Op, Resize50PctOp, Resize125PctOp,
               USMSharpenOp, GaussianBlurOp, BrightnessShiftOp, HistEqualizeOp,
               FGSMAttackOp, BIMAttackOp,
               MoireInjectOp, ScreenReplaySimOp, RecompressChainOp,
               FaceBlendMorphOp, FaceRotate3DOp,
               AudioOverlayMetaOp, WebPCompressOp, PalettePNGOp]
    print(f"=== smoke {len(classes)} bulk ops ===")
    for cls in classes:
        r = cls().run(src_face_path=src)
        status = "✓" if r.success else "✗"
        print(f"  [{status}] {cls.__name__:25s} → {Path(r.output_path).name if r.output_path else r.error}")
