"""Local image post-processing operators (face_align, jpeg_85, resize_bicubic).

Critical bug fix (2026-06-20): these op names were in operator_list everywhere
but NOT in OPERATOR_REGISTRY → chain steps mock-pass-through → only 1-2 of 5
chain steps actually ran → tier2 gemini one-shot caught the unmasked face-swap.

These are simple PIL/cv2 functions, no LLM call needed. Register as real ops
so chain "face_align → inswapper → gfpgan → jpeg_85 → resize" actually runs
all 5 steps.

After this fix all 4 methods immediately get more "fully processed" attack
images that have post-processing artifact masking.
"""
from __future__ import annotations
import os, time, uuid, logging
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


class FaceAlignOperator(ApiImageOperator):
    """Affine align: detect face, crop with margin, resize to 512×512.
    Reuses sandbox's insightface FaceAnalysis (already loaded by Tier-1)."""
    model_id = "face_align"
    family = "preprocess"
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

    @staticmethod
    def _square_resize(img, size, border_frac=0.0):
        """Letterbox-pad to square (replicate border) then resize — no aspect
        stretch. border_frac adds extra context so the face stays detectable."""
        import cv2
        h, w = img.shape[:2]
        side = max(h, w)
        bd = int(side * border_frac)
        top = bd + (side - h) // 2; bot = bd + (side - h + 1) // 2
        left = bd + (side - w) // 2; right = bd + (side - w + 1) // 2
        padded = cv2.copyMakeBorder(img, top, bot, left, right, cv2.BORDER_REPLICATE)
        return cv2.resize(padded, (size, size), interpolation=cv2.INTER_LANCZOS4)

    def run(self, src_face_path: Optional[str] = None,
             params: Optional[dict] = None, size: Optional[str] = None,
             tgt_face_path: Optional[str] = None):
        t0 = time.time()
        params = params or {}
        target_size = int(params.get("target_size", 512))
        margin = float(params.get("margin", 0.25))
        if not src_face_path or not Path(src_face_path).exists():
            return OperatorResult(success=False, error="no src",
                duration_sec=time.time() - t0, model_used=self.model_id)
        try:
            self._ensure_face()
            import cv2
            bgr = cv2.imread(src_face_path)
            if bgr is None:
                return OperatorResult(success=False, error="cv2 read fail",
                    duration_sec=time.time() - t0, model_used=self.model_id)
            faces = self._face_app.get(bgr)
            if not faces:
                # no face: letterbox-pad whole image to square (no stretch) then resize
                out = self._square_resize(bgr, target_size)
            else:
                f = faces[0]
                x1, y1, x2, y2 = f.bbox.astype(int).tolist()
                h, w = bgr.shape[:2]
                mw = int((x2 - x1) * margin); mh = int((y2 - y1) * margin)
                x1 = max(0, x1 - mw); y1 = max(0, y1 - mh)
                x2 = min(w, x2 + mw); y2 = min(h, y2 + mh)
                crop = bgr[y1:y2, x1:x2]
                # ★ FIX: do NOT stretch the (non-square) crop to a square — that
                # distorts the face until insightface can no longer detect it,
                # poisoning every downstream swap step. Instead pad to square with
                # a replicate border (keeps aspect + leaves detector context),
                # then resize. Verified to restore detection (score ~0.89).
                out = self._square_resize(crop, target_size, border_frac=0.35)
            out_path = self.out_dir / f"{self.name}_{uuid.uuid4().hex[:8]}.png"
            cv2.imwrite(str(out_path), out)
            return OperatorResult(success=True, output_path=str(out_path),
                cost_usd=0.0, duration_sec=time.time() - t0,
                model_used=self.model_id,
                raw_response=f"aligned to {target_size}x{target_size}")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200],
                duration_sec=time.time() - t0, model_used=self.model_id)


class JpegCompressOperator(ApiImageOperator):
    """JPEG re-encode at quality q (default 85). Masks high-freq artifacts."""
    model_id = "jpeg_85"
    family = "postprocess"
    cost_per_call = 0.0

    def __init__(self, client=None, out_dir: str = "/tmp/face_attack_outputs"):
        self.client = client
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str: return self.__class__.__name__

    def run(self, src_face_path: Optional[str] = None,
             params: Optional[dict] = None, size: Optional[str] = None,
             tgt_face_path: Optional[str] = None):
        t0 = time.time()
        params = params or {}
        q = int(params.get("quality", 85))
        if not src_face_path or not Path(src_face_path).exists():
            return OperatorResult(success=False, error="no src",
                duration_sec=time.time() - t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            out_path = self.out_dir / f"jpeg_q{q}_{uuid.uuid4().hex[:8]}.jpg"
            im.save(str(out_path), "JPEG", quality=q, optimize=True)
            return OperatorResult(success=True, output_path=str(out_path),
                cost_usd=0.0, duration_sec=time.time() - t0,
                model_used=self.model_id, raw_response=f"jpeg q={q}")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200],
                duration_sec=time.time() - t0, model_used=self.model_id)


class ResizeBicubicOperator(ApiImageOperator):
    """Bicubic resize (scale or fixed size). Adds sensor-like resampling."""
    model_id = "resize_bicubic"
    family = "postprocess"
    cost_per_call = 0.0

    def __init__(self, client=None, out_dir: str = "/tmp/face_attack_outputs"):
        self.client = client
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str: return self.__class__.__name__

    def run(self, src_face_path: Optional[str] = None,
             params: Optional[dict] = None, size: Optional[str] = None,
             tgt_face_path: Optional[str] = None):
        t0 = time.time()
        params = params or {}
        scale = float(params.get("scale", 0.9))
        if not src_face_path or not Path(src_face_path).exists():
            return OperatorResult(success=False, error="no src",
                duration_sec=time.time() - t0, model_used=self.model_id)
        try:
            im = Image.open(src_face_path).convert("RGB")
            w, h = im.size
            new_w, new_h = max(64, int(w * scale)), max(64, int(h * scale))
            # downsample then upsample (real-world image-laundering pattern)
            small = im.resize((new_w, new_h), Image.BICUBIC)
            back = small.resize((w, h), Image.BICUBIC)
            out_path = self.out_dir / f"resize_s{scale:.2f}_{uuid.uuid4().hex[:8]}.png"
            back.save(str(out_path), "PNG")
            return OperatorResult(success=True, output_path=str(out_path),
                cost_usd=0.0, duration_sec=time.time() - t0,
                model_used=self.model_id, raw_response=f"resize x{scale}")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200],
                duration_sec=time.time() - t0, model_used=self.model_id)


class GFPGANRestoreOperator(ApiImageOperator):
    """Face restoration via GFPGAN-style Gaussian-blur+sharpen approximation.
    Real GFPGAN weights are ~340MB; for this 'masking' purpose a sharpen+
    bilateral filter is sufficient and cost-0."""
    model_id = "gfpgan"
    family = "postprocess"
    cost_per_call = 0.0

    def __init__(self, client=None, out_dir: str = "/tmp/face_attack_outputs"):
        self.client = client
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str: return self.__class__.__name__

    def run(self, src_face_path: Optional[str] = None,
             params: Optional[dict] = None, size: Optional[str] = None,
             tgt_face_path: Optional[str] = None):
        t0 = time.time()
        params = params or {}
        strength = float(params.get("strength", 0.5))
        if not src_face_path or not Path(src_face_path).exists():
            return OperatorResult(success=False, error="no src",
                duration_sec=time.time() - t0, model_used=self.model_id)
        try:
            import cv2
            bgr = cv2.imread(src_face_path)
            if bgr is None:
                return OperatorResult(success=False, error="cv2 read fail",
                    duration_sec=time.time() - t0, model_used=self.model_id)
            # bilateral (smooth skin keeping edges) + unsharp mask (sharpen detail)
            smooth = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
            gauss = cv2.GaussianBlur(bgr, (0, 0), sigmaX=3)
            sharp = cv2.addWeighted(bgr, 1 + strength, gauss, -strength, 0)
            # blend: bilateral on skin areas + sharp on details
            out = cv2.addWeighted(smooth, 0.5, sharp, 0.5, 0)
            out_path = self.out_dir / f"gfpgan_s{strength:.1f}_{uuid.uuid4().hex[:8]}.png"
            cv2.imwrite(str(out_path), out)
            return OperatorResult(success=True, output_path=str(out_path),
                cost_usd=0.0, duration_sec=time.time() - t0,
                model_used=self.model_id, raw_response=f"gfpgan-proxy s={strength}")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200],
                duration_sec=time.time() - t0, model_used=self.model_id)


class LightingOperator(ApiImageOperator):
    """Photometric relight: imposes a lighting condition on an already-forged
    image so swap/reenact families (which have no text-prompt path) can still
    realize the grid's `lighting` axis.

    mode:
      dark        gamma-darken + global brightness drop  (mean ~179 → ~95)
      strong      brighten + mild highlight clip          (overexposed look)
      back_light  bright rim + darkened face center       (silhouette / haloed)
      normal      identity pass-through (axis already satisfied)
    """
    model_id = "relight"
    family = "postprocess"
    cost_per_call = 0.0

    def __init__(self, client=None, out_dir: str = "/tmp/face_attack_outputs"):
        self.client = client
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str: return self.__class__.__name__

    @staticmethod
    def _gamma(bgr: np.ndarray, g: float) -> np.ndarray:
        inv = 1.0 / max(1e-3, g)
        lut = np.clip(((np.arange(256) / 255.0) ** inv) * 255.0, 0, 255).astype(np.uint8)
        import cv2
        return cv2.LUT(bgr, lut)

    @staticmethod
    def _radial_mask(h: int, w: int, invert: bool = False) -> np.ndarray:
        """Center-bright radial falloff in [0,1]; invert → center-dark."""
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h / 2.0, w / 2.0
        d = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
        m = np.clip(1.0 - d / 1.4142, 0.0, 1.0)
        return (1.0 - m) if invert else m

    def run(self, src_face_path: Optional[str] = None,
             params: Optional[dict] = None, size: Optional[str] = None,
             tgt_face_path: Optional[str] = None):
        t0 = time.time()
        params = params or {}
        mode = str(params.get("mode", "dark")).lower()
        if not src_face_path or not Path(src_face_path).exists():
            return OperatorResult(success=False, error="no src",
                duration_sec=time.time() - t0, model_used=self.model_id)
        try:
            import cv2
            bgr = cv2.imread(src_face_path)
            if bgr is None:
                return OperatorResult(success=False, error="cv2 read fail",
                    duration_sec=time.time() - t0, model_used=self.model_id)
            f = bgr.astype(np.float32)
            h, w = bgr.shape[:2]
            if mode in ("dark", "low_light", "night"):
                out = self._gamma(bgr, 0.45)              # crush mid/shadows
                out = np.clip(out.astype(np.float32) * 0.72, 0, 255).astype(np.uint8)
            elif mode in ("strong", "bright", "harsh"):
                out = self._gamma(bgr, 1.7)               # lift toward highlights
                out = np.clip(out.astype(np.float32) * 1.18, 0, 255).astype(np.uint8)
            elif mode in ("back_light", "backlight", "rim"):
                rim = self._radial_mask(h, w, invert=True)[:, :, None]   # bright edges
                bright = np.clip(f * (1.0 + 0.9 * rim), 0, 255)
                center_dark = self._radial_mask(h, w)[:, :, None]        # dark center
                out = np.clip(bright * (1.0 - 0.45 * center_dark), 0, 255).astype(np.uint8)
            else:  # normal / unknown → identity
                out = bgr
            out_path = self.out_dir / f"relight_{mode}_{uuid.uuid4().hex[:8]}.png"
            cv2.imwrite(str(out_path), out)
            return OperatorResult(success=True, output_path=str(out_path),
                cost_usd=0.0, duration_sec=time.time() - t0,
                model_used=self.model_id, raw_response=f"relight {mode}")
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:200],
                duration_sec=time.time() - t0, model_used=self.model_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    src = "/data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces/0_row0_real.png"
    print("=== local postprocess ops smoke ===")
    for cls in (FaceAlignOperator, JpegCompressOperator, ResizeBicubicOperator,
                 GFPGANRestoreOperator):
        op = cls()
        r = op.run(src_face_path=src)
        print(f"  {cls.__name__}: success={r.success} out={r.output_path} "
              f"err={r.error} {r.duration_sec:.2f}s")
