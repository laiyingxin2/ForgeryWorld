"""Local SimSwap-256 operator (Chen et al. 2020, MM'20). 2nd local face-swap op
beyond InSwapper-128, proves L0 attack-op diversity.

Pipeline:
  1. Detect src + tgt faces (insightface buffalo_l, reused).
  2. Compute src ArcFace 512-d embedding via simswap_arcface_model.onnx
     (input: 1×3×112×112 BGR, normalized).
  3. Align tgt face crop to 256×256.
  4. Run simswap_256.onnx (input1: tgt_256, input2: src_embed) → swapped_256.
  5. Paste swapped 256-crop back into the original tgt image (face_recon).

Weights live at:
  /data/disk4/lyx_ICML/hf_models_lyx/01_face_swap/netrunner-exe__Insight-Swap-models-onnx/
    simswap_256.onnx
    simswap_arcface_model.onnx
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

SIMSWAP_DIR = "/data/disk4/lyx_ICML/hf_models_lyx/01_face_swap/netrunner-exe__Insight-Swap-models-onnx"
SIMSWAP_ONNX = f"{SIMSWAP_DIR}/simswap_256.onnx"
ARCFACE_ONNX = f"{SIMSWAP_DIR}/simswap_arcface_model.onnx"


class LocalSimSwapOperator(ApiImageOperator):
    """SimSwap-256 (Chen et al. 2020). 2nd local face-swap op."""
    model_id = "simswap_256_local"
    family = "profile_swap"   # SimSwap historically better at off-angle than InSwapper
    cost_per_call = 0.0
    default_size = "1024x1024"

    # class-level lazy caches
    _swap_sess = None
    _arc_sess = None
    _face_app = None
    _tgt_pool: Optional[list] = None

    def __init__(self, client=None,
                 out_dir: str = "/tmp/face_attack_outputs",
                 tgt_face_pool: Optional[list] = None,
                 simswap_onnx: str = SIMSWAP_ONNX,
                 arcface_onnx: str = ARCFACE_ONNX):
        self.client = client
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.simswap_onnx = simswap_onnx
        self.arcface_onnx = arcface_onnx
        self._tgt_pool = tgt_face_pool or sorted(
            Path("/data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces").glob("*.png")
        )

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @classmethod
    def _ensure_loaded(cls, simswap_onnx: str, arcface_onnx: str):
        if cls._swap_sess is not None:
            return
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        cls._swap_sess = ort.InferenceSession(simswap_onnx, providers=providers)
        cls._arc_sess = ort.InferenceSession(arcface_onnx, providers=providers)
        from insightface.app import FaceAnalysis
        cls._face_app = FaceAnalysis(name='buffalo_l',
                                      providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        cls._face_app.prepare(ctx_id=0, det_size=(640, 640))

    @staticmethod
    def _align_face_to_size(img_bgr: np.ndarray, bbox: np.ndarray,
                             out_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Crop bbox region with 25% margin, resize to out_size×out_size.
        Returns (crop_resized_bgr, place_info=(x1,y1,x2,y2,scale))."""
        import cv2
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int).tolist()
        mw = int((x2 - x1) * 0.25)
        mh = int((y2 - y1) * 0.25)
        x1 = max(0, x1 - mw); y1 = max(0, y1 - mh)
        x2 = min(w, x2 + mw); y2 = min(h, y2 + mh)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None
        resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LANCZOS4)
        return resized, np.array([x1, y1, x2, y2])

    def _arcface_embed(self, face_112_bgr: np.ndarray) -> np.ndarray:
        # SimSwap's ArcFace wants 1×3×112×112 float32 in [-1, 1], RGB order.
        x = face_112_bgr[:, :, ::-1].astype(np.float32)  # BGR -> RGB
        x = (x - 127.5) / 127.5
        x = np.transpose(x, (2, 0, 1))[None].copy()
        emb = self._arc_sess.run(None, {self._arc_sess.get_inputs()[0].name: x})[0]
        emb = emb.astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-8)  # unit id vector — SimSwap expects normalized
        return emb

    def run(self, src_face_path: Optional[str] = None,
             tgt_face_path: Optional[str] = None,
             params: Optional[dict] = None, size: Optional[str] = None):
        t0 = time.time()
        params = params or {}
        if not src_face_path or not Path(src_face_path).exists():
            return OperatorResult(success=False, error="no src face",
                                   duration_sec=time.time() - t0, model_used=self.model_id)
        tgt = tgt_face_path
        if not tgt:
            cands = [str(p) for p in self._tgt_pool if str(p) != src_face_path]
            if not cands:
                return OperatorResult(success=False, error="no tgt pool",
                                       duration_sec=time.time() - t0, model_used=self.model_id)
            tgt = random.choice(cands)
        try:
            self._ensure_loaded(self.simswap_onnx, self.arcface_onnx)
            import cv2
            src_bgr = cv2.imread(src_face_path)
            tgt_bgr = cv2.imread(tgt)
            if src_bgr is None or tgt_bgr is None:
                return OperatorResult(success=False, error="cv2 read failed",
                                       duration_sec=time.time() - t0, model_used=self.model_id)
            src_faces = self._face_app.get(src_bgr)
            tgt_faces = self._face_app.get(tgt_bgr)
            if not src_faces or not tgt_faces:
                return OperatorResult(success=False,
                    error=f"no face (src={len(src_faces)} tgt={len(tgt_faces)})",
                    duration_sec=time.time() - t0, model_used=self.model_id)
            # src 112 for ArcFace — use 5-point ALIGNED crop (norm_crop), not a loose
            # bbox resize. A mis-framed crop yields a degraded id embedding and the
            # swap fails to carry the source identity (arc≈0.15).
            from insightface.utils import face_align
            src_112 = face_align.norm_crop(src_bgr, src_faces[0].kps, image_size=112)
            src_emb = self._arcface_embed(src_112)
            # tgt: 5-point ALIGNED 256 crop + the affine M (image -> aligned frame).
            # SimSwap's generator works in this aligned frame; warping its output back
            # with inverse(M) (instead of a loose bbox paste) matches the result to the
            # target's actual face geometry → far stronger identity transfer.
            M = face_align.estimate_norm(tgt_faces[0].kps, image_size=256)
            tgt_256 = cv2.warpAffine(tgt_bgr, M, (256, 256), borderValue=0.0)
            # SimSwap inference: target input is RGB 256, normalized to [-1, 1].
            tgt_in = tgt_256[:, :, ::-1].astype(np.float32)  # BGR -> RGB
            tgt_in = (tgt_in - 127.5) / 127.5
            tgt_in = np.transpose(tgt_in, (2, 0, 1))[None].copy()
            in0 = self._swap_sess.get_inputs()[0].name
            in1 = self._swap_sess.get_inputs()[1].name
            swapped = self._swap_sess.run(None, {in0: tgt_in, in1: src_emb})[0]
            # SimSwap output is RGB in [0, 1] (NOT [-1, 1]); decode accordingly.
            swap_rgb = np.clip(swapped[0].transpose(1, 2, 0) * 255.0,
                               0, 255).astype(np.uint8)
            swap_img = swap_rgb[:, :, ::-1]  # RGB -> BGR for cv2
            # warp the swapped 256 face back into the original image with inverse(M),
            # blended through a feathered elliptical mask (no rectangular seam).
            h, w = tgt_bgr.shape[:2]
            IM = cv2.invertAffineTransform(M)
            warped = cv2.warpAffine(swap_img, IM, (w, h),
                                    borderMode=cv2.BORDER_REPLICATE)
            face_mask = np.zeros((256, 256), np.float32)
            cv2.ellipse(face_mask, (128, 128), (int(128 * 0.86), int(128 * 0.96)),
                        0, 0, 360, 1.0, -1)
            mask = cv2.warpAffine(face_mask, IM, (w, h), borderValue=0.0)
            k = max(3, (min(h, w) // 30) | 1)  # odd kernel
            mask = cv2.GaussianBlur(mask, (k, k), 0)[:, :, None]
            out = (warped.astype(np.float32) * mask +
                   tgt_bgr.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
            out_path = self.out_dir / f"{self.name}_{uuid.uuid4().hex[:8]}.png"
            cv2.imwrite(str(out_path), out)
            return OperatorResult(
                success=True, output_path=str(out_path), cost_usd=0.0,
                raw_response=f"simswap src={Path(src_face_path).name} → tgt={Path(tgt).name}",
                duration_sec=time.time() - t0, model_used=self.model_id,
            )
        except Exception as e:
            return OperatorResult(success=False, error=str(e)[:300],
                                   duration_sec=time.time() - t0, model_used=self.model_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    op = LocalSimSwapOperator()
    src = "/data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces/0_row0_real.png"
    print(f"smoke: src = {src}")
    r = op.run(src_face_path=src)
    print(f"  success: {r.success}")
    print(f"  output:  {r.output_path}")
    print(f"  duration: {r.duration_sec:.1f}s")
    if not r.success:
        print(f"  error:   {r.error}")
