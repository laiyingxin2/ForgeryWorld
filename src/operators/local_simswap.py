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
        # input expected as 1×3×112×112 float32 in [-1, 1] (BGR)
        x = face_112_bgr.astype(np.float32)
        x = (x - 127.5) / 127.5
        x = np.transpose(x, (2, 0, 1))[None]
        emb = self._arc_sess.run(None, {self._arc_sess.get_inputs()[0].name: x})[0]
        return emb.astype(np.float32)

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
            # src 112-crop for ArcFace
            src_112, _ = self._align_face_to_size(src_bgr, src_faces[0].bbox, 112)
            if src_112 is None:
                return OperatorResult(success=False, error="src crop empty",
                                       duration_sec=time.time() - t0, model_used=self.model_id)
            src_emb = self._arcface_embed(src_112)
            # tgt 256-crop
            tgt_256, place = self._align_face_to_size(tgt_bgr, tgt_faces[0].bbox, 256)
            if tgt_256 is None:
                return OperatorResult(success=False, error="tgt crop empty",
                                       duration_sec=time.time() - t0, model_used=self.model_id)
            # SimSwap inference: input is BGR 256 [-1, 1]
            tgt_in = tgt_256.astype(np.float32)
            tgt_in = (tgt_in - 127.5) / 127.5
            tgt_in = np.transpose(tgt_in, (2, 0, 1))[None]
            in0 = self._swap_sess.get_inputs()[0].name
            in1 = self._swap_sess.get_inputs()[1].name
            swapped = self._swap_sess.run(None, {in0: tgt_in, in1: src_emb})[0]
            # back to uint8 BGR
            swap_img = np.clip((swapped[0].transpose(1, 2, 0) * 127.5 + 127.5),
                                0, 255).astype(np.uint8)
            # paste back
            x1, y1, x2, y2 = place
            crop_h, crop_w = (y2 - y1), (x2 - x1)
            swap_resized = cv2.resize(swap_img, (crop_w, crop_h),
                                       interpolation=cv2.INTER_LANCZOS4)
            out = tgt_bgr.copy()
            out[y1:y2, x1:x2] = swap_resized
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
