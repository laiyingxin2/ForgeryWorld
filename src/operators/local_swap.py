"""本地 face-swap operators (替代不可用的 viviai image gen).

InSwapper-128 是 industry-standard face swap (Roop / Deep-Live-Cam 都用):
- insightface 1.0.1 已装 in fakevlm env
- weights: /data/disk4/lyx_ICML/hf_models_lyx/01_face_swap/ezioruan__inswapper_128.onnx/inswapper_128.onnx
- 输入: src face image + tgt face image
- 输出: src 的脸换到 tgt 上

实现:
1. FaceAnalysis 抓 face embedding from src
2. inswapper 把 embedding 应用到 tgt 的 face 区域
3. Blend + 输出
"""
from __future__ import annotations
import os
import time
import uuid
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    from operators.api_image import OperatorResult, ApiImageOperator
except ImportError:
    # standalone test
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from operators.api_image import OperatorResult, ApiImageOperator


_log = logging.getLogger(__name__)

# 默认 ONNX 路径
INSWAPPER_ONNX = "/data/disk4/lyx_ICML/hf_models_lyx/01_face_swap/ezioruan__inswapper_128.onnx/inswapper_128.onnx"
INSIGHTFACE_ROOT = os.path.expanduser("~/.insightface")


class LocalInSwapperOperator(ApiImageOperator):
    """InSwapper-128 本地 face-swap.

    Replaces NanoBananaPro/Two/One (all 503 model_not_found on viviai).
    Generates a real face-swap attack image.
    """
    model_id = "inswapper_128_local"
    family = "frontal_swap"
    cost_per_call = 0.0
    default_size = "1024x1024"

    # lazy-loaded
    _swapper = None
    _face_analyser = None
    _tgt_pool = None  # list of target face image paths to swap onto

    def __init__(self, client=None, out_dir="/tmp/face_attack_outputs",
                 tgt_face_pool: Optional[list] = None,
                 onnx_path: str = INSWAPPER_ONNX):
        # NB: client is unused for local op, kept for ApiImageOperator interface
        self.client = client
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_path = onnx_path
        # default target face pool = our cropped real_faces (swap onto random one)
        self._tgt_pool = tgt_face_pool or sorted(
            (Path("/data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces").glob("*.png"))
        )

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @classmethod
    def _ensure_loaded(cls, onnx_path):
        if cls._swapper is not None: return
        import insightface
        from insightface.app import FaceAnalysis
        # FaceAnalysis for landmark + embedding
        cls._face_analyser = FaceAnalysis(name='buffalo_l',
                                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        cls._face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        # InSwapper model
        cls._swapper = insightface.model_zoo.get_model(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

    def run(self, src_face_path=None, tgt_face_path=None, params=None, size=None):
        """Execute face-swap.

        - src_face_path: source face (whose identity we project)
        - tgt_face_path: target face (whose pose/lighting/composition we keep)
          if None, pick random from self._tgt_pool different from src
        """
        params = params or {}
        t0 = time.time()
        if not src_face_path or not Path(src_face_path).exists():
            return OperatorResult(success=False, error="no src face",
                                   duration_sec=time.time() - t0, model_used=self.model_id)

        # pick tgt different from src
        tgt = tgt_face_path
        if not tgt:
            candidates = [str(p) for p in self._tgt_pool if str(p) != src_face_path]
            if not candidates:
                return OperatorResult(success=False, error="no tgt pool",
                                       duration_sec=time.time() - t0, model_used=self.model_id)
            import random
            tgt = random.choice(candidates)

        try:
            self._ensure_loaded(self.onnx_path)
            import cv2
            src_img = cv2.imread(src_face_path)
            tgt_img = cv2.imread(tgt)
            if src_img is None or tgt_img is None:
                return OperatorResult(success=False, error="cv2 read failed",
                                       duration_sec=time.time() - t0, model_used=self.model_id)

            src_faces = self._face_analyser.get(src_img)
            tgt_faces = self._face_analyser.get(tgt_img)
            if not src_faces or not tgt_faces:
                return OperatorResult(success=False,
                                       error=f"no face detected (src={len(src_faces)} tgt={len(tgt_faces)})",
                                       duration_sec=time.time() - t0, model_used=self.model_id)

            src_face = src_faces[0]
            tgt_face = tgt_faces[0]
            # do swap: project src identity onto tgt
            output = self._swapper.get(tgt_img.copy(), tgt_face, src_face, paste_back=True)

            out_path = self.out_dir / f"{self.name}_{uuid.uuid4().hex[:8]}.png"
            cv2.imwrite(str(out_path), output)
            return OperatorResult(
                success=True,
                output_path=str(out_path),
                cost_usd=0.0,
                raw_response=f"swap src={Path(src_face_path).name} → tgt={Path(tgt).name}",
                duration_sec=time.time() - t0,
                model_used=self.model_id,
            )
        except Exception as e:
            return OperatorResult(
                success=False, error=str(e)[:300],
                duration_sec=time.time() - t0, model_used=self.model_id,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    op = LocalInSwapperOperator()
    src = "/data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces/0_row0_real.png"
    print(f"smoke test: src = {src}")
    r = op.run(src_face_path=src)
    print(f"  success: {r.success}")
    print(f"  output:  {r.output_path}")
    print(f"  duration: {r.duration_sec:.1f}s")
    if not r.success:
        print(f"  error:   {r.error}")
