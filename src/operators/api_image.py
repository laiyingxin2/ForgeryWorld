"""Layer 0 attack operators based on viviai image-generation API.

5 个 API-based op:
  - NanoBananaPro    : gemini-3-pro-image-preview     - id_diff 高质量 (2K/4K)
  - NanoBananaTwo    : gemini-3.1-flash-image-preview - id_diff/morph (最新, 2K/4K)
  - NanoBananaOne    : gemini-2.5-flash-image         - 快速 morph/视角
  - GptImageTwo      : gpt-image-2                    - 局部修复 / artifact 覆盖
  - ClaudeVisionPlan : claude-opus-4-7 vision         - 看图选下一个 op (planner, 不输出图)

每个 op 实现统一 BaseOperator.run() 接口, 输出落到磁盘并返回路径.

成本估算 (per call, USD):
  NanoBananaPro    ~$0.06 (按次 1.5x 倍率)
  NanoBananaTwo    ~$0.04
  NanoBananaOne    ~$0.02
  GptImageTwo      ~$0.04
"""
from __future__ import annotations
import os
import base64
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from viviai_client import ViviClient


_log = logging.getLogger(__name__)


# ────────────────────────── Base abstract op ────────────────────────

@dataclass
class OperatorResult:
    """Layer 0 op 统一返回值."""
    success: bool
    output_path: str = ""           # 落盘位置
    cost_usd: float = 0.0
    raw_response: str = ""          # 原始 API 返回 (debug 用)
    duration_sec: float = 0.0
    error: Optional[str] = None
    model_used: str = ""


class ApiImageOperator:
    """Base class for viviai image-gen attack operators.

    Subclasses 只需 define:
      - model_id  : viviai model name
      - family    : attack family in {frontal_swap, profile_swap, id_diff, reenact, morph, ...}
      - cost_per_call : approximate USD
    plus override _build_prompt() if needed.
    """
    model_id: str = ""
    family: str = "id_diff"
    cost_per_call: float = 0.04
    default_size: str = "1024x1024"

    def __init__(
        self,
        client: Optional[ViviClient] = None,
        out_dir: Union[str, Path] = "/tmp/face_attack_outputs",
    ):
        self.client = client or ViviClient()
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def _build_prompt(
        self,
        src_face_path: Optional[str],
        tgt_face_path: Optional[str],
        params: dict,
    ) -> str:
        """Override per-op. Default = generic ID-preserving edit instruction."""
        instr = params.get("instruction") or (
            "Re-render this face preserving the identity, "
            "but change the lighting to warm indoor scene, "
            "remove visible artifacts, output as a natural photo."
        )
        return instr

    def _save_b64_image(self, b64: str, suffix: str = ".png") -> str:
        """Decode b64 → save → return path."""
        path = self.out_dir / f"{self.name}_{uuid.uuid4().hex[:8]}{suffix}"
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        return str(path)

    # fallback chain: 当前 model 失败时降级 (BUG #6 修)
    FALLBACK_CHAIN = {
        "gemini-3-pro-image-preview": "gemini-3.1-flash-image-preview",
        "gemini-3.1-flash-image-preview": "gemini-2.5-flash-image",
        "gemini-2.5-flash-image": None,
        "gpt-image-2": None,
    }

    def run(
        self,
        src_face_path: Optional[str] = None,
        tgt_face_path: Optional[str] = None,
        params: Optional[dict] = None,
        size: Optional[str] = None,
    ) -> OperatorResult:
        """Execute the operator. Returns OperatorResult.

        Note: viviai gen_image API 只接 text prompt + size, 暂不直接接受 image input.
        想要 image-to-image edit 时, 把 src 图编码进 prompt (vision-grounded prompt).
        显式 fallback chain (BUG #6 修): pro → 2 → 1 失败降级.
        """
        params = params or {}
        t0 = time.time()
        prompt = self._build_prompt(src_face_path, tgt_face_path, params)
        size = size or self.default_size

        current_model = self.model_id
        original_model = self.model_id
        last_err = None
        while current_model:
            try:
                results = self.client.gen_image(
                    model=current_model,
                    prompt=prompt,
                    n=1,
                    size=size,
                )
                if results:
                    # 成功! 记录降级 (如果有)
                    if current_model != original_model:
                        _log.warning(f"  fallback: {original_model} → {current_model}")
                    self.model_id = current_model
                    break
                else:
                    raise RuntimeError("empty result")
            except Exception as e:
                last_err = e
                next_model = self.FALLBACK_CHAIN.get(current_model)
                if next_model is None:
                    self.model_id = original_model
                    return OperatorResult(
                        success=False,
                        cost_usd=0.0,
                        duration_sec=time.time() - t0,
                        error=f"All models failed: {str(e)[:150]}",
                        model_used=current_model,
                    )
                current_model = next_model
                _log.info(f"  retrying with fallback {current_model}...")
        self.model_id = original_model
        # 此处 results 应已 set
        try:
            if not results:
                return OperatorResult(
                    success=False,
                    cost_usd=0.0,
                    duration_sec=time.time() - t0,
                    error="No image returned",
                    model_used=current_model,
                )
            first = results[0]
            if first.startswith("http"):
                import requests
                r = requests.get(first, timeout=60)
                r.raise_for_status()
                path = self.out_dir / f"{self.name}_{uuid.uuid4().hex[:8]}.png"
                with open(path, "wb") as f:
                    f.write(r.content)
                output_path = str(path)
            else:
                output_path = self._save_b64_image(first)

            return OperatorResult(
                success=True,
                output_path=output_path,
                cost_usd=self.cost_per_call,
                raw_response=first[:200],
                duration_sec=time.time() - t0,
                model_used=current_model,
            )
        except Exception as e:
            return OperatorResult(
                success=False,
                cost_usd=0.0,
                duration_sec=time.time() - t0,
                error=str(e)[:300],
                model_used=current_model,
            )


# ────────────────────────── Concrete ops ────────────────────────────

class NanoBananaPro(ApiImageOperator):
    """gemini-3-pro-image-preview — highest quality, 2K/4K support."""
    model_id = "gemini-3-pro-image-preview"
    family = "id_diff"
    cost_per_call = 0.06
    default_size = "2048x2048"

    def _build_prompt(self, src_face_path, tgt_face_path, params):
        identity_hint = params.get("identity_hint", "a person with similar features")
        scene = params.get("scene", "natural indoor lighting, slight smile, looking at camera")
        return (
            f"Generate a high-quality photorealistic portrait of {identity_hint}, "
            f"with {scene}. The result must look like an authentic smartphone selfie, "
            f"no digital artifacts, natural skin texture and pores, sharp eyes."
        )


class NanoBananaTwo(ApiImageOperator):
    """gemini-3.1-flash-image-preview — latest, 2K/4K, balanced quality/speed."""
    model_id = "gemini-3.1-flash-image-preview"
    family = "id_diff"
    cost_per_call = 0.04
    default_size = "2048x2048"

    def _build_prompt(self, src_face_path, tgt_face_path, params):
        identity_hint = params.get("identity_hint", "a person")
        action = params.get("action", "looking straight at the camera with neutral expression")
        return (
            f"Photorealistic selfie of {identity_hint}, {action}. "
            "Lighting: soft daylight from window. Camera: phone front camera, slight grain. "
            "Render natural skin imperfections, micro-shadows."
        )


class NanoBananaOne(ApiImageOperator):
    """gemini-2.5-flash-image — cheap, fast morph / 视角."""
    model_id = "gemini-2.5-flash-image"
    family = "morph"
    cost_per_call = 0.02
    default_size = "1024x1024"

    def _build_prompt(self, src_face_path, tgt_face_path, params):
        viewpoint = params.get("viewpoint", "3/4 profile turned slightly left")
        return (
            f"Portrait of a person, {viewpoint}, photorealistic, natural lighting. "
            "Keep the same identity features but change the head pose."
        )


class GptImageTwo(ApiImageOperator):
    """gpt-image-2 — inpainting / artifact masking."""
    model_id = "gpt-image-2"
    family = "restoration"
    cost_per_call = 0.04
    default_size = "1024x1024"

    def _build_prompt(self, src_face_path, tgt_face_path, params):
        instr = params.get(
            "instruction",
            "Restore this face to look natural and authentic. Remove any visible "
            "blending artifacts, smooth jaw edges, normalize skin color across cheeks "
            "and forehead, fix asymmetric eye highlights."
        )
        return instr


class ClaudeVisionPlanner:
    """Not an image gen, but a vision-guided next-op planner.

    Reads current image, suggests which operator to apply next.
    Used as Layer 4 augmentation for the pipeline planner.
    """
    model_id = "claude-opus-4-7"
    family = "planner"
    cost_per_call = 0.015

    def __init__(self, client: Optional[ViviClient] = None):
        self.client = client or ViviClient()

    def suggest_next_op(
        self,
        current_image_path: str,
        attack_family: str,
        available_ops: list,
        tier1_metrics: dict,
    ) -> dict:
        """Returns {next_op, params_hint, reason}."""
        prompt = f"""You are a face forgery red-team planner. Current image is at hand.
Target attack family: {attack_family}
Available operators: {', '.join(available_ops)}
Current Tier-1 metrics: {tier1_metrics}

Suggest the SINGLE most useful next operator + key parameter to apply,
to improve the chance of bypassing a deepfake detector.

Return JSON: {{"next_op": "...", "params_hint": {{...}}, "reason": "one sentence"}}"""
        return self.client.chat_vision_json(
            self.model_id, prompt, current_image_path,
            temperature=0.1, max_tokens=400,
        )


# ────────────────────────── Registry ────────────────────────────────

OPERATOR_REGISTRY = {
    "nano_banana_pro": NanoBananaPro,
    "nano_banana_two": NanoBananaTwo,
    "nano_banana_one": NanoBananaOne,
    "gpt_image_two": GptImageTwo,
}


def list_api_operators() -> list:
    return list(OPERATOR_REGISTRY.keys())


# ────────────────────────── Smoke test ─────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("api_image_smoke")

    api_key = os.environ["VIVIAI_KEY"]  # set VIVIAI_KEY env var; key not committed
    client = ViviClient(api_key=api_key)

    # 测试: 用 NanoBananaOne (最便宜) 生成一张图
    op = NanoBananaOne(client=client, out_dir="/tmp/face_attack_outputs")
    log.info(f"Testing {op.name} ({op.model_id}) ...")
    result = op.run(
        params={"viewpoint": "frontal", "instruction": None},
        size="1024x1024",
    )

    if result.success:
        log.info(f"  ✓ output_path = {result.output_path}")
        log.info(f"  ✓ cost        = ${result.cost_usd:.4f}")
        log.info(f"  ✓ duration    = {result.duration_sec:.1f}s")
        log.info(f"  ✓ model       = {result.model_used}")
        size_kb = Path(result.output_path).stat().st_size / 1024
        log.info(f"  ✓ file size   = {size_kb:.0f} KB")
    else:
        log.error(f"  ✗ failed: {result.error}")

    log.info(f"\nAvailable API ops: {list_api_operators()}")
