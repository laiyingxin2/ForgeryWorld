"""FakeVLM judge wrapper — uses the fine-tuned LLaVA-1.5-7B fakevlm checkpoint
at /cpfs01/oss_dataset/lyx/Forgery/fakevlm_check/checkpoints_5epo/llava-1.5-7b-fakevlm/.

Provides:
  • FakeVLMJudge.score(image_path) -> dict
      {'label': 'real'|'fake', 'confidence': float, 'reason': str,
       'evasion_score': float}    # evasion = 1 - confidence_fake (higher = harder to detect)

Lazy loads on first call. Designed for the fakevlm conda env
(torch 2.8 + transformers 4.45).
"""
from __future__ import annotations
import os, re
from pathlib import Path
from typing import Optional, Union
from PIL import Image


JUDGE_PROMPT = (
    "USER: <image>\nYou are a forgery detection expert. Carefully analyze this face image "
    "for manipulation artifacts (seams, lighting inconsistency, unnatural texture, asymmetry, "
    "ID mismatch). Answer strictly in this format:\n"
    "LABEL: [real|fake]\nCONFIDENCE: <0..1>\nREASON: <one short sentence>\nASSISTANT:"
)

DEFAULT_CKPT = "/cpfs01/oss_dataset/lyx/Forgery/fakevlm_check/checkpoints_5epo/llava-1.5-7b-fakevlm"
DEFAULT_BASE = "/cpfs01/bob_workspace/students/lyx/Model_download/llava-hf/llava-1.5-7b-hf"


def _parse_response(text: str) -> dict:
    """Parse 'LABEL: ... CONFIDENCE: ... REASON: ...' or be lenient."""
    label = "fake"  # default conservative
    confidence = 0.5
    reason = ""

    m_label = re.search(r"LABEL\s*[:：]\s*(real|fake)", text, re.I)
    if m_label:
        label = m_label.group(1).lower()

    m_conf = re.search(r"CONFIDENCE\s*[:：]\s*([0-9]*\.?[0-9]+)", text, re.I)
    if m_conf:
        try:
            confidence = float(m_conf.group(1))
            if confidence > 1.0:  # sometimes the model says "85"
                confidence = confidence / 100.0
        except ValueError:
            pass

    m_reason = re.search(r"REASON\s*[:：]\s*(.+?)(?:\n|$)", text, re.I | re.S)
    if m_reason:
        reason = m_reason.group(1).strip()
    else:
        reason = text.strip()[:200]

    # confidence_fake = how sure it's fake
    conf_fake = confidence if label == "fake" else (1.0 - confidence)
    return {
        "label": label,
        "confidence": confidence,
        "confidence_fake": conf_fake,
        "reason": reason,
        "evasion_score": 1.0 - conf_fake,   # ↑ better = harder to detect
        "raw": text,
    }


class FakeVLMJudge:
    def __init__(self,
                 ckpt_path: str = DEFAULT_CKPT,
                 base_model_path: str = DEFAULT_BASE,
                 device: str = "cuda",
                 dtype: str = "bfloat16",
                 lazy: bool = True):
        self.ckpt_path = ckpt_path
        self.base_model_path = base_model_path
        self.device = device
        self.dtype_str = dtype
        self.model = None
        self.processor = None
        if not lazy:
            self._load()

    def _load(self):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[self.dtype_str]

        # processor comes from base (tokenizer + image_processor)
        proc_src = self.base_model_path if Path(self.base_model_path).exists() else self.ckpt_path
        self.processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=True)

        # full safetensors weights at ckpt_path
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.ckpt_path, torch_dtype=dtype, low_cpu_mem_usage=True,
        ).to(self.device).eval()
        print(f"[FakeVLM] loaded from {self.ckpt_path} → {self.device} ({self.dtype_str})")

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def score(self, image: Union[str, Path, Image.Image],
              max_new_tokens: int = 80) -> dict:
        import torch
        if not self.loaded:
            self._load()

        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB") if image.mode != "RGB" else image

        # llava-1.5 expects "USER: <image>\n... ASSISTANT:"
        inputs = self.processor(text=JUDGE_PROMPT, images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=0.0,
                pad_token_id=self.processor.tokenizer.pad_token_id
                              or self.processor.tokenizer.eos_token_id,
            )
        full = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        # strip echoed prompt
        out_text = full.split("ASSISTANT:", 1)[-1].strip()
        return _parse_response(out_text)

    def batch_score(self, image_paths: list, max_new_tokens: int = 80) -> list:
        return [self.score(p, max_new_tokens=max_new_tokens) for p in image_paths]


# ---------------------- smoke test --------------------------------
if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/disk4/lyx_ICML/hf_models_lyx/04_id_preserving/InstantX__InstantID/examples/0.png"

    print(f"[FakeVLM smoke test] testing on {img_path}")
    judge = FakeVLMJudge(lazy=False)  # eager load
    r = judge.score(img_path)
    print(f"\nResult:")
    for k, v in r.items():
        if k != "raw":
            print(f"  {k}: {v}")
    print(f"\nRaw response:\n  {r['raw']}")
