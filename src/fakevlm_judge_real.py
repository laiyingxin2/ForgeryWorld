"""FakeVLM real-weight judge for sandbox Tier-2.

★ 用户 confirm: 用现成 FakeVLM (LLaVA-1.5-7B fine-tuned on deepfake tasks)
   权重: /cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/checkpoints/multi_20260329_132526_llava-1.5-7b

替代 sandbox.py 的 tier2_llm_judge(viviai gemini) — 用本地真权重做 detector.
论文 framing: "我们用 FakeVLM (PR'24 SOTA deepfake detector) 作 Tier-2 不是 LLM API,这是公平比较".

启动方式 2 选 1:
  (A) vLLM server: 见 FakeVLM/FakeVLM_Official/scripts/eval_vllm.py
      bash deploy_fakevlm_vllm.sh → openai-compatible localhost:8000
  (B) Direct PyTorch inference: 见 FakeVLM/FakeVLM_Official/scripts/eval.py
      慢但无需起服务

本文件实现 (A) wrapper. 一个 vLLM server start 后, 直接当 viviai 调用.
"""
from __future__ import annotations
import os
import logging
import base64
import json
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

import requests

_log = logging.getLogger(__name__)


# ────────────────────────── Config ────────────────────────────────

@dataclass
class FakeVLMJudgeConfig:
    # The VALIDATED faithful checkpoint (95% bal-acc on gold). The old default
    # `multi_20260329…` ckpt was a broken multi-task retrain that emitted captions /
    # bare "Real" for every image — it produced 100% of the spurious historical
    # bypasses. Do NOT point this back at the multi_ ckpt.
    ckpt_path: str = "/data/disk4/lyx_ICML/self_evolution_forgery/scripts/fakevlm_correct_ckpt"
    # vLLM server endpoint — the stacked GPU7 server that serves fakevlm_correct_ckpt
    vllm_endpoint: str = "http://localhost:8001/v1"
    vllm_api_key: str = "EMPTY"
    # 推理参数
    temperature: float = 0.1
    max_tokens: int = 96   # verdict is in the first 1-2 sentences; keep low for speed
    timeout: int = 60
    lora_model_name: Optional[str] = None  # if set, override base ckpt name (e.g. "defender")


# ────────────────────────── Judge ─────────────────────────────────

# Matches FakeVLM training distribution exactly (fakeclue format). The model is
# SFT'd to answer "This is a {fake|real} image. <artifact reasoning>".
# NOTE: do NOT prepend a literal "<image>" — the vLLM chat API injects the image
# token from the image_url content; a second literal token yields empty output.
_PROMPT_TEMPLATE = """Does the image looks real/fake?"""


class FakeVLMJudge:
    """Drop-in replacement for sandbox.tier2_llm_judge using local FakeVLM weights.

    Same return format as sandbox.tier2_llm_judge:
      {model, is_fake, confidence, attack_family_guess, reasoning, raw_text, success}
    """

    def __init__(self, config: Optional[FakeVLMJudgeConfig] = None):
        self.cfg = config or FakeVLMJudgeConfig()
        self._verify_ckpt()

    def _verify_ckpt(self):
        p = Path(self.cfg.ckpt_path)
        if not p.exists():
            _log.warning(f"FakeVLM ckpt not found at {p}; fallback to viviai needed")
            return False
        # Check safetensors files exist
        sftens = list(p.glob("*.safetensors"))
        if not sftens:
            _log.warning(f"No .safetensors in {p}")
            return False
        total_mb = sum(f.stat().st_size for f in sftens) / 1024 / 1024
        _log.info(f"FakeVLM ckpt OK: {len(sftens)} safetensors files, {total_mb:.0f} MB total")
        return True

    def is_server_up(self) -> bool:
        try:
            base = self.cfg.vllm_endpoint
            if base.endswith("/v1"):
                base = base[:-3]
            r = requests.get(base + "/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _parse_verdict(raw: str) -> tuple:
        """Map FakeVLM free-text to (is_fake, confidence, decided).

        Replicates the EXACT protocol from FakeVLM's official eval_vllm.py (the one
        that scored 98.9% on fakeclue_test): scan the FIRST sentence — if it contains
        'real' it's real (checked first, so "real photo ... not a fake" → real, which
        correctly handles negation); elif it contains 'fake' it's fake; otherwise fall
        back to the second sentence. An explicit <answer> tag, if present, wins.

        is_fake = (pred == 0) where eval labels 1=real, 0=fake.

        `decided` is False when the model returned NO real/fake verdict word anywhere
        (a generic caption, empty/newline garbage, or a VQA hallucination). The caller
        MUST NOT treat an undecided output as "real" — doing so manufactured 31% of the
        spurious historical bypasses (broken-ckpt captions defaulting to real@0.5). An
        undecided judgement means the detector FAILED to rule, not that it was fooled.
        """
        import re
        s = (raw or "").strip()
        low = s.lower()
        m = re.findall(r'<answer>\s*(real|fake)\s*</answer>', low)
        if m:
            return (m[-1] == "fake"), 0.9, True
        parts = low.split('.')
        first = parts[0] if parts else low
        if 'real' in first:
            return False, 0.85, True
        if 'fake' in first:
            return True, 0.85, True
        # fallback: second sentence (matches eval_vllm.py)
        second = parts[1] if len(parts) > 1 else ""
        if 'real' in second:
            return False, 0.7, True
        if 'fake' in second:
            return True, 0.7, True
        # nothing decisive anywhere → UNDECIDED. Not a verdict, not a bypass.
        return False, 0.0, False

    def judge(self, image_path: Union[str, Path]) -> dict:
        """Return same shape as sandbox.tier2_llm_judge.

        If server is down, return error stub (caller should fallback to viviai).
        """
        if not self.is_server_up():
            return {
                "model": "fakevlm_local_OFFLINE",
                "is_fake": False, "confidence": 0.5,
                "attack_family_guess": "unknown",
                "reasoning": "[vLLM server not running at " +
                             self.cfg.vllm_endpoint + "; run deploy_fakevlm_vllm.sh first]",
                "raw_text": "", "success": False,
            }
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            # NO system prompt: FakeVLM's fakeclue SFT + the validated eval_vllm.py
            # protocol use only the bare question. An off-distribution system prompt
            # ("anti-deepfake detector, output JSON") made the model hallucinate
            # manipulation on real faces and parrot the prompt text back.
            payload = {
                "model": self.cfg.lora_model_name or self.cfg.ckpt_path,
                "messages": [
                    {"role": "user",
                     "content": [
                         {"type": "image_url",
                          "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                         {"type": "text", "text": _PROMPT_TEMPLATE},
                     ]},
                ],
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            }
            r = requests.post(
                self.cfg.vllm_endpoint + "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.cfg.vllm_api_key}"},
                timeout=self.cfg.timeout,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            is_fake, conf, decided = self._parse_verdict(raw)
            return {
                "model": (self.cfg.lora_model_name or
                          f"fakevlm/{Path(self.cfg.ckpt_path).name}"),
                "is_fake": is_fake,
                "confidence": conf,
                "attack_family_guess": "unknown",
                "reasoning": raw[:500] if decided else f"[UNDECIDED — no real/fake verdict] {raw[:200]}",
                "raw_text": raw,
                # An undecided output (caption / garbage / hallucination) is NOT a
                # successful judgement. success=False keeps sandbox.verify from counting
                # it as tier2_says_real → no spurious bypass (matches the Q17 "never
                # silent bypass on tier2 failure" invariant).
                "success": bool(decided),
            }
        except Exception as e:
            return {
                "model": f"fakevlm/{Path(self.cfg.ckpt_path).name}",
                "is_fake": False, "confidence": 0.5,
                "attack_family_guess": "unknown",
                "reasoning": f"[error: {e}]",
                "raw_text": "", "success": False,
            }


# ────────────────────────── 启动 vLLM server 脚本生成 ──────────────

DEPLOY_SCRIPT_TEMPLATE = """#!/bin/bash
# 启动 FakeVLM vLLM server (作 sandbox Tier-2 detector)
# 端口 8000 → openai-compatible
# 需 GPU 显存 ~18GB (LLaVA-1.5-7B fp16)

CKPT={ckpt_path}
PORT=8000

cd /cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/FakeVLM_Official

# 选项 1: vLLM (推荐, 快)
vllm serve "$CKPT" \\
    --port $PORT \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.85 \\
    --max-model-len 4096 \\
    --enforce-eager

# 选项 2 (vLLM 不行时): 用 transformers 直接 inference (慢)
# python scripts/eval.py --ckpt "$CKPT" --image_path "$1"
"""


def emit_deploy_script(out_path: str | Path,
                        ckpt_path: str = "/cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/checkpoints/multi_20260329_132526_llava-1.5-7b"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(DEPLOY_SCRIPT_TEMPLATE.format(ckpt_path=ckpt_path))
    os.chmod(out_path, 0o755)
    return str(out_path)


# ────────────────────────── Smoke test ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    judge = FakeVLMJudge()

    print("=== Check ckpt + server ===")
    print(f"  ckpt exists: {Path(judge.cfg.ckpt_path).exists()}")
    print(f"  server up:   {judge.is_server_up()}")

    if judge.is_server_up():
        sample = "/data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces/0_row0_real.png"
        result = judge.judge(sample)
        print(f"\n=== Judge result ===")
        for k, v in result.items():
            print(f"  {k}: {str(v)[:200]}")
    else:
        out = emit_deploy_script("/data/disk4/lyx_ICML/self_evolution_forgery/outputs/deploy_fakevlm_vllm.sh")
        print(f"\n=== Server not running. Emitted deploy script: {out} ===")
        print(f"  Run: bash {out}")
        print(f"  Then re-run this smoke test")
