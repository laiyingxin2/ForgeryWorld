"""STaSC-style correction loop for Method 3 (paper 2503.08681).

之前 method 3 = passive SFT 训 defender on (image, fake_label). 缺 STaSC 的核心:
  "model corrects its own previous wrong answer"

STaSC pipeline (external/STaSC/src/executors/stasc.py:44-106):
  1. init-gen: model produces initial answer (here: defender's initial verdict on attack image)
  2. sample N corrections of own answer (here: re-prompt with "your verdict was wrong, correct it")
  3. filter `improving` (init wrong & corr right) or `non_decreasing`
  4. FT only on improvements
  5. re-eval

Face-forgery adaptation:
  - For each (image, ground_truth_fake) where defender got it WRONG (said real):
    - Ask defender: "your previous verdict was 'real'; we know it's fake. Re-analyze
      and explain WHY it's fake."
    - Get N=3 corrected reasoning samples.
    - Filter: keep ONLY those whose corrected answer correctly says 'fake' with
      coherent reasoning citing specific artifacts.
  - These (image, corrected_reasoning, 'fake') triples become the R+1 SFT pool.

Reference: external/STaSC/src/prompts/stasc_builder.py:99-119 _CORRECTION prompt
"""
from __future__ import annotations
import json
import re
import logging
import time
import requests
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fakevlm_judge_real import FakeVLMJudge, FakeVLMJudgeConfig

_log = logging.getLogger(__name__)


# STaSC correction prompt (paper verbatim adaptation to face-KYC):
# original: "Consider the previous answer.  Generate a correction.  Disregard
#            the information you already have …"
_CORRECTION_PROMPT_TEMPLATE = """<image>
Your previous verdict on this face image was: "{prev_verdict}".

Ground truth from our authorized red-team metadata: this image IS a forgery
(family: {family}, pipeline: {pipeline_hint}).

Consider where your previous reasoning went wrong. Re-analyze the image from
scratch, disregarding any biases from your prior verdict. Cite specific
visible artifacts (frequency cues, lighting inconsistency, micro-texture, eye
specular, hairline, etc.) that support the correct verdict.

Output STRICTLY:
<think>
[your re-analysis citing >=2 specific artifacts]
</think>
<answer>fake</answer>"""


@dataclass
class CorrectionSample:
    image_path: str
    family: str
    pipeline_hint: str
    prev_verdict: str        # "real" (the wrong one)
    corrected_reasoning: str
    corrected_answer: str    # ideally "fake"
    improvement: bool        # True iff prev_verdict != "fake" AND new answer == "fake"


def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_correction(
    image_path: str,
    prev_verdict: str,
    family: str,
    pipeline_hint: str,
    vllm_endpoint: str = "http://localhost:8000/v1",
    lora_model: str = "defender",
    n_samples: int = 3,
    temperature: float = 0.6,
    max_tokens: int = 350,
) -> list[CorrectionSample]:
    """Ask the (LoRA) FakeVLM to correct its previous wrong verdict.
    Returns N samples; caller filters for improving ones."""
    prompt = _CORRECTION_PROMPT_TEMPLATE.format(
        prev_verdict=prev_verdict, family=family,
        pipeline_hint=pipeline_hint,
    )
    img_b64 = _img_to_b64(image_path)
    samples = []
    for i in range(n_samples):
        payload = {
            "model": lora_model,
            "messages": [
                {"role": "system",
                 "content": "You are a face-KYC anti-deepfake detector being trained "
                            "to correct your own mistakes via authorized internal red-team. "
                            "Output strict format only."},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ]},
            ],
            "temperature": temperature, "max_tokens": max_tokens,
        }
        try:
            r = requests.post(f"{vllm_endpoint}/chat/completions", json=payload,
                               headers={"Authorization": "Bearer EMPTY"}, timeout=45)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            _log.warning(f"correction call {i} failed: {e}")
            continue
        ans_match = re.search(r'<answer>\s*(real|fake)\s*</answer>', raw, re.IGNORECASE)
        ans = ans_match.group(1).lower() if ans_match else "unknown"
        reasoning_match = re.search(r'<think>(.*?)</think>', raw, re.S | re.IGNORECASE)
        reasoning = (reasoning_match.group(1).strip() if reasoning_match
                     else raw[:300])
        improvement = (prev_verdict.lower() != "fake" and ans == "fake")
        samples.append(CorrectionSample(
            image_path=image_path, family=family, pipeline_hint=pipeline_hint,
            prev_verdict=prev_verdict, corrected_reasoning=reasoning,
            corrected_answer=ans, improvement=improvement,
        ))
    return samples


def collect_correction_stats(samples: list[CorrectionSample]) -> dict:
    """Port of STaSC src/helper/stasc.py:133-149 — transition matrix."""
    I_to_C = sum(1 for s in samples if s.prev_verdict.lower() != "fake"
                  and s.corrected_answer == "fake")
    I_to_I = sum(1 for s in samples if s.prev_verdict.lower() != "fake"
                  and s.corrected_answer != "fake")
    C_to_C = sum(1 for s in samples if s.prev_verdict.lower() == "fake"
                  and s.corrected_answer == "fake")
    C_to_I = sum(1 for s in samples if s.prev_verdict.lower() == "fake"
                  and s.corrected_answer != "fake")
    total = max(len(samples), 1)
    return {
        "I→C (improving)":     f"{I_to_C}/{total} = {I_to_C/total:.2%}",
        "I→I (still wrong)":   f"{I_to_I}/{total} = {I_to_I/total:.2%}",
        "C→C (kept correct)":  f"{C_to_C}/{total} = {C_to_C/total:.2%}",
        "C→I (regression!)":   f"{C_to_I}/{total} = {C_to_I/total:.2%}",
    }


def samples_to_sft_jsonl(samples: list[CorrectionSample],
                          out_path: str | Path,
                          mode: str = "improving") -> int:
    """Convert improving correction samples to swift SFT format.

    mode='improving' → STaSC filter, only keep init-wrong & corr-right (paper preferred)
    mode='all'       → keep all corrections (more data, noisier)
    """
    n = 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in samples:
            keep = (mode == "all") or (mode == "improving" and s.improvement)
            if not keep: continue
            rec = {
                "messages": [
                    {"role": "user",
                     "content": "<image>\nIs this face image real or fake? "
                                "Reason step by step and conclude with <answer>real|fake</answer>."},
                    {"role": "assistant",
                     "content": f"<think>\n{s.corrected_reasoning}\n</think>\n"
                                f"<answer>{s.corrected_answer}</answer>"},
                ],
                "images": [s.image_path],
                "meta": {"family": s.family, "source": "stasc_correction",
                          "prev_verdict": s.prev_verdict,
                          "improvement": s.improvement},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# ────────────────────────── smoke ──────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    # smoke without live vLLM — synthesize fake samples to test plumbing
    samples = [
        CorrectionSample(
            image_path="/tmp/face_attack_outputs/LocalInSwapperOperator_d1388435.png",
            family="frontal_swap",
            pipeline_hint="face_align → inswapper_128_local → jpeg_85",
            prev_verdict="real",
            corrected_reasoning="Inspecting the upper cheek region I notice "
                                "asymmetric pore distribution and a faint seam at "
                                "the temple, indicating a face-swap operation.",
            corrected_answer="fake",
            improvement=True,
        ),
        CorrectionSample(
            image_path="/tmp/face_attack_outputs/LocalInSwapperOperator_d1388435.png",
            family="frontal_swap",
            pipeline_hint="face_align → inswapper_128_local",
            prev_verdict="real",
            corrected_reasoning="Re-analysis still finds no obvious artifact; "
                                "this image appears authentic.",
            corrected_answer="real",        # no improvement
            improvement=False,
        ),
    ]
    stats = collect_correction_stats(samples)
    print("=== STaSC correction stats ===")
    for k, v in stats.items(): print(f"  {k:25s}: {v}")
    n_written = samples_to_sft_jsonl(samples, "/tmp/stasc_smoke.jsonl",
                                       mode="improving")
    print(f"\n  improving samples written → /tmp/stasc_smoke.jsonl ({n_written})")
    print(f"\nLive vLLM mode:")
    print(f"  get_correction(image, prev_verdict, family, pipeline_hint)")
    print(f"  → returns N=3 candidate corrections → filter by .improvement → train R+1")
