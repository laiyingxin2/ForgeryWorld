"""P1-B: 3-detector cross-eval for paper-grade defender comparison.

Detectors:
  A) FakeVLM + LoRA (defender) — vLLM model="defender"
  B) FakeVLM base (no LoRA) — vLLM model=<base ckpt path>
  C) gemini-2.5-flash — viviai (different vision LLM family)

For a directory of attack PNGs (or a list), each detector judges fake/real and
returns a verdict. Output: per-image table + 3-way confusion matrix +
"defender beats base" rate (how often LoRA catches what base misses).
"""
from __future__ import annotations
import argparse, json, base64, time, os, sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field

import requests

# allow standalone run
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fakevlm_judge_real import FakeVLMJudge, FakeVLMJudgeConfig
from viviai_client import ViviClient


_BASE_CKPT = "/cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/checkpoints/multi_20260329_132526_llava-1.5-7b"

_GEMINI_PROMPT = """You are a forensic deepfake detector inspecting a face image submitted to a KYC system.
Return STRICTLY JSON: {"is_fake": true|false, "confidence": 0.0-1.0, "reasoning": "<one paragraph>"}.
Cite visible artifacts, lighting, texture, frequency cues."""


@dataclass
class JudgeVerdict:
    detector_id: str
    is_fake: bool
    confidence: float
    reasoning: str = ""
    raw: str = ""
    success: bool = True
    error: Optional[str] = None


def judge_with_fakevlm(image_path: str, lora: bool = True,
                        endpoint: str = "http://localhost:8000/v1") -> JudgeVerdict:
    cfg = FakeVLMJudgeConfig(vllm_endpoint=endpoint,
                              lora_model_name="defender" if lora else None)
    j = FakeVLMJudge(cfg)
    r = j.judge(image_path)
    return JudgeVerdict(
        detector_id="FakeVLM+LoRA" if lora else "FakeVLM-base",
        is_fake=bool(r["is_fake"]), confidence=float(r["confidence"]),
        reasoning=str(r.get("reasoning", ""))[:300],
        raw=str(r.get("raw_text", ""))[:200],
        success=bool(r.get("success", False)),
        error=None if r.get("success", False) else "judge failed",
    )


def judge_with_gemini(image_path: str, client: ViviClient,
                       model: str = "gemini-2.5-flash") -> JudgeVerdict:
    try:
        parsed = client.chat_vision_json(
            model, _GEMINI_PROMPT, image_path,
            temperature=0.1, max_tokens=400,
        )
        return JudgeVerdict(
            detector_id=f"viviai/{model}",
            is_fake=bool(parsed.get("is_fake", False)),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=str(parsed.get("reasoning", ""))[:300],
            raw=json.dumps(parsed)[:200],
            success=True,
        )
    except Exception as e:
        return JudgeVerdict(detector_id=f"viviai/{model}",
                             is_fake=False, confidence=0.5,
                             success=False, error=str(e)[:120])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True,
                    help="paths to PNG attack images (or one dir glob)")
    ap.add_argument("--out", required=True, help="output JSON report path")
    ap.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--gemini-model", default="gemini-2.5-flash")
    ap.add_argument("--gemini-key",
        default=os.environ.get("VIVIAI_KEY", ""))
    args = ap.parse_args()

    # collect image list (handle dir or glob)
    images: list[Path] = []
    for p in args.images:
        pp = Path(p)
        if pp.is_dir():
            images.extend(sorted(pp.glob("*.png")))
        elif pp.exists():
            images.append(pp)
    if not images:
        print(f"no images found in {args.images}", file=sys.stderr); sys.exit(1)
    print(f"[cross_eval] {len(images)} images, 3 detectors")

    client = ViviClient(api_key=args.gemini_key)

    per_image = []
    counts = {"A_fake": 0, "B_fake": 0, "C_fake": 0,
              "ABC_agree_fake": 0, "ABC_agree_real": 0,
              "A_catches_B_misses": 0, "B_catches_A_misses": 0,
              "A_C_agree_B_disagree": 0,
              "A_success": 0, "B_success": 0, "C_success": 0}
    n = len(images)

    for i, img in enumerate(images):
        t0 = time.time()
        a = judge_with_fakevlm(str(img), lora=True, endpoint=args.vllm_endpoint)
        b = judge_with_fakevlm(str(img), lora=False, endpoint=args.vllm_endpoint)
        c = judge_with_gemini(str(img), client, model=args.gemini_model)
        dt = time.time() - t0
        rec = {"image": str(img), "duration_s": round(dt, 2),
               "A_defender_LoRA": asdict(a),
               "B_FakeVLM_base":   asdict(b),
               "C_gemini":         asdict(c)}
        per_image.append(rec)
        if a.success: counts["A_success"] += 1
        if b.success: counts["B_success"] += 1
        if c.success: counts["C_success"] += 1
        if a.is_fake: counts["A_fake"] += 1
        if b.is_fake: counts["B_fake"] += 1
        if c.is_fake: counts["C_fake"] += 1
        if a.is_fake == b.is_fake == c.is_fake == True:
            counts["ABC_agree_fake"] += 1
        if a.is_fake == b.is_fake == c.is_fake == False:
            counts["ABC_agree_real"] += 1
        if a.is_fake and not b.is_fake:
            counts["A_catches_B_misses"] += 1
        if b.is_fake and not a.is_fake:
            counts["B_catches_A_misses"] += 1
        if a.is_fake == c.is_fake and a.is_fake != b.is_fake:
            counts["A_C_agree_B_disagree"] += 1
        print(f"  [{i+1}/{n}] {img.name}: A={a.is_fake} B={b.is_fake} C={c.is_fake} "
              f"(A_LoRA conf={a.confidence:.2f}) [{dt:.1f}s]")

    report = {
        "n_images": n,
        "counts": counts,
        "detector_fake_rates": {
            "A_defender_LoRA":  f"{counts['A_fake']}/{n} = {counts['A_fake']/n:.1%}",
            "B_FakeVLM_base":   f"{counts['B_fake']}/{n} = {counts['B_fake']/n:.1%}",
            "C_gemini":         f"{counts['C_fake']}/{n} = {counts['C_fake']/n:.1%}",
        },
        "agreement": {
            "ABC_all_fake":  f"{counts['ABC_agree_fake']}/{n}",
            "ABC_all_real":  f"{counts['ABC_agree_real']}/{n}",
            "A_catches_extras_vs_B": counts['A_catches_B_misses'],
            "B_catches_extras_vs_A": counts['B_catches_A_misses'],
        },
        "per_image": per_image,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[cross_eval] report → {args.out}")
    print(f"  detector_fake_rates: {report['detector_fake_rates']}")
    print(f"  agreement: {report['agreement']}")


if __name__ == "__main__":
    main()
