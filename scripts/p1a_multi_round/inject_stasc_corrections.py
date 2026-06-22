"""BUG-6 fix: Use STaSC correction loop to GENERATE new SFT data when sandbox
catches everything (SFT pool stagnation problem).

P1-A R0/R1/R2 stuck at 113 SFT pool because: when defender LoRA catches all attacks,
no new bypass cases → no new (image, fake) labelled SFT data → R1/R2 LoRA train on
the SAME R0 data → no progress.

STaSC fix: pick the most-confidently-flagged "real" attack images (false negatives
from defender's view) and ask defender to re-analyze + correct. Filter improvements.

Usage:
  python inject_stasc_corrections.py \
    --image-dir <attack images dir> \
    --metadata <jsonl with family+pipeline_hint per image> \
    --out <new_sft.jsonl> \
    --n-per-image 3
"""
from __future__ import annotations
import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lv5_correction_loop import (
    get_correction, collect_correction_stats, samples_to_sft_jsonl,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True,
                    help="dir containing attack images (PNG)")
    ap.add_argument("--metadata", required=True,
                    help="jsonl: {image_path, family, pipeline_hint, prev_verdict}")
    ap.add_argument("--out", required=True, help="new SFT jsonl (swift format)")
    ap.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--lora-model", default="defender")
    ap.add_argument("--n-per-image", type=int, default=3,
                    help="N candidate corrections per image")
    ap.add_argument("--mode", choices=["improving", "all"], default="improving")
    args = ap.parse_args()

    # collect metadata records
    if not Path(args.metadata).exists():
        print(f"ERROR: metadata file {args.metadata} missing", file=sys.stderr)
        sys.exit(1)
    records = [json.loads(l) for l in open(args.metadata) if l.strip()]
    print(f"[stasc-inject] {len(records)} attack images to correct")

    all_samples = []
    for i, rec in enumerate(records):
        img = rec.get("image_path")
        if not img or not Path(img).exists():
            print(f"  [{i+1}] skip — image not found: {img}")
            continue
        family = rec.get("family", "frontal_swap")
        pipeline_hint = rec.get("pipeline_hint", "")
        prev_verdict = rec.get("prev_verdict", "real")  # default = false-negative case
        print(f"  [{i+1}/{len(records)}] correcting {Path(img).name} "
              f"(family={family}, prev={prev_verdict})")
        try:
            samples = get_correction(
                image_path=img, prev_verdict=prev_verdict,
                family=family, pipeline_hint=pipeline_hint,
                vllm_endpoint=args.vllm_endpoint, lora_model=args.lora_model,
                n_samples=args.n_per_image,
            )
            all_samples.extend(samples)
        except Exception as e:
            print(f"    err: {e}")
            continue

    # transition stats
    stats = collect_correction_stats(all_samples)
    print(f"\n[stasc-inject] transition stats over {len(all_samples)} corrections:")
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")

    # write SFT jsonl
    n = samples_to_sft_jsonl(all_samples, args.out, mode=args.mode)
    print(f"\n[stasc-inject] wrote {n} (mode={args.mode}) SFT records → {args.out}")
    print(f"[stasc-inject] cat this onto cumulative SFT pool for next-round LoRA training")


if __name__ == "__main__":
    main()
