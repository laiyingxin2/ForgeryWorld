"""Dense soft-reward re-scoring of forgeries (local GPU only, NO API, near-free).

Binary bypass (detector says 'real'?) gives ZERO signal against a 95%-accurate
detector: every forgery is caught, so the attacker can't tell a near-miss from a
hopeless attempt and self-evolution stalls. Following QDRT (2506.07121) / RADAR
(2601.03981), we read back the detector's CONTINUOUS p_real instead of the 0/1
label, so a near-miss forgery (p_real=0.45) scores higher than a hopeless one
(p_real=0.02). Rising mean p_real across rounds = the attacker IS getting closer
even while hard-bypass stays 0.

p_real is computed WITHOUT logprobs from the live server (it serves text only):
we teacher-force the two FakeVLM answers and softmax their sequence-logprobs:
    p_real = exp(LL("This is a real image.")) /
             (exp(LL("real")) + exp(LL("fake")))
on the SAME faithful base the :8001/:8002 servers serve (optionally + a round LoRA).

Usage:
  python soft_score.py --round-sft <coevo>/round_sft_r0.jsonl ... \
      --label R0 R1 R2 --out <coevo>/soft_margin.json [--lora <dir>] [--device cuda:0]
Each --round-sft is a jsonl of {"image": path, ...}; --label names each (parallel).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", "/data/disk4/lyx_ICML/self_evolution_forgery/outputs/lv5/.hf_cache")

import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

CKPT = "/data/disk4/lyx_ICML/self_evolution_forgery/scripts/fakevlm_correct_ckpt"
BASE_PROC = "/cpfs01/bob_workspace/students/lyx/Model_download/llava-hf/llava-1.5-7b-hf"
PROMPT = "<image>Does the image looks real/fake?"
ANS_REAL = "This is a real image."
ANS_FAKE = "This is a fake image."


def seq_logprob(model, processor, image, prompt, answer, device, max_len=1024):
    """Sum log p(answer tokens | prompt, image) under teacher forcing."""
    full = processor(images=[image], text=[prompt + " " + answer],
                     return_tensors="pt", truncation=True, max_length=max_len)
    ponly = processor(images=[image], text=[prompt],
                      return_tensors="pt", truncation=True, max_length=max_len)
    plen = int(ponly["input_ids"].shape[1])
    full = {k: v.to(device) for k, v in full.items()}
    with torch.no_grad():
        logits = model(**full).logits  # [1, T, V]
    ids = full["input_ids"][0]
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    total, T = 0.0, ids.shape[0]
    # token t is predicted from logits at t-1; score answer span [plen, T)
    for t in range(max(plen, 1), T):
        total += float(logp[t - 1, ids[t]])
    return total


def p_real(model, processor, image, device):
    lr = seq_logprob(model, processor, image, PROMPT, ANS_REAL, device)
    lf = seq_logprob(model, processor, image, PROMPT, ANS_FAKE, device)
    m = max(lr, lf)
    er, ef = torch.tensor(lr - m).exp(), torch.tensor(lf - m).exp()
    return float(er / (er + ef))


def load_images(sft):
    out = []
    if not os.path.exists(sft):
        return out
    for l in open(sft):
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        img = d.get("image") or (d.get("images") or [None])[0]
        if img and os.path.exists(img):
            out.append(img)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-sft", nargs="+", required=True)
    ap.add_argument("--label", nargs="+", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert len(args.round_sft) == len(args.label), "one --label per --round-sft"

    print(f"[{time.strftime('%H:%M:%S')}] loading {CKPT}", flush=True)
    processor = AutoProcessor.from_pretrained(BASE_PROC)
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = 14
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = "default"
    model = LlavaForConditionalGeneration.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16, device_map=args.device)
    if args.lora and os.path.exists(os.path.join(args.lora, "adapter_config.json")):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
        print(f"  + LoRA {args.lora}", flush=True)
    model.eval()

    curve = []
    for label, sft in zip(args.label, args.round_sft):
        imgs = load_images(sft)
        ps = []
        for img in imgs:
            try:
                ps.append(p_real(model, processor, Image.open(img).convert("RGB"), args.device))
            except Exception as e:
                print(f"  skip {os.path.basename(img)}: {e}", flush=True)
        mean_p = sum(ps) / len(ps) if ps else None
        soft_byp = (sum(1 for p in ps if p > 0.5) / len(ps)) if ps else None
        row = {"label": label, "n": len(ps),
               "mean_p_real": round(mean_p, 4) if mean_p is not None else None,
               "max_p_real": round(max(ps), 4) if ps else None,
               "hard_bypass": round(soft_byp, 4) if soft_byp is not None else None}
        curve.append(row)
        print(f"[{label}] n={row['n']} mean_p_real={row['mean_p_real']} "
              f"max={row['max_p_real']} bypass(p>0.5)={row['hard_bypass']}", flush=True)

    json.dump({"lora": args.lora, "curve": curve}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
