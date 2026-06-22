"""Co-evolution: train ONE round of defender LoRA on the FAITHFUL FakeVLM base.

Fixes vs scripts/p1a_multi_round/train_one_round.py:
  1. base = the published llava-1.5-7b-fakevlm (staged fakevlm_correct_ckpt),
     NOT the multi_ multi-task ckpt -> the LoRA is composable with the 8002
     server, which serves the SAME base.
  2. RAW-completion prompt format '<image>Does the image looks real/fake? <answer>',
     matching the served eval path -> the LoRA actually transfers at inference.
     (The old vicuna 'USER: ... ASSISTANT:' format never matched the served prompt.)
  3. Answer-only loss masking (prompt tokens -> -100) so the update teaches the
     verdict, not prompt regurgitation -> less drift.

Replay / data composition is the caller's job (build_round_data.py): --data is a
swift-format jsonl of {messages:[user,assistant], images:[path]} already mixing
this round's gated bypasses + reals + a replay sample of prior rounds.
"""
from __future__ import annotations
import argparse, json, os, sys, time

os.environ.setdefault("HF_HOME", "/data/disk4/lyx_ICML/self_evolution_forgery/outputs/lv5/.hf_cache")
os.environ.setdefault("TMPDIR", "/data/disk4/lyx_ICML/self_evolution_forgery/outputs/lv5/.tmpdir")
for d in [os.environ["HF_HOME"], os.environ["TMPDIR"]]:
    os.makedirs(d, exist_ok=True)

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

CORRECT_CKPT = "/data/disk4/lyx_ICML/self_evolution_forgery/scripts/fakevlm_correct_ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=CORRECT_CKPT)
    # the official HF processor expands <image> correctly (the staged ckpt's copied
    # processor lacks image_seq_length config -> 575/576 token mismatch). Tokenizer
    # vocab is identical, so the LoRA still transfers to the served detector.
    ap.add_argument("--base-proc",
        default="/cpfs01/bob_workspace/students/lyx/Model_download/llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--data", required=True, help="swift-format jsonl (messages+images)")
    ap.add_argument("--out", required=True, help="LoRA adapter output dir")
    ap.add_argument("--prev-lora", default=None, help="warm-start from previous round's LoRA")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading processor {args.base_proc}")
    processor = AutoProcessor.from_pretrained(args.base_proc)
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = 14
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = "default"

    print(f"[{time.strftime('%H:%M:%S')}] loading FakeVLM base {args.base}")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map=args.device)

    if args.prev_lora and os.path.exists(os.path.join(args.prev_lora, "adapter_config.json")):
        from peft import PeftModel
        print(f"[{time.strftime('%H:%M:%S')}] warm-start from {args.prev_lora} (Evolving base)")
        model = PeftModel.from_pretrained(model, args.prev_lora, is_trainable=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] fresh LoRA r={args.lora_r}")
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM, bias="none"))
    model.print_trainable_parameters()

    class DefenderDataset(Dataset):
        def __init__(self, jsonl_path):
            self.samples = []
            for l in open(jsonl_path):
                l = l.strip()
                if not l:
                    continue
                r = json.loads(l)
                if r.get("images") and os.path.exists(r["images"][0]):
                    self.samples.append(r)
            print(f"  loaded {len(self.samples)} defender samples (image-verified)")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            d = self.samples[idx]
            try:
                image = Image.open(d["images"][0]).convert("RGB")
            except Exception:
                return None
            msgs = d["messages"]
            q = msgs[0]["content"].replace("<image>", "").strip()
            a = msgs[1]["content"].strip()
            # RAW-completion: matches the served prompt '<image>Does the image looks real/fake?'
            return {"image": image, "prompt": f"<image>{q}", "answer": a}

    ds = DefenderDataset(args.data)
    if len(ds) == 0:
        print("ERROR: empty dataset", file=sys.stderr)
        sys.exit(1)

    def collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        b = batch[0]  # batch_size=1: clean per-sample answer masking
        full = processor(images=[b["image"]], text=[b["prompt"] + " " + b["answer"]],
                         return_tensors="pt", truncation=True, max_length=args.max_len)
        # length of the prompt span (incl. expanded image tokens) to mask
        prompt_only = processor(images=[b["image"]], text=[b["prompt"]],
                                return_tensors="pt", truncation=True, max_length=args.max_len)
        plen = int(prompt_only["input_ids"].shape[1])
        labels = full["input_ids"].clone()
        if 0 < plen < labels.shape[1]:
            labels[:, :plen] = -100   # answer-only loss
        full["labels"] = labels
        return {k: v.to(args.device) for k, v in full.items()}

    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    print(f"[{time.strftime('%H:%M:%S')}] training {len(ds)} samples × {args.epochs} epochs, lr={args.lr}")
    final_epoch_loss = None
    for epoch in range(args.epochs):
        epoch_loss, epoch_steps = 0.0, 0
        for batch in loader:
            if batch is None:
                continue
            try:
                out = model(**batch)
                loss = out.loss
                if loss is None or torch.isnan(loss):
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                epoch_steps += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  step err: {e}")
                continue
        final_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  [epoch {epoch+1}/{args.epochs}] avg loss = {final_epoch_loss:.4f} over {epoch_steps} steps")

    print(f"[{time.strftime('%H:%M:%S')}] saving to {args.out}")
    model.save_pretrained(args.out)
    meta = {"data_path": args.data, "out_path": args.out, "prev_lora": args.prev_lora,
            "base": args.base, "epochs": args.epochs, "lr": args.lr, "lora_r": args.lora_r,
            "n_samples": len(ds), "final_epoch_loss": final_epoch_loss,
            "format": "raw_completion_answer_masked",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  train_meta.json: {meta}")


if __name__ == "__main__":
    main()
