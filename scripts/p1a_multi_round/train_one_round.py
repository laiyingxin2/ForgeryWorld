"""P1-A: train one round of defender LoRA.

Same hyperparams as outputs/lv5/attacker_lv5/train_defender.py but:
  - reads from --data (so multi-round can swap pool each round)
  - writes to --out (so each round has its own LoRA dir)
  - reports final train loss for monotonicity tracking
"""
from __future__ import annotations
import argparse, json, os, sys, time

os.environ.setdefault("HF_HOME", "/data/disk4/lyx_ICML/self_evolution_forgery/outputs/lv5/.hf_cache")
os.environ.setdefault("TMPDIR", "/data/disk4/lyx_ICML/self_evolution_forgery/outputs/lv5/.tmpdir")
os.environ.setdefault("MODELSCOPE_CACHE", os.environ["HF_HOME"] + "/modelscope")
for d in [os.environ["HF_HOME"], os.environ["TMPDIR"], os.environ["MODELSCOPE_CACHE"]]:
    os.makedirs(d, exist_ok=True)

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",
        default="/cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/checkpoints/multi_20260329_132526_llava-1.5-7b")
    ap.add_argument("--base-proc",
        default="/cpfs01/bob_workspace/students/lyx/Model_download/llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--data", required=True, help="swift format jsonl (messages+images)")
    ap.add_argument("--out", required=True, help="LoRA adapter output dir")
    ap.add_argument("--prev-lora", default=None,
                    help="if set, warm-start from previous round's LoRA")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading processor")
    processor = AutoProcessor.from_pretrained(args.base_proc)
    if not hasattr(processor, 'patch_size') or processor.patch_size is None:
        processor.patch_size = 14
    if not hasattr(processor, 'vision_feature_select_strategy') or processor.vision_feature_select_strategy is None:
        processor.vision_feature_select_strategy = "default"

    print(f"[{time.strftime('%H:%M:%S')}] loading FakeVLM base")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map=args.device
    )

    # Warm-start: load previous LoRA, then make it trainable
    if args.prev_lora and os.path.exists(os.path.join(args.prev_lora, "adapter_config.json")):
        from peft import PeftModel
        print(f"[{time.strftime('%H:%M:%S')}] warm-start from {args.prev_lora}")
        model = PeftModel.from_pretrained(model, args.prev_lora, is_trainable=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] fresh LoRA r={args.lora_r}")
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM, bias="none",
        )
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()

    class DefenderDataset(Dataset):
        def __init__(self, jsonl_path):
            self.samples = []
            for l in open(jsonl_path):
                r = json.loads(l)
                if r.get("images") and os.path.exists(r["images"][0]):
                    self.samples.append(r)
            print(f"  loaded {len(self.samples)} defender samples (image-verified)")
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            d = self.samples[idx]
            img_path = d["images"][0]
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                return None
            msgs = d["messages"]
            user_clean = msgs[0]['content'].replace('<image>', '').strip()
            prompt = f"USER: <image>\n{user_clean} ASSISTANT: {msgs[1]['content']}"
            return {"image": image, "prompt": prompt}

    ds = DefenderDataset(args.data)
    if len(ds) == 0:
        print("ERROR: empty dataset", file=sys.stderr); sys.exit(1)

    def collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        images = [b["image"] for b in batch]
        prompts = [b["prompt"] for b in batch]
        inputs = processor(images=images, text=prompts, return_tensors="pt",
                            padding=True, truncation=True, max_length=args.max_len)
        inputs["labels"] = inputs["input_ids"].clone()
        return {k: v.to(args.device) for k, v in inputs.items()}

    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    print(f"[{time.strftime('%H:%M:%S')}] training, {len(ds)} samples × {args.epochs} epochs, lr={args.lr}")
    final_epoch_loss = None
    for epoch in range(args.epochs):
        epoch_loss = 0.0; epoch_steps = 0
        for batch in loader:
            if batch is None: continue
            try:
                out = model(**batch)
                loss = out.loss
                if loss is None or torch.isnan(loss): continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()); epoch_steps += 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            except Exception as e:
                print(f"  step err: {e}"); continue
        avg = epoch_loss / max(epoch_steps, 1)
        final_epoch_loss = avg
        print(f"  [epoch {epoch+1}/{args.epochs}] avg loss = {avg:.4f} over {epoch_steps} steps")

    print(f"[{time.strftime('%H:%M:%S')}] saving to {args.out}")
    model.save_pretrained(args.out)

    # write training metadata
    meta = {
        "data_path": args.data,
        "out_path": args.out,
        "prev_lora": args.prev_lora,
        "epochs": args.epochs, "lr": args.lr, "lora_r": args.lora_r,
        "n_samples": len(ds),
        "final_epoch_loss": final_epoch_loss,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  train_meta.json: {meta}")


if __name__ == "__main__":
    main()
