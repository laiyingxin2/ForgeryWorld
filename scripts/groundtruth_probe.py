"""Ground-truth probe: does LLM.generate() discriminate real vs fake on the gold
images in THIS env, and which SamplingParams/max_model_len reproduce the validated
full-sentence output (vs the terse 'Real' collapse the 8001 server shows)?

Loads the ckpt directly on a free GPU (set CUDA_VISIBLE_DEVICES) and runs the EXACT
validated prompt on a handful of gold real+fake images under two configs:
  A) eval_vllm.py params : max_model_len=800,  max_tokens=4096
  B) server params       : max_model_len=2048, max_tokens=64
"""
import json, os
from PIL import Image
from vllm import LLM, SamplingParams

CKPT = "/cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/checkpoints/multi_20260329_132526_llava-1.5-7b"
GOLD = json.load(open("/cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/results/llava-1.5-7b-fakevlm_fakeclue_test.json"))
PROMPT = "<image>Does the image looks real/fake?"


def pick(label, k):
    out = []
    for x in GOLD:
        if x['label'] == label and os.path.exists(x['image_path']):
            out.append(x)
        if len(out) >= k:
            break
    return out


sample = [(x, "REAL") for x in pick(1, 3)] + [(x, "FAKE") for x in pick(0, 3)]
imgs = [Image.open(x['image_path']).convert("RGB") for x, _ in sample]

for tag, mml, mt in [("A_eval", 800, 4096), ("B_server", 2048, 64)]:
    print(f"\n===== config {tag}: max_model_len={mml} max_tokens={mt} =====", flush=True)
    llm = LLM(model=CKPT, dtype="float16", max_model_len=mml,
              gpu_memory_utilization=0.30, tensor_parallel_size=1, enforce_eager=True)
    sp = SamplingParams(max_tokens=mt, temperature=0.0)
    reqs = [{"prompt": PROMPT, "multi_modal_data": {"image": im}} for im in imgs]
    outs = llm.generate(reqs, sp, use_tqdm=False)
    for (x, gt), o in zip(sample, outs):
        print(f"  [{gt}] {o.outputs[0].text[:90]!r}", flush=True)
    del llm
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
