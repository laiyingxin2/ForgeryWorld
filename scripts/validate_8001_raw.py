"""Validate the raw-template server (8001) vs default-template server (8000) on gold.

8001 serves the SAME FakeVLM ckpt but with --chat-template fakevlm_raw.jinja, so the
chat API renders '<image>Does the image looks real/fake?' (the validated eval format).
Expect 8001 to recover ~99% real-acc / ~96% fake-acc; 8000 (vicuna wrapper) flips reals.
"""
import base64, json, os, re, time, requests

GOLD = json.load(open("/cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/results/llava-1.5-7b-fakevlm_fakeclue_test.json"))
CKPT = "/cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/checkpoints/multi_20260329_132526_llava-1.5-7b"
SERVERS = {
    "8000_default": ("http://localhost:8000/v1/chat/completions", CKPT),
    "8001_raw":     ("http://localhost:8001/v1/chat/completions", "fakevlm_raw"),
}
N = 20  # per class

def parse(raw):
    s = (raw or "").strip(); low = s.lower()
    m = re.findall(r'<answer>\s*(real|fake)\s*</answer>', low)
    if m: return 0 if m[-1] == "fake" else 1
    parts = low.split('.'); first = parts[0] if parts else low
    if 'real' in first: return 1
    if 'fake' in first: return 0
    second = parts[1] if len(parts) > 1 else ""
    if 'real' in second: return 1
    if 'fake' in second: return 0
    return 1

def call(ep, model, b64):
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "Does the image looks real/fake?"}]}],
        "temperature": 0.0, "max_tokens": 64}
    r = requests.post(ep, json=payload, headers={"Authorization": "Bearer EMPTY"}, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def pick(label, k):
    out = []
    for x in GOLD:
        if x['label'] == label and os.path.exists(x['image_path']):
            out.append(x)
        if len(out) >= k: break
    return out

sample = [(x, 1) for x in pick(1, N)] + [(x, 0) for x in pick(0, N)]
print(f"sample: {N} real + {N} fake (gold pred_label all correct on raw-generate)")

stats = {s: {"r": [0, 0], "f": [0, 0]} for s in SERVERS}
examples = {s: [] for s in SERVERS}
t0 = time.time()
for x, label in sample:
    with open(x['image_path'], "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    for s, (ep, model) in SERVERS.items():
        try:
            raw = call(ep, model, b64)
            pred = parse(raw)
        except Exception as e:
            raw = f"[ERR {e}]"; pred = -1
        bucket = "r" if label == 1 else "f"
        if pred == label: stats[s][bucket][0] += 1
        else: stats[s][bucket][1] += 1
        if len(examples[s]) < 3 and label == 1:
            examples[s].append(f"real->pred{pred} :: {raw[:75]!r}")

print(f"\n(elapsed {time.time()-t0:.0f}s)")
for s in SERVERS:
    st = stats[s]; rr, rw = st["r"]; fr, fw = st["f"]
    ra = rr/(rr+rw) if rr+rw else 0; fa = fr/(fr+fw) if fr+fw else 0
    print(f"\n[{s}] real_acc={ra:.0%} ({rr}/{rr+rw})  fake_acc={fa:.0%} ({fr}/{fr+fw})  bal_acc={(ra+fa)/2:.0%}")
    for e in examples[s]: print("   ", e)
