"""A/B test: default vicuna chat template vs raw-completion template override.

Validated eval_vllm.py feeds the model the RAW string '<image>Does the image looks
real/fake?' via model.generate (no role wrapper). The OpenAI /chat/completions path
wraps it as 'USER: <image>\\n... ASSISTANT:', which is off-distribution for the
fakeclue SFT and flips real faces to fake. We override chat_template to reproduce
the raw format and compare accuracy on gold images.
"""
import base64, json, os, re, time, requests

EP = "http://localhost:8000/v1/chat/completions"
MODEL = "/cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/checkpoints/multi_20260329_132526_llava-1.5-7b"

# Minimal template -> emits exactly: <image>Does the image looks real/fake?
RAW_TMPL = ("{% for message in messages %}"
            "{% for content in message['content'] | selectattr('type','equalto','image') %}{{ '<image>' }}{% endfor %}"
            "{% for content in message['content'] | selectattr('type','equalto','text') %}{{ content['text'] }}{% endfor %}"
            "{% endfor %}")

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

def call(img_b64, mode):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": "Does the image looks real/fake?"},
        ]}],
        "temperature": 0.0, "max_tokens": 96,
    }
    if mode == "raw":
        payload["chat_template"] = RAW_TMPL
    r = requests.post(EP, json=payload, headers={"Authorization": "Bearer EMPTY"}, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

GOLD = json.load(open("/cpfs01/bob_workspace/students/lyx/ICML/FakeVLM/Origin/FakeVLM/results/llava-1.5-7b-fakevlm_fakeclue_test.json"))
reals = [x for x in GOLD if x['label'] == 1]
fakes = [x for x in GOLD if x['label'] == 0]

def pick(lst, k):
    out = []
    for x in lst:
        if os.path.exists(x['image_path']):
            out.append(x)
        if len(out) >= k: break
    return out

sample = [(x, 1) for x in pick(reals, 10)] + [(x, 0) for x in pick(fakes, 10)]
print(f"sample: {sum(1 for _,l in sample if l==1)} real + {sum(1 for _,l in sample if l==0)} fake")

stats = {"default": {"r": [0, 0], "f": [0, 0]}, "raw": {"r": [0, 0], "f": [0, 0]}}
for x, label in sample:
    with open(x['image_path'], "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    for mode in ("default", "raw"):
        raw = call(b64, mode)
        pred = parse(raw)
        bucket = "r" if label == 1 else "f"
        stats[mode][bucket][0 if pred == label else 1] += 1
        if x is sample[0][0] or x is sample[10][0]:
            print(f"  [{mode}] label={label} pred={pred} :: {raw[:90]!r}")

for mode in ("default", "raw"):
    s = stats[mode]
    rr, rw = s["r"]; fr, fw = s["f"]
    real_acc = rr / (rr + rw) if rr + rw else 0
    fake_acc = fr / (fr + fw) if fr + fw else 0
    bal = (real_acc + fake_acc) / 2
    print(f"[{mode:7s}] real_acc={real_acc:.2f} ({rr}/{rr+rw})  fake_acc={fake_acc:.2f} ({fr}/{fr+fw})  bal_acc={bal:.2f}")
