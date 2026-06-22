"""P1-A: balance SFT pool with real-positives.

Reads defender_sft_v2.jsonl (sharegpt-conversations schema, all `<answer>fake</answer>`),
converts to swift format (messages + images), then appends N real-positive records
where the LLaVA target output is `<answer>real</answer>` with one-line forensic
justification. This fixes the catastrophic-forgetting bug where LoRA was trained on
100% fake-positives and forgot to ever say `real`.

Usage:
  python balance_sft_pool.py \
    --in-pool /data/disk4/lyx_ICML/self_evolution_forgery/outputs/lv5/defender_sft_v2.jsonl \
    --real-faces-dir /data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces \
    --n-real 50 \
    --out /data/disk4/lyx_ICML/self_evolution_forgery/outputs/lv5/attacker_lv5/defender_swift_balanced.jsonl
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path


_REAL_PROMPT = "<image>\nIs this face image real or fake? Reason step by step and conclude with <answer>real|fake</answer>."

_REAL_JUSTIFICATIONS = [
    "Sharp natural skin micro-texture with consistent pore distribution, eye specularities match the dominant light direction, no halo or color-bleed around the face contour, and high-frequency hair detail is intact.",
    "Eyebrow hair strands resolve cleanly, eyelashes cast crisp shadows, pore density matches expected human skin statistics, no GAN-style symmetric eye artifact, and the frequency spectrum is consistent with a CMOS-sensor capture rather than a generator.",
    "Skin-tone gradient across cheek-to-jaw is smooth and physically plausible, no swap-mask seam at the temple or jawline, irises retain natural limbal-ring contrast, and the image carries normal sensor noise rather than a low-pass-filtered fake.",
    "Specular highlight on the lip and tip-of-nose is sharp and direction-consistent, micro-shadows under chin and earlobe align with the same light source, no soft-blur artifact around the hairline, and texture preserves natural pore irregularity.",
    "Eye reflections show coherent room reflections, dental edges are sharply resolved without warping, ear cartilage micro-shadow detail is preserved, and there is no detectable spatial-frequency falloff typical of a forged synthesis.",
]


def conv_to_swift(rec: dict) -> dict | None:
    """defender_sft_v2 (sharegpt conversations) -> swift messages+images."""
    convs = rec.get("conversations") or []
    if len(convs) < 2:
        return None
    user_val = convs[0].get("value", "")
    asst_val = convs[1].get("value", "")
    image_path = rec.get("image") or (rec.get("meta") or {}).get("image_path") or ""
    if not image_path:
        return None
    return {
        "messages": [
            {"role": "user", "content": user_val},
            {"role": "assistant", "content": asst_val},
        ],
        "images": [image_path],
        "meta": {**(rec.get("meta") or {}), "label": "fake"},
    }


def make_real_positive(img_path: str) -> dict:
    just = random.choice(_REAL_JUSTIFICATIONS)
    assistant = f"<think>\n{just}\n</think>\n<answer>real</answer>"
    return {
        "messages": [
            {"role": "user", "content": _REAL_PROMPT},
            {"role": "assistant", "content": assistant},
        ],
        "images": [img_path],
        "meta": {"label": "real", "source": "real_faces_pool"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-pool", required=True,
                    help="defender_sft_v2.jsonl (sharegpt conversations)")
    ap.add_argument("--real-faces-dir", required=True)
    ap.add_argument("--n-real", type=int, default=50,
                    help="number of real-positive records to synthesize")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    # convert fake-positives
    in_path = Path(args.in_pool)
    swift_fake = []
    if in_path.exists():
        for line in in_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            sw = conv_to_swift(r)
            if sw is not None:
                swift_fake.append(sw)
    print(f"  converted {len(swift_fake)} fake-positive records", flush=True)

    # synthesize real-positives
    real_imgs = sorted(p for p in Path(args.real_faces_dir).glob("*.png"))
    if not real_imgs:
        print(f"  ERROR: no real faces found in {args.real_faces_dir}", file=sys.stderr)
        sys.exit(1)
    real_records = []
    for i in range(args.n_real):
        img = real_imgs[i % len(real_imgs)]
        real_records.append(make_real_positive(str(img)))
    print(f"  synthesized {len(real_records)} real-positive records "
          f"from {len(real_imgs)} unique faces", flush=True)

    # merge + shuffle
    pool = swift_fake + real_records
    random.shuffle(pool)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # report balance
    n_real = sum(1 for r in pool if (r.get("meta") or {}).get("label") == "real")
    n_fake = sum(1 for r in pool if (r.get("meta") or {}).get("label") == "fake")
    print(f"\n[balanced pool] total={len(pool)}, fake={n_fake} ({n_fake/len(pool)*100:.0f}%), "
          f"real={n_real} ({n_real/len(pool)*100:.0f}%)", flush=True)
    print(f"[balanced pool] written → {args.out}", flush=True)


if __name__ == "__main__":
    main()
