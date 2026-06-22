"""Assemble ONE round's defender training data for co-evolution.

Reads the attacker round's `defender_sft_{mode}.jsonl` (ShareGPT CoT export) and
re-emits training data in FakeVLM's NATIVE raw-completion format so the LoRA
hardens the actually-served detector:
    user      : "<image>Does the image looks real/fake?"
    assistant : "This is a fake image."   (forgeries)  /  "This is a real image."

Composition (the catastrophic-forgetting fix the prior P1-A lacked):
  • this round's forgeries  -> label fake   (the new hard positives)
  • trusted-real src faces  -> label real   (oversampled ~50% to hold real-acc)
  • REPLAY buffer: a sample of EVERY prior round's forgeries -> label fake
Also persists a cumulative replay_store.jsonl and writes guard_set.jsonl (prior
forgeries) so the wrapper can run the STaSC Non-Decreasing regression check.

Usage:
  python build_round_data.py --round 1 \
     --attacker-sft <m3 round dir>/defender_sft_v2.jsonl \
     --coevo-dir <run>/coevo \
     --src-real f1.png f2.png ... \
     --replay-per-round 8 --out-data <run>/coevo/train_r1.jsonl
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path

PROMPT = "Does the image looks real/fake?"
ANS_FAKE = "This is a fake image."
ANS_REAL = "This is a real image."


def _rec(image, is_fake):
    return {"messages": [{"role": "user", "content": f"<image>{PROMPT}"},
                         {"role": "assistant", "content": ANS_FAKE if is_fake else ANS_REAL}],
            "images": [image], "label": "fake" if is_fake else "real"}


def _read_attacker_forgeries(sft_path):
    """Return [(image_path, bypass_succeeded)] for image-verified forgery entries."""
    out = []
    if not os.path.exists(sft_path):
        return out
    for l in open(sft_path):
        l = l.strip()
        if not l:
            continue
        try:
            d = json.loads(l)
        except Exception:
            continue
        img = d.get("image") or (d.get("images") or [None])[0]
        if not img or not os.path.exists(img):
            continue
        byp = bool(d.get("meta", {}).get("bypass_succeeded", False))
        out.append((img, byp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--attacker-sft", required=True)
    ap.add_argument("--coevo-dir", required=True)
    ap.add_argument("--src-real", nargs="+", required=True)
    ap.add_argument("--replay-per-round", type=int, default=8)
    ap.add_argument("--out-data", required=True)
    # M3-overhaul: hold out a fraction of the real pool from TRAINING to use as a
    # never-trained real-acc guard set (catches a defender that wins by paranoia).
    # Default 0.0 = legacy behavior (no holdout) so in-flight runs are unaffected.
    ap.add_argument("--guard-real-frac", type=float, default=0.0)
    args = ap.parse_args()

    coevo = Path(args.coevo_dir)
    coevo.mkdir(parents=True, exist_ok=True)
    replay_path = coevo / "replay_store.jsonl"
    random.seed(1234 + args.round)

    forgeries = _read_attacker_forgeries(args.attacker_sft)
    this_round_imgs = [img for img, _ in forgeries]
    n_bypass = sum(1 for _, b in forgeries if b)
    print(f"[R{args.round}] attacker forgeries: {len(forgeries)} "
          f"(bypassed detector: {n_bypass})")

    # ── replay: sample prior rounds' forgeries from the cumulative store ──
    prior = []
    if replay_path.exists():
        for l in replay_path.read_text().splitlines():
            l = l.strip()
            if not l:
                continue
            d = json.loads(l)
            if d.get("round", -1) < args.round and os.path.exists(d.get("image", "")):
                prior.append(d)
    # group by round, sample up to replay-per-round from each prior round
    by_round = {}
    for d in prior:
        by_round.setdefault(d["round"], []).append(d["image"])
    replay_imgs = []
    for rd, imgs in sorted(by_round.items()):
        k = min(args.replay_per_round, len(imgs))
        replay_imgs += random.sample(imgs, k)
    print(f"[R{args.round}] replay: {len(replay_imgs)} forgeries from {len(by_round)} prior round(s)")

    # ── assemble training records ──
    recs = []
    for img in this_round_imgs:
        recs.append(_rec(img, is_fake=True))
    for img in replay_imgs:
        recs.append(_rec(img, is_fake=True))
    n_fake = len(recs)
    # oversample reals to ~50% of fakes (class balance so detector keeps real-acc)
    reals_all = [r for r in args.src_real if os.path.exists(r)]
    # ── hold out never-trained reals for the real-acc guard floor ──
    guard_reals = []
    reals = reals_all
    if args.guard_real_frac > 0.0 and len(reals_all) >= 3:
        rr = random.Random(20260621)  # FIXED split across rounds (consistent holdout)
        shuffled = sorted(reals_all)
        rr.shuffle(shuffled)
        n_hold = max(1, min(len(reals_all) - 2, round(len(reals_all) * args.guard_real_frac)))
        guard_reals = shuffled[:n_hold]
        reals = shuffled[n_hold:]            # only these are trained on
        print(f"[R{args.round}] real holdout: {len(guard_reals)} guard / {len(reals)} train")
    if reals:
        target_real = max(len(reals), n_fake // 2)
        for i in range(target_real):
            recs.append(_rec(reals[i % len(reals)], is_fake=False))
    random.shuffle(recs)

    Path(args.out_data).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_data, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[R{args.round}] wrote {len(recs)} train recs "
          f"({n_fake} fake / {len(recs)-n_fake} real) -> {args.out_data}")

    # ── guard set = prior forgeries (regression check); write BEFORE appending ──
    guard_path = coevo / f"guard_set_r{args.round}.jsonl"
    with open(guard_path, "w") as f:
        for img in replay_imgs:
            f.write(json.dumps({"image": img, "label": "fake"}) + "\n")
    print(f"[R{args.round}] guard set: {len(replay_imgs)} prior forgeries -> {guard_path}")

    # ── real-acc guard set = never-trained held-out reals (paranoia check) ──
    guard_reals_path = coevo / f"guard_reals_r{args.round}.jsonl"
    with open(guard_reals_path, "w") as f:
        for img in guard_reals:
            f.write(json.dumps({"image": img, "label": "real"}) + "\n")
    print(f"[R{args.round}] real-acc guard: {len(guard_reals)} held-out reals -> {guard_reals_path}")

    # ── append THIS round's forgeries to the cumulative replay store ──
    with open(replay_path, "a") as f:
        for img, byp in forgeries:
            f.write(json.dumps({"image": img, "round": args.round, "bypass": byp}) + "\n")

    # summary for the wrapper to log the arms-race curve
    summary = {"round": args.round, "n_forgeries": len(forgeries), "n_bypass": n_bypass,
               "bypass_rate": (n_bypass / len(forgeries)) if forgeries else 0.0,
               "n_replay": len(replay_imgs), "n_train": len(recs),
               "n_fake": n_fake, "n_real": len(recs) - n_fake,
               "guard_set": str(guard_path)}
    (coevo / f"data_summary_r{args.round}.json").write_text(json.dumps(summary, indent=2))
    print(f"[R{args.round}] summary: {summary}")


if __name__ == "__main__":
    main()
