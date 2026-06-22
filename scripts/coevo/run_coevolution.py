"""Method-3 TRUE co-evolution driver (arms race).

Per round R:
  1. Attacker (the v2 self-evolving stack, state accumulates in a SHARED out dir)
     attacks the CURRENT detector D_R served on :8002.
  2. Isolate THIS round's forgeries by diffing face_attack_outputs/ (filesystem),
     join with defender_sft_v2.jsonl to recover per-image bypass labels.
  3. Assemble defender training data = this round's forgeries (fake) + trusted-real
     sources (real) + a REPLAY sample of every prior round (fake)  [forgetting fix].
  4. Train defender LoRA D_{R+1}, warm-started from D_R (STaSC "Evolving" base).
  5. STaSC Non-Decreasing guard: if D_{R+1} regresses on prior forgeries beyond
     --guard-drop, reject it and keep D_R.
  6. Hot-reload the accepted LoRA into :8002; next round faces the harder detector.

M1/M2/M4 are unaffected: they keep hitting the FROZEN :8001. The ONLY structural
difference for M3 is this train+reload between rounds.

Metric (literature: arms-race ASR is non-monotone by design): we log per-round
bypass_rate vs D_R, defender guard-recall, and per-round delta -> convergence.
"""
from __future__ import annotations
import argparse, base64, json, os, subprocess, sys, time
from pathlib import Path
import requests


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def list_outputs(d):
    p = Path(d)
    return set(str(x) for x in p.glob("*")) if p.exists() else set()


def judge_fake_rate(endpoint, images, timeout=60):
    """Run images through the CURRENTLY-loaded detector on :endpoint; return the
    fraction judged FAKE (FakeVLM eval parse: first sentence — 'real' first)."""
    if not images:
        return None, 0
    n_fake, n = 0, 0
    for img in images:
        if not os.path.exists(img):
            continue
        b64 = base64.b64encode(open(img, "rb").read()).decode()
        body = {"model": "fakevlm_lora", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Does the image looks real/fake?"}]}]}
        try:
            r = requests.post(f"{endpoint}/chat/completions", json=body, timeout=timeout)
            txt = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log(f"  judge err {os.path.basename(img)}: {e}")
            continue
        first = txt.strip().split(".")[0].lower()
        is_fake = ("real" not in first) and ("fake" in first)
        n_fake += int(is_fake)
        n += 1
    return (n_fake / n if n else None), n


def set_lora(endpoint, path):
    if path is None:
        requests.post(f"{endpoint}/unload_lora_adapter", json={"lora_name": "defender"}, timeout=30)
        log("  detector -> base (no LoRA)")
    else:
        r = requests.post(f"{endpoint}/load_lora_adapter",
                          json={"lora_name": "defender", "lora_path": os.path.abspath(path)}, timeout=60)
        log(f"  detector -> {path} ({r.json().get('status')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default="/data/disk4/lyx_ICML/self_evolution_forgery")
    ap.add_argument("--py", default="/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python")
    ap.add_argument("--out", required=True, help="run dir; m3 attacker state lives in <out>/m3")
    ap.add_argument("--endpoint", default="http://localhost:8002/v1")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--briefs", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--src-pool", nargs="+", required=True)
    ap.add_argument("--train-gpu", default="1")
    ap.add_argument("--replay-per-round", type=int, default=8)
    ap.add_argument("--guard-drop", type=float, default=0.15,
                    help="reject D_{R+1} if prior-forgery recall drops more than this")
    ap.add_argument("--guard-real-floor", type=float, default=0.80,
                    help="reject D_{R+1} if real-acc on held-out reals falls below this "
                         "(blocks a defender that wins by paranoia). 0 disables.")
    ap.add_argument("--guard-real-frac", type=float, default=0.25,
                    help="fraction of the real pool held out from training for the real-acc guard")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    # the orchestrator subprocess runs with cwd=src/, so ALL paths it receives must
    # be absolute or they resolve under src/outputs (silent data loss).
    args.out = os.path.abspath(args.out)
    args.proj = os.path.abspath(args.proj)
    args.src_pool = [os.path.abspath(p) for p in args.src_pool]

    proj = Path(args.proj)
    m3 = Path(args.out) / "m3"
    coevo = m3 / "coevo"
    lora_root = coevo / "lora"
    coevo.mkdir(parents=True, exist_ok=True)
    lora_root.mkdir(parents=True, exist_ok=True)
    fao = m3 / "face_attack_outputs"
    sft_path = m3 / "defender_sft_v2.jsonl"
    coevo_scripts = proj / "scripts" / "coevo"

    # round 0 starts from the frozen base detector
    set_lora(args.endpoint, None)
    prev_lora = None
    curve = []

    for R in range(args.rounds):
        log(f"════════ ROUND {R} (detector = {'base' if prev_lora is None else Path(prev_lora).name}) ════════")
        before = list_outputs(fao)

        # 1. attacker round (shared out dir -> skills accumulate = attacker self-evolves)
        cmd = [args.py, "orchestrator.py", "--mode", "v2", "--rounds", "1",
               "--briefs", str(args.briefs), "--rollouts", str(args.rollouts),
               "--multi-agent-preset", "w6_full",
               "--tier2-backend", "fakevlm_local", "--fakevlm-endpoint", args.endpoint,
               "--src-pool", *args.src_pool, "--out", str(m3)]
        runlog = open(coevo / f"attacker_r{R}.log", "w")
        log(f"  attacker: orchestrator --rounds 1 (briefs={args.briefs} rollouts={args.rollouts})")
        rc = subprocess.run(cmd, cwd=str(proj / "src"), stdout=runlog, stderr=subprocess.STDOUT)
        runlog.close()
        if rc.returncode != 0:
            log(f"  !! attacker round failed (rc={rc.returncode}); see attacker_r{R}.log"); sys.exit(1)

        # 2. isolate this round's NEW forgery images, join with bypass labels
        new_imgs = list_outputs(fao) - before
        bypass_by_img = {}
        if sft_path.exists():
            for l in sft_path.read_text().splitlines():
                l = l.strip()
                if not l:
                    continue
                d = json.loads(l)
                img = d.get("image")
                if img:
                    bypass_by_img[img] = bool(d.get("meta", {}).get("bypass_succeeded", False))
        round_sft = coevo / f"round_sft_r{R}.jsonl"
        n_new, n_byp = 0, 0
        with open(round_sft, "w") as f:
            for img in sorted(new_imgs):
                if img not in bypass_by_img:
                    continue
                byp = bypass_by_img[img]
                f.write(json.dumps({"image": img, "meta": {"bypass_succeeded": byp}}) + "\n")
                n_new += 1
                n_byp += int(byp)
        bypass_rate = (n_byp / n_new) if n_new else 0.0
        log(f"  round forgeries: {n_new} new (bypassed D_{R}: {n_byp}, rate={bypass_rate:.0%})")

        # 3. assemble training data (+replay +reals) and a guard set
        train_data = coevo / f"train_r{R}.jsonl"
        rc = subprocess.run([args.py, str(coevo_scripts / "build_round_data.py"),
                             "--round", str(R), "--attacker-sft", str(round_sft),
                             "--coevo-dir", str(coevo), "--src-real", *args.src_pool,
                             "--replay-per-round", str(args.replay_per_round),
                             "--guard-real-frac", str(args.guard_real_frac),
                             "--out-data", str(train_data)])
        if rc.returncode != 0:
            log("  !! build_round_data failed"); sys.exit(1)

        # 4. train D_{R+1}, warm from D_R (Evolving)
        out_lora = lora_root / f"r{R}"
        log(f"  training D_{R+1} on GPU {args.train_gpu} (warm from {'-' if not prev_lora else Path(prev_lora).name})")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.train_gpu)
        tcmd = [args.py, str(coevo_scripts / "train_defender_round.py"),
                "--data", str(train_data), "--out", str(out_lora),
                "--epochs", str(args.epochs), "--device", "cuda:0"]
        if prev_lora:
            tcmd += ["--prev-lora", str(prev_lora)]
        tlog = open(coevo / f"train_r{R}.log", "w")
        rc = subprocess.run(tcmd, env=env, stdout=tlog, stderr=subprocess.STDOUT)
        tlog.close()
        if rc.returncode != 0 or not (out_lora / "adapter_config.json").exists():
            log(f"  !! training failed (rc={rc.returncode}); see train_r{R}.log"); sys.exit(1)

        # 5. Guard: (a) STaSC Non-Decreasing on prior forgeries (recall must not drop),
        #    (b) M3-overhaul real-acc floor on never-trained held-out reals. Without (b)
        #    a defender wins trivially by flagging EVERYTHING fake (recall=1, real-acc=0)
        #    -> the arms race degenerates. (b) rejects that paranoid collapse.
        guard_imgs = []
        gpath = coevo / f"guard_set_r{R}.jsonl"
        if gpath.exists():
            guard_imgs = [json.loads(l)["image"] for l in gpath.read_text().splitlines() if l.strip()]
        guard_reals = []
        grpath = coevo / f"guard_reals_r{R}.jsonl"
        if grpath.exists():
            guard_reals = [json.loads(l)["image"] for l in grpath.read_text().splitlines() if l.strip()]

        accepted = True
        recall_old = recall_new = real_acc_new = None
        reject_reason = None

        if guard_imgs:                                    # D_R still loaded -> baseline recall
            recall_old, _ = judge_fake_rate(args.endpoint, guard_imgs)
        set_lora(args.endpoint, out_lora)                 # swap in candidate D_{R+1}
        if guard_imgs:
            recall_new, _ = judge_fake_rate(args.endpoint, guard_imgs)
        if guard_reals:
            fake_on_reals, _ = judge_fake_rate(args.endpoint, guard_reals)
            real_acc_new = (1.0 - fake_on_reals) if fake_on_reals is not None else None

        if recall_old is not None and recall_new is not None and recall_new < recall_old - args.guard_drop:
            reject_reason = (f"prior-forgery recall {recall_old:.0%}->{recall_new:.0%} "
                             f"(drop > {args.guard_drop:.0%})")
        elif (args.guard_real_floor > 0 and real_acc_new is not None
              and real_acc_new < args.guard_real_floor):
            reject_reason = (f"held-out real-acc {real_acc_new:.0%} < floor "
                             f"{args.guard_real_floor:.0%} (defender too paranoid)")

        if reject_reason:
            accepted = False
            log(f"  guard REJECT: {reject_reason}; reverting to D_{R}")
            set_lora(args.endpoint, prev_lora)
        else:
            log(f"  guard OK: recall {recall_old}->{recall_new}, real_acc={real_acc_new}")

        if accepted:
            prev_lora = str(out_lora)

        curve.append({"round": R, "bypass_rate_vs_D_R": bypass_rate,
                      "n_forgeries": n_new, "n_bypass": n_byp,
                      "guard_recall_old": recall_old, "guard_recall_new": recall_new,
                      "guard_real_acc_new": real_acc_new,
                      "defender_accepted": accepted,
                      "detector_after": "base" if prev_lora is None else Path(prev_lora).name})
        (coevo / "armsrace_curve.json").write_text(json.dumps(curve, indent=2))
        log(f"  round {R} done. curve so far: " +
            ", ".join(f"R{c['round']}={c['bypass_rate_vs_D_R']:.0%}" for c in curve))

    log("════════ CO-EVOLUTION COMPLETE ════════")
    log("arms-race curve (bypass_rate vs the round's detector D_R):")
    for c in curve:
        log(f"  R{c['round']}: bypass={c['bypass_rate_vs_D_R']:.0%} "
            f"({c['n_bypass']}/{c['n_forgeries']})  defender={c['detector_after']}  "
            f"accepted={c['defender_accepted']}")
    print(json.dumps(curve, indent=2))


if __name__ == "__main__":
    main()
