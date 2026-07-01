"""Method-3 co-evolution driver v2 — anti-collapse levers (NEW file; v1 untouched).

The v1 loop (run_coevolution.py) collapses: a gradient-trained defender retrained
EVERY round crushes the API-only (non-fine-tunable) attacker by round 1, bypass->0,
no arms race. v2 adds three literature-grounded levers that keep the loop ALIVE
without touching the attacker, all as driver-level knobs (no edits to v1 / train):

  (1) DEFENDER THROTTLE  --defender-period K   [TTUR, 1706.08500]
      Retrain the defender only every K rounds. Between updates the attacker hits a
      FROZEN detector for K rounds, so the slow (in-context) player gets time on the
      clock to find a foothold before the fast (gradient) player adapts.

  (2) WEAK / TWO-TIMESCALE DEFENDER UPDATE  --lora-r 8 --epochs 1 --lr 1e-5
      Each defender update is a SMALL step (rank-8, 1 epoch) that deliberately
      under-fits the newest attack family, leaving exploitable headroom — the
      "slower discriminator" side of TTUR. (Passed straight to train_defender_round.)

  (3) BYPASS FLOOR  --bypass-floor 0.10        [ACE residual-ASR, 2511.19218]
      Reject a candidate D_{R+1} that drives attacker bypass on the round's own
      forgeries below the floor (defender "winning too hard"). Converts the guard
      from "reward paranoia" into "preserve a gap" so the race can't degenerate.

Plus v1's STaSC Non-Decreasing guard + real-acc floor (forgetting + paranoia fix).
Logs armsrace_curve_v2.json with detector-update events marked.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# reuse v1 helpers (no duplication, no edits to v1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_coevolution import log, list_outputs, judge_fake_rate, set_lora  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default="/data/disk4/lyx_ICML/self_evolution_forgery")
    ap.add_argument("--py", default="/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python")
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="http://localhost:8002/v1")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--briefs", type=int, default=6)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--src-pool", nargs="+", required=True)
    ap.add_argument("--train-gpu", default="1")
    ap.add_argument("--replay-per-round", type=int, default=8)
    ap.add_argument("--guard-drop", type=float, default=0.15)
    ap.add_argument("--guard-real-floor", type=float, default=0.80)
    ap.add_argument("--guard-real-frac", type=float, default=0.25)
    # ── v2 anti-collapse levers ──
    ap.add_argument("--defender-period", type=int, default=2,
                    help="retrain defender only every K rounds (TTUR throttle). K=1 = v1.")
    ap.add_argument("--lora-r", type=int, default=8, help="weak/two-timescale LoRA rank")
    ap.add_argument("--epochs", type=int, default=1, help="weak update: 1 epoch/round")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bypass-floor", type=float, default=0.10,
                    help="reject D_{R+1} if it drives round-forgery bypass below this "
                         "(ACE residual-ASR; preserve a gap). 0 disables.")
    ap.add_argument("--detector-base", default=None,
                    help="train LoRA on THIS base instead of the strong FakeVLM ckpt "
                         "(weak-start: pass the vanilla llava the :8006 server serves so "
                         "the adapter composes on the naive detector). Default = strong ckpt.")
    ap.add_argument("--preset", choices=["w1_cheap", "w6_full"], default="w1_cheap",
                    help="L2 fan-out preset for the attacker orchestrator (cheap=all flash).")
    args = ap.parse_args()

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

    set_lora(args.endpoint, None)
    prev_lora = None
    curve = []

    for R in range(args.rounds):
        train_this_round = (R % args.defender_period == 0)
        det_name = "base" if prev_lora is None else Path(prev_lora).name
        log(f"════ ROUND {R} (detector={det_name}, train={'YES' if train_this_round else 'no(throttle)'}) ════")
        before = list_outputs(fao)

        # 1. attacker round vs the CURRENT (possibly frozen-by-throttle) detector
        cmd = [args.py, "orchestrator.py", "--mode", "v2", "--rounds", "1",
               "--briefs", str(args.briefs), "--rollouts", str(args.rollouts),
               "--multi-agent-preset", args.preset,
               "--tier2-backend", "fakevlm_local", "--fakevlm-endpoint", args.endpoint,
               "--src-pool", *args.src_pool, "--out", str(m3)]
        runlog = open(coevo / f"attacker_r{R}.log", "w")
        log(f"  attacker: orchestrator --rounds 1 (briefs={args.briefs} rollouts={args.rollouts})")
        rc = subprocess.run(cmd, cwd=str(proj / "src"), stdout=runlog, stderr=subprocess.STDOUT)
        runlog.close()
        if rc.returncode != 0:
            log(f"  !! attacker round failed (rc={rc.returncode})"); sys.exit(1)

        # 2. isolate this round's forgeries + bypass labels
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
        round_imgs, n_new, n_byp = [], 0, 0
        with open(round_sft, "w") as f:
            for img in sorted(new_imgs):
                if img not in bypass_by_img:
                    continue
                byp = bypass_by_img[img]
                f.write(json.dumps({"image": img, "meta": {"bypass_succeeded": byp}}) + "\n")
                round_imgs.append(img)
                n_new += 1
                n_byp += int(byp)
        bypass_rate = (n_byp / n_new) if n_new else 0.0
        log(f"  round forgeries: {n_new} (sandbox-bypassed D_{R}: {n_byp}, rate={bypass_rate:.0%})")

        # ── HONEST DETECTOR-LEVEL bypass (weak-start headline) ──
        # The sandbox bypass above is gated by face_valid (tier1 face/identity check),
        # which masks the DETECTOR axis we are actually evolving. For the weak-start
        # suppression curve we measure how often the CURRENT detector D_R calls THIS
        # round's forgeries "real", over ALL new forgery images (not just face_valid
        # ones). r0 (naive base) -> high; trained rounds -> drops = the hardening curve.
        all_round_imgs = sorted(new_imgs)
        (coevo / f"round_all_r{R}.jsonl").write_text(
            "\n".join(json.dumps({"image": p}) for p in all_round_imgs))
        det_bypass = None
        if all_round_imgs:
            fr_all, _ = judge_fake_rate(args.endpoint, all_round_imgs)
            det_bypass = (1.0 - fr_all) if fr_all is not None else None
        log(f"  detector-level bypass vs D_{R} over ALL {len(all_round_imgs)} forgeries: "
            f"{det_bypass:.0%}" if det_bypass is not None else "  detector-level bypass: n/a")

        accepted = None
        recall_old = recall_new = real_acc_new = cand_bypass = None
        reject_reason = None

        if not train_this_round:
            log(f"  throttle: defender frozen at {det_name} this round (no train)")
        else:
            # 3. build training data (+replay +reals +held-out real guard)
            train_data = coevo / f"train_r{R}.jsonl"
            rc = subprocess.run([args.py, str(coevo_scripts / "build_round_data.py"),
                                 "--round", str(R), "--attacker-sft", str(round_sft),
                                 "--coevo-dir", str(coevo), "--src-real", *args.src_pool,
                                 "--replay-per-round", str(args.replay_per_round),
                                 "--guard-real-frac", str(args.guard_real_frac),
                                 "--out-data", str(train_data)])
            if rc.returncode != 0:
                log("  !! build_round_data failed"); sys.exit(1)

            # 4. WEAK / two-timescale defender update (rank-r, few epochs), warm from D_R
            out_lora = lora_root / f"r{R}"
            log(f"  training D_{R+1} (rank={args.lora_r} epochs={args.epochs} lr={args.lr}) "
                f"GPU {args.train_gpu} warm={'-' if not prev_lora else Path(prev_lora).name}")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.train_gpu)
            tcmd = [args.py, str(coevo_scripts / "train_defender_round.py"),
                    "--data", str(train_data), "--out", str(out_lora),
                    "--epochs", str(args.epochs), "--lr", str(args.lr),
                    "--lora-r", str(args.lora_r), "--device", "cuda:0"]
            if args.detector_base:
                tcmd += ["--base", args.detector_base]
            if prev_lora:
                tcmd += ["--prev-lora", str(prev_lora)]
            tlog = open(coevo / f"train_r{R}.log", "w")
            rc = subprocess.run(tcmd, env=env, stdout=tlog, stderr=subprocess.STDOUT)
            tlog.close()
            if rc.returncode != 0 or not (out_lora / "adapter_config.json").exists():
                log(f"  !! training failed (rc={rc.returncode})"); sys.exit(1)

            # 5. guards: (a) Non-Decreasing on prior forgeries, (b) real-acc floor,
            #    (c) v2 BYPASS FLOOR on this round's forgeries (preserve a gap).
            guard_imgs, guard_reals = [], []
            gp = coevo / f"guard_set_r{R}.jsonl"
            if gp.exists():
                guard_imgs = [json.loads(l)["image"] for l in gp.read_text().splitlines() if l.strip()]
            grp = coevo / f"guard_reals_r{R}.jsonl"
            if grp.exists():
                guard_reals = [json.loads(l)["image"] for l in grp.read_text().splitlines() if l.strip()]

            if guard_imgs:
                recall_old, _ = judge_fake_rate(args.endpoint, guard_imgs)
            set_lora(args.endpoint, out_lora)  # mount candidate D_{R+1}
            if guard_imgs:
                recall_new, _ = judge_fake_rate(args.endpoint, guard_imgs)
            if guard_reals:
                fr, _ = judge_fake_rate(args.endpoint, guard_reals)
                real_acc_new = (1.0 - fr) if fr is not None else None
            if round_imgs and args.bypass_floor > 0:
                fr_round, _ = judge_fake_rate(args.endpoint, round_imgs)
                cand_bypass = (1.0 - fr_round) if fr_round is not None else None

            if recall_old is not None and recall_new is not None and recall_new < recall_old - args.guard_drop:
                reject_reason = f"prior-forgery recall {recall_old:.0%}->{recall_new:.0%} drop>{args.guard_drop:.0%}"
            elif (args.guard_real_floor > 0 and real_acc_new is not None
                  and real_acc_new < args.guard_real_floor):
                reject_reason = f"held-out real-acc {real_acc_new:.0%} < floor {args.guard_real_floor:.0%}"
            elif (args.bypass_floor > 0 and cand_bypass is not None
                  and cand_bypass < args.bypass_floor):
                reject_reason = (f"candidate drives bypass {cand_bypass:.0%} < floor "
                                 f"{args.bypass_floor:.0%} (defender winning too hard; keep gap)")

            if reject_reason:
                accepted = False
                log(f"  guard REJECT: {reject_reason}; revert to D_{R}")
                set_lora(args.endpoint, prev_lora)
            else:
                accepted = True
                prev_lora = str(out_lora)
                log(f"  guard OK: recall {recall_old}->{recall_new} real_acc={real_acc_new} cand_bypass={cand_bypass}")

        curve.append({"round": R, "trained": train_this_round,
                      "det_bypass_vs_D_R": det_bypass, "n_all_forgeries": len(all_round_imgs),
                      "bypass_rate_vs_D_R": bypass_rate, "n_forgeries": n_new, "n_bypass": n_byp,
                      "guard_recall_old": recall_old, "guard_recall_new": recall_new,
                      "guard_real_acc_new": real_acc_new, "cand_bypass": cand_bypass,
                      "defender_accepted": accepted,
                      "detector_after": "base" if prev_lora is None else Path(prev_lora).name})
        (coevo / "armsrace_curve_v2.json").write_text(json.dumps(curve, indent=2))
        log("  curve (detector-level): " + ", ".join(
            f"R{c['round']}=" + (f"{c['det_bypass_vs_D_R']:.0%}" if c['det_bypass_vs_D_R'] is not None else "na")
            + ('*' if c['trained'] else '') for c in curve))

    log("════ CO-EVOLUTION v2 COMPLETE ════")
    print(json.dumps(curve, indent=2))


if __name__ == "__main__":
    main()
