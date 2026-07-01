"""Method-5 driver: POPULATION archive co-evolution (the flagship).

M2 = single attacker self-evolving vs a FROZEN detector.
M3 = single attacker co-evolving vs a TRAINABLE detector (one lineage).
M5 = a POPULATION of K attacker lineages co-evolving vs the same trainable
     detector, with a shared coverage archive + novelty-based lineage selection.

Literature imports (repos cloned under literature/repos/):
  • DGM (2505.22954)   : population/archive of agent lineages, not a single trajectory.
  • GEA (2602.04837)   : shared archive + novelty selection -> preserves attack diversity
                          against a strengthening defender (don't just chase mean bypass).
  • HGM (2510.21614)   : promote a lineage by its *descendant* coverage productivity, not
                          its instantaneous bypass (Clade-Metaproductivity, simplified to
                          cumulative-new-coverage here).
  • EvoTest (2510.13220): candidate defender accepted only if it improves a BALANCED
                          objective (recall AND real-acc) -> reuses run_coevolution's guard.

Per round R:
  1. Every lineage attacks the CURRENT detector D_R (parallel subprocesses, shared endpoint).
  2. Read each lineage's per-family bypass (reports/) -> lineage coverage set.
  3. Novelty selection: rank lineages by NEW families added to the shared archive
     (GEA). coverage_cum (the population headline curve) = |shared archive|.
  4. Union ALL lineages' new forgeries -> train defender D_{R+1} (+replay +reals).
  5. Guard with the M3-overhaul real-acc floor, hot-reload. Next round is harder.

Headline figure: population coverage_cum + per-round bypass (oscillating arms race),
directly comparable to M3's single-lineage curve and to the QD/ARMs literature.
"""
from __future__ import annotations
import argparse, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_coevolution import log, list_outputs, judge_fake_rate, set_lora  # noqa: E402


def jaccard_dist(a, b):
    """Behavioral distance between two lineages' family-coverage sets (GEA novelty)."""
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def gea_score(stat, others, children):
    """GEA selection score: performance x sqrt(behavioral novelty), boosted by new
    archive coverage (HGM productivity) and damped by DGM child-count (don't over-exploit
    one parent). A small performance floor keeps purely-novel lineages rankable in the
    defender-dominates regime where every bypass_rate is ~0."""
    fams = set(stat["families_bypassed"])
    perf = stat["global_bypass_rate"]
    if perf is None:
        perf = (stat["n_bypass"] / stat["n_forgeries"]) if stat["n_forgeries"] else 0.0
    nov = (sum(jaccard_dist(fams, o["families_bypassed"]) for o in others) / len(others)
           if others else 1.0)
    new_bonus = 1.0 + stat["n_new_families"]
    child_pen = 1.0 / (1.0 + children.get(stat["lineage"], 0))
    return (perf + 0.05) * math.sqrt(nov + 1e-3) * new_bonus * child_pen


def branch_lineage(parent_dir: Path, victim_dir: Path):
    """DGM/GEA steady-state branching: reseed the victim lineage's accumulated memory/skill
    state from the parent so the population concentrates on productive lineages while keeping
    K live slots. Forgery streams (face_attack_outputs) stay per-lineage so the round-diff
    bookkeeping is unaffected."""
    for sub in ("reasoning_bank", "chroma_persist"):
        src, dst = parent_dir / sub, victim_dir / sub
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    src = parent_dir / "novelty_history.json"
    if src.exists():
        shutil.copy2(src, victim_dir / "novelty_history.json")


def latest_report(lin_dir: Path):
    """Newest reports/r*_v2.json for a lineage; returns parsed dict or None."""
    reps = sorted((lin_dir / "reports").glob("r*_v2.json"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not reps:
        return None
    try:
        return json.loads(reps[0].read_text())
    except Exception:
        return None


def covered_families(report, thresh=0.0):
    """Families this lineage BYPASSED this round (bypass rate > thresh)."""
    if not report:
        return set()
    fbr = report.get("diagnosis", {}).get("family_bypass_rates", {}) or {}
    return {fam for fam, rate in fbr.items() if (rate or 0.0) > thresh}


def isolate_round_forgeries(fao_dir, before_set, sft_path):
    """Diff a lineage's face_attack_outputs, join with its defender_sft bypass labels.
    Returns (list[(image, bypass)], n_new, n_bypass)."""
    new_imgs = list_outputs(fao_dir) - before_set
    bypass_by_img = {}
    if sft_path.exists():
        for l in sft_path.read_text().splitlines():
            l = l.strip()
            if not l:
                continue
            try:
                d = json.loads(l)
            except Exception:
                continue
            img = d.get("image")
            if img:
                bypass_by_img[img] = bool(d.get("meta", {}).get("bypass_succeeded", False))
    out = []
    for img in sorted(new_imgs):
        if img in bypass_by_img:
            out.append((img, bypass_by_img[img]))
    n_byp = sum(1 for _, b in out if b)
    return out, len(out), n_byp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default="/data/disk4/lyx_ICML/self_evolution_forgery")
    ap.add_argument("--py", default="/cpfs01/bob_workspace/miniconda3/envs/fakevlm/bin/python")
    ap.add_argument("--out", required=True, help="run dir; lineages live in <out>/m5/lin_*")
    ap.add_argument("--endpoint", default="http://localhost:8002/v1")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--lineages", type=int, default=3, help="K attacker lineages (population)")
    ap.add_argument("--briefs", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=2)
    ap.add_argument("--src-pool", nargs="+", required=True)
    ap.add_argument("--train-gpu", default="1")
    ap.add_argument("--replay-per-round", type=int, default=8)
    ap.add_argument("--guard-drop", type=float, default=0.15)
    ap.add_argument("--guard-real-floor", type=float, default=0.80)
    ap.add_argument("--guard-real-frac", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--preset", choices=["w1_cheap", "w6_full"], default="w1_cheap",
                    help="L2 fan-out preset for the attacker lineages (cheap=all flash).")
    ap.add_argument("--detector-base", default=None,
                    help="train LoRA on THIS base instead of the strong FakeVLM ckpt "
                         "(weak-start: pass the vanilla llava the served endpoint uses).")
    args = ap.parse_args()

    args.out = os.path.abspath(args.out)
    args.proj = os.path.abspath(args.proj)
    args.src_pool = [os.path.abspath(p) for p in args.src_pool]

    proj = Path(args.proj)
    m5 = Path(args.out) / "m5"
    coevo = m5 / "coevo"          # SHARED across lineages (replay store, guard, lora)
    lora_root = coevo / "lora"
    coevo.mkdir(parents=True, exist_ok=True)
    lora_root.mkdir(parents=True, exist_ok=True)
    coevo_scripts = proj / "scripts" / "coevo"

    lineages = [m5 / f"lin_{k}" for k in range(args.lineages)]
    for lin in lineages:
        (lin / "face_attack_outputs").mkdir(parents=True, exist_ok=True)

    set_lora(args.endpoint, None)
    prev_lora = None
    shared_archive = set()        # GEA cumulative coverage archive (families ever bypassed)
    children_count = {k: 0 for k in range(args.lineages)}  # DGM: times a lineage seeded a child
    genealogy = []                # per-round (parent -> victim) branch events
    curve = []

    for R in range(args.rounds):
        det = "base" if prev_lora is None else Path(prev_lora).name
        log(f"════════ M5 ROUND {R} (detector = {det}, K={args.lineages} lineages) ════════")

        before = {k: list_outputs(lin / "face_attack_outputs") for k, lin in enumerate(lineages)}

        # 1. all lineages attack D_R in parallel (shared endpoint; distinct out dirs)
        procs = []
        for k, lin in enumerate(lineages):
            cmd = [args.py, "legacy/orchestrator.py", "--mode", "v2", "--rounds", "1",
                   "--briefs", str(args.briefs), "--rollouts", str(args.rollouts),
                   "--multi-agent-preset", args.preset,
                   "--tier2-backend", "fakevlm_local", "--fakevlm-endpoint", args.endpoint,
                   "--src-pool", *args.src_pool, "--out", str(lin)]
            lg = open(coevo / f"attacker_r{R}_lin{k}.log", "w")
            # orchestrator.py imports siblings from BOTH src/ (viviai_client) and
            # src/legacy/ (trajectory_schema); give it an absolute PYTHONPATH for both.
            att_env = dict(os.environ)
            _pp = os.pathsep.join([str(proj / "src"), str(proj / "src" / "legacy")])
            att_env["PYTHONPATH"] = (_pp + os.pathsep + att_env["PYTHONPATH"]
                                     if att_env.get("PYTHONPATH") else _pp)
            p = subprocess.Popen(cmd, cwd=str(proj / "src"), stdout=lg,
                                 stderr=subprocess.STDOUT, env=att_env)
            procs.append((k, p, lg))
            log(f"  lineage {k}: orchestrator launched (pid {p.pid})")
        for k, p, lg in procs:
            rc = p.wait()
            lg.close()
            if rc != 0:
                log(f"  !! lineage {k} attacker failed (rc={rc}); see attacker_r{R}_lin{k}.log")
                sys.exit(1)

        # 2-3. per-lineage coverage + forgeries; novelty selection (GEA)
        all_forgeries = []        # union across lineages for defender training
        lineage_stats = []
        round_new_families = set()
        for k, lin in enumerate(lineages):
            rep = latest_report(lin)
            fams = covered_families(rep)
            forg, n_new, n_byp = isolate_round_forgeries(
                lin / "face_attack_outputs", before[k], lin / "defender_sft_v2.jsonl")
            all_forgeries.extend(forg)
            new_fams = fams - shared_archive
            round_new_families |= fams
            g_byp = (rep or {}).get("diagnosis", {}).get("global_bypass_rate", None)
            lineage_stats.append({"lineage": k, "families_bypassed": sorted(fams),
                                  "new_families": sorted(new_fams), "n_new_families": len(new_fams),
                                  "n_forgeries": n_new, "n_bypass": n_byp,
                                  "global_bypass_rate": g_byp})
            log(f"  lineage {k}: bypass_fams={sorted(fams)} NEW={sorted(new_fams)} "
                f"forg={n_new} byp={n_byp}")

        # GEA population selection: score every lineage by performance x novelty x new-coverage,
        # damped by DGM child-count, then BRANCH (reseed the worst slot from the best) so the
        # population genuinely evolves instead of being K independent lineages.
        for s in lineage_stats:
            others = [o for o in lineage_stats if o["lineage"] != s["lineage"]]
            s["gea_score"] = gea_score(s, others, children_count)
        scored = sorted(lineage_stats, key=lambda s: s["gea_score"], reverse=True)
        best, worst = scored[0], scored[-1]
        shared_archive |= round_new_families
        coverage_cum = len(shared_archive)
        log(f"  >> GEA scores: " + ", ".join(
            f"lin{s['lineage']}={s['gea_score']:.3f}" for s in scored))
        log(f"  >> best lineage={best['lineage']} (score {best['gea_score']:.3f}, "
            f"+{best['n_new_families']} new fams); coverage_cum={coverage_cum} "
            f"archive={sorted(shared_archive)}")

        branch_event = None
        # branch only on non-final rounds (next round must exist to use the reseeded slot)
        if (args.lineages >= 2 and best["lineage"] != worst["lineage"]
                and best["gea_score"] > worst["gea_score"] and R < args.rounds - 1):
            branch_lineage(lineages[best["lineage"]], lineages[worst["lineage"]])
            children_count[best["lineage"]] += 1
            branch_event = {"round": R, "parent": best["lineage"], "victim": worst["lineage"],
                            "parent_score": best["gea_score"], "victim_score": worst["gea_score"]}
            genealogy.append(branch_event)
            log(f"  >> BRANCH: lineage {worst['lineage']} (score {worst['gea_score']:.3f}) "
                f"reseeded as descendant of lineage {best['lineage']} "
                f"(score {best['gea_score']:.3f}); children_count={children_count}")

        # 4. union forgeries -> one merged round_sft for the defender
        round_sft = coevo / f"round_sft_r{R}.jsonl"
        with open(round_sft, "w") as f:
            for img, byp in all_forgeries:
                f.write(json.dumps({"image": img, "meta": {"bypass_succeeded": byp}}) + "\n")
        n_pop = len(all_forgeries)
        n_pop_byp = sum(1 for _, b in all_forgeries if b)
        pop_bypass_rate = (n_pop_byp / n_pop) if n_pop else 0.0
        log(f"  population forgeries: {n_pop} (bypassed D_{R}: {n_pop_byp}, "
            f"rate={pop_bypass_rate:.0%})")

        train_data = coevo / f"train_r{R}.jsonl"
        rc = subprocess.run([args.py, str(coevo_scripts / "build_round_data.py"),
                             "--round", str(R), "--attacker-sft", str(round_sft),
                             "--coevo-dir", str(coevo), "--src-real", *args.src_pool,
                             "--replay-per-round", str(args.replay_per_round),
                             "--guard-real-frac", str(args.guard_real_frac),
                             "--out-data", str(train_data)])
        if rc.returncode != 0:
            log("  !! build_round_data failed"); sys.exit(1)

        # 4b. train defender D_{R+1} (warm from D_R)
        out_lora = lora_root / f"r{R}"
        log(f"  training D_{R+1} on GPU {args.train_gpu} (warm from {'-' if not prev_lora else Path(prev_lora).name})")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.train_gpu)
        tcmd = [args.py, str(coevo_scripts / "train_defender_round.py"),
                "--data", str(train_data), "--out", str(out_lora),
                "--epochs", str(args.epochs), "--device", "cuda:0"]
        if args.detector_base:
            tcmd += ["--base", args.detector_base]
        if prev_lora:
            tcmd += ["--prev-lora", str(prev_lora)]
        tlog = open(coevo / f"train_r{R}.log", "w")
        rc = subprocess.run(tcmd, env=env, stdout=tlog, stderr=subprocess.STDOUT)
        tlog.close()
        if rc.returncode != 0 or not (out_lora / "adapter_config.json").exists():
            log(f"  !! training failed (rc={rc.returncode}); see train_r{R}.log"); sys.exit(1)

        # 5. guard: recall non-decreasing AND real-acc floor (M3-overhaul)
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
        if guard_imgs:
            recall_old, _ = judge_fake_rate(args.endpoint, guard_imgs)
        set_lora(args.endpoint, out_lora)
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

        curve.append({"round": R, "population_bypass_rate": pop_bypass_rate,
                      "coverage_cum": coverage_cum,
                      "best_lineage": best["lineage"], "best_new_families": best["n_new_families"],
                      "best_score": best["gea_score"], "branch": branch_event,
                      "children_count": dict(children_count),
                      "n_forgeries": n_pop, "n_bypass": n_pop_byp,
                      "lineages": lineage_stats,
                      "guard_recall_old": recall_old, "guard_recall_new": recall_new,
                      "guard_real_acc_new": real_acc_new, "defender_accepted": accepted,
                      "shared_archive": sorted(shared_archive),
                      "detector_after": "base" if prev_lora is None else Path(prev_lora).name})
        (coevo / "armsrace_curve_m5.json").write_text(json.dumps(curve, indent=2))
        log(f"  round {R} done. coverage_cum so far: " +
            ", ".join(f"R{c['round']}={c['coverage_cum']}" for c in curve))

    (coevo / "genealogy_m5.json").write_text(json.dumps(genealogy, indent=2))
    log("════════ M5 POPULATION CO-EVOLUTION COMPLETE ════════")
    log(f"  branch events: {genealogy}")
    for c in curve:
        log(f"  R{c['round']}: pop_bypass={c['population_bypass_rate']:.0%} "
            f"coverage_cum={c['coverage_cum']} best_lin={c['best_lineage']} "
            f"defender={c['detector_after']} accepted={c['defender_accepted']} "
            f"real_acc={c['guard_real_acc_new']}")
    print(json.dumps(curve, indent=2))


if __name__ == "__main__":
    main()
