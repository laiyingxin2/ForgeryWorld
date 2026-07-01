"""Co-evolution round driver — the OUTER-OUTER arms-race loop.

This is the piece that makes the framework a *self-evolving* red-team in the sense
the 2025-26 literature uses the term (CEMMA / MAGIC / CHASE / ShellForge): the
attacker and the defender are alternately updated, with the QD archive acting as the
data interface between them (CEMMA). Without it, our two-layer system is only an
open-ended QD attacker against a FROZEN target — which is why the budget-100 run's
bypass-coverage plateaus (the literature-predicted behavior of a static-target run).

Design (our own content; ShellForge "multi-view detector retrained on accumulated
evasions" + graded fitness to un-degenerate the QD search):

  Defender = FakeVLM (frozen, :8001) AND a trainable SURROGATE.
    The surrogate is a lightweight logistic regression over the forensic feature
    vector sandbox.py *already* computes for every verify() (fft / niqe / maniqa /
    landmark-symmetry / arcface / ssim / lpips) — a genuine second "view", no extra
    model or GPU. Its smooth P(real) score is what fixes the degenerate near-binary
    FakeVLM signal: MAP-Elites now has a gradient to climb toward the defender's
    blind spots.

  Round r:
    1. run the inner QD attacker against the CURRENT effective defender D_r
       (graded fitness = blend(FakeVLM real-prob, surrogate P(real));
        bypass = FakeVLM fooled AND surrogate says real).
    2. harvest every generated forgery's feature vector (the archive-as-interface).
    3. retrain the surrogate on {real seed faces = real, accumulated forgeries =
       fake} -> S_{r+1}, hardening the defender.
    4. log per-round ASR, newly-covered bypass cells, and ROBUSTNESS GAIN
       (fraction of this round's evasions the retrained surrogate now catches).

  Arms-race instrumentation (MAGIC): we flag DEGENERACY (defender domination — the
  modal co-evolution outcome) when ASR collapses to ~0 and the attacker stops
  finding new bypass cells for consecutive rounds. Genuine progression = ASR
  oscillates / new cells keep appearing while robustness gain stays > 0.

Two run modes (mirrors outer_poet):
  --mock : a synthetic detector with a fixed blind spot; validates the whole loop +
           the robustness-gain / degeneracy math with no GPU or image generation.
  real   : wraps InnerMapElites with the surrogate-fused score_fn.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_log = logging.getLogger("coevolution")

# Forensic features sandbox.tier1 produces. Order is fixed (the surrogate's input
# layout). Cross-reference metrics (arcface/ssim/lpips) are usually sentinel (-1) in
# the inner loop (no source face is passed to verify), so they standardize to ~0
# weight; the no-reference ones (fft/niqe/maniqa/landmark) carry the signal.
FEATURE_KEYS = [
    "fft_artifact_score", "niqe", "maniqa", "landmark_consistency",
    "arcface_id_sim", "ssim_vs_src", "lpips_vs_src",
]


def tier1_to_features(tier1: Dict[str, Any]) -> np.ndarray:
    """Map a sandbox tier1 dict to the fixed feature vector; sentinels -> NaN
    (imputed to the train mean at fit time)."""
    out = np.full(len(FEATURE_KEYS), np.nan, dtype=np.float64)
    for i, k in enumerate(FEATURE_KEYS):
        v = tier1.get(k) if tier1 else None
        if v is None:
            continue
        v = float(v)
        if v == -1.0:          # sandbox "no face / no source" sentinel
            continue
        out[i] = v
    return out


# ────────────────────────────── surrogate defender ──────────────────────────────

class SurrogateDefender:
    """Numpy logistic regression P(real) over the forensic feature vector.

    Dependency-free (no sklearn): standardize with stored train mean/std, impute NaN
    to the mean, full-batch gradient descent with L2. Deliberately lightweight — it
    is a *second detector view* whose job is to create arms-race pressure, not to be
    SOTA. A weak surrogate (forensic stats of high-realism bypasses ~ real faces)
    sustains the arms race; a strong one drives the defender-domination regime — both
    are legitimate, and the degeneracy guard distinguishes them.
    """

    def __init__(self, l2: float = 1e-2, lr: float = 0.3, epochs: int = 800):
        self.l2, self.lr, self.epochs = l2, lr, epochs
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.is_fitted = False
        self.train_acc: float = 0.0

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        # impute NaN -> train mean (mean is 0 after centering, so fill with mean_)
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X[inds] = np.take(self.mean_, inds[1])
        return (X - self.mean_) / self.std_

    def fit(self, pos_feats: List[np.ndarray], neg_feats: List[np.ndarray]) -> bool:
        if not pos_feats or not neg_feats:
            return False  # need both classes
        Xp, Xn = np.array(pos_feats, float), np.array(neg_feats, float)
        X = np.vstack([Xp, Xn])
        y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
        self.mean_ = np.nanmean(X, axis=0)
        self.mean_ = np.where(np.isnan(self.mean_), 0.0, self.mean_)
        std = np.nanstd(X, axis=0)
        self.std_ = np.where((std < 1e-6) | np.isnan(std), 1.0, std)
        Xs = self._prep(X)
        n, d = Xs.shape
        w = np.zeros(d)
        b = 0.0
        # class-balanced weights (bypasses are few)
        wp, wn = n / (2 * len(Xp)), n / (2 * len(Xn))
        sw = np.where(y == 1, wp, wn)
        for _ in range(self.epochs):
            z = Xs @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = (p - y) * sw
            gw = Xs.T @ g / n + self.l2 * w
            gb = g.mean()
            w -= self.lr * gw
            b -= self.lr * gb
        self.w, self.b = w, b
        self.is_fitted = True
        pred = (self.predict_real_batch(X) >= 0.5).astype(float)
        self.train_acc = float((pred == y).mean())
        return True

    def predict_real(self, feat: np.ndarray) -> float:
        if not self.is_fitted:
            return 1.0  # untrained defender admits everything (FakeVLM decides alone)
        xs = self._prep(feat.reshape(1, -1))
        z = float((xs @ self.w).item() + self.b)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def predict_real_batch(self, X: np.ndarray) -> np.ndarray:
        xs = self._prep(X)
        z = xs @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def to_json(self) -> dict:
        return {"is_fitted": self.is_fitted, "train_acc": round(self.train_acc, 4),
                "w": None if self.w is None else self.w.tolist(), "b": self.b,
                "mean": None if self.mean_ is None else self.mean_.tolist(),
                "std": None if self.std_ is None else self.std_.tolist(),
                "feature_keys": FEATURE_KEYS}


class EffectiveDefender:
    """Frozen FakeVLM fused with the current surrogate.

    fitness  : graded score MAP-Elites maximizes = blend(FakeVLM real-prob,
               surrogate P(real)). Smooth even when FakeVLM is near-binary.
    bypass   : the attacker wins only if it fools BOTH (AND-gate) — so a defender
               update that catches a forgery removes it from the bypass set.
    """

    def __init__(self, surrogate: Optional[SurrogateDefender], alpha: float = 0.5,
                 real_thresh: float = 0.5):
        self.surrogate = surrogate
        self.alpha = alpha
        self.real_thresh = real_thresh

    @property
    def active(self) -> bool:
        return self.surrogate is not None and self.surrogate.is_fitted

    def fitness(self, fakevlm_real: float, feat: np.ndarray) -> float:
        if not self.active:
            return fakevlm_real
        s = self.surrogate.predict_real(feat)
        return self.alpha * fakevlm_real + (1.0 - self.alpha) * s

    def bypass(self, fakevlm_says_real: bool, feat: np.ndarray) -> bool:
        if not fakevlm_says_real:
            return False
        if not self.active:
            return True
        return self.surrogate.predict_real(feat) >= self.real_thresh


# ────────────────────────────── round result interface ──────────────────────────────

@dataclass
class CoSample:
    cell: Tuple[str, ...]
    features: np.ndarray
    fakevlm_real: float
    eff_bypass: bool

    def to_json(self) -> dict:
        return {"cell": list(self.cell), "fakevlm_real": round(self.fakevlm_real, 4),
                "eff_bypass": self.eff_bypass}


@dataclass
class RoundResult:
    samples: List[CoSample]
    n_cells: int = 0
    archive_path: Optional[str] = None


# ────────────────────────────── the driver ──────────────────────────────

@dataclass
class RoundLog:
    round: int
    n_samples: int
    asr: float                 # attack success rate vs the effective defender D_r
    n_bypass: int
    new_bypass_cells: int      # cells bypassed for the first time this round
    cum_bypass_cells: int
    robustness_gain: float     # frac of THIS round's evasions the retrained surrogate now catches
    surrogate_trained: bool
    surrogate_train_acc: float
    degenerate: bool


class CoEvolutionDriver:
    def __init__(self, run_round: Callable[[int, EffectiveDefender], RoundResult],
                 real_features: List[np.ndarray], out_dir: Path,
                 alpha: float = 0.5, real_thresh: float = 0.5,
                 degeneracy_patience: int = 2):
        # run_round(round_idx, effective_defender) -> RoundResult  (real or mock)
        # real_features: forensic vectors of genuine faces = the surrogate's positives
        self.run_round = run_round
        self.real_features = real_features
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        self.real_thresh = real_thresh
        self.degeneracy_patience = degeneracy_patience

        self.surrogate = SurrogateDefender()
        self.neg_features: List[np.ndarray] = []     # accumulated forgeries (the archive interface)
        self.cum_bypass_cells: set = set()
        self.logs: List[RoundLog] = []

    def _degenerate(self) -> bool:
        """MAGIC-style guard: defender domination = the attacker once succeeded and
        was then driven to ASR~0 with no new bypass cells for `patience` consecutive
        rounds. The peak>0 precondition is essential: ASR~0 from the very first round
        means the attacker never bypassed at all (budget too small / operators too
        weak) — a DIFFERENT failure (attacker collapse), not defender domination."""
        if len(self.logs) < self.degeneracy_patience:
            return False
        if max(l.asr for l in self.logs) <= 1e-9:
            return False
        tail = self.logs[-self.degeneracy_patience:]
        return all(l.asr <= 1e-9 and l.new_bypass_cells == 0 for l in tail)

    def run(self, rounds: int) -> List[RoundLog]:
        for r in range(rounds):
            eff = EffectiveDefender(self.surrogate, self.alpha, self.real_thresh)
            rr = self.run_round(r, eff)
            n = len(rr.samples)
            byp = [s for s in rr.samples if s.eff_bypass]
            asr = len(byp) / n if n else 0.0
            byp_cells = {s.cell for s in byp}
            new_cells = byp_cells - self.cum_bypass_cells

            # harvest: every forgery this round becomes a defender training negative
            self.neg_features.extend(s.features for s in rr.samples)
            trained = self.surrogate.fit(self.real_features, self.neg_features)

            # robustness gain = how many of THIS round's evasions the retrained
            # surrogate would now reject (ShellForge per-round hardening signal)
            if byp and trained:
                caught = sum(1 for s in byp
                             if self.surrogate.predict_real(s.features) < self.real_thresh)
                rgain = caught / len(byp)
            else:
                rgain = 0.0

            self.cum_bypass_cells |= byp_cells
            log = RoundLog(
                round=r, n_samples=n, asr=round(asr, 4), n_bypass=len(byp),
                new_bypass_cells=len(new_cells), cum_bypass_cells=len(self.cum_bypass_cells),
                robustness_gain=round(rgain, 4), surrogate_trained=trained,
                surrogate_train_acc=round(self.surrogate.train_acc, 4),
                degenerate=False,
            )
            self.logs.append(log)
            log.degenerate = self._degenerate()
            _log.info("round %d: ASR=%.3f n_bypass=%d new_cells=%d cum_cells=%d "
                      "rgain=%.3f surr_acc=%.3f%s", r, asr, len(byp), len(new_cells),
                      len(self.cum_bypass_cells), rgain, self.surrogate.train_acc,
                      "  [DEGENERATE]" if log.degenerate else "")
            if log.degenerate:
                _log.warning("defender domination detected (ASR~0, no new cells for "
                             "%d rounds) — arms race has collapsed", self.degeneracy_patience)
                break
        if self.logs and max(l.asr for l in self.logs) <= 1e-9:
            _log.warning("ATTACKER COLLAPSE: no bypass in any round — raise --inner-budget "
                         "or strengthen operators; this is NOT defender domination")
        self._save()
        return self.logs

    def _save(self) -> None:
        from evolve.artifacts import SCHEMA_VERSION
        (self.out_dir / "coevolution.json").write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "feature_keys": FEATURE_KEYS,
            "rounds": [asdict(l) for l in self.logs],
            "final_surrogate": self.surrogate.to_json(),
            "n_real_positives": len(self.real_features),
            "n_accumulated_negatives": len(self.neg_features),
            "alpha": self.alpha, "real_thresh": self.real_thresh,
        }, indent=2, ensure_ascii=False))


# ────────────────────────────── mock factory (logic validation) ──────────────────────────────

def make_mock(rounds_seed: int = 0, n_per_round: int = 80,
              cells: Optional[List[str]] = None) -> Tuple[Callable, List[np.ndarray]]:
    """Synthetic detector with a FROZEN blind spot. A 'sample' is a forensic vector
    drawn per (synthetic) cell; FakeVLM-mock says real iff a fixed hidden direction
    fires. As the surrogate trains on accumulated evasions it learns that direction,
    so ASR vs the AND-gated defender falls round over round (robustness gain), and
    eventually the attacker is dominated (degeneracy fires) — validating the math."""
    rng = np.random.default_rng(rounds_seed)
    cells = cells or [f"fam{i}" for i in range(6)]
    d = len(FEATURE_KEYS)
    # frozen FakeVLM blind spot: a half-space in feature space it mislabels as real
    fv_dir = rng.standard_normal(d)
    fv_dir /= np.linalg.norm(fv_dir)
    fv_bias = -0.3

    # real faces (surrogate positives): cluster near origin, distinct from the blind spot
    real_features = [rng.standard_normal(d) * 0.5 for _ in range(120)]

    def run_round(r: int, eff: EffectiveDefender) -> RoundResult:
        srng = np.random.default_rng(1000 + r)
        samples = []
        for _ in range(n_per_round):
            cell = (cells[srng.integers(len(cells))],)
            feat = srng.standard_normal(d) * 0.8 + 0.6 * fv_dir  # attacker biases toward blind spot
            z = float(feat @ fv_dir) + fv_bias
            fv_real_prob = 1.0 / (1.0 + math.exp(-3.0 * z))
            fv_says_real = fv_real_prob >= 0.5
            eb = eff.bypass(fv_says_real, feat)
            samples.append(CoSample(cell=cell, features=feat,
                                    fakevlm_real=fv_real_prob, eff_bypass=eb))
        return RoundResult(samples=samples, n_cells=len(cells))

    return run_round, real_features


# ────────────────────────────── real factory (wraps InnerMapElites) ──────────────────────────────

def make_real(inner_kwargs: dict, inner_budget: int, n_seed: int,
              real_face_paths: List[str], alpha: float, real_thresh: float,
              base_out: Path, warm_cap: int = 16,
              warm_start: bool = True) -> Tuple[Callable, List[np.ndarray]]:
    """Build a run_round that drives an InnerMapElites archive per round, scored by the
    surrogate-fused EffectiveDefender, plus the real-face positives.

    warm_start: carry the top-`warm_cap` elite GENOTYPES (prioritizing prior bypasses)
    into the next round's seed population, re-scored against the hardened defender. This
    is the fix for the attacker-vs-defender asymmetry: without it the inner archive is
    rebuilt from random seeds every round, so the defender accumulates negatives while
    the attacker forgets — the forgery cannot monotonically improve. Set False to run
    the no-warm-start ablation (the degenerate baseline)."""
    import sys
    _SRC = Path(__file__).resolve().parent.parent
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from sandbox import tier1_function_checks            # noqa: E402
    from evolve.inner_mapelites import InnerMapElites, _SRC as _IM_SRC  # noqa: E402

    # surrogate positives: forensic features of genuine faces (no FakeVLM call needed)
    real_features: List[np.ndarray] = []
    for p in real_face_paths:
        try:
            real_features.append(tier1_to_features(tier1_function_checks(p)))
        except Exception as e:
            _log.warning("tier1 on real face %s failed: %s", p, str(e)[:80])

    inner_kwargs.pop("out_dir", None)   # driver controls layout: <base_out>/roundN
    base_out = Path(base_out)
    carry: Dict[str, List[Dict[str, str]]] = {"descriptors": []}   # warm-start state

    def run_round(r: int, eff: EffectiveDefender) -> RoundResult:
        recorder: List[CoSample] = []

        def score_fn(verdict, descriptor):
            feat = tier1_to_features(verdict.tier1)
            # SEARCH signal: prefer the MC-augmentation graded real-prob (de-saturates a
            # near-binary detector into a gradient the inner QD loop can climb); fall back
            # to the single-shot real-prob when graded MC is off.
            t2 = verdict.tier2 or {}
            fv_real = t2.get("graded_real_prob")
            if fv_real is None:
                fv_real = InnerMapElites._fitness_from_verdict(verdict)
            # EVAL signal stays binary single-shot: a true bypass = the real detector was
            # fooled on the unaugmented image (the citable headline metric).
            fv_says_real = bool(verdict.sandbox_pass)
            fitness = eff.fitness(float(fv_real), feat)
            eb = eff.bypass(fv_says_real, feat)
            recorder.append(CoSample(
                cell=tuple(descriptor[a] for a in engine.ax.grid_names),
                features=feat, fakevlm_real=fv_real, eff_bypass=eb))
            return fitness, eb

        engine = InnerMapElites(out_dir=base_out / f"round{r}", score_fn=score_fn,
                                **inner_kwargs)
        engine.run(budget=inner_budget, n_seed=n_seed,
                   seed_descriptors=carry["descriptors"] if warm_start else None)

        if warm_start:
            # carry the best genotypes forward — prioritize ones that bypassed THIS
            # round's defender, then by fitness, capped to bound re-eval cost.
            elites = sorted(engine.archive.all_elites(),
                            key=lambda e: (e.bypass, e.fitness), reverse=True)
            carry["descriptors"] = [dict(e.descriptor) for e in elites[:warm_cap]]

        return RoundResult(samples=recorder, n_cells=engine.archive.n_cells(),
                           archive_path=str(base_out / f"round{r}" / "archive.json"))

    return run_round, real_features


# ────────────────────────────── CLI ──────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Co-evolution arms-race driver")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--inner-budget", type=int, default=30)
    ap.add_argument("--n-seed", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="fitness blend weight on FakeVLM vs surrogate")
    ap.add_argument("--real-thresh", type=float, default=0.5)
    ap.add_argument("--degeneracy-patience", type=int, default=2)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent.parent
                                         / "runs" / "coevo_demo"))
    ap.add_argument("--mock", action="store_true",
                    help="synthetic detector — validate the loop with no GPU")
    ap.add_argument("--seed-list", default=None,
                    help="real mode: clean face list (positives + inner seeds)")
    ap.add_argument("--seed-faces-dir",
                    default=str(Path(__file__).resolve().parent.parent.parent
                                / "data" / "pool_scut_asian"))
    ap.add_argument("--n-real-positives", type=int, default=60)
    ap.add_argument("--warm-cap", type=int, default=16,
                    help="real mode: # top genotypes carried into the next round's seed")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="real mode: disable attacker warm-start (degenerate ablation)")
    ap.add_argument("--tier2", choices=["fakevlm_local", "viviai"], default="fakevlm_local",
                    help="inner detector/judge: fakevlm_local (:8001, frozen FakeVLM) or "
                         "viviai (online gemini API; needs VIVIAI_KEY). The online judge is "
                         "weaker → more bypass signal for a quick mechanism test, but its "
                         "bypass numbers are NOT the FakeVLM result (test-only, do not cite).")
    ap.add_argument("--graded-mc-n", type=int, default=0,
                    help="real mode: # MC-augmentation views the detector re-judges per "
                         "candidate to produce a GRADED [0,1] search fitness (de-saturates "
                         "a near-binary judge). 0=off (single-shot). Multiplies tier2 "
                         "queries per candidate by N; cheap on fakevlm_local, costs N× API "
                         "calls on viviai. The binary bypass/EVAL metric is unaffected.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    out = Path(a.out)

    if a.mock:
        run_round, real_features = make_mock(rounds_seed=a.seed,
                                             n_per_round=max(a.inner_budget, 40))
    else:
        # gather real-face positives from the clean seed list
        seed_list = a.seed_list
        if seed_list is None:
            cand = Path(a.seed_faces_dir + "_clean.txt")
            seed_list = str(cand) if cand.exists() else None
        paths: List[str] = []
        if seed_list and Path(seed_list).exists():
            paths = [ln.strip() for ln in Path(seed_list).read_text().splitlines()
                     if ln.strip() and Path(ln.strip()).exists()]
        if not paths:
            raise SystemExit(f"no clean seed faces (need {a.seed_faces_dir}_clean.txt "
                             f"or --seed-list)")
        rng = random.Random(a.seed)
        pos_paths = rng.sample(paths, k=min(a.n_real_positives, len(paths)))
        inner_kwargs = dict(seed_faces_dir=Path(a.seed_faces_dir),
                            seed_list=Path(seed_list), seed=a.seed,
                            tier2_backend=a.tier2, graded_mc_n=a.graded_mc_n)
        run_round, real_features = make_real(
            inner_kwargs, a.inner_budget, a.n_seed, pos_paths, a.alpha, a.real_thresh,
            base_out=out, warm_cap=a.warm_cap, warm_start=not a.no_warm_start)

    from evolve.artifacts import write_manifest
    write_manifest(out, layer="coevolution", seed=a.seed,
                   detector_signature="mock" if a.mock else f"{a.tier2}+surrogate_logreg",
                   extra={"rounds": a.rounds, "inner_budget": a.inner_budget,
                          "n_seed": a.n_seed, "alpha": a.alpha,
                          "real_thresh": a.real_thresh, "mock": a.mock,
                          "tier2": a.tier2,
                          "n_real_positives": len(real_features),
                          "warm_start": (not a.no_warm_start) and not a.mock,
                          "warm_cap": a.warm_cap,
                          "degeneracy_patience": a.degeneracy_patience})
    driver = CoEvolutionDriver(run_round, real_features, out_dir=out,
                               alpha=a.alpha, real_thresh=a.real_thresh,
                               degeneracy_patience=a.degeneracy_patience)
    logs = driver.run(a.rounds)
    print("\n=== co-evolution summary ===")
    for l in logs:
        print(f"  round {l.round}: ASR={l.asr:.3f} bypass={l.n_bypass} "
              f"new_cells={l.new_bypass_cells} rgain={l.robustness_gain:.3f} "
              f"surr_acc={l.surrogate_train_acc:.3f}"
              f"{'  [DEGENERATE]' if l.degenerate else ''}")
    print(f"  -> {out/'coevolution.json'}")


if __name__ == "__main__":
    main()
