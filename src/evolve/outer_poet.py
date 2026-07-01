"""OUTER scenario-evolution layer (POET + PATA-EC novelty + OMNI-EPIC generation).

This is the open-ended outer loop that sits ON TOP of the inner MAP-Elites. It
evolves a population of *scenarios* (evolve/scenario.py): each scenario constrains
the inner descriptor space to a KYC sub-region and is evaluated by spawning an inner
ForgeryArchive against the frozen FakeVLM detector. The loop keeps scenarios that are
challenging-but-solvable (POET minimal-criterion band) AND behaviourally novel
(PATA-EC k-NN on the per-family bypass signature), and lets strong elites TRANSFER
between compatible scenarios.

We write our own modules (per project guidance) but follow the reference algorithms:
  - POET `pass_mc(score)`     -> score within [mc_lower, mc_upper]
  - POET novelty             -> mean distance to k nearest behaviour vectors in
                                (active ∪ archived) population (PATA-EC characterization)
  - POET reproduce/transfer  -> mutate a parent scenario; copy compatible elites
  - OMNI-EPIC generation     -> LLMScenarioGenerator invents novel scenarios (optional)

The inner engine is injected via `inner_factory` so the outer logic is unit-testable
without paying real generation cost; `--mock` uses a synthetic detector.

CLI:
    python -m evolve.outer_poet --epochs 4 --inner-budget 12 --out runs/outer_demo
    python -m evolve.outer_poet --mock --epochs 6           # fast logic check, no GPU
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import sys
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evolve.inner_mapelites import (  # noqa: E402
    InnerMapElites, AxisSpace, DEFAULT_AXES,
)
from evolve.scenario import (  # noqa: E402
    Scenario, TemplateScenarioGenerator, LLMScenarioGenerator,
)
from evolve.artifacts import SCHEMA_VERSION, write_manifest  # noqa: E402
from evolve.inner_engine import (  # noqa: E402
    InnerResult, MapElitesInner, make_inner_engine, INNER_ENGINE_NAMES,
)

_log = logging.getLogger("outer.poet")


# ────────────────────────── behaviour / novelty ──────────────────────────

def _euclid(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def novelty_knn(behavior: List[float], others: List[List[float]], k: int = 5) -> float:
    """Mean distance to the k nearest behaviour vectors (PATA-EC novelty).
    Empty neighbourhood -> maximally novel (1.0)."""
    if not others:
        return 1.0
    dists = sorted(_euclid(behavior, o) for o in others)
    kk = dists[:k]
    return sum(kk) / len(kk)


# ────────────────────────── inner result adapter ──────────────────────────
# InnerResult + the pluggable engines (mapelites / orchestrator_v1|v2 / method4) live
# in evolve/inner_engine.py so M1–M5 can each be the INNER stage of this outer loop.

def _real_inner_factory(families: List[str], seed_list: Optional[Path],
                        base_out: Path) -> Callable[[Scenario, int, int], InnerResult]:
    """Default factory: a real InnerMapElites constrained to the scenario sub-region."""
    return MapElitesInner(families, seed_list, base_out)


def _mock_inner_factory(families: List[str]) -> Callable[[Scenario, int, int], InnerResult]:
    """Synthetic inner: a toy detector where bypass-ease depends on the scenario's
    constraints (e.g. 'replay_video' + 'dark' is easier). Lets us validate the POET
    MC/novelty/transfer machinery in seconds, no GPU."""
    ease = {"replay_video": 0.5, "deepfake_injection": 0.6, "mask_3d": 0.4,
            "print_photo": 0.3, "mask_2d": 0.25, "paper_cut": 0.2, "live": 0.05,
            "dark": 0.15, "back_light": 0.1, "strong": -0.05, "normal": 0.0,
            "reenact": 0.3, "swap": 0.1, "id_diff": 0.15}

    def factory(sc: Scenario, budget: int, seed: int) -> InnerResult:
        rng = random.Random(seed ^ hash(sc.id) & 0xffffffff)
        flat = [v for vals in sc.axis_constraints.values() for v in vals]
        base_p = max(0.0, min(0.95, 0.1 + sum(ease.get(v, 0.0) for v in flat)))
        per_fam = {}
        for f in families:
            p = max(0.0, min(0.95, base_p + ease.get(f, 0.0) + rng.uniform(-0.1, 0.1)))
            per_fam[f] = p
        behavior = [per_fam[f] for f in families]
        n_elites = budget
        n_bypass = sum(1 for _ in range(budget) if rng.random() < base_p)
        score = n_bypass / n_elites if n_elites else 0.0
        bypass_elites = [{"descriptor": {"forgery_family": rng.choice(families)},
                          "fitness": round(rng.uniform(0.5, 0.95), 3),
                          "image_path": f"mock/{sc.id}_{i}.png"}
                         for i in range(n_bypass)]
        return InnerResult(score, behavior, n_bypass, len(set(behavior)), n_elites, bypass_elites)
    return factory


# ────────────────────────── outer loop ──────────────────────────

class OuterPOET:
    def __init__(
        self,
        inner_factory: Callable[[Scenario, int, int], InnerResult],
        scenario_gen,
        families: List[str],
        out_dir: Path,
        inner_budget: int = 12,
        mc_lower: float = 0.05,
        mc_upper: float = 0.85,
        capacity: int = 12,
        novelty_k: int = 5,
        max_children: int = 6,
        max_admitted: int = 2,
        seed: int = 0,
    ):
        self.inner = inner_factory
        self.gen = scenario_gen
        self.families = families
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.inner_budget = inner_budget
        self.mc_lower, self.mc_upper = mc_lower, mc_upper
        self.capacity = capacity
        self.novelty_k = novelty_k
        self.max_children = max_children
        self.max_admitted = max_admitted
        self.rng = random.Random(seed)
        self._seed = seed

        self.active: List[Scenario] = []
        self.archived: List[Scenario] = []     # novelty memory + history
        self.transfers: List[dict] = []
        self._transfer_seen: set = set()       # dedup keys so re-checks don't double-count
        self._epoch = 0
        self._log_f = (self.out_dir / "outer_log.jsonl").open("w")

    def pass_mc(self, score: float) -> bool:
        return self.mc_lower <= score <= self.mc_upper

    def _evaluate(self, sc: Scenario, seed: int) -> Scenario:
        res = self.inner(sc, self.inner_budget, seed)
        sc.score = res.score
        sc.behavior = res.behavior
        sc.n_bypass = res.n_bypass
        sc.n_cells = res.n_cells
        sc._inner = res    # transient, not serialized
        return sc

    def _all_behaviors(self, exclude: Optional[Scenario] = None) -> List[List[float]]:
        pool = self.active + self.archived
        return [s.behavior for s in pool if s.behavior is not None and s is not exclude]

    # ── elite transfer between compatible scenarios ──
    def _compatible(self, descriptor: Dict[str, str], sc: Scenario) -> bool:
        for axis, allowed in sc.axis_constraints.items():
            v = descriptor.get(axis)
            if v is not None and v not in allowed:
                return False
        return True

    def _transfer(self) -> int:
        """Count bypass elites that ALSO satisfy another active scenario's constraints
        — a transfer means a method discovered in scenario A is portable to scenario B
        (POET cross-niche transfer). We record it as evidence of generality."""
        n = 0
        for src in self.active:
            res = getattr(src, "_inner", None)
            if res is None:
                continue
            for ei, el in enumerate(res.bypass_elites):
                for dst in self.active:
                    if dst is src:
                        continue
                    if self._compatible(el["descriptor"], dst):
                        key = (src.id, dst.id, ei)
                        if key in self._transfer_seen:
                            continue
                        self._transfer_seen.add(key)
                        self.transfers.append({"from": src.id, "to": dst.id,
                                               "fitness": el["fitness"]})
                        n += 1
        return n

    def _admit_children(self, epoch: int) -> List[Scenario]:
        parents = [s for s in self.active if s.score is not None and self.pass_mc(s.score)]
        if not parents:
            parents = list(self.active)
        children: List[Scenario] = []
        for _ in range(self.max_children):
            parent = self.rng.choice(parents) if parents else None
            child = self.gen.sample(parent=parent, gen=epoch)
            child = self._evaluate(child, seed=self.rng.randint(0, 2**31 - 1))
            if not self.pass_mc(child.score):
                self.archived.append(child)      # remember for novelty, don't activate
                continue
            child.novelty = novelty_knn(child.behavior, self._all_behaviors(), self.novelty_k)
            children.append(child)
        children.sort(key=lambda c: c.novelty, reverse=True)
        admitted = children[:self.max_admitted]
        self.archived.extend(children[self.max_admitted:])
        return admitted

    def _prune(self):
        if len(self.active) <= self.capacity:
            return
        # archive the oldest (lowest gen) beyond capacity — POET remove_oldest
        self.active.sort(key=lambda s: (s.gen, -(s.novelty or 0.0)))
        overflow = self.active[:len(self.active) - self.capacity]
        self.archived.extend(overflow)
        self.active = self.active[len(overflow):]

    def _log_epoch(self, epoch: int, n_transfer: int):
        rec = {
            "epoch": epoch, "active": len(self.active), "archived": len(self.archived),
            "transfers_cum": len(self.transfers), "transfers_epoch": n_transfer,
            "scenarios": [
                {"id": s.id, "name": s.name, "score": round(s.score, 3) if s.score is not None else None,
                 "novelty": round(s.novelty, 3), "n_bypass": s.n_bypass,
                 "constraints": s.axis_constraints}
                for s in self.active
            ],
        }
        self._log_f.write(json.dumps(rec, ensure_ascii=False) + "\n"); self._log_f.flush()
        _log.info("epoch %d | active=%d archived=%d transfers=%d (best score=%.2f)",
                  epoch, len(self.active), len(self.archived), len(self.transfers),
                  max((s.score or 0.0) for s in self.active) if self.active else 0.0)

    def run(self, epochs: int, n_seed_scenarios: int = 4) -> Dict[str, Any]:
        write_manifest(self.out_dir, layer="outer", seed=self._seed,
                       extra={"epochs": epochs, "n_seed_scenarios": n_seed_scenarios,
                              "inner_budget": self.inner_budget,
                              "mc_band": [self.mc_lower, self.mc_upper],
                              "capacity": self.capacity, "families": list(self.families)})
        # seed population
        for _ in range(n_seed_scenarios):
            sc = self.gen.sample(parent=None, gen=0)
            sc = self._evaluate(sc, seed=self.rng.randint(0, 2**31 - 1))
            sc.novelty = novelty_knn(sc.behavior, self._all_behaviors(exclude=sc), self.novelty_k)
            self.active.append(sc)
        self._log_epoch(0, self._transfer())

        for epoch in range(1, epochs + 1):
            self._epoch = epoch
            admitted = self._admit_children(epoch)
            self.active.extend(admitted)
            self._prune()
            n_transfer = self._transfer()
            self._log_epoch(epoch, n_transfer)

        self._log_f.close()
        return self._save()

    def _save(self) -> Dict[str, Any]:
        summary = {
            "schema_version": SCHEMA_VERSION,
            # the PATA-EC behavior vector on each scenario is per-family bypass rate,
            # in THIS axis order — record it so the vector is interpretable later.
            "behavior_axes": list(self.families),
            "n_active": len(self.active),
            "n_archived": len(self.archived),
            "n_transfers": len(self.transfers),
            "mc_band": [self.mc_lower, self.mc_upper],
            "active": [s.to_json() for s in self.active],
            "archived": [s.to_json() for s in self.archived],
            "transfers": self.transfers,
        }
        # drop transient _inner before serialization handled by asdict (not a field)
        (self.out_dir / "scenarios.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))
        _log.info("OUTER DONE active=%d archived=%d transfers=%d -> %s",
                  len(self.active), len(self.archived), len(self.transfers),
                  self.out_dir / "scenarios.json")
        return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--n-seed-scenarios", type=int, default=4)
    ap.add_argument("--inner-budget", type=int, default=12)
    ap.add_argument("--capacity", type=int, default=12)
    ap.add_argument("--mc-lower", type=float, default=0.05)
    ap.add_argument("--mc-upper", type=float, default=0.85)
    ap.add_argument("--axes", default=str(DEFAULT_AXES))
    ap.add_argument("--seed-list", default=None)
    ap.add_argument("--out", default=str(_SRC.parent / "runs" / "outer_demo"))
    ap.add_argument("--mock", action="store_true", help="synthetic inner detector (fast, no GPU)")
    ap.add_argument("--inner-engine", choices=list(INNER_ENGINE_NAMES), default="mapelites",
                    help="which method drives the INNER stage: mapelites (M6/M2 QD), "
                         "orchestrator_v1 (M1), orchestrator_v2 (M2), method4 (M4). "
                         "Selecting any of these makes that method two-layer.")
    ap.add_argument("--use-llm", action="store_true", help="OMNI-EPIC LLM scenario gen (needs VIVIAI_KEY)")
    ap.add_argument("--inner-rounds", type=int, default=1,
                    help="long-horizon: rounds of self-evolution INSIDE each scenario "
                         "(legacy engines; >1 gives a round-over-round bypass curve)")
    ap.add_argument("--shared-bank", action="store_true",
                    help="long-horizon: share ONE skill/markov/reasoning bank across ALL "
                         "scenarios (orchestrator_v1/v2 only) instead of cold-starting each")
    ap.add_argument("--evade-weight", type=float, default=0.3,
                    help="continuous bypass-proximity reward weight (0=off, binary bypass only; "
                         ">0 gives the family search a gradient toward detector-evasion)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ax = AxisSpace.load(Path(a.axes))
    families = list(ax.grid_axes["forgery_family"])
    axis_vocab = ax.all_axes
    rng = random.Random(a.seed)

    if a.use_llm:
        try:
            from viviai_client import ViviClient
            client = ViviClient()
            scenario_gen = LLMScenarioGenerator(axis_vocab, client=client, rng=rng)
        except Exception as e:
            _log.warning("LLM gen unavailable (%s); template fallback", str(e)[:120])
            scenario_gen = TemplateScenarioGenerator(axis_vocab, rng)
    else:
        scenario_gen = TemplateScenarioGenerator(axis_vocab, rng)

    out = Path(a.out)
    if a.mock:
        factory = _mock_inner_factory(families)
    else:
        seed_list = Path(a.seed_list) if a.seed_list else None
        factory = make_inner_engine(a.inner_engine, families, seed_list, out / "inner",
                                    rounds=a.inner_rounds, shared_bank=a.shared_bank,
                                    evade_weight=a.evade_weight)

    engine = OuterPOET(
        inner_factory=factory, scenario_gen=scenario_gen, families=families,
        out_dir=out, inner_budget=a.inner_budget, mc_lower=a.mc_lower,
        mc_upper=a.mc_upper, capacity=a.capacity, seed=a.seed,
    )
    engine.run(epochs=a.epochs, n_seed_scenarios=a.n_seed_scenarios)


if __name__ == "__main__":
    main()
