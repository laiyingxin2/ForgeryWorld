"""Pluggable INNER engines — the bridge that makes M1–M5 genuinely two-layer.

The OUTER POET layer (outer_poet.py) drives ONE inner engine per scenario through a
callable `engine(scenario, budget, seed) -> InnerResult`. Until now the only inner was
`InnerMapElites` (the M6 / M2-思想 QD inner). This module additionally wraps the LEGACY
single-layer orchestrators as inner engines, so selecting `--inner-engine` turns each
historical method into the INNER stage of the SAME outer scenario loop:

    mapelites        InnerMapElites               (M6 / M2 QD inner)        [in-process]
    orchestrator_v1  legacy/orchestrator.py v1    (M1 baseline)            [subprocess]
    orchestrator_v2  legacy/orchestrator.py v2    (M2 self-evolution)      [subprocess]
    method4          legacy/method4_orchestrator  (M4 face-cluster Pareto) [subprocess]

That is what "把 M1–M5 也升级为两层" means in code: 外层场景 (POET MC-band + k-NN novelty +
transfer) × 内层手法 (any of the engines above). Before this file the two-layer wiring
existed for MAP-Elites ONLY; the legacy methods were parallel single-layer tracks.

Contract every engine satisfies (decoupled from the inner implementation):
  - `behavior` = per-family bypass-rate vector over `families` — the PATA-EC scenario
    characterization the outer novelty/MC machinery consumes (two scenarios that break
    the same families the same way are not novel).
  - `score`    = the scenario's overall bypass rate — the POET minimal-criterion value.

The legacy adapters translate a Scenario into the orchestrator CLI (forgery_family
constraint → which seed faces to attack via --src-pool; budget → --briefs/--rollouts)
and parse the report JSON back into an InnerResult. They drive the frozen FakeVLM at
:8001 (`--tier2-backend fakevlm_local`). They are structurally complete but need a
real-infra smoke (legacy pipeline = gemini brain + GPU ops); `MapElitesInner` is the
validated in-process path.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evolve.scenario import Scenario  # noqa: E402

_log = logging.getLogger("inner_engine")

FAKEVLM_ENDPOINT = "http://localhost:8001/v1"
INNER_ENGINE_NAMES = ("mapelites", "orchestrator_v1", "orchestrator_v2", "method4")


class InnerResult:
    """What the outer loop needs from one inner run, decoupled from the engine."""
    def __init__(self, score: float, behavior: List[float], n_bypass: int,
                 n_cells: int, n_elites: int, bypass_elites: List[dict]):
        self.score = score
        self.behavior = behavior
        self.n_bypass = n_bypass
        self.n_cells = n_cells
        self.n_elites = n_elites
        self.bypass_elites = bypass_elites    # [{descriptor, fitness, image_path}]


def _vector_over(rate_map: Dict[str, float], families: List[str]) -> List[float]:
    """Project a {family -> bypass_rate} dict onto the fixed `families` order; absent
    families are 0.0 (the scenario never exercised them = no observed bypass)."""
    return [float(rate_map.get(f, 0.0)) for f in families]


# ────────────────────────── in-process MAP-Elites inner (M6 / M2) ──────────────────────────

class MapElitesInner:
    """Default inner: a real InnerMapElites constrained to the scenario sub-region.

    This is the validated path (identical to outer_poet's former _real_inner_factory).
    """

    def __init__(self, families: List[str], seed_list: Optional[Path], base_out: Path):
        self.families = families
        self.seed_list = seed_list
        self.base_out = Path(base_out)

    def __call__(self, sc: Scenario, budget: int, seed: int) -> InnerResult:
        from evolve.inner_mapelites import InnerMapElites
        engine = InnerMapElites(
            seed_list=self.seed_list,
            out_dir=self.base_out / sc.id,
            axis_overrides=sc.axis_constraints,
            seed=seed,
        )
        engine.run(budget=budget, n_seed=max(2, budget // 4))
        arch = engine.archive
        n_elites = arch.n_elites()
        n_bypass = arch.n_bypass()
        score = n_bypass / n_elites if n_elites else 0.0
        behavior = arch.family_bypass_rates(self.families)
        bypass_elites = [{"descriptor": e.descriptor, "fitness": e.fitness,
                          "image_path": e.image_path}
                         for e in arch.all_elites() if e.bypass]
        return InnerResult(score, behavior, n_bypass, arch.n_cells(), n_elites, bypass_elites)


# ────────────────────────── legacy subprocess inners (M1 / M2 / M4) ──────────────────────────

class _LegacyInner:
    """Common scaffolding for the subprocess-driven legacy orchestrators.

    Subclasses define the module, the per-round report glob, and how a report dict maps
    to (family_bypass_rates, n, n_bypass). A Scenario narrows the run through --src-pool
    (a subset of seed faces) + --briefs/--rollouts derived from the outer budget.
    """
    module: str = ""
    extra_flags: List[str] = []
    # subclasses whose orchestrator supports the --bank-dir long-horizon flag set this.
    supports_bank_dir: bool = False

    def __init__(self, families: List[str], seed_list: Optional[Path], base_out: Path,
                 python_exe: Optional[str] = None, fakevlm_endpoint: str = FAKEVLM_ENDPOINT,
                 max_src: int = 16, timeout: int = 60 * 60,
                 rounds: int = 1, shared_bank: bool = False,
                 evade_weight: float = 0.3):
        self.families = families
        self.seed_list = seed_list
        self.base_out = Path(base_out)
        self.python_exe = python_exe or sys.executable
        self.fakevlm_endpoint = fakevlm_endpoint
        self.max_src = max_src
        self.timeout = timeout
        # ★ Long-horizon: rounds>1 = within-scenario round-over-round self-evolution;
        # shared_bank = one bank dir shared across ALL outer scenarios (Voyager fixed-ckpt
        # = implicit POET transfer of the accumulated skill/markov/reasoning state).
        self.rounds = max(1, int(rounds))
        self.shared_bank = bool(shared_bank)
        self.evade_weight = float(evade_weight)
        self.bank_dir = self.base_out / "_bank"

    def _src_pool(self, sc: Scenario, seed: int) -> List[str]:
        """A scenario-scoped subset of the clean seed faces to attack."""
        if not self.seed_list or not Path(self.seed_list).exists():
            return []
        faces = [ln.strip() for ln in Path(self.seed_list).read_text().splitlines()
                 if ln.strip() and Path(ln.strip()).exists()]
        if not faces:
            return []
        import random as _r
        rng = _r.Random(seed ^ (hash(sc.id) & 0xffffffff))
        rng.shuffle(faces)
        return faces[:self.max_src]

    def _budget_to_briefs(self, budget: int) -> tuple:
        # rollouts=1 keeps each brief a single attack attempt; briefs ~ budget.
        return max(1, budget // 1), 1

    def _cmd(self, sc: Scenario, budget: int, seed: int, out: Path,
             src_pool: List[str]) -> List[str]:
        briefs, rollouts = self._budget_to_briefs(budget)
        cmd = [self.python_exe, "-m", self.module,
               "--rounds", str(self.rounds), "--briefs", str(briefs),
               "--rollouts", str(rollouts),
               "--tier2-backend", "fakevlm_local",
               "--fakevlm-endpoint", self.fakevlm_endpoint,
               "--out", str(out)]
        # Long-horizon: point every scenario's bank at ONE shared dir so the skill/
        # markov/reasoning state accumulates across scenarios instead of cold-starting.
        if self.shared_bank and self.supports_bank_dir:
            cmd += ["--bank-dir", str(self.bank_dir)]
        if self.supports_bank_dir:
            cmd += ["--evade-weight", str(self.evade_weight)]
        cmd += self.extra_flags
        if src_pool:
            cmd += ["--src-pool", *src_pool]
        return cmd

    def _parse(self, out: Path) -> tuple:
        """Return (family_bypass_rates: dict, n: int, n_bypass: int). Subclass-specific."""
        raise NotImplementedError

    def __call__(self, sc: Scenario, budget: int, seed: int) -> InnerResult:
        out = self.base_out / sc.id
        out.mkdir(parents=True, exist_ok=True)
        src_pool = self._src_pool(sc, seed)
        cmd = self._cmd(sc, budget, seed, out, src_pool)
        _log.info("[%s] scenario=%s -> %s", self.module, sc.id, " ".join(cmd[:8]))
        # legacy/orchestrator.py uses bare imports (`from trajectory_schema import …`)
        # that only resolve when src/legacy is on PYTHONPATH. Inject both src and
        # src/legacy so the `python -m legacy.<mod>` subprocess can import its siblings.
        env = dict(os.environ)
        extra_pp = os.pathsep.join([str(_SRC), str(_SRC / "legacy")])
        env["PYTHONPATH"] = (extra_pp + os.pathsep + env["PYTHONPATH"]
                             if env.get("PYTHONPATH") else extra_pp)
        try:
            subprocess.run(cmd, cwd=str(_SRC), check=True, timeout=self.timeout,
                           capture_output=True, text=True, env=env)
        except subprocess.CalledProcessError as e:
            _log.warning("[%s] failed on %s: %s", self.module, sc.id,
                         (e.stderr or "")[-300:])
            return InnerResult(0.0, _vector_over({}, self.families), 0, 0, 0, [])
        except subprocess.TimeoutExpired:
            _log.warning("[%s] TIMEOUT on %s", self.module, sc.id)
            return InnerResult(0.0, _vector_over({}, self.families), 0, 0, 0, [])

        rates, n, n_bypass = self._parse(out)
        behavior = _vector_over(rates, self.families)
        score = (n_bypass / n) if n else (sum(behavior) / len(behavior) if behavior else 0.0)
        n_cells = sum(1 for r in behavior if r > 0.0)
        return InnerResult(score, behavior, n_bypass, n_cells, n or n_bypass, [])


class OrchestratorInner(_LegacyInner):
    """M1 (v1) / M2 (v2): legacy/orchestrator.py. Report: reports/r0_<mode>.json with
    diagnosis.{global_bypass_rate, family_bypass_rates}."""
    module = "legacy.orchestrator"
    supports_bank_dir = True

    def __init__(self, mode: str, **kw):
        assert mode in ("v1", "v2")
        self.mode = mode
        super().__init__(**kw)
        self.extra_flags = ["--mode", mode]
        if mode == "v2":
            self.extra_flags += ["--multi-agent-preset", "w6_full"]

    def _parse(self, out: Path) -> tuple:
        reps = sorted((out / "reports").glob(f"r*_{self.mode}.json"))
        if not reps:
            return {}, 0, 0
        diag = json.loads(reps[-1].read_text()).get("diagnosis", {})
        rates = diag.get("family_bypass_rates", {}) or {}
        gbr = float(diag.get("global_bypass_rate", 0.0))
        # the report stores rates, not raw counts; reconstruct a nominal n from the
        # outer budget is not available here, so carry the global rate as the score and
        # leave counts as a rate-scaled proxy (outer only needs score+behavior).
        n = 100
        n_bypass = int(round(gbr * n))
        return rates, n, n_bypass


class Method4Inner(_LegacyInner):
    """M4: legacy/method4_orchestrator.py. Report: method4_summary.json with
    rounds[].results[] {family, bypass}; we aggregate per-family bypass rates."""
    module = "legacy.method4_orchestrator"
    extra_flags = ["--preset", "w6_full"]

    def _parse(self, out: Path) -> tuple:
        fp = out / "method4_summary.json"
        if not fp.exists():
            return {}, 0, 0
        summary = json.loads(fp.read_text())
        cnt: Dict[str, List[int]] = {}    # family -> [bypass, total]
        n = n_bypass = 0
        for rnd in summary.get("rounds", []):
            for res in rnd.get("results", []):
                fam = res.get("family")
                if fam is None:
                    continue
                n += 1
                b = 1 if res.get("bypass") else 0
                n_bypass += b
                cnt.setdefault(fam, [0, 0])
                cnt[fam][0] += b
                cnt[fam][1] += 1
        rates = {f: (bp / tot if tot else 0.0) for f, (bp, tot) in cnt.items()}
        return rates, n, n_bypass


# ────────────────────────── factory ──────────────────────────

def make_inner_engine(name: str, families: List[str], seed_list: Optional[Path],
                      base_out: Path, python_exe: Optional[str] = None,
                      fakevlm_endpoint: str = FAKEVLM_ENDPOINT,
                      rounds: int = 1, shared_bank: bool = False,
                      evade_weight: float = 0.3):
    """Return a callable inner engine `engine(scenario, budget, seed) -> InnerResult`.

    name ∈ INNER_ENGINE_NAMES. `mapelites` runs in-process; the legacy engines spawn
    their orchestrator as a subprocess against the frozen FakeVLM at `fakevlm_endpoint`.

    Long-horizon knobs (legacy engines only): `rounds`>1 runs round-over-round
    self-evolution inside each scenario; `shared_bank`=True shares ONE skill/markov/
    reasoning bank across all outer scenarios (orchestrator_v1/v2 only — method4 has no
    --bank-dir flag and silently ignores it via supports_bank_dir=False).
    """
    if name == "mapelites":
        return MapElitesInner(families, seed_list, base_out)
    common = dict(families=families, seed_list=seed_list, base_out=base_out,
                  python_exe=python_exe, fakevlm_endpoint=fakevlm_endpoint,
                  rounds=rounds, shared_bank=shared_bank, evade_weight=evade_weight)
    if name == "orchestrator_v1":
        return OrchestratorInner(mode="v1", **common)
    if name == "orchestrator_v2":
        return OrchestratorInner(mode="v2", **common)
    if name == "method4":
        return Method4Inner(**common)
    raise ValueError(f"unknown inner engine {name!r}; choose from {INNER_ENGINE_NAMES}")
