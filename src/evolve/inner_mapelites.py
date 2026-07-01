"""Inner-layer MAP-Elites for the two-layer open-ended face-forgery red-team.

This is the INNER loop: it evolves a concrete forgery *method* via a Quality-
Diversity (MAP-Elites) archive over fine-grained attack descriptors, against the
frozen FakeVLM detector (:8001) as the fitness signal. The OUTER scenario-evolution
loop (POET / OMNI-EPIC style) is a later layer that will spawn one of these archives
per scenario.

Design (参考 RainbowPlus archive.py, but our own content):
  - Archive = dict keyed by a GRID descriptor tuple -> list of elites (multi-elite
    per cell, top-K by fitness). Grid axes are the few axes we actively diversify
    (forgery_family x pai x lighting); the other ~9 axes ride along as metadata tags.
  - One iteration: pick a parent elite -> mutate its descriptor (resample a few axes)
    -> instantiate a concrete generation (operator + prompt + optional post-process
    chain) -> score with the frozen detector -> insert into its cell (top-K eviction).
  - Realism is NOT a gate (two-layer redesign): every generated candidate is inserted
    and ranked only by detector fitness, so OOD / cross-species / faceless bypasses
    are kept. The sandbox emits a `face_type` label we record, never a reject.

Axes + value vocab + operator/prompt mappings live in configs/evolve_axes.yaml.

CLI:
    python -m evolve.inner_mapelites --budget 40 --out runs/inner_demo \
        --seed-faces-dir ../data/pool_scut_asian
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

# src/ is the import root for this project (operators.*, sandbox, fakevlm_judge_real)
_SRC = Path(__file__).resolve().parent.parent
import sys
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from operators import OPERATOR_REGISTRY, resolve_op  # noqa: E402
from sandbox import SandboxVerifier  # noqa: E402
from viviai_client import ViviClient  # noqa: E402
from evolve.artifacts import SCHEMA_VERSION, write_manifest  # noqa: E402

_log = logging.getLogger("inner_mapelites")

DEFAULT_AXES = _SRC.parent / "configs" / "evolve_axes.yaml"
FAKEVLM_ENDPOINT = "http://localhost:8001/v1"
FAKEVLM_CKPT = "/data/disk4/lyx_ICML/self_evolution_forgery/scripts/fakevlm_correct_ckpt"

# Families whose operators consume the natural-language prompt (diffusion / synth /
# edit) realize the `lighting` axis through build_prompt() text fragments. Families
# NOT in this set (swap, reenact) have no text path, so they realize lighting via a
# photometric `relight` post-op instead. Mapping lighting-value -> relight mode:
TEXT_FAMILIES = {"entire_synthesis", "attribute_edit", "id_diff"}
LIGHTING_RELIGHT_MODE = {
    "normal": None,            # identity — no relight needed
    "strong": "strong",
    "back_light": "back_light",
    "dark": "dark",
}


# ────────────────────────────── descriptor space ──────────────────────────────

@dataclass
class AxisSpace:
    """Parsed evolve_axes.yaml: which axes grid the archive, the value vocab, and
    the descriptor->generation mappings."""
    grid_axes: Dict[str, List[str]]
    tag_axes: Dict[str, List[str]]
    family_ops: Dict[str, List[str]]
    post_ops: Dict[str, str]
    prompt_fragments: Dict[str, Dict[str, str]]

    @property
    def grid_names(self) -> List[str]:
        return list(self.grid_axes.keys())

    @property
    def all_axes(self) -> Dict[str, List[str]]:
        return {**self.grid_axes, **self.tag_axes}

    @classmethod
    def load(cls, path: Path) -> "AxisSpace":
        cfg = yaml.safe_load(Path(path).read_text())
        return cls(
            grid_axes=cfg["grid_axes"],
            tag_axes=cfg["tag_axes"],
            family_ops=cfg["family_ops"],
            post_ops=cfg.get("post_ops", {}),
            prompt_fragments=cfg.get("prompt_fragments", {}),
        )


def sample_descriptor(ax: AxisSpace, rng: random.Random) -> Dict[str, str]:
    return {name: rng.choice(values) for name, values in ax.all_axes.items()}


def mutate_descriptor(parent: Dict[str, str], ax: AxisSpace, rng: random.Random,
                      n_axes: int = 2) -> Dict[str, str]:
    """Resample `n_axes` randomly chosen axes of the parent descriptor."""
    child = dict(parent)
    names = list(ax.all_axes.keys())
    for name in rng.sample(names, k=min(n_axes, len(names))):
        child[name] = rng.choice(ax.all_axes[name])
    return child


def cell_key(descriptor: Dict[str, str], ax: AxisSpace) -> Tuple[str, ...]:
    return tuple(descriptor[name] for name in ax.grid_names)


def build_prompt(descriptor: Dict[str, str], ax: AxisSpace) -> str:
    """Compose a natural-language scenario string from the descriptor so diffusion /
    synthesis operators are steered by lighting / attribute / pose / occlusion /
    environment / cross-species tags."""
    parts = ["a photorealistic close-up portrait of a person"]
    for axis_name, frag_map in ax.prompt_fragments.items():
        val = descriptor.get(axis_name)
        frag = frag_map.get(val) if val else None
        if frag:
            parts.append(frag)
    parts.append("sharp natural skin texture, shot on a phone camera")
    return ", ".join(parts)


# ────────────────────────────── archive ──────────────────────────────

@dataclass
class Elite:
    id: str
    descriptor: Dict[str, str]
    cell: Tuple[str, ...]
    fitness: float                 # detector real-probability (higher = better attack)
    bypass: bool                   # frozen detector judged it real
    face_type: str                 # {face, low_id_face, non_face} — label only
    image_path: str
    op_name: str
    post_op: Optional[str]
    prompt: str
    parent_id: Optional[str]
    gen: int

    def to_json(self) -> dict:
        d = asdict(self)
        d["cell"] = list(self.cell)
        d["fitness"] = round(float(self.fitness), 4)
        return d


class ForgeryArchive:
    """dict[cell_tuple -> list[Elite]], multi-elite per cell with top-K eviction.

    Never drops on face grounds; ranks only by detector fitness (two-layer redesign).
    """

    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self._cells: Dict[Tuple[str, ...], List[Elite]] = {}

    def add(self, elite: Elite) -> bool:
        """Insert; keep top-K per cell by fitness. Returns True if it was kept."""
        bucket = self._cells.setdefault(elite.cell, [])
        bucket.append(elite)
        bucket.sort(key=lambda e: e.fitness, reverse=True)
        del bucket[self.top_k:]
        return elite in bucket

    def all_elites(self) -> List[Elite]:
        return [e for bucket in self._cells.values() for e in bucket]

    def random_parent(self, rng: random.Random) -> Optional[Elite]:
        pool = self.all_elites()
        return rng.choice(pool) if pool else None

    # ── stats ──
    def n_cells(self) -> int:
        return len(self._cells)

    def n_elites(self) -> int:
        return sum(len(b) for b in self._cells.values())

    def n_bypass(self) -> int:
        return sum(1 for e in self.all_elites() if e.bypass)

    def face_type_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.all_elites():
            out[e.face_type] = out.get(e.face_type, 0) + 1
        return out

    def family_bypass_rates(self, families: List[str]) -> List[float]:
        """Per-family bypass rate over the archive, in the given family order.
        Used by the OUTER layer as a scenario behavior characterization (PATA-EC):
        two scenarios that break the SAME families the same way are not novel."""
        cnt = {f: [0, 0] for f in families}   # family -> [bypass, total]
        for e in self.all_elites():
            fam = e.cell[0] if e.cell else None
            if fam in cnt:
                cnt[fam][1] += 1
                cnt[fam][0] += 1 if e.bypass else 0
        return [(cnt[f][0] / cnt[f][1]) if cnt[f][1] else 0.0 for f in families]

    def save(self, path: Path, grid_axes: Optional[List[str]] = None) -> None:
        # cells are keyed by a delimiter-joined string (JSON objects can't have tuple
        # keys); `grid_axes` + `cell_delimiter` make the keys decodable without
        # re-deriving the axis order from code.
        data = {
            "schema_version": SCHEMA_VERSION,
            "grid_axes": grid_axes or [],
            "cell_delimiter": "|",
            "n_cells": self.n_cells(),
            "n_elites": self.n_elites(),
            "n_bypass": self.n_bypass(),
            "face_type_counts": self.face_type_counts(),
            "cells": {
                "|".join(k): [e.to_json() for e in bucket]
                for k, bucket in self._cells.items()
            },
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ────────────────────────────── inner loop ──────────────────────────────

class InnerMapElites:
    def __init__(
        self,
        axes_path: Path = DEFAULT_AXES,
        seed_faces_dir: Path = _SRC.parent / "data" / "pool_scut_asian",
        seed_list: Optional[Path] = None,
        out_dir: Path = _SRC.parent / "runs" / "inner_demo",
        top_k: int = 4,
        fitness_floor: float = 0.0,    # 0.0 == never drop on quality (top-K still evicts)
        tier2_backend: str = "fakevlm_local",
        seed: int = 0,
        axis_overrides: Optional[Dict[str, List[str]]] = None,
        score_fn: Optional[Callable[[Any, Dict[str, str]], Tuple[float, bool]]] = None,
        graded_mc_n: int = 0,
    ):
        # score_fn lets an OUTER co-evolution driver override how a verdict maps to
        # (fitness, bypass): e.g. fuse the frozen FakeVLM signal with a trainable
        # surrogate defender so MAP-Elites optimizes a graded score and a bypass
        # counts only if it fools the *current* (evolving) defender. Default keeps the
        # frozen-detector behavior (FakeVLM real-prob / sandbox_pass).
        self.score_fn = score_fn
        self._seed = seed
        self._axes_path = axes_path
        self.ax = AxisSpace.load(axes_path)
        if axis_overrides:
            # An OUTER scenario narrows the inner descriptor space to a sub-region
            # (e.g. only {dark, back_light} lighting, only the swap family). Intersect
            # each override with the axis's real vocab so a scenario can't introduce
            # an unknown value; ignore empties.
            for name, vals in axis_overrides.items():
                if name in self.ax.grid_axes:
                    keep = [v for v in self.ax.grid_axes[name] if v in set(vals)]
                    if keep:
                        self.ax.grid_axes[name] = keep
                elif name in self.ax.tag_axes:
                    keep = [v for v in self.ax.tag_axes[name] if v in set(vals)]
                    if keep:
                        self.ax.tag_axes[name] = keep
        self.rng = random.Random(seed)
        self.archive = ForgeryArchive(top_k=top_k)
        self.fitness_floor = fitness_floor

        # Prefer a QC'd clean list (one path per line) when present — a loose/multi-
        # /undetectable seed poisons every faces[0]-indexing operator. Fall back to a
        # raw glob of the pool dir.
        if seed_list is None:
            cand = Path(str(seed_faces_dir) + "_clean.txt")
            seed_list = cand if cand.exists() else None
        if seed_list and Path(seed_list).exists():
            self.seed_faces = [ln.strip() for ln in Path(seed_list).read_text().splitlines()
                               if ln.strip() and Path(ln.strip()).exists()]
            _log.info("using QC'd seed list %s (%d faces)", seed_list, len(self.seed_faces))
        else:
            self.seed_faces = sorted(str(p) for p in Path(seed_faces_dir).glob("*")
                                     if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        if not self.seed_faces:
            raise FileNotFoundError(f"no seed faces under {seed_faces_dir} / {seed_list}")

        self.out_dir = Path(out_dir)
        self.gen_dir = self.out_dir / "gen"
        self.gen_dir.mkdir(parents=True, exist_ok=True)

        if tier2_backend == "fakevlm_local":
            # client is unused on the fakevlm_local path (tier2 = local vLLM judge),
            # but SandboxVerifier still constructs a ViviClient if none is passed and
            # ViviClient hard-requires a key. Pass a placeholder so we don't need one.
            placeholder = ViviClient(api_key="unused-fakevlm-local")
            self.verifier = SandboxVerifier(
                client=placeholder,
                tier2_backend="fakevlm_local",
                fakevlm_endpoint=FAKEVLM_ENDPOINT,
                fakevlm_ckpt_path=FAKEVLM_CKPT,
                tier3_enabled=False,
                graded_mc_n=graded_mc_n,
            )
        else:
            self.verifier = SandboxVerifier(tier2_backend=tier2_backend, tier3_enabled=False,
                                            graded_mc_n=graded_mc_n)

        self._gen_counter = 0

    # ── generation ──

    def _pick_faces(self) -> Tuple[str, str]:
        a = self.rng.choice(self.seed_faces)
        b = self.rng.choice(self.seed_faces)
        return a, b

    def generate(self, descriptor: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Instantiate a concrete forgery from a descriptor.

        forgery_family -> base operator; the lighting/attribute/pose/etc tags steer
        the prompt; post_process / perturbation / pai tags may chain a post-op.
        Returns {image_path, op_name, post_op, prompt} or None on hard failure.
        """
        family = descriptor["forgery_family"]
        op_candidates = self.ax.family_ops.get(family, [])
        op_candidates = [k for k in (resolve_op(c) for c in op_candidates)
                         if k in OPERATOR_REGISTRY]
        if not op_candidates:
            _log.warning("no registered op for family=%s", family)
            return None
        op_key = self.rng.choice(op_candidates)
        Op = OPERATOR_REGISTRY[op_key]
        op = Op(out_dir=str(self.gen_dir))

        prompt = build_prompt(descriptor, self.ax)
        base, donor = self._pick_faces()
        params = {"prompt": prompt, "instruction": prompt,
                  "seed": self.rng.randint(0, 2**31 - 1)}
        res = op.run(src_face_path=base, tgt_face_path=donor, params=params)
        if not res.success or not res.output_path:
            _log.info("base op %s failed: %s", op_key, res.error)
            return None
        cur = res.output_path

        # Lighting axis: text-driven families already baked it into the prompt; swap/
        # reenact have no text path, so realize the `lighting` value photometrically.
        relight_used = None
        if family not in TEXT_FAMILIES:
            mode = LIGHTING_RELIGHT_MODE.get(descriptor.get("lighting"))
            Relight = OPERATOR_REGISTRY.get("relight")
            if mode and Relight is not None:
                rres = Relight(out_dir=str(self.gen_dir)).run(
                    src_face_path=cur, params={"mode": mode})
                if rres.success and rres.output_path:
                    cur = rres.output_path
                    relight_used = f"relight:{mode}"

        # optional post-process / perturbation / PAI chain
        post_used = relight_used
        for axis in ("post_process", "perturbation", "pai"):
            val = descriptor.get(axis)
            post_key = self.ax.post_ops.get(val) if val else None
            if not post_key:
                continue
            post_key = resolve_op(post_key)
            PostOp = OPERATOR_REGISTRY.get(post_key)
            if PostOp is None:
                continue
            pres = PostOp(out_dir=str(self.gen_dir)).run(src_face_path=cur, params={})
            if pres.success and pres.output_path:
                cur = pres.output_path
                post_used = f"{relight_used}+{post_key}" if relight_used else post_key
                break   # one post stage per candidate keeps the chain interpretable

        return {"image_path": cur, "op_name": op_key, "post_op": post_used,
                "prompt": prompt}

    # ── evaluation ──

    @staticmethod
    def _fitness_from_verdict(verdict) -> float:
        """Detector real-probability in [0,1] (higher = more convincingly real)."""
        t2 = verdict.tier2 or {}
        if not t2.get("success", False):
            return 0.0
        conf = float(t2.get("confidence", 0.5))
        return conf if not t2.get("is_fake", False) else (1.0 - conf)

    def evaluate(self, image_path: str, family: str):
        return self.verifier.verify(forged_path=image_path, attack_family=family)

    # ── main loop ──

    def _coerce_descriptor(self, desc: Dict[str, str]) -> Dict[str, str]:
        """Make a (possibly stale / cross-scenario) descriptor valid under the CURRENT
        axis vocab: keep each carried value only if it is still in this axis's vocab
        (a scenario override may have narrowed it), fill every missing/invalid axis
        with a fresh random draw. Lets a warm-started genotype survive axis changes."""
        full = sample_descriptor(self.ax, self.rng)
        for name, vals in self.ax.all_axes.items():
            v = desc.get(name)
            if v in vals:
                full[name] = v
        return full

    def run(self, budget: int, n_seed: int = 6,
            seed_descriptors: Optional[List[Dict[str, str]]] = None) -> ForgeryArchive:
        # seed_descriptors warm-starts the archive from a prior round's winning
        # genotypes (re-evaluated against the CURRENT defender). This is what makes the
        # ATTACKER persist across co-evolution rounds — without it each round rebuilds a
        # fresh archive from random seeds and the forgery structurally cannot improve.
        warm = list(seed_descriptors or [])
        write_manifest(self.out_dir, layer="inner", seed=self._seed,
                       detector_signature=self.verifier.detector_signature,
                       axes_path=self._axes_path,
                       extra={"budget": budget, "n_seed": n_seed,
                              "n_warm_start": len(warm),
                              "top_k": self.archive.top_k,
                              "n_seed_faces": len(self.seed_faces),
                              "grid_axes": self.ax.grid_names})
        log = self.out_dir / "iterations.jsonl"
        log_f = log.open("w")

        def _step(descriptor, parent_id, gen):
            gen_info = self.generate(descriptor)
            if gen_info is None:
                return None
            verdict = self.evaluate(gen_info["image_path"], descriptor["forgery_family"])
            if self.score_fn is not None:
                fitness, bypass = self.score_fn(verdict, descriptor)
            else:
                fitness, bypass = self._fitness_from_verdict(verdict), bool(verdict.sandbox_pass)
            self._gen_counter += 1
            elite = Elite(
                id=uuid.uuid4().hex[:10],
                descriptor=descriptor,
                cell=cell_key(descriptor, self.ax),
                fitness=fitness,
                bypass=bypass,
                face_type=verdict.face_type,
                image_path=gen_info["image_path"],
                op_name=gen_info["op_name"],
                post_op=gen_info["post_op"],
                prompt=gen_info["prompt"],
                parent_id=parent_id,
                gen=gen,
            )
            kept = fitness >= self.fitness_floor and self.archive.add(elite)
            rec = {"i": self._gen_counter, "gen": gen, "cell": list(elite.cell),
                   "fitness": round(fitness, 3), "bypass": elite.bypass,
                   "face_type": elite.face_type, "op": elite.op_name,
                   "post_op": elite.post_op, "kept": kept, "parent": parent_id}
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_f.flush()
            _log.info("[%d] cell=%s fit=%.2f bypass=%s face=%s op=%s kept=%s",
                      self._gen_counter, elite.cell, fitness, elite.bypass,
                      elite.face_type, elite.op_name, kept)
            return elite

        # seed: warm-start from carried genotypes first (re-scored vs the current
        # defender), then top up with random draws to reach at least n_seed.
        for desc in warm:
            _step(self._coerce_descriptor(desc), parent_id=None, gen=0)
        for _ in range(max(0, n_seed - len(warm))):
            _step(sample_descriptor(self.ax, self.rng), parent_id=None, gen=0)

        # evolve
        for it in range(budget):
            parent = self.archive.random_parent(self.rng)
            if parent is None:
                desc = sample_descriptor(self.ax, self.rng)
                pid = None
            else:
                desc = mutate_descriptor(parent.descriptor, self.ax, self.rng)
                pid = parent.id
            _step(desc, parent_id=pid, gen=it + 1)

        log_f.close()
        self.archive.save(self.out_dir / "archive.json", grid_axes=self.ax.grid_names)
        _log.info("DONE cells=%d elites=%d bypass=%d face_types=%s",
                  self.archive.n_cells(), self.archive.n_elites(),
                  self.archive.n_bypass(), self.archive.face_type_counts())
        return self.archive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--n-seed", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--axes", default=str(DEFAULT_AXES))
    ap.add_argument("--seed-faces-dir", default=str(_SRC.parent / "data" / "pool_scut_asian"))
    ap.add_argument("--seed-list", default=None,
                    help="text file of clean face paths (one per line); "
                         "auto-detects <pool>_clean.txt if omitted")
    ap.add_argument("--out", default=str(_SRC.parent / "runs" / "inner_demo"))
    ap.add_argument("--tier2", default="fakevlm_local",
                    choices=["fakevlm_local", "viviai"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    engine = InnerMapElites(
        axes_path=Path(a.axes),
        seed_faces_dir=Path(a.seed_faces_dir),
        seed_list=Path(a.seed_list) if a.seed_list else None,
        out_dir=Path(a.out),
        top_k=a.top_k,
        tier2_backend=a.tier2,
        seed=a.seed,
    )
    engine.run(budget=a.budget, n_seed=a.n_seed)


if __name__ == "__main__":
    main()
