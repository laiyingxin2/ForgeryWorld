"""Scenario representation + generation for the OUTER layer.

A *scenario* is an open-ended KYC/liveness context that constrains the INNER
MAP-Elites descriptor space to a sub-region and gives it a natural-language framing
(e.g. "remote bank onboarding, applicant filming a printed photo under a desk lamp").
The outer POET/OMNI-EPIC loop evolves a population of these.

Concretely a Scenario carries:
  - axis_constraints : {grid/tag axis -> allowed value subset}, narrowing what the
                       inner archive may sample (passed to InnerMapElites.axis_overrides).
  - description      : NL context string (used for logging / future LLM prompt steering).
  - name            : short slug.

Generation is PLUGGABLE so the layer runs end-to-end without an LLM key:
  - TemplateScenarioGenerator : combinatorial sampling over the axis vocab + a slot
                                template. Deterministic given a seed. Always available.
  - LLMScenarioGenerator      : OMNI-EPIC style — asks a ViviClient chat model to
                                invent a novel KYC scenario as JSON {name, description,
                                constraints}; falls back to the template on any failure.

We keep the constraint vocab anchored to configs/evolve_axes.yaml so a scenario can
never reference an axis value the inner loop doesn't understand.
"""
from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger("outer.scenario")


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    axis_constraints: Dict[str, List[str]]
    parent_id: Optional[str] = None
    gen: int = 0
    # filled in by the outer loop after the inner archive runs:
    score: Optional[float] = None             # MC score (inner bypass rate)
    behavior: Optional[List[float]] = None    # PATA-EC: per-family bypass-rate vector
    n_bypass: int = 0
    n_cells: int = 0
    novelty: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)


# A small KYC-context template bank keyed by the dominant constraint, so the NL
# description is coherent with the axis sub-region the scenario actually enforces.
_PAI_CONTEXT = {
    "print_photo": "the applicant holds a printed photo up to the verification camera",
    "replay_video": "the applicant replays a recorded video of the target on a second screen",
    "mask_2d": "the applicant wears a flat 2D paper mask of the target's face",
    "mask_3d": "the applicant wears a silicone 3D mask of the target",
    "paper_cut": "a paper cut-out of the target's eyes is held over the applicant's face",
    "deepfake_injection": "a deepfake stream is injected into the camera feed via a virtual device",
    "live": "the applicant presents their live face to the camera",
}
_LIGHT_CONTEXT = {
    "dark": "in a dim, under-lit room",
    "strong": "under harsh direct lighting",
    "back_light": "back-lit by a bright window behind them",
    "normal": "under even ambient light",
}
_ENV_CONTEXT = {
    "indoor_phone": "captured on a handheld phone indoors",
    "outdoor_phone": "captured on a phone outdoors",
    "pc_webcam": "captured on a laptop webcam",
    "pad_cam": "captured on a tablet camera",
}


class TemplateScenarioGenerator:
    """Combinatorial scenario generator — no LLM. Samples a sub-region of the axis
    vocab and fills a KYC template. `constrain_axes` decides which axes get pinned to
    a narrow subset (the scenario's defining theme); the rest stay unconstrained so
    the inner loop still diversifies them."""

    def __init__(self, axis_vocab: Dict[str, List[str]], rng: Optional[random.Random] = None,
                 constrain_axes: Optional[List[str]] = None):
        self.vocab = axis_vocab
        self.rng = rng or random.Random(0)
        # default themed axes: PAI + lighting (+ environment if present)
        self.constrain_axes = constrain_axes or [
            a for a in ("pai", "lighting", "environment") if a in axis_vocab
        ]

    def _pin(self, axis: str) -> List[str]:
        vals = self.vocab.get(axis, [])
        if not vals:
            return []
        k = 1 if len(vals) <= 3 else self.rng.randint(1, 2)
        return self.rng.sample(vals, k=k)

    def _describe(self, constraints: Dict[str, List[str]]) -> str:
        pai = (constraints.get("pai") or ["live"])[0]
        light = (constraints.get("lighting") or ["normal"])[0]
        env = (constraints.get("environment") or [None])[0]
        parts = [_PAI_CONTEXT.get(pai, "the applicant presents to the verification camera"),
                 _LIGHT_CONTEXT.get(light, "")]
        if env and env in _ENV_CONTEXT:
            parts.append(_ENV_CONTEXT[env])
        return "Remote KYC liveness check: " + ", ".join(p for p in parts if p) + "."

    def sample(self, parent: Optional[Scenario] = None, gen: int = 0) -> Scenario:
        if parent is None:
            constraints = {a: self._pin(a) for a in self.constrain_axes}
        else:
            # mutate the parent: re-pin one constrained axis
            constraints = {k: list(v) for k, v in parent.axis_constraints.items()}
            axis = self.rng.choice(self.constrain_axes)
            constraints[axis] = self._pin(axis)
        constraints = {k: v for k, v in constraints.items() if v}
        slug = "_".join((constraints.get(a) or ["any"])[0] for a in self.constrain_axes)
        return Scenario(
            id=uuid.uuid4().hex[:10],
            name=f"sc_{slug}"[:48],
            description=self._describe(constraints),
            axis_constraints=constraints,
            parent_id=parent.id if parent else None,
            gen=gen,
        )


class LLMScenarioGenerator:
    """OMNI-EPIC style generator: ask a chat LLM to invent a NOVEL KYC attack
    scenario as JSON. Falls back to the template generator on any error (no key,
    bad JSON, unknown axis values), so the outer loop never hard-depends on the LLM."""

    _SYS = ("You are a red-team scenario designer for a face-verification (KYC) "
            "liveness detector. Invent ONE novel, concrete spoofing scenario that is "
            "challenging but plausible. Respond with STRICT JSON only.")

    def __init__(self, axis_vocab: Dict[str, List[str]], client=None,
                 model: str = "gemini-2.5-flash", rng: Optional[random.Random] = None,
                 constrain_axes: Optional[List[str]] = None):
        self.vocab = axis_vocab
        self.client = client
        self.model = model
        self.fallback = TemplateScenarioGenerator(axis_vocab, rng, constrain_axes)

    def _prompt(self, parent: Optional[Scenario]) -> str:
        vocab_str = json.dumps({a: self.vocab[a] for a in self.fallback.constrain_axes
                                if a in self.vocab})
        seed_txt = f"\nVary meaningfully from this prior scenario: {parent.description}" if parent else ""
        return (f"Allowed axis values (choose subsets):\n{vocab_str}\n"
                f"Return JSON: {{\"name\": str, \"description\": str, "
                f"\"constraints\": {{axis: [values]}}}}. Only use listed axis values."
                f"{seed_txt}")

    def _coerce(self, obj: dict, parent: Optional[Scenario], gen: int) -> Optional[Scenario]:
        try:
            cons_in = obj.get("constraints", {})
            constraints: Dict[str, List[str]] = {}
            for axis, vals in cons_in.items():
                if axis in self.vocab:
                    keep = [v for v in (vals if isinstance(vals, list) else [vals])
                            if v in self.vocab[axis]]
                    if keep:
                        constraints[axis] = keep
            if not constraints:
                return None
            return Scenario(
                id=uuid.uuid4().hex[:10],
                name=str(obj.get("name", "sc_llm"))[:48],
                description=str(obj.get("description", ""))[:400],
                axis_constraints=constraints,
                parent_id=parent.id if parent else None,
                gen=gen,
            )
        except Exception:
            return None

    def sample(self, parent: Optional[Scenario] = None, gen: int = 0) -> Scenario:
        if self.client is not None:
            try:
                txt = self.client.chat_text(self.model, self._prompt(parent),
                                            system=self._SYS, temperature=0.9,
                                            max_tokens=400)
                start, end = txt.find("{"), txt.rfind("}")
                if start >= 0 and end > start:
                    sc = self._coerce(json.loads(txt[start:end + 1]), parent, gen)
                    if sc is not None:
                        _log.info("LLM scenario: %s", sc.name)
                        return sc
            except Exception as e:
                _log.warning("LLM scenario gen failed (%s); using template", str(e)[:120])
        return self.fallback.sample(parent, gen)
