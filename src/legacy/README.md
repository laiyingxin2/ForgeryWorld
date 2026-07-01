# legacy/ — deprecated enumerated-pool architecture

These 38 modules implement the **old fixed-dimension / enumerated operator-pool**
attack system (Methods M1–M4, the lv5 self-edit loop, the multi-agent generators,
the ACE skill library, the co-evolution orchestrator, and their analysis/eval
scripts). They were superseded on 2026-06-29 by the **two-layer open-ended
evolutionary red-team** under `src/evolve/`:

- `evolve/inner_mapelites.py` — inner MAP-Elites over fine-grained forgery descriptors
- `evolve/scenario.py` + `evolve/outer_poet.py` — outer POET/OMNI-EPIC scenario evolution
- `evolve/metrics.py` — frozen-detector-valid metrics (coverage / best-so-far / weak-family ASR)

## Why moved, not deleted
Conservative cleanup: nothing is destroyed. The new pipeline's live import closure is
small and self-contained — `operators/`, `evolve/`, `sandbox.py`, `viviai_client.py`,
`fakevlm_judge_real.py` — and verified to import none of these files. They are parked
here (reversible: move back to `src/` to restore) so the active source tree shows only
the two-layer architecture.

## What still lives in src/ (NOT moved)
- `operators/`  — Layer-0 attack operator pool (reused as inner primitives)
- `evolve/`     — the new two-layer pipeline
- `sandbox.py`  — Tier-1/2 verifier (FakeVLM :8001 fitness signal)
- `viviai_client.py`, `fakevlm_judge_real.py` — detector/LLM clients

## Caveat
Scripts under `../../scripts/` (p6_selfevo_metrics, coevo/*, etc.) were written against
this old architecture and import these modules by flat name; they will not run while the
files live here. They are part of the same deprecated flow and were intentionally left
untouched. Restore the relevant module to `src/` if an old experiment must be re-run.
