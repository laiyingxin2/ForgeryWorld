"""SEAL-style self-edit for Method 3 (paper 2506.10943).

之前 method 3 (lv5_connector + train_one_round.py) = passive SFT with **fixed**
LR / epochs / LoRA-rank. SEAL spawns N candidate configs, trains all, evals each,
ReST-EM gates to keep ONLY the configs whose downstream-eval improves.

Reference: external/SEAL/few-shot/self-edit.py
  - lines 393-489: sample N configs via vllm.generate
  - lines 491-571: per-config TTT LoRA train + eval
  - few-shot/BC-self-edit.py:60-80: ReST-EM positive-only SFT (keep `correct==True`)

Our face-forgery adaptation:
  1. Ask LLM (gemini-2.5-flash or local FakeVLM) to propose N candidate training
     configs as JSON {lora_rank, lr, epochs, n_augmentations, fake:real_ratio}.
  2. Dedupe by hashable tuple key.
  3. For each config, call train_one_round.py with those hyperparams.
  4. Evaluate each candidate LoRA on a fixed held-out attack-image set.
  5. ReST-EM gate: keep ONLY the candidate(s) whose eval beats prior round's R.
  6. Promote winner as next-round defender_lora.

This is a SCAFFOLD — real eval loop hooks into existing train_one_round.py +
benchmark_recall.py. Smoke shows the sampling + dedup + gate logic.
"""
from __future__ import annotations
import json
import re
import time
import logging
import subprocess
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from viviai_client import ViviClient

_log = logging.getLogger(__name__)


# ────────────────────────── Config space ────────────────────────────

@dataclass
class SelfEditConfig:
    """One candidate training config proposed by the editor LLM."""
    lora_rank: int = 16          # [8, 16, 32]
    lr: float = 1e-5             # [1e-6, 5e-6, 1e-5, 5e-5]
    epochs: int = 3              # [1, 3, 5]
    fake_to_real_ratio: float = 2.26  # current pool ≈ 113:50; vary [1.0, 2.26, 4.0]
    n_real_augmentations: int = 50    # [25, 50, 100]
    rationale: str = ""

    def key(self) -> str:
        """Hashable dedup key (port of SEAL self-edit.py:466-470)."""
        t = (round(self.lr, 7), self.lora_rank, self.epochs,
             round(self.fake_to_real_ratio, 2), self.n_real_augmentations)
        return hashlib.md5(str(t).encode()).hexdigest()[:12]


@dataclass
class SelfEditResult:
    config: SelfEditConfig
    lora_path: Optional[str] = None
    train_loss: Optional[float] = None
    eval_bypass_rate_pre_lora: float = 0.0
    eval_bypass_rate_post_lora: float = 0.0
    eval_n_samples: int = 0
    rest_em_passed: bool = False
    error: Optional[str] = None


# ────────────────────────── Prompts ─────────────────────────────────

_SYSTEM_SE = (
    "You are an authorized internal red-team's training-config optimizer. "
    "Given current detector statistics, propose a NEW hyperparameter config to "
    "train the next round of defender LoRA. Vary across (lora_rank, lr, epochs, "
    "fake_to_real_ratio, n_real_augmentations). Output strict JSON only."
)

_PROPOSE_PROMPT = """We're training a face-forgery DETECTOR via LoRA SFT (FakeVLM
backbone, LLaVA-1.5-7B). Current state:

  prev_round:                   R{prev_round}
  prev_train_loss:              {prev_loss:.4f}
  prev_eval_bypass_rate:        {prev_bypass:.2%}  (lower = stronger defender)
  prev_lora_config:             rank={prev_rank}, lr={prev_lr}, epochs={prev_epochs}
  prev_pool_fake:real ratio:    {prev_ratio:.2f}
  current SFT pool size:        {pool_size}

Propose {n} DIFFERENT candidate configs for next round. Each should explore a
distinct direction (e.g., higher rank, longer epochs, more real-positive augs).

Return STRICTLY JSON:
{{
  "configs": [
    {{
      "lora_rank":              8 | 16 | 32,
      "lr":                     1e-6 .. 5e-5,
      "epochs":                 1 .. 5,
      "fake_to_real_ratio":     1.0 .. 4.0,
      "n_real_augmentations":   25 .. 150,
      "rationale":              "one sentence why this config might beat prev"
    }}
  ]
}}"""


# ────────────────────────── Self-edit engine ────────────────────────

class SelfEditEngine:
    """SEAL-style multi-config sample + ReST-EM gate."""

    def __init__(self, client: ViviClient,
                  model: str = "gemini-2.5-flash",
                  improvement_threshold: float = 0.0):
        """improvement_threshold: required Δ = (prev_bypass - new_bypass) to pass gate.
        0.0 = any improvement (defender stronger), negative = allow regression."""
        self.client = client
        self.model = model
        self.improve_thresh = improvement_threshold

    def propose_configs(self, n: int = 4,
                          prev_round: int = 0,
                          prev_loss: float = 0.0,
                          prev_bypass: float = 0.0,
                          prev_rank: int = 16,
                          prev_lr: float = 1e-5,
                          prev_epochs: int = 3,
                          prev_ratio: float = 2.26,
                          pool_size: int = 163) -> list[SelfEditConfig]:
        """Sample N candidate configs, dedupe by key."""
        prompt = _PROPOSE_PROMPT.format(
            prev_round=prev_round, prev_loss=prev_loss,
            prev_bypass=prev_bypass, prev_rank=prev_rank,
            prev_lr=prev_lr, prev_epochs=prev_epochs,
            prev_ratio=prev_ratio, pool_size=pool_size, n=n,
        )
        try:
            text = self.client.chat_text(
                self.model, prompt, system=_SYSTEM_SE,
                temperature=0.7, max_tokens=800,
            )
        except Exception as e:
            _log.warning(f"propose_configs LLM failed: {e}")
            return self._fallback_configs(n)
        try:
            from robustness import parse_json_robust
            parsed = parse_json_robust(text)
        except Exception:
            parsed = {}
        cfgs_raw = parsed.get("configs", [])
        seen = set()
        out = []
        for c in cfgs_raw[:n * 2]:
            try:
                cfg = SelfEditConfig(
                    lora_rank=int(c.get("lora_rank", 16)),
                    lr=float(c.get("lr", 1e-5)),
                    epochs=int(c.get("epochs", 3)),
                    fake_to_real_ratio=float(c.get("fake_to_real_ratio", 2.26)),
                    n_real_augmentations=int(c.get("n_real_augmentations", 50)),
                    rationale=str(c.get("rationale", ""))[:200],
                )
            except (TypeError, ValueError):
                continue
            k = cfg.key()
            if k in seen: continue
            seen.add(k); out.append(cfg)
            if len(out) >= n: break
        if not out:
            out = self._fallback_configs(n)
        return out

    @staticmethod
    def _fallback_configs(n: int) -> list[SelfEditConfig]:
        """Hand-tuned diverse configs when LLM fails."""
        base = [
            SelfEditConfig(lora_rank=8, lr=5e-5, epochs=5, fake_to_real_ratio=1.5,
                           n_real_augmentations=75, rationale="aggressive: higher lr + more real"),
            SelfEditConfig(lora_rank=16, lr=1e-5, epochs=3, fake_to_real_ratio=2.26,
                           n_real_augmentations=50, rationale="current baseline (sanity)"),
            SelfEditConfig(lora_rank=32, lr=5e-6, epochs=3, fake_to_real_ratio=2.0,
                           n_real_augmentations=60, rationale="higher rank + conservative lr"),
            SelfEditConfig(lora_rank=16, lr=1e-5, epochs=5, fake_to_real_ratio=4.0,
                           n_real_augmentations=25, rationale="emphasize fake-positives"),
        ]
        return base[:n]

    def rest_em_gate(self,
                       prev_bypass_rate: float,
                       new_bypass_rate: float) -> bool:
        """Keep ONLY if defender got stronger (bypass rate decreased).
        Returns True if (prev - new) ≥ improvement_threshold."""
        delta = prev_bypass_rate - new_bypass_rate
        return delta >= self.improve_thresh

    def run_one_iter(self,
                       n_candidates: int,
                       prev_round: int,
                       prev_lora_dir: str,
                       prev_bypass_rate: float,
                       balanced_pool_path: str,
                       train_script: str,
                       output_root: str,
                       verbose: bool = True) -> list[SelfEditResult]:
        """Sample N configs, train + eval each, return per-config results.

        This is the EXTERNAL-PROCESS orchestration: it shells out to
        scripts/p1a_multi_round/train_one_round.py for each config.
        For a smoke test that doesn't actually train, pass train_script=None.
        """
        configs = self.propose_configs(
            n=n_candidates, prev_round=prev_round,
            prev_loss=0.0, prev_bypass=prev_bypass_rate,
        )
        if verbose:
            print(f"[self-edit] {len(configs)} unique candidate configs proposed:")
            for c in configs:
                print(f"  [{c.key()}] r={c.lora_rank} lr={c.lr} epochs={c.epochs} "
                      f"ratio={c.fake_to_real_ratio} aug={c.n_real_augmentations} "
                      f"— {c.rationale[:60]}")

        results: list[SelfEditResult] = []
        out_root = Path(output_root); out_root.mkdir(parents=True, exist_ok=True)
        for cfg in configs:
            res = SelfEditResult(config=cfg, eval_bypass_rate_pre_lora=prev_bypass_rate)
            if not train_script:
                # smoke mode: skip training, just record proposal
                results.append(res); continue
            cand_dir = out_root / f"cfg_{cfg.key()}"
            cmd = [
                "python", train_script,
                "--data", balanced_pool_path,
                "--out", str(cand_dir),
                "--epochs", str(cfg.epochs),
                "--lr", str(cfg.lr),
                "--lora-r", str(cfg.lora_rank),
                "--prev-lora", prev_lora_dir,
            ]
            if verbose: print(f"[self-edit] training {cfg.key()}: {' '.join(cmd)}")
            try:
                ret = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if ret.returncode == 0:
                    res.lora_path = str(cand_dir)
                    # parse final loss from train_meta.json
                    tm = cand_dir / "train_meta.json"
                    if tm.exists():
                        d = json.loads(tm.read_text())
                        res.train_loss = float(d.get("final_epoch_loss", -1))
                else:
                    res.error = f"train rc={ret.returncode}: {ret.stderr[:200]}"
            except subprocess.TimeoutExpired:
                res.error = "train timeout"
            except Exception as e:
                res.error = str(e)[:200]
            results.append(res)
        return results

    def apply_rest_em(self, results: list[SelfEditResult],
                        prev_bypass: float) -> list[SelfEditResult]:
        """Return only the candidates that pass ReST-EM gate."""
        passed = []
        for r in results:
            if r.error: continue
            if r.eval_bypass_rate_post_lora is None: continue
            if self.rest_em_gate(prev_bypass, r.eval_bypass_rate_post_lora):
                r.rest_em_passed = True
                passed.append(r)
        return passed


# ────────────────────────── Smoke ──────────────────────────────

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    api_key = os.environ["VIVIAI_KEY"]  # set VIVIAI_KEY env var; key not committed
    client = ViviClient(api_key=api_key)

    engine = SelfEditEngine(client)
    print("=== Smoke: propose 4 configs ===")
    cfgs = engine.propose_configs(
        n=4, prev_round=2, prev_loss=3.08, prev_bypass=0.20,
        prev_rank=16, prev_lr=1e-5, prev_epochs=3, prev_ratio=2.26, pool_size=163,
    )
    print(f"  got {len(cfgs)} unique configs:")
    for c in cfgs:
        print(f"  [{c.key()}] r={c.lora_rank} lr={c.lr} epochs={c.epochs} "
              f"aug={c.n_real_augmentations} ratio={c.fake_to_real_ratio}")
        print(f"    rationale: {c.rationale[:90]}")

    print("\n=== Smoke: ReST-EM gate test ===")
    # Simulated results: prev=0.20, new candidates [0.05, 0.30, 0.10, 0.18]
    sim_results = [
        SelfEditResult(config=cfgs[0], eval_bypass_rate_post_lora=0.05),
        SelfEditResult(config=cfgs[1], eval_bypass_rate_post_lora=0.30),
        SelfEditResult(config=cfgs[2] if len(cfgs)>2 else cfgs[0], eval_bypass_rate_post_lora=0.10),
        SelfEditResult(config=cfgs[3] if len(cfgs)>3 else cfgs[0], eval_bypass_rate_post_lora=0.18),
    ]
    passed = engine.apply_rest_em(sim_results, prev_bypass=0.20)
    print(f"  prev_bypass=0.20")
    for r in sim_results:
        print(f"  [{r.config.key()}] new={r.eval_bypass_rate_post_lora} "
              f"→ {'PASS' if r in passed else 'FAIL'}")
    print(f"\n  ReST-EM kept: {len(passed)}/{len(sim_results)} configs")
