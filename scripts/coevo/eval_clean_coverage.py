"""Valid-measurement re-scoring of existing red-team runs (NO new runs, FREE).

Implements the "comparison requires valid measurement" practice (arXiv:2601.18076)
+ structural-validity coverage audit (arXiv:2605.15118): a still-image forgery
detector cannot be meaningfully "bypassed" by families that produce no still-image
manipulation of the source face. Those families inflate raw coverage as a
MEASUREMENT ARTIFACT, not a real bypass. We recompute coverage / best-so-far over
the VALID (modality-coherent) families only, and report the delta.

VALID (digital still-image face manipulation):
    frontal_swap, profile_swap, id_diff, morph, adv_patch
INVALID / no-op for a still-image detector (audio / temporal / physical / liveness):
    audio_synth (voice clone, no image), reenact (head-motion liveness),
    3d_mask (physical presentation), replay (screen recapture liveness)

Usage:
  python eval_clean_coverage.py LABEL1=<dir_with_reports> LABEL2=<dir> ...
  e.g. python eval_clean_coverage.py \
      M1=outputs/p8_faithful/p8_20260621_1840/m1 \
      M2=outputs/p8_faithful/p8_20260621_1840/m2
Each <dir> must contain reports/round_*_v*.json with diagnosis.family_bypass_rates.
"""
from __future__ import annotations
import glob
import json
import os
import re
import sys

VALID = {"frontal_swap", "profile_swap", "id_diff", "morph", "adv_patch"}
INVALID = {"audio_synth", "reenact", "3d_mask", "replay"}


def round_reports(d):
    """Return [(round_idx, family_bypass_rates_dict)] sorted by round."""
    out = []
    for f in glob.glob(os.path.join(d, "reports", "round_*_v*.json")):
        m = re.search(r"round_(\d+)_v", os.path.basename(f))
        if not m:
            continue
        try:
            data = json.load(open(f))
        except Exception:
            continue
        fb = (data.get("diagnosis", {}) or {}).get("family_bypass_rates", {}) or {}
        out.append((int(m.group(1)), fb))
    out.sort(key=lambda x: x[0])
    return out


def analyze(label, d):
    reps = round_reports(d)
    if not reps:
        return None
    raw_cov, val_cov = set(), set()
    rows = []
    raw_best = val_best = 0.0
    for r, fb in reps:
        byp = {k: v for k, v in fb.items() if v and v > 0}
        raw_cov |= set(byp)
        val_byp = {k: v for k, v in byp.items() if k in VALID}
        val_cov |= set(val_byp)
        raw_best = max([raw_best] + list(byp.values()))
        val_best = max([val_best] + list(val_byp.values()))
        rows.append({
            "round": r,
            "raw_cov_cum": len(raw_cov), "valid_cov_cum": len(val_cov),
            "raw_best": round(raw_best, 3), "valid_best": round(val_best, 3),
            "valid_families": sorted(val_cov),
            "excluded_families": sorted(set(byp) & INVALID),
        })
    return {"label": label, "dir": d, "rows": rows}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    results = []
    for arg in sys.argv[1:]:
        if "=" not in arg:
            print(f"skip (need LABEL=dir): {arg}")
            continue
        label, d = arg.split("=", 1)
        res = analyze(label, d)
        if res is None:
            print(f"[{label}] no round_*_v*.json reports under {d}/reports")
            continue
        results.append(res)

    for res in results:
        print(f"\n===== {res['label']}  ({res['dir']}) =====")
        print(f"{'rnd':>3} | {'raw_cov':>7} {'valid_cov':>9} | "
              f"{'raw_best':>8} {'val_best':>8} | valid_families  [excluded no-op]")
        for x in res["rows"]:
            print(f"{x['round']:>3} | {x['raw_cov_cum']:>7} {x['valid_cov_cum']:>9} | "
                  f"{x['raw_best']:>8.0%} {x['valid_best']:>8.0%} | "
                  f"{x['valid_families']}  [{','.join(x['excluded_families']) or '-'}]")

    print("\n===== HEADLINE (last round, valid-only) =====")
    for res in results:
        last = res["rows"][-1]
        print(f"  {res['label']:>6}: valid_coverage_cum={last['valid_cov_cum']} "
              f"valid_best_so_far={last['valid_best']:.0%}  "
              f"(raw was cov={last['raw_cov_cum']} best={last['raw_best']:.0%})")

    out = {"valid_families": sorted(VALID), "excluded_families": sorted(INVALID),
           "results": results}
    op = "outputs/clean_coverage_eval.json"
    json.dump(out, open(op, "w"), indent=2)
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
