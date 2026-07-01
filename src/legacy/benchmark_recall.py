"""P2: public-benchmark black-recall evaluation.

Reads K real + K fake samples from a benchmark directory (default FF++) and
measures detector recall under attack (黑召回). Compares defender LoRA vs
base FakeVLM. Output: per-detector recall + ROC-style table.

Black-recall target (INTERNSHIP_PLAN_LAI.md KPI): ≥ 0.93 (baseline 0.88, +5pp).

Usage:
  python benchmark_recall.py \
    --real-dir /cpfs01/oss_dataset/lyx/Forgery/test/ff++/real \
    --fake-dir /cpfs01/oss_dataset/lyx/Forgery/test/ff++/fake \
    --n-per-class 50 \
    --out /data/disk4/.../bench_ff++.json
"""
from __future__ import annotations
import argparse, json, time, random, sys, os
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_eval import judge_with_fakevlm, judge_with_gemini, JudgeVerdict
from viviai_client import ViviClient


def collect_images(directory: str, n: int, seed: int = 42) -> list[Path]:
    """Recursively gather PNG/JPG up to n samples, shuffled deterministically."""
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
        paths.extend(Path(directory).rglob(ext))
    random.Random(seed).shuffle(paths)
    return paths[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--fake-dir", required=True)
    ap.add_argument("--n-per-class", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detectors", nargs="+",
                    default=["defender_lora", "fakevlm_base"],
                    choices=["defender_lora", "fakevlm_base", "gemini"],
                    help="which detectors to evaluate")
    ap.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--gemini-model", default="gemini-2.5-flash")
    ap.add_argument("--gemini-key",
        default=os.environ.get("VIVIAI_KEY", ""))
    args = ap.parse_args()

    real = collect_images(args.real_dir, args.n_per_class)
    fake = collect_images(args.fake_dir, args.n_per_class)
    print(f"[bench] {len(real)} real + {len(fake)} fake; detectors={args.detectors}")
    if not real or not fake:
        print("ERROR: not enough samples", file=sys.stderr); sys.exit(1)

    client = ViviClient(api_key=args.gemini_key) if "gemini" in args.detectors else None

    rows = []
    for label, paths in [("real", real), ("fake", fake)]:
        for i, p in enumerate(paths):
            verdicts = {}
            t0 = time.time()
            if "defender_lora" in args.detectors:
                verdicts["defender_lora"] = asdict(judge_with_fakevlm(
                    str(p), lora=True, endpoint=args.vllm_endpoint))
            if "fakevlm_base" in args.detectors:
                verdicts["fakevlm_base"] = asdict(judge_with_fakevlm(
                    str(p), lora=False, endpoint=args.vllm_endpoint))
            if "gemini" in args.detectors:
                verdicts["gemini"] = asdict(judge_with_gemini(
                    str(p), client, model=args.gemini_model))
            dt = time.time() - t0
            rows.append({"label": label, "image": str(p),
                          "verdicts": verdicts, "duration_s": round(dt, 1)})
            if (i+1) % 10 == 0:
                print(f"  {label} {i+1}/{len(paths)} done")

    # ── compute metrics per detector ──
    metrics = {}
    for det in args.detectors:
        tp = sum(1 for r in rows if r["label"] == "fake"
                  and r["verdicts"].get(det, {}).get("is_fake") is True)
        fn = sum(1 for r in rows if r["label"] == "fake"
                  and r["verdicts"].get(det, {}).get("is_fake") is False)
        tn = sum(1 for r in rows if r["label"] == "real"
                  and r["verdicts"].get(det, {}).get("is_fake") is False)
        fp = sum(1 for r in rows if r["label"] == "real"
                  and r["verdicts"].get(det, {}).get("is_fake") is True)
        total = tp + fn + tn + fp
        recall_fake = tp / max(tp + fn, 1)
        precision_fake = tp / max(tp + fp, 1)
        accuracy = (tp + tn) / max(total, 1)
        specificity = tn / max(tn + fp, 1)
        f1 = 2 * precision_fake * recall_fake / max(precision_fake + recall_fake, 1e-9)
        metrics[det] = {
            "TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "black_recall (fake recall)": round(recall_fake, 4),
            "precision_fake": round(precision_fake, 4),
            "specificity (real recall)": round(specificity, 4),
            "accuracy": round(accuracy, 4),
            "F1_fake": round(f1, 4),
        }

    out = {"benchmark": {"real_dir": args.real_dir,
                         "fake_dir": args.fake_dir,
                         "n_per_class": args.n_per_class,
                         "n_real_actual": len(real), "n_fake_actual": len(fake)},
           "metrics": metrics,
           "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"\n[bench] → {args.out}")
    for det, m in metrics.items():
        br = m['black_recall (fake recall)']
        kpi = "✓ ≥0.93" if br >= 0.93 else "✗ <0.93"
        print(f"  {det:18s}: black_recall={br:.4f} {kpi}  "
              f"F1={m['F1_fake']:.4f}  acc={m['accuracy']:.4f}")


if __name__ == "__main__":
    main()
