"""Per-operator validation harness for the inner-layer generation models.

Runs each forgery-family operator ONCE with a controlled descriptor (default: a
`dark`-lighting + `cross_species_ood` prompt) on a known-good base/donor face, scores
the output with the frozen FakeVLM detector (:8001), and reports whether the result
matches intent: success, latency, output brightness (does `dark` actually go dark?),
face-present signal (landmark_consistency), identity retention (arcface vs base),
detector verdict (is_fake / confidence / bypass), and our face_type label.

Use this to test generation models one at a time and tune the YAML mappings before
scaling the MAP-Elites budget.

    python -m evolve.test_operators                 # all default ops
    python -m evolve.test_operators --ops sdxl_t2i instructpix2pix
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from operators import OPERATOR_REGISTRY, resolve_op  # noqa: E402
from sandbox import SandboxVerifier  # noqa: E402
from viviai_client import ViviClient  # noqa: E402
from evolve.inner_mapelites import AxisSpace, build_prompt, DEFAULT_AXES, \
    FAKEVLM_ENDPOINT, FAKEVLM_CKPT  # noqa: E402

_log = logging.getLogger("test_operators")

# the 7 generation ops, grouped by family (post-process ops tested via the chain)
DEFAULT_OPS = ["inswapper_128_local", "simswap_256_local", "liveportrait",
               "sdxl_t2i", "instructpix2pix", "instantid", "ipadapter_face"]


def mean_brightness(path: str) -> float:
    from PIL import Image
    import numpy as np
    return float(np.asarray(Image.open(path).convert("L")).mean())


def build_test_descriptor(ax: AxisSpace, family: str) -> dict:
    """A fixed, intent-heavy descriptor: dark + cross-species + a few steers."""
    return {
        "forgery_family": family,
        "pai": "live",
        "lighting": "dark",
        "generator_family": "diffusion",
        "blend_region": "full_face",
        "semantic_attribute": "none",
        "identity_source": "cross_species_ood",
        "environment_sensor": "phone_cam",
        "pose_yaw": "frontal",
        "occlusion": "none",
        "post_process": "none",
        "perturbation": "none",
    }


# op_key -> family (so we build the right prompt + know if src is needed)
OP_FAMILY = {
    "inswapper_128_local": "swap", "simswap_256_local": "swap",
    "liveportrait": "reenact", "sdxl_t2i": "entire_synthesis",
    "instructpix2pix": "attribute_edit",
    "instantid": "id_diff", "ipadapter_face": "id_diff",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", nargs="*", default=DEFAULT_OPS)
    ap.add_argument("--base", default=str(_SRC.parent / "data" / "real_faces" / "0_row0_real.png"))
    ap.add_argument("--donor", default=str(_SRC.parent / "data" / "real_faces" / "1_row0_real.png"))
    ap.add_argument("--out", default=str(_SRC.parent / "runs" / "op_test"))
    ap.add_argument("--axes", default=str(DEFAULT_AXES))
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir = Path(a.out); (out_dir / "gen").mkdir(parents=True, exist_ok=True)
    ax = AxisSpace.load(Path(a.axes))

    verifier = SandboxVerifier(
        client=ViviClient(api_key="unused-fakevlm-local"),
        tier2_backend="fakevlm_local", fakevlm_endpoint=FAKEVLM_ENDPOINT,
        fakevlm_ckpt_path=FAKEVLM_CKPT, tier3_enabled=False,
    )
    print(f"base brightness  = {mean_brightness(a.base):.1f}")
    print(f"donor brightness = {mean_brightness(a.donor):.1f}\n")

    rows = []
    for op_key in a.ops:
        key = resolve_op(op_key)
        Op = OPERATOR_REGISTRY.get(key)
        if Op is None:
            print(f"[skip] {op_key}: not registered"); continue
        family = OP_FAMILY.get(key, "id_diff")
        desc = build_test_descriptor(ax, family)
        prompt = build_prompt(desc, ax)
        op = Op(out_dir=str(out_dir / "gen"))
        t0 = time.time()
        res = op.run(src_face_path=a.base, tgt_face_path=a.donor,
                     params={"prompt": prompt, "instruction": prompt, "seed": 0})
        dur = time.time() - t0
        if not res.success or not res.output_path:
            print(f"[FAIL] {op_key:22s} ({dur:5.1f}s)  err={res.error}")
            rows.append({"op": op_key, "success": False, "error": res.error,
                         "dur": round(dur, 1)})
            continue
        verdict = verifier.verify(forged_path=res.output_path, src_face_path=a.base,
                                  attack_family=family)
        t2 = verdict.tier2 or {}
        bright = mean_brightness(res.output_path)
        row = {
            "op": op_key, "family": family, "success": True,
            "dur": round(dur, 1), "out": res.output_path,
            "brightness": round(bright, 1),
            "landmark": round(verdict.tier1.get("landmark_consistency", -1), 3),
            "arcface_vs_base": round(verdict.tier1.get("arcface_id_sim", -1), 3),
            "is_fake": t2.get("is_fake"), "conf": t2.get("confidence"),
            "bypass": bool(verdict.sandbox_pass), "face_type": verdict.face_type,
        }
        rows.append(row)
        print(f"[ OK ] {op_key:22s} ({dur:5.1f}s)  bright={bright:5.1f}  "
              f"lm={row['landmark']:.2f}  arc={row['arcface_vs_base']:.2f}  "
              f"fake={row['is_fake']}  bypass={row['bypass']}  face={row['face_type']}")
        print(f"        -> {res.output_path}")

    (out_dir / "op_test_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_dir/'op_test_results.json'}")


if __name__ == "__main__":
    main()
