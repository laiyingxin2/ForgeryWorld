"""LivePortrait reenactment worker — runs INSIDE the isolated `forgery_img` env.

Source portrait + ONE driving frame → ONE animated IMAGE. Identity comes from the
source; expression + head pose come from the driving frame. Image-only (no video):
LivePortrait detects that the driving input is a single image and emits a `.jpg`.

Invoked as a subprocess by operators/local_reenact_advpatch.py so the KwaiVGI repo
+ its torch/onnx deps never load in the orchestrator's `fakevlm` env.

CLI:
    python liveportrait_worker.py --src face.png --driving drv.png --out out.png

Prints exactly one line on success:  OK <output_path>
Exits non-zero with a traceback on failure.
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

LP_REPO = Path("/data/disk4/lyx_ICML/third_party/LivePortrait")


def _partial(cls, kwargs):
    return cls(**{k: v for k, v in kwargs.items() if hasattr(cls, k)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)      # source portrait (identity kept)
    ap.add_argument("--driving", required=True)  # single driving frame (motion source)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(LP_REPO))
    from src.config.argument_config import ArgumentConfig
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / f".lp_{out.stem}"
    work.mkdir(parents=True, exist_ok=True)
    # Copy the driving frame into the work dir: LivePortrait dumps a cached `.pkl`
    # motion template next to the driving file, and we don't want to pollute the
    # shared driving-pool directory.
    drv = work / Path(a.driving).name
    shutil.copy(a.driving, drv)

    args = ArgumentConfig(
        source=str(a.src), driving=str(drv), output_dir=str(work),
        flag_pasteback=True, flag_stitching=True, flag_do_crop=True,
        flag_do_torch_compile=False,
    )
    pipe = LivePortraitPipeline(
        inference_cfg=_partial(InferenceConfig, args.__dict__),
        crop_cfg=_partial(CropConfig, args.__dict__),
    )
    wfp, _concat = pipe.execute(args)
    if not wfp or not Path(wfp).exists():
        raise RuntimeError("liveportrait produced no output image")
    # LivePortrait writes a .jpg; re-encode as lossless PNG so this op matches the
    # other operators (no JPEG artifacts — those are a forensic signal the detector
    # keys on, and would confound the reenact-family verdict).
    import cv2
    img = cv2.imread(str(wfp))
    if img is None:
        raise RuntimeError(f"failed to read liveportrait output {wfp}")
    cv2.imwrite(str(out), img)
    shutil.rmtree(work, ignore_errors=True)
    print(f"OK {out}")


if __name__ == "__main__":
    main()
