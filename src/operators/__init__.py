"""Layer 0 — Attack Operator Pool.

15 个 operator 混合本地 + API:
- local_swap.py     : InSwapper-128, SimSwap-256, Roop                  (frontal_swap/profile_swap)
- local_reenact.py  : LivePortrait, FaceVid2Vid                          (reenact)
- local_morph.py    : StyleGAN-morph                                     (morph)
- local_3dmask.py   : DECA/FLAME                                         (3d_mask)
- local_replay.py   : screen replay + Moiré + recompress (custom code)   (replay)
- local_advpatch.py : PGD on FAS CNN (torchattacks)                       (adv_patch)
- local_audio.py    : XTTS voice clone                                   (audio_synth)
- api_image.py      : viviai nanobanana 1/2/pro + gpt-image-2            (id_diff/morph 高质量)
"""

from operators.api_image import (
    ApiImageOperator,
    NanoBananaPro,
    NanoBananaTwo,
    NanoBananaOne,
    GptImageTwo,
    OPERATOR_REGISTRY,
)
# ★ 本地 face-swap (替代失效的 nano_banana viviai endpoints)
try:
    from operators.local_swap import LocalInSwapperOperator
    OPERATOR_REGISTRY["inswapper_128_local"] = LocalInSwapperOperator
except ImportError as _e:
    LocalInSwapperOperator = None
    print(f"[operators] LocalInSwapper unavailable: {_e}")

# ★ P0-D: 2nd 本地 face-swap (Chen 2020 SimSwap, profile-friendly)
try:
    from operators.local_simswap import LocalSimSwapOperator
    OPERATOR_REGISTRY["simswap_256_local"] = LocalSimSwapOperator
except ImportError as _e:
    LocalSimSwapOperator = None
    print(f"[operators] LocalSimSwap unavailable: {_e}")

# ★ CRITICAL FIX (2026-06-20): basic image postprocess ops were referenced by
# chain steps everywhere but NOT registered → all became mock pass-through →
# attack image had only 1-2 real steps → tier2 gemini one-shot caught it.
# Adding face_align/jpeg_85/resize_bicubic/gfpgan as real local ops.
try:
    from operators.local_postprocess import (
        FaceAlignOperator, JpegCompressOperator,
        ResizeBicubicOperator, GFPGANRestoreOperator, LightingOperator,
    )
    OPERATOR_REGISTRY["face_align"] = FaceAlignOperator
    OPERATOR_REGISTRY["jpeg_85"] = JpegCompressOperator
    OPERATOR_REGISTRY["resize_bicubic"] = ResizeBicubicOperator
    OPERATOR_REGISTRY["gfpgan"] = GFPGANRestoreOperator
    OPERATOR_REGISTRY["relight"] = LightingOperator
except ImportError as _e:
    print(f"[operators] local_postprocess unavailable: {_e}")

# ★ 2026-06-20: reenact + adv_patch real ops (real LivePortrait + PGD)
try:
    from operators.local_reenact_advpatch import (
        LivePortraitOperator, AdvPatchPGDOperator,
    )
    OPERATOR_REGISTRY["liveportrait"] = LivePortraitOperator
    OPERATOR_REGISTRY["adv_patch_pgd"] = AdvPatchPGDOperator
except ImportError as _e:
    print(f"[operators] local_reenact_advpatch unavailable: {_e}")

# ★ 2026-06-20: bulk 18 lightweight real ops (cov 9 families)
try:
    from operators.local_methods_bulk import (
        JpegQ70Op, JpegQ95Op, Resize50PctOp, Resize125PctOp,
        USMSharpenOp, GaussianBlurOp, BrightnessShiftOp, HistEqualizeOp,
        FGSMAttackOp, BIMAttackOp,
        MoireInjectOp, ScreenReplaySimOp, RecompressChainOp,
        FaceBlendMorphOp, FaceRotate3DOp,
        AudioOverlayMetaOp, WebPCompressOp, PalettePNGOp,
    )
    OPERATOR_REGISTRY["jpeg_70"] = JpegQ70Op
    OPERATOR_REGISTRY["jpeg_95"] = JpegQ95Op
    OPERATOR_REGISTRY["resize_50pct"] = Resize50PctOp
    OPERATOR_REGISTRY["resize_125pct"] = Resize125PctOp
    OPERATOR_REGISTRY["usm_sharpen"] = USMSharpenOp
    OPERATOR_REGISTRY["gaussian_blur"] = GaussianBlurOp
    OPERATOR_REGISTRY["brightness_shift"] = BrightnessShiftOp
    OPERATOR_REGISTRY["hist_equalize"] = HistEqualizeOp
    OPERATOR_REGISTRY["fgsm_attack"] = FGSMAttackOp
    OPERATOR_REGISTRY["bim_attack"] = BIMAttackOp
    OPERATOR_REGISTRY["moire_inject"] = MoireInjectOp
    OPERATOR_REGISTRY["screen_replay_sim"] = ScreenReplaySimOp
    OPERATOR_REGISTRY["recompress_chain"] = RecompressChainOp
    OPERATOR_REGISTRY["stylegan_morph"] = FaceBlendMorphOp
    OPERATOR_REGISTRY["deca_3dmask"] = FaceRotate3DOp
    OPERATOR_REGISTRY["xtts_audio"] = AudioOverlayMetaOp
    OPERATOR_REGISTRY["webp_compress"] = WebPCompressOp
    OPERATOR_REGISTRY["png_palette"] = PalettePNGOp
except ImportError as _e:
    print(f"[operators] local_methods_bulk unavailable: {_e}")

# ★ 2026-06-24: diffusion ID/synthesis/edit operators (image-only forgery family).
# Run in the ISOLATED forgery_img conda env via subprocess; weights already on disk
# (SDXL base, SD1.5 base, InstructPix2Pix, IP-Adapter plus-face, InstantID).
try:
    from operators.local_id_diffusion import (
        InstructPix2PixOperator, IPAdapterFaceOperator,
        SDXLSynthOperator, InstantIDOperator,
    )
    OPERATOR_REGISTRY["instructpix2pix"] = InstructPix2PixOperator
    OPERATOR_REGISTRY["ipadapter_face"] = IPAdapterFaceOperator
    OPERATOR_REGISTRY["sdxl_t2i"] = SDXLSynthOperator
    OPERATOR_REGISTRY["instantid"] = InstantIDOperator
except ImportError as _e:
    print(f"[operators] local_id_diffusion unavailable: {_e}")

# ★ 2026-06-20 fix: legacy/phantom op names → real registry keys. Seed libraries,
# MCTS mutation maps, brief_hints and persisted DB chains across the codebase still
# emit the OLD names (inswapper_128, simswap_256, roop, facevid2vid, replay_sim).
# Those are NOT registry keys, so they errored at execution and dropped the
# identity-producing step → faceless forgeries that trivially "bypass" a face
# detector. Resolve every tool name through this alias at dispatch time.
OP_ALIAS = {
    "inswapper_128": "inswapper_128_local",
    "simswap_256": "simswap_256_local",
    "simswap": "simswap_256_local",
    "roop": "inswapper_128_local",      # same frontal-swap family, no separate impl
    "facevid2vid": "liveportrait",       # same reenact family, no separate impl
    "replay_sim": "screen_replay_sim",
}


def resolve_op(name: str) -> str:
    """Map a possibly-legacy op name to its canonical OPERATOR_REGISTRY key."""
    return OP_ALIAS.get(name, name)


__all__ = [
    "ApiImageOperator",
    "NanoBananaPro",
    "NanoBananaTwo",
    "NanoBananaOne",
    "GptImageTwo",
    "LocalInSwapperOperator",
    "LocalSimSwapOperator",
    "OPERATOR_REGISTRY",
    "OP_ALIAS",
    "resolve_op",
]
