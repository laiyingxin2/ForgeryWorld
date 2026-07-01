"""3-Tier Sandbox Verifier — Layer 5 of Baseline #2.

3 个 detector tier 按 UI-TARS-2 task-family-routing 风格组合 (不是 3-way 投票):

  Tier 1 (Function-based, deterministic, 每 rollout 必跑, 免费)
    - ArcFace ID-cos sim (现 placeholder, W2 接 insightface)
    - SSIM / LPIPS (现 PIL+skimage, W2 升 lpips lib)
    - NIQE / MANIQA (现 simple, W2 接 piq)
    - FFT spectral artifact (numpy 即可)
    - Face landmark consistency (W2 接 mediapipe / dlib)

  Tier 2 (LLM-as-Judge, 主 detector, UI-TARS-2 Generative ORM 风格)
    - W1-W5: gemini-2.5-flash (最便宜, $0.0015/call)
    - W6+:   gemini-3-pro-preview (准确, $0.005/call, F1>0.8 expected)
    - JSON: {is_fake, confidence, attack_family_guess, reasoning}

  Tier 3 (Forensic cross-check, Agent0-VL grounded, W6+ 启用 top-20%)
    - claude-opus-4-7 vision, 强制引用 Tier-1 数值证据
    - 防 Tier-2 hallucination
    - $0.015/call, 不每 rollout 都跑

调用范式:
    from sandbox import SandboxVerifier
    sb = SandboxVerifier(tier2_model="gemini-2.5-flash", tier3_enabled=False)
    verdict = sb.verify("path/to/forged.png", src_face_path="path/to/src.png",
                        attack_family="frontal_swap")
    # verdict["sandbox_pass"] = True/False
    # verdict["tier1"] / verdict["tier2"] / verdict["tier3"] 详细数据

成本估算 (W1-W3, gemini-2.5-flash only):
    单次 verify ≈ $0.002 (Tier-2 一次 call)
    一 round 32 brief × 8 rollout = 256 verify ≈ $0.5
    100 round 总计 ≈ $50

W6+ 升级到 gemini-3-pro + Tier-3 claude:
    单次 verify ≈ $0.008 (含 20% Tier-3 抽样)
    一 round ≈ $2
    总预算仍 < $300
"""
from __future__ import annotations
import os
import sys
import json
import time
import hashlib
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Union, Literal

import numpy as np
from PIL import Image

# 同目录的 viviai_client
from viviai_client import ViviClient


# ────────────────────────── Tier-1 metrics ─────────────────────────────

# Real-lib globals; lazily initialized (cost = 1-time GPU/CPU model load)
_FACE_APP = None       # insightface FaceAnalysis (ArcFace embedding)
_LPIPS_NET = None      # lpips.LPIPS(net='alex') torch model
_MP_FACE_MESH = None   # mediapipe FaceMesh
_REAL_LIBS_TRIED = False
_REAL_LIBS_OK = {
    "insightface": False, "skimage": False, "lpips": False,
    "piq": False, "mediapipe": False,
}


def _try_init_real_libs():
    """Lazy 1-shot: try to import + warm-load real metric libs.
    Sets _REAL_LIBS_OK flags; missing libs fall back to placeholder."""
    global _REAL_LIBS_TRIED, _FACE_APP, _LPIPS_NET, _MP_FACE_MESH
    if _REAL_LIBS_TRIED:
        return
    _REAL_LIBS_TRIED = True
    try:
        from insightface.app import FaceAnalysis
        _FACE_APP = FaceAnalysis(name='buffalo_l',
                                  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        _FACE_APP.prepare(ctx_id=0, det_size=(640, 640))
        _REAL_LIBS_OK["insightface"] = True
    except Exception as e:
        logging.getLogger(__name__).info(f"[tier1] insightface unavailable: {e}")
    try:
        import skimage  # noqa
        _REAL_LIBS_OK["skimage"] = True
    except Exception:
        pass
    try:
        import lpips, torch
        _LPIPS_NET = lpips.LPIPS(net='alex', verbose=False)
        if torch.cuda.is_available():
            _LPIPS_NET = _LPIPS_NET.cuda()
        _LPIPS_NET.eval()
        _REAL_LIBS_OK["lpips"] = True
    except Exception as e:
        logging.getLogger(__name__).info(f"[tier1] lpips unavailable: {e}")
    try:
        import piq  # noqa
        _REAL_LIBS_OK["piq"] = True
    except Exception:
        pass
    # mediapipe 0.10+ removed the legacy `solutions` API. Use insightface's
    # 2d106 landmarks (already loaded via FaceAnalysis above) instead.
    if _REAL_LIBS_OK["insightface"]:
        _REAL_LIBS_OK["mediapipe"] = True  # landmark path = insightface 2d106
        logging.getLogger(__name__).info("[tier1] landmark path: insightface buffalo_l 2d106")


def _load_image(path: Union[str, Path]) -> np.ndarray:
    """Load as RGB uint8 numpy."""
    img = Image.open(str(path)).convert("RGB")
    return np.array(img)


def _ssim_real(a: np.ndarray, b: np.ndarray) -> float:
    """skimage full-spectrum SSIM. Returns [0, 1]."""
    from skimage.metrics import structural_similarity as ssim_fn
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = np.array(Image.fromarray(a).resize((w, h)))
        b = np.array(Image.fromarray(b).resize((w, h)))
    return float(ssim_fn(a, b, channel_axis=-1, data_range=255))


def _ssim_quick(a: np.ndarray, b: np.ndarray) -> float:
    """Cheap luminance-correlation SSIM fallback (when skimage missing)."""
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = np.array(Image.fromarray(a).resize((w, h)))
        b = np.array(Image.fromarray(b).resize((w, h)))
    a_l = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float64)
    b_l = (0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]).astype(np.float64)
    mu_a, mu_b = a_l.mean(), b_l.mean()
    var_a, var_b = a_l.var(), b_l.var()
    cov = ((a_l - mu_a) * (b_l - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    )
    return float(np.clip(ssim, 0.0, 1.0))


def _fft_artifact_score(img: np.ndarray) -> float:
    """High-frequency energy ratio. Deepfakes often have suppressed HF.
    Returns scalar in [0, 1] — higher = more HF (more "natural").
    """
    if img.ndim == 3:
        gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
    else:
        gray = img.astype(np.float64)
    F = np.fft.fft2(gray)
    F = np.fft.fftshift(F)
    mag = np.log1p(np.abs(F))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r_low = min(h, w) // 8
    r_high = min(h, w) // 3
    # mask
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low_mask = dist < r_low
    high_mask = (dist > r_high)
    low_energy = mag[low_mask].mean() + 1e-8
    high_energy = mag[high_mask].mean() + 1e-8
    ratio = high_energy / (low_energy + high_energy)
    return float(np.clip(ratio, 0.0, 1.0))


def _niqe_real(img: np.ndarray) -> float:
    """piq NIQE (no-reference, lower = better, real OUT-of-OG image stat).
    Returns float, typical face range 3-15."""
    import torch
    from piq import brisque  # NIQE not in all piq builds; BRISQUE is equiv NR metric
    t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        score = brisque(t, data_range=1.0, reduction='none')
    return float(score.item())


def _niqe_quick(img: np.ndarray) -> float:
    """Cheap laplacian-variance NIQE fallback."""
    if img.ndim == 3:
        gray = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
    else:
        gray = img.astype(np.float64)
    pad = np.pad(gray, 1, mode="edge")
    lap = (
        pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] - 4 * pad[1:-1, 1:-1]
    )
    sharpness = lap.var()
    niqe = 12.0 - np.log1p(sharpness) * 1.5
    return float(np.clip(niqe, 0.0, 20.0))


def _lpips_real(a: np.ndarray, b: np.ndarray) -> float:
    """LPIPS perceptual distance via AlexNet (Zhang 2018). Higher = more dissimilar."""
    import torch
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = np.array(Image.fromarray(a).resize((w, h)))
        b = np.array(Image.fromarray(b).resize((w, h)))
    # to torch [-1, 1] (lpips convention)
    ta = torch.from_numpy(a).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    tb = torch.from_numpy(b).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    if next(_LPIPS_NET.parameters()).is_cuda:
        ta, tb = ta.cuda(), tb.cuda()
    with torch.no_grad():
        d = _LPIPS_NET(ta, tb)
    return float(d.item())


def _maniqa_real(img: np.ndarray) -> float:
    """piq MANIQA-like proxy via TV+sharpness; real MANIQA needs a checkpoint not bundled.
    Use piq.total_variation as a cheap stand-in for now."""
    import torch
    from piq import total_variation
    t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        tv = total_variation(t, reduction='mean', norm_type='l2')
    return float(tv.item())


def _arcface_id_sim_real(img: np.ndarray, src_img: np.ndarray) -> float:
    """insightface buffalo_l ArcFace embedding cosine. Returns [-1, 1], 1=same id."""
    import cv2
    # insightface expects BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    src_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
    faces_a = _FACE_APP.get(img_bgr)
    faces_b = _FACE_APP.get(src_bgr)
    if not faces_a or not faces_b:
        return -1.0
    ea = faces_a[0].normed_embedding
    eb = faces_b[0].normed_embedding
    return float(np.dot(ea, eb))


def _arcface_id_sim_placeholder(img: np.ndarray, src_img: Optional[np.ndarray]) -> float:
    """Placeholder when insightface missing: naive grayscale L2."""
    if src_img is None:
        return -1.0
    h = w = 64
    a = np.array(Image.fromarray(img).resize((w, h)).convert("L") if img.ndim == 3
                 else Image.fromarray(img).resize((w, h))).astype(np.float64) / 255.0
    b = np.array(Image.fromarray(src_img).resize((w, h)).convert("L") if src_img.ndim == 3
                 else Image.fromarray(src_img).resize((w, h))).astype(np.float64) / 255.0
    if a.ndim == 3: a = a.mean(axis=-1)
    if b.ndim == 3: b = b.mean(axis=-1)
    l2 = float(np.sqrt(((a - b) ** 2).mean()))
    return float(1.0 - np.clip(l2, 0.0, 1.0))


def _landmark_consistency_real(img: np.ndarray, src_img: Optional[np.ndarray]) -> float:
    """insightface 2d106 landmark symmetry. Returns [0,1]: 1=symmetric (real-like), 0=asymmetric.
    Fake-swap chains often distort eye/mouth corners → drops symmetry."""
    if _FACE_APP is None:
        return -1.0
    import cv2
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    faces = _FACE_APP.get(img_bgr)
    if not faces:
        return 0.0
    f = faces[0]
    lm = getattr(f, 'landmark_2d_106', None)
    if lm is None or len(lm) < 106:
        return -1.0
    # face center x = bbox midpoint
    bbox = f.bbox  # [x1, y1, x2, y2]
    cx = (bbox[0] + bbox[2]) / 2.0
    fw = bbox[2] - bbox[0] + 1e-6
    # mirror landmarks across cx
    lm_x = lm[:, 0]
    mirror_x = 2 * cx - lm_x
    # for each landmark, find nearest mirror counterpart (lazy: same-index L2)
    err = float(np.abs(lm_x - mirror_x).mean() / fw)
    return float(np.clip(1.0 - err, 0.0, 1.0))


def tier1_function_checks(
    forged_path: Union[str, Path],
    src_face_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Layer 5 Tier-1: deterministic function metrics. Real libs when available, fallback otherwise.

    Returns dict with 7 metrics + `metrics_source` indicating which path ran each.
    """
    _try_init_real_libs()
    img = _load_image(forged_path)
    src = _load_image(src_face_path) if src_face_path else None

    out = {"fft_artifact_score": _fft_artifact_score(img)}
    src_used = {}

    # ArcFace
    if _REAL_LIBS_OK["insightface"] and src is not None:
        try:
            out["arcface_id_sim"] = _arcface_id_sim_real(img, src)
            src_used["arcface_id_sim"] = "insightface_buffalo_l"
        except Exception as e:
            out["arcface_id_sim"] = _arcface_id_sim_placeholder(img, src)
            src_used["arcface_id_sim"] = f"placeholder({type(e).__name__})"
    else:
        out["arcface_id_sim"] = _arcface_id_sim_placeholder(img, src)
        src_used["arcface_id_sim"] = "placeholder"

    # SSIM
    if src is not None:
        if _REAL_LIBS_OK["skimage"]:
            try:
                out["ssim_vs_src"] = _ssim_real(img, src)
                src_used["ssim_vs_src"] = "skimage"
            except Exception:
                out["ssim_vs_src"] = _ssim_quick(img, src)
                src_used["ssim_vs_src"] = "fallback"
        else:
            out["ssim_vs_src"] = _ssim_quick(img, src)
            src_used["ssim_vs_src"] = "luminance_proxy"
    else:
        out["ssim_vs_src"] = -1.0

    # LPIPS
    if _REAL_LIBS_OK["lpips"] and src is not None:
        try:
            out["lpips_vs_src"] = _lpips_real(img, src)
            src_used["lpips_vs_src"] = "lpips_alex"
        except Exception as e:
            out["lpips_vs_src"] = -1.0
            src_used["lpips_vs_src"] = f"err({type(e).__name__})"
    else:
        out["lpips_vs_src"] = -1.0

    # NIQE (real = BRISQUE NR)
    if _REAL_LIBS_OK["piq"]:
        try:
            out["niqe"] = _niqe_real(img)
            src_used["niqe"] = "piq_brisque"
            out["maniqa"] = _maniqa_real(img)
            src_used["maniqa"] = "piq_tv_proxy"
        except Exception as e:
            out["niqe"] = _niqe_quick(img)
            out["maniqa"] = -1.0
            src_used["niqe"] = f"fallback({type(e).__name__})"
    else:
        out["niqe"] = _niqe_quick(img)
        out["maniqa"] = -1.0
        src_used["niqe"] = "laplacian_proxy"

    # landmark (uses insightface 2d106; old name "mediapipe" kept for backwards compat)
    if _REAL_LIBS_OK["mediapipe"]:
        try:
            out["landmark_consistency"] = _landmark_consistency_real(img, src)
            src_used["landmark_consistency"] = "insightface_2d106"
        except Exception:
            out["landmark_consistency"] = -1.0
            src_used["landmark_consistency"] = "err"
    else:
        out["landmark_consistency"] = -1.0
        src_used["landmark_consistency"] = "missing"

    out["metrics_source"] = src_used
    return out


# ────────────────────────── Tier-2: LLM-as-Judge ──────────────────────

_TIER2_PROMPT = """You are a forensic analyst inspecting a face image submitted to a KYC system.
Your job is to judge if the image is a real human face captured live, or a forgery (deepfake / face-swap / morph / replay / 3D mask / synthesized).

Return STRICTLY this JSON, no extra text:
{{
  "is_fake": true | false,
  "confidence": 0.0-1.0,  // your confidence that the image IS fake
  "attack_family_guess": "frontal_swap" | "profile_swap" | "id_diff" | "reenact" | "morph" | "3d_mask" | "replay" | "adv_patch" | "audio_synth" | "unknown",
  "reasoning": "one paragraph forensic argument (cite visible artifacts, lighting, texture, frequency cues)"
}}"""


def tier2_llm_judge(
    forged_path: Union[str, Path],
    client: ViviClient,
    model: str = "gemini-2.5-flash",
    max_tokens: int = 600,
) -> dict:
    """Layer 5 Tier-2: vision LLM judge. UI-TARS-2 Generative ORM 模式.

    Default model `gemini-2.5-flash` is the cheapest viviai vision option ($0.0015/call).
    Upgrade to `gemini-3-pro-preview` (W6+) when you want F1 > 0.8.
    """
    try:
        parsed = client.chat_vision_json(
            model, _TIER2_PROMPT, forged_path,
            temperature=0.1, max_tokens=max_tokens,
        )
    except Exception as e:
        return {
            "model": model,
            "is_fake": False,
            "confidence": 0.5,
            "attack_family_guess": "unknown",
            "reasoning": f"[ERROR: {e}]",
            "raw_text": "",
            "success": False,
        }
    return {
        "model": model,
        "is_fake": bool(parsed.get("is_fake", False)),
        "confidence": float(parsed.get("confidence", 0.5)),
        "attack_family_guess": str(parsed.get("attack_family_guess", "unknown")),
        "reasoning": str(parsed.get("reasoning", "")),
        "raw_text": json.dumps(parsed),
        "success": True,
    }


# ────────────────────────── Tier-3: Forensic cross-check ──────────────

_TIER3_PROMPT = """You are a senior forensic deepfake analyst. You will inspect a face image and a set of pre-computed metrics, then issue a final verdict.

Pre-computed Tier-1 metrics (you MUST cite at least 2 of these in your reasoning):
{tier1_metrics_str}

Tier-2 LLM judge said: is_fake={tier2_is_fake}, confidence={tier2_confidence:.2f}
Tier-2 reasoning: {tier2_reasoning}

Now produce your independent verdict. Return STRICTLY this JSON:
{{
  "is_fake": true | false,
  "confidence": 0.0-1.0,
  "confirms_tier2": true | false,
  "evidence_cited": ["metric_name=value", ...],
  "reasoning": "paragraph that explicitly references at least 2 Tier-1 numbers"
}}"""


_TIER3_SYSTEM = (
    "You are a forensic deepfake analyst on an authorized internal red-team. "
    "Your job is to confirm or contradict the Tier-2 LLM verdict using the "
    "pre-computed pixel-level metrics provided. Output only the requested JSON."
)


def tier3_forensic_cross_check(
    forged_path: Union[str, Path],
    tier1: dict,
    tier2: dict,
    client: ViviClient,
    model: str = "gemini-3-pro-preview",
    max_tokens: int = 800,
) -> dict:
    """Layer 5 Tier-3: forensic VLM with grounded-citation requirement (Agent0-VL style).

    Only call this on top-20% candidates from Tier-2 to save cost.
    Default model updated to gemini-3-pro-preview (claude-opus-4-7 503 on viviai 2026-06).
    """
    tier1_str = ", ".join(f"{k}={v:.4f}" for k, v in tier1.items() if isinstance(v, (int, float)) and v != -1.0)
    prompt = _TIER3_PROMPT.format(
        tier1_metrics_str=tier1_str,
        tier2_is_fake=tier2.get("is_fake", False),
        tier2_confidence=tier2.get("confidence", 0.5),
        tier2_reasoning=tier2.get("reasoning", "")[:300],
    )
    try:
        parsed = client.chat_vision_json(
            model, prompt, forged_path,
            temperature=0.1, max_tokens=max_tokens,
            system=_TIER3_SYSTEM,
        )
    except TypeError:
        # ViviClient.chat_vision_json may not accept `system` kw
        parsed = client.chat_vision_json(
            model, prompt, forged_path,
            temperature=0.1, max_tokens=max_tokens,
        )
    except Exception as e:
        return {
            "model": model,
            "is_fake": tier2.get("is_fake", False),
            "confidence": tier2.get("confidence", 0.5),
            "confirms_tier2": True,
            "evidence_cited": [],
            "reasoning": f"[ERROR: {e}]",
            "success": False,
        }
    return {
        "model": model,
        "is_fake": bool(parsed.get("is_fake", False)),
        "confidence": float(parsed.get("confidence", 0.5)),
        "confirms_tier2": bool(parsed.get("confirms_tier2", True)),
        "evidence_cited": list(parsed.get("evidence_cited", [])),
        "reasoning": str(parsed.get("reasoning", "")),
        "raw_text": json.dumps(parsed),
        "success": True,
    }


# ────────────────────────── Bypass validity gate ─────────────────────

# Families whose threat model REQUIRES inducing a (target) identity. A bypass in
# these families must keep a recognizable face that carries a plausible identity.
IDENTITY_FAMILIES = {
    "frontal_swap", "profile_swap", "id_diff", "reenact", "morph", "3d_mask",
}
MIN_ARCFACE_IDENTITY = 0.30  # below this the "face" carries no usable identity


def _face_type(tier1: dict, attack_family: Optional[str]) -> str:
    """Label a forgery by what kind of face it carries — never rejects it.

    Two-layer open-ended redesign (2026-06-29): realism is no longer a hard gate.
    OOD / cross-species / faceless samples that fool the frozen detector are
    legitimate failure modes worth keeping (cf. Hendrycks Natural Adversarial
    Examples). So instead of a boolean accept/reject we emit a descriptor label
    that the MAP-Elites archive records; the inner loop gates only on detector
    fitness, not on face validity.

    Returns one of:
      "non_face"    : no detectable face geometry (insightface sentinels)
      "low_id_face" : a face is present but carries little/no target identity
                      (random / garbage / cross-species) — only meaningful for
                      identity-driven families
      "face"        : a face with a plausible identity
    """
    if not tier1:
        return "non_face"
    arc = tier1.get("arcface_id_sim", -1.0)
    lm = tier1.get("landmark_consistency", -1.0)
    # Face presence is decided by landmark geometry only. arc == -1.0 means the
    # ArcFace identity cosine could not be COMPUTED (no source reference passed,
    # e.g. pure text-to-image synthesis), which is NOT the same as "no face" — a
    # valid landmark proves a face is present. Gating non_face on arc here wrongly
    # labelled every src-less synthesis as faceless.
    if lm is None or lm <= 0.0:   # insightface found no face geometry
        return "non_face"
    if attack_family in IDENTITY_FAMILIES and (arc == -1.0 or arc < MIN_ARCFACE_IDENTITY):
        return "low_id_face"
    return "face"


def _augment_view(img_bgr: "np.ndarray", rng) -> "np.ndarray":
    """One LIGHT perturbation of a BGR image for the MC-augmentation graded proxy.

    Deliberately mild (small JPEG/brightness/blur/noise) — the point is to probe how
    *robustly* the detector calls the image fake, not to evade it. A candidate sitting
    near the detector's decision boundary flips on more of these views than one deep in
    fake-space, so the FRACTION-still-fake is a graded [0,1] signal even when each single
    verdict is near-binary. (Ilyas et al. 2018, label-only black-box, arXiv:1804.08598.)"""
    import cv2
    out = img_bgr
    q = int(rng.integers(70, 96))
    ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if ok:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is not None:
            out = dec
    b = 1.0 + float(rng.uniform(-0.08, 0.08))
    out = np.clip(out.astype(np.float32) * b, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if rng.random() < 0.5:
        noise = rng.normal(0.0, 2.0, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


# ────────────────────────── Orchestrator ──────────────────────────────

@dataclass
class SandboxVerdict:
    """Layer 5 完整输出, 直接序列化进 trajectory.verdicts."""
    sandbox_pass: bool
    bypass_confirmed_by: list = field(default_factory=list)
    tier1: dict = field(default_factory=dict)
    tier2: dict = field(default_factory=dict)
    tier3: Optional[dict] = None
    cost_usd: float = 0.0
    detector_signature: str = ""
    timestamp: float = 0.0
    face_type: str = ""          # {face, low_id_face, non_face} — label, not a gate


class SandboxVerifier:
    """3-tier sandbox 入口. 每个 rollout 完成后调用 .verify().

    ★ Lv5 切换: tier2_backend 决定 Tier-2 detector:
      - 'viviai': gemini-2.5-flash / gemini-3-pro via viviai (default, 不需 GPU)
      - 'fakevlm_local': 本地 FakeVLM via vLLM (Lv5 用, 需 GPU)
    """

    # 当前定价 (viviai 2026-06), 用于成本估算
    _COST = {
        "gemini-2.5-flash": 0.0015,
        "gemini-3-pro-preview": 0.005,
        "gemini-3.5-flash": 0.002,
        "claude-opus-4-7": 0.015,
        "claude-sonnet-4-6": 0.005,
        "fakevlm_local": 0.0,  # 本地推理无 API 成本 (只算 GPU 时间)
    }

    def __init__(
        self,
        client: Optional[ViviClient] = None,
        tier2_model: str = "gemini-2.5-flash",
        tier3_model: str = "claude-opus-4-7",
        tier3_enabled: bool = False,
        tier3_top_quantile: float = 0.2,
        confidence_threshold: float = 0.5,
        tier2_backend: str = "viviai",          # ★ Lv5 switch
        fakevlm_ckpt_path: Optional[str] = None,
        fakevlm_endpoint: str = "http://localhost:8001/v1",
        graded_mc_n: int = 0,                    # MC-augmentation graded proxy views
        graded_mc_seed: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        self.client = client or ViviClient()
        self.tier2_model = tier2_model
        self.tier3_model = tier3_model
        self.tier3_enabled = tier3_enabled
        self.tier3_top_quantile = tier3_top_quantile
        self.confidence_threshold = confidence_threshold
        self.tier2_backend = tier2_backend
        self.graded_mc_n = int(graded_mc_n)
        self.graded_mc_seed = int(graded_mc_seed)
        self.logger = logger or logging.getLogger(__name__)

        # ★ Lv5: 懒加载 FakeVLM judge (只在用时 import)
        self._fakevlm_judge = None
        if tier2_backend == "fakevlm_local":
            try:
                from fakevlm_judge_real import FakeVLMJudge, FakeVLMJudgeConfig
                cfg = FakeVLMJudgeConfig(
                    # default to the VALIDATED faithful ckpt, never the broken multi_ retrain
                    ckpt_path=fakevlm_ckpt_path or
                        "/data/disk4/lyx_ICML/self_evolution_forgery/scripts/fakevlm_correct_ckpt",
                    vllm_endpoint=fakevlm_endpoint,
                )
                self._fakevlm_judge = FakeVLMJudge(cfg)
                if not self._fakevlm_judge.is_server_up():
                    if os.environ.get("ALLOW_VIVIAI_FALLBACK") == "1":
                        self.logger.warning(
                            f"FakeVLM vLLM server not up at {fakevlm_endpoint} — "
                            f"ALLOW_VIVIAI_FALLBACK=1 set, falling back to viviai per call (NON-PAPER)"
                        )
                    else:
                        raise RuntimeError(
                        f"tier2_backend=fakevlm_local but FakeVLM vLLM server is NOT up at "
                        f"{fakevlm_endpoint}. Refusing to silently fall back to viviai — that "
                        f"would invalidate paper-grade bypass numbers. Start the server "
                        f"(scripts/fakevlm_raw_server.py --port 8001) and retry. "
                        f"Set ALLOW_VIVIAI_FALLBACK=1 to opt into the (non-paper) fallback."
                    )
            except ImportError:
                self.logger.warning("FakeVLMJudge import failed; using viviai")

    @property
    def detector_signature(self) -> str:
        if (self.tier2_backend == "fakevlm_local" and self._fakevlm_judge is not None):
            # Encode the ACTUAL ckpt + endpoint port so runs against different
            # checkpoints are distinguishable. The old bare `fakevlm_local` label
            # collapsed the broken multi_ ckpt and the faithful ckpt into one
            # signature, which is what let the wrong-ckpt contamination hide.
            from pathlib import Path as _P
            cfg = self._fakevlm_judge.cfg
            ckpt = _P(cfg.ckpt_path).name
            port = cfg.vllm_endpoint.rstrip("/").rsplit(":", 1)[-1].split("/")[0]
            tier2_label = f"fakevlm_local[{ckpt}@{port}]"
        else:
            tier2_label = self.tier2_model
        sig = f"tier1_func+tier2_{tier2_label}"
        if self.graded_mc_n > 0:
            sig += f"+gradedMC{self.graded_mc_n}"
        if self.tier3_enabled:
            sig += f"+tier3_{self.tier3_model}"
        return sig

    def _judge_image(self, path: Union[str, Path]) -> dict:
        """Run the configured Tier-2 backend on one image path. Tags the dict with
        `_backend` so the caller can attribute cost. The single source of the tier2
        verdict, reused by both verify() and the MC-augmentation graded proxy."""
        if (self.tier2_backend == "fakevlm_local" and self._fakevlm_judge is not None
                and self._fakevlm_judge.is_server_up()):
            t2 = self._fakevlm_judge.judge(path)
            t2["_backend"] = "fakevlm_local"
        else:
            t2 = tier2_llm_judge(path, self.client, model=self.tier2_model)
            t2["_backend"] = self.tier2_model
        return t2

    def _augmented_real_prob(self, forged_path: Union[str, Path], n_aug: int) -> dict:
        """MC-augmentation graded proxy. Judge `n_aug` light perturbations of the image;
        return mean P(real) and the fraction judged real — a graded [0,1] search signal
        that de-saturates a near-binary detector (see _augment_view)."""
        import cv2, tempfile, os
        img = cv2.imread(str(forged_path))
        if img is None:
            return {"graded_real_prob": 0.0, "graded_frac_real": 0.0, "graded_n_aug": 0}
        rng = np.random.default_rng(self.graded_mc_seed ^ (hash(str(forged_path)) & 0xffffffff))
        probs: list = []
        n_real = 0
        with tempfile.TemporaryDirectory() as td:
            for i in range(n_aug):
                aug = _augment_view(img, rng)
                ap = os.path.join(td, f"aug{i}.png")
                cv2.imwrite(ap, aug)
                t2 = self._judge_image(ap)
                if not t2.get("success", False):
                    continue
                conf = float(t2.get("confidence", 0.5))
                pr = conf if not t2.get("is_fake", False) else (1.0 - conf)
                probs.append(pr)
                if not t2.get("is_fake", False):
                    n_real += 1
        if not probs:
            return {"graded_real_prob": 0.0, "graded_frac_real": 0.0, "graded_n_aug": 0}
        return {"graded_real_prob": float(np.mean(probs)),
                "graded_frac_real": float(n_real / len(probs)),
                "graded_n_aug": len(probs)}

    def verify(
        self,
        forged_path: Union[str, Path],
        src_face_path: Optional[Union[str, Path]] = None,
        attack_family: Optional[str] = None,
        force_tier3: bool = False,
    ) -> SandboxVerdict:
        """Run full sandbox. Returns SandboxVerdict.

        force_tier3=True 跳过 quantile gate, 强制跑 Tier-3 (debug 用).
        """
        t0 = time.time()
        cost = 0.0

        # Tier-1 (cheap, always)
        tier1 = tier1_function_checks(forged_path, src_face_path)

        # ★ Tier-2 (LLM-judge, switchable backend)
        tier2 = self._judge_image(forged_path)
        cost += (self._COST.get("fakevlm_local", 0.0)
                 if tier2.get("_backend") == "fakevlm_local"
                 else self._COST.get(self.tier2_model, 0.005))

        # ★ MC-augmentation graded proxy (SEARCH signal): re-judge N light perturbations
        # so a near-binary detector yields a graded [0,1] real-prob the evolutionary inner
        # loop can climb. The single-shot verdict below still defines the binary bypass
        # (the citable EVAL metric); this only enriches tier2 with graded_real_prob.
        if self.graded_mc_n > 0:
            g = self._augmented_real_prob(forged_path, self.graded_mc_n)
            tier2.update(g)
            cost += g.get("graded_n_aug", 0) * (
                self._COST.get("fakevlm_local", 0.0)
                if tier2.get("_backend") == "fakevlm_local"
                else self._COST.get(self.tier2_model, 0.005))

        # Tier-3 gate
        tier3 = None
        run_tier3 = self.tier3_enabled and (
            force_tier3
            # 简化版 top-quantile gate: 只对 Tier-2 confidence 在边界附近的样本跑
            # (i.e. 真正模糊的 case)
            or abs(tier2["confidence"] - self.confidence_threshold) < 0.2
        )
        if run_tier3:
            tier3 = tier3_forensic_cross_check(
                forged_path, tier1, tier2, self.client, model=self.tier3_model
            )
            cost += self._COST.get(self.tier3_model, 0.015)

        # Sandbox pass logic
        # ★ Q17 修: tier2 调用失败时绝不能 silent bypass=True. 必须 success=True 才能判 bypass.
        tier2_clear_fake = tier2.get("is_fake", False) and tier2.get("confidence", 0.0) >= self.confidence_threshold
        tier2_says_real = (not tier2_clear_fake) and tier2.get("success", False)
        confirmed_by = []
        if tier2_says_real:
            confirmed_by.append("tier2")
        if tier3 is not None and tier3.get("success"):
            tier3_clear_fake = tier3.get("is_fake", False) and tier3.get("confidence", 0.0) >= self.confidence_threshold
            tier3_says_real = not tier3_clear_fake
            if tier3_says_real:
                confirmed_by.append("tier3")
        # 如果 Tier-3 disabled, Tier-2 单独决定; 启用时, 必须 Tier-2+Tier-3 都说 real
        if self.tier3_enabled and tier3 is not None:
            detector_fooled = ("tier2" in confirmed_by) and ("tier3" in confirmed_by)
        else:
            detector_fooled = "tier2" in confirmed_by

        # ★ 2026-06-29 two-layer redesign: realism is no longer a hard gate. We
        # label the face type for the archive but a bypass is now decided purely by
        # whether the frozen detector was fooled — OOD / cross-species / faceless
        # samples that fool it are legitimate failure modes worth keeping (cf.
        # Hendrycks Natural Adversarial Examples). The old _face_valid AND-gate
        # (Bug-4b) is replaced by this label so we stop discarding such samples.
        face_type = _face_type(tier1, attack_family)
        sandbox_pass = detector_fooled

        return SandboxVerdict(
            sandbox_pass=sandbox_pass,
            bypass_confirmed_by=confirmed_by,
            tier1=tier1,
            tier2=tier2,
            tier3=tier3,
            cost_usd=round(cost, 5),
            detector_signature=self.detector_signature,
            timestamp=t0,
            face_type=face_type,
        )

    def verify_to_dict(self, *args, **kwargs) -> dict:
        """Same as verify() but returns plain dict for jsonl write."""
        return asdict(self.verify(*args, **kwargs))


# ────────────────────────── Smoke test ───────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("sandbox_smoke")

    api_key = os.environ["VIVIAI_KEY"]  # set VIVIAI_KEY env var; key not committed
    client = ViviClient(api_key=api_key)

    # 找一张样本图测试
    candidates = [
        "/data/disk4/lyx_ICML/hf_models_lyx/04_id_preserving/InstantX__InstantID/examples/0.png",
        "/data/disk4/lyx_ICML/hf_models_lyx/04_id_preserving/InstantX__InstantID/examples/yann-lecun_resize.jpg",
    ]
    sample = next((c for c in candidates if Path(c).exists()), None)
    if not sample:
        log.error("No sample image found. Provide one path on CLI: python sandbox.py <image.png>")
        if len(sys.argv) > 1:
            sample = sys.argv[1]
        else:
            sys.exit(1)
    log.info(f"Using sample: {sample}")

    # --- W1-W5 配置: 仅 Tier-1 + Tier-2 (cheap) ---
    sb_cheap = SandboxVerifier(
        client=client,
        tier2_model="gemini-2.5-flash",
        tier3_enabled=False,
    )
    log.info("=" * 60)
    log.info(f"W1-W5 配置 (cheap): {sb_cheap.detector_signature}")
    v_cheap = sb_cheap.verify(sample, src_face_path=sample)  # 自己跟自己比 → ID-sim=1
    log.info(f"  sandbox_pass = {v_cheap.sandbox_pass}")
    log.info(f"  tier1 = {json.dumps(v_cheap.tier1, indent=2)}")
    log.info(f"  tier2 = {v_cheap.tier2.get('model')}: is_fake={v_cheap.tier2.get('is_fake')}, "
             f"conf={v_cheap.tier2.get('confidence'):.2f}")
    log.info(f"  tier2 reasoning preview: {v_cheap.tier2.get('reasoning', '')[:150]}")
    log.info(f"  cost = ${v_cheap.cost_usd:.4f}")

    # --- W6+ 配置: 完整 3-tier (gemini-3-pro + claude cross-check) ---
    # 注释掉以节约 smoke test 成本; 解开来跑 W6 验证
    # sb_full = SandboxVerifier(
    #     client=client,
    #     tier2_model="gemini-3-pro-preview",
    #     tier3_enabled=True,
    #     tier3_model="claude-opus-4-7",
    # )
    # log.info("=" * 60)
    # log.info(f"W6+ 配置 (full): {sb_full.detector_signature}")
    # v_full = sb_full.verify(sample, src_face_path=sample, force_tier3=True)
    # log.info(f"  sandbox_pass = {v_full.sandbox_pass}")
    # log.info(f"  bypass_confirmed_by = {v_full.bypass_confirmed_by}")
    # log.info(f"  tier3 confirms_tier2 = {v_full.tier3.get('confirms_tier2') if v_full.tier3 else 'N/A'}")
    # log.info(f"  tier3 evidence_cited = {v_full.tier3.get('evidence_cited') if v_full.tier3 else 'N/A'}")
    # log.info(f"  cost = ${v_full.cost_usd:.4f}")

    log.info("=" * 60)
    log.info("Smoke test 完成. 接下来:")
    log.info("  1. 装 numpy + Pillow (基本上都有)")
    log.info("  2. W2 升级 Tier-1: pip install insightface lpips piq mediapipe")
    log.info("  3. W2 加 attack operator wrapper (operators/ 目录)")
    log.info("  4. W3 多 agent benchmark gen 接入这个 sandbox")
