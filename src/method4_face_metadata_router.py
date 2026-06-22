"""Method 4 — Face-metadata structural router (Eevee-inspired, face-forgery adapted).

User instruction (适应性 不照抄): NOT a copy of Eevee's LLM-mutated router.

Insight: Eevee's "router" is needed because text inputs are heterogeneous in
unknown ways → must learn the partition. Our inputs are face images, and we
already have a fast deterministic feature extractor (insightface buffalo_l)
that gives gender + age + pose + bbox + landmarks. Use these as STRUCTURED
cluster IDs — cheaper, interpretable, no LLM mutation needed.

Cluster space (8 buckets total per family, kept small to avoid sparsity):
  gender:      {male, female}                  → 2
  age_group:   {young<35, adult>=35}           → 2
  pose:        {frontal, profile}              → 2
                                                 = 2×2×2 = 8 clusters

For each (family, cluster) pair we maintain a Pareto-front skill snippet pool.
Total slots: 9 family × 8 cluster = 72 (vs Eevee's K=3-5 global slots), but
each slot can grow K=3 Pareto snippets, so 72 × 3 = 216 specialized prompts
discoverable. In practice most slots stay empty (cold start) and only
high-frequency clusters get filled.

Lazy-init: face_app loaded only when first called.
"""
from __future__ import annotations
import logging
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

# Lazy globals (loaded on first call)
_FACE_APP = None


def _ensure_face_app():
    global _FACE_APP
    if _FACE_APP is not None:
        return
    from insightface.app import FaceAnalysis
    _FACE_APP = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    _FACE_APP.prepare(ctx_id=0, det_size=(640, 640))


# ────────────────────────── Metadata schema ──────────────────────────

@dataclass
class FaceMetadata:
    gender: str = "unknown"           # male | female | unknown
    age: float = -1.0                 # years
    age_group: str = "unknown"        # young | adult | unknown
    pose: str = "unknown"             # frontal | profile | unknown
    yaw_deg: float = 0.0              # head yaw in degrees
    has_face: bool = False
    cluster_id: str = "unknown"       # composed bucket id

    def is_cold(self) -> bool:
        return not self.has_face or self.cluster_id == "unknown"


# ────────────────────────── Cluster bucket logic ──────────────────────

_AGE_THRESHOLD = 35     # adult vs young split
_POSE_YAW_THRESH = 25   # degrees, profile when |yaw| > threshold


def _pose_from_landmarks_or_bbox(face) -> tuple[str, float]:
    """Approximate yaw from 2d106 landmark asymmetry, fallback to bbox aspect.
    Returns (pose_str, yaw_degrees)."""
    # insightface buffalo_l provides .pose = [yaw, pitch, roll] in recent versions
    pose_arr = getattr(face, "pose", None)
    if pose_arr is not None and len(pose_arr) >= 1:
        try:
            yaw = float(pose_arr[0])
            return ("profile" if abs(yaw) > _POSE_YAW_THRESH else "frontal"), yaw
        except (TypeError, ValueError):
            pass
    # fallback: use landmark_2d_106 left/right eye-cheek distance asymmetry
    lm = getattr(face, "landmark_2d_106", None)
    bbox = face.bbox  # [x1,y1,x2,y2]
    fw = bbox[2] - bbox[0] + 1e-6
    if lm is not None and len(lm) >= 100:
        # take nose tip vs face center as proxy
        nose = lm[80]   # nose tip index in 2d106 schema
        face_cx = (bbox[0] + bbox[2]) / 2.0
        offset_ratio = (nose[0] - face_cx) / fw
        yaw_proxy = offset_ratio * 90.0  # rough mapping
        return ("profile" if abs(yaw_proxy) > _POSE_YAW_THRESH else "frontal"), yaw_proxy
    return "frontal", 0.0


def extract_metadata(image_path: str) -> FaceMetadata:
    """Run insightface on image, build deterministic cluster ID."""
    _ensure_face_app()
    import cv2
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return FaceMetadata()
    faces = _FACE_APP.get(bgr)
    if not faces:
        return FaceMetadata()
    f = faces[0]  # primary face
    # gender: insightface gives 0 (female) / 1 (male)
    g_int = int(getattr(f, "gender", -1))
    gender = ("male" if g_int == 1 else "female" if g_int == 0 else "unknown")
    # age
    age = float(getattr(f, "age", -1))
    age_group = ("young" if 0 < age < _AGE_THRESHOLD else
                  "adult" if age >= _AGE_THRESHOLD else "unknown")
    # pose
    pose, yaw = _pose_from_landmarks_or_bbox(f)
    # cluster_id: structured + stable hash
    cluster_id = f"{gender}_{age_group}_{pose}"
    return FaceMetadata(
        gender=gender, age=age, age_group=age_group,
        pose=pose, yaw_deg=yaw, has_face=True, cluster_id=cluster_id,
    )


def all_possible_clusters() -> list[str]:
    """Enumerate all (gender, age_group, pose) combos used as keys."""
    out = []
    for g in ("male", "female", "unknown"):
        for a in ("young", "adult", "unknown"):
            for p in ("frontal", "profile", "unknown"):
                out.append(f"{g}_{a}_{p}")
    return out


# ────────────────────────── Smoke ──────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os, glob
    faces = sorted(glob.glob(
        "/data/disk4/lyx_ICML/self_evolution_forgery/data/real_faces/*.png"))
    print(f"=== face_metadata extraction on {len(faces)} real faces ===")
    for p in faces:
        meta = extract_metadata(p)
        print(f"  {os.path.basename(p):25s}  cluster={meta.cluster_id:30s}  "
              f"age={meta.age:.0f}  yaw={meta.yaw_deg:6.1f}°")
    print(f"\n  total possible clusters: {len(all_possible_clusters())} "
          f"(some unknown — cold start)")
