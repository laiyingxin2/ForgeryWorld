#!/usr/bin/env python3
"""Detect, frontal-filter, and crop faces into a base src-pool of clean headshots.

Reuses the project's insightface buffalo_l (same detector as the attack operators),
so cropped faces are guaranteed detectable downstream.

Usage:
  python crop_faces.py --in 'data/asian_kyc/files/**/*.jpg' --out data/pool_asian_kyc \
      --size 512 --margin 0.4 --min-det 0.6 --max-yaw 30 --per-id 3
"""
import argparse, glob, os, sys
import numpy as np
import cv2
from insightface.app import FaceAnalysis


def yaw_from_kps(kps):
    # kps: 5x2 (l-eye, r-eye, nose, l-mouth, r-mouth). Crude yaw proxy:
    # nose horizontal position relative to eye-midpoint, normalized by eye distance.
    eye_mid = (kps[0] + kps[1]) / 2.0
    eye_dist = np.linalg.norm(kps[0] - kps[1]) + 1e-6
    off = (kps[2][0] - eye_mid[0]) / eye_dist
    return abs(off) * 90.0  # ~degrees, monotone proxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="glob of input images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--margin", type=float, default=0.4, help="crop margin around bbox")
    ap.add_argument("--min-det", type=float, default=0.6, help="min det score")
    ap.add_argument("--max-yaw", type=float, default=35.0, help="reject near-profile")
    ap.add_argument("--per-id", type=int, default=0, help="cap crops per parent dir (0=all)")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    paths = sorted(glob.glob(args.inp, recursive=True))
    print(f"[crop] {len(paths)} input images", flush=True)
    kept, per_id = 0, {}
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            continue
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        if float(f.det_score) < args.min_det:
            continue
        if f.kps is not None and yaw_from_kps(f.kps) > args.max_yaw:
            continue
        idkey = os.path.basename(os.path.dirname(p))
        if args.per_id and per_id.get(idkey, 0) >= args.per_id:
            continue
        x1, y1, x2, y2 = f.bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = max(w, h) * (1 + args.margin) / 2
        X1, Y1 = int(max(0, cx - half)), int(max(0, cy - half))
        X2, Y2 = int(min(img.shape[1], cx + half)), int(min(img.shape[0], cy + half))
        crop = img[Y1:Y2, X1:X2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)
        stem = f"{args.prefix}{idkey}_{os.path.splitext(os.path.basename(p))[0]}.png"
        cv2.imwrite(os.path.join(args.out, stem), crop)
        per_id[idkey] = per_id.get(idkey, 0) + 1
        kept += 1
    print(f"[crop] kept {kept} frontal crops -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
