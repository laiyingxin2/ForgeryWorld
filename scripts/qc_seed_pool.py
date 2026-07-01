"""QC the SCUT Asian seed pool for the inner MAP-Elites loop.

Keep an image only if insightface (buffalo_l, the same detector the sandbox uses)
finds exactly ONE face with a solid detection score and a usable face-box size.
A loose / multi-face / tiny-face seed poisons every downstream swap+reenact step
(the operators all index faces[0]), so we filter once, up front, and persist the
clean list. The inner loop reads this list instead of globbing the raw pool.

Writes:
  <pool>_clean.txt   one absolute path per line (kept images)
  <pool>_qc.json     full per-image verdict + summary stats
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="/data/disk4/lyx_ICML/self_evolution_forgery/data/pool_scut_asian")
    ap.add_argument("--min-det", type=float, default=0.60)
    ap.add_argument("--min-face-frac", type=float, default=0.06,
                    help="min face-box area as fraction of image area")
    ap.add_argument("--max-faces", type=int, default=1,
                    help="reject images with more detected faces than this")
    ap.add_argument("--ctx", type=int, default=0, help="insightface ctx_id (0=GPU, -1=CPU)")
    a = ap.parse_args()

    pool = Path(a.pool).resolve()
    imgs = sorted(p for p in pool.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not imgs:
        print(f"no images under {pool}", file=sys.stderr); sys.exit(1)

    from insightface.app import FaceAnalysis
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if a.ctx >= 0 else ['CPUExecutionProvider']
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=a.ctx, det_size=(640, 640))

    kept, rejected, records = [], [], []
    t0 = time.time()
    for i, p in enumerate(imgs):
        bgr = cv2.imread(str(p))
        if bgr is None:
            rejected.append(p); records.append({"path": str(p), "ok": False, "reason": "unreadable"})
            continue
        H, W = bgr.shape[:2]
        faces = app.get(bgr)
        n = len(faces)
        if n == 0:
            reason = "no_face"
        elif n > a.max_faces:
            reason = f"multi_face({n})"
        else:
            f = faces[0]
            det = float(f.det_score)
            x1, y1, x2, y2 = f.bbox.tolist()
            frac = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / float(W * H)
            if det < a.min_det:
                reason = f"low_det({det:.2f})"
            elif frac < a.min_face_frac:
                reason = f"small_face({frac:.3f})"
            else:
                reason = None
        if reason is None:
            kept.append(p)
            records.append({"path": str(p), "ok": True, "det": round(det, 3), "frac": round(frac, 3)})
        else:
            rejected.append(p)
            records.append({"path": str(p), "ok": False, "reason": reason})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(imgs)} kept={len(kept)} ({time.time()-t0:.0f}s)", flush=True)

    clean_txt = pool.parent / f"{pool.name}_clean.txt"
    qc_json = pool.parent / f"{pool.name}_qc.json"
    clean_txt.write_text("\n".join(str(p) for p in kept) + ("\n" if kept else ""))
    qc_json.write_text(json.dumps({
        "pool": str(pool), "total": len(imgs), "kept": len(kept), "rejected": len(rejected),
        "min_det": a.min_det, "min_face_frac": a.min_face_frac, "max_faces": a.max_faces,
        "records": records,
    }, indent=2))
    print(f"DONE total={len(imgs)} kept={len(kept)} rejected={len(rejected)} "
          f"({100*len(kept)/len(imgs):.1f}% kept) -> {clean_txt}")


if __name__ == "__main__":
    main()
