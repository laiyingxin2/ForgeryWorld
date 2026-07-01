"""Diffusion ID/edit worker — runs INSIDE the isolated `forgery_img` conda env.

Invoked as a subprocess by operators/local_id_diffusion.py so the heavy diffusion
deps (diffusers + insightface + SDXL/SD1.5 bases) never have to be imported in the
orchestrator's `fakevlm` env. One process = one generation (weights load lazily,
cached across the process if a method is called repeatedly via --batch, but the
default contract is one-shot).

CLI:
    python id_diffusion_worker.py --method instructpix2pix \
        --src /path/face.png --out /path/out.png --prompt "..." --seed 0

Methods (image-only forgery operators):
    instructpix2pix : instruction-guided edit of the source image (self-contained)
    ipadapter_face  : SD1.5 + IP-Adapter plus-face — identity-conditioned re-gen
    sdxl_t2i        : SDXL text-to-image entire-face synthesis (no identity input)
    instantid       : SDXL + InstantID — strong tuning-free identity preservation

Prints exactly one line on success:  OK <output_path>
Exits non-zero with a traceback on failure.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# ── local model paths (all already on disk) ──────────────────────────
ZOO = Path("/data/disk4/lyx_ICML/hf_models_lyx")
SD15_BASE = ZOO / "07_bases" / "stable-diffusion-v1-5"
SDXL_BASE = ZOO / "07_bases" / "stabilityai__stable-diffusion-xl-base-1.0"
IP2P_BASE = ZOO / "08_editing" / "timbrooks__instruct-pix2pix"
IPADAPTER_DIR = ZOO / "03_ip_adapter" / "h94__IP-Adapter"
INSTANTID_DIR = ZOO / "04_id_preserving" / "InstantX__InstantID"
SDXL_VAE_FP16 = ZOO / "07_bases" / "madebyollin__sdxl-vae-fp16-fix"  # fixes fp16 VAE overflow
ANTELOPE_ROOT = ZOO / "02_encoders" / "insight_root"  # has models/antelopev2/*.onnx

DEFAULT_PROMPT = (
    "a photorealistic close-up selfie portrait of a person, natural indoor "
    "lighting, sharp eyes, natural skin texture, shot on a phone camera"
)


def _device_dtype():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    return dev, dt


def run_instructpix2pix(src, out, prompt, seed, steps):
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline
    from PIL import Image
    dev, dt = _device_dtype()
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        str(IP2P_BASE), torch_dtype=dt, safety_checker=None)
    pipe.to(dev)
    img = Image.open(src).convert("RGB").resize((512, 512))
    g = torch.Generator(dev).manual_seed(seed)
    edit = prompt or "make it look like a natural authentic photo, fix artifacts"
    res = pipe(edit, image=img, num_inference_steps=steps or 20,
               image_guidance_scale=1.5, guidance_scale=7.0, generator=g).images[0]
    res.save(out)


def run_sdxl_t2i(src, out, prompt, seed, steps):
    import torch
    from diffusers import StableDiffusionXLPipeline
    dev, dt = _device_dtype()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(SDXL_BASE), torch_dtype=dt, variant="fp16", use_safetensors=True)
    pipe.to(dev)
    g = torch.Generator(dev).manual_seed(seed)
    res = pipe(prompt or DEFAULT_PROMPT, num_inference_steps=steps or 30,
               guidance_scale=5.0, generator=g).images[0]
    res.save(out)


def run_ipadapter_face(src, out, prompt, seed, steps):
    import torch
    from diffusers import StableDiffusionPipeline
    from PIL import Image
    dev, dt = _device_dtype()
    pipe = StableDiffusionPipeline.from_pretrained(
        str(SD15_BASE), torch_dtype=dt, safety_checker=None)
    pipe.load_ip_adapter(str(IPADAPTER_DIR), subfolder="models",
                         weight_name="ip-adapter-plus-face_sd15.safetensors")
    pipe.set_ip_adapter_scale(0.7)
    pipe.to(dev)
    face = Image.open(src).convert("RGB").resize((512, 512))
    g = torch.Generator(dev).manual_seed(seed)
    res = pipe(prompt or DEFAULT_PROMPT, ip_adapter_image=face,
               num_inference_steps=steps or 30, guidance_scale=7.0,
               generator=g).images[0]
    res.save(out)


def run_instantid(src, out, prompt, seed, steps):
    # InstantID needs its custom SDXL pipeline file + IdentityNet controlnet.
    import torch, cv2, numpy as np
    from PIL import Image
    from insightface.app import FaceAnalysis
    from diffusers.models import ControlNetModel, AutoencoderKL
    sys.path.insert(0, str(Path(__file__).parent))
    from instantid_pipeline import StableDiffusionXLInstantIDPipeline, draw_kps  # vendored

    dev, dt = _device_dtype()
    app = FaceAnalysis(name="antelopev2", root=str(ANTELOPE_ROOT),
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    controlnet = ControlNetModel.from_pretrained(
        str(INSTANTID_DIR / "ControlNetModel"), torch_dtype=dt)
    # SDXL's stock VAE overflows in fp16 → neon RGB channel-split artifacts. Swap in
    # the fp16-fixed VAE so decoding stays clean at half precision.
    vae = AutoencoderKL.from_pretrained(str(SDXL_VAE_FP16), torch_dtype=dt)
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        str(SDXL_BASE), controlnet=controlnet, vae=vae, torch_dtype=dt)
    pipe.load_ip_adapter_instantid(str(INSTANTID_DIR / "ip-adapter.bin"))
    pipe.to(dev)

    # SDXL is trained at ~1024px and emits neon-noise garbage when asked to generate
    # at a small size. Inputs are often ~320px face crops, and the pipeline inherits
    # the control-image resolution → corrupt output. Upscale the source to a 1024px
    # working canvas (kept square; faces are roughly centered) and run detection, kps
    # drawing, and generation all at that size.
    W = H = 1024
    src_pil = Image.open(src).convert("RGB").resize((W, H))
    img = cv2.cvtColor(np.array(src_pil), cv2.COLOR_RGB2BGR)
    faces = app.get(img)
    if not faces:
        raise RuntimeError("no face detected by antelopev2")
    face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))[-1]
    emb = face["embedding"]
    kps = draw_kps(src_pil, face["kps"])
    g = torch.Generator(dev).manual_seed(seed)
    res = pipe(prompt=prompt or DEFAULT_PROMPT, image_embeds=emb, image=kps,
               width=W, height=H,
               controlnet_conditioning_scale=0.8, ip_adapter_scale=0.8,
               num_inference_steps=steps or 30, guidance_scale=5.0,
               generator=g).images[0]
    res.save(out)


METHODS = {
    "instructpix2pix": run_instructpix2pix,
    "sdxl_t2i": run_sdxl_t2i,
    "ipadapter_face": run_ipadapter_face,
    "instantid": run_instantid,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    METHODS[a.method](a.src, a.out, a.prompt, a.seed, a.steps)
    if not Path(a.out).exists():
        raise RuntimeError("worker finished but output file missing")
    print(f"OK {a.out}")


if __name__ == "__main__":
    main()
