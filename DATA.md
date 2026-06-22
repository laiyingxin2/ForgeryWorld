# Data setup

None of the datasets are committed (privacy + size). Recreate the layout below
under `data/`. The face pools are the SOURCE faces the attacker forges from and the
defender's "real" supervision; the detector is FakeVLM (LLaVA-1.5-7B).

```
data/
├── real_faces/          # 10 Western CelebA-Spoof crops (smoke only)
├── scut_fbp5500/        # SCUT-FBP5500 raw (HF parquet mirror)
├── pool_scut_asian/     # 2266 cropped Asian frontal headshots  ← PRIMARY base pool
├── pool_scut_curated/   # 60 hand-curated symlinks into pool_scut_asian
├── asian_kyc/           # UniqueData asian-kyc SAMPLE (~75 imgs, ND license = smoke only)
├── pool_asian_kyc/      # 18 crops from asian_kyc
├── chinese_lips/        # BAAI Chinese-LiPS image.zip (REJECTED for base: low-res)
└── pool_chinese_lips/   # 42 crops (low-res/blurry, not used in paper runs)
```

## 1. SCUT-FBP5500  (PRIMARY paper base pool — academic license)

2000s+ Asian frontal headshots; the cleanest KYC-distribution source.

```bash
# Option A: HF parquet mirror (used to build pool_scut_asian)
pip install datasets pillow
python - <<'PY'
from datasets import load_dataset
import os
ds = load_dataset("Roronotalt/scut-fbp", split="train")   # parquet mirror
os.makedirs("data/scut_fbp5500/imgs", exist_ok=True)
for i, ex in enumerate(ds):
    if str(ex.get("race","")).lower() == "asian":          # keep Asian subset (~4000)
        ex["image"].save(f"data/scut_fbp5500/imgs/{i:05d}.png")
PY

# Option B: official release (Baidu link on the SCUT-FBP5500 GitHub)
#   https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
```

Then crop to frontal headshots with insightface buffalo_l (yaw filter):

```bash
python scripts/crop_faces.py \
    --src data/scut_fbp5500/imgs --out data/pool_scut_asian --size 512
# 4000 -> ~2266 clean frontal crops

# curated 60-image subset used by the weak-start runs (symlinks):
mkdir -p data/pool_scut_curated
ls data/pool_scut_asian/*.png | shuf -n 60 | \
  while read f; do ln -s "$(readlink -f "$f")" "data/pool_scut_curated/$(basename "$f")"; done
```

## 2. asian_kyc  (smoke only — CC-BY-NC-**ND**, derivatives technically disallowed)

```bash
pip install huggingface_hub
huggingface-cli download UniqueData/asian-kyc-photo-dataset \
    --repo-type dataset --local-dir data/asian_kyc
# public repo is only a ~75-image / 235MB SAMPLE (5 ids); full set is commercial.
python scripts/crop_faces.py --src data/asian_kyc --out data/pool_asian_kyc --size 512
```

## 3. Chinese-LiPS  (NOT used in paper runs — frames upscale to mush)

```bash
huggingface-cli download BAAI/Chinese-LiPS --repo-type dataset \
    --local-dir data/chinese_lips        # CC-BY-NC-SA; 207 speakers, talking-face video
unzip data/chinese_lips/image.zip -d data/chinese_lips/
python scripts/crop_faces.py --src data/chinese_lips --out data/pool_chinese_lips --size 512
```

## 4. real_faces (Western smoke pool, 10 imgs)

CelebA-Spoof crops; any 10 frontal real-face PNGs at `data/real_faces/*.png` work.

## Detector checkpoint (FakeVLM)

The served detector base is the published `llava-1.5-7b-fakevlm`; the faithful
checkpoint is staged at `scripts/fakevlm_correct_ckpt/` (gitignored). The weak-start
servers instead serve vanilla `llava-hf/llava-1.5-7b-hf`. Set paths via the `--base`
flag on the server / `train_defender_round.py` (see scripts/).

## License note

SCUT-FBP5500 = academic use. asian_kyc = CC-BY-NC-ND (ND blocks derivative forgeries
→ smoke only). Chinese-LiPS = CC-BY-NC-SA (derivatives allowed). Use SCUT as the
primary base for any released results.
