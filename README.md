# [ICLR'26] Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution

This repository contains code for reproducing the AbC qualitative experiments
from the paper **"Concept-TRAK: Understanding how diffusion models learn
concepts through concept-level attribution"**.

- [Arxiv](https://arxiv.org/abs/2507.06547)

> [Notice] This repository currently focuses on the AbC free-query qualitative
> pipeline. The default workflow generates an image from a compositional prompt
> and attributes individual target concepts inside that prompt to influential
> training examples.

---

# TL;DR

Concept-TRAK extends data attribution to the concept level: instead of asking
which training examples influenced an entire generated image, it asks which
training examples influenced a specific object, style, or concept inside the
image.

The shortest path to run the current AbC free-query reproduction is:

```bash
bash requirements.sh
conda activate concept-trak
bash scripts/abc/run_download.sh
bash scripts/abc/download_train_grad.sh
bash scripts/abc/run_qual_free.sh
```

By default, `run_qual_free.sh` runs the prompt
`Pikachu in the style of Pablo Picasso` and produces separate qualitative
attribution results for `Pikachu` and `Pablo Picasso`.

---

# Prerequisites

1. Install dependencies.

```bash
bash requirements.sh
conda activate concept-trak
```

2. Download the AbC benchmark assets and LAION subset metadata.

```bash
bash scripts/abc/run_download.sh
```

This prepares the expected data layout under:

```text
data/abc
```

> [Note] TRAK/dattri may not work reliably on Blackwell GPUs. We recommend
> using H100/A100 class GPUs when possible.

---

# Train Gradients

The AbC qualitative pipeline needs precomputed train-side projected gradients
at:

```text
experiments/abc/results/grads/attn2-dps-NFE10-norm-ddim-gs_7.5
```

## Option 1: Download precomputed train gradients

```bash
bash scripts/abc/download_train_grad.sh
```

This downloads:

```text
train_grad-0.npy
...
train_grad-15.npy
```

into the expected directory.

## Option 2: Compute train gradients locally

```bash
bash scripts/abc/run_train_grad.sh
```

The default settings match the released precomputed artifact:

```text
LAYER=attn2
F=dps
NUM_SPLIT=16
NFE=10
TRAIN_GUIDANCE_SCALE=7.5
DTYPE=fp16
BATCH_SIZE=4
NORMALIZE=1
DDIM_INVERSION=1
```

For a multi-GPU run:

```bash
GPU_IDS="0 1 2 3" bash scripts/abc/run_train_grad_norm_4gpu.sh
```

---

# Experiment

Run the default free-query qualitative attribution experiment:

```bash
bash scripts/abc/run_qual_free.sh
```

Default configuration:

```text
PROMPT="Pikachu in the style of Pablo Picasso"
TARGET_CONCEPTS="Pikachu,Pablo Picasso"
SEED=0
GPU_ID=4
```

For the default composition, the script uses the complementary prompt as the
negative prompt:

```text
target=Pikachu       -> negative_prompt="in the style of Pablo Picasso"
target=Pablo Picasso -> negative_prompt="Pikachu"
```

Results are saved under:

```text
experiments/abc/results/qual/free_query/
```

Each target concept gets its own directory, for example:

```text
pikachu-in-the-style-of-pablo-picasso-...-target_pikachu-...
pikachu-in-the-style-of-pablo-picasso-...-target_pablo-picasso-...
```

Each directory contains:

```text
query.png
query_noise.pt
query_xT.pt
query_meta.pt
concept_grad.npy
topk-lambda_*.jpg
topk-prompts-lambda_*.txt
metrics.json
```

---

# Custom Prompt

Run Concept-TRAK on another compositional prompt:

```bash
PROMPT="Pikachu in the style of graffiti art" \
TARGET_CONCEPTS="Pikachu,graffiti art" \
bash scripts/abc/run_qual_free.sh
```

If the automatically inferred negative prompts are not suitable, provide them
manually in the same order:

```bash
PROMPT="Pikachu in the style of graffiti art" \
TARGET_CONCEPTS="Pikachu,graffiti art" \
NEGATIVE_PROMPTS="in the style of graffiti art,Pikachu" \
bash scripts/abc/run_qual_free.sh
```

Useful overrides:

```bash
GPU_ID=5 bash scripts/abc/run_qual_free.sh
EPOCHS=64 bash scripts/abc/run_qual_free.sh
FORCE_RECOMPUTE_CONCEPT_GRAD=1 bash scripts/abc/run_qual_free.sh
```

If `concept_grad.npy` already exists and the saved metadata matches the current
prompt, target concept, negative prompt, and seed, `qual.py` reuses it and
skips the concept-gradient loop.

---

# Hugging Face Artifacts

Maintainers can upload the local train-gradient artifact to the configured
public Hugging Face dataset repo:

```bash
bash scripts/abc/upload_train_grad.sh
```

Users can download the same artifact with:

```bash
bash scripts/abc/download_train_grad.sh
```

---

# References

This reproduction pipeline relies on the AbC benchmark assets and LAION subset
used by the original Concept-TRAK experiments.

---

# BibTeX

```bibtex
@article{park2025concepttrak,
  title={Concept-TRAK: Understanding how diffusion models learn concepts through concept-level attribution},
  author={Park, Yonghyun and Lai, Chieh-Hsin and Hayakawa, Satoshi and Takida, Yuhta and Murata, Naoki and Liao, Wei-Hsiang and Choi, Woosung and Cheuk, Kin Wai and Koo, Junghyun and Mitsufuji, Yuki},
  journal={arXiv preprint arXiv:2507.06547},
  year={2025}
}
```
