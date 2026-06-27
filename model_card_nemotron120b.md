---
license: other
license_name: nvidia-open-model-license
license_link: LICENSE
base_model:
- nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
tags:
- mlx
- turboquant
- quantization
- apple-silicon
- nemotron
- moe
- mamba
- hybrid
language:
- en
---

# Nemotron-3-Super-120B-A12B-tq3

TurboQuant 3-bit quantized version of [nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) using [TurboQuant-MLX](https://github.com/manjunathshiva/turboquant-mlx).

This is the first publicly available **single-laptop** quantization of NVIDIA's 120B hybrid Mamba-Transformer MoE — runnable on a 64 GB Apple Silicon MacBook with no calibration data required.

## Model Details

- **Base Model**: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 (hybrid Mamba + Sparse Attention + MoE, 120 B params total, ~12 B active per token)
- **Architecture**: 88 layers, hybrid override pattern `MEMEMEM*EMEMEMEM*...` (M = Mamba, E = MoE, * = Attention)
- **Experts**: 512 routed experts + 1 shared expert, latent MoE with `moe_latent_size = 1024`
- **Quantization**: TurboQuant 3-bit (Hadamard rotation + Lloyd-Max codebook)
- **Group size**: 64
- **Calibration data**: **none** — TurboQuant is data-free
- **Size**: ~50 GB on disk (vs ~240 GB BF16, ~4.8× smaller)
- **Runs on**: 64 GB Apple Silicon MacBook (M-series). 48 GB and below: insufficient.

## Requirements

```bash
# macOS with Apple Silicon (M1/M2/M3/M4), 64 GB unified memory minimum
pip install turboquant-mlx-full mlx-lm>=0.31.3
```

> ⚠️ `mlx-lm` must be **0.31.3 or newer** — earlier versions don't know about
> Nemotron-H's latent-MoE projections (`fc1_latent_proj` / `fc2_latent_proj`)
> or the MTP head, and loading will fail.

## Quick Start

### Download the model

```bash
huggingface-cli download manjunathshiva/Nemotron-3-Super-120B-A12B-tq3 \
    --local-dir ~/models/nemotron-3-super-120b-tq3
```

### Generate text

```bash
python -m turboquant_mlx.generate \
    --model ~/models/nemotron-3-super-120b-tq3 \
    --prompt "Why is the sky blue? Explain in simple terms." \
    --max-tokens 200 \
    --min-tokens 50 \
    --temp 0.7
```

The `--min-tokens 50` flag is recommended for Nemotron-3 Super — the model
emits a `<think>` reasoning trace before its final answer, and you want enough
tokens for both phases.

### From Python (mlx-lm)

```python
from mlx_lm import load, generate

model, tokenizer = load("manjunathshiva/Nemotron-3-Super-120B-A12B-tq3")
response = generate(
    model,
    tokenizer,
    prompt="Why is the sky blue? Explain in simple terms.",
    max_tokens=200,
    temp=0.7,
)
print(response)
```

### With KV cache compression (long contexts)

For long prompts, compress the KV cache too:

```bash
python -m turboquant_mlx.demo_kv \
    --model ~/models/nemotron-3-super-120b-tq3 \
    --prompt "Your long prompt here..." \
    --max-tokens 200 \
    --tq-bits 3
```

## Results

Measured on a 64 GB MacBook (M-series) with macOS, MLX, and `turboquant-mlx-full`.

| Configuration | Size | Fits 64 GB? | Speed |
|---|---|---|---|
| BF16 (original)              | ~240 GB | ❌ no | n/a |
| **TurboQuant 3-bit (this repo)** | **~50 GB** | ✅ yes | **~19 tok/s** |

### Long-form reasoning run

A 974-token reasoning + answer run on the prompt *"Why is the sky blue?"*:

| Metric | Value |
|---|---|
| Output tokens          | 974 |
| Speed                  | ~19 tok/s |
| Peak unified memory    | ~54 GB |
| Output quality         | Coherent reasoning, structured `<think>` trace preserved |

## How It Works

TurboQuant applies, in one shot with no calibration data:

1. **Hadamard rotation** — a reversible orthogonal transform that flattens
   weight outliers, so all values land in a narrow range that 3-bit
   quantization can represent without large error.
2. **Lloyd-Max codebook** — 8 optimal scalar values (since 3 bits = 8 levels)
   chosen to minimize total quantization error. The codebook used by this
   model is fixed and embedded in `config.json`:
   `[-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152]`
3. **Group-wise scaling** — per-group float16 scales (group size 64) preserve
   per-channel dynamic range.
4. **Latent-MoE quantization** — Nemotron-3 Super's 512 experts share a
   1024-dim latent space. Quantizing that shared space compresses every
   expert at once. This is where most of the 190 GB savings come from.

The same recipe works for any architecture without per-model calibration.

## Architecture Notes

Nemotron-3 Super is a *hybrid* model — the layer pattern alternates between:

- **M** — Mamba state-space layers (cheap for long context)
- **E** — Mixture-of-Experts (512 routed experts, latent-MoE design)
- **\*** — Sparse attention (used only where it helps)

Plus 1 MTP (multi-token-prediction) layer. There are no dense MLP layers —
all FFN compute goes through the MoE.

`turboquant-mlx-full` quantizes:

- Mamba `in_proj` / `out_proj` linears
- Attention QKV / O linears
- The **latent-MoE projections** (`fc1_latent_proj`, `fc2_latent_proj`) — the
  shared expert pantry
- The shared expert and MTP layer linears

Embeddings, layer norms, and small bias-style tensors stay in BF16 / FP16.

## License

Released under the **NVIDIA Open Model License** (same as the base model).
See https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/

## Acknowledgements

- **NVIDIA** — for releasing Nemotron-3-Super-120B-A12B-BF16 openly
- **Google Research** — for the original TurboQuant algorithm
- **Apple** — for MLX and the unified-memory architecture that makes this fit
- **`mlx-lm`** maintainers — for landing Nemotron-H + latent-MoE + MTP support
  in 0.31.3, without which this quantization would not load

## Citation

```bibtex
@article{zandieh2025turboquant,
  title         = {TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author        = {Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  year          = {2025},
  eprint        = {2504.19874},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2504.19874}
}
```

## Repository

- TurboQuant-MLX (the conversion tool): https://github.com/manjunathshiva/turboquant-mlx
- Issues / questions: https://github.com/manjunathshiva/turboquant-mlx/issues
