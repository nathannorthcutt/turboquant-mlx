---
library_name: mlx
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507/blob/main/LICENSE
pipeline_tag: text-generation
base_model: Qwen/Qwen3-235B-A22B-Instruct-2507
tags:
- mlx
- turboquant
- moe
---

# Qwen3-235B-A22B-Instruct-2507-tq3-g32

**Full 3-bit** TurboQuant quantization of [Qwen/Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) — **3-bit attention *and* 3-bit experts** (group size 32) — produced with [TurboQuant-MLX](https://github.com/manjunathshiva/turboquant-mlx).

This is the **recall-critical sibling** of the smaller [**tq3a-tq2e hybrid**](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32). The hybrid drops experts to 2-bit for a 70.5 GB footprint but, as a result, can flip the *leading digit* of an exact in-context literal (e.g. a password) toward a spurious value. Keeping experts at **3-bit** fixes that — at the cost of a larger checkpoint (**103 GB** vs 70.5 GB). Pick this build if exact factual recall matters; pick the hybrid if you want the smallest footprint.

## Model Details

- **Base model**: Qwen/Qwen3-235B-A22B-Instruct-2507 — Mixture-of-Experts (`qwen3_moe`)
- **Architecture**: 94 layers, hidden size 4096, **128 routed experts (top-8)**, MoE intermediate 1536, ~235B total / ~22B active params
- **Quantization**: **TurboQuant** (Hadamard rotation + Lloyd-Max codebook), **uniform 3-bit**, group size 32
  - Attention (`q/k/v/o_proj` + `lm_head`) → 3-bit
  - Experts (`gate/up/down_proj` of every expert) → **3-bit** (vs 2-bit in the hybrid)
  - Routers (`mlp.gate`) → full precision (auto-skipped — never quantized)
- **Size**: **103 GB** across **21** shards (vs ~470 GB BF16, and vs 70.5 GB for the 2-bit-expert hybrid)

### Why a full-3-bit build?

On the 2-bit-expert hybrid, the always-on attention path is 3-bit but the experts are 2-bit. Diagnosis (top-8 logit probing across several in-context codes) showed the 2-bit experts inject a faint token prior that overrides the *weakest* copy signal — the **leading digit** of a multi-digit literal, which has the shortest matched prefix. Trailing digits and the alphabetic parts (with longer, sharper matches) are recalled perfectly; only the first digit could collapse toward a hallucinated value. Raising the experts to **3-bit removes that prior entirely** — the leading digit is then recalled at ~1.0 confidence. This build trades the hybrid's memory savings for that fidelity.

> **Note:** this is an instruct model. Use the chat template (the streaming generator applies it automatically).

## Quality

Passes **all 6 stress probes** on a 64 GB Mac (40 GB cache, greedy decode):

| Probe | Result |
|---|---|
| Long essay | ✅ Coherent, well-structured ~400-word essay |
| Multi-step math | ✅ Correct — 24 × $7 with a 15% bulk discount → **$142.80** |
| Code generation | ✅ Correct memoized `nth_fib` (0-indexed, valid examples) |
| Strict JSON formatting | ✅ Exact array of the five largest planets, no prose |
| Repetition / degeneration | ✅ Listed 1–15 cleanly and stopped |
| **In-context needle recall** | ✅ **`RAVEN-stone-91`** — the digit the hybrid flipped to `-51` is now correct |

The headline: the hybrid's one failure mode is **fixed**. The needle that returned `RAVEN-stone-51` on the 2-bit-expert hybrid comes back as the correct **`RAVEN-stone-91`** here, with the leading digit recalled at ~0.98 confidence (it was 0.13, and lost to a spurious `5`, at 2-bit). Raising the experts from 2-bit to 3-bit removes that prior entirely.

## Running on limited RAM (expert streaming)

At **103 GB** this model fits *fully resident* only on a **128 GB+** Apple Silicon machine. On smaller Macs it runs by **streaming MoE experts from disk** — each token pages in only its router-selected experts (LRU-cached), so the big expert tensors are never all in memory at once. Output is **bit-identical** to the fully-resident model.

Because this build is larger than the hybrid, a given cache budget holds a *smaller fraction* of the experts, so streaming is slower than the 70.5 GB hybrid at the same budget:

| Machine | Cache budget | Expert hit-rate | Decode speed | Peak memory |
|---|---|---|---|---|
| M4 Mac mini, 16 GB | `--cache-budget-gb 6` | ~0% | ~0.1 tok/s | 10.1 GB |
| 64 GB Apple Silicon | `--cache-budget-gb 40` | **86.3%** | **~1.3 tok/s** (sustained) | 46.5 GB |

On the **64 GB Mac** a 40 GB cache holds ~40% of the ~100 GB of experts, but temporal locality over a sustained run lifts the hit-rate to **86.3%** (measured across the 6-probe stress). Decode is bursty — ~1.3 tok/s on long generations, dropping on short prompts (prefill-dominated) or when a task churns into a colder expert slice. That's noticeably slower than the 70.5 GB [hybrid](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32) (94.1% hit / ~4–6 tok/s at the same budget) — full 3-bit experts cost both disk and speed, and buy exact-recall fidelity in return. Peak memory was **46.5 GB**, comfortably inside 64 GB.

On the 16 GB mini the 6 GB cache holds almost none of the ~100 GB of experts, so nearly every token falls through to the SSD — it runs, but it's a correctness demo, not a usable speed. **Use a 64 GB+ Mac with `--cache-budget-gb 40-50`** (bump `sudo sysctl iogpu.wired_limit_mb=57344` past the ~48 GB default Metal wired cap), faster Thunderbolt NVMe storage, or — if exact recall is *not* critical — the smaller [**hybrid build**](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32).

Expert streaming ships in TurboQuant-MLX **0.6.0+**:

```bash
pip install "turboquant-mlx-full>=0.6.0"

python -m turboquant_mlx.stream.stream_generate \
    --model manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3-g32 \
    --prompt "Explain why the sky is blue." \
    --max-tokens 512 --cache-budget-gb 40
```

## Requirements

```bash
# macOS with Apple Silicon (M1/M2/M3/M4)
pip install turboquant-mlx-full mlx-lm
```

## Quick Start

Fully resident (128 GB+ machine), via mlx-lm:

```python
from mlx_lm import load, generate

model, tokenizer = load("manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3-g32")
response = generate(
    model, tokenizer,
    prompt="Write a Python function that returns the nth Fibonacci number.",
    max_tokens=512,
)
print(response)
```

On smaller machines, use the streaming generator shown above.

## How It Works

TurboQuant applies:
1. **Hadamard rotation** — random ±1 scaling to decorrelate weights before quantization
2. **Lloyd-Max codebook** — optimal scalar quantization via k-means
3. **Group-wise scaling** — per-group float16 scales for precision

This achieves better quality than standard affine quantization at the same bit-width. Here *every* Linear and expert projection is quantized to 3-bit — including the experts — which is what preserves exact in-context recall that the 2-bit-expert hybrid can drop.

### Reproducing the conversion

```bash
python -m turboquant_mlx.convert \
    --hf-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --mlx-path /Volumes/SSD/qwen3-235b-tq3-g32 \
    --bits 3 -g 32 --streaming
```

A uniform `--bits 3` (no `--mlp-bits` override) keeps experts at 3-bit; `--streaming` writes each quantized layer to a shard and frees it, so the full 235B converts in **~8-12 GB of RAM** (peak) — it was produced on a **16 GB Mac mini**. Point `--mlx-path` at a drive with ≥110 GB free.

## License

Apache 2.0 (same as the base model). Quantization tooling: TurboQuant-MLX.

Copyright 2026 Manjunath Janardhan.

## Citation

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  year={2025},
  eprint={2504.19874},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2504.19874}
}
```
