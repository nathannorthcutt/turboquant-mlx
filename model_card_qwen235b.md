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
- hybrid-quant
---

# Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32

**Hybrid** TurboQuant quantization of [Qwen/Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) — **3-bit attention, 2-bit experts** (group size 32) — produced with [TurboQuant-MLX](https://github.com/manjunathshiva/turboquant-mlx).

A 235B-parameter MoE compressed from ~470 GB (BF16) to **70.51 GB**, small enough to *stream* on a 16 GB Mac and run fully resident on a 96 GB+ Apple Silicon machine.

## Model Details

- **Base model**: Qwen/Qwen3-235B-A22B-Instruct-2507 — Mixture-of-Experts (`qwen3_moe`)
- **Architecture**: 94 layers, hidden size 4096, **128 routed experts (top-8)**, MoE intermediate 1536, ~235B total / ~22B active params
- **Quantization**: **Hybrid TurboQuant** (Hadamard rotation + Lloyd-Max codebook), group size 32
  - **Attention → 3-bit** (`q/k/v/o_proj` + `lm_head`): 377 Linear layers
  - **Experts → 2-bit** (`gate/up/down_proj` of every expert): 282 SwitchLinear layers
  - **Routers → full precision** (`mlp.gate` is auto-skipped — never quantized)
- **Size**: **70.51 GB** across 15 shards (vs ~470 GB BF16 — a **6.7× reduction**)

### Why hybrid (tq3a-tq2e)?

The experts dominate the parameter count, so dropping them to **2-bit** is where almost all the memory savings come from. Qwen3-235B routes **top-8 of 128 experts** per token, and that redundancy averages out 2-bit quantization noise cleanly — the same effect that lets [gpt-oss-120b](https://huggingface.co/manjunathshiva/gpt-oss-120b-tq3) (128 experts) hold up at 2-bit while [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) (32 experts) collapses into word-salad. The shared **attention** path is hit on *every* token with no expert averaging to hide errors, so it's kept at the safer **3-bit**. Routers stay full precision because a wrong expert selection is unrecoverable.

> **Note:** this is an instruct model. Use the chat template (the streaming generator applies it automatically).

## Quality

Validated with a 6-probe stress suite on a **64 GB Apple Silicon Mac** (40 GB expert cache, greedy decode). **Five of six probes pass cleanly:**

| Probe | Result |
|---|---|
| Long essay (coherence over length) | ✅ Coherent, well-structured ~400-word essay |
| Multi-step math | ✅ Correct — 24 × $7 with a 15% bulk discount → **$142.80**, every step right |
| Code generation | ✅ Correct memoized `nth_fib` with docstring + valid examples |
| Strict JSON formatting | ✅ Exact JSON array, correct five largest planets, no prose |
| Repetition / degeneration | ✅ Listed 1–15 cleanly and stopped — no looping |
| In-context needle recall | ⚠️ Recalled the password shape but flipped one digit (`RAVEN-stone-91` → `-51`) |

Multi-step arithmetic — usually the soft spot for 2-bit-expert MoEs — was **exactly right** here, confirming that 2-bit experts hold at Qwen3's 128-expert / top-8 routing (this is the first `qwen3_moe` model validated under TurboQuant expert streaming). The one blemish was fine-grained factual recall: a specific alphanumeric token from the context came back with one digit changed.

> **Caveat — exact recall.** The observed soft spot is precise factual recall (exact codes, IDs, digit strings) rather than reasoning: in stress testing one alphanumeric needle returned with a single flipped digit (`91`→`51`), while reasoning, math, code, and formatting were all correct. Probing showed the 2-bit experts inject a faint prior that only overrides the *weakest* copy signal — the **leading digit** of a multi-digit literal; trailing digits and alphabetic spans are recalled perfectly.
>
> **Need exact recall?** Use the full-3-bit sibling — [**Qwen3-235B-A22B-Instruct-2507-tq3-g32**](https://huggingface.co/manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3-g32) (3-bit experts, ~103 GB). Raising the experts to 3-bit removes that prior entirely: it recalls the same needle correctly (`RAVEN-stone-91`). Choose **this hybrid** for the smallest footprint (70.5 GB); choose the **tq3** build when exact literal recall matters.

## Running on limited RAM (expert streaming)

At **70.51 GB** the model only fits *fully resident* on a **96 GB+** Apple Silicon machine. On a **64 GB** or **16 GB** Mac it still runs by **streaming MoE experts from disk**: each token pages in only its router-selected experts (LRU-cached), so the 282 big expert tensors are never all in memory at once. Resident memory stays a few GB and output is **bit-identical** to the fully-resident model.

Measured (greedy decode):

| Machine | Cache budget | Expert hit-rate | Disk read / token | Decode speed | Peak memory |
|---|---|---|---|---|---|
| M4 Mac mini, 16 GB | `--cache-budget-gb 6` | ~38% | ~3.2 GB | ~0.2 tok/s | 10.1 GB |
| 64 GB Apple Silicon | `--cache-budget-gb 40` | **94.1%** | ~0.28 GB | **~4–6 tok/s** (warm) | 46 GB |

On the **64 GB Mac** a 40 GB cache holds ~60% of the ~67 GB of experts, but temporal locality lifts the hit-rate to **94.1%** — so once the working set is warm, decode runs at the model's compute-bound **~4–6 tok/s** (the math, format, and repetition probes ran at this rate). Throughput is **bursty**: the first generation and tasks that route into a colder slice of experts still stall on the SSD (the 6-probe run read 269.5 GB total), pulling those tests below 1 tok/s until their experts page in. Peak memory was **46 GB**, comfortably inside 64 GB.

On **16 GB** it is far more disk-bound: the per-token working set is ~4.1 GB (top-8 × 94 layers × ~5.5 MB/expert), and a 6 GB cache holds less than a fifth of the ~67 GB of experts, so most tokens fall through to the SSD (~724 MB/s here). It runs, but slowly.

**Levers that raise throughput:**
- **More cache** — `--cache-budget-gb 40-50` on a 64 GB Mac keeps most hot experts resident (the 94% hit-rate above). Bump `sudo sysctl iogpu.wired_limit_mb=57344` to go past the ~48 GB default Metal wired cap.
- **Faster storage** — a Thunderbolt NVMe (~2-3 GB/s) vs the internal SSD directly multiplies streaming speed, since SSD bandwidth is the limiter on cache misses.
- **96 GB+ Mac** — skip streaming entirely; load fully resident.

Expert streaming for `qwen3_moe` ships in TurboQuant-MLX **0.6.0+**:

```bash
pip install "turboquant-mlx-full>=0.6.0"

python -m turboquant_mlx.stream.stream_generate \
    --model manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32 \
    --prompt "Explain why the sky is blue." \
    --max-tokens 512 --cache-budget-gb 40
```

## Requirements

```bash
# macOS with Apple Silicon (M1/M2/M3/M4)
pip install turboquant-mlx-full mlx-lm
```

## Quick Start

Fully resident (96 GB+ machine), via mlx-lm:

```python
from mlx_lm import load, generate

model, tokenizer = load("manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32")
response = generate(
    model, tokenizer,
    prompt="Write a Python function that returns the nth Fibonacci number.",
    max_tokens=512,
)
print(response)
```

On 16-64 GB, use the streaming generator shown above.

## How It Works

TurboQuant applies:
1. **Hadamard rotation** — random ±1 scaling to decorrelate weights before quantization
2. **Lloyd-Max codebook** — optimal scalar quantization via k-means
3. **Group-wise scaling** — per-group float16 scales for precision

This achieves better quality than standard affine quantization at the same bit-width. MoE models with many experts (here, 128 with top-8 routing) carry enough redundancy to absorb **2-bit** expert quantization, while the always-on attention path is kept at **3-bit** for safety — the *hybrid* that this checkpoint ships.

### Reproducing the conversion

```bash
python -m turboquant_mlx.convert \
    --hf-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --mlx-path /Volumes/SSD/qwen3-235b-tq3a-tq2e-g32 \
    --bits 3 --mlp-bits 2 -g 32 --streaming
```

`--mlp-bits 2` drops the experts to 2-bit while `--bits 3` keeps attention at 3-bit; `--streaming` writes each quantized layer to a shard and frees it, so the full 235B model converts in **~8-12 GB of RAM** (peak) — it was produced on a **16 GB Mac mini in ~18 minutes**. Point `--mlx-path` at a drive with ≥70 GB free.

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
