# Layer- and expert-streaming for trillion-class MoE on 64 GB Macs

**Branch:** `poc/layer-streaming-1t`
**Status:** design — no code yet
**Goal:** run a 1T-class MoE (Kimi K2, DeepSeek-V3 class) on a single 64 GB
Apple Silicon Mac by combining TurboQuant 3-bit weights with AirLLM-style
on-disk streaming. Trade tokens/sec for memory; aim for a usable-if-patient
range (≥ 0.1 tok/s), not a benchmark winner.

---

## 1. Why we even looked at AirLLM

AirLLM (lyogavin/airllm, Apache 2.0) advertises "70B on 4 GB VRAM" by
splitting the model into per-layer files on disk and loading one layer
at a time during the forward pass. It has an MLX backend, so on paper
it's the closest existing reference for what we'd build.

A read of the cloned repo (`/tmp/airllm-poc/` at HEAD on 2026-05-16)
turns up a less rosy picture than the README implies — see §3.

---

## 2. What AirLLM actually does on MLX today

Cloned source: `air_llm/airllm/`

### 2.1 Per-layer split-and-save (one-time preprocessing)

`utils.py:188 split_and_save_layers` walks `model.safetensors.index.json`,
groups tensors by `model.layers.{i}.*` prefix, and writes one shard per
layer. On the MLX backend the persister is
`persist/mlx_model_persister.py:77 persist_model`, which converts each
layer's state dict to fp16 numpy and saves as `<layer>.mlx.npz`. A
sibling `.mlx.done` marker file signals "fully written" so partial runs
don't corrupt the cache.

Layer naming is HF-conventional: `model.embed_tokens`,
`model.layers.{0..N-1}`, `model.norm`, `lm_head`. A whole MoE block
(all 8 Mixtral experts, all 256 Qwen3-MoE experts, etc.) lands in a
**single** `model.layers.{i}.mlx.npz` file.

### 2.2 The streaming forward pass (MLX path)

`airllm_llama_mlx.py:265 model_generate` is the core loop. For the
prompt pass:

```python
# embed
self.tok_embeddings = nn.Embedding(...)
self.tok_embeddings.update(persister.load_model('model.embed_tokens', ...))
x = self.tok_embeddings(x); mx.eval(x); del self.tok_embeddings; gc.collect()

for il in range(n_layers):
    l = TransformerBlock(args=self.model_args)
    l.update(persister.load_model(f'model.layers.{il}', ...)['layers'][il])
    x, c = l(x, mask=mask)
    mx.eval(x)
    cache.append(c)            # per-layer KV stays in RAM across tokens
    del l; gc.collect()        # weights are evicted every layer
```

Decode pass (`airllm_llama_mlx.py:367`) is the same shape, reusing
`cache[i]`.

Key properties of the MLX path:

| | |
|---|---|
| Per-step RAM | **one transformer block's weights** + KV-cache-list + activations |
| Per-token disk read | **whole model** (all layers, embed, norm, lm_head) |
| Layer architecture | hard-coded Llama-2 (`RMSNorm`/`RoPE`/GQA `Attention`/SiLU `FeedForward` in `airllm_llama_mlx.py:71–177`) |
| Quantization | **none** — weights stored as fp16 `.npz`, no codebook, no scales |
| Prefetching | **none on MLX** — the `ThreadPoolExecutor` + `torch.cuda.Stream` pipeline lives in `airllm_base.py:441–487` and is CUDA-only |
| Eviction policy | `del l; gc.collect()` — eager, every layer, no LRU |
| MoE awareness | **none** — `airllm_mixtral.py` is a 22-line trivial subclass that only disables BetterTransformer |

### 2.3 The "compression" feature is a CUDA-only red herring

The 4bit/8bit option uses `bitsandbytes` (`utils.py:157
compress_layer_state_dict`), which is CUDA-only. On the MLX path
compression is silently unavailable — you get fp16 npz files at full
size. So on Mac, AirLLM is reading the **fp16** model from disk every
token. For a 1T model that's ~2 TB streamed per token over NVMe.

---

## 3. Gap analysis vs. what we need

| Gap | AirLLM today | Required for our target |
|---|---|---|
| **G1. Quantized layer files** | fp16 `.npz` | TurboQuant 3-bit packed indices + per-group fp16 scales + per-codebook table |
| **G2. Expert-level granularity** | one file per *layer* (all experts inside) | one file (or one shard offset) per *expert*, so router top-k drives the read |
| **G3. Lazy paging** | `mx.load(npz)` loads the entire tensor eagerly | safetensors mmap so the OS page cache and SSD readahead do the work |
| **G4. Prefetch on MLX** | none | overlap layer N compute with layer N+1 disk fetch (threadpool or `mx.async_load`) |
| **G5. Eviction policy** | eager `del` every step | LRU on a fixed RAM budget so hot experts stay resident |
| **G6. MoE-architecture support** | trivial Mixtral subclass with no expert routing in the loop | a real MoE block driver that consumes router logits and only materializes top-k experts |
| **G7. Architecture coverage** | Llama-2 only on MLX | Nemotron-H, Qwen3-MoE, DeepSeek-V3, Kimi K2 — each is its own block class in MLX |

G1 and G2 together are the load-bearing pair. G2 alone (smaller reads)
buys order-of-magnitude speedups on MoE; G1 alone buys a 5–6× cut on
disk bandwidth. Both stack.

---

## 4. Throughput model

64 GB unified-memory Mac, NVMe SSD sustained ≈ 5 GB/s sequential read,
no kernel-level surprises. The decode-time bottleneck is *I/O*, not
compute — once compute is hidden behind I/O via prefetch, throughput
is just `bytes_read_per_token / disk_bandwidth`.

### 4.1 Dense 1T

| Build | On disk | Bytes/token | Floor latency | Tok/s |
|---|---|---|---|---|
| fp16 (AirLLM as-is) | 2 TB | 2 TB | 400 s | 0.0025 |
| TQ 2-bit g32 | 250 GB | 250 GB | 50 s | 0.02 |
| TQ 3-bit g32 | 375 GB | 375 GB | 75 s | 0.013 |

Dense 1T is technically runnable but practically a slideshow.
**Not the target.**

### 4.2 MoE 1T (Kimi K2 / DeepSeek-V3 class, ~30 B active per token)

Active params per token = attention + shared + top-k experts ≈ 30 B.

| Build | Active-bytes/token | Floor latency | Tok/s |
|---|---|---|---|
| fp16 (AirLLM "expert-aware" — doesn't exist) | 60 GB | 12 s | 0.08 |
| TQ 2-bit g32 expert-aware | 7.5 GB | 1.5 s | 0.67 |
| **TQ 3-bit g32 expert-aware** | **11 GB** | **2.2 s** | **0.45** |

With realistic overheads (codebook dequant, kernel launch, KV cache
maintenance) shave 30–50%. Realistic target: **0.2–0.4 tok/s**, i.e.
**1 token every 2.5–5 s**. Slow, but interactive-ish for chat-length
outputs; a 500-token response in ~25–40 min.

### 4.3 The prefetch ceiling

If layer N's compute takes 1 s and layer N+1's I/O takes 2.2 s, the
critical path is the slower of the two (2.2 s) plus a fixed overhead.
Without prefetch you pay 1 + 2.2 = 3.2 s per layer-step; with it, 2.2 s.
~30% speedup, which matches the AirLLM README's CUDA claim. Worth
having but not the headline lever.

### 4.4 Sanity check vs. the AirLLM 70B-on-4GB number

AirLLM's claim is the dense-4bit-on-CUDA case: 70 B × 0.5 byte = 35 GB
on disk, streamed in 7 s/token over PCIe + NVMe combined. Their
published numbers are ~0.5–1 tok/s on consumer hardware. Our MoE 1T
math (0.2–0.4 tok/s) is in the same ballpark per byte streamed — the
1T number works only because **active-params << total params**.

---

## 5. What we'd build

### 5.1 Module layout

```
turboquant_mlx/
└── stream/                       # NEW
    ├── __init__.py
    ├── persist.py                # TQ-aware per-expert sharding (replaces airllm utils.split_and_save_layers)
    ├── loader.py                 # mmap + async fetch + LRU cache
    ├── block.py                  # MoE-aware block driver (consumes router top-k)
    ├── archs/
    │   ├── llama.py              # parity check vs airllm_llama_mlx.py on a small model
    │   ├── mixtral.py            # first real MoE (8 experts) — minimal interesting case
    │   ├── qwen3_moe.py
    │   ├── deepseek_v3.py
    │   └── kimi_k2.py
    └── stream_generate.py        # entrypoint, mirrors generate.py
```

### 5.2 On-disk format

Per-expert sharded TQ checkpoint:

```
<model>.tq-stream/
├── manifest.json                 # arch, n_layers, n_experts_per_layer, group_size,
│                                 # tq codebook (per-bit-width), layer→file mapping
├── meta/                         # tied across layers
│   ├── embed_tokens.tq           # codebook + packed indices + scales
│   ├── norm.tq
│   └── lm_head.tq                # often tied to embed; if so, just a symlink
└── layers/
    └── {i}/
        ├── attn.tq               # qkv + o (3-bit per --attn-bits if hybrid)
        ├── attn_norm.fp16        # tiny, stays loaded
        ├── ffn_norm.fp16
        ├── router.fp16           # tiny, stays loaded
        ├── shared_expert.tq      # if arch has one (DSV3, Nemotron-3)
        └── experts/
            ├── 000.tq
            ├── 001.tq
            └── ... (up to 511 for nemotron-3, 128 for DSV3, 384 for K2)
```

The `.tq` file is a safetensors layout (`mx.load(...,
return_metadata=True)` friendly) holding three arrays — `codebook`
(shape `[2**bits]`, fp16), `indices` (shape `[out_features,
in_features // group_size, group_size]`, packed uint8), `scales`
(shape `[out_features, in_features // group_size]`, fp16). Reusing
safetensors gives us mmap for free.

**Why one file per expert and not one large per-layer file with
offsets?** Both work. Per-file is simpler, plays well with the OS page
cache, and means the LRU evictor can call `madvise(MADV_DONTNEED)` per
expert. Per-layer-with-offsets gives one fewer syscall per expert but
needs a custom range-load path. Start with per-file; revisit if open()
overhead shows up in profiling.

### 5.3 Hot loop (MoE-aware, decode pass)

```python
# pseudo-code, MoE block step
def moe_block_step(x, layer_idx, kv_cache, loader, lru):
    # tiny tensors — always resident
    attn_norm, ffn_norm = lru.get_resident(layer_idx, ['attn_norm', 'ffn_norm'])
    router            = lru.get_resident(layer_idx, ['router'])

    # attention — already prefetched by previous step
    attn_w = lru.get(layer_idx, 'attn', prefetched=True)
    x = attn_norm(x); x = attn_w(x, kv_cache); del attn_w  # eviction managed by LRU

    # routing
    h = ffn_norm(x)
    routing_logits = router(h)
    topk_experts = topk_indices(routing_logits, k=K)   # shape [B*L, K]

    # kick off prefetch of likely-next experts (heuristic: top-k from a
    # smoothed running estimate, or just the same set from the next layer)
    loader.prefetch_next_experts(layer_idx + 1, topk_experts)

    # materialize and fuse only the experts this token actually needs
    out = zeros_like(h)
    for e in unique(topk_experts):
        we = lru.get(layer_idx, f'experts/{e}')
        out += routing_weight[e] * we(h)
    return x + out
```

The eviction predicate is *"keep if predicted hot in next 2 layers,
else drop"*. A naive LRU is the floor; a router-trace-trained predictor
is the ceiling.

### 5.4 Persistence pipeline

Read the existing TQ checkpoint (produced by `convert.py`), split per
expert, write the streaming layout. Roughly:

```python
def make_stream_checkpoint(tq_in: Path, out: Path, arch_spec):
    weights = safetensors_open(tq_in / 'model.safetensors.index.json')
    for layer_idx, name in arch_spec.iter_layer_components(weights):
        # name = 'layer.7.experts.42' or 'layer.7.attn'
        write_tq_shard(out / shard_path(name), weights, name)
    write_manifest(out / 'manifest.json', arch_spec, ...)
```

This is the moral equivalent of AirLLM's
`utils.py:188 split_and_save_layers` but TQ- and MoE-aware. **We reuse
no AirLLM code** — the path mapping (`map_torch_to_mlx` in
`mlx_model_persister.py:16`) is Llama-2 hardcoded and doesn't help us.

---

## 6. PoC milestones

Each stage gates the next. Stop and re-plan if a stage misses its
acceptance bar.

| M | Goal | Acceptance | Effort |
|---|---|---|---|
| **M1** | Reproduce AirLLM's MLX layer-streaming on a small Llama (Llama-3.2-1B) using **our own loader** (mmap'd safetensors, no npz). | Generates same tokens as `mlx_lm` baseline within rounding; peak RAM ≤ size of one transformer block + activations. | 1–2 days |
| **M2** | Swap the fp16 layer reads for TQ-quantized layer reads on the same Llama-3.2-1B (post-`convert.py`). | Same generation parity; on-disk shrinks ~5×; tok/s within 30% of M1. | 2 days |
| **M3** | Mixtral-8x7B (or Qwen3-30B-A3B) MoE block driver. Loads only top-k experts per token, not all experts. | Generates coherent text; per-token bytes-read ≈ (active params × bytes/param) within 2×. | 3–5 days |
| **M4** | Threadpool prefetch (layer N+1 fetch overlaps layer N compute). MLX has `mx.async_eval`/`stream` primitives — investigate before rolling our own. | ≥ 25% speedup over M3 on Mixtral; **no** generation regressions. | 2 days |
| **M5** | DeepSeek-V3 (or equivalent reachable 1T-class) end-to-end. Accept any tok/s ≥ 0.1. | Coherent 256-token output on a stress prompt; documented on a 64 GB Mac. | 1–2 weeks (mostly arch-specific block code, KV cache, attention variants) |

Total: ~4 weeks elapsed if everything goes well, more realistically
6–8 with debugging the long tail of MoE arch quirks (latent MoE,
multi-token-prediction heads, hybrid Mamba/attention layouts).

---

## 7. Open questions / decision points

1. **MLX `mx.async_eval` vs. ThreadPoolExecutor.** MLX has lazy
   evaluation; if `mx.load` is non-blocking up until the first op that
   forces materialization, we may be able to express prefetch in pure
   MLX without a thread pool. Needs a 10-line measurement before M4.

2. **Should we ship this as part of `turboquant-mlx-full`, or a new
   sibling package `turboquant-stream`?** Lean toward the latter — keeps
   the install footprint of the core small for users who don't need it.
   Decide at M3 when the API is stable enough.

3. **KV cache strategy for very long context on small RAM.** Per-layer
   KV scales linearly with sequence length. For a 1T model at 8K
   context, KV alone can be tens of GB even at fp16. We already have
   `TurboQuantKVCache`; making it the default in the streaming path
   keeps RAM steady. **Wire this in at M3, not later.**

4. **Speculative expert prefetch.** Naive policy: prefetch the same
   top-k expert IDs from layer N for layer N+1. Better: train a tiny
   predictor from a routing-trace log. Don't bother before M4.

5. **Whether to attempt a *true* 1T (Kimi K2-class, ~1 T total,
   ~30 B active) or settle for DeepSeek-V3 (671 B total, 37 B active)
   as the headline.** K2's per-expert size is smaller (more experts,
   so more variety, but each individually cheaper to load); V3 has
   better-documented Apple Silicon attempts. Pick at M4 based on what
   actually downloads cleanly.

6. **Disk footprint.** A TQ-3-bit copy of Kimi K2 is ~375 GB. The
   original BF16 is ~2 TB. The streaming layout adds ~5% overhead for
   per-expert shard headers. Need to verify the user's disk has room.

---

## 8. What's reusable from AirLLM, concretely

Short list — most of AirLLM doesn't help us. What does:

- **The high-level driver shape** — "build empty block → load weights
  → run → del" — `airllm_llama_mlx.py:304–323`. Sound pattern; we'll
  mirror it.
- **The KV-cache-list-per-layer convention** —
  `airllm_llama_mlx.py:266–316`. Already how MLX users handle this; no
  innovation but a clean reference.
- **The `<layer>.done` marker convention** —
  `mlx_model_persister.py:71`. Cheap, catches half-written shards
  during preprocessing crashes. Worth copying.
- **The HF→cache→split flow** — `utils.py:341
  find_or_create_local_splitted_path`. We'll write our own (TQ- and
  MoE-aware), but the shape of "if no local split, pull from HF, split,
  cache" is exactly right.

What we won't reuse: `map_torch_to_mlx`, `airllm_base.py` (CUDA-only),
the bitsandbytes compression path, the BetterTransformer fallback
plumbing, the `tok_embeddings`-renaming weight-key surgery.

---

## 9. Why not just contribute upstream to AirLLM

Tempting, but no. Upstream is structured around HF Transformers'
`GenerationMixin` (see `airllm_base.py:46`), which we don't use in
turboquant-mlx. MoE-aware streaming would require gutting
`airllm_mixtral.py` and replacing it; that's a fork, not a contribution.
We also need TQ-quantized files as a first-class storage format, which
they have no reason to adopt. Cleaner to build alongside and credit
AirLLM in the README + paper as prior art.

---

## 10. Decision needed before code starts

Pick one of:

- **A.** Greenlight M1+M2 (1B Llama parity + TQ swap, ~3–4 days). Then
  reassess at the M2/M3 boundary based on measured throughput.
- **B.** Greenlight all the way through M3 (Mixtral MoE block, ~1 week
  end-to-end). Larger commitment but the result is the first
  interesting demo.
- **C.** Hold pending arXiv submission. The streaming work doesn't
  affect the paper as submitted; it'd be a follow-up.

Recommendation: **A** — small step, fast feedback, low cost if the
numbers don't pan out on real hardware.
