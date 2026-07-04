"""Batched streaming generation: runs B independent sequences simultaneously,
sharing the streaming expert cache so popular experts are loaded once per step
instead of B times.

The batch is driven through the model as a *single* forward pass (inputs of
shape ``(B, 1)`` per decode step). Every sequence's router selections therefore
arrive at each MoE block in one ``indices`` tensor of shape ``(B, 1, K)``, so:

  * ``StreamingSwitchLinear`` sees the whole batch at once and gathers the
    UNION of experts across all B sequences in a single ``ExpertCache.gather``
    — a popular expert routed to by several sequences is read from disk once
    per step, not once per sequence. Expected 20-60% overlap on typical
    queries turns directly into that many fewer expert disk reads.
  * The MoE math for the whole batch runs as one fused ``polar_multi_gather_qmv``
    call over the B×K routings (true parallel forward — no per-sequence Python
    loop inside the layer). See ``StreamingSwitchLinear.__call__`` batch path.

Differing prompt lengths are handled by LEFT-padding to a common length and
passing a boolean attention mask so real tokens never attend to pad positions.
Left-padding is correct for RoPE because attention scores depend on the
*relative* offset between query and key, which a uniform per-sequence shift
leaves unchanged. Equal-length prompts need no padding and are the
unconditionally-correct case (the mask reduces to plain causal).

NOTE: this drives the model with ``model(x, cache=cache, mask=mask)``. Standard
mlx_lm decoder models accept that ``mask`` kwarg; a model that does not will
raise a clear TypeError rather than silently mis-attending.

Usage:
    python -m turboquant_mlx.stream.batch_generate \\
        --model <model_path> \\
        --prompts "prompt one" "prompt two" "prompt three" \\
        --max-tokens 256 \\
        [all the normal stream_generate flags for the model]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time

import mlx.core as mx

import turboquant_mlx.compat  # noqa: F401 — registers upstream patches on import
from mlx_lm.models.cache import make_prompt_cache

from turboquant_mlx.sampling import eos_token_ids
from .loader import load_streaming


def _rss_gb() -> float:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return int(out) / 1024 / 1024


def _tokenize_prompts(tok, prompts, use_chat_template: bool) -> list[list[int]]:
    """Encode each prompt to a list of token ids, applying the chat template
    (per prompt) when available and not disabled."""
    seqs = []
    for p in prompts:
        text = p
        if use_chat_template and hasattr(tok, "apply_chat_template"):
            text = tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True,
                tokenize=False,
            )
        ids = tok.encode(text) if isinstance(text, str) else list(text)
        seqs.append(list(ids))
    return seqs


def _left_pad(seqs, pad_id: int):
    """Left-pad ``seqs`` to a common length. Returns (padded (B, L) int32 array,
    real_lengths list, pad_counts list)."""
    lengths = [len(s) for s in seqs]
    L = max(lengths)
    B = len(seqs)
    padded = [[pad_id] * (L - len(s)) + list(s) for s in seqs]
    pad_counts = [L - n for n in lengths]
    return mx.array(padded, dtype=mx.int32), lengths, pad_counts


def _prefill_mask(pad_counts, L: int):
    """Boolean attention mask (B, 1, L, L) for a left-padded prefill: keep
    position (i, j) iff j <= i (causal) AND j is a real token (j >= pad_count).
    True = attend."""
    B = len(pad_counts)
    j = mx.arange(L).reshape(1, 1, 1, L)
    i = mx.arange(L).reshape(1, 1, L, 1)
    causal = j <= i                                  # (1,1,L,L)
    pad = mx.array(pad_counts, dtype=mx.int32).reshape(B, 1, 1, 1)
    not_pad_key = j >= pad                            # (B,1,1,L) -> broadcasts
    return causal & not_pad_key                       # (B,1,L,L) bool


def _decode_mask(pad_counts, k_len: int):
    """Boolean attention mask (B, 1, 1, k_len) for one decode step: a single new
    query attends to every non-left-pad key (all past positions are <= current,
    so causality is automatic). True = attend."""
    B = len(pad_counts)
    j = mx.arange(k_len).reshape(1, 1, 1, k_len)
    pad = mx.array(pad_counts, dtype=mx.int32).reshape(B, 1, 1, 1)
    return j >= pad                                   # (B,1,1,k_len) bool


def _sample(logits, temp: float):
    """Sample one token per row from (B, vocab) logits. Returns (B,) int32."""
    if temp <= 0.0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)
    return mx.random.categorical(logits * (1.0 / temp)).astype(mx.int32)


def batch_generate(model, tok, prompts, *, max_tokens=256, temp=0.7,
                   use_chat_template=True, verbose=True):
    """Generate continuations for every prompt in ``prompts`` in one batched,
    expert-cache-sharing pass. Returns a list of generated-token-id lists (one
    per prompt, EOS excluded)."""
    eos_ids = eos_token_ids(tok)
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is None:
        pad_id = next(iter(eos_ids)) if eos_ids else 0

    seqs = _tokenize_prompts(tok, prompts, use_chat_template)
    B = len(seqs)
    padded, lengths, pad_counts = _left_pad(seqs, pad_id)
    L = int(padded.shape[1])

    cache = make_prompt_cache(model)

    # ---- Prefill (one batched pass) -------------------------------------
    mask = _prefill_mask(pad_counts, L)
    logits = model(padded, cache=cache, mask=mask)        # (B, L, vocab)
    # Left-padding aligns every sequence's last real token at column L-1.
    next_tokens = _sample(logits[:, -1, :], temp)         # (B,)
    mx.eval(next_tokens)

    generated: list[list[int]] = [[] for _ in range(B)]
    done = [False] * B
    toks = next_tokens.tolist()
    for b in range(B):
        if toks[b] in eos_ids:
            done[b] = True
        else:
            generated[b].append(int(toks[b]))

    # ---- Decode loop (batched, shared expert cache) ---------------------
    step = 0
    while not all(done) and max(len(g) for g in generated) < max_tokens:
        step += 1
        k_len = L + step                                  # keys held after this token
        dmask = _decode_mask(pad_counts, k_len)
        x = next_tokens.reshape(B, 1)                     # (B, 1)
        logits = model(x, cache=cache, mask=dmask)        # (B, 1, vocab)
        next_tokens = _sample(logits[:, -1, :], temp)     # (B,)
        mx.eval(next_tokens)
        toks = next_tokens.tolist()
        for b in range(B):
            if done[b]:
                continue
            t = int(toks[b])
            if t in eos_ids:
                done[b] = True
            else:
                generated[b].append(t)

    if verbose:
        for b in range(B):
            text = tok.decode(generated[b]) if generated[b] else ""
            print(f"\n--- sequence {b} ({len(generated[b])} tok) ---")
            print(text)

    return generated


def main():
    p = argparse.ArgumentParser(
        description="Batched stream-generate from a TurboQuant MoE model "
                    "(B sequences share one streaming expert cache)."
    )
    p.add_argument("--model", required=True,
                   help="Local path or HF repo id of a TurboQuant model.")
    p.add_argument("--prompts", nargs="+", required=True,
                   help="One or more prompts; each becomes a sequence in the batch.")
    p.add_argument("--batch-size", type=int, default=0,
                   help="If >0, cap the number of prompts processed to this many "
                        "(defaults to using every prompt given).")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--cache-budget-gb", type=float, default=3.0,
                   help="Max resident expert memory (LRU-evicted), shared across "
                        "the whole batch.")
    p.add_argument("--prefetch-workers", type=int, default=8,
                   help="Threads for parallel per-layer expert reads. 1 = serial.")
    p.add_argument("--prefetch-ahead", type=int, default=-1,
                   help="Speculatively prefetch this many upcoming layers' experts "
                        "(from the previous step's batch-union routing). "
                        "Default: auto (1 internal NVMe, 0 otherwise); 0 disables.")
    p.add_argument("--pin-file", default=None,
                   help="JSON {'pin': [[layer, expert], ...]} of hot experts to keep "
                        "permanently resident.")
    p.add_argument("--max-active-experts", type=int, default=4,
                   help="Cap router top_k to min(native, this) on every MoE block. "
                        "Default 4; 0 = native routing.")
    p.add_argument("--use-page-cache", dest="use_page_cache", action="store_true",
                   default=None,
                   help="Force the OS page cache ON for expert reads. Default: auto.")
    p.add_argument("--no-page-cache", dest="use_page_cache", action="store_false",
                   help="Force F_NOCACHE (page cache off). Default: auto.")
    p.add_argument("--warmup-file", default=None,
                   help="Histogram JSON for cross-session cache warmup.")
    p.add_argument("--warmup-gb", type=float, default=20.0,
                   help="Warmup budget in GB. Ignored without --warmup-file.")
    p.add_argument("--perm-file", default=None,
                   help="perm.json from calibrate_experts.py analyze (for models "
                        "repacked by stream/repack.py).")
    p.add_argument("--fast", action="store_true",
                   help="Disable QJL correction for faster decode.")
    p.add_argument("--no-chat-template", action="store_true")
    args = p.parse_args()

    prompts = args.prompts
    if args.batch_size and args.batch_size > 0:
        prompts = prompts[: args.batch_size]

    t0 = time.time()
    prefetch_ahead = args.prefetch_ahead if args.prefetch_ahead >= 0 else None
    model, tok, cache = load_streaming(
        args.model, cache_budget_gb=args.cache_budget_gb, fast=args.fast,
        prefetch_workers=args.prefetch_workers, prefetch_ahead=prefetch_ahead,
        pin_file=args.pin_file, max_active_experts=args.max_active_experts,
        use_page_cache=args.use_page_cache,
        warmup_file=args.warmup_file, warmup_gb=args.warmup_gb,
        perm_path=args.perm_file,
    )
    print(f"[batch] loaded in {time.time() - t0:.1f}s | resident RSS={_rss_gb():.2f} GB "
          f"| batch of {len(prompts)} sequences")

    print("=" * 60)
    t = time.time()
    generated = batch_generate(
        model, tok, prompts,
        max_tokens=args.max_tokens, temp=args.temp,
        use_chat_template=not args.no_chat_template, verbose=True,
    )
    dt = time.time() - t
    print("=" * 60)
    n = sum(len(g) for g in generated)
    print(f"[batch] {n} generated tok across {len(prompts)} seq in {dt:.1f}s = "
          f"{n / dt:.1f} tok/s (aggregate) | peak RSS={_rss_gb():.2f} GB | "
          f"mlx_peak={mx.get_peak_memory() / 1e9:.2f} GB")
    s = cache.stats()
    print(f"[batch] expert cache: hit_rate={s['hit_rate']:.1%} "
          f"(resident {s['cache_hit_rate']:.1%} + prefetch {s['prefetch_hit_rate']:.1%}) "
          f"resident={s['resident_gb']:.2f} GB")
    print(f"[batch] disk: critical_read={s['bytes_read_gb']:.1f} GB "
          f"prefetched={s['bytes_prefetched_gb']:.1f} GB total={s['bytes_total_gb']:.1f} GB")
    print(f"[batch] coalescing: {s['expert_reads']} expert-loads in {s['read_runs']} "
          f"range-reads = {s['experts_per_read']:.2f} experts/read")

    if args.warmup_file:
        m = cache.dump_histogram(args.warmup_file, model_id=args.model,
                                 k=args.max_active_experts)
        print(f"[batch] histogram saved: {m} (layer,expert) pairs -> {args.warmup_file}")


if __name__ == "__main__":
    main()
