"""Speculative decoding for a disk-streamed TurboQuant target model.

Standard speculative sampling (Chen et al., 2023, "Accelerating Large Language
Model Decoding with Speculative Sampling"). A small, fully-resident *draft*
model proposes K tokens autoregressively; the large *target* model — whose MoE
experts stream from disk via ``load_streaming`` — verifies all K in ONE batched
forward pass. Verification goes through the streaming model's multi-token
(prefill) path, so the per-layer expert disk I/O is amortised over K+1 positions
instead of paid once per token. Every accepted token is distributionally
identical to sampling from the target directly, so output quality is unchanged.

Example (Qwen3-235B target, Qwen3-1.7B draft):

    python -m turboquant_mlx.stream.speculative_generate \\
        --model <qwen3-235b-tq> \\
        --draft-model Qwen/Qwen3-1.7B \\
        --prompt "Explain why the sky is blue." \\
        --max-tokens 256 --draft-steps 4 --temp 0.7 --cache-budget-gb 3

Design notes / deviations from the original task spec
-----------------------------------------------------
1. **Cache rollback uses ``trim()``, not "restore snapshot + re-run".**
   Both ``mlx_lm``'s ``KVCache`` and TurboQuant's ``TurboQuantKVCache`` expose a
   trimmable interface (``is_trimmable()`` / ``trim(n)`` — trimming just rewinds
   the append offset). After the K+1-token verification pass the target KV cache
   is rewound to the accepted prefix with a single ``trim`` per layer. This is
   chosen over the spec's restore+re-run approach for two reasons:
     * Re-running the target on the accepted tokens issues a *second* streaming
       forward pass — the exact disk I/O speculative decoding exists to avoid.
     * The spec's re-run fed ``accepted_tokens`` (which begins at ``draft[0]``
       and *ends with the correction token*). That would drop ``KV(token)``,
       wrongly store ``KV(correction)``, and shift every position by one —
       corrupting the cache. Trimming the already-correct verified KV is exact.
   A ``state``-slicing fallback is used for any cache that reports itself
   non-trimmable.

2. **The draft KV cache is kept exactly consistent** (trimmed on partial
   acceptance, and given a one-token catch-up feed on full acceptance, where the
   draft never ran on its own last proposal). An off-by-one draft cache would
   misalign every subsequent draft prediction and collapse the acceptance rate,
   so leaving it "slightly inconsistent" is not viable in practice.

3. ``temp == 0`` produces a clean one-hot distribution / argmax sample rather
   than the ``softmax(logits * 1e6)`` trick, which overflows to NaN on large
   logits.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import time

import numpy as np
import mlx.core as mx

import turboquant_mlx.compat  # noqa: F401  (side-effect: upstream shims)

from turboquant_mlx.generate import load_turboquant, resolve_model_path
from .loader import load_streaming


# Recommended draft model for the Qwen3-235B streaming target: same tokenizer
# family (vocab-compatible), fully resident at ~1.5 GB, and a high acceptance
# rate because it shares the target's pretraining distribution.
_RECOMMENDED_DRAFT = "Qwen/Qwen3-1.7B"


def _rss_gb() -> float:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return int(out) / 1024 / 1024


# --------------------------------------------------------------------------- #
# Model / cache setup
# --------------------------------------------------------------------------- #
def load_both(target_model_path, draft_model_path, cache_budget_gb=3.0,
              fast=False, prefetch_workers=8, prefetch_ahead=None,
              pin_file=None, max_active_experts=4, use_page_cache=None,
              warmup_file=None, warmup_gb=0.0, perm_path=None):
    """Load a disk-streaming target model and a fully-resident draft model.

    Returns ``(target_model, target_tokenizer, expert_cache, draft_model,
    draft_tokenizer)``. All ``*expert*`` / streaming kwargs apply to the target
    only; the draft is small enough to stay wholly in RAM.
    """
    # Draft: small model, fully resident (no expert streaming).
    draft_path = str(resolve_model_path(draft_model_path))
    draft_model, draft_tokenizer = load_turboquant(draft_path, lazy=False, fast=fast)
    mx.eval(draft_model.parameters())

    # Target: large MoE model, experts paged from disk.
    target_model, target_tokenizer, expert_cache = load_streaming(
        target_model_path, cache_budget_gb=cache_budget_gb, fast=fast,
        prefetch_workers=prefetch_workers, prefetch_ahead=prefetch_ahead,
        pin_file=pin_file, max_active_experts=max_active_experts,
        use_page_cache=use_page_cache, warmup_file=warmup_file,
        warmup_gb=warmup_gb, perm_path=perm_path,
    )

    # Speculative sampling is only valid when both models share a vocabulary
    # (token id i means the same thing to both). Same vocab_size is a cheap,
    # sufficient proxy for the same-family draft/target pairs this targets.
    dv = getattr(draft_tokenizer, "vocab_size", None)
    tv = getattr(target_tokenizer, "vocab_size", None)
    if dv is not None and tv is not None and dv != tv:
        print(f"[spec] WARNING: tokenizer vocab mismatch "
              f"draft={dv} target={tv} — acceptance may be low / ids may not align")

    return target_model, target_tokenizer, expert_cache, draft_model, draft_tokenizer


def _make_cache(model):
    """Create a per-layer KV cache for ``model`` (handles hybrid models)."""
    from mlx_lm.models.cache import make_prompt_cache
    return make_prompt_cache(model)


def _cache_offset(cache_list) -> int:
    """Current sequence length held by the cache (all layers share an offset)."""
    return int(getattr(cache_list[0], "offset", 0)) if cache_list else 0


def _slice_state_to(c, desired_len):
    """Fallback trim for a non-trimmable cache: slice K/V state on the seq axis.

    Used only when a cache reports ``is_trimmable() == False``. Tries the
    common mlx_lm layout (heads, seq, dim) first, then a couple of fallbacks.
    """
    state = c.state
    if not (isinstance(state, tuple) and len(state) == 2):
        return  # unknown layout — leave as-is (correctness handled by target trim)
    k, v = state
    if k is None:
        return
    for seq_dim in (2, 1, 0):
        if k.ndim > seq_dim and k.shape[seq_dim] > desired_len:
            idx = [slice(None)] * k.ndim
            idx[seq_dim] = slice(None, desired_len)
            k = k[tuple(idx)]
            v = v[tuple(idx)]
            break
    c.state = (k, v)
    if hasattr(c, "offset"):
        c.offset = desired_len


def _trim_cache_to(cache_list, desired_len):
    """Rewind every layer cache to hold exactly ``desired_len`` tokens."""
    for c in cache_list:
        cur = int(getattr(c, "offset", 0))
        if cur <= desired_len:
            continue
        n = cur - desired_len
        trimmable = (hasattr(c, "trim") and
                     (not hasattr(c, "is_trimmable") or c.is_trimmable()))
        if trimmable:
            c.trim(n)
        else:
            _slice_state_to(c, desired_len)


# --------------------------------------------------------------------------- #
# Forward step + sampling primitives
# --------------------------------------------------------------------------- #
def _model_step(model, tokens, cache):
    """Run ``model`` on ``tokens`` (list / 1-D array / 2-D array), extending the
    KV cache, and return the per-position logits of shape ``(seq_len, vocab)``.

    Multi-token inputs go through the model's prefill path (the model builds its
    own causal mask internally), which is what amortises expert disk I/O.
    """
    if isinstance(tokens, mx.array):
        x = tokens if tokens.ndim == 2 else tokens[None]
    else:
        x = mx.array(tokens)[None]
    logits = model(x, cache=cache)
    mx.eval(logits)
    return logits[0]  # (seq_len, vocab)


def _sample_token(logits_vec, temp=1.0):
    """Sample a single token id from a 1-D logits vector."""
    if temp == 0:
        return int(mx.argmax(logits_vec).item())
    probs = mx.softmax(logits_vec / temp)
    return int(mx.random.categorical(mx.log(probs)).item())


def _get_probs(logits_vec, temp=1.0):
    """Full probability distribution over the vocab from a 1-D logits vector.

    ``temp == 0`` returns a one-hot at the argmax (greedy) rather than scaling
    the logits, which would overflow ``exp`` to NaN on large-magnitude logits.
    """
    if temp == 0:
        oh = mx.zeros_like(logits_vec)
        oh[int(mx.argmax(logits_vec).item())] = 1.0
        return oh
    return mx.softmax(logits_vec / temp)


def accept_reject(draft_token, target_probs, draft_probs, temp=1.0):
    """One position of speculative sampling.

    Returns ``(accepted: bool, correction_token: int | None)``.

    Accept ``draft_token`` with probability ``min(1, p_target(x)/p_draft(x))``.
    On rejection, sample a correction from the residual distribution
    ``max(0, p_target - p_draft)`` renormalised — guaranteeing the emitted token
    is distributed exactly as ``p_target``.
    """
    t_p = float(target_probs[draft_token].item())
    d_p = float(draft_probs[draft_token].item())
    accept_prob = min(1.0, t_p / max(d_p, 1e-10))
    if random.random() < accept_prob:
        return True, None
    adjusted = mx.maximum(target_probs - draft_probs, mx.zeros_like(target_probs))
    z = float(adjusted.sum().item())
    if z < 1e-10:
        # Degenerate residual (e.g. greedy where target == draft argmax but the
        # accept coin still failed on a rounding edge): fall back to the target.
        correction = _sample_token(mx.log(mx.maximum(target_probs, 1e-20)), temp)
    else:
        correction = int(mx.random.categorical(mx.log(adjusted / z)).item())
    return False, correction


# --------------------------------------------------------------------------- #
# Main speculative loop
# --------------------------------------------------------------------------- #
def speculative_generate(
    target_model, target_cache, draft_model, draft_cache,
    prompt_tokens, max_tokens=256, draft_steps=4, temp=0.7,
    eos_ids=None, tokenizer=None, verbose=True,
):
    """Generate up to ``max_tokens`` tokens by speculative decoding.

    ``target_cache`` / ``draft_cache`` are per-layer KV cache lists (from
    :func:`_make_cache`), assumed empty. Returns ``(generated_ids, stats)``.

    Invariant maintained across iterations: both caches hold every *confirmed*
    token except the last one, which is re-fed as the first draft/verify input.
    """
    eos_ids = set(eos_ids or ())
    prompt_tokens = list(prompt_tokens)

    # ---- Prefill both models over the prompt (target: one streaming pass) ----
    t_logits = _model_step(target_model, prompt_tokens, target_cache)
    _model_step(draft_model, prompt_tokens, draft_cache)

    tokens = list(prompt_tokens)          # full confirmed sequence
    generated: list[int] = []

    # First token comes from the target so generation starts on-distribution.
    token = _sample_token(t_logits[-1], temp)
    generated.append(token)
    tokens.append(token)

    n_iters = n_draft_tokens = n_accepted = 0
    printed_chars = 0
    stop = token in eos_ids

    def _emit(new_ids):
        nonlocal printed_chars
        if verbose and tokenizer is not None:
            full = tokenizer.decode(generated)
            print(full[printed_chars:], end="", flush=True)
            printed_chars = len(full)

    if stop:
        # First target token was already EOS — nothing to speculate.
        _emit(generated)
    while len(generated) < max_tokens and not stop:
        K = min(draft_steps, max_tokens - len(generated))
        if K <= 0:
            break

        # ---- DRAFT PHASE: propose K tokens autoregressively (resident) ----
        draft_tokens: list[int] = []
        draft_probs_list = []
        d_token = token
        for _ in range(K):
            d_logits = _model_step(draft_model, [d_token], draft_cache)
            probs = _get_probs(d_logits[-1], temp)
            d_token = _sample_token(d_logits[-1], temp)
            draft_tokens.append(d_token)
            draft_probs_list.append(probs)
            n_draft_tokens += 1

        # ---- VERIFY PHASE: one batched streaming pass over [token]+drafts ----
        # verify_input[j] -> t_logits[j] predicts the token that FOLLOWS it, so
        # t_logits[i] is the target distribution for draft_tokens[i], and
        # t_logits[K] is the bonus distribution after the last draft token.
        verify_input = [token] + draft_tokens
        t_logits = _model_step(target_model, verify_input, target_cache)
        t_probs_list = [_get_probs(t_logits[i], temp) for i in range(K)]

        # ---- ACCEPT / REJECT ----
        accepted_tokens: list[int] = []
        n_acc = 0
        for i in range(K):
            ok, correction = accept_reject(
                draft_tokens[i], t_probs_list[i], draft_probs_list[i], temp
            )
            if ok:
                accepted_tokens.append(draft_tokens[i])
                n_acc += 1
            else:
                accepted_tokens.append(correction)  # residual-sampled correction
                break

        if n_acc == K:
            # Every draft accepted: the target's K+1-th logit gives a free
            # bonus token (its KV is NOT in the cache — it's the new last token).
            accepted_tokens.append(_sample_token(t_logits[K], temp))

        # ---- COMMIT + KV SYNC ----------------------------------------------
        # Truncate at the first EOS if one was produced this round.
        emit_ids = accepted_tokens
        for j, tk in enumerate(accepted_tokens):
            if tk in eos_ids:
                emit_ids = accepted_tokens[:j + 1]
                stop = True
                break

        tokens.extend(emit_ids)
        generated.extend(emit_ids)
        token = emit_ids[-1]
        n_accepted += n_acc
        n_iters += 1

        # Restore the invariant: caches hold all confirmed tokens except the
        # last. desired == len(tokens) - 1. The target (verified K+1 ahead) is
        # trimmed down; the draft (which never ran on its own final proposal)
        # is trimmed on partial accept or given a one-token catch-up on full.
        desired = len(tokens) - 1
        _trim_cache_to(target_cache, desired)
        draft_off = _cache_offset(draft_cache)
        if draft_off > desired:
            _trim_cache_to(draft_cache, desired)
        elif draft_off < desired:
            _model_step(draft_model, tokens[draft_off:desired], draft_cache)

        _emit(emit_ids)

    if verbose and tokenizer is not None:
        print()

    stats = {
        "iterations": n_iters,
        "draft_tokens": n_draft_tokens,
        "accepted": n_accepted,
        "generated": len(generated),
        "acceptance_rate": n_accepted / max(n_draft_tokens, 1),
        "tokens_per_iter": len(generated) / max(n_iters, 1),
    }
    return generated, stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Speculative decoding: small resident draft + disk-streamed "
                    "TurboQuant MoE target (amortises expert I/O over K tokens)."
    )
    p.add_argument("--model", required=True,
                   help="Local path or HF repo id of the streaming TurboQuant target.")
    p.add_argument("--draft-model", required=True,
                   help="Small, fully-resident draft model (local path or HF repo id). "
                        f"Recommended for Qwen3-235B targets: {_RECOMMENDED_DRAFT} "
                        "(same tokenizer family, ~1.5 GB, high acceptance).")
    p.add_argument("--draft-steps", type=int, default=4,
                   help="K: draft tokens proposed per verification pass (default 4).")
    p.add_argument("--prompt", default="Why is the sky blue?")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.7)
    # --- target streaming flags (mirror stream_generate.py) ---
    p.add_argument("--cache-budget-gb", type=float, default=3.0,
                   help="Max resident expert memory for the target (LRU-evicted).")
    p.add_argument("--prefetch-workers", type=int, default=8,
                   help="Threads for parallel per-layer expert reads (1 = serial).")
    p.add_argument("--prefetch-ahead", type=int, default=-1,
                   help="Speculatively prefetch this many upcoming layers' experts. "
                        "Default: auto (1 on internal NVMe, 0 otherwise); 0 disables.")
    p.add_argument("--pin-file", default=None,
                   help="JSON {'pin': [[layer, expert], ...]} of hot experts to keep resident.")
    p.add_argument("--max-active-experts", type=int, default=4,
                   help="Cap router top_k to min(native, this) on every target MoE block. "
                        "Default 4; 0 = native routing.")
    p.add_argument("--use-page-cache", dest="use_page_cache", action="store_true",
                   default=None, help="Force the OS page cache ON for target expert reads.")
    p.add_argument("--no-page-cache", dest="use_page_cache", action="store_false",
                   help="Force F_NOCACHE (page cache off) for target expert reads.")
    p.add_argument("--warmup-file", default=None,
                   help="Histogram JSON for cross-session target cache warmup.")
    p.add_argument("--warmup-gb", type=float, default=20.0,
                   help="Warmup budget in GB (ignored without --warmup-file).")
    p.add_argument("--perm-file", default=None,
                   help="perm.json for a stream/repack.py-repacked target (logical->physical).")
    p.add_argument("--fast", action="store_true",
                   help="Disable QJL correction on both models for faster decode.")
    p.add_argument("--no-chat-template", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    prefetch_ahead = args.prefetch_ahead if args.prefetch_ahead >= 0 else None
    (target_model, tok, expert_cache,
     draft_model, _draft_tok) = load_both(
        args.model, args.draft_model,
        cache_budget_gb=args.cache_budget_gb, fast=args.fast,
        prefetch_workers=args.prefetch_workers, prefetch_ahead=prefetch_ahead,
        pin_file=args.pin_file, max_active_experts=args.max_active_experts,
        use_page_cache=args.use_page_cache,
        warmup_file=args.warmup_file, warmup_gb=args.warmup_gb,
        perm_path=args.perm_file,
    )
    print(f"[spec] loaded target(streaming) + draft({args.draft_model}) in "
          f"{time.time() - t0:.1f}s | resident RSS={_rss_gb():.2f} GB")
    print(f"[spec] tip: {_RECOMMENDED_DRAFT} is the recommended draft for "
          f"Qwen3-235B targets (same family, ~1.5 GB).")

    prompt = args.prompt
    if not args.no_chat_template and hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True
        )
    prompt_tokens = tok.encode(prompt) if isinstance(prompt, str) else prompt

    from turboquant_mlx.sampling import eos_token_ids
    eos_ids = eos_token_ids(tok)

    target_cache = _make_cache(target_model)
    draft_cache = _make_cache(draft_model)

    print("=" * 60)
    t = time.time()
    generated, stats = speculative_generate(
        target_model, target_cache, draft_model, draft_cache,
        prompt_tokens, max_tokens=args.max_tokens, draft_steps=args.draft_steps,
        temp=args.temp, eos_ids=eos_ids, tokenizer=tok, verbose=True,
    )
    dt = time.time() - t
    print("=" * 60)

    n = len(generated)
    print(f"[spec] {n} generated tok in {dt:.1f}s = {n / dt:.1f} tok/s (end-to-end) | "
          f"peak RSS={_rss_gb():.2f} GB | mlx_peak={mx.get_peak_memory() / 1e9:.2f} GB")
    print(f"[spec] speculative stats: acceptance_rate={stats['acceptance_rate']:.1%}, "
          f"tokens_per_iter={stats['tokens_per_iter']:.1f}, "
          f"draft_tokens_tried={stats['draft_tokens']} "
          f"accepted={stats['accepted']} iterations={stats['iterations']}")
    # Rough model: sequential decode = 1 target call/token; speculative =
    # ~(alpha*K + 1) confirmed tokens per ~2 target calls (verify + correction).
    speedup = (stats["acceptance_rate"] * args.draft_steps + 1) / 2
    print(f"[spec] effective speedup over sequential: ~{speedup:.1f}x")

    s = expert_cache.stats()
    print(f"[spec] expert cache: hit_rate={s['hit_rate']:.1%} "
          f"(resident {s['cache_hit_rate']:.1%} + prefetch {s['prefetch_hit_rate']:.1%}) "
          f"resident={s['resident_gb']:.2f} GB")
    print(f"[spec] disk: critical_read={s['bytes_read_gb']:.1f} GB "
          f"prefetched={s['bytes_prefetched_gb']:.1f} GB total={s['bytes_total_gb']:.1f} GB")

    if args.warmup_file:
        m = expert_cache.dump_histogram(
            args.warmup_file, model_id=args.model, k=args.max_active_experts)
        print(f"[spec] histogram saved: {m} (layer,expert) pairs -> {args.warmup_file}")


if __name__ == "__main__":
    main()
