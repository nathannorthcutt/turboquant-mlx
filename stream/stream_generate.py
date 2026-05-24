"""Generate from a TurboQuant MoE model with experts streamed from disk.

Runs a model whose weights exceed available RAM by keeping only the resident
tensors in memory and streaming router-selected experts on demand.

Example (Qwen3.6-35B-A3B, ~16 GB on disk, runs in ~5 GB RAM):

    python -m turboquant_mlx.stream.stream_generate \\
        --model manjunathshiva/Qwen3.6-35B-A3B-tq3-g32 \\
        --prompt "Explain why the sky is blue." \\
        --max-tokens 256 --cache-budget-gb 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time

import mlx.core as mx

import turboquant_mlx.compat  # noqa: F401
from mlx_lm import generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

from .loader import load_streaming


def _rss_gb() -> float:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return int(out) / 1024 / 1024


def main():
    p = argparse.ArgumentParser(
        description="Stream-generate from a TurboQuant MoE model (experts paged from disk)."
    )
    p.add_argument("--model", required=True, help="Local path or HF repo id of a TurboQuant model.")
    p.add_argument("--prompt", default="Why is the sky blue?")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--cache-budget-gb", type=float, default=3.0,
                   help="Max resident expert memory (LRU-evicted). Lower = less RAM, more disk reads.")
    p.add_argument("--fast", action="store_true", help="Disable QJL correction for faster decode.")
    p.add_argument("--no-chat-template", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    model, tok, cache = load_streaming(args.model, cache_budget_gb=args.cache_budget_gb, fast=args.fast)
    print(f"[stream] loaded in {time.time() - t0:.1f}s | resident RSS={_rss_gb():.2f} GB")

    prompt = args.prompt
    if not args.no_chat_template and hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], add_generation_prompt=True
        )

    sampler = make_sampler(temp=args.temp)
    print("=" * 60)
    t = time.time()
    text = mlx_generate(model, tok, prompt=prompt, max_tokens=args.max_tokens,
                        sampler=sampler, verbose=True)
    dt = time.time() - t
    print("=" * 60)
    n = args.max_tokens
    print(f"[stream] {n} tok in {dt:.1f}s = {n / dt:.1f} tok/s | "
          f"peak RSS={_rss_gb():.2f} GB | mlx_peak={mx.get_peak_memory() / 1e9:.2f} GB")
    s = cache.stats()
    print(f"[stream] expert cache: hit_rate={s['hit_rate']:.1%} resident={s['resident_gb']:.2f} GB "
          f"disk_read={s['bytes_read_gb']:.1f} GB")


if __name__ == "__main__":
    main()
