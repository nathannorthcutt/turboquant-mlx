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
    p.add_argument("--prefetch-workers", type=int, default=8,
                   help="Threads for parallel per-layer expert reads. 1 = serial baseline.")
    p.add_argument("--prefetch-ahead", type=int, default=-1,
                   help="Speculatively prefetch this many upcoming layers' experts "
                        "(predicted from the previous token's routing). "
                        "Default: auto (1 on internal NVMe, 0 otherwise). Pass 0 to "
                        "disable, 1 to force-enable. Prefetch helps only on fast NVMe "
                        "with spare bandwidth (~neutral on a saturated USB bus) and "
                        "self-disables if the storage proves bandwidth-bound.")
    p.add_argument("--pin-file", default=None,
                   help="JSON {'pin': [[layer, expert], ...]} of hot experts to keep "
                        "permanently resident (from calibrate_experts.py).")
    p.add_argument("--max-active-experts", type=int, default=4,
                   help="Cap router top_k to min(native, this) on every MoE block "
                        "(Flash-MoE K-reduction: ~2x less streamed disk I/O at no quality "
                        "cost up to K=4 on validated models). Default 4; 0 = native routing.")
    p.add_argument("--use-page-cache", dest="use_page_cache", action="store_true",
                   default=None,
                   help="Force the OS page cache ON for expert reads ('trust the OS'; "
                        "~2.4x faster decode when the model fits in RAM). Default: auto "
                        "by model-size-vs-RAM.")
    p.add_argument("--no-page-cache", dest="use_page_cache", action="store_false",
                   help="Force F_NOCACHE (page cache off). Default: auto by model-size-vs-RAM.")
    p.add_argument("--warmup-file", default=None,
                   help="Path to histogram JSON for cross-session cache warmup. "
                        "Loaded at startup (if it exists) and updated at session end. "
                        "Generate with --dump-warmup on a first run, or use the same "
                        "path for all runs to accumulate history.")
    p.add_argument("--warmup-gb", type=float, default=20.0,
                   help="Warmup budget in GB: pre-load this many GB of hot experts at "
                        "startup from the warmup file. Default 20 GB. Ignored if "
                        "--warmup-file is not set.")
    p.add_argument("--perm-file", default=None,
                   help="Path to perm.json from calibrate_experts.py analyze. "
                        "Use with a model repacked by stream/repack.py (expert-stacks-only "
                        "repack) so the reader translates logical->physical expert ids. "
                        "Do NOT use with models repacked by repack_experts.py, which "
                        "permutes the router rows and needs no translation.")
    p.add_argument("--use-ane", action="store_true", default=False,
                   help="Route single-token attention to the Apple Neural Engine via CoreML. "
                        "Frees wired GPU memory for the expert hot tier. "
                        "Requires: pip install coremltools (macOS only). "
                        "First-run compiles attention models per sequence-length bucket "
                        "(~30s each, cached).")
    p.add_argument("--fast", action="store_true", help="Disable QJL correction for faster decode.")
    p.add_argument("--no-chat-template", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    # -1 sentinel = "auto" -> None lets load_streaming pick by storage type.
    prefetch_ahead = args.prefetch_ahead if args.prefetch_ahead >= 0 else None
    model, tok, cache = load_streaming(
        args.model, cache_budget_gb=args.cache_budget_gb, fast=args.fast,
        prefetch_workers=args.prefetch_workers, prefetch_ahead=prefetch_ahead,
        pin_file=args.pin_file, max_active_experts=args.max_active_experts,
        use_page_cache=args.use_page_cache,
        warmup_file=args.warmup_file, warmup_gb=args.warmup_gb,
        perm_path=args.perm_file, use_ane=args.use_ane,
    )
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
    # Count tokens actually generated (the model may stop at EOS before
    # max_tokens) — dividing max_tokens by wall-time overstates the rate.
    n = len(tok.encode(text))
    print(f"[stream] {n} generated tok in {dt:.1f}s = {n / dt:.1f} tok/s (end-to-end) | "
          f"peak RSS={_rss_gb():.2f} GB | mlx_peak={mx.get_peak_memory() / 1e9:.2f} GB")
    s = cache.stats()
    print(f"[stream] expert cache: hit_rate={s['hit_rate']:.1%} "
          f"(resident {s['cache_hit_rate']:.1%} + prefetch {s['prefetch_hit_rate']:.1%}) "
          f"resident={s['resident_gb']:.2f} GB")
    print(f"[stream] disk: critical_read={s['bytes_read_gb']:.1f} GB "
          f"prefetched={s['bytes_prefetched_gb']:.1f} GB total={s['bytes_total_gb']:.1f} GB "
          f"| prefetched_experts={s['prefetched']} dropped_unused={s['prefetch_dropped']}")
    print(f"[stream] coalescing: {s['expert_reads']} expert-loads in {s['read_runs']} "
          f"range-reads = {s['experts_per_read']:.2f} experts/read")

    if args.use_ane:
        from .ane_loader import ane_stats
        a = ane_stats()
        print(f"[ANE] attention: {a['ane']} ANE calls ({a['ane_rate']:.1%}) "
              f"/ {a['fallback']} GPU fallback / {a['total']} total")

    # Persist this session's routing histogram for the next run's warmup. Same
    # path is both load source and dump destination; the dump REPLACES the file
    # (last-session-wins) rather than merging counts.
    if args.warmup_file:
        n = cache.dump_histogram(
            args.warmup_file,
            model_id=args.model,
            k=args.max_active_experts,
        )
        print(f"[stream] histogram saved: {n} (layer,expert) pairs -> {args.warmup_file}")


if __name__ == "__main__":
    main()
