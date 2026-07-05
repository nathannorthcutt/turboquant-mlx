#!/usr/bin/env python3
"""tq — single entry point for turboquant-mlx streaming inference.

One script bootstraps the model (download + all repacks + warmup) and runs
inference. All state lives under a stable root (default ``~/.turboquant_mlx``):

    ~/.turboquant_mlx/
      config.json                    persisted defaults (model_id, paths, flags)
      models/<slug>/                 downloaded model + repack companions
        model_freqsorted/            freq-sorted variant (+ perm.json)
      warmup/<slug>.json             routing histogram
      perf/<slug>-<ts>.json          per-run perf log
      perf/<slug>-<ts>-misses.jsonl  per-miss trace (only with --profile)

Usage:
    python scripts/tq.py setup                        # one-time bootstrap
    python scripts/tq.py run "Why is the sky blue?"   # inference
    python scripts/tq.py benchmark                    # standard prompt suite
    python scripts/tq.py config show                  # inspect defaults
    python scripts/tq.py profile-report <perf.json>   # read a saved perf log

Design note: every turboquant_mlx / mlx import is LAZY (inside a function), so
this file parses and ``config``/``profile-report`` run on a machine that has not
installed the package yet — that is what lets ``setup`` do the install first.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
DEFAULT_ROOT = os.path.expanduser("~/.turboquant_mlx")
DEFAULT_MODEL = "manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32"

# setup / run defaults (also documented in each subcommand's --help)
DEF_MAX_TOKENS = 512
DEF_TEMP = 0.7
def _default_cache_budget_gb() -> float:
    """Scale hot-tier budget against iogpu.wired_limit_mb (the actual Metal ceiling).

    For Qwen3-235B the non-streaming resident weights (attention Q/K/V/O across
    94 layers + embeddings) consume ~28 GB of wired Metal memory. The expert
    streaming cache must fit in the remainder, leaving at least 6 GB for Metal
    compute buffers, KV cache, and command-buffer overhead.
    """
    try:
        import subprocess
        wired_mb = int(subprocess.check_output(
            ["sysctl", "-n", "iogpu.wired_limit_mb"], stderr=subprocess.DEVNULL
        ).strip())
        wired_gb = wired_mb / 1024
    except Exception:
        return 8.0  # safe fallback
    # 28 GB resident estimate + 6 GB compute headroom; remainder is expert cache.
    RESIDENT_ESTIMATE_GB = 28.0
    COMPUTE_HEADROOM_GB  = 6.0
    budget = wired_gb - RESIDENT_ESTIMATE_GB - COMPUTE_HEADROOM_GB
    return round(max(4.0, min(budget, 20.0)), 1)

DEF_CACHE_BUDGET_GB = _default_cache_budget_gb()
DEF_K = 4

WARMUP_PROMPT = "The key insight in transformer attention is"
WARMUP_TOKENS = 64

BENCH_MAX_TOKENS = 256
STANDARD_PROMPTS = [
    "Explain the key insight behind attention in transformers.",
    "Write a Python function that finds all prime numbers up to N using the "
    "Sieve of Eratosthenes.",
    "What are the trade-offs between microservices and monolithic architectures?",
    "Derive the quadratic formula from ax^2 + bx + c = 0.",
    "Summarize the plot of Hamlet in three sentences.",
]

# companion-file prefixes written by the repack tools; excluded when we look for
# the model's own shards.
_COMPANION_PREFIXES = ("model_wts", "model_fused", "model_s8")
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

# status glyphs
_TICK = "✓"   # ✓
_ARROW = "→"  # →


# --------------------------------------------------------------------------
# tiny status helpers
# --------------------------------------------------------------------------
def _skip(msg: str) -> None:
    print(f"[{_TICK}] {msg} (already done)")


def _start(msg: str) -> None:
    print(f"[{_ARROW}] {msg} …", flush=True)


def _done(msg: str) -> None:
    print(f"[{_TICK}] {msg}")


# --------------------------------------------------------------------------
# path / slug helpers
# --------------------------------------------------------------------------
def slugify(model_id: str) -> str:
    """``manjunathshiva/Qwen3-235B-A22B-Instruct-2507-tq3a-tq2e-g32`` -> ``manjunathshiva-qwen3-235b-a22b-instruct-2507-tq3a-tq2e-g32``.

    ``/`` becomes ``-``; ``_`` is preserved; everything is lowercased.
    """
    return model_id.replace("/", "-").lower()


def root_dir() -> str:
    return os.environ.get("TURBOQUANT_MLX_HOME", DEFAULT_ROOT)


def _p(*parts: str) -> str:
    return os.path.join(root_dir(), *parts)


def config_path() -> str:
    return _p("config.json")


def model_dir(slug: str) -> str:
    return _p("models", slug)


def freqsorted_dir(slug: str) -> str:
    return os.path.join(model_dir(slug), "model_freqsorted")


def warmup_path(slug: str) -> str:
    return _p("warmup", f"{slug}.json")


def perf_dir() -> str:
    return _p("perf")


def _ensure_dirs() -> None:
    for d in ("models", "warmup", "perf"):
        os.makedirs(_p(d), exist_ok=True)


def repo_root() -> str:
    """Repo root = the directory that holds ``pyproject.toml`` (and ``mlx-lm/``).

    ``scripts/tq.py`` normally lives one level below it; walk up as a fallback so
    the script keeps working if it is moved.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "pyproject.toml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(here)  # best-effort: parent of scripts/


def _model_shards(directory: str) -> list[str]:
    """The model's own weight shards in ``directory`` (companions excluded)."""
    out = []
    for f in glob.glob(os.path.join(glob.escape(directory), "model*.safetensors")):
        base = os.path.basename(f)
        if any(base.startswith(pfx) for pfx in _COMPANION_PREFIXES):
            continue
        out.append(f)
    return sorted(out)


def _has_companion(directory: str, prefix: str) -> bool:
    return bool(glob.glob(os.path.join(glob.escape(directory), f"{prefix}*.safetensors")))


def _layer_of(key: str):
    m = _LAYER_RE.search(key)
    return int(m.group(1)) if m else -1


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def load_config() -> dict:
    try:
        with open(config_path()) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def save_config(cfg: dict) -> None:
    _ensure_dirs()
    with open(config_path(), "w") as f:
        json.dump(cfg, f, indent=2)


# --------------------------------------------------------------------------
# profiler (deep instrumentation, --profile only)
# --------------------------------------------------------------------------
class Profiler:
    """Near-zero-overhead-when-absent instrumentation for a streaming run.

    Nothing here runs unless ``--profile`` constructs a Profiler and calls
    ``attach`` / ``on_token``; the run path guards every call behind
    ``if profiler:``. When attached it monkeypatches three bound methods on the
    ``ExpertCache`` instance (``on_layer_start``, ``gather``, ``gather_fused``)
    to time the disk / gpu / routing phases and trace misses. The wrappers add a
    few ``perf_counter`` calls (~1 µs each) per layer per token.

    Several signals read ExpertCache internals (``_od``, ``_pinned``,
    ``_staging``, ``_inflight``, ``misses``, ``prefetch_hits``). They are
    best-effort approximations for diagnostics, not exact accounting.
    """

    def __init__(self, misses_path: str | None = None):
        self.misses_path = misses_path
        self._fh = open(misses_path, "w") if misses_path else None
        self.token_idx = 0

        # per-decoder-layer latency samples (ms), one per layer per token
        self.layer_latencies: list[float] = []
        self._last_layer_ts = None

        # per-token phase breakdown (running sums in seconds)
        self.t_disk = 0.0     # blocked on critical-path pread (misses only)
        self.t_gpu = 0.0      # between a gather returning and the next starting
        self.t_routing = 0.0  # inside the routing/prefetch remap (on_layer_start)
        self._last_gather_end = None

        # mlx memory samples every 10 tokens: [token, active_bytes, peak_bytes]
        self.memory_timeline: list[list[int]] = []

        # prefetch rescue rate in 50-token windows
        self._win = 50
        self._win_start = 0
        self._win_prefetch0 = 0
        self._win_miss0 = 0
        self.prefetch_rescue_timeline: list[list] = []

        self._cache = None

    # -- lifecycle -------------------------------------------------------
    def attach(self, cache) -> None:
        self._cache = cache
        self._win_prefetch0 = cache.prefetch_hits
        self._win_miss0 = cache.misses

        orig_ols = cache.on_layer_start
        orig_gather = cache.gather
        orig_fused = cache.gather_fused

        def wrapped_ols(layer_idx, experts):
            now = time.perf_counter()
            if self._last_layer_ts is not None:
                self.layer_latencies.append((now - self._last_layer_ts) * 1000.0)
            self._last_layer_ts = now
            t = time.perf_counter()
            r = orig_ols(layer_idx, experts)
            self.t_routing += time.perf_counter() - t
            return r

        def wrapped_gather(wkey, skey, experts):
            self._pre_gather()
            missed = [e for e in experts if self._would_miss(wkey, e)]
            m0 = self._cache.misses
            t = time.perf_counter()
            r = orig_gather(wkey, skey, experts)
            dt = time.perf_counter() - t
            self._post_gather(wkey, dt, missed, self._cache.misses - m0)
            return r

        def wrapped_fused(fwkey, fskey, layer_idx, experts):
            self._pre_gather()
            missed = [e for e in experts if self._would_miss(fwkey, e)]
            m0 = self._cache.misses
            t = time.perf_counter()
            r = orig_fused(fwkey, fskey, layer_idx, experts)
            dt = time.perf_counter() - t
            self._post_gather(fwkey, dt, missed, self._cache.misses - m0)
            return r

        cache.on_layer_start = wrapped_ols
        cache.gather = wrapped_gather
        cache.gather_fused = wrapped_fused

    def _pre_gather(self) -> None:
        now = time.perf_counter()
        if self._last_gather_end is not None:
            self.t_gpu += now - self._last_gather_end

    def _post_gather(self, wkey, dt, missed, delta_misses) -> None:
        if delta_misses > 0:
            self.t_disk += dt
            if self._fh is not None and missed:
                lat_ms = round(dt * 1000.0 / len(missed), 3)
                lyr = _layer_of(wkey)
                for e in missed:
                    self._fh.write(
                        json.dumps([self.token_idx, lyr, int(e), lat_ms]) + "\n"
                    )
        self._last_gather_end = time.perf_counter()

    def _would_miss(self, wkey, e) -> bool:
        c = self._cache
        ck = (wkey, e)
        return not (
            ck in c._od or ck in c._pinned
            or ck in c._staging or ck in c._inflight
        )

    def on_token(self, token_idx: int) -> None:
        """Called once per generated (yielded) token from the run loop."""
        self.token_idx = token_idx + 1  # gathers for the NEXT token log under this
        c = self._cache
        if token_idx % 10 == 0:
            import mlx.core as mx
            self.memory_timeline.append(
                [token_idx, int(mx.get_active_memory()), int(mx.get_peak_memory())]
            )
        if token_idx > 0 and token_idx % self._win == 0:
            dp = c.prefetch_hits - self._win_prefetch0
            dm = c.misses - self._win_miss0
            rate = dp / (dp + dm) if (dp + dm) > 0 else 0.0
            self.prefetch_rescue_timeline.append([self._win_start, round(rate, 4)])
            self._win_start = token_idx
            self._win_prefetch0 = c.prefetch_hits
            self._win_miss0 = c.misses

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # -- reporting -------------------------------------------------------
    def _pct(self, p: float) -> float:
        lat = self._sorted_lat
        if not lat:
            return 0.0
        k = min(len(lat) - 1, int(round((p / 100.0) * (len(lat) - 1))))
        return round(lat[k], 3)

    def summary(self) -> dict:
        self._sorted_lat = sorted(self.layer_latencies)
        ntok = max(1, self.token_idx)
        return {
            "layer_latency_ms": {
                "p50": self._pct(50), "p95": self._pct(95),
                "p99": self._pct(99), "samples": len(self._sorted_lat),
            },
            "phase_mean_ms_per_token": {
                "t_disk": round(self.t_disk * 1000.0 / ntok, 3),
                "t_gpu": round(self.t_gpu * 1000.0 / ntok, 3),
                "t_routing": round(self.t_routing * 1000.0 / ntok, 3),
            },
            "note": "phase/miss figures are approximate diagnostics "
                    "(read from ExpertCache internals)",
        }

    def print_summary(self) -> None:
        s = self.summary()
        ll = s["layer_latency_ms"]
        ph = s["phase_mean_ms_per_token"]
        print(f"[profile] layer latency ms  p50={ll['p50']} p95={ll['p95']} "
              f"p99={ll['p99']} (n={ll['samples']})")
        print(f"[profile] per-token phase ms t_disk={ph['t_disk']} "
              f"t_gpu={ph['t_gpu']} t_routing={ph['t_routing']}")
        if self.prefetch_rescue_timeline:
            last = self.prefetch_rescue_timeline[-1]
            print(f"[profile] prefetch rescue windows: {len(self.prefetch_rescue_timeline)} "
                  f"(latest {last[1]:.1%} from token {last[0]})")


# --------------------------------------------------------------------------
# generation core (shared by run / benchmark / warmup)
# --------------------------------------------------------------------------
def _rss_gb() -> float:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
        return int(out) / 1024 / 1024
    except Exception:
        return 0.0


def _resolve_model_paths(model_id: str, use_freqsort: bool):
    """Pick the on-disk model path + perm file for a run.

    Freq-sort reorders only the expert stacks, so the router still emits *logical*
    ids — the reader must translate via ``perm.json`` (written next to the
    reordered shards by ``setup``). Returns ``(path, perm_path, used_freqsort)``.
    """
    slug = slugify(model_id)
    fq = freqsorted_dir(slug)
    perm = os.path.join(fq, "perm.json")
    if use_freqsort and _model_shards(fq) and os.path.isfile(perm):
        return fq, perm, True
    md = model_dir(slug)
    if _model_shards(md):
        return md, None, False
    # Not downloaded locally — fall back to the HF id (load_streaming resolves it).
    return model_id, None, False


def run_generation(model_id: str, prompt: str, *, max_tokens: int, temp: float,
                   cache_budget_gb: float, k: int, use_ane: bool,
                   use_freqsort: bool, chat_template: bool,
                   profiler: Profiler | None = None,
                   warmup_write: bool = True, verbose: bool = True) -> dict:
    """Load the streaming model, generate, and return a perf record dict.

    Imports are local so this module stays importable before ``setup`` installs
    the package. When ``profiler`` is given, it is attached to the expert cache
    for the duration of the run.
    """
    import mlx.core as mx
    import mlx.core.metal as mx_metal
    import turboquant_mlx.compat  # noqa: F401  (registers mlx-lm compat shims)
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from turboquant_mlx.stream.loader import load_streaming

    # Check iogpu.wired_limit_mb — on Apple Silicon this caps how much memory
    # Metal can wire for GPU use (default ~75% of RAM). Large models need it raised.
    try:
        import subprocess as _sp
        phys_bytes = int(
            _sp.check_output(["sysctl", "-n", "hw.memsize"],
                             stderr=_sp.DEVNULL).strip()
        )
        phys_gb = phys_bytes / 1024**3
        wired_mb = int(
            _sp.check_output(["sysctl", "-n", "iogpu.wired_limit_mb"],
                             stderr=_sp.DEVNULL).strip()
        )
        # Attention layers alone are ~28 GB; add expert cache + overhead.
        # Warn when wired headroom is tight (< 90% of RAM).
        recommended_mb = int(phys_gb * 0.92 * 1024)
        if wired_mb < recommended_mb:
            print(f"\n  [!] iogpu.wired_limit_mb={wired_mb} may be too low for this model.")
            print(f"      Recommended: sudo sysctl iogpu.wired_limit_mb={recommended_mb}")
            print(f"      To persist across reboots: add that line to /etc/sysctl.conf\n")
        # Set MLX's own limit just below the wired cap so allocations fail fast.
        mx.set_memory_limit(wired_mb * 1024 * 1024)
        mx.set_cache_memory_limit(int(phys_bytes * 0.03))
    except Exception:
        pass

    slug = slugify(model_id)
    wpath = warmup_path(slug)
    model_path, perm_path, used_freqsort = _resolve_model_paths(model_id, use_freqsort)

    # warmup_gb: pre-load hot experts up to the cache budget so we start near
    # steady state. load_streaming disables warmup when warmup_gb == 0, so only
    # pass a positive value once a histogram exists.
    warmup_gb = cache_budget_gb if os.path.isfile(wpath) else 0.0

    t0 = time.time()
    model, tok, cache = load_streaming(
        model_path, cache_budget_gb=cache_budget_gb, max_active_experts=k,
        use_page_cache=None, warmup_file=wpath, warmup_gb=warmup_gb,
        perm_path=perm_path, use_ane=use_ane,
    )
    load_sec = time.time() - t0
    print(f"[tq] loaded in {load_sec:.1f}s | resident RSS={_rss_gb():.2f} GB")
    # Force-evaluate all resident model parameters into Metal memory now, before
    # inference, so we can measure the true wired footprint and detect OOM early
    # with a useful message rather than a cryptic Metal command-buffer failure.
    params = model.parameters()
    flat_params = []
    def _flatten(x):
        if isinstance(x, mx.array):
            flat_params.append(x)
        elif isinstance(x, dict):
            for v in x.values(): _flatten(v)
        elif isinstance(x, (list, tuple)):
            for v in x: _flatten(v)
    _flatten(params)
    mx.eval(*flat_params)
    try:
        metal_active = mx.metal.get_active_memory() / 1024**3
        metal_peak   = mx.metal.get_peak_memory()   / 1024**3
        metal_cache  = mx.metal.get_cache_memory()  / 1024**3
        print(f"[tq] Metal memory — active={metal_active:.2f} GB  peak={metal_peak:.2f} GB  cache={metal_cache:.2f} GB")
    except Exception:
        pass

    text_prompt = prompt
    if chat_template and hasattr(tok, "apply_chat_template"):
        text_prompt = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True
        )

    if profiler is not None:
        profiler.attach(cache)

    sampler = make_sampler(temp=temp)
    text = ""
    ttft = None
    prompt_tokens = 0
    gen_tokens = 0
    t_start = time.perf_counter()
    for i, resp in enumerate(stream_generate(
            model, tok, text_prompt, max_tokens=max_tokens, sampler=sampler)):
        if ttft is None:
            ttft = time.perf_counter() - t_start
        text += resp.text
        prompt_tokens = resp.prompt_tokens
        gen_tokens = resp.generation_tokens
        if verbose:
            print(resp.text, end="", flush=True)
        if profiler is not None:
            profiler.on_token(i)
    total_sec = time.perf_counter() - t_start
    if verbose:
        print()

    s = cache.stats()
    tok_per_sec = (gen_tokens / total_sec) if total_sec > 0 else 0.0

    flags = []
    if getattr(cache.reader, "has_fused_gate_up", False):
        flags.append("fused_gate_up")
    if getattr(cache.reader, "has_interleaved", False):
        flags.append("interleaved")
    if used_freqsort:
        flags.append("freq_sorted")

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_id": model_id,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": gen_tokens,
        "tok_per_sec": round(tok_per_sec, 3),
        "ttft_sec": round(ttft or 0.0, 3),
        "total_sec": round(total_sec, 3),
        "load_sec": round(load_sec, 3),
        "peak_rss_gb": round(_rss_gb(), 3),
        "peak_mlx_gb": round(mx.get_peak_memory() / 1e9, 3),
        "cache_hit_rate": round(s["cache_hit_rate"], 4),
        "prefetch_hit_rate": round(s["prefetch_hit_rate"], 4),
        "resident_gb": round(s["resident_gb"], 3),
        "bytes_read_gb": round(s["bytes_read_gb"], 3),
        "experts_per_read": round(s["experts_per_read"], 3),
        "use_ane": use_ane,
        "use_freqsort": used_freqsort,
        "cache_budget_gb": cache_budget_gb,
        "k": k,
        "flags": flags,
    }

    print(f"[tq] {gen_tokens} tok in {total_sec:.1f}s = {tok_per_sec:.1f} tok/s "
          f"| ttft={record['ttft_sec']:.2f}s | hit_rate={s['hit_rate']:.1%} "
          f"| peak_mlx={record['peak_mlx_gb']:.1f} GB")

    if profiler is not None:
        record["profile"] = profiler.summary()
        record["memory_timeline"] = profiler.memory_timeline
        record["prefetch_rescue_timeline"] = profiler.prefetch_rescue_timeline
        profiler.print_summary()

    # Persist this session's routing histogram for the next run's warmup.
    if warmup_write:
        n = cache.dump_histogram(wpath, model_id=model_path, k=k)
        print(f"[tq] histogram saved: {n} (layer,expert) pairs -> {wpath}")

    return record


def _write_perf(slug: str, payload) -> str:
    _ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(perf_dir(), f"{slug}-{ts}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _misses_path_for(slug: str) -> str:
    _ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(perf_dir(), f"{slug}-{ts}-misses.jsonl")


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
def _step_install() -> None:
    if (importlib.util.find_spec("turboquant_mlx") is not None
            and importlib.util.find_spec("mlx_lm") is not None):
        _skip("dependencies installed")
        return
    _start("installing dependencies (huggingface_hub, mlx-lm, turboquant-mlx)")
    root = repo_root()
    # huggingface_hub >= 1.0 ships the `hf` CLI that replaced `huggingface-cli`.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub"])
    mlxlm = os.path.join(root, "mlx-lm")
    if os.path.isdir(mlxlm):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", mlxlm])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", root])
    _done("dependencies installed")


def _step_download(model_id: str, slug: str) -> None:
    md = model_dir(slug)
    if _model_shards(md):
        _skip(f"model downloaded ({md})")
        return
    _start(f"downloading {model_id} -> {md}")
    os.makedirs(md, exist_ok=True)
    # Use the `hf` CLI (huggingface_hub >= 1.0). It picks up the token from
    # `hf login` automatically and handles resumable downloads.
    try:
        subprocess.check_call(["hf", "download", model_id, "--local-dir", md])
    except FileNotFoundError:
        print(
            "[tq] The `hf` CLI is required to download models.\n"
            "     Install / upgrade with:  pip install -U huggingface_hub\n"
            "     Then authenticate with:  hf login",
            file=sys.stderr,
        )
        raise SystemExit(2)
    _done(f"model downloaded ({md})")


def _step_interleaved(slug: str) -> None:
    md = model_dir(slug)
    if _has_companion(md, "model_wts"):
        _skip("interleaved repack (model_wts-*)")
        return
    _start("interleaved repack (weight+scales -> 1 pread/expert)")
    from turboquant_mlx.stream import repack_interleaved as ri
    total = 0
    for shard in _model_shards(md):
        dst = os.path.join(md, ri._companion_name(os.path.basename(shard)))
        n_pairs, _ = ri._build_interleaved_shard(shard, dst, include_resident=False)
        if n_pairs == 0 and os.path.exists(dst):
            os.remove(dst)
        total += n_pairs
    _done(f"interleaved repack ({total} pairs)")


def _step_fused(slug: str) -> None:
    md = model_dir(slug)
    if _has_companion(md, "model_fused"):
        _skip("fused gate+up repack (model_fused-*)")
        return
    _start("fused gate+up repack")
    from turboquant_mlx.stream import repack_fused_gate_up as rf
    total = 0
    for shard in _model_shards(md):
        dst = os.path.join(md, rf._companion_name(os.path.basename(shard)))
        n_layers, _ = rf._build_fused_shard(shard, dst)
        if n_layers == 0 and os.path.exists(dst):
            os.remove(dst)
        total += n_layers
    _done(f"fused gate+up repack ({total} layers)")


def _step_warmup(model_id: str, slug: str, cache_budget_gb: float, k: int) -> None:
    wpath = warmup_path(slug)
    if os.path.isfile(wpath):
        _skip("warmup histogram")
        return
    _start(f"warmup run ({WARMUP_TOKENS} tokens) to build routing histogram")
    _ensure_dirs()
    # Run in a subprocess so a Metal OOM abort (C++ exception, uncatchable in Python)
    # doesn't kill the setup process — Metal crashes call std::terminate() which
    # bypasses Python exception handling entirely.
    import subprocess as _sp
    import sys as _sys
    cmd = [
        _sys.executable, "-c",
        (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "from scripts.tq import run_generation;"
            "run_generation(sys.argv[2], sys.argv[3],"
            " max_tokens=int(sys.argv[4]), temp=0.0,"
            " cache_budget_gb=float(sys.argv[5]), k=int(sys.argv[6]),"
            " use_ane=False, use_freqsort=False, chat_template=False,"
            " warmup_write=True, verbose=True)"
        ),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        model_id, WARMUP_PROMPT,
        str(WARMUP_TOKENS), str(cache_budget_gb), str(k),
    ]
    result = _sp.run(cmd, capture_output=False)
    if result.returncode == 0 and os.path.isfile(wpath):
        _done("warmup histogram")
    else:
        print(f"\n  [!] warmup failed (exit {result.returncode}) — skipping histogram.")
        print(f"      Inference will still work; re-run setup after closing other apps to retry.")


def _step_freqsort(slug: str) -> None:
    fq = freqsorted_dir(slug)
    if _model_shards(fq) and os.path.isfile(os.path.join(fq, "perm.json")):
        _skip("freq-sort repack (model_freqsorted/)")
        return
    wpath = warmup_path(slug)
    if not os.path.isfile(wpath):
        _skip("freq-sort repack (no histogram yet — re-run setup once warmup succeeds)")
        return
    _start("freq-sort repack (hot experts -> low file offsets)")
    md = model_dir(slug)
    import shutil
    from turboquant_mlx.stream import repack as rp

    with open(wpath) as f:
        hist = json.load(f)

    # Reorder ALL model*.safetensors including companions so the freqsorted dir
    # is self-contained and its companions stay consistent with the reorder.
    shards = sorted(glob.glob(os.path.join(glob.escape(md), "model*.safetensors")))
    experts_per_layer = rp._experts_per_layer(
        [s for s in shards if not any(
            os.path.basename(s).startswith(p) for p in _COMPANION_PREFIXES)])
    perm_by_layer = rp._freq_perm_by_layer(hist, experts_per_layer)

    os.makedirs(fq, exist_ok=True)
    # Copy non-weight files (config, tokenizer, index) into the freqsorted dir.
    for name in os.listdir(md):
        s = os.path.join(md, name)
        if name.endswith(".safetensors") or not os.path.isfile(s):
            continue
        shutil.copy2(s, os.path.join(fq, name))

    metadata = {"repacked": "true", "freq_sort": "true"}
    total = 0
    for shard in shards:
        dst = os.path.join(fq, os.path.basename(shard))
        _, n_reordered, _ = rp._repack_shard(shard, dst, perm_by_layer, metadata)
        total += n_reordered

    # The router still emits logical ids; write the derived perm so `run` can
    # pass perm_path to translate logical -> physical at read time.
    with open(os.path.join(fq, "perm.json"), "w") as f:
        json.dump({"perm": perm_by_layer}, f)
    _done(f"freq-sort repack ({total} expert stacks reordered)")


def cmd_setup(args) -> int:
    cfg = load_config()
    model_id = args.model or cfg.get("model_id") or DEFAULT_MODEL
    cache_budget_gb = (args.cache_budget_gb if args.cache_budget_gb is not None
                       else cfg.get("cache_budget_gb", DEF_CACHE_BUDGET_GB))
    k = args.k if args.k is not None else cfg.get("k", DEF_K)
    slug = slugify(model_id)
    _ensure_dirs()

    # Persist the model id so later `run` with no --model uses the same one.
    cfg.setdefault("model_id", model_id)
    save_config(cfg)

    print(f"[tq] setup for {model_id}\n     root={root_dir()}\n     slug={slug}\n")
    _step_install()
    _step_download(model_id, slug)
    _step_interleaved(slug)
    _step_fused(slug)
    _step_warmup(model_id, slug, cache_budget_gb, k)
    _step_freqsort(slug)
    print("\n[tq] setup complete. Run:  python scripts/tq.py run \"Your prompt\"")
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def _resolve_run_opts(args, cfg):
    model_id = args.model or cfg.get("model_id") or DEFAULT_MODEL
    max_tokens = (args.max_tokens if args.max_tokens is not None
                  else cfg.get("max_tokens", DEF_MAX_TOKENS))
    temp = args.temp if args.temp is not None else cfg.get("temp", DEF_TEMP)
    cache_budget_gb = (args.cache_budget_gb if args.cache_budget_gb is not None
                       else cfg.get("cache_budget_gb", DEF_CACHE_BUDGET_GB))
    k = args.k if args.k is not None else cfg.get("k", DEF_K)
    use_ane = args.use_ane if args.use_ane is not None else cfg.get("use_ane", False)
    return model_id, max_tokens, temp, cache_budget_gb, k, use_ane


def cmd_run(args) -> int:
    cfg = load_config()
    model_id, max_tokens, temp, cache_budget_gb, k, use_ane = _resolve_run_opts(args, cfg)
    slug = slugify(model_id)

    profiler = None
    if args.profile:
        profiler = Profiler(misses_path=_misses_path_for(slug))

    try:
        record = run_generation(
            model_id, args.prompt, max_tokens=max_tokens, temp=temp,
            cache_budget_gb=cache_budget_gb, k=k, use_ane=use_ane,
            use_freqsort=not args.no_freqsort,
            chat_template=not args.no_chat_template, profiler=profiler,
        )
    finally:
        if profiler is not None:
            profiler.close()

    if args.perf_log:
        path = _write_perf(slug, record)
        print(f"[tq] perf log -> {path}")
        if profiler is not None and profiler.misses_path:
            print(f"[tq] miss trace -> {profiler.misses_path}")
    return 0


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------
def _mean_std(xs):
    import statistics
    if not xs:
        return 0.0, 0.0
    m = statistics.mean(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, sd


def cmd_benchmark(args) -> int:
    cfg = load_config()
    model_id = args.model or cfg.get("model_id") or DEFAULT_MODEL
    cache_budget_gb = (args.cache_budget_gb if args.cache_budget_gb is not None
                       else cfg.get("cache_budget_gb", DEF_CACHE_BUDGET_GB))
    k = args.k if args.k is not None else cfg.get("k", DEF_K)
    use_ane = args.use_ane if args.use_ane is not None else cfg.get("use_ane", False)
    slug = slugify(model_id)

    records = []
    for i, prompt in enumerate(STANDARD_PROMPTS, 1):
        print(f"\n{'=' * 64}\n[tq] benchmark {i}/{len(STANDARD_PROMPTS)}: {prompt[:56]}…\n"
              f"{'=' * 64}")
        profiler = Profiler(misses_path=_misses_path_for(slug)) if args.profile else None
        try:
            rec = run_generation(
                model_id, prompt, max_tokens=BENCH_MAX_TOKENS, temp=DEF_TEMP,
                cache_budget_gb=cache_budget_gb, k=k, use_ane=use_ane,
                use_freqsort=not args.no_freqsort, chat_template=True,
                profiler=profiler, verbose=False,
            )
        finally:
            if profiler is not None:
                profiler.close()
                rec["miss_trace"] = profiler.misses_path
        records.append(rec)

    tps = [r["tok_per_sec"] for r in records]
    ttft = [r["ttft_sec"] for r in records]
    hit = [r["cache_hit_rate"] for r in records]

    print(f"\n{'=' * 64}\n[tq] benchmark summary ({model_id})\n{'=' * 64}")
    print(f"{'#':<3}{'tok/s':>9}{'ttft_s':>9}{'hit%':>8}  prompt")
    for i, r in enumerate(records, 1):
        print(f"{i:<3}{r['tok_per_sec']:>9.2f}{r['ttft_sec']:>9.2f}"
              f"{r['cache_hit_rate'] * 100:>8.1f}  {r['prompt'][:44]}")
    m_tps, s_tps = _mean_std(tps)
    m_ttft, s_ttft = _mean_std(ttft)
    m_hit, s_hit = _mean_std(hit)
    print("-" * 64)
    print(f"{'avg':<3}{m_tps:>9.2f}{m_ttft:>9.2f}{m_hit * 100:>8.1f}  "
          f"(stddev tok/s={s_tps:.2f} ttft={s_ttft:.2f} hit={s_hit * 100:.1f})")

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_id": model_id,
        "max_tokens": BENCH_MAX_TOKENS,
        "aggregate": {
            "tok_per_sec": {"mean": round(m_tps, 3), "stddev": round(s_tps, 3)},
            "ttft_sec": {"mean": round(m_ttft, 3), "stddev": round(s_ttft, 3)},
            "cache_hit_rate": {"mean": round(m_hit, 4), "stddev": round(s_hit, 4)},
        },
        "runs": records,
    }
    path = _write_perf(slug, payload)
    print(f"\n[tq] benchmark perf log -> {path}")
    return 0


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
_CONFIG_KEYS = ("model_id", "cache_budget_gb", "k", "max_tokens", "temp", "use_ane")


def cmd_config(args) -> int:
    cfg = load_config()
    updates = {}
    if args.model is not None:
        updates["model_id"] = args.model
    if args.cache_budget_gb is not None:
        updates["cache_budget_gb"] = args.cache_budget_gb
    if args.k is not None:
        updates["k"] = args.k
    if args.max_tokens is not None:
        updates["max_tokens"] = args.max_tokens
    if args.temp is not None:
        updates["temp"] = args.temp
    if args.use_ane is not None:
        updates["use_ane"] = args.use_ane

    if args.subcommand == "show" or not updates:
        print(f"[tq] config ({config_path()})")
        if not cfg:
            print("     (empty — all defaults in effect)")
        merged = {
            "model_id": cfg.get("model_id", DEFAULT_MODEL),
            "cache_budget_gb": cfg.get("cache_budget_gb", DEF_CACHE_BUDGET_GB),
            "k": cfg.get("k", DEF_K),
            "max_tokens": cfg.get("max_tokens", DEF_MAX_TOKENS),
            "temp": cfg.get("temp", DEF_TEMP),
            "use_ane": cfg.get("use_ane", False),
        }
        for key, val in merged.items():
            src = "set" if key in cfg else "default"
            print(f"     {key:<16} = {val}   ({src})")
        return 0

    cfg.update(updates)
    save_config(cfg)
    print(f"[tq] config updated ({config_path()}):")
    for key, val in updates.items():
        print(f"     {key} = {val}")
    return 0


# --------------------------------------------------------------------------
# profile-report (pure stdlib — no mlx)
# --------------------------------------------------------------------------
def _print_run_report(r: dict) -> None:
    print(f"  model         : {r.get('model_id')}")
    print(f"  timestamp     : {r.get('timestamp')}")
    print(f"  prompt        : {str(r.get('prompt', ''))[:60]}")
    print(f"  tokens        : {r.get('prompt_tokens')} prompt / "
          f"{r.get('generated_tokens')} generated")
    print(f"  throughput    : {r.get('tok_per_sec')} tok/s   "
          f"ttft {r.get('ttft_sec')}s   total {r.get('total_sec')}s")
    print(f"  memory        : peak_rss {r.get('peak_rss_gb')} GB   "
          f"peak_mlx {r.get('peak_mlx_gb')} GB   resident {r.get('resident_gb')} GB")
    print(f"  cache         : hit {r.get('cache_hit_rate')}   "
          f"prefetch {r.get('prefetch_hit_rate')}   "
          f"experts/read {r.get('experts_per_read')}   "
          f"read {r.get('bytes_read_gb')} GB")
    print(f"  flags         : {', '.join(r.get('flags', [])) or '(none)'}  "
          f"| freqsort={r.get('use_freqsort')} ane={r.get('use_ane')} "
          f"k={r.get('k')} budget={r.get('cache_budget_gb')} GB")

    prof = r.get("profile")
    if prof:
        ll = prof.get("layer_latency_ms", {})
        ph = prof.get("phase_mean_ms_per_token", {})
        print("  --- profile ---")
        print(f"  layer latency : p50 {ll.get('p50')} ms  p95 {ll.get('p95')} ms  "
              f"p99 {ll.get('p99')} ms  (n={ll.get('samples')})")
        print(f"  phase / token : disk {ph.get('t_disk')} ms  gpu {ph.get('t_gpu')} ms  "
              f"routing {ph.get('t_routing')} ms")
    mt = r.get("memory_timeline")
    if mt:
        peak = max(row[2] for row in mt)
        print(f"  mem timeline  : {len(mt)} samples, peak {peak / 1e9:.2f} GB")
    pr = r.get("prefetch_rescue_timeline")
    if pr:
        rates = [w[1] for w in pr]
        print(f"  rescue windows: {len(pr)} windows, "
              f"rate min {min(rates):.1%} max {max(rates):.1%} "
              f"last {rates[-1]:.1%}")


def cmd_profile_report(args) -> int:
    try:
        with open(args.perf_file) as f:
            data = json.load(f)
    except (FileNotFoundError, ValueError) as e:
        print(f"[tq] cannot read perf file: {e}", file=sys.stderr)
        return 2

    print("=" * 64)
    print(f"perf report: {args.perf_file}")
    print("=" * 64)

    if isinstance(data, dict) and "runs" in data:
        # benchmark payload
        agg = data.get("aggregate", {})
        print(f"benchmark: {data.get('model_id')}  ({len(data['runs'])} prompts, "
              f"max_tokens={data.get('max_tokens')})")
        print("-" * 64)
        print(f"{'#':<3}{'tok/s':>9}{'ttft_s':>9}{'hit%':>8}  prompt")
        for i, r in enumerate(data["runs"], 1):
            print(f"{i:<3}{r.get('tok_per_sec', 0):>9.2f}{r.get('ttft_sec', 0):>9.2f}"
                  f"{r.get('cache_hit_rate', 0) * 100:>8.1f}  {str(r.get('prompt',''))[:44]}")
        print("-" * 64)
        if agg:
            t = agg.get("tok_per_sec", {})
            tt = agg.get("ttft_sec", {})
            h = agg.get("cache_hit_rate", {})
            print(f"{'avg':<3}{t.get('mean', 0):>9.2f}{tt.get('mean', 0):>9.2f}"
                  f"{h.get('mean', 0) * 100:>8.1f}  "
                  f"(stddev tok/s={t.get('stddev', 0):.2f} "
                  f"ttft={tt.get('stddev', 0):.2f})")
        if any(r.get("profile") for r in data["runs"]):
            print("\nper-prompt profile:")
            for i, r in enumerate(data["runs"], 1):
                print(f"\n[{i}] {str(r.get('prompt',''))[:56]}")
                _print_run_report(r)
    elif isinstance(data, list):
        for i, r in enumerate(data, 1):
            print(f"\n[{i}]")
            _print_run_report(r)
    else:
        _print_run_report(data)
    return 0


# --------------------------------------------------------------------------
# argparse
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tq",
        description="Single entry point for turboquant-mlx streaming inference "
                    "(setup, run, benchmark, config, profile-report).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # setup
    sp = sub.add_parser("setup", help="Download model, install deps, run all repacks + warmup.")
    sp.add_argument("--model", default=None, help=f"HF model id (default: {DEFAULT_MODEL}).")
    sp.add_argument("--cache-budget-gb", type=float, default=None,
                    help=f"Resident expert budget for the warmup run (default {DEF_CACHE_BUDGET_GB}).")
    sp.add_argument("--k", type=int, default=None,
                    help=f"Max active experts for the warmup run (default {DEF_K}).")
    sp.set_defaults(func=cmd_setup)

    # run
    rp = sub.add_parser("run", help="Generate from a prompt.")
    rp.add_argument("prompt", help="The prompt text.")
    rp.add_argument("--model", default=None, help="Override HF model id.")
    rp.add_argument("--max-tokens", type=int, default=None,
                    help=f"Max tokens to generate (default {DEF_MAX_TOKENS}).")
    rp.add_argument("--temp", type=float, default=None,
                    help=f"Sampling temperature (default {DEF_TEMP}).")
    rp.add_argument("--cache-budget-gb", type=float, default=None,
                    help=f"Resident expert memory budget in GB (default {DEF_CACHE_BUDGET_GB}).")
    rp.add_argument("--k", type=int, default=None,
                    help=f"Max active experts / router top_k cap (default {DEF_K}).")
    rp.add_argument("--use-ane", dest="use_ane", action="store_const", const=True,
                    default=None, help="Enable ANE attention (macOS; first run compiles CoreML).")
    rp.add_argument("--no-freqsort", action="store_true",
                    help="Use the original layout instead of the freq-sorted variant.")
    rp.add_argument("--profile", action="store_true",
                    help="Enable deep instrumentation (per-layer timing, phase "
                         "breakdown, miss trace, memory + rescue timelines).")
    rp.add_argument("--perf-log", dest="perf_log", action="store_true", default=True,
                    help="Save timing to perf/ (default on).")
    rp.add_argument("--no-perf-log", dest="perf_log", action="store_false",
                    help="Do not write a perf log.")
    rp.add_argument("--no-chat-template", action="store_true", help="Pass the raw prompt.")
    rp.set_defaults(func=cmd_run)

    # benchmark
    bp = sub.add_parser("benchmark", help="Run the 5 standard prompts and report stats.")
    bp.add_argument("--model", default=None, help="Override HF model id.")
    bp.add_argument("--cache-budget-gb", type=float, default=None,
                    help=f"Resident expert memory budget in GB (default {DEF_CACHE_BUDGET_GB}).")
    bp.add_argument("--k", type=int, default=None,
                    help=f"Max active experts (default {DEF_K}).")
    bp.add_argument("--use-ane", dest="use_ane", action="store_const", const=True,
                    default=None, help="Enable ANE attention.")
    bp.add_argument("--no-freqsort", action="store_true",
                    help="Use the original layout instead of the freq-sorted variant.")
    bp.add_argument("--profile", action="store_true", help="Enable deep instrumentation.")
    bp.set_defaults(func=cmd_benchmark)

    # config
    cp = sub.add_parser("config", help="Show or set persisted defaults.")
    cp.add_argument("subcommand", nargs="?", choices=["show"], default=None,
                    help="'show' to print the current config.")
    cp.add_argument("--model", default=None, help="Set the default model id.")
    cp.add_argument("--cache-budget-gb", type=float, default=None,
                    help="Set the default cache budget (GB).")
    cp.add_argument("--k", type=int, default=None, help="Set the default max active experts.")
    cp.add_argument("--max-tokens", type=int, default=None, help="Set the default max tokens.")
    cp.add_argument("--temp", type=float, default=None, help="Set the default temperature.")
    cp.add_argument("--use-ane", dest="use_ane", action="store_const", const=True,
                    default=None, help="Set ANE attention default on.")
    cp.add_argument("--no-use-ane", dest="use_ane", action="store_const", const=False,
                    help="Set ANE attention default off.")
    cp.set_defaults(func=cmd_config)

    # profile-report
    pr = sub.add_parser("profile-report", help="Print a summary of a saved perf JSON.")
    pr.add_argument("perf_file", help="Path to a perf/<slug>-<ts>.json file.")
    pr.set_defaults(func=cmd_profile_report)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
