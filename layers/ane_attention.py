"""ANE (Apple Neural Engine) offload for scaled-dot-product attention.

Called from mlx_lm/mlx_lm/models/base.py ``scaled_dot_product_attention`` in
the turboquant fork of mlx-lm. Single-token (decode) attention runs on the ANE
via CoreML, freeing Metal's wired GPU pool for the streaming expert hot tier.

Design notes:
- All MLX / coremltools imports are lazy (inside functions). This module must
  import cleanly on a machine without either package (e.g. Linux dev box) so
  ``py_compile`` and plain ``import`` succeed.
- ``scale`` is a CoreML runtime input, not baked as a constant. One compiled
  model is valid for any scale mlx_lm passes; the cache key needs no float key.

KB: none (deployment glue; no research entry applies)
"""

from __future__ import annotations

import os
import threading

# Sequence-length buckets. For a given seq_len we pick the smallest bucket >=
# seq_len and pad K/V (and mask padding to -inf). seq_len beyond the last bucket
# falls back to the original MLX SDPA.
_BUCKETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

# fp16 cannot represent true -inf without overflow surprises in the softmax
# numerics; -1e4 is safely below any real score and drives softmax to ~0.
_NEG_INF_F16 = -1e4


def _build_sdpa_model(n_heads, n_kv_heads, head_dim, bucket_seq_len, scale):
    """Compile a CoreML MIL program for single-query GQA scaled-dot-product attention.

    Inputs (all fp16):
        q:     [1, n_heads,    1,              head_dim]
        k:     [1, n_kv_heads, bucket_seq_len, head_dim]
        v:     [1, n_kv_heads, bucket_seq_len, head_dim]
        mask:  [1, 1,          1,              bucket_seq_len]  (additive: 0 or -inf)
        scale: scalar                                          (multiplies scores)

    Output (fp16):
        out:   [1, n_heads, 1, head_dim]

    ``scale`` is a runtime input, not a baked constant, so one compiled model is
    correct for any scale mlx_lm passes. The ``scale`` argument here is unused for
    the graph body but kept in the signature for API symmetry / documentation.
    """
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    repeat_factor = n_heads // n_kv_heads

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, n_heads,    1,              head_dim), dtype=types.fp16),
            mb.TensorSpec(shape=(1, n_kv_heads, bucket_seq_len, head_dim), dtype=types.fp16),
            mb.TensorSpec(shape=(1, n_kv_heads, bucket_seq_len, head_dim), dtype=types.fp16),
            mb.TensorSpec(shape=(1, 1,          1,              bucket_seq_len), dtype=types.fp16),
            mb.TensorSpec(shape=(), dtype=types.fp16),
        ],
        opset_version=ct.target.iOS17,
    )
    def sdpa_prog(q, k, v, mask, scale):
        # GQA: repeat K/V heads to match the query head count.
        if repeat_factor > 1:
            k = mb.tile(x=k, reps=[1, repeat_factor, 1, 1])
            v = mb.tile(x=v, reps=[1, repeat_factor, 1, 1])
        k_t = mb.transpose(x=k, perm=[0, 1, 3, 2])          # [1, h, d, B]
        scores = mb.matmul(x=q, y=k_t)                       # [1, h, 1, B]
        scores = mb.mul(x=scores, y=scale)                   # runtime scale
        scores = mb.add(x=scores, y=mask)                    # additive mask
        probs = mb.softmax(x=scores, axis=-1)                # [1, h, 1, B]
        out = mb.matmul(x=probs, y=v)                        # [1, h, 1, d]
        return out

    mlmodel = ct.convert(
        sdpa_prog,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,   # ANE + GPU + CPU, ANE preferred
        minimum_deployment_target=ct.target.macOS14,
    )
    return mlmodel


def _cache_dir() -> str:
    d = os.path.expanduser("~/.turboquant_mlx/ane_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_path(n_heads, n_kv_heads, head_dim, bucket) -> str:
    return os.path.join(
        _cache_dir(),
        f"{n_heads}h_{n_kv_heads}kv_{head_dim}d_B{bucket}.mlpackage",
    )


def _load_or_compile(n_heads, n_kv_heads, head_dim, bucket, scale):
    """Return an MLModel for this (config, bucket), compiling+caching on miss.

    Compilation is ~30s per bucket; results are cached on disk so subsequent
    runs (and subsequent buckets across sessions) load from cache.
    """
    import coremltools as ct

    path = _cache_path(n_heads, n_kv_heads, head_dim, bucket)
    if os.path.exists(path):
        return ct.models.MLModel(path)
    print(f"[ANE] compiling attention model for bucket={bucket} "
          f"({n_heads}h/{n_kv_heads}kv/{head_dim}d) ... (one-time, ~30s)", flush=True)
    m = _build_sdpa_model(n_heads, n_kv_heads, head_dim, bucket, scale)
    m.save(path)
    return m


class ANEAttentionDispatcher:
    """Routes single-token attention to the ANE via CoreML.

    Called directly from the turboquant fork of mlx-lm's
    ``scaled_dot_product_attention`` in ``mlx_lm/models/base.py``.

    Returns None to signal fallback (caller uses mx.fast.scaled_dot_product_attention).
    Compile-on-demand per sequence-length bucket.

    Thread-safety: model compilation is guarded by ``self._lock``; CoreML
    ``predict`` is stateless and thread-safe per CoreML's guarantee.
    """

    _BUCKETS = _BUCKETS

    def __init__(self):
        self._installed = False   # set by install()
        self._models: dict = {}   # (n_heads, n_kv_heads, head_dim, bucket) -> MLModel
        self._lock = threading.Lock()
        self._available = self._check_coreml()
        self.calls = 0
        self.ane_calls = 0
        self.fallback_calls = 0

    def _check_coreml(self) -> bool:
        try:
            import coremltools  # noqa: F401
            return True
        except ImportError:
            print("[ANE] coremltools not found; attention stays on GPU "
                  "(install with: pip install coremltools)")
            return False

    def _get_bucket(self, seq_len: int):
        for b in self._BUCKETS:
            if b >= seq_len:
                return b
        return None  # > max bucket -> fallback

    def __call__(self, q, k, v, scale, mask=None):
        """Attempt ANE dispatch. Returns None to signal caller should use MLX fallback.

        Called from mlx_lm fork's ``scaled_dot_product_attention`` in base.py.
        Only intercepts single-query decode (q.shape[-2] == 1) within bucket range.
        """
        self.calls += 1

        # Only intercept single-query decode. Prefill stays on GPU matmul path.
        if not self._available or q.ndim != 4 or q.shape[-2] != 1:
            self.fallback_calls += 1
            return None

        seq_len = k.shape[-2]
        bucket = self._get_bucket(seq_len)
        if bucket is None:
            self.fallback_calls += 1
            return None

        n_heads = q.shape[1]
        n_kv_heads = k.shape[1]
        head_dim = q.shape[-1]
        if n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
            self.fallback_calls += 1
            return None

        model_key = (n_heads, n_kv_heads, head_dim, bucket)

        with self._lock:
            if model_key not in self._models:
                try:
                    self._models[model_key] = _load_or_compile(
                        n_heads, n_kv_heads, head_dim, bucket, float(scale))
                except Exception as e:
                    print(f"[ANE] failed to compile bucket {bucket}: {e} "
                          f"— disabling ANE attention")
                    self._available = False
                    self.fallback_calls += 1
                    return None
            mlmodel = self._models[model_key]

        try:
            import numpy as np
            import mlx.core as mx

            if mask is not None and not isinstance(mask, str):
                mx.eval(q, k, v, mask)
            else:
                mx.eval(q, k, v)

            q_np = np.array(q, dtype=np.float16)
            k_np = np.array(k, dtype=np.float16)
            v_np = np.array(v, dtype=np.float16)

            pad_len = bucket - seq_len
            if pad_len > 0:
                k_pad = np.zeros(
                    (k_np.shape[0], k_np.shape[1], pad_len, head_dim),
                    dtype=np.float16)
                v_pad = np.zeros_like(k_pad)
                k_np = np.concatenate([k_np, k_pad], axis=-2)
                v_np = np.concatenate([v_np, v_pad], axis=-2)

            mask_np = np.full((1, 1, 1, bucket), _NEG_INF_F16, dtype=np.float16)
            mask_np[:, :, :, :seq_len] = 0.0
            if mask is not None and not isinstance(mask, str):
                orig_mask_np = np.array(mask, dtype=np.float16)
                m = orig_mask_np.reshape(-1)[-seq_len:]
                mask_np[:, :, :, :len(m)] += m.reshape(1, 1, 1, -1)

            scale_np = np.array(float(scale), dtype=np.float16)

            preds = mlmodel.predict({
                "q": q_np, "k": k_np, "v": v_np,
                "mask": mask_np, "scale": scale_np,
            })
            out_np = preds["out"] if "out" in preds else next(iter(preds.values()))
            result = mx.array(out_np).astype(mx.float16)
            self.ane_calls += 1
            return result

        except Exception as e:
            print(f"[ANE] runtime error: {e} — falling back to GPU for this call")
            self.fallback_calls += 1
            return None

    def install(self):
        """Mark the dispatcher as active. Idempotent."""
        if self._installed:
            return
        self._installed = True
        print(f"[ANE] attention dispatcher active "
              f"({'CoreML available' if self._available else 'GPU fallback mode'})")

    def uninstall(self):
        """Deactivate the dispatcher. Idempotent."""
        self._installed = False

    def warmup(self, n_heads, n_kv_heads, head_dim, scale, buckets=None):
        """Pre-compile the given buckets (default: all) in a background thread.

        Returns immediately; compilation proceeds off the calling thread so the
        first decode token does not pay the ~30s/bucket compile latency. Guarded
        by ``self._lock`` so it can't race the on-demand compile in ``__call__``.
        """
        if not self._available:
            return None
        buckets = list(buckets) if buckets is not None else list(self._BUCKETS)

        def _run():
            done = 0
            for b in buckets:
                model_key = (n_heads, n_kv_heads, head_dim, b)
                try:
                    with self._lock:
                        if model_key not in self._models:
                            self._models[model_key] = _load_or_compile(
                                n_heads, n_kv_heads, head_dim, b, float(scale))
                    done += 1
                    print(f"[ANE] warmup {done}/{len(buckets)} buckets ready "
                          f"(bucket={b})", flush=True)
                except Exception as e:
                    print(f"[ANE] warmup: bucket {b} failed: {e}", flush=True)
            print(f"[ANE] warmup complete: {done}/{len(buckets)} buckets compiled",
                  flush=True)

        t = threading.Thread(target=_run, name="ane-warmup", daemon=True)
        t.start()
        return t

    def stats(self) -> dict:
        return {
            "total": self.calls,
            "ane": self.ane_calls,
            "fallback": self.fallback_calls,
            "ane_rate": (self.ane_calls / self.calls) if self.calls else 0.0,
        }


# Module-level singleton.
_dispatcher = None


def get_dispatcher() -> "ANEAttentionDispatcher":
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = ANEAttentionDispatcher()
    return _dispatcher
