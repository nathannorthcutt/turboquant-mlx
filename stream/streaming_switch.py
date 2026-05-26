"""Streaming replacement for PolarQuantizedSwitchLinear.

Holds no resident expert weights. On each forward it discovers the
router-selected experts, loads *only those* expert slices from the mmap'd
safetensors (through an LRU ``ExpertCache``), and runs the exact same fused
polar kernels as the resident layer — assembling a small (n_selected, ...)
weight stack and remapping the routing indices to local positions.

Output is numerically identical to PolarQuantizedSwitchLinear because it uses
the same kernels on the same expert bytes; only the *set* of experts present
in memory at any moment differs.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx.core.rotation import rotate_input
from turboquant_mlx.kernels.polar_gather_qmv import polar_gather_qmv
from turboquant_mlx.kernels.polar_multi_gather_qmv import polar_multi_gather_qmv


class ExpertCache:
    """Process-wide LRU cache of per-expert (weight, scales) mx.arrays.

    Shared across every streaming layer so a single byte budget bounds the
    total resident expert memory. Keyed by (weight_key, expert) which is
    globally unique (weight_key already encodes layer + projection).

    The experts missing for a single ``gather`` are read in one batch and,
    when ``prefetch_workers > 1``, their ``pread``s are fanned across a thread
    pool. ``pread`` releases the GIL, so the per-layer disk stall drops from the
    sum of the slice reads to roughly the slowest one. MLX array construction
    and ``eval`` stay on the calling thread, so the produced tensors are
    bit-identical to the serial path.
    """

    def __init__(self, reader, budget_bytes: int, *, prefetch_workers: int = 8):
        self.reader = reader
        self.budget = budget_bytes
        self.cur = 0
        self._od: "OrderedDict[tuple, tuple]" = OrderedDict()
        # stats
        self.hits = 0
        self.misses = 0
        self.bytes_read = 0
        # parallel prefetch
        self._workers = max(1, int(prefetch_workers))
        self._pool = (ThreadPoolExecutor(max_workers=self._workers)
                      if self._workers > 1 else None)

    # -- loading -------------------------------------------------------
    def _load_serial(self, miss_pairs):
        out = {}
        for wkey, skey, e in miss_pairs:
            w = self.reader.read_expert(wkey, e)
            s = self.reader.read_expert(skey, e)
            mx.eval(w, s)  # force the slice copy out of mmap into an MLX buffer
            out[(wkey, e)] = (w, s, w.nbytes + s.nbytes)
        return out

    def _load_parallel(self, miss_pairs):
        # fan the (GIL-releasing) preads across the pool, then build + eval the
        # MLX arrays on this thread so MLX is never touched from a worker.
        tasks = []
        for wkey, skey, e in miss_pairs:
            tasks.append((wkey, e))
            tasks.append((skey, e))
        bufs = list(self._pool.map(lambda t: self.reader.read_expert_np(*t), tasks))
        arrs = [mx.array(buf, dtype=dt) for (buf, dt) in bufs]
        mx.eval(*arrs)
        out = {}
        for i, (wkey, _skey, e) in enumerate(miss_pairs):
            w, s = arrs[2 * i], arrs[2 * i + 1]
            out[(wkey, e)] = (w, s, w.nbytes + s.nbytes)
        return out

    def _evict(self):
        while self.cur > self.budget and len(self._od) > 1:
            _, (_w, _s, nb) = self._od.popitem(last=False)  # oldest
            self.cur -= nb

    def gather(self, wkey: str, skey: str, experts: list[int]):
        """Return stacked (n, out, packed) weight + (n, out, ng) scales for
        ``experts`` (in the given order)."""
        # classify against the current resident set, then load all misses in
        # one (optionally parallel) batch.
        miss_pairs = []
        for e in experts:
            ck = (wkey, e)
            if ck in self._od:
                self.hits += 1
            else:
                self.misses += 1
                miss_pairs.append((wkey, skey, e))

        if miss_pairs:
            loaded = (self._load_parallel(miss_pairs) if self._pool
                      else self._load_serial(miss_pairs))
            for ck, entry in loaded.items():
                self._od[ck] = entry
                self.cur += entry[2]
                self.bytes_read += entry[2]

        # Build the stack *before* evicting so freshly loaded slices are still
        # present (and held by ``ws``/``ss``, so eviction can't free them out
        # from under this call).
        ws, ss = [], []
        for e in experts:
            ck = (wkey, e)
            w, s, _ = self._od[ck]
            self._od.move_to_end(ck)
            ws.append(w)
            ss.append(s)

        self._evict()
        return mx.stack(ws, axis=0), mx.stack(ss, axis=0)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def stats(self):
        tot = self.hits + self.misses
        return {
            "hit_rate": (self.hits / tot) if tot else 0.0,
            "hits": self.hits,
            "misses": self.misses,
            "resident_experts": len(self._od),
            "resident_gb": self.cur / 1e9,
            "bytes_read_gb": self.bytes_read / 1e9,
        }


class StreamingSwitchLinear(nn.Module):
    """Drop-in for PolarQuantizedSwitchLinear that streams experts from disk."""

    def __init__(
        self,
        *,
        input_dims: int,
        output_dims: int,
        num_experts: int,
        bits: int,
        group_size: int,
        needs_rotation: bool,
        codebook: mx.array,
        signs: mx.array,
        weight_key: str,
        scales_key: str,
        cache: ExpertCache,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_experts = num_experts
        self.bits = bits
        self.group_size = group_size
        self._needs_rotation = needs_rotation
        # small resident tensors
        self.codebook = codebook
        self.signs = signs
        # streaming wiring (underscored so they are not treated as params)
        self._weight_key = weight_key
        self._scales_key = scales_key
        self._cache = cache
        self.freeze()

    def _dequantize_selected(self, w_sel: mx.array, s_sel: mx.array) -> mx.array:
        """Dequantize only the selected experts to float16 (prefill path).

        Mirrors PolarQuantizedSwitchLinear._dequantize_all but on an
        (n_sel, out, packed) stack instead of all num_experts.
        """
        from turboquant_mlx.core.packing import unpack_indices
        from turboquant_mlx.core.codebook import dequantize_scalar

        n_sel = w_sel.shape[0]
        n_groups = self.input_dims // self.group_size
        idx = unpack_indices(w_sel, self.bits, self.input_dims)
        w_deq = dequantize_scalar(idx, self.codebook)
        w_deq = w_deq.reshape(n_sel, self.output_dims, n_groups, self.group_size)
        w_deq = w_deq * mx.expand_dims(s_sel, axis=-1)
        return w_deq.reshape(n_sel, self.output_dims, self.input_dims)

    def __call__(self, x, indices, sorted_indices=False):
        if self._needs_rotation:
            x = rotate_input(x, self.signs)

        n_tokens = 1 if x.ndim <= 2 else math.prod(x.shape[:-2])
        k = indices.shape[-1] if indices.ndim >= 1 else 1

        # Discover selected experts (forces a tiny GPU->CPU sync on the
        # routing indices — unavoidable: we must know what to load).
        idx_global = np.asarray(
            indices.reshape(-1).astype(mx.uint32).tolist(), dtype=np.int64
        )
        sel = sorted(set(int(v) for v in idx_global))
        remap = {e: i for i, e in enumerate(sel)}
        w_sel, s_sel = self._cache.gather(self._weight_key, self._scales_key, sel)
        idx_local_flat = mx.array([remap[int(v)] for v in idx_global], dtype=mx.uint32)

        if n_tokens == 1:
            x_flat = x.reshape(-1)
            y = polar_gather_qmv(
                w_sel, s_sel, self.codebook,
                x_flat, idx_local_flat, self.bits, self.group_size,
            )
            return y.reshape(list(indices.shape) + [1, self.output_dims])

        if n_tokens == k:
            x_2d = x.reshape(k, self.input_dims)
            y = polar_multi_gather_qmv(
                w_sel, s_sel, self.codebook,
                x_2d, idx_local_flat, self.bits, self.group_size,
            )
            return y.reshape(list(indices.shape) + [1, self.output_dims])

        # Prefill: dequantize only the selected experts, gather_mm locally.
        w_deq = self._dequantize_selected(w_sel, s_sel)
        idx_local = idx_local_flat.reshape(indices.shape)
        return mx.gather_mm(
            x, w_deq.swapaxes(-1, -2),
            rhs_indices=idx_local, sorted_indices=False,
        )

    def _extra_repr(self):
        return (
            f"input_dims={self.input_dims}, output_dims={self.output_dims}, "
            f"num_experts={self.num_experts}, bits={self.bits}, "
            f"group_size={self.group_size}, STREAMING key={self._weight_key}"
        )
