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
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx.core.rotation import rotate_input
from turboquant_mlx.kernels.polar_gather_qmv import polar_gather_qmv
from turboquant_mlx.kernels.polar_multi_gather_qmv import polar_multi_gather_qmv
from turboquant_mlx.kernels.polar_gather_qmm import (
    polar_gather_qmm, supports as _gather_qmm_supports,
)


class ExpertCache:
    """Process-wide LRU cache of per-expert (weight, scales) mx.arrays, with
    optional cross-layer speculative prefetch.

    Shared across every streaming layer so a single byte budget bounds the
    total resident expert memory. Keyed by (weight_key, expert) which is
    globally unique (weight_key already encodes layer + projection).

    The experts missing for a single ``gather`` are read in one batch and,
    when ``prefetch_workers > 1``, their ``pread``s are fanned across a thread
    pool. ``pread`` releases the GIL, so the per-layer disk stall drops from the
    sum of the slice reads to roughly the slowest one. MLX array construction
    and ``eval`` stay on the calling thread, so the produced tensors are
    bit-identical to the serial path.

    **Speculative prefetch** (``prefetch_ahead > 0``): when a layer starts we
    kick off *background* disk reads for the experts the previous token used at
    the next layer(s) — a near-free, high-accuracy predictor because MoE routing
    is strongly correlated token-to-token. The reads land in ``_staging`` as raw
    numpy buffers; the next ``gather`` claims them on the calling thread (where
    it builds the mx.array), so the disk latency overlaps with compute and *no
    MLX is ever touched off the main thread*. Output stays bit-identical:
    prefetch changes only *when* bytes are read, never *which* experts a gather
    returns (mispredicted slices simply age out of staging unused).
    """

    def __init__(self, reader, budget_bytes: int, *, prefetch_workers: int = 8,
                 prefetch_ahead: int = 1,
                 staging_budget_bytes: int = 1_500_000_000,
                 pin_keys=None):
        self.reader = reader
        self.budget = budget_bytes
        self.cur = 0
        self._od: "OrderedDict[tuple, tuple]" = OrderedDict()
        # stats
        self.hits = 0             # served resident from _od
        self.misses = 0           # critical-path disk reads
        self.prefetch_hits = 0    # served from a completed background prefetch
        self.bytes_read = 0           # critical-path bytes
        self.bytes_prefetched = 0     # background prefetch bytes
        self.prefetched = 0           # experts prefetched
        self.prefetch_dropped = 0     # prefetched then evicted from staging unused
        self.read_runs = 0            # coalesced range reads issued (critical path)
        self.expert_reads = 0         # experts loaded on the critical path
        # parallel read pool (shared by critical-path misses and prefetch)
        self._workers = max(1, int(prefetch_workers))
        self._pool = (ThreadPoolExecutor(max_workers=self._workers)
                      if self._workers > 1 else None)
        # speculative-prefetch state (needs the pool to run in background)
        self._prefetch_ahead = max(0, int(prefetch_ahead)) if self._pool else 0
        # saturation throttle: after warming up, if prefetch is rescuing too few
        # of the would-be misses (its reads arrive too late because the bus is
        # saturated), turn prefetch OFF so it stops wasting bandwidth. Judged by
        # the RESCUE RATE — prefetch_hits / (prefetch_hits + misses), NOT raw
        # utilization, since prefetch over-fetches but can still rescue a large
        # share of misses on fast storage. Self-disables on the bus-bound USB
        # 235B (~0.1% rescue); stays on for fast NVMe (tens of %).
        self._throttle_warmup = 2000
        self._throttle_min_rescue = 0.03
        self._throttle_decided = False
        self._staging_cap = max(0, int(staging_budget_bytes))
        self._staging: "OrderedDict[tuple, tuple]" = OrderedDict()  # key -> (np_buf, dt)
        self._staging_bytes = 0
        self._inflight: set = set()
        self._lock = threading.Lock()        # guards _staging / _inflight / stats
        self._layer_keys: dict = {}          # layer_idx -> [(wkey, skey), ...]
        self._last_experts: dict = {}        # layer_idx -> experts (previous token)
        # calibration trace: per-decode-token expert selections, for #2/#3
        self._trace_on = False
        self._trace: list = []               # [(layer_idx, tuple(sorted experts)), ...]
        # frequency-based pinning (#2): these (wkey, expert) keys never evict;
        # they live in a separate dict so eviction stays O(1) on the LRU set.
        self._pin_keys: set = set(pin_keys or ())
        self._pinned: dict = {}              # (wkey, e) -> (w, s, nbytes) wired mx.arrays
        self._pin_hits = 0
        # per-token routing cache (#1): the trigger projection (gate_proj) computes
        # the GPU->CPU routing sync once and stashes it here keyed by layer_idx; the
        # two non-trigger projections (up/down) reuse it, skipping 2/3 of the syncs.
        self._layer_routing_cache: dict = {}  # layer_idx -> (sel, idx_local_flat)
        # rotated-input cache (#4.3): in SwiGLU MoE gate_proj and up_proj receive
        # the identical layer input x, so the Hadamard rotation is computed once
        # by the trigger (gate) and reused by up_proj. Keyed by layer_idx.
        self._rotated_x_cache: dict = {}     # layer_idx -> mx.array (rotated x)
        # fused gate+up cache (fused-gate_up optimization): when a fused companion
        # is present the gate projection runs ONE kernel over the fused
        # (2*out) weight stack, splits the result, returns the gate half and
        # stashes (gate_out, up_out) here; the up projection pops its half.
        self._fused_gate_up_cache: dict = {}  # layer_idx -> (gate_out, up_out)
        # warmup histogram (#3): live per-(layer, expert) access counts for
        # cross-session cache warmup. Populated in on_layer_start.
        self._hist: dict = {}                # (layer_idx, expert_id) -> count
        self._trigger_calls: int = 0         # on_layer_start calls (1 per layer per token)

    # -- speculative prefetch -----------------------------------------
    def register_layer(self, layer_idx: int, proj_keys: list):
        """Record the (weight_key, scales_key) of every projection in a layer so
        the predictor can prefetch all of them for the next layer at once."""
        self._layer_keys[layer_idx] = list(proj_keys)

    def on_layer_start(self, layer_idx: int, experts: list):
        """Called once per layer per token (from the trigger projection).

        Prefetches the next layer(s)' predicted experts in the background, then
        records this layer's selection for the *next* token to predict from.
        """
        # once enough has been prefetched, judge whether it's worth continuing
        if (self._prefetch_ahead and not self._throttle_decided
                and self.prefetched >= self._throttle_warmup):
            self._throttle_decided = True
            rescue = self.prefetch_hits / max(1, self.prefetch_hits + self.misses)
            if rescue < self._throttle_min_rescue:
                self._prefetch_ahead = 0
                print(f"[stream] prefetch self-disabled: rescue rate {rescue:.0%} < "
                      f"{self._throttle_min_rescue:.0%} (storage appears bandwidth-bound)")
        if self._prefetch_ahead:
            for nxt in range(layer_idx + 1, layer_idx + 1 + self._prefetch_ahead):
                pred = self._last_experts.get(nxt)
                if pred:
                    self._prefetch_layer(nxt, pred)
        self._last_experts[layer_idx] = experts
        # warmup histogram (#3): count each (layer, expert) access live.
        for e in experts:
            key = (layer_idx, e)
            self._hist[key] = self._hist.get(key, 0) + 1
        # count triggers (1 per layer per token) so total_tokens can be derived.
        self._trigger_calls += 1

    def dump_histogram(self, path: str, model_id: str = "", k: int = 0) -> int:
        """Write per-(layer,expert) access counts to a JSON file.

        Returns the number of (layer,expert) pairs written.
        Format: {"model_id": ..., "k": ..., "total_tokens": ...,
                 "hist": [[layer, expert, count], ...]}
        Sorted by count descending so the warmup loader can take a prefix.
        """
        import json
        n_layers = len(self._layer_keys)
        total_tokens = (self._trigger_calls // n_layers) if n_layers else 0
        rows = sorted(
            ([lyr, e, c] for (lyr, e), c in self._hist.items()),
            key=lambda r: r[2],
            reverse=True,
        )
        with open(path, "w") as f:
            json.dump({
                "model_id": model_id,
                "k": k,
                "total_tokens": total_tokens,
                "hist": rows,
            }, f)
        return len(rows)

    def load_histogram(self, path: str, model_id: str = "", k: int = 0) -> dict:
        """Load a histogram JSON. Returns the raw dict.

        Validates model_id and k if non-empty (skip, warn, do not crash on
        mismatch). Does NOT pre-warm the cache — warmup is done by the caller
        (loader.py).
        """
        import json
        with open(path) as f:
            data = json.load(f)
        if model_id and data.get("model_id") and data["model_id"] != model_id:
            print(f"[stream] histogram model_id mismatch: file={data['model_id']!r} "
                  f"expected={model_id!r} (using anyway)")
        if k and data.get("k") and data["k"] != k:
            print(f"[stream] histogram k mismatch: file={data['k']} "
                  f"expected={k} (using anyway)")
        return data

    def _prefetch_layer(self, layer_idx: int, predicted: list):
        keys = self._layer_keys.get(layer_idx)
        if not keys or self._pool is None:
            return
        # With a fused gate+up companion, gate and up are served from ONE fused
        # read keyed by the fused weight key. Prefetch that once (on the gate
        # projection) and skip the separate up prefetch; down is unchanged.
        fused = getattr(self.reader, 'has_fused_gate_up', False)
        with self._lock:
            for wkey, skey in keys:
                if fused and wkey.endswith("up_proj.weight"):
                    continue  # served by the fused gate prefetch below
                if fused and wkey.endswith("gate_proj.weight"):
                    fwkey = wkey.replace("gate_proj.weight", "gate_up_proj.weight")
                    fskey = skey.replace("gate_proj.scales", "gate_up_proj.scales")
                    for e in predicted:
                        ck = (fwkey, e)
                        if (ck in self._od or ck in self._pinned
                                or ck in self._staging or ck in self._inflight):
                            continue
                        self._inflight.add(ck)
                        self._pool.submit(
                            self._prefetch_one_fused, fwkey, fskey, layer_idx, e)
                    continue
                for e in predicted:
                    ck = (wkey, e)
                    # already resident (LRU or pinned), staged, or being read
                    if (ck in self._od or ck in self._pinned
                            or ck in self._staging or ck in self._inflight):
                        continue
                    self._inflight.add(ck)
                    self._pool.submit(self._prefetch_one, wkey, skey, e)

    # -- calibration trace (#2 frequencies / #3 co-activation) --------
    def set_trace(self, on: bool = True):
        self._trace_on = on

    def record_trace(self, layer_idx: int, experts):
        # called once per decode token per layer with that token's selected set
        if self._trace_on:
            self._trace.append((layer_idx, tuple(experts)))

    def dump_trace(self, path: str) -> int:
        import json
        with open(path, "w") as f:
            json.dump(self._trace, f)
        return len(self._trace)

    def _prefetch_one(self, wkey: str, skey: str, e: int):
        # disk-only background work; never touches MLX (built later on main thread)
        try:
            # Pair-read when the reader interleaves weight+scales as companion
            # files: one pread covers both, halving the syscalls for this expert
            # (falls back to two reads when the method / interleaving is absent).
            if hasattr(self.reader, 'read_expert_pair_np'):
                (w_np, w_dt), (s_np, s_dt) = self.reader.read_expert_pair_np(
                    wkey, skey, e)
                wbuf = (w_np, w_dt)                       # (np_array, mlx_dtype)
                sbuf = (s_np, s_dt)
            else:
                wbuf = self.reader.read_expert_np(wkey, e)   # (np_array, mlx_dtype)
                sbuf = self.reader.read_expert_np(skey, e)
            nb = wbuf[0].nbytes + sbuf[0].nbytes
            with self._lock:
                self._inflight.discard((wkey, e))
                # Store both projections under one key so eviction can never
                # orphan half a pair (drop the weight but keep its scales).
                self._staging[(wkey, e)] = (wbuf, sbuf)
                self._staging_bytes += nb
                self.bytes_prefetched += nb
                self.prefetched += 1
                # bound staging: drop oldest (mispredicted) entries FIFO
                while self._staging_bytes > self._staging_cap and len(self._staging) > 1:
                    _, (wb, sb) = self._staging.popitem(last=False)
                    self._staging_bytes -= (wb[0].nbytes + sb[0].nbytes)
                    self.prefetch_dropped += 1
        except Exception:
            with self._lock:
                self._inflight.discard((wkey, e))

    def _prefetch_one_fused(self, fwkey: str, fskey: str, layer_idx: int, e: int):
        # Background fused gate+up read (never touches MLX). Staged under the
        # fused weight key so gather_fused claims it exactly like a normal miss.
        try:
            (w_np, w_dt), (s_np, s_dt) = self.reader.read_fused_gate_up_pair_np(
                layer_idx, e)
            wbuf = (w_np, w_dt)
            sbuf = (s_np, s_dt)
            nb = w_np.nbytes + s_np.nbytes
            with self._lock:
                self._inflight.discard((fwkey, e))
                self._staging[(fwkey, e)] = (wbuf, sbuf)
                self._staging_bytes += nb
                self.bytes_prefetched += nb
                self.prefetched += 1
                while self._staging_bytes > self._staging_cap and len(self._staging) > 1:
                    _, (wb, sb) = self._staging.popitem(last=False)
                    self._staging_bytes -= (wb[0].nbytes + sb[0].nbytes)
                    self.prefetch_dropped += 1
        except Exception:
            with self._lock:
                self._inflight.discard((fwkey, e))

    # -- loading -------------------------------------------------------
    def _load_coalesced_fused(self, experts, fwkey, layer_idx):
        """Load every missed fused gate+up expert for one layer, coalescing
        contiguous positions into range reads (mirrors _load_coalesced but via
        the fused reader). Returns {(fwkey, e): (np_w, dt_w, np_s, dt_s, nbytes)}.
        """
        experts = sorted(experts)
        runs = []
        i = 0
        while i < len(experts):
            j = i
            while j + 1 < len(experts) and experts[j + 1] == experts[j] + 1:
                j += 1
            runs.append((experts[i], experts[j] - experts[i] + 1))
            i = j + 1
        self.read_runs += len(runs)
        self.expert_reads += len(experts)

        wb, sb = {}, {}

        def _read(t):
            start, count = t
            (w_buf, w_dt), (s_buf, s_dt) = self.reader.read_fused_gate_up_range_np(
                layer_idx, start, count)
            return start, count, w_buf, w_dt, s_buf, s_dt

        results = (list(self._pool.map(_read, runs)) if self._pool
                   else [_read(t) for t in runs])
        for start, count, w_buf, w_dt, s_buf, s_dt in results:
            for k in range(count):
                wb[start + k] = (w_buf[k], w_dt)
                sb[start + k] = (s_buf[k], s_dt)

        out = {}
        for e in experts:
            wbuf, wdt = wb[e]
            sbuf, sdt = sb[e]
            nb = wbuf.nbytes + sbuf.nbytes
            out[(fwkey, e)] = (wbuf, wdt, sbuf, sdt, nb)
        return out

    def _load_coalesced(self, miss_pairs):
        """Load every missed expert for one projection, coalescing *contiguous*
        expert positions into a single range ``pread`` (the #3 layout win).

        All pairs in one gather share the same (wkey, skey) — gather is called
        per projection — so we sort the missed experts, split them into runs of
        consecutive positions, and read each run once (weight + scales) instead
        of once per expert. On a co-activation-ordered checkpoint a token's
        misses fall into far fewer runs, so this turns many small random reads
        into a few larger sequential ones. The per-expert slices are then built
        into MLX arrays on this (the calling) thread, as before.
        """
        wkey, skey = miss_pairs[0][0], miss_pairs[0][1]
        experts = sorted(p[2] for p in miss_pairs)
        runs = []
        i = 0
        while i < len(experts):
            j = i
            while j + 1 < len(experts) and experts[j + 1] == experts[j] + 1:
                j += 1
            runs.append((experts[i], experts[j] - experts[i] + 1))
            i = j + 1
        self.read_runs += len(runs)
        self.expert_reads += len(experts)

        wb, sb = {}, {}
        if hasattr(self.reader, 'read_range_pair_np'):
            # Pair-read: one task per run reads BOTH the weight and scales range
            # (a single pread when the companion files are interleaved), so a run
            # costs one task instead of two — halving the per-run disk syscalls.
            def _read_pair(t):
                start, count = t
                (w_buf, w_dt), (s_buf, s_dt) = self.reader.read_range_pair_np(
                    wkey, skey, start, count)
                return start, count, w_buf, w_dt, s_buf, s_dt

            results = (list(self._pool.map(_read_pair, runs)) if self._pool
                       else [_read_pair(t) for t in runs])
            for start, count, w_buf, w_dt, s_buf, s_dt in results:
                for k in range(count):
                    wb[start + k] = (w_buf[k], w_dt)
                    sb[start + k] = (s_buf[k], s_dt)
        else:
            # Legacy path: two independent range reads per run (weight, scales).
            def _read(t):
                key, start, count = t
                buf, dt = self.reader.read_range_np(key, start, count)
                return key, start, buf, dt

            tasks = []
            for start, count in runs:
                tasks.append((wkey, start, count))
                tasks.append((skey, start, count))
            results = (list(self._pool.map(_read, tasks)) if self._pool
                       else [_read(t) for t in tasks])

            for key, start, buf, dt in results:
                tgt = wb if key == wkey else sb
                for k in range(buf.shape[0]):
                    tgt[start + k] = (buf[k], dt)

        # Store the raw numpy slices (pageable unified RAM), not mx.arrays (which
        # would be wired GPU memory). The K active experts are wired only during
        # each gather's mx.stack, then released. Numeric output is unchanged.
        out = {}
        for e in experts:
            wbuf, wdt = wb[e]
            sbuf, sdt = sb[e]
            nb = wbuf.nbytes + sbuf.nbytes
            out[(wkey, e)] = (wbuf, wdt, sbuf, sdt, nb)
        return out

    def _evict(self):
        while self.cur > self.budget and len(self._od) > 1:
            _, (_, _, _, _, nb) = self._od.popitem(last=False)  # oldest
            self.cur -= nb

    def _insert(self, ck, entry):
        """Place a freshly loaded (np_w, dt_w, np_s, dt_s, nbytes) entry. Pinned
        keys go to the never-evicted dict; everything else into the LRU set.

        Pinned experts are hot on nearly every token, so they are *wired* into
        mx.arrays once at insert (a one-time cost) and stored as a 3-tuple
        (w, s, nbytes) — every subsequent gather then uses them zero-copy. LRU
        (_od) entries stay as pageable 5-tuples and pay an mx.array() per gather.
        """
        if ck in self._pin_keys:
            np_w, dt_w, np_s, dt_s, nb = entry
            w = mx.array(np_w, dtype=dt_w)
            s = mx.array(np_s, dtype=dt_s)
            mx.eval(w, s)                    # one-time wire cost at insert
            self._pinned[ck] = (w, s, nb)    # 3-tuple: mx.array format
        else:
            self._od[ck] = entry
        self.cur += entry[4]

    def gather(self, wkey: str, skey: str, experts: list[int]):
        """Return stacked (n, out, packed) weight + (n, out, ng) scales for
        ``experts`` (in the given order)."""
        return self._gather_impl(wkey, skey, experts, fused_layer_idx=None)

    def gather_fused(self, fwkey: str, fskey: str, layer_idx: int,
                     experts: list[int]):
        """Fused gate+up variant of :meth:`gather`.

        Returns stacked ``(n, 2*out, packed)`` weight + ``(n, 2*out, ng)`` scales
        for ``experts`` — the gate rows then the up rows — read via the fused
        companion (reader.read_fused_gate_up_*). Entries live in the same LRU as
        regular experts, keyed by the fused weight key; they are ~2x the size of a
        single-projection entry (they cover gate AND up), which the byte budget
        accounts for naturally. ``layer_idx`` selects the fused companion tensor.
        """
        return self._gather_impl(fwkey, fskey, experts, fused_layer_idx=layer_idx)

    def _gather_impl(self, wkey: str, skey: str, experts: list[int], *,
                     fused_layer_idx=None):
        """Return stacked weight + scales for ``experts`` (in the given order).

        ``fused_layer_idx`` None is the normal per-projection path; when set, the
        misses are loaded as fused gate+up experts for that layer (the LRU key is
        still ``wkey`` — the fused weight key — so staging/eviction are shared)."""
        # classify each requested expert against the pinned set, the LRU
        # resident set, and the prefetch staging area; load the rest in a batch.
        miss_pairs = []
        staged = []  # (expert, wbuf, sbuf) — read in background, build here
        awaited = []  # experts whose read is already in-flight (don't duplicate)
        with self._lock:
            for e in experts:
                ck = (wkey, e)
                if ck in self._pinned:
                    self.hits += 1
                    self._pin_hits += 1
                    continue
                if ck in self._od:
                    self.hits += 1
                    continue
                if ck in self._staging:
                    wbuf, sbuf = self._staging.pop(ck)
                    self._staging_bytes -= (wbuf[0].nbytes + sbuf[0].nbytes)
                    staged.append((e, wbuf, sbuf))
                    self.prefetch_hits += 1
                elif ck in self._inflight:
                    # A prefetch is already reading this expert. Don't issue a
                    # duplicate pread — wait for it to land (below). Count as a
                    # miss for now; retroactively corrected if it arrives.
                    awaited.append(e)
                    self.misses += 1
                else:
                    self.misses += 1
                    miss_pairs.append((wkey, skey, e))

        # Build prefetch-staged slices into MLX arrays on THIS thread (no disk;
        # the bytes were already read in the background). Lock released above so
        # background reads keep flowing while we build.
        for e, wbuf, sbuf in staged:
            np_w, dt_w = wbuf
            np_s, dt_s = sbuf
            nb = np_w.nbytes + np_s.nbytes
            self._insert((wkey, e), (np_w, dt_w, np_s, dt_s, nb))

        # Poll staging for the in-flight experts before giving up and reading
        # them ourselves — overlaps with the background prefetch instead of
        # racing it with a duplicate read. Bounded spin (~20 ms).
        if awaited:
            t_wait = time.time()
            while awaited and (time.time() - t_wait) < 0.020:
                with self._lock:
                    still_waiting = []
                    for e in awaited:
                        ck = (wkey, e)
                        if ck in self._staging:
                            wbuf, sbuf = self._staging.pop(ck)
                            self._staging_bytes -= wbuf[0].nbytes + sbuf[0].nbytes
                            np_w, dt_w = wbuf
                            np_s, dt_s = sbuf
                            nb = np_w.nbytes + np_s.nbytes
                            self._insert(ck, (np_w, dt_w, np_s, dt_s, nb))
                            self.prefetch_hits += 1
                            self.misses -= 1   # retroactively correct the stat
                        elif ck not in self._inflight:
                            # prefetch failed/dropped: read it ourselves
                            miss_pairs.append((wkey, skey, e))
                        else:
                            still_waiting.append(e)
                    awaited = still_waiting
                if awaited:
                    time.sleep(0.001)
            # timed out: read whatever never landed on the critical path
            for e in awaited:
                miss_pairs.append((wkey, skey, e))

        if miss_pairs:
            if fused_layer_idx is not None:
                loaded = self._load_coalesced_fused(
                    [p[2] for p in miss_pairs], wkey, fused_layer_idx)
            else:
                loaded = self._load_coalesced(miss_pairs)
            for ck, entry in loaded.items():
                self._insert(ck, entry)
                self.bytes_read += entry[4]

        # Build the stack *before* evicting so freshly loaded slices are still
        # present (and held by ``ws``/``ss``, so eviction can't free them out
        # from under this call).
        ws, ss = [], []
        for e in experts:
            ck = (wkey, e)
            entry = self._pinned.get(ck) or self._od.get(ck)
            if len(entry) == 3:
                # pinned: already-wired mx.arrays (w, s, nbytes) — zero-copy.
                w, s, _ = entry
                ws.append(w)
                ss.append(s)
            else:
                # LRU (_od): pageable 5-tuple, wire only these K active experts;
                # the stack is consumed by the Metal kernel and released.
                np_w, dt_w, np_s, dt_s, _ = entry
                self._od.move_to_end(ck)
                ws.append(mx.array(np_w, dtype=dt_w))
                ss.append(mx.array(np_s, dtype=dt_s))

        self._evict()
        w_stack = mx.stack(ws, axis=0)
        s_stack = mx.stack(ss, axis=0)
        # Return lazy: the downstream kernel (polar_gather_qmv / prefill GEMM)
        # triggers the eval, flushing the graph once instead of firing a
        # per-projection CPU<->GPU barrier here. Output is bit-identical; only
        # the timing of Metal command submission changes.
        return w_stack, s_stack

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        with self._lock:
            self._staging.clear()
            self._staging_bytes = 0
            self._inflight.clear()

    def stats(self):
        served = self.hits + self.prefetch_hits   # neither incurred critical-path disk
        tot = served + self.misses
        return {
            # effective (latency) hit-rate: fraction served without a blocking read
            "hit_rate": (served / tot) if tot else 0.0,
            "cache_hit_rate": (self.hits / tot) if tot else 0.0,
            "prefetch_hit_rate": (self.prefetch_hits / tot) if tot else 0.0,
            "hits": self.hits,
            "prefetch_hits": self.prefetch_hits,
            "pin_hits": self._pin_hits,
            "misses": self.misses,
            "prefetched": self.prefetched,
            "prefetch_dropped": self.prefetch_dropped,
            "resident_experts": len(self._od) + len(self._pinned),
            "pinned_experts": len(self._pinned),
            "resident_gb": self.cur / 1e9,
            "bytes_read_gb": self.bytes_read / 1e9,
            "bytes_prefetched_gb": self.bytes_prefetched / 1e9,
            "bytes_total_gb": (self.bytes_read + self.bytes_prefetched) / 1e9,
            "read_runs": self.read_runs,
            "expert_reads": self.expert_reads,
            # experts per coalesced read: 1.0 = no coalescing, higher = #3 working
            "experts_per_read": (self.expert_reads / self.read_runs) if self.read_runs else 0.0,
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
        layer_idx: int = -1,
        is_trigger: bool = False,
        trit: bool = False,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_experts = num_experts
        self.bits = bits
        self.group_size = group_size
        # Ternary (1.58-bit) experts pack 20 base-3 trits per uint32 instead of
        # bit-packing; the kernels and dequant switch to base-3 decode off this.
        # Detect from the 3-entry codebook if the caller didn't say (self-describing).
        self.trit = bool(trit) or codebook.size == 3
        if self.trit:
            # The Metal kernels hardcode n_codes=3 under trit; a mismatched
            # codebook would read out of bounds on the GPU. Fail loud instead.
            if codebook.size != 3:
                raise ValueError(
                    "Ternary (trit) mode requires a 3-entry codebook, "
                    f"got size {codebook.size}."
                )
            # Match the resident layer (ternary is always bit-width 2) so the
            # (bits, group_size, trit) kernel-cache key stays consistent.
            self.bits = 2
        self._needs_rotation = needs_rotation
        # small resident tensors
        self.codebook = codebook
        self.signs = signs
        # streaming wiring (underscored so they are not treated as params)
        self._weight_key = weight_key
        self._scales_key = scales_key
        self._cache = cache
        # speculative-prefetch wiring: exactly one projection per layer is the
        # trigger (fires the next-layer prefetch once per token).
        self._layer_idx = layer_idx
        self._is_trigger = is_trigger
        # up_proj shares its (rotated) input with gate_proj (#4.3). Inferred from
        # the weight key so no other file (loader.py) needs to pass a flag.
        self._is_up = weight_key.endswith("up_proj.weight")
        # gate_proj drives the fused gate+up read (#fused): it runs one kernel
        # over the fused (2*out) stack, splits, and caches the up half.
        self._is_gate = weight_key.endswith("gate_proj.weight")
        self.freeze()

    def _dequantize_selected(self, w_sel: mx.array, s_sel: mx.array,
                             out_dims: int | None = None) -> mx.array:
        """Dequantize only the selected experts to float16 (prefill path).

        Mirrors PolarQuantizedSwitchLinear._dequantize_all but on an
        (n_sel, out, packed) stack instead of all num_experts. ``out_dims``
        defaults to self.output_dims; the fused gate+up path passes 2*out.
        """
        from turboquant_mlx.core.packing import unpack_indices, unpack_trits
        from turboquant_mlx.core.codebook import dequantize_scalar

        if out_dims is None:
            out_dims = self.output_dims
        n_sel = w_sel.shape[0]
        n_groups = self.input_dims // self.group_size
        if self.trit:
            idx = unpack_trits(w_sel, self.input_dims)
        else:
            idx = unpack_indices(w_sel, self.bits, self.input_dims)
        w_deq = dequantize_scalar(idx, self.codebook)
        w_deq = w_deq.reshape(n_sel, out_dims, n_groups, self.group_size)
        w_deq = w_deq * mx.expand_dims(s_sel, axis=-1)
        return w_deq.reshape(n_sel, out_dims, self.input_dims)

    def __call__(self, x, indices, sorted_indices=False):
        cache = self._cache
        # Fused gate+up fast path: up_proj returns the up half that gate_proj
        # already computed + cached this token — no rotation, routing sync,
        # disk read, or kernel. (Falls through to the normal path if absent,
        # e.g. gate hasn't run yet — the original gate/up tensors still exist.)
        if self._is_up and getattr(cache.reader, 'has_fused_gate_up', False):
            cached_gu = cache._fused_gate_up_cache.pop(self._layer_idx, None)
            if cached_gu is not None:
                return cached_gu[1]  # up_out, already shaped like a projection out

        if self._needs_rotation:
            if self._is_trigger:
                # gate_proj: compute the Hadamard rotation and cache it for the
                # sibling up_proj (identical layer input x).
                x = rotate_input(x, self.signs)
                self._cache._rotated_x_cache[self._layer_idx] = x
            elif self._is_up:
                # up_proj: reuse gate_proj's rotation if present (same x).
                xr = self._cache._rotated_x_cache.get(self._layer_idx)
                x = xr if xr is not None else rotate_input(x, self.signs)
            else:
                # down_proj: different x (post-SwiGLU activation), recompute.
                x = rotate_input(x, self.signs)

        n_tokens = 1 if x.ndim <= 2 else math.prod(x.shape[:-2])
        k = indices.shape[-1] if indices.ndim >= 1 else 1

        # Discover selected experts. This forces a GPU->CPU sync on the routing
        # indices, which are IDENTICAL across a layer's three projections (gate/
        # up/down) for a given token. The trigger projection (gate_proj) does the
        # sync once and caches the result under its layer_idx; the non-trigger
        # projections (up/down) reuse it, skipping 2/3 of the syncs per token.
        cached = (None if self._is_trigger
                  else cache._layer_routing_cache.get(self._layer_idx))
        if cached is not None:
            sel, idx_local_flat = cached
        else:
            # Trigger, or non-trigger cache miss: vectorized routing on GPU.
            # Avoids the GPU→CPU sync + Python scalar loops of the old path.
            flat = indices.reshape(-1).astype(mx.uint32)  # (K,) or (B*K,)

            # Sort to bring identical expert ids together.
            sort_order = mx.argsort(flat)          # positions that would sort flat
            sorted_flat = flat[sort_order]         # expert ids in ascending order

            # Unique mask: True at the first occurrence of each value.
            changed = mx.concatenate([
                mx.array([True], dtype=mx.bool_),
                sorted_flat[1:] != sorted_flat[:-1],
            ])
            sel_mx = sorted_flat[changed]          # unique sorted expert ids

            # Rank of each sorted position within the unique set (0-based).
            ranks_in_sorted = mx.cumsum(changed.astype(mx.uint32)) - 1  # (K,)

            # Inverse permutation of sort_order: inv_sort[sort_order[i]] = i.
            inv_sort = mx.argsort(sort_order)
            idx_local_flat = ranks_in_sorted[inv_sort]  # remap for original order

            # Materialise sel as a Python list — K≤4 integers, negligible cost.
            # Required by cache.gather / on_layer_start (Python dict keys).
            mx.eval(sel_mx)
            sel = [int(v) for v in sel_mx.tolist()]
            # Only the trigger caches its routing and fires the next-layer
            # speculative prefetch / calibration trace (once per token).
            if self._is_trigger:
                cache._layer_routing_cache[self._layer_idx] = (sel, idx_local_flat)
                cache.on_layer_start(self._layer_idx, sel)
                if n_tokens == 1:
                    cache.record_trace(self._layer_idx, sel)

        # Fused gate+up: gate_proj loads the fused (2*out) stack, runs ONE kernel,
        # splits the output into gate / up halves, caches the up half for the
        # sibling up_proj, and returns the gate half. Saves a kernel launch and a
        # disk read pair per MoE layer per token.
        if self._is_gate and getattr(cache.reader, 'has_fused_gate_up', False):
            fwkey = self._weight_key.replace("gate_proj.weight", "gate_up_proj.weight")
            fskey = self._scales_key.replace("gate_proj.scales", "gate_up_proj.scales")
            w_sel, s_sel = cache.gather_fused(fwkey, fskey, self._layer_idx, sel)
            result = self._project(x, indices, w_sel, s_sel, idx_local_flat,
                                   n_tokens, k, 2 * self.output_dims)
            # Output feature axis is last; first half is gate, second is up
            # (fused stacked gate rows then up rows along the output dim).
            gate_out, up_out = mx.split(result, 2, axis=-1)
            cache._fused_gate_up_cache[self._layer_idx] = (gate_out, up_out)
            return gate_out

        w_sel, s_sel = cache.gather(self._weight_key, self._scales_key, sel)
        return self._project(x, indices, w_sel, s_sel, idx_local_flat,
                             n_tokens, k, self.output_dims)

    def _project(self, x, indices, w_sel, s_sel, idx_local_flat,
                 n_tokens, k, out_dims):
        """Run the routed projection kernel for the selected experts and reshape.

        Shared by the normal per-projection path (``out_dims == output_dims``)
        and the fused gate+up path (``out_dims == 2*output_dims``, split by the
        caller). ``w_sel``/``s_sel`` are the gathered (n_sel, out_dims, ...) stacks.
        """
        # --- Batched multi-sequence decode (stream/batch_generate.py) --------
        # B sequences each generate one token per step, so their routing arrives
        # as indices of shape (B, 1, K): B sequences × K experts. Every
        # sequence's selection lives in this single tensor, so `sel` above is
        # already the UNION across the batch and the single cache.gather that
        # just ran loaded every expert any sequence needs — a popular expert is
        # read from disk once per step, not once per sequence (the whole point
        # of batched serving). The MoE math for the entire batch is then one
        # fused kernel call over the B×K routings (true parallel forward, no
        # per-sequence Python loop). on_layer_start / prefetch above likewise
        # saw the batch union, so speculative prefetch is batch-aware for free.
        #
        # Detection is deliberately strict — 3-D indices, batch > 1, singleton
        # token axis — so single-sequence decode and every prefill shape fall
        # through unchanged (byte-identical to the pre-batch implementation).
        is_batch_decode = (
            indices.ndim >= 3 and indices.shape[0] > 1 and indices.shape[1] == 1
        )
        if is_batch_decode:
            B = int(indices.shape[0])
            if n_tokens == B * k:
                # down_proj: x is already one activation vector per (seq, expert)
                # routing — (B, 1, K, 1, in) -> (B*K, in). (k == 1 collapses here
                # too, which is correct: one vector per routing either way.)
                x_rows = x.reshape(B * k, self.input_dims)
            else:
                # gate_proj / up_proj: one shared input vector per sequence
                # (B, 1, 1, 1, in). Broadcast it across that sequence's K experts
                # so each routing has its input row, aligned with idx_local_flat.
                x_tok = x.reshape(B, self.input_dims)
                x_rows = mx.broadcast_to(
                    x_tok.reshape(B, 1, self.input_dims),
                    (B, k, self.input_dims),
                ).reshape(B * k, self.input_dims)
            # idx_local_flat is (B*K,) in (sequence, expert-slot) order — the same
            # order as x_rows — so one multi_gather covers the whole batch. The
            # kernel chunks large N internally, so B*K needs no manual tiling.
            y = polar_multi_gather_qmv(
                w_sel, s_sel, self.codebook,
                x_rows, idx_local_flat, self.bits, self.group_size,
                trit=self.trit,
            )
            return y.reshape(list(indices.shape) + [1, out_dims])

        if n_tokens == 1:
            x_flat = x.reshape(-1)
            y = polar_gather_qmv(
                w_sel, s_sel, self.codebook,
                x_flat, idx_local_flat, self.bits, self.group_size,
                trit=self.trit,
            )
            return y.reshape(list(indices.shape) + [1, out_dims])

        if n_tokens == k:
            x_2d = x.reshape(k, self.input_dims)
            y = polar_multi_gather_qmv(
                w_sel, s_sel, self.codebook,
                x_2d, idx_local_flat, self.bits, self.group_size,
                trit=self.trit,
            )
            return y.reshape(list(indices.shape) + [1, out_dims])

        # Prefill (n_tokens > k): one shared input vector per token routed to k
        # experts. Preferred path mirrors the resident layer — tiled GEMM
        # directly on the packed weights (polar_gather_qmm), which reads the
        # weights in a single pass and materializes nothing. The kernel requires
        # ascending-sorted indices, so we flatten the (n_tokens, k) routings to
        # per-row form, sort, run, and un-sort the outputs. Falls back to the
        # dequant + gather_mm path when the kernel can't handle output_dims.
        if _gather_qmm_supports(out_dims):
            # Broadcast each token's shared x across its k experts -> per-routing
            # rows aligned with idx_local_flat (token-major flatten).
            x_rows = mx.broadcast_to(
                x.reshape(n_tokens, 1, self.input_dims),
                (n_tokens, k, self.input_dims),
            ).reshape(n_tokens * k, self.input_dims)
            idx_flat = idx_local_flat.reshape(-1)
            order = mx.argsort(idx_flat)
            inv = mx.argsort(order)
            y = polar_gather_qmm(
                w_sel, s_sel, self.codebook,
                x_rows[order], idx_flat[order].astype(mx.uint32),
                self.bits, self.group_size, trit=self.trit,
            )
            y = y[inv]  # restore original (token-major) order
            return y.reshape(list(indices.shape) + [1, out_dims])

        # Fallback: dequantize only the selected experts, gather_mm locally.
        w_deq = self._dequantize_selected(w_sel, s_sel, out_dims)
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
