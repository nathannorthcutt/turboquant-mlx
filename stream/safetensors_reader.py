"""Read individual MoE expert slices out of mmap'd safetensors shards.

TurboQuant stores each expert projection as a stacked tensor

    weight: (num_experts, output_dims, packed_cols)  uint32
    scales: (num_experts, output_dims, n_groups)      float16

contiguous along the expert axis, so expert ``e`` is a single contiguous byte
range. This reader builds an offset index across all shards and hands back
just the bytes for one expert as an ``mx.array`` — never materializing the
full (256, ...) tensor. That is the primitive the streaming MoE block uses to
keep resident memory bounded.

Only the dtypes that actually get streamed are supported (U32 weights, F16
scales); everything else in the model is loaded resident the normal way.
"""

from __future__ import annotations

import fcntl
import glob
import json
import os
import re
import struct
from dataclasses import dataclass

import numpy as np
import mlx.core as mx


def _layer_from_key(key: str) -> int | None:
    """Extract the transformer layer index from a full tensor key.

    e.g. ``model.layers.42.mlp.switch_mlp.gate_proj.weight`` -> 42.
    Returns None when the key is not layer-scoped.
    """
    m = re.search(r'\.layers\.(\d+)\.', key)
    return int(m.group(1)) if m else None

# macOS: tell the OS not to keep this fd's data in the unified buffer cache,
# so streaming 14+ GB of expert slices doesn't balloon resident page cache on
# a memory-constrained machine. Advisory; pairs with our own LRU expert cache.
_F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)

# safetensors dtype string -> (numpy dtype, itemsize, mlx dtype)
_DTYPES = {
    "U32": (np.uint32, 4, mx.uint32),
    "F16": (np.float16, 2, mx.float16),
    "F32": (np.float32, 4, mx.float32),
}


@dataclass
class _TensorLoc:
    file_idx: int          # index into self._mmaps
    dtype: str             # safetensors dtype string
    shape: tuple           # full tensor shape
    abs_begin: int         # absolute byte offset of element 0 in the file


class SafetensorsExpertReader:
    """Offset index + mmap pool for reading per-expert weight/scale slices.

    Parameters
    ----------
    model_path : str
        Directory holding ``model*.safetensors``.
    use_page_cache : bool
        If False (default), set ``F_NOCACHE`` so expert reads bypass the OS
        unified buffer cache — correct when the model is much larger than RAM
        (16 GB mini streaming a big MoE), where letting the page cache grow
        would thrash. If True ("trust the OS", Flash-MoE's finding), leave the
        page cache enabled so LRU-eviction re-reads come back from warm RAM
        instead of disk — a win on roomy machines where the model file fits in
        free RAM. See ``trust_os_ab.py`` for the measured tradeoff.
    """

    def __init__(self, model_path: str, use_page_cache: bool = False,
                 perm_path: str | None = None):
        self.model_path = model_path
        self.use_page_cache = use_page_cache
        # glob.escape so a model_path containing [ ] etc. still matches literally.
        all_files = sorted(glob.glob(os.path.join(glob.escape(model_path), "model*.safetensors")))
        # Companion files (model_wts-*.safetensors interleaved weight+scales,
        # model_fused-*.safetensors fused gate+up) also match the glob above;
        # keep them out of the main index and handle them separately.
        files = [f for f in all_files
                 if not os.path.basename(f).startswith(("model_wts", "model_fused"))]
        wts_files = [f for f in all_files if os.path.basename(f).startswith("model_wts")]
        fused_files = [f for f in all_files if os.path.basename(f).startswith("model_fused")]
        if not files:
            raise FileNotFoundError(f"No model*.safetensors in {model_path}")
        self._files = files
        self._fds: list[int] = []
        self._index: dict[str, _TensorLoc] = {}
        # Interleaved weight+scales companion files (see stream/repack_interleaved.py).
        # _wts_fds is a SEPARATE fd pool; _TensorLoc.file_idx in _wts_index indexes
        # into _wts_fds, never _fds — keep the two pools distinct to avoid fd mixups.
        self._wts_fds: list[int] = []
        self._wts_index: dict[str, _TensorLoc] = {}
        # Fused gate+up companion files (see stream/repack_fused_gate_up.py). Its
        # fds are a SEPARATE pool; _fused_index maps a transformer layer index to
        # that layer's fused weight/scales locations (one fused gate_up per layer).
        self._fused_fds: list[int] = []
        self._fused_index: dict[int, dict] = {}  # layer -> {"weight","scales","wkey","skey"}

        def _open_nocache(path: str) -> int:
            fd = os.open(path, os.O_RDONLY)
            if not use_page_cache:
                try:
                    fcntl.fcntl(fd, _F_NOCACHE, 1)
                except OSError:
                    pass  # F_NOCACHE is best-effort
            return fd

        # Set when main shards are replaced-in-place interleaved files
        # (repack_interleaved.py --replace).  In that case _wts_index entries
        # point into _fds (same fd pool) rather than a separate _wts_fds pool,
        # so we store the fd index relative to _fds.
        self._wts_use_main_fds: bool = False

        for fi, f in enumerate(files):
            fd = _open_nocache(f)
            self._fds.append(fd)
            header_len = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            header = json.loads(os.pread(fd, header_len, 8))
            data_start = 8 + header_len  # tensor data section begins here
            md = header.get("__metadata__") or {}
            replaced = md.get("replaced_originals") == "true"
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                begin, _end = meta["data_offsets"]
                loc = _TensorLoc(
                    file_idx=fi,
                    dtype=meta["dtype"],
                    shape=tuple(meta["shape"]),
                    abs_begin=data_start + begin,
                )
                if replaced and name.endswith("_wts"):
                    # Interleaved weight+scales rows embedded in the main shard.
                    # File index refers to _fds (not _wts_fds); flag this so the
                    # read path uses the right pool.
                    self._wts_index[name] = loc
                    self._wts_use_main_fds = True
                else:
                    self._index[name] = loc

        # Index the interleaved companion files exactly like the main files, but
        # into _wts_index / _wts_fds. Each entry maps a `<wkey>_wts` key to the
        # contiguous (E, w_stride + s_stride) U8 block on disk.
        for wfi, f in enumerate(wts_files):
            fd = _open_nocache(f)
            self._wts_fds.append(fd)
            header_len = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            header = json.loads(os.pread(fd, header_len, 8))
            data_start = 8 + header_len
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                begin, _end = meta["data_offsets"]
                self._wts_index[name] = _TensorLoc(
                    file_idx=wfi,
                    dtype=meta["dtype"],
                    shape=tuple(meta["shape"]),
                    abs_begin=data_start + begin,
                )
        # True when at least one weight+scales pair can be read in a single pread.
        self.has_interleaved: bool = bool(self._wts_index)

        # Index the fused gate+up companion files into _fused_index / _fused_fds.
        # Each companion holds `<prefix>gate_up_proj.weight` (E, 2*out, packed) and
        # `<prefix>gate_up_proj.scales` (E, 2*out, n_groups) per MoE layer; we key
        # by layer index since there is exactly one fused gate_up per layer.
        for ffi, f in enumerate(fused_files):
            fd = _open_nocache(f)
            self._fused_fds.append(fd)
            header_len = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
            header = json.loads(os.pread(fd, header_len, 8))
            data_start = 8 + header_len
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                layer = _layer_from_key(name)
                if layer is None:
                    continue
                begin, _end = meta["data_offsets"]
                loc = _TensorLoc(
                    file_idx=ffi,
                    dtype=meta["dtype"],
                    shape=tuple(meta["shape"]),
                    abs_begin=data_start + begin,
                )
                slot = self._fused_index.setdefault(layer, {})
                if name.endswith("gate_up_proj.weight"):
                    slot["weight"] = loc
                    slot["wkey"] = name
                elif name.endswith("gate_up_proj.scales"):
                    slot["scales"] = loc
                    slot["skey"] = name
        # True when at least one layer's fused gate+up can be read directly.
        self.has_fused_gate_up: bool = bool(self._fused_index)

        # Optional logical->physical expert remap. The router emits *logical*
        # expert ids; after a repack (see stream/repack.py) the on-disk order is
        # permuted, so we must translate before computing byte offsets.
        # self._perm[weight_key][logical_id] -> physical slot on disk.
        self._perm: dict[str, np.ndarray] = {}
        if perm_path is not None:
            self._load_perm(perm_path)

    def _load_perm(self, perm_path: str) -> None:
        with open(perm_path) as fh:
            perm_by_layer = json.load(fh)["perm"]
        # Cache the inverse permutation per layer so multiple weight keys in the
        # same layer share one array.
        inv_by_layer: dict[str, np.ndarray] = {}
        for name, loc in self._index.items():
            layer = _layer_from_key(name)
            if layer is None:
                continue
            layer_str = str(layer)
            if layer_str not in perm_by_layer:
                continue
            inv = inv_by_layer.get(layer_str)
            if inv is None:
                perm_list = np.asarray(perm_by_layer[layer_str], dtype=np.int64)
                # perm_list[physical_pos] = logical_id. Invert it:
                # inv[logical_id] = physical_pos.
                inv = np.empty(perm_list.shape[0], dtype=np.int32)
                inv[perm_list] = np.arange(perm_list.shape[0], dtype=np.int32)
                inv_by_layer[layer_str] = inv
            # Only remap stacked expert tensors whose expert axis matches.
            if loc.shape and loc.shape[0] == inv.shape[0]:
                self._perm[name] = inv

    # -- introspection -------------------------------------------------
    def has(self, key: str) -> bool:
        return key in self._index

    def shape(self, key: str) -> tuple:
        return self._index[key].shape

    def num_experts(self, key: str) -> int:
        return self._index[key].shape[0]

    # -- the hot path --------------------------------------------------
    def read_expert_np(self, key: str, expert: int):
        """Return ``(numpy_array, mlx_dtype)`` for slice ``[expert]``.

        This is the disk half of :meth:`read_expert`: a positional ``os.pread``
        (which releases the GIL and ignores the file offset, so it is safe to
        call concurrently on the same fd) plus a zero-copy ``np.frombuffer``
        view. The MLX array — which we keep on a single thread — is built by
        the caller. Splitting it this way lets the expert cache fan the
        per-layer ``pread``s across a thread pool without ever touching MLX
        from a worker thread.
        """
        loc = self._index[key]
        # Translate logical expert id -> physical slot on disk when a repack
        # permutation is active. No-op when self._perm is empty.
        if key in self._perm:
            expert = int(self._perm[key][expert])
        np_dt, itemsize, mlx_dt = _DTYPES[loc.dtype]
        per_expert = 1
        for d in loc.shape[1:]:
            per_expert *= d
        nbytes = per_expert * itemsize
        start = loc.abs_begin + expert * nbytes
        # pread on an F_NOCACHE fd: copies just this expert's bytes, without
        # growing the resident page cache.
        raw = os.pread(self._fds[loc.file_idx], nbytes, start)
        buf = np.frombuffer(raw, dtype=np_dt, count=per_expert).reshape(loc.shape[1:])
        return buf, mlx_dt

    def read_range_np(self, key: str, e_start: int, count: int):
        """Read ``count`` *consecutive* experts [e_start, e_start+count) in ONE
        positional ``pread`` and return ``(numpy_array, mlx_dtype)`` of shape
        ``(count, d0, d1, ...)``. Coalescing adjacent experts into a single
        larger read is the win behind the co-activation on-disk layout (#3):
        the streaming cache calls this once per contiguous run instead of once
        per expert, cutting syscalls and turning random reads into sequential.
        """
        loc = self._index[key]
        np_dt, itemsize, mlx_dt = _DTYPES[loc.dtype]
        per_expert = 1
        for d in loc.shape[1:]:
            per_expert *= d
        each = per_expert * itemsize

        if key in self._perm:
            # Logical range [e_start, e_start+count) may not be contiguous on
            # disk after a repack. Map to physical slots and coalesce.
            phys = self._perm[key][e_start:e_start + count].astype(np.int64)
            p_min = int(phys.min())
            p_max = int(phys.max())
            if p_max - p_min == count - 1:
                # Physical slots form one contiguous run — the repack win: one
                # pread covers the whole (co-activated) run. Read it in physical
                # order, then reorder so output index i = logical expert
                # e_start+i (whose physical slot is phys[i]).
                start = loc.abs_begin + p_min * each
                raw = os.pread(self._fds[loc.file_idx], each * count, start)
                block = np.frombuffer(
                    raw, dtype=np_dt, count=per_expert * count
                ).reshape((count, *loc.shape[1:]))
                reorder = phys - p_min  # block[reorder[i]] is logical e_start+i
                return block[reorder], mlx_dt
            # Non-contiguous physical layout: fall back to per-expert reads.
            out = np.empty((count, *loc.shape[1:]), dtype=np_dt)
            for i in range(count):
                buf, _ = self.read_expert_np(key, e_start + i)
                out[i] = buf
            return out, mlx_dt

        start = loc.abs_begin + e_start * each
        raw = os.pread(self._fds[loc.file_idx], each * count, start)
        buf = np.frombuffer(raw, dtype=np_dt, count=per_expert * count).reshape(
            (count, *loc.shape[1:]))
        return buf, mlx_dt

    # -- interleaved weight+scales (one pread covers both) -------------
    def read_expert_pair_np(self, wkey: str, skey: str, expert: int):
        """Return ``((w_np, w_mlx_dt), (s_np, s_mlx_dt))`` for one expert.

        When an interleaved companion file covers ``(wkey, skey)`` this issues a
        SINGLE ``pread`` spanning the expert's weight AND scales bytes (they are
        laid out contiguously on disk — see stream/repack_interleaved.py). With
        no companion it falls back to two separate reads: identical output, no
        speedup. Logical->physical expert remap (repack perm) is applied in both
        paths.
        """
        wts_key = wkey + "_wts"
        if wts_key in self._wts_index:
            loc = self._wts_index[wts_key]
            # Same logical->physical translation as read_expert_np. The companion
            # rows are in on-disk (physical) order, so we remap before indexing.
            if wkey in self._perm:
                expert = int(self._perm[wkey][expert])
            wloc = self._index[wkey]
            sloc = self._index[skey]
            row_bytes = loc.shape[1]  # w_stride + s_stride

            w_stride = 1
            for d in wloc.shape[1:]:
                w_stride *= d
            w_stride *= 4  # U32

            # replaced_originals: interleaved rows live in the main shard fds.
            fds = self._fds if self._wts_use_main_fds else self._wts_fds
            start = loc.abs_begin + expert * row_bytes
            raw = os.pread(fds[loc.file_idx], row_bytes, start)

            w_np_dt = _DTYPES["U32"][0]
            s_np_dt = _DTYPES["F16"][0]
            _, _, w_mlx_dt = _DTYPES["U32"]
            _, _, s_mlx_dt = _DTYPES["F16"]

            w_buf = np.frombuffer(raw[:w_stride], dtype=w_np_dt).reshape(wloc.shape[1:])
            s_buf = np.frombuffer(raw[w_stride:], dtype=s_np_dt).reshape(sloc.shape[1:])
            return (w_buf, w_mlx_dt), (s_buf, s_mlx_dt)

        # Fallback: two separate reads (read_expert_np applies perm itself).
        return self.read_expert_np(wkey, expert), self.read_expert_np(skey, expert)

    def read_range_pair_np(self, wkey: str, skey: str, e_start: int, count: int):
        """Return ``((w_batch, w_mlx_dt), (s_batch, s_mlx_dt))`` for ``count``
        experts starting at ``e_start``; ``w_batch.shape == (count, *wshape[1:])``.

        One ``pread`` when an interleaved companion covers the pair (and the
        physical run is contiguous), else two range reads. Mirrors read_range_np's
        perm handling: a logical range that maps to a contiguous physical run is a
        single read + reorder; a fragmented range falls back to per-expert pair
        reads.
        """
        wts_key = wkey + "_wts"
        if wts_key in self._wts_index:
            loc = self._wts_index[wts_key]
            wloc = self._index[wkey]
            sloc = self._index[skey]
            row_bytes = loc.shape[1]  # w_stride + s_stride

            w_stride = 1
            for d in wloc.shape[1:]:
                w_stride *= d
            w_stride *= 4  # U32

            w_np_dt = _DTYPES["U32"][0]
            s_np_dt = _DTYPES["F16"][0]
            _, _, w_mlx_dt = _DTYPES["U32"]
            _, _, s_mlx_dt = _DTYPES["F16"]

            fds = self._fds if self._wts_use_main_fds else self._wts_fds
            if wkey in self._perm:
                phys = self._perm[wkey][e_start:e_start + count].astype(np.int64)
                p_min, p_max = int(phys.min()), int(phys.max())
                if p_max - p_min == count - 1:
                    # Contiguous physical run: one pread, then reorder to logical.
                    start = loc.abs_begin + p_min * row_bytes
                    raw = os.pread(fds[loc.file_idx], row_bytes * count, start)
                    block = np.frombuffer(raw, dtype=np.uint8).reshape(count, row_bytes)
                    reorder = phys - p_min  # block[reorder[i]] is logical e_start+i
                    block = block[reorder]
                else:
                    # Fragmented physical layout: per-expert pair reads. Pass
                    # LOGICAL ids so read_expert_pair_np applies the perm itself.
                    rows = np.empty((count, row_bytes), dtype=np.uint8)
                    for i in range(count):
                        (wp, _), (sp, _) = self.read_expert_pair_np(wkey, skey, e_start + i)
                        rows[i, :w_stride] = wp.reshape(-1).view(np.uint8)
                        rows[i, w_stride:] = sp.reshape(-1).view(np.uint8)
                    block = rows
            else:
                start = loc.abs_begin + e_start * row_bytes
                raw = os.pread(fds[loc.file_idx], row_bytes * count, start)
                block = np.frombuffer(raw, dtype=np.uint8).reshape(count, row_bytes)

            # Split each row at w_stride and reinterpret. tobytes() gives a
            # contiguous, writable buffer so the reshaped views are well-defined
            # even after the fancy-index reorder above.
            w_block = np.frombuffer(
                np.ascontiguousarray(block[:, :w_stride]).tobytes(), dtype=w_np_dt
            ).reshape(count, *wloc.shape[1:])
            s_block = np.frombuffer(
                np.ascontiguousarray(block[:, w_stride:]).tobytes(), dtype=s_np_dt
            ).reshape(count, *sloc.shape[1:])
            return (w_block, w_mlx_dt), (s_block, s_mlx_dt)

        # Fallback: two separate range reads.
        w_batch, w_dt = self.read_range_np(wkey, e_start, count)
        s_batch, s_dt = self.read_range_np(skey, e_start, count)
        return (w_batch, w_dt), (s_batch, s_dt)

    # -- fused gate+up (one fused expert covers both projections) ------
    def read_fused_gate_up_pair_np(self, layer_idx: int, expert: int):
        """Read fused gate+up weight and scales for one expert.

        Returns ``((w_np, w_dt), (s_np, s_dt))`` where ``w_np`` has shape
        ``(2*out_features, in_features/32)`` — the first ``out_features`` rows are
        the gate projection, the second ``out_features`` are the up projection
        (see stream/repack_fused_gate_up.py). Reads the fused expert directly
        (weight + scales) from the companion; raises ``RuntimeError`` when no
        fused companion covers ``layer_idx``. Applies the same logical->physical
        expert remap as read_expert_pair_np when a repack perm is active.
        """
        slot = self._fused_index.get(layer_idx)
        if not slot or "weight" not in slot or "scales" not in slot:
            raise RuntimeError(
                f"no fused gate_up companion for layer {layer_idx}")
        wloc, sloc = slot["weight"], slot["scales"]

        # Fused rows follow the original on-disk (physical) expert order, so we
        # remap the logical id via the GATE key's perm (fused == gate ++ up).
        gate_wkey = slot["wkey"].replace("gate_up_proj.weight", "gate_proj.weight")
        if gate_wkey in self._perm:
            expert = int(self._perm[gate_wkey][expert])

        w_np_dt, w_isz, w_mlx_dt = _DTYPES[wloc.dtype]
        s_np_dt, s_isz, s_mlx_dt = _DTYPES[sloc.dtype]
        w_per = 1
        for d in wloc.shape[1:]:
            w_per *= d
        s_per = 1
        for d in sloc.shape[1:]:
            s_per *= d

        w_raw = os.pread(self._fused_fds[wloc.file_idx],
                         w_per * w_isz, wloc.abs_begin + expert * w_per * w_isz)
        s_raw = os.pread(self._fused_fds[sloc.file_idx],
                         s_per * s_isz, sloc.abs_begin + expert * s_per * s_isz)
        w_buf = np.frombuffer(w_raw, dtype=w_np_dt, count=w_per).reshape(wloc.shape[1:])
        s_buf = np.frombuffer(s_raw, dtype=s_np_dt, count=s_per).reshape(sloc.shape[1:])
        return (w_buf, w_mlx_dt), (s_buf, s_mlx_dt)

    def read_fused_gate_up_range_np(self, layer_idx: int, e_start: int, count: int):
        """Range read fused gate+up for experts ``[e_start, e_start+count)``.

        Returns ``((w_batch, w_dt), (s_batch, s_dt))`` with
        ``w_batch.shape == (count, 2*out_features, in_features/32)``. One pread
        per tensor for the whole run (mirrors read_range_np); a repack perm that
        fragments the physical run falls back to per-expert fused reads.
        """
        slot = self._fused_index.get(layer_idx)
        if not slot or "weight" not in slot or "scales" not in slot:
            raise RuntimeError(
                f"no fused gate_up companion for layer {layer_idx}")
        wloc, sloc = slot["weight"], slot["scales"]

        w_np_dt, w_isz, w_mlx_dt = _DTYPES[wloc.dtype]
        s_np_dt, s_isz, s_mlx_dt = _DTYPES[sloc.dtype]
        w_per = 1
        for d in wloc.shape[1:]:
            w_per *= d
        s_per = 1
        for d in sloc.shape[1:]:
            s_per *= d
        w_each, s_each = w_per * w_isz, s_per * s_isz
        w_fd, s_fd = self._fused_fds[wloc.file_idx], self._fused_fds[sloc.file_idx]

        gate_wkey = slot["wkey"].replace("gate_up_proj.weight", "gate_proj.weight")
        if gate_wkey in self._perm:
            phys = self._perm[gate_wkey][e_start:e_start + count].astype(np.int64)
            p_min, p_max = int(phys.min()), int(phys.max())
            if p_max - p_min == count - 1:
                # Contiguous physical run: one pread each, then reorder to logical.
                w_raw = os.pread(w_fd, w_each * count, wloc.abs_begin + p_min * w_each)
                s_raw = os.pread(s_fd, s_each * count, sloc.abs_begin + p_min * s_each)
                w_block = np.frombuffer(w_raw, dtype=w_np_dt, count=w_per * count).reshape(
                    (count, *wloc.shape[1:]))
                s_block = np.frombuffer(s_raw, dtype=s_np_dt, count=s_per * count).reshape(
                    (count, *sloc.shape[1:]))
                reorder = phys - p_min  # block[reorder[i]] is logical e_start+i
                return (w_block[reorder], w_mlx_dt), (s_block[reorder], s_mlx_dt)
            # Fragmented physical layout: per-expert fused reads (pass LOGICAL ids).
            w_out = np.empty((count, *wloc.shape[1:]), dtype=w_np_dt)
            s_out = np.empty((count, *sloc.shape[1:]), dtype=s_np_dt)
            for i in range(count):
                (wb, _), (sb, _) = self.read_fused_gate_up_pair_np(layer_idx, e_start + i)
                w_out[i] = wb
                s_out[i] = sb
            return (w_out, w_mlx_dt), (s_out, s_mlx_dt)

        w_raw = os.pread(w_fd, w_each * count, wloc.abs_begin + e_start * w_each)
        s_raw = os.pread(s_fd, s_each * count, sloc.abs_begin + e_start * s_each)
        w_block = np.frombuffer(w_raw, dtype=w_np_dt, count=w_per * count).reshape(
            (count, *wloc.shape[1:]))
        s_block = np.frombuffer(s_raw, dtype=s_np_dt, count=s_per * count).reshape(
            (count, *sloc.shape[1:]))
        return (w_block, w_mlx_dt), (s_block, s_mlx_dt)

    def read_expert(self, key: str, expert: int) -> mx.array:
        """Return slice ``[expert]`` of a stacked tensor as an mx.array.

        For a tensor of shape (E, d0, d1, ...) this returns shape (d0, d1, ...)
        and copies only that expert's bytes into an MLX buffer.
        """
        buf, mlx_dt = self.read_expert_np(key, expert)
        return mx.array(buf, dtype=mlx_dt)

    def close(self):
        for fd in self._fds:
            try:
                os.close(fd)
            except Exception:
                pass
        self._fds = []
        for fd in self._wts_fds:
            try:
                os.close(fd)
            except Exception:
                pass
        self._wts_fds = []
        for fd in self._fused_fds:
            try:
                os.close(fd)
            except Exception:
                pass
        self._fused_fds = []

    def __del__(self):
        self.close()
