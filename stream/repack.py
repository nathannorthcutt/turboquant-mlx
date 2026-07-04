"""Physically reorder a TurboQuant MoE checkpoint's expert tensors on disk.

Two ordering modes are supported; both produce the same output format (a
per-layer physical ordering applied to the ``switch_mlp`` expert stacks) and
differ only in how that ordering is derived:

**Co-activation mode (default, ``--perm``)**
Given a per-layer permutation ``perm.json`` from
``calibrate_experts.py analyze`` (shape ``{"perm": {layer: [expert_ids ...]}}``,
where ``perm[layer][physical_pos] = logical_id``), this writes a copy of the
model in which, for every MoE layer, the ``switch_mlp`` expert stacks are
reordered along their expert axis so that::

    new_tensor[i] = old_tensor[perm[layer][i]]

i.e. physical slot ``i`` of the output holds what used to be logical expert
``perm[layer][i]``. Frequently co-activated experts thus land adjacent on disk,
so the streaming reader coalesces a token's selected experts into fewer, longer
contiguous ``pread``s.

**Frequency-sort mode (``--freq-sort``)**
Here the ``--perm`` argument is instead an access-frequency *histogram* in the
shape produced by ``ExpertCache.dump_histogram`` (see
``stream/streaming_switch.py``)::

    {"hist": [[layer, expert, count], ...], ...}

Within each layer the experts are sorted by descending access count, so physical
slot 0 holds the single most-accessed expert, slot 1 the next, and so on.
Experts absent from the histogram (zero recorded accesses) are appended last in
their original id order. The derived ordering has the same
``perm[layer][physical_pos] = logical_id`` meaning as co-activation mode and is
applied identically. The intent differs, though: rather than making reads
faster, a frequency-first layout keeps the hot working set in the low-offset
region of each stacked tensor, so the OS page cache and the LRU expert cache
naturally retain it and reads for hot experts are *avoided* entirely.

Unlike ``repack_experts.py`` (which also permutes the *router* rows, making the
model emit physical ids directly), this tool reorders **only** the expert
stacks. The router still emits *logical* ids at runtime; the reader translates
logical -> physical via its ``perm_path`` argument (see
``SafetensorsExpertReader``). The output records a ``__metadata__`` marker
``{"repacked": "true", "perm_hash": <sha256 of perm.json>}`` so a repacked model
can be detected and matched to the perm that produced it.

Byte layout is preserved: reordering along axis 0 does not change any tensor's
byte size, so every tensor's ``data_offsets`` are identical to the input — only
the bytes of the reordered expert stacks move (and the header gains the metadata
marker). The reorder is done at the byte level (rows of ``prod(shape[1:]) *
itemsize`` bytes are gathered), so it is dtype-agnostic and never has to
interpret exotic element types.

Examples:
  # co-activation ordering from a calibrate_experts perm
  python -m turboquant_mlx.stream.repack \
      --model .../qwen3.5-122b-tq3 --perm /tmp/perm_122b.json \
      --out /tmp/qwen3.5-122b-tq3-coact

  # frequency ordering from a dump_histogram histogram
  python -m turboquant_mlx.stream.repack \
      --model .../qwen3-235b-tq3 --perm /tmp/histogram.json --freq-sort \
      --out /tmp/qwen3-235b-tq3-freqsort
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import struct
import sys

import numpy as np

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

# safetensors dtype string -> element size in bytes. We only ever need the
# *size* of an element to reorder axis 0 (whole rows move as opaque bytes), so
# this covers every safetensors dtype without any numpy interpretation.
_ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def _layer_from_key(key: str) -> int | None:
    """Extract the transformer layer index from a full tensor key."""
    m = _LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def _reorder_axis0(blob: bytes, shape: list[int], itemsize: int,
                   perm_list) -> bytes:
    """Return ``blob`` with axis-0 rows gathered so ``new[i] = old[perm[i]]``.

    ``blob`` is the raw little-endian bytes of a tensor of the given ``shape``
    and element size. The reorder is performed on opaque fixed-size rows, so it
    is independent of the actual element dtype.
    """
    num_experts = shape[0]
    row_bytes = itemsize
    for d in shape[1:]:
        row_bytes *= d
    rows = np.frombuffer(blob, dtype=np.uint8).reshape(num_experts, row_bytes)
    idx = np.asarray(perm_list, dtype=np.int64)
    return rows[idx].tobytes()


def _experts_per_layer(shards) -> dict[int, int]:
    """Peek every shard header to find each MoE layer's expert count ``E``.

    ``E`` (the length of a ``switch_mlp`` stack's leading axis) is needed to
    build a *full-length* frequency permutation — the histogram only lists
    experts that were actually accessed, so zero-access experts have to be
    reconstructed from the model's own tensor shapes. Only the header is read
    (a few KB per shard), not the tensor data.
    """
    counts: dict[int, int] = {}
    for path in shards:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for k, v in header.items():
            if k == "__metadata__":
                continue
            layer = _layer_from_key(k)
            if layer is None or "switch_mlp" not in k:
                continue
            shape = v.get("shape")
            if shape:
                # Same layer's tensors share E; last writer wins (identical).
                counts[layer] = shape[0]
    return counts


def _hist_to_counts(data: dict) -> dict[int, dict[int, int]]:
    """Normalise a dump_histogram payload to ``{layer: {expert: count}}``.

    Accepts the canonical list-of-triples shape written by
    ``ExpertCache.dump_histogram`` (``{"hist": [[layer, expert, count], ...]}``)
    and, defensively, a nested-dict shape (``{"hist": {layer: {expert: count}}}``)
    or a bare payload without the ``"hist"`` wrapper.
    """
    hist = data.get("hist", data) if isinstance(data, dict) else data
    per_layer: dict[int, dict[int, int]] = {}
    if isinstance(hist, list):
        for row in hist:
            lyr, e, c = int(row[0]), int(row[1]), int(row[2])
            per_layer.setdefault(lyr, {})[e] = per_layer.setdefault(lyr, {}).get(e, 0) + c
    elif isinstance(hist, dict):
        for lyr_str, experts in hist.items():
            lyr = int(lyr_str)
            bucket = per_layer.setdefault(lyr, {})
            for e_str, c in experts.items():
                bucket[int(e_str)] = bucket.get(int(e_str), 0) + int(c)
    else:
        raise ValueError("unrecognised histogram shape (expected list or dict)")
    return per_layer


def _freq_perm_by_layer(data: dict, experts_per_layer: dict[int, int]) -> dict:
    """Derive ``{layer_str: [logical_id ...]}`` from an access histogram.

    Slot ``i`` holds the ``i``-th most-accessed expert (descending count, ties
    broken by ascending expert id for determinism). Experts absent from the
    histogram (zero accesses) are appended last in their original id order. The
    result has the same meaning as a co-activation perm:
    ``perm[layer][physical_pos] = logical_id``.
    """
    per_layer_counts = _hist_to_counts(data)
    perm_by_layer: dict[str, list[int]] = {}
    for layer, E in experts_per_layer.items():
        counts = per_layer_counts.get(layer, {})
        # Only experts within [0, E) are valid physical/logical ids.
        present = sorted(
            (e for e in counts if 0 <= e < E),
            key=lambda e: (-counts[e], e),
        )
        present_set = set(present)
        absent = [e for e in range(E) if e not in present_set]
        perm_by_layer[str(layer)] = present + absent
    return perm_by_layer


def _repack_shard(src_path: str, dst_path: str, perm_by_layer: dict,
                  metadata: dict) -> tuple[int, int, int]:
    """Rewrite one shard with expert stacks reordered.

    Returns ``(n_tensors, n_reordered, bytes_written)``.
    """
    with open(src_path, "rb") as f:
        raw = f.read()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    data_start = 8 + n

    entries = [(k, v) for k, v in header.items() if k != "__metadata__"]

    # Build the new header: __metadata__ (merged) first, then every tensor with
    # its ORIGINAL data_offsets (byte sizes are unchanged by an axis-0 reorder,
    # so the offsets stay valid verbatim).
    new_header: dict = {}
    md = dict(header.get("__metadata__") or {})
    md.update(metadata)
    new_header["__metadata__"] = md
    for k, v in entries:
        new_header[k] = v

    hjson = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    hjson += b" " * ((-len(hjson)) % 8)  # pad header to 8-byte alignment
    base = 8 + len(hjson)               # data section start in the output file

    n_reordered = 0
    written = 0
    with open(dst_path, "wb") as out:
        out.write(struct.pack("<Q", len(hjson)))
        out.write(hjson)
        for k, v in entries:
            begin, end = v["data_offsets"]
            blob = raw[data_start + begin:data_start + end]
            layer = _layer_from_key(k)
            if layer is not None and "switch_mlp" in k:
                perm_list = perm_by_layer.get(str(layer))
                shape = v["shape"]
                if perm_list is not None and shape and shape[0] == len(perm_list):
                    itemsize = _ITEMSIZE[v["dtype"]]
                    blob = _reorder_axis0(blob, shape, itemsize, perm_list)
                    n_reordered += 1
            # Seek to the tensor's declared offset before writing, so the output
            # is correct regardless of the header's key ordering.
            out.seek(base + begin)
            out.write(blob)
            written += len(blob)

    return len(entries), n_reordered, written


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Input model dir (local path).")
    ap.add_argument("--perm", required=True,
                    help="Co-activation perm.json from calibrate_experts, OR (with "
                         "--freq-sort) an access histogram from dump_histogram.")
    ap.add_argument("--out", required=True, help="Output model dir.")
    ap.add_argument("--freq-sort", action="store_true",
                    help="Interpret --perm as a dump_histogram access histogram and "
                         "order each layer's experts by descending access count "
                         "(most-accessed first), instead of using a co-activation perm.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite a non-empty --out directory.")
    args = ap.parse_args()

    # Resolve HF-cache style refs when available; fall back to the raw path so
    # the tool stays usable standalone.
    try:
        from turboquant_mlx.generate import resolve_model_path
        src = str(resolve_model_path(args.model))
    except Exception:
        src = args.model

    if not os.path.isdir(src):
        print(f"[repack] error: model dir not found: {src}", file=sys.stderr)
        return 2

    if os.path.exists(args.out) and os.path.isdir(args.out) and os.listdir(args.out):
        if not args.force:
            print(f"[repack] error: --out {args.out} exists and is non-empty "
                  f"(pass --force to overwrite)", file=sys.stderr)
            return 2
    os.makedirs(args.out, exist_ok=True)

    with open(args.perm, "rb") as f:
        perm_bytes = f.read()
    perm_hash = hashlib.sha256(perm_bytes).hexdigest()
    perm_data = json.loads(perm_bytes)

    shards = sorted(glob.glob(os.path.join(glob.escape(src), "model*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no model*.safetensors in {src}")

    if args.freq_sort:
        # Derive the physical ordering from the access histogram. Needs each
        # layer's expert count (from the model itself) so zero-access experts
        # can be placed last. Output shape matches the co-activation path:
        # {layer_str: [logical_id ...]} with perm[i] = logical id in slot i.
        experts_per_layer = _experts_per_layer(shards)
        perm_by_layer = _freq_perm_by_layer(perm_data, experts_per_layer)
        print(f"[repack] freq-sort: derived ordering for {len(perm_by_layer)} "
              f"layer(s) from histogram", flush=True)
        metadata = {"repacked": "true", "freq_sort": "true", "perm_hash": perm_hash}
    else:
        perm_by_layer = perm_data["perm"]  # {layer_str: [ids ...]}
        metadata = {"repacked": "true", "perm_hash": perm_hash}

    # Copy everything that is NOT a weight shard (config, tokenizer, index, ...).
    for name in os.listdir(src):
        if name.endswith(".safetensors"):
            continue
        s = os.path.join(src, name)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.out, name))

    total = len(shards)
    total_reordered = 0
    for i, shard in enumerate(shards, 1):
        dst = os.path.join(args.out, os.path.basename(shard))
        n_tensors, n_reordered, written = _repack_shard(
            shard, dst, perm_by_layer, metadata)
        total_reordered += n_reordered
        print(f"[repack] shard {i}/{total} {os.path.basename(shard)}: "
              f"{n_tensors} tensors, {n_reordered} reordered, "
              f"{written / (1024 * 1024):.1f} MB written", flush=True)

    print(f"[repack] done -> {args.out} "
          f"({total_reordered} expert stacks reordered, perm_hash={perm_hash[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
