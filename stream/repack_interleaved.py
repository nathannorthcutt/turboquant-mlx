"""Co-locate each expert's packed weight and scales contiguously on disk.

The streaming reader loads one MoE expert projection with two ``pread``s:
one for the ``(E, out, packed_cols)`` U32 weight slice and one for the
``(E, out, n_groups)`` F16 scales slice. That is two syscalls per expert, per
projection, per selected expert — 2 × 3 projections × L layers × K selected
experts per decoded token.

This tool writes a *companion* safetensors file next to each original shard in
which, for every ``(layer, projection)`` pair, the weight and scales bytes for
each expert are packed into ONE contiguous row::

    row e = [ weight_bytes_e | scales_bytes_e ]

so the reader can cover both with a single ``pread``. The companion file holds
only these combined ``<weight_key>_wts`` tensors (typed ``U8`` — the row is an
opaque byte blob the reader splits at ``w_stride_bytes``); non-expert tensors
stay in the original shards and are read the normal way.

The interleave is a pure byte-level rearrangement (like ``repack.py``'s axis-0
reorder): weight rows and scales rows are concatenated per expert without
interpreting the element dtype. The companion header records
``{"interleaved": "true", "weight_dtype": "U32", "scales_dtype": "F16"}`` so the
reader can auto-detect it.

Note on disk cost: the companion DUPLICATES the expert weight+scales bytes
(which are the bulk of a MoE checkpoint), so ``--in-place`` roughly adds the
model's expert-byte footprint again on disk — it avoids copying the *non-expert*
tensors and the whole-model duplication that ``--out`` incurs, not the expert
bytes themselves.

Examples:
  # default: write companions in-place beside the model's existing shards
  python -m turboquant_mlx.stream.repack_interleaved --model .../qwen3.6-35b-tq3

  # copy the model to --out and write companions there
  python -m turboquant_mlx.stream.repack_interleaved \
      --model .../qwen3.6-35b-tq3 --out /tmp/qwen3.6-35b-tq3-wts
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import struct
import sys

import numpy as np

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

# safetensors dtype string -> element size in bytes. We only need element sizes
# to carve fixed-size per-expert rows out of opaque bytes, so this stays
# dtype-agnostic (mirrors repack.py's _ITEMSIZE).
_ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

_WTS_SUFFIX = "_wts"


def _companion_name(shard_basename: str) -> str:
    """``model-00001-of-00030.safetensors`` -> ``model_wts-00001-of-00030.safetensors``.

    Also handles an unsharded ``model.safetensors`` -> ``model_wts.safetensors``.
    """
    assert shard_basename.startswith("model")
    return "model_wts" + shard_basename[len("model"):]


def _is_companion(basename: str) -> bool:
    return basename.startswith("model_wts")


def _row_bytes(shape, dtype: str) -> int:
    """Bytes for one expert (all dims except the leading expert axis)."""
    n = _ITEMSIZE[dtype]
    for d in shape[1:]:
        n *= d
    return n


def _build_interleaved_shard(src_path: str, dst_path: str,
                             include_resident: bool = False) -> tuple[int, int]:
    """Write the companion for one shard. Returns ``(n_pairs, bytes_written)``.

    For every ``switch_mlp`` ``.weight`` tensor whose ``.scales`` sibling lives
    in the SAME shard, emit one ``<weight_key>_wts`` U8 tensor of shape
    ``(E, w_stride + s_stride)`` whose row ``e`` is ``weight_e ++ scales_e``.

    When ``include_resident`` is set (``--replace`` mode), every OTHER tensor of
    the source shard — non-expert resident weights, router, norms, and any
    expert tensor not folded into a pair — is also copied verbatim, so the
    output is a *complete, self-contained* replacement for the original shard
    rather than an additive companion. The interleaved expert bytes then exist
    only once (in this file), eliminating the duplication that a plain companion
    incurs.
    """
    with open(src_path, "rb") as f:
        raw = f.read()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    data_start = 8 + n

    # Discover (weight, scales) pairs that both live in this shard.
    combined: list[tuple[str, bytes, int, int]] = []  # (wts_key, bytes, E, row)
    consumed: set[str] = set()  # source keys folded into a pair (skip on copy)
    for wkey, wmeta in header.items():
        if wkey == "__metadata__":
            continue
        if "switch_mlp" not in wkey or not wkey.endswith(".weight"):
            continue
        skey = wkey[: -len(".weight")] + ".scales"
        smeta = header.get(skey)
        if smeta is None:
            # Scales sibling is in a different shard (uncommon). Skip — the
            # reader falls back to two separate reads for this pair.
            continue

        wshape, sshape = wmeta["shape"], smeta["shape"]
        if not wshape or not sshape or wshape[0] != sshape[0]:
            continue
        E = wshape[0]

        wbegin, wend = wmeta["data_offsets"]
        sbegin, send = smeta["data_offsets"]
        wblob = raw[data_start + wbegin:data_start + wend]
        sblob = raw[data_start + sbegin:data_start + send]

        w_stride = _row_bytes(wshape, wmeta["dtype"])
        s_stride = _row_bytes(sshape, smeta["dtype"])
        if len(wblob) != E * w_stride or len(sblob) != E * s_stride:
            raise ValueError(
                f"{wkey}: byte size mismatch (w {len(wblob)} vs {E*w_stride}, "
                f"s {len(sblob)} vs {E*s_stride})")

        wrows = np.frombuffer(wblob, dtype=np.uint8).reshape(E, w_stride)
        srows = np.frombuffer(sblob, dtype=np.uint8).reshape(E, s_stride)
        rows = np.ascontiguousarray(np.hstack([wrows, srows]))  # (E, w+s)
        combined.append((wkey + _WTS_SUFFIX, rows.tobytes(), E, w_stride + s_stride))
        consumed.add(wkey)
        consumed.add(skey)

    if not combined:
        return 0, 0

    # In --replace mode, gather every tensor NOT folded into a pair so the
    # output shard loses no data when the original is deleted.
    residents: list[tuple[str, dict, bytes]] = []  # (key, meta, bytes)
    if include_resident:
        for key, meta in header.items():
            if key == "__metadata__" or key in consumed:
                continue
            begin, end = meta["data_offsets"]
            residents.append((key, meta, raw[data_start + begin:data_start + end]))

    # Lay tensors out back-to-back and build the header.
    md = {
        "interleaved": "true",
        "weight_dtype": "U32",
        "scales_dtype": "F16",
    }
    if include_resident:
        md["replaced_originals"] = "true"
    new_header: dict = {"__metadata__": md}
    offset = 0
    for wts_key, blob, E, row in combined:
        new_header[wts_key] = {
            "dtype": "U8",
            "shape": [E, row],
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
    for key, meta, blob in residents:
        new_header[key] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)

    hjson = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    hjson += b" " * ((-len(hjson)) % 8)  # pad header to 8-byte alignment

    written = 0
    with open(dst_path, "wb") as out:
        out.write(struct.pack("<Q", len(hjson)))
        out.write(hjson)
        for _wts_key, blob, _E, _row in combined:
            out.write(blob)
            written += len(blob)
        for _key, _meta, blob in residents:
            out.write(blob)
            written += len(blob)

    return len(combined), written


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Input model dir (local path).")
    ap.add_argument("--out", default=None,
                    help="Output dir: copies the model and writes companions there. "
                         "Omit for in-place (companions written beside --model shards).")
    ap.add_argument("--in-place", action="store_true",
                    help="Write companions directly into --model (default when --out "
                         "is omitted). No full-model copy.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing companion files / a non-empty --out.")
    ap.add_argument("--replace", action="store_true",
                    help="Replace originals instead of writing additive companions: "
                         "each companion also copies the shard's non-expert tensors, "
                         "then the original model-*.safetensors is DELETED and the "
                         "companion renamed to the original's name. Eliminates the "
                         "expert-byte duplication a plain companion incurs, but is "
                         "IRREVERSIBLE and requires --force. Needs a reader that "
                         "recognises interleaved data under the model-* name.")
    args = ap.parse_args()

    # Replacing originals is destructive and irreversible: require --force even
    # if the output dir has no pre-existing companions to clash with.
    if args.replace and not args.force:
        print("[repack-wts] error: --replace deletes the original shards; "
              "pass --force to confirm.", file=sys.stderr)
        return 2

    try:
        from turboquant_mlx.generate import resolve_model_path
        src = str(resolve_model_path(args.model))
    except Exception:
        src = args.model

    if not os.path.isdir(src):
        print(f"[repack-wts] error: model dir not found: {src}", file=sys.stderr)
        return 2

    in_place = args.in_place or args.out is None

    if in_place:
        out_dir = src
    else:
        out_dir = args.out
        if os.path.exists(out_dir) and os.path.isdir(out_dir) and os.listdir(out_dir):
            if not args.force:
                print(f"[repack-wts] error: --out {out_dir} exists and is non-empty "
                      f"(pass --force to overwrite)", file=sys.stderr)
                return 2
        os.makedirs(out_dir, exist_ok=True)
        # Copy the whole model (shards + config/tokenizer/index) so --out is a
        # self-contained model with companions.
        for name in os.listdir(src):
            if _is_companion(name):
                continue  # never copy pre-existing companions from the source
            s = os.path.join(src, name)
            d = os.path.join(out_dir, name)
            if os.path.isfile(s):
                shutil.copy2(s, d)

    # Original (non-companion) shards drive the pairing.
    shards = sorted(
        f for f in glob.glob(os.path.join(glob.escape(src), "model*.safetensors"))
        if not _is_companion(os.path.basename(f))
    )
    if not shards:
        raise FileNotFoundError(f"no model*.safetensors in {src}")

    # Guard against clobbering companions that already exist.
    if not args.force:
        existing = [
            os.path.join(out_dir, _companion_name(os.path.basename(s)))
            for s in shards
        ]
        clash = [p for p in existing if os.path.exists(p)]
        if clash:
            print(f"[repack-wts] error: companion file(s) already exist, e.g. "
                  f"{clash[0]} (pass --force to overwrite)", file=sys.stderr)
            return 2

    total = len(shards)
    total_pairs = 0
    total_bytes = 0
    # For --replace: (original_basename, companion_path) for each shard that
    # produced a companion; used to delete + rename after all writes succeed.
    replace_plan: list[tuple[str, str]] = []
    for i, shard in enumerate(shards, 1):
        base = os.path.basename(shard)
        companion = _companion_name(base)
        dst = os.path.join(out_dir, companion)
        n_pairs, written = _build_interleaved_shard(
            shard, dst, include_resident=args.replace)
        total_pairs += n_pairs
        total_bytes += written
        if n_pairs == 0:
            # No expert pairs in this shard: don't leave an empty companion.
            # In --replace mode the original stays (it holds resident-only data).
            if os.path.exists(dst):
                os.remove(dst)
            print(f"[repack-wts] shard {i}/{total} {base}: "
                  f"no expert pairs (no companion written)", flush=True)
        else:
            replace_plan.append((base, dst))
            print(f"[repack-wts] shard {i}/{total} {base}: "
                  f"{n_pairs} weight+scales pairs -> {companion} "
                  f"({written / (1024 * 1024):.1f} MB)", flush=True)

    if args.replace:
        # Verify every planned companion was written before touching originals,
        # so a partial run never deletes a shard whose replacement is missing.
        missing = [dst for _base, dst in replace_plan if not os.path.exists(dst)]
        if missing:
            print(f"[repack-wts] error: companion(s) missing, refusing to delete "
                  f"originals (e.g. {missing[0]})", file=sys.stderr)
            return 2
        for base, dst in replace_plan:
            original = os.path.join(out_dir, base)
            if os.path.exists(original):
                os.remove(original)
            os.replace(dst, original)  # companion becomes the primary shard
        print(f"[repack-wts] replaced {len(replace_plan)} original shard(s) with "
              f"interleaved+resident shards of the same name.")
        print("[repack-wts] WARNING: originals deleted. This format needs a "
              "reader that recognises interleaved data under the model-* name "
              "(has_interleaved=True). The current safetensors_reader keys "
              "interleaving on the model_wts-* filename and reads expert strides "
              "from the original tensor entries, so it will NOT read these "
              "replaced shards without a compatible update.", file=sys.stderr)

    where = "in-place" if in_place else out_dir
    kind = "replaced shards" if args.replace else "companion files"
    print(f"[repack-wts] done ({where}): {total_pairs} interleaved pairs, "
          f"{total_bytes / (1024 * 1024 * 1024):.2f} GB in {kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
