"""Fuse each MoE expert's gate_proj and up_proj into one gate_up_proj tensor.

In a SwiGLU MoE the gate and up projections of an expert (a) always receive the
*identical* rotated input x and (b) are always both computed on every forward
pass. Concatenating their packed weights along the output-feature axis

    gate: (E, out, packed_cols)  U32          gate_scales: (E, out, n_groups) F16
    up:   (E, out, packed_cols)  U32          up_scales:   (E, out, n_groups) F16
    ------------------------------------------------------------------------
    gate_up: (E, 2*out, packed_cols) U32      gate_up_scales: (E, 2*out, n_groups) F16

lets a single ``polar_gather_qmm`` call compute both projections at once (the
result is split ``[:out]`` / ``[out:]`` to recover gate and up). At K=4 this
saves one kernel launch per MoE layer per token, doubles the kernel's output
dim (better GPU occupancy), and halves the expert reads (one fused pread pair
instead of a gate pair + an up pair). This is the standard fused-gate_up
optimization (vLLM, TRT-LLM).

This tool writes a *companion* safetensors file next to each original shard
(``model_fused-N-of-M.safetensors``). The companion holds ONLY the fused
``gate_up_proj.weight`` / ``gate_up_proj.scales`` tensors for the MoE layers
whose gate+up (weight AND scales) all live in that shard. Everything else — the
original gate_proj / up_proj (left intact so the streaming reader can still fall
back to them), down_proj, attention, norms, router — stays in the original
shards and is read the normal way. down_proj is NOT fused (its input differs).

Fused-key naming: the fused key is derived from the gate key by replacing
``gate_proj`` -> ``gate_up_proj`` (e.g.
``model.layers.3.mlp.switch_mlp.gate_proj.weight`` ->
``model.layers.3.mlp.switch_mlp.gate_up_proj.weight``). The reader and the
streaming dispatch derive the same key by the same rule, so the on-disk name
matches whatever prefix the model uses (``switch_mlp`` or ``experts``).

The fuse is a pure byte-level rearrangement (dtype-agnostic, like
repack_interleaved.py): each output row is copied whole, so no element is ever
interpreted. The companion header records
``{"fused_gate_up": "true", "weight_dtype": "U32", "scales_dtype": "F16"}`` so
the reader can auto-detect it.

Note on disk cost: the companion DUPLICATES the gate+up expert bytes (it does
not delete the originals), so it adds roughly the model's gate+up footprint
again on disk.

Examples:
  # default: write companions in-place beside the model's existing shards
  python -m turboquant_mlx.stream.repack_fused_gate_up --model .../qwen3-235b-tq3

  # copy the model to --out and write companions there
  python -m turboquant_mlx.stream.repack_fused_gate_up \
      --model .../qwen3-235b-tq3 --out /tmp/qwen3-235b-tq3-fused
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import struct
import sys

import numpy as np

# safetensors dtype string -> element size in bytes. We only need element sizes
# to carve fixed-size per-output-row blobs out of opaque bytes, so this stays
# dtype-agnostic (mirrors repack_interleaved.py's _ITEMSIZE).
_ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def _companion_name(shard_basename: str) -> str:
    """``model-00001-of-00030.safetensors`` -> ``model_fused-00001-of-00030.safetensors``.

    Also handles an unsharded ``model.safetensors`` -> ``model_fused.safetensors``.
    """
    assert shard_basename.startswith("model")
    return "model_fused" + shard_basename[len("model"):]


def _is_companion(basename: str) -> bool:
    # Both repack companions must be excluded when discovering source shards.
    return basename.startswith("model_wts") or basename.startswith("model_fused")


def _fuse_pair(raw: bytes, data_start: int, gmeta: dict, umeta: dict):
    """Concatenate gate + up along the output axis. Returns ``(blob, shape, dtype)``.

    ``gate`` / ``up`` are each ``(E, out, *inner)``; the result is
    ``(E, 2*out, *inner)`` where, per expert ``e``, the ``out`` gate rows come
    first and the ``out`` up rows second. Purely a whole-row byte copy — the
    element dtype is never interpreted.
    """
    gshape, ushape = gmeta["shape"], umeta["shape"]
    if gshape != ushape:
        raise ValueError(f"gate/up shape mismatch: {gshape} vs {ushape}")
    dtype = gmeta["dtype"]
    if dtype != umeta["dtype"]:
        raise ValueError(f"gate/up dtype mismatch: {dtype} vs {umeta['dtype']}")
    E, out = gshape[0], gshape[1]
    isz = _ITEMSIZE[dtype]
    inner = 1
    for d in gshape[2:]:
        inner *= d
    row_bytes = inner * isz  # bytes per output row

    gbegin, gend = gmeta["data_offsets"]
    ubegin, uend = umeta["data_offsets"]
    gblob = raw[data_start + gbegin:data_start + gend]
    ublob = raw[data_start + ubegin:data_start + uend]
    if len(gblob) != E * out * row_bytes or len(ublob) != E * out * row_bytes:
        raise ValueError(
            f"byte size mismatch (g {len(gblob)}, u {len(ublob)}, "
            f"expected {E * out * row_bytes})")

    grows = np.frombuffer(gblob, dtype=np.uint8).reshape(E, out, row_bytes)
    urows = np.frombuffer(ublob, dtype=np.uint8).reshape(E, out, row_bytes)
    fused = np.ascontiguousarray(np.concatenate([grows, urows], axis=1))
    fused_shape = [E, 2 * out] + list(gshape[2:])
    return fused.tobytes(), fused_shape, dtype


def _build_fused_shard(src_path: str, dst_path: str) -> tuple[int, int]:
    """Write the fused companion for one shard. Returns ``(n_layers, bytes_written)``.

    For every ``gate_proj.weight`` whose ``up_proj.weight`` sibling and both
    ``.scales`` siblings live in the SAME shard, emit a fused
    ``gate_up_proj.weight`` and ``gate_up_proj.scales``.
    """
    with open(src_path, "rb") as f:
        raw = f.read()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    data_start = 8 + n

    # (key, dtype, shape, blob) for each fused tensor.
    tensors: list[tuple[str, str, list, bytes]] = []
    for gate_wkey in list(header.keys()):
        if gate_wkey == "__metadata__":
            continue
        if not gate_wkey.endswith("gate_proj.weight"):
            continue
        up_wkey = gate_wkey.replace("gate_proj.weight", "up_proj.weight")
        gate_skey = gate_wkey.replace("gate_proj.weight", "gate_proj.scales")
        up_skey = gate_wkey.replace("gate_proj.weight", "up_proj.scales")
        # All four siblings must live in this shard to fuse in-place.
        if any(k not in header for k in (up_wkey, gate_skey, up_skey)):
            continue

        fused_wkey = gate_wkey.replace("gate_proj.weight", "gate_up_proj.weight")
        fused_skey = gate_wkey.replace("gate_proj.weight", "gate_up_proj.scales")

        w_blob, w_shape, w_dtype = _fuse_pair(
            raw, data_start, header[gate_wkey], header[up_wkey])
        s_blob, s_shape, s_dtype = _fuse_pair(
            raw, data_start, header[gate_skey], header[up_skey])
        tensors.append((fused_wkey, w_dtype, w_shape, w_blob))
        tensors.append((fused_skey, s_dtype, s_shape, s_blob))

    if not tensors:
        return 0, 0

    # Lay tensors out back-to-back and build the header.
    new_header: dict = {
        "__metadata__": {
            "fused_gate_up": "true",
            "weight_dtype": "U32",
            "scales_dtype": "F16",
        }
    }
    offset = 0
    for key, dtype, shape, blob in tensors:
        new_header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)

    hjson = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    hjson += b" " * ((-len(hjson)) % 8)  # pad header to 8-byte alignment

    written = 0
    with open(dst_path, "wb") as out:
        out.write(struct.pack("<Q", len(hjson)))
        out.write(hjson)
        for _key, _dtype, _shape, blob in tensors:
            out.write(blob)
            written += len(blob)

    return len(tensors) // 2, written  # two tensors (weight+scales) per layer


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
    args = ap.parse_args()

    try:
        from turboquant_mlx.generate import resolve_model_path
        src = str(resolve_model_path(args.model))
    except Exception:
        src = args.model

    if not os.path.isdir(src):
        print(f"[repack-fused] error: model dir not found: {src}", file=sys.stderr)
        return 2

    in_place = args.in_place or args.out is None

    if in_place:
        out_dir = src
    else:
        out_dir = args.out
        if os.path.exists(out_dir) and os.path.isdir(out_dir) and os.listdir(out_dir):
            if not args.force:
                print(f"[repack-fused] error: --out {out_dir} exists and is non-empty "
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

    # Original (non-companion) shards drive the fusing.
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
            print(f"[repack-fused] error: companion file(s) already exist, e.g. "
                  f"{clash[0]} (pass --force to overwrite)", file=sys.stderr)
            return 2

    total = len(shards)
    total_layers = 0
    total_bytes = 0
    for i, shard in enumerate(shards, 1):
        companion = _companion_name(os.path.basename(shard))
        dst = os.path.join(out_dir, companion)
        n_layers, written = _build_fused_shard(shard, dst)
        total_layers += n_layers
        total_bytes += written
        if n_layers == 0:
            # No fusable MoE layers in this shard: don't leave an empty companion.
            if os.path.exists(dst):
                os.remove(dst)
            print(f"[repack-fused] shard {i}/{total} {os.path.basename(shard)}: "
                  f"no gate/up pairs (no companion written)", flush=True)
        else:
            print(f"[repack-fused] shard {i}/{total} {os.path.basename(shard)}: "
                  f"{n_layers} fused gate+up layers -> {companion} "
                  f"({written / (1024 * 1024):.1f} MB)", flush=True)

    where = "in-place" if in_place else out_dir
    print(f"[repack-fused] done ({where}): {total_layers} fused gate+up layers, "
          f"{total_bytes / (1024 * 1024 * 1024):.2f} GB in companion files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
