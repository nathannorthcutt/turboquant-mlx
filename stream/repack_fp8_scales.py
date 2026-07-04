"""Re-encode MoE scales from F16 to fp8 (E4M3) to halve scales I/O.

Each MoE projection stores its per-group magnitudes as an F16 ``scales`` tensor
of shape ``(E, out, in/g)``. For Qwen3-235B gate_proj (in=4096, out=1536, g=32)
that is ``(128, 1536, 128)`` ~= 50 MB/layer — consistently ~24% of every
expert's streamed bytes. Halving the scales element width (F16 -> fp8) trims
~12% off all cold-path streamed bytes.

This tool writes *companion* files ``model_s8-N-of-M.safetensors`` next to each
original shard. For every ``.scales`` tensor that lives in F16, it emits a
same-shape, same-key tensor typed ``F8_E4M3`` whose values are the original
F16 scales quantized to fp8 E4M3 (clamp to +/-448, round-to-nearest-even). No
weights (U32) and no non-scales tensors are duplicated.

The companion header records ``{"fp8_scales": "true",
"scales_dtype": "F8_E4M3"}`` so a reader can auto-detect it.

fp8 re-encoding is LOSSY. Run ``stream/fp8_scales_probe.py`` as a quality gate
before trusting these companions for inference.

Example:
  python -m turboquant_mlx.stream.repack_fp8_scales \
      --model .../qwen3-235b-tq3 [--force]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

_S8_PREFIX = "model_s8"


# --------------------------------------------------------------------------
# fp8 E4M3 encode / decode (pure numpy)
# --------------------------------------------------------------------------
# E4M3: sign=1 bit, exponent=4 bits (bias=7), mantissa=3 bits. No infinities;
# the only specials are NaN at 0x7F / 0xFF. Max normal = 1.110b * 2^8 = 448.0.


def _f16_to_f8e4m3(arr_f16: np.ndarray) -> np.ndarray:
    """Convert a float16 array to fp8 E4M3 (returned as uint8 bit patterns).

    Values outside ``[-448, 448]`` are clamped. Rounding is nearest-even (via
    ``np.round``). Subnormals (very small magnitudes) are flushed to zero —
    ternary scales are essentially never that small.
    """
    # Arithmetic in float32 to avoid float16 rounding polluting the bit math.
    arr = arr_f16.astype(np.float32)

    sign = (arr < 0).astype(np.uint8)
    abs_arr = np.abs(arr)

    # Clamp to E4M3 range (max normal = 448.0).
    abs_arr = np.clip(abs_arr, 0.0, 448.0)

    result = np.zeros(arr.shape, dtype=np.uint8)
    nonzero = abs_arr > 0.0

    if np.any(nonzero):
        # true_exp = floor(log2(|x|)); stored_exp = true_exp + bias(7), 4 bits.
        log2 = np.floor(np.log2(np.where(nonzero, abs_arr, 1.0))).astype(np.int32)
        stored_exp = np.clip(log2 + 7, 0, 15).astype(np.int32)

        # Mantissa: (|x| / 2^true_exp - 1) * 8, rounded to nearest-even.
        scale = np.power(2.0, log2.astype(np.float32))
        mantissa_f = np.where(
            nonzero, (abs_arr / np.where(scale > 0, scale, 1.0) - 1.0) * 8.0, 0.0
        )
        mantissa = np.clip(np.round(mantissa_f).astype(np.int32), 0, 7)

        fp8_val = ((stored_exp << 3) | mantissa).astype(np.uint8)
        result = np.where(nonzero, fp8_val, np.uint8(0))

    result = result | (sign << 7)
    return result.astype(np.uint8)


def _f8e4m3_to_f32(arr_u8: np.ndarray) -> np.ndarray:
    """Decode an fp8 E4M3 uint8 array back to float32."""
    arr_u8 = arr_u8.astype(np.uint8)
    sign = ((arr_u8 >> 7) & 1).astype(np.float32)
    exp_bits = ((arr_u8 >> 3) & 0x0F).astype(np.int32)
    mant_bits = (arr_u8 & 0x07).astype(np.int32)

    # 0x7F / 0xFF are NaN in E4M3; treat as 0.
    is_nan = (arr_u8 & 0x7F) == 0x7F

    normal = (exp_bits > 0) & ~is_nan
    subnormal = (exp_bits == 0) & (mant_bits > 0) & ~is_nan

    val = np.zeros(arr_u8.shape, dtype=np.float32)
    val = np.where(
        normal,
        (1.0 - 2.0 * sign)
        * np.power(2.0, (exp_bits - 7).astype(np.float32))
        * (1.0 + mant_bits / 8.0),
        val,
    )
    val = np.where(
        subnormal,
        (1.0 - 2.0 * sign) * (2.0 ** -6) * (mant_bits / 8.0),
        val,
    )
    return val


# --------------------------------------------------------------------------
# companion naming
# --------------------------------------------------------------------------
def _companion_name(shard_basename: str) -> str:
    """``model-00001-of-00030.safetensors`` -> ``model_s8-00001-of-00030...``.

    Also handles an unsharded ``model.safetensors`` -> ``model_s8.safetensors``.
    """
    assert shard_basename.startswith("model")
    return _S8_PREFIX + shard_basename[len("model"):]


def _is_companion(basename: str) -> bool:
    # Both this tool's fp8 companion and repack_interleaved's weight companion
    # match ``model*.safetensors``; exclude either from the source pairing.
    return basename.startswith(_S8_PREFIX) or basename.startswith("model_wts")


# --------------------------------------------------------------------------
# per-shard repack
# --------------------------------------------------------------------------
def _build_fp8_shard(src_path: str, dst_path: str) -> tuple[int, int]:
    """Write the fp8-scales companion for one shard.

    Returns ``(n_scales_tensors, bytes_written)``. Emits one ``F8_E4M3`` tensor
    (same key, same shape) for every F16 ``.scales`` tensor in the source shard.
    """
    with open(src_path, "rb") as f:
        raw = f.read()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    data_start = 8 + n

    converted: list[tuple[str, list, bytes]] = []  # (key, shape, fp8 bytes)
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        if not key.endswith(".scales") or meta["dtype"] != "F16":
            continue
        begin, end = meta["data_offsets"]
        blob = raw[data_start + begin:data_start + end]
        f16 = np.frombuffer(blob, dtype=np.float16)
        fp8 = _f16_to_f8e4m3(f16)  # uint8, one byte per element
        converted.append((key, list(meta["shape"]), fp8.tobytes()))

    if not converted:
        return 0, 0

    new_header: dict = {
        "__metadata__": {
            "fp8_scales": "true",
            "scales_dtype": "F8_E4M3",
        }
    }
    offset = 0
    for key, shape, blob in converted:
        new_header[key] = {
            "dtype": "F8_E4M3",
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
        for _key, _shape, blob in converted:
            out.write(blob)
            written += len(blob)

    return len(converted), written


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Input model dir (local path).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing fp8-scales companion files.")
    args = ap.parse_args()

    try:
        from turboquant_mlx.generate import resolve_model_path
        src = str(resolve_model_path(args.model))
    except Exception:
        src = args.model

    if not os.path.isdir(src):
        print(f"[repack-s8] error: model dir not found: {src}", file=sys.stderr)
        return 2

    shards = sorted(
        f for f in glob.glob(os.path.join(glob.escape(src), "model*.safetensors"))
        if not _is_companion(os.path.basename(f))
    )
    if not shards:
        raise FileNotFoundError(f"no model*.safetensors in {src}")

    if not args.force:
        existing = [
            os.path.join(src, _companion_name(os.path.basename(s)))
            for s in shards
        ]
        clash = [p for p in existing if os.path.exists(p)]
        if clash:
            print(f"[repack-s8] error: companion file(s) already exist, e.g. "
                  f"{clash[0]} (pass --force to overwrite)", file=sys.stderr)
            return 2

    total = len(shards)
    total_tensors = 0
    total_bytes = 0
    for i, shard in enumerate(shards, 1):
        companion = _companion_name(os.path.basename(shard))
        dst = os.path.join(src, companion)
        n_tensors, written = _build_fp8_shard(shard, dst)
        total_tensors += n_tensors
        total_bytes += written
        if n_tensors == 0:
            if os.path.exists(dst):
                os.remove(dst)
            print(f"[repack-s8] shard {i}/{total} {os.path.basename(shard)}: "
                  f"no F16 scales (no companion written)", flush=True)
        else:
            print(f"[repack-s8] shard {i}/{total} {os.path.basename(shard)}: "
                  f"{n_tensors} scales -> {companion} "
                  f"({written / (1024 * 1024):.1f} MB fp8)", flush=True)

    print(f"[repack-s8] done: {total_tensors} scales tensors re-encoded to fp8, "
          f"{total_bytes / (1024 * 1024 * 1024):.2f} GB in companion files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
