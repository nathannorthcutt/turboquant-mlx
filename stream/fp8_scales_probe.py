"""Quality gate for the F16 -> fp8 (E4M3) scales re-encoding.

Reads a sample of ``.scales`` tensors straight off disk, round-trips them
through fp8 E4M3 (``encode`` then ``decode``), and reports how much precision
is lost. It also simulates the downstream effect: dequantizing random ternary
weights with the original vs. fp8-decoded scales and measuring the L2 deviation
of a matmul output.

No mlx / coremltools / torch — pure numpy + struct, so it runs on the dev box.
It does NOT load the model for inference; it works on raw tensors from disk.

Verdict is PASS when, across every sampled tensor:
  * max relative error  < 0.01
  * mean relative error < 0.001
  * weight-output L2 deviation < 0.001

Usage:
  python -m turboquant_mlx.stream.fp8_scales_probe --model .../qwen3-235b-tq3 [--layers 8]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import struct
import sys

import numpy as np

# Reuse the exact encode/decode used by the repack tool so the probe measures
# the real thing, not a re-implementation.
from stream.repack_fp8_scales import _f16_to_f8e4m3, _f8e4m3_to_f32

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")

# Thresholds — a tensor PASSES only if it clears all three.
_MAX_REL_ERR = 0.01
_MEAN_REL_ERR = 0.001
_L2_DEV = 0.001


# --------------------------------------------------------------------------
# minimal safetensors slice reader (no mlx)
# --------------------------------------------------------------------------
def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def _read_first_expert_f16(path: str, meta: dict, data_start: int) -> np.ndarray:
    """Read scales slice ``[0]`` of a stacked ``(E, out, in/g)`` F16 tensor.

    Reading one expert (~0.4 MB) is enough for error statistics and keeps the
    probe's memory footprint tiny even for 50 MB/layer scales tensors.
    """
    shape = meta["shape"]
    per_expert = 1
    for d in shape[1:]:
        per_expert *= d
    nbytes = per_expert * 2  # F16
    begin = meta["data_offsets"][0]
    with open(path, "rb") as f:
        f.seek(data_start + begin)
        blob = f.read(nbytes)
    return np.frombuffer(blob, dtype=np.float16).reshape(shape[1:])


# --------------------------------------------------------------------------
# per-tensor round-trip stats
# --------------------------------------------------------------------------
def _roundtrip_stats(f16: np.ndarray) -> dict:
    orig = f16.astype(np.float32)
    decoded = _f8e4m3_to_f32(_f16_to_f8e4m3(f16))
    abs_err = np.abs(orig - decoded)
    rel_err = abs_err / (np.abs(orig) + 1e-8)

    # Histogram of error magnitudes (log-ish buckets).
    edges = np.array([0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, np.inf])
    hist, _ = np.histogram(abs_err, bins=edges)

    return {
        "max_abs_err": float(abs_err.max()),
        "mean_abs_err": float(abs_err.mean()),
        "max_rel_err": float(rel_err.max()),
        "mean_rel_err": float(rel_err.mean()),
        "frac_exact": float(np.mean(abs_err == 0.0)),
        "hist": hist,
        "hist_edges": edges,
    }


# --------------------------------------------------------------------------
# downstream weight-output deviation
# --------------------------------------------------------------------------
def _unpack_ternary_from_u32(packed: np.ndarray) -> np.ndarray:
    """Map random uint32 to ternary values in {-1, 0, +1}.

    This is a stand-in for tq3's real bit unpacking: each uint32 yields 16
    2-bit fields decoded as {0:0, 1:+1, 2:-1, 3:0}. The exact packing does not
    affect what this probe measures — the deviation comes entirely from the
    scale difference, and any {-1,0,+1} distribution exercises that faithfully.
    """
    packed = packed.astype(np.uint32)
    fields = np.empty(packed.shape + (16,), dtype=np.int8)
    for i in range(16):
        two = (packed >> (2 * i)) & 0x3
        fields[..., i] = np.select(
            [two == 1, two == 2], [1, -1], default=0
        ).astype(np.int8)
    return fields.reshape(packed.shape[:-1] + (packed.shape[-1] * 16,))


def _weight_output_deviation(scales_f16: np.ndarray, rng: np.random.Generator,
                             group_size: int = 32) -> float:
    """Dequantize random ternary weights with F16 vs fp8 scales; return the
    normalized L2 deviation of ``W @ x``.

    ``scales_f16`` is one expert's ``(out, n_groups)`` slice; the ternary weight
    is generated with matching ``(out, n_groups * group_size)`` shape so scales
    broadcast per group exactly as in real dequant.
    """
    out, n_groups = scales_f16.shape
    in_features = n_groups * group_size
    n_u32 = in_features // 16  # 16 ternary trits per uint32

    packed = rng.integers(0, 2 ** 32, size=(out, n_u32), dtype=np.uint32)
    ternary = _unpack_ternary_from_u32(packed).astype(np.float32)  # (out, in)

    s_orig = scales_f16.astype(np.float32)
    s_fp8 = _f8e4m3_to_f32(_f16_to_f8e4m3(scales_f16))

    # Expand per-group scales across group_size columns.
    exp_orig = np.repeat(s_orig, group_size, axis=1)  # (out, in)
    exp_fp8 = np.repeat(s_fp8, group_size, axis=1)

    w_orig = ternary * exp_orig
    w_fp8 = ternary * exp_fp8

    x = rng.standard_normal(in_features).astype(np.float32)
    y_orig = w_orig @ x
    y_fp8 = w_fp8 @ x

    denom = np.linalg.norm(y_orig)
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(y_orig - y_fp8) / denom)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Model dir (local path).")
    ap.add_argument("--layers", type=int, default=8,
                    help="Sample scales from the first N transformer layers.")
    ap.add_argument("--group-size", type=int, default=32,
                    help="Quantization group size (for the weight-output sim).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        from turboquant_mlx.generate import resolve_model_path
        src = str(resolve_model_path(args.model))
    except Exception:
        src = args.model

    if not os.path.isdir(src):
        print(f"[probe-s8] error: model dir not found: {src}", file=sys.stderr)
        return 2

    shards = sorted(
        f for f in glob.glob(os.path.join(glob.escape(src), "model*.safetensors"))
        if not (os.path.basename(f).startswith("model_s8")
                or os.path.basename(f).startswith("model_wts"))
    )
    if not shards:
        print(f"[probe-s8] error: no model*.safetensors in {src}", file=sys.stderr)
        return 2

    # Collect F16 .scales tensors from the first N layers, all projections.
    samples: list[tuple[str, str, dict, int]] = []  # (key, path, meta, data_start)
    for path in shards:
        header, data_start = _read_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if not key.endswith(".scales") or meta["dtype"] != "F16":
                continue
            m = _LAYER_RE.search(key)
            layer = int(m.group(1)) if m else -1
            if layer >= args.layers or layer < 0:
                continue
            samples.append((key, path, meta, data_start))

    if not samples:
        print(f"[probe-s8] error: no F16 .scales tensors found in first "
              f"{args.layers} layers of {src}", file=sys.stderr)
        return 2

    samples.sort(key=lambda t: (int(_LAYER_RE.search(t[0]).group(1)), t[0]))

    rng = np.random.default_rng(args.seed)
    print(f"[probe-s8] model: {src}")
    print(f"[probe-s8] sampling {len(samples)} scales tensors "
          f"(first {args.layers} layers)\n")

    hdr = (f"{'tensor':<58} {'max_rel':>9} {'mean_rel':>9} "
           f"{'exact%':>7} {'W-L2':>9}  verdict")
    print(hdr)
    print("-" * len(hdr))

    all_pass = True
    agg = {"max_rel_err": 0.0, "mean_rel_err": 0.0, "max_l2": 0.0}
    total_hist = None
    edges = None

    for key, path, meta, data_start in samples:
        f16 = _read_first_expert_f16(path, meta, data_start)
        st = _roundtrip_stats(f16)
        l2 = _weight_output_deviation(f16, rng, group_size=args.group_size)

        ok = (st["max_rel_err"] < _MAX_REL_ERR
              and st["mean_rel_err"] < _MEAN_REL_ERR
              and l2 < _L2_DEV)
        all_pass &= ok

        agg["max_rel_err"] = max(agg["max_rel_err"], st["max_rel_err"])
        agg["mean_rel_err"] = max(agg["mean_rel_err"], st["mean_rel_err"])
        agg["max_l2"] = max(agg["max_l2"], l2)
        if total_hist is None:
            total_hist = st["hist"].astype(np.int64).copy()
            edges = st["hist_edges"]
        else:
            total_hist += st["hist"].astype(np.int64)

        short = key if len(key) <= 58 else "..." + key[-55:]
        print(f"{short:<58} {st['max_rel_err']:>9.2e} {st['mean_rel_err']:>9.2e} "
              f"{st['frac_exact'] * 100:>6.1f}% {l2:>9.2e}  "
              f"{'PASS' if ok else 'FAIL'}")

    print("\n[probe-s8] error-magnitude histogram (abs err, all sampled tensors):")
    labels = ["==0", "<1e-6", "<1e-4", "<1e-3", "<1e-2", "<1e-1", ">=1e-1"]
    # np.histogram with edges [0,1e-6,...] puts exact-0 into the first bucket
    # together with (0,1e-6); report the true exact-0 count separately.
    total = int(total_hist.sum())
    for lbl, cnt in zip(labels, total_hist):
        frac = cnt / total if total else 0.0
        bar = "#" * int(frac * 40)
        print(f"  {lbl:>7}: {int(cnt):>10}  {frac * 100:5.1f}%  {bar}")

    print(f"\n[probe-s8] worst-case across all tensors:")
    print(f"  max_rel_err  = {agg['max_rel_err']:.3e}  (threshold < {_MAX_REL_ERR})")
    print(f"  mean_rel_err = {agg['mean_rel_err']:.3e}  (threshold < {_MEAN_REL_ERR})")
    print(f"  weight L2    = {agg['max_l2']:.3e}  (threshold < {_L2_DEV})")

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\n[probe-s8] VERDICT: {verdict}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
