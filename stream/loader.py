"""Load a TurboQuant MoE model with its experts streamed from disk.

Reuses ``load_turboquant(lazy=True)`` to build the full model with weights
left mmap-backed and *unmaterialized*, then swaps every MoE expert layer
(``switch_mlp.{gate,up,down}_proj``) for a ``StreamingSwitchLinear`` before any
forward runs — so the big (num_experts, ...) expert tensors are never
evaluated into RAM. Everything else (embeddings, norms, attention, router,
shared expert) stays resident as usual.
"""

from __future__ import annotations

import glob
import os
import time

import mlx.core as mx
import mlx.nn as nn

from turboquant_mlx.generate import load_turboquant, resolve_model_path
from turboquant_mlx.layers.polar_switch_linear import PolarQuantizedSwitchLinear

from .safetensors_reader import SafetensorsExpertReader
from .streaming_switch import ExpertCache, StreamingSwitchLinear


class _BF16StreamingSwitchLinear(nn.Module):
    """Streaming wrapper for unquantized bfloat16 switch expert projections.

    Some model checkpoints store a small number of MoE layers as full bfloat16
    rather than polar-quantized uint32. Without this wrapper all 128-expert
    stacks for those layers accumulate as wired Metal allocations across the
    forward pass, exhausting the device wired-memory limit.

    This wrapper reads only the k router-selected expert slices from disk on
    each forward call (via the safetensors reader) and releases them after the
    computation — the same principle as StreamingSwitchLinear, but without the
    polar-quantization kernel.
    """

    def __init__(self, weight_key: str, reader: SafetensorsExpertReader,
                 output_dims: int, input_dims: int, num_experts: int):
        super().__init__()
        self._weight_key = weight_key
        self._reader = reader
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.num_experts = num_experts
        self.freeze()

    def __call__(self, x, indices, sorted_indices=False):
        # Sync routing indices to Python (one GPU->CPU round-trip per call).
        flat = indices.reshape(-1)
        mx.eval(flat)
        flat_list = [int(v) for v in flat.tolist()]
        unique = sorted(set(flat_list))

        # Read only the selected experts from disk; wire only those slices.
        # BF16 is stored as uint16 in numpy — use .view() to reinterpret the
        # bits as bfloat16 rather than converting the values. F16 and others
        # are stored as their native numpy dtype and can be passed directly.
        ws = []
        for e in unique:
            np_w, mlx_dt = self._reader.read_expert_np(self._weight_key, e)
            if mlx_dt == mx.bfloat16:
                ws.append(mx.array(np_w).view(mx.bfloat16))
            else:
                ws.append(mx.array(np_w, dtype=mlx_dt))
        w_stack = mx.stack(ws, axis=0)  # [n_sel, out, in]

        # Remap global expert ids to local 0..n_sel-1 indices.
        idx_map = {e: i for i, e in enumerate(unique)}
        idx_local = mx.array(
            [idx_map[e] for e in flat_list], dtype=mx.uint32,
        ).reshape(indices.shape)

        return mx.gather_mm(x, w_stack.swapaxes(-1, -2), rhs_indices=idx_local)

_PROJS = ("gate_proj", "up_proj", "down_proj")

# When the model file fits comfortably in RAM, trusting the OS page cache makes
# LRU-eviction re-reads come back from warm RAM instead of disk — measured 2.44x
# faster decode on a streamed 35B-A3B (scripts/flash_moe/trust_os_ab.py). When
# the model is larger than RAM (16 GB mini on a 70 GB MoE), the page cache would
# thrash and F_NOCACHE is correct. These fractions are the three-way decision:
# < FIT   -> clearly fits, trust OS
# < NEAR  -> near-fit (model ~10% over RAM), OS page cache still covers the
#            overflow from the ~(RAM - wired_pool) pageable space, trust OS
# >= NEAR -> too large, F_NOCACHE to avoid page-cache thrash
_PAGE_CACHE_RAM_FRACTION_FIT = 0.60    # model < 60% of RAM: clearly fits, trust OS
_PAGE_CACHE_RAM_FRACTION_NEAR = 1.15   # model < 115% of RAM: near-fit, also trust OS


def _total_ram_bytes() -> int:
    try:  # AttributeError too: os.sysconf is absent on some platforms (Windows)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        import subprocess
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))


def _auto_page_cache(model_path: str, cache_budget_gb: float = 0.0) -> bool:
    """True iff the OS page cache should stay on for expert reads.

    Three-way decision by model-size-vs-RAM (see the fraction constants above):
    a model that clearly fits, or one that is only near-fit (~10% over RAM) with
    enough pageable RAM left after the wired application cache, both benefit from
    the OS page cache; a model too large for the pageable space uses F_NOCACHE to
    avoid thrash. cache_budget_gb is the application LRU's wired-RAM claim, which
    is subtracted from effective pageable RAM for the near-fit check.
    """
    try:
        # glob.escape so a model_path with [ ] etc. still matches; no files ->
        # we can't size the model, so fail safe to F_NOCACHE rather than 0 bytes.
        files = glob.glob(os.path.join(glob.escape(model_path), "model*.safetensors"))
        # Exclude interleaved weight+scales companion files (model_wts-*): they
        # duplicate expert bytes already counted in the main shards, so including
        # them would ~double the measured model size and skew the fit decision.
        files = [f for f in files if not os.path.basename(f).startswith("model_wts")]
        if not files:
            return False
        model_bytes = sum(os.path.getsize(f) for f in files)
        ram = _total_ram_bytes()
    except Exception:
        return False  # any uncertainty -> the always-safe F_NOCACHE path
    # Near-fit: model is within ~15% of total RAM. The OS page cache can absorb
    # the overflow through the pageable space, so F_NOCACHE is counterproductive.
    # Exception: when the application LRU claims large wired RAM (> 8 GB), also
    # require model < effective_pageable × 1.10 to avoid thrash with a big cache.
    effective_pageable = ram - int(cache_budget_gb * 1e9)
    fits_comfortably = model_bytes < _PAGE_CACHE_RAM_FRACTION_FIT * ram
    fits_near_by_ram = model_bytes < _PAGE_CACHE_RAM_FRACTION_NEAR * ram
    fits_near_by_budget = model_bytes < effective_pageable * 1.10
    _LARGE_CACHE_GB = 8.0  # only apply budget check when cache is large
    fits_near = fits_near_by_ram and (
        cache_budget_gb <= _LARGE_CACHE_GB or fits_near_by_budget
    )
    use_pc = fits_comfortably or fits_near
    if fits_comfortably:
        reason = "trust-OS (model comfortably fits, F_NOCACHE off)"
    elif fits_near:
        reason = "trust-OS (model ~10% over RAM, page-cache covers overflow, F_NOCACHE off)"
    else:
        reason = "F_NOCACHE on (model too large for page-cache)"
    print(f"[stream] page-cache auto: model {model_bytes/1e9:.1f} GB vs RAM {ram/1e9:.1f} GB "
          f"-> {reason} (override with use_page_cache=/--use-page-cache)")
    return use_pc


def _is_internal_nvme(path: str) -> bool:
    """Best-effort heuristic: is this path on an internal NVMe (vs USB/network)?

    On macOS, check diskutil info for the filesystem's device. Internal Apple SSD
    shows as 'APPLE SSD' in the model string. Falls back to False (conservative)
    on any error.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["diskutil", "info", "-plist", path],
            capture_output=True, timeout=3
        )
        if result.returncode != 0:
            return False
        import plistlib
        info = plistlib.loads(result.stdout)
        # 'MediaName' or 'IORegistryEntryName' on Apple SSDs contains 'APPLE SSD'
        # 'BusProtocol' is 'PCI-Express' for NVMe internal, 'USB' for external
        bus = info.get("BusProtocol", "")
        media = info.get("MediaName", "") + info.get("IORegistryEntryName", "")
        return "PCI-Express" in bus or "APPLE SSD" in media.upper()
    except Exception:
        return False


def _cap_active_experts(layers, max_active: int) -> None:
    """Cap router top_k on every MoE block to ``min(native, max_active)``.

    The "K-reduction" lever (Flash-MoE): when experts stream from disk, per-token
    disk I/O scales with the number of *active* experts, not the total. Lowering
    top_k is mechanically clean — ``argpartition`` then selects fewer experts and
    ``norm_topk_prob`` renormalizes the gate weights over them — and the streaming
    switch loads only the selected experts. Measured on Qwen3.6-35B-A3B-tq3-g32
    (256 experts, native top_k=8): 8->4 is byte-identical on the 6-test stress
    harness and cuts streamed disk reads ~2x (78.9->37.8 GB) for ~1.4x decode in
    the disk-bound regime; K=2 collapses (broken JSON). Caps only — never raises.
    """
    if not max_active or max_active <= 0:
        return
    changed = []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "top_k") or not hasattr(mlp, "switch_mlp"):
            continue
        native = int(mlp.top_k)
        new_k = min(native, max_active)
        if new_k != native:
            mlp.top_k = new_k
            changed.append(native)
    if changed:
        print(f"[stream] K-reduction: capped router top_k {changed[0]}->{min(changed[0], max_active)} "
              f"on {len(changed)} MoE blocks (~2x less disk I/O; pass "
              f"max_active_experts=0 / --max-active-experts 0 to use native routing)")


def _expert_bytes(reader) -> int:
    """Per-expert byte cost across all three projections (gate+up+down), counting
    both the packed weight and the scales — the same cost calculation as
    calibrate_experts._model_expert_info.

    Finds one switch_mlp weight key in the reader index, sizes its (and its
    scales sibling's) per-expert slice from ``shape[1:]`` × ``itemsize``, and
    multiplies by 3 for the three projections. Returns 0 when the reader exposes
    no switch_mlp expert tensors, so the caller can skip warmup gracefully.
    """
    itemsize = {"U32": 4, "F16": 2, "F32": 4}
    wkey = None
    for key in reader._index:
        if "switch_mlp" in key and key.endswith(".weight"):
            wkey = key
            break
    if wkey is None:
        return 0
    skey = wkey[: -len(".weight")] + ".scales"
    per_proj = 0
    for key in (wkey, skey):
        loc = reader._index.get(key)
        if loc is None:
            continue
        n = 1
        for d in loc.shape[1:]:  # bytes per expert = all dims except the expert axis
            n *= d
        per_proj += n * itemsize.get(loc.dtype, 0)
    return per_proj * 3


def _warm_cache(cache, model_path, warmup_file, warmup_gb, k):
    """Pre-load the hottest experts from a saved histogram into the LRU cache,
    up to a GB budget, so the session starts near steady-state hit rate instead
    of cold. Called at startup from ``load_streaming`` when a warmup file exists.

    The histogram rows are sorted by count descending, so a prefix that fits the
    byte budget is the hottest set. Loads go through ``cache.gather`` (which
    inserts into the LRU and evicts to respect the cache budget), so if the cache
    budget is smaller than ``warmup_gb`` the LRU simply keeps the most recently
    loaded — correct behavior, not an error.
    """
    data = cache.load_histogram(warmup_file, model_id=model_path, k=k)
    cost_per_expert = _expert_bytes(cache.reader)
    if cost_per_expert == 0:
        print(f"[stream] warmup: no switch_mlp expert tensors found in reader; "
              f"skipping warmup from {warmup_file}")
        return

    # Take rows (hottest first) until the byte budget is exhausted. All experts
    # cost the same, so the first overflow ends the prefix.
    budget_bytes = int(warmup_gb * 1e9)
    warmup_set = []
    running_total = 0
    for row in data.get("hist", []):
        if running_total + cost_per_expert > budget_bytes:
            break
        warmup_set.append((int(row[0]), int(row[1])))
        running_total += cost_per_expert

    if not warmup_set:
        print(f"[stream] warmup: budget {warmup_gb:.1f} GB fits no experts "
              f"(per-expert {cost_per_expert / 1e9:.3f} GB) from {warmup_file}")
        return

    # Sort by on-disk offset so the loads walk the file monotonically (random
    # reads -> sequential). Offset ~= weight-key base + expert * per-expert
    # stride; the weight key comes from the layer's first registered projection.
    if cache._layer_keys:
        itemsize = {"U32": 4, "F16": 2, "F32": 4}

        def _offset(le):
            layer_idx, expert = le
            proj_keys = cache._layer_keys.get(layer_idx)
            if not proj_keys:
                return 0
            wkey = proj_keys[0][0]
            loc = cache.reader._index.get(wkey)
            if loc is None:
                return 0
            n = 1
            for d in loc.shape[1:]:
                n *= d
            per_expert_weight_bytes = n * itemsize.get(loc.dtype, 0)
            return loc.abs_begin + expert * per_expert_weight_bytes

        warmup_set.sort(key=_offset)

    # Group by layer and load every projection of each warm expert. gather()
    # coalesces the sorted experts within a projection into range reads.
    from collections import defaultdict
    by_layer = defaultdict(set)
    for layer_idx, expert in warmup_set:
        by_layer[layer_idx].add(expert)

    loaded = 0
    proj_keys = []
    t0 = time.time()
    for layer_idx in sorted(by_layer.keys()):
        experts = sorted(by_layer[layer_idx])
        proj_keys = cache._layer_keys.get(layer_idx, [])
        for wkey, skey in proj_keys:
            cache.gather(wkey, skey, experts)
        loaded += len(experts)
    print(f"[stream] warmup: loaded {loaded} experts × {len(proj_keys or [])} projections "
          f"(~{loaded * cost_per_expert / 1e9:.1f} GB) from {warmup_file} "
          f"in {time.time() - t0:.1f}s")


def load_streaming(model_path, cache_budget_gb: float = 3.0, fast: bool = False,
                   prefetch_workers: int = 8, prefetch_ahead: int | None = None,
                   pin_file: str | None = None, max_active_experts: int = 4,
                   use_page_cache: bool | None = None,
                   warmup_file: str | None = None, warmup_gb: float = 0.0,
                   perm_path: str | None = None, use_ane: bool = False):
    """Returns (model, tokenizer, cache).

    cache_budget_gb bounds total resident expert memory (LRU-evicted).
    prefetch_workers parallelizes per-layer expert reads (1 = serial baseline).
    prefetch_ahead speculatively prefetches this many upcoming layers' experts
    (predicted from the previous token's routing); 0 disables prefetch. None
    (default) auto-decides by storage: 1 on internal NVMe (fast, spare
    bandwidth), 0 on external/network storage.
    pin_file is an optional JSON {"pin": [[layer, expert], ...]} of hot experts
    to keep permanently resident (never LRU-evicted) — see calibrate_experts.py.
    max_active_experts caps router top_k to min(native, this) on every MoE block
    (the Flash-MoE K-reduction lever: ~2x less streamed disk I/O at no quality
    cost up to K=4 on validated models). Default 4; set 0 to use native routing.
    use_page_cache controls the OS page cache for expert reads. None (default)
    auto-decides by model-size-vs-RAM: trust the OS (page cache on) when the
    model fits comfortably in RAM (~2.4x faster decode), F_NOCACHE when it does
    not (avoids page-cache thrash on a memory-constrained machine). True/False
    force it.
    warmup_file is an optional histogram JSON (see ExpertCache.dump_histogram);
    when it exists at startup the top experts by count are pre-loaded up to
    warmup_gb, so the session begins near steady-state hit rate instead of cold.
    warmup_gb bounds that pre-load; 0 (default) disables warmup even if a file
    is given.
    perm_path is an optional perm.json from calibrate_experts.py analyze. Pass
    when loading a model repacked by stream/repack.py (which permutes only the
    expert stacks, not the router); the reader then translates logical->physical
    expert ids at read time. Do NOT pass when loading a model repacked by
    repack_experts.py (which permutes the router too, making translation
    transparent).
    use_ane: Route single-token scaled-dot-product attention to the Apple Neural
    Engine via CoreML (macOS only). Frees wired GPU memory for the streaming
    expert hot tier. Requires coremltools (pip install coremltools). First run
    compiles CoreML models per sequence-length bucket (~30s/bucket, cached to
    ~/.turboquant_mlx/ane_cache/).
    """
    local_path = str(resolve_model_path(model_path))
    if prefetch_ahead is None:
        prefetch_ahead = 1 if _is_internal_nvme(local_path) else 0
        print(f"[stream] prefetch-ahead auto: "
              f"{'internal NVMe' if prefetch_ahead else 'external/network storage'}, "
              f"setting prefetch_ahead={prefetch_ahead}")
    if use_page_cache is None:
        use_page_cache = _auto_page_cache(local_path, cache_budget_gb=cache_budget_gb)
    model, tok = load_turboquant(local_path, lazy=True, fast=fast)
    reader = SafetensorsExpertReader(local_path, use_page_cache=use_page_cache,
                                     perm_path=perm_path)
    if reader.has_interleaved:
        print("[stream] interleaved weight+scales companion files detected — "
              "1 pread per expert (halved syscalls)")
    else:
        print("[stream] no interleaved companion files; run stream/repack_interleaved.py "
              "to enable 1-pread-per-expert mode")
    cache = ExpertCache(
        reader, int(cache_budget_gb * 1e9),
        prefetch_workers=prefetch_workers,
        prefetch_ahead=prefetch_ahead,
    )

    # Load the hot-expert pin spec (frequency-based pinning, #2). Keyed by layer
    # so we can pin all three projections of each hot expert.
    pin_layers: dict = {}
    if pin_file:
        import json
        with open(pin_file) as f:
            for layer, expert in json.load(f).get("pin", []):
                pin_layers.setdefault(int(layer), set()).add(int(expert))

    # Locate the transformer layer stack and its weight-key prefix. Multimodal
    # MoEs (qwen3_5_moe) nest it under `language_model.model.layers`; text-only
    # MoEs (deepseek_v2/v3, …) use `model.model.layers`.
    if hasattr(model, "language_model"):
        layers = model.language_model.model.layers
        prefix = "language_model.model.layers"
    else:
        layers = model.model.layers
        prefix = "model.layers"
    swapped = 0
    pin_keys: set = set()
    for i, layer in enumerate(layers):
        sm = getattr(layer.mlp, "switch_mlp", None)
        if sm is None:
            continue
        proj_keys = []
        for proj in _PROJS:
            res = getattr(sm, proj, None)
            if not isinstance(res, PolarQuantizedSwitchLinear):
                continue
            cb, sg = res.codebook, res.signs
            mx.eval(cb, sg)  # tiny — pin resident, let the rest of res be freed
            wkey = f"{prefix}.{i}.mlp.switch_mlp.{proj}.weight"
            skey = f"{prefix}.{i}.mlp.switch_mlp.{proj}.scales"
            for e in pin_layers.get(i, ()):  # pin every projection of a hot expert
                pin_keys.add((wkey, e))
            st = StreamingSwitchLinear(
                input_dims=res.input_dims,
                output_dims=res.output_dims,
                num_experts=res.num_experts,
                bits=res.bits,
                group_size=res.group_size,
                needs_rotation=res._needs_rotation,
                codebook=cb,
                signs=sg,
                weight_key=wkey,
                scales_key=skey,
                cache=cache,
                layer_idx=i,
                # one trigger per layer fires the next-layer prefetch; gate_proj
                # is first in _PROJS so it fires with maximum lead time.
                is_trigger=(proj == _PROJS[0]),
                trit=res.trit,
            )
            setattr(sm, proj, st)
            proj_keys.append((wkey, skey))
            swapped += 1
        if proj_keys:
            cache.register_layer(i, proj_keys)

    cache._pin_keys = pin_keys
    pin_note = f", pinned {len(pin_keys)} hot expert-projections" if pin_keys else ""
    print(f"[stream] swapped {swapped} expert projections to streaming "
          f"(budget {cache_budget_gb:.1f} GB{pin_note})")

    # Second pass: wrap any remaining large floating-point switch_mlp projections.
    # Some model checkpoints store a subset of MoE layers as full-precision
    # (bfloat16 or float16) rather than polar-quantized uint32. The first pass
    # skips them (not PolarQuantizedSwitchLinear). Without streaming, those
    # [128, out, in] stacks wire into Metal as each layer executes and accumulate
    # across the forward pass, exhausting the wired-memory limit.
    #
    # Detection: inspect the model weight directly (ndim==3, shape[0] matches
    # num_experts) rather than trusting the reader dtype string, which can vary
    # (BF16 vs F16) across checkpoints and wasn't reliable in practice.
    n_fp = 0
    for i, layer in enumerate(layers):
        sm = getattr(getattr(layer, 'mlp', None), 'switch_mlp', None)
        if sm is None:
            continue
        for proj in _PROJS:
            res = getattr(sm, proj, None)
            if res is None:
                continue
            if isinstance(res, (StreamingSwitchLinear, _BF16StreamingSwitchLinear)):
                continue  # already streaming
            w = getattr(res, 'weight', None)
            if not isinstance(w, mx.array) or w.ndim != 3 or w.shape[0] < 8:
                continue
            # Large unstreamed stacked expert weight. Verify the reader has the
            # disk key so read_expert_np won't KeyError at inference time.
            wkey = f"{prefix}.{i}.mlp.switch_mlp.{proj}.weight"
            loc = reader._index.get(wkey)
            if loc is None:
                print(f"[stream] WARNING: unstreamed expert weight {wkey!r} not in "
                      f"reader index — will wire {w.nbytes >> 20} MB at inference "
                      f"(OOM risk at layer {i})")
                continue
            if loc.dtype not in ("BF16", "F16", "F32"):
                print(f"[stream] WARNING: unstreamed expert weight {wkey!r} has "
                      f"unexpected dtype {loc.dtype!r}; skipping streaming wrap")
                continue
            n_exp, out_dims, in_dims = loc.shape
            st = _BF16StreamingSwitchLinear(
                weight_key=wkey,
                reader=reader,
                output_dims=out_dims,
                input_dims=in_dims,
                num_experts=n_exp,
            )
            setattr(sm, proj, st)
            n_fp += 1
    if n_fp:
        print(f"[stream] full-precision fallback: wrapped {n_fp} unquantized expert "
              f"projections as disk-backed streamers")

    _cap_active_experts(layers, max_active_experts)

    # Cross-session cache warmup: pre-load the hottest experts recorded by a
    # previous session's histogram, starting this session near steady state.
    if warmup_file and os.path.isfile(warmup_file) and warmup_gb > 0:
        _warm_cache(cache, local_path, warmup_file, warmup_gb, max_active_experts)

    if use_ane:
        from .ane_loader import install_ane_attention
        install_ane_attention(model=model, warmup=True)
        print("[stream] ANE attention dispatcher active — single-token attention routed to ANE")

    return model, tok, cache
