"""Compatibility shims for upstream libraries.

Importing this module is a side-effect: it patches third-party classes in
place to work around gaps between bleeding-edge model releases and the
installed library versions. All patches are idempotent and self-disable
once upstream catches up.
"""


def _patch_nemotron_h_pattern():
    """Teach NemotronHConfig about MLP ("-") block types.

    Nemotron 3 models encode layer types as a string like
    "M-M-M-MM-M-M*-M-M*-..." where M=mamba, *=attention, -=MLP, E=MoE.
    transformers' NemotronHConfig hard-codes its pattern alphabet in two
    places — `_pattern_to_list` (decodes the string) and
    `validate_layers_block_type` (checks the resulting list). Both miss
    "mlp". mlx-lm already handles the "-" block type, so extending the
    config's alphabet is enough to unblock loading.
    """
    try:
        from transformers.models.nemotron_h.configuration_nemotron_h import (
            NemotronHConfig,
        )
    except ImportError:
        return

    try:
        NemotronHConfig._pattern_to_list("-")
    except KeyError:
        @staticmethod
        def _pattern_to_list(pattern: str) -> list:
            mapping = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
            return [mapping[c] for c in pattern]
        NemotronHConfig._pattern_to_list = _pattern_to_list

    valid_types = {"mamba", "attention", "moe", "mlp"}

    @staticmethod
    def validate_layers_block_type(self):
        if not isinstance(self.layers_block_type, list):
            raise ValueError(
                f"`layers_block_type` must be a list of strings. Got type: "
                f"{type(self.layers_block_type)}"
            )
        invalid = set(self.layers_block_type) - valid_types
        if invalid:
            raise ValueError(
                f"`layers_block_type` contains invalid types: {invalid}. "
                f"Must be one of: {valid_types}"
            )
        if getattr(self, "num_nextn_predict_layers", 0) > 0:
            if self.mtp_layers_block_type is None:
                raise ValueError(
                    "mtp_layers_block_type is required when "
                    "num_nextn_predict_layers > 0."
                )
            if not isinstance(self.mtp_layers_block_type, list):
                raise ValueError(
                    f"`mtp_layers_block_type` must be a list of strings. "
                    f"Got type: {type(self.mtp_layers_block_type)}"
                )
            invalid = set(self.mtp_layers_block_type) - valid_types
            if invalid:
                raise ValueError(
                    f"`mtp_layers_block_type` contains invalid types: "
                    f"{invalid}. Must be one of: {valid_types}"
                )

    NemotronHConfig.validate_layers_block_type = validate_layers_block_type

    # huggingface_hub's dataclass machinery captures validator functions into
    # __class_validators__ at class-definition time, so swapping the attribute
    # above is not enough — we also have to replace the entry in that list.
    validators = getattr(NemotronHConfig, "__class_validators__", None)
    if validators is not None:
        for i, fn in enumerate(validators):
            if getattr(fn, "__name__", "") == "validate_layers_block_type":
                # validate_layers_block_type is defined as @staticmethod, so
                # __class_validators__ stores the underlying function; use the
                # same shape here.
                validators[i] = validate_layers_block_type.__func__
                break


_patch_nemotron_h_pattern()


def _patch_moe_layer_barrier():
    """Force mx.eval(h) after every transformer layer in streamed MoE models.

    Without this, MLX builds the entire N-layer computation graph lazily and
    submits it as one Metal command buffer. For Qwen3-235B the prefill pass
    touches nearly every expert across all 94 layers simultaneously — the
    combined working set exceeds the Metal wired-memory limit and crashes.

    Per-layer eval breaks the graph into 94 small command buffers. Each holds
    one layer's expert stack (~1-2 GB) rather than the full ~36+ GB union.

    Installed here rather than editing the mlx_lm source because mlx_lm may
    resolve to the pip-installed site-packages copy, not the local checkout.
    The patch is idempotent and targets the actual imported module.
    """
    import mlx.core as mx

    _MOE_DECODER_STACKS = [
        ("mlx_lm.models.qwen3_moe",   "Qwen3MoeModel"),
        ("mlx_lm.models.qwen3_5_moe", "Qwen3_5MoeModel"),
        ("mlx_lm.models.deepseek_v2", "DeepseekV2Model"),
        ("mlx_lm.models.deepseek_v3", "DeepseekV3Model"),
        ("mlx_lm.models.mixtral",     "MixtralModel"),
    ]

    for mod_path, cls_name in _MOE_DECODER_STACKS:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is None or getattr(cls, "_tq_layer_barrier", False):
            continue

        orig_call = cls.__call__

        def _make_patched(orig):
            def _patched_call(self, inputs, cache=None, *args, **kwargs):
                # Only patch the inner decoder stack (has both .layers and .norm).
                # The outer lm-head Model wraps this and should pass through.
                if not hasattr(self, "layers") or not hasattr(self, "norm"):
                    return orig(self, inputs, cache=cache, *args, **kwargs)

                from mlx_lm.models.base import create_attention_mask

                import os
                _diag = os.environ.get("TQ_MEM_DIAG", "0") == "1"

                h = (self.embed_tokens(inputs)
                     if hasattr(self, "embed_tokens") and isinstance(inputs, mx.array)
                     else inputs)

                if cache is None:
                    cache = [None] * len(self.layers)

                mask = create_attention_mask(h, cache[0])

                if _diag:
                    if mask is not None:
                        mx.eval(h, mask)
                    else:
                        mx.eval(h)
                    _mask_repr = (
                        None if mask is None
                        else list(mask.shape) if hasattr(mask, 'shape')
                        else type(mask).__name__
                    )
                    print(
                        f"[tq-diag] barrier-loop active  layers={len(self.layers)}"
                        f"  h={list(h.shape)}  mask={_mask_repr}"
                        f"  active={mx.get_active_memory()//1048576} MB"
                        f"  peak={mx.get_peak_memory()//1048576} MB",
                        flush=True,
                    )

                for i, (layer, c) in enumerate(zip(self.layers, cache)):
                    h = layer(h, mask, c)
                    # One Metal command buffer per layer — prevents the full
                    # 94-layer graph from being submitted at once.
                    mx.eval(h)
                    # Release freed Metal buffers back to the OS immediately.
                    # Without this, MLX's buffer cache accumulates across 94
                    # layers and pushes total wired memory past the wired limit
                    # even though peak active data is small.
                    if callable(getattr(mx, "clear_cache", None)):
                        mx.clear_cache()
                    if _diag:
                        print(
                            f"[tq-diag] layer {i:3d} ok"
                            f"  active={mx.get_active_memory()//1048576:6d} MB"
                            f"  peak={mx.get_peak_memory()//1048576:6d} MB",
                            flush=True,
                        )

                return self.norm(h)

            _patched_call._tq_layer_barrier = True
            return _patched_call

        cls.__call__ = _make_patched(orig_call)
        cls._tq_layer_barrier = True
        print(f"[tq-compat] installed barrier+clear_cache on {mod_path}.{cls_name}", flush=True)


def _check_mlx_version():
    """Warn if mx.clear_cache is missing (MLX < 0.8)."""
    import mlx.core as mx
    if not callable(getattr(mx, "clear_cache", None)):
        import warnings
        warnings.warn(
            "[tq-compat] mx.clear_cache() not available in this MLX version. "
            "Per-layer buffer release is disabled. OOM risk is higher.",
            stacklevel=2,
        )


_patch_moe_layer_barrier()
_check_mlx_version()
