"""ANE attention offload: patches mx.fast.scaled_dot_product_attention to run on ANE.

Call install_ane_attention() after load_streaming() to activate. The dispatcher
compiles CoreML attention models on first use per sequence-length bucket and caches
them to ~/.turboquant_mlx/ane_cache/. Each bucket takes ~30s to compile once;
subsequent runs load from cache.

The ANE has its own memory pool separate from Metal's wired pool (iogpu.wired_limit_mb),
so routing attention there frees wired headroom for the streaming expert hot tier.
"""

from .ane_attention import ANEAttentionDispatcher, get_dispatcher  # noqa: F401


def get_dispatcher_if_active():
    """Return the dispatcher if it has been installed and CoreML is available, else None.

    Called from mlx_lm.models.base.scaled_dot_product_attention in the forked mlx-lm.
    Returns None when turboquant's ANE offload is not in use, keeping the hot path free.
    """
    d = get_dispatcher()
    if d._installed and d._available:
        return d
    return None


def install_ane_attention(model=None, warmup=True, buckets=None):
    """Patch mx.fast.scaled_dot_product_attention to dispatch single-token attention
    to the ANE via CoreML. Safe to call multiple times (idempotent).

    Args:
        model:   If provided, infer n_heads/n_kv_heads/head_dim/scale from the model
                 to pre-compile warmup buckets without waiting for first inference.
        warmup:  Pre-compile all buckets in a background thread.
        buckets: Override bucket list for warmup (default: all _BUCKETS).
    """
    d = get_dispatcher()
    d.install()

    if warmup and model is not None:
        try:
            # Infer attention config from the model's first transformer layer.
            n_heads, n_kv_heads, head_dim, scale = _infer_attention_config(model)
            d.warmup(n_heads, n_kv_heads, head_dim, scale, buckets=buckets)
        except Exception as e:
            print(f"[ANE] warmup skipped: could not infer attention config ({e})")

    return d


def uninstall_ane_attention():
    get_dispatcher().uninstall()


def _iter_attention_layers(model):
    """Yield candidate attention modules across common mlx_lm layouts.

    Text-only MoEs expose ``model.model.layers``; multimodal MoEs
    (e.g. qwen3_5_moe) nest under ``model.language_model.model.layers``. Also
    accept a bare ``model.layers`` for models that expose the stack directly.
    """
    stacks = []
    lm = getattr(model, "language_model", None)
    if lm is not None:
        inner = getattr(getattr(lm, "model", None), "layers", None)
        if inner is not None:
            stacks.append(inner)
    inner = getattr(getattr(model, "model", None), "layers", None)
    if inner is not None:
        stacks.append(inner)
    direct = getattr(model, "layers", None)
    if direct is not None:
        stacks.append(direct)

    for layers in stacks:
        for layer in layers:
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is not None:
                yield attn


def _infer_attention_config(model):
    """Walk the model to find the first attention layer and read its config."""
    for attn in _iter_attention_layers(model):
        n_heads = int(getattr(attn, "n_heads", getattr(attn, "num_heads", -1)))
        n_kv_heads = int(getattr(attn, "n_kv_heads",
                                 getattr(attn, "num_kv_heads", n_heads)))
        head_dim = int(getattr(attn, "head_dim", getattr(attn, "d_head", -1)))
        scale = float(getattr(attn, "scale", head_dim ** -0.5 if head_dim > 0 else 0.0))
        if n_heads > 0 and head_dim > 0:
            return n_heads, n_kv_heads, head_dim, scale
    raise ValueError("could not find attention config in model")


def ane_stats() -> dict:
    return get_dispatcher().stats()
