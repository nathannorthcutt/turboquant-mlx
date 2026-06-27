# qwen3.5-122b-tq3

TurboQuant 3-bit quantized version of [Qwen/Qwen2.5-122B-A10B](https://huggingface.co/Qwen/Qwen2.5-122B-A10B) using [TurboQuant-MLX](https://github.com/manjunathshiva/turboquant-mlx).

## Model Details

- **Base Model**: Qwen/Qwen2.5-122B-A10B (MoE: 122B total params, 10B active)
- **Quantization**: TurboQuant 3-bit (Hadamard rotation + Lloyd-Max codebook)
- **Parameters**: ~122B total, ~10B active
- **Size**: ~50 GB (vs ~240 GB BF16)
- **Group Size**: 64

## Requirements

```bash
# macOS with Apple Silicon (M1/M2/M3/M4)
pip install turboquant-mlx-full mlx-lm
```

## Quick Start

### Generate text

```bash
python -m turboquant_mlx.generate \
    --model ~/path/to/qwen3.5-122b-tq3 \
    --prompt "Why is the sky blue? Explain in simple terms." \
    --max-tokens 200 \
    --temp 0.7
```

Or using mlx-lm:

```python
from mlx_lm import load, generate

model, tokenizer = load("manjunathshiva/qwen3.5-122b-tq3")
response = generate(
    model,
    tokenizer,
    prompt="Why is the sky blue? Explain in simple terms.",
    max_tokens=200,
    temp=0.7
)
print(response)
```

### With KV Cache Compression

For longer prompts/context, use KV cache compression:

```bash
python -m turboquant_mlx.demo_kv \
    --model ~/path/to/qwen3.5-122b-tq3 \
    --prompt "Your long prompt here..." \
    --max-tokens 200 \
    --tq-bits 3
```

## Results

| Configuration | Size | Speed (M4 Ultra) |
|---------------|------|------------------|
| BF16 (original) | ~240 GB | Doesn't fit 64GB |
| TurboQuant 3-bit | ~50 GB | **26.5 tok/s** |

## How It Works

TurboQuant applies:
1. **Hadamard rotation** - random +/- 1 scaling to decorrelate weights
2. **Lloyd-Max codebook** - optimal scalar quantization via k-means
3. **Group-wise scaling** - per-group float16 scales for precision

This achieves much better quality than standard affine quantization at the same bit-width. The 3-bit on 3-bit (weights + KV cache) works cleanly on 100B+ models which have enough redundancy to absorb stacked noise.

## License

Apache 2.0 (same as base model)

## Citation

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  year={2025},
  eprint={2504.19874},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2504.19874}
}
```