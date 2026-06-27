# gpt-oss-120b-tq3

TurboQuant 3-bit quantized version of [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) using [TurboQuant-MLX](https://github.com/manjunathshiva/turboquant-mlx).

## Model Details

- **Base Model**: openai/gpt-oss-120b (MoE: 120B params)
- **Quantization**: TurboQuant 3-bit (Hadamard rotation + Lloyd-Max codebook)
- **Parameters**: ~120B
- **Size**: ~48 GB (vs ~240 GB BF16)
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
    --model ~/path/to/gpt-oss-120b-tq3 \
    --prompt "Why is the sky blue? Explain in simple terms." \
    --max-tokens 200 \
    --temp 0.7
```

Or using mlx-lm:

```python
from mlx_lm import load, generate

model, tokenizer = load("manjunathshiva/gpt-oss-120b-tq3")
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
    --model ~/path/to/gpt-oss-120b-tq3 \
    --prompt "Your long prompt here..." \
    --max-tokens 200 \
    --tq-bits 3
```

## Results

| Configuration | Size | Speed (M4 Ultra) |
|---------------|------|------------------|
| BF16 (original) | ~240 GB | Doesn't fit 64GB |
| Affine 4-bit (mlx-community) | ~65.8 GB | Doesn't fit 64GB |
| MXFP4 (original) | ~63.5 GB | Doesn't fit 64GB |
| TurboQuant 3-bit | ~48 GB | **44 tok/s** |
| TurboQuant 2-bit | ~32 GB | 51 tok/s (poor quality) |

With KV cache compression:
| Configuration | KV Size | Speed |
|---------------|---------|-------|
| TQ 3-bit weights + FP16 KV | 45.0 MB | 6.4 tok/s |
| TQ 3-bit weights + TQ 3-bit KV | 11.83 MB | **8.7 tok/s** (3.8x smaller and *faster*) |

## How It Works

TurboQuant applies:
1. **Hadamard rotation** - random +/- 1 scaling to decorrelate weights
2. **Lloyd-Max codebook** - optimal scalar quantization via k-means
3. **Group-wise scaling** - per-group float16 scales for precision

This achieves much better quality than standard affine quantization at the same bit-width. On 100B+ models, the 4x smaller KV cache reduces memory bandwidth more than dequant adds — generation is *faster* than FP16 baseline!

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