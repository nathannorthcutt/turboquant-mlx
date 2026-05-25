# Changelog

All notable changes to this project are documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.1] - 2026-05-25

### Fixed

- **Packaging: `turboquant_mlx.stream` was omitted from the 0.4.0
  distribution.** The `[tool.setuptools] packages` list enumerates packages
  explicitly and the new `stream` subpackage was not added, so 0.4.0 shipped
  without the streaming code (`import turboquant_mlx.stream` failed after a
  PyPI install; the feature was only reachable from a source checkout). Added
  `turboquant_mlx.stream` to the packages list. Expert streaming now works
  from `pip install turboquant-mlx-full`.

## [0.4.0] - 2026-05-25

### Added

- **Expert streaming for MoE models (`turboquant_mlx.stream`).** Run MoE
  checkpoints whose weights exceed available RAM by paging only the
  router-selected experts from disk per token (LRU-cached), keeping the full
  `(num_experts, ...)` expert tensors out of memory. Output is bit-identical
  to the fully-resident model. New CLI:
  `python -m turboquant_mlx.stream.stream_generate --model <repo> --cache-budget-gb <GB>`.
  - Validated on a 16 GB Mac mini running the ~16 GB
    `Qwen3.6-35B-A3B-tq3-g32` (`qwen3_5_moe`, 256 experts): 3.9 GB peak RSS
    at `--cache-budget-gb 2` (~3 tok/s) up to 9.4 GB / ~4.5 tok/s at
    `--cache-budget-gb 8`. Disk read is the throughput limiter; a larger cache
    budget raises the expert hit-rate and cuts per-token SSD traffic.
  - Uses `os.pread` + macOS `F_NOCACHE` so streaming tens of GB of expert
    slices doesn't balloon resident page cache; RSS tracks MLX managed memory.

### Fixed

- `stream_generate` reported throughput by dividing `--max-tokens` by
  wall-time, overstating tok/s whenever the model stopped at EOS before the
  cap. It now counts the tokens actually generated.

### Notes

- Streaming currently targets the `qwen3_5_moe` expert layout
  (`language_model.model.layers[*].mlp.switch_mlp.{gate,up,down}_proj`);
  generalizing to other MoE architectures is future work.

## [0.3.0] - 2026-05-03

### Changed

- **License relicensed from MIT to Apache-2.0.** The MIT license that covered
  versions 0.1.x and 0.2.0 is preserved verbatim in `LICENSE-MIT` for
  reference; users of those versions retain their MIT rights. All new
  contributions and releases (0.3.0 and later) are governed by `LICENSE`
  (Apache-2.0).
- `pyproject.toml` `license` field updated to `Apache-2.0`; classifier
  updated accordingly. Author field corrected to legal name
  (`Manjunath Janardhan`).

### Added

- `NOTICE` file with copyright + attribution boilerplate (Apache-2.0
  requires this to propagate into derivative works).
- `CITATION.cff` so GitHub renders a "Cite this repository" widget and
  academic users have a defined citation form.
- `CONTRIBUTING.md` with the Developer Certificate of Origin (DCO) and
  `Signed-off-by` instructions for contributors.

### Notes

- Why the relicense: Apache-2.0 adds an explicit patent grant, mandates
  NOTICE propagation in derivative works, and is the standard open-source
  license for ML/quantization tooling. MIT remains valid for all 0.1.x and
  0.2.0 releases.

## [0.2.0] - 2026-04-30

KV cache v0.2: mixed K/V bits (`--kv-k-bits`/`--kv-v-bits`),
attention-sink protection (`--kv-min-tokens`), per-head_dim PPL harness,
production CLI for `turboquant-generate` and `turboquant-serve`. Validated
on GPT-OSS-20B/120B and Qwen3.5-122B.

## [0.1.6] - 2026-04-12

Hybrid quantization (`--attn-bits`/`--mlp-bits`) targeting 48 GB Apple
Silicon Macs. Long-context Metal kernel fixes.

## Earlier versions

See `git log` for the full history of versions 0.1.0 through 0.1.5.
