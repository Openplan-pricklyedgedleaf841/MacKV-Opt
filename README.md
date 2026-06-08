# MacKV-Opt

[English](README.md) | [简体中文](README.zh-CN.md)

MacKV-Opt is a local tool for choosing safer context and KV cache settings when
running LLMs on Apple Silicon Macs. It keeps the selected model unchanged, does
not upload prompts or outputs, and helps Ollama users avoid memory pressure,
swap stalls, and trial-and-error `num_ctx` tuning.

![MacKV-Opt architecture](docs/assets/mackv-opt-architecture.svg)

![Validation snapshot](docs/assets/validation-snapshot.svg)

![Baseline summary example](docs/assets/baseline-summary-example.svg)

## What It Optimizes

Long-context inference grows KV cache memory roughly linearly with context
length. On unified-memory Macs, an oversized context can push the system into
memory pressure and swap; an undersized context wastes available memory and
forces users to shorten prompts.

MacKV-Opt improves this workflow by:

- estimating model weights, KV cache, runtime overhead, and memory headroom;
- choosing a `num_ctx` that fits the requested memory budget;
- recommending llama.cpp KV cache types when that runtime is available;
- warning when Ollama, llama.cpp, model metadata, or memory sampling are missing;
- comparing default Ollama, manual `num_ctx`, and MacKV-Opt runs with the same
  model and prompts.

The practical effect is not magic speedup. The benefit is targeted: fewer
failed long-context runs, less swap-driven slowdown, and a larger usable context
when the Mac has memory headroom that default or manual settings do not use.

## Quick Start

Install from the repository:

```bash
python -m pip install -e .
```

Check the machine:

```bash
mackv-opt doctor
```

Use the simplest planner entry:

```bash
mackv-opt auto llama3.1:8b --memory-budget 12GiB
```

Run only when you are ready to call the local Ollama API:

```bash
mackv-opt auto llama3.1:8b \
  --memory-budget 12GiB \
  --prompt "Summarize this document" \
  --execute
```

Plan a specific target context:

```bash
mackv-opt plan llama3.1:8b --target-context 64k --memory-budget 12GiB
```

If Ollama metadata is incomplete, provide model metadata manually:

```bash
mackv-opt plan llama3.1:8b \
  --target-context 64k \
  --memory-budget 12GiB \
  --model-size 4.8GiB \
  --hidden-size 4096 \
  --layers 32 \
  --heads 32 \
  --kv-heads 8 \
  --hardware-memory 16GiB
```

## How It Works

The planner uses this budget model:

```text
estimated_total = model_weights + KV_cache(context, layers, hidden, GQA_ratio, KV_type) + runtime_overhead
```

It searches from the requested context downward and selects the largest
configuration that fits the memory budget. It prefers higher-quality KV cache
types when they fit, falls back to smaller KV types when memory is tight, and
reduces context only when no cache strategy fits.

For Ollama, MacKV-Opt emits safe `num_ctx` and `num_gpu` options. For
llama.cpp, it also emits `--ctx-size`, `--cache-type-k`, `--cache-type-v`, and
`--kv-offload` arguments when the local runtime exposes those flags.

## Compared With Other Options

| Approach | What users do | Main drawback | MacKV-Opt role |
| --- | --- | --- | --- |
| Default Ollama | Run the model with default options | May use a conservative or unsuitable context for the task | Measures the default baseline |
| Manual `num_ctx` | Guess a context value | Easy to overshoot memory or leave capacity unused | Estimates a safer value before running |
| llama.cpp flags | Tune low-level cache/context flags directly | Powerful but easy to misconfigure | Converts memory budget into concrete args |
| MLX/other runtimes | Use a different local inference stack | May require changing workflow or model format | Keeps Ollama-first workflow and selected model |
| KV compression libraries | Add algorithm-specific compression | Often requires code integration or runtime changes | Uses available runtime controls first |

## Validate Improvement On Your Mac

Create three comparable run folders:

```bash
mackv-opt baseline-template \
  --output-dir experiments/m2-16gb \
  --models llama3.1:8b \
  --contexts 8k,16k,32k \
  --memory-budget 12GiB
```

Run the generated `run.sh` files under:

```text
experiments/m2-16gb/llama3.1-8b/default/
experiments/m2-16gb/llama3.1-8b/manual-num-ctx/
experiments/m2-16gb/llama3.1-8b/mackv-opt/
```

Compare the results:

```bash
mackv-opt compare \
  default=experiments/m2-16gb/llama3.1-8b/default/full-run.json \
  manual-num-ctx=experiments/m2-16gb/llama3.1-8b/manual-num-ctx/full-run.json \
  mackv-opt=experiments/m2-16gb/llama3.1-8b/mackv-opt/full-run.json \
  --baseline-label default \
  --format markdown
```

For a multi-model matrix, use:

```bash
./scripts/run_macos_matrix.sh
MACKV_EXECUTE=1 MACKV_MEMORY_BUDGET=20GiB ./scripts/run_macos_matrix.sh
```

See [docs/MAC_VALIDATION_CHECKLIST.md](docs/MAC_VALIDATION_CHECKLIST.md) for
16GB, 32GB, and 64GB validation presets.

## Testing Without A Mac

You can test most of the project without Apple Silicon hardware:

- run the full unit suite on Windows or Linux with `python -m pytest -q`;
- run planner commands with manual metadata and `--hardware-memory`;
- run `bench`, `experiment`, `needle`, and `qa` in `--dry-run` mode;
- generate baseline folders and comparison reports using fixture JSON;
- use GitHub Actions macOS runners for packaging and CLI smoke checks;
- use a borrowed, remote, or self-hosted Apple Silicon Mac for executable
  Ollama validation.

What non-Mac testing cannot prove: Apple unified-memory pressure, Metal runtime
behavior, Ollama model throughput, and max stable context on a target Mac. Those
need an Apple Silicon machine with Ollama and the selected models installed.

## CLI Reference

Common commands:

```bash
mackv-opt profile --json
mackv-opt doctor --json
mackv-opt capabilities --json
mackv-opt collect --output-dir experiments/m2-16gb/collect --models llama3.1:8b --json
mackv-opt audit experiments/m2-16gb/collect/manifest.json --json
mackv-opt auto llama3.1:8b --memory-budget 12GiB
mackv-opt run llama3.1:8b --target-context 64k --memory-budget 12GiB
mackv-opt bench --models llama3.1:8b --contexts 8k,16k,32k --dry-run --json
mackv-opt experiment llama3.1:8b --contexts 8k,16k,32k --memory-budget 12GiB --dry-run --json
mackv-opt report full-run.json --table performance
mackv-opt plot-memory full-run.json --output memory-series.svg
```

`run`, `bench`, `needle`, `qa`, and `experiment` call local inference only when
`--execute` is passed.

## Safety And Privacy

- No model replacement.
- No weight rewriting.
- No automatic model quantization.
- No cloud service.
- No prompt or output upload.
- No persistent Ollama configuration changes unless the user runs the printed
  command or passes `--execute`.

## Repository Profile

- About: KV cache and context planner for longer local LLM contexts on Apple
  Silicon Macs with Ollama-compatible benchmarks.
- Homepage: <https://github.com/Lin-Aurora/MacKV-Opt#readme>
- Topics: `apple-silicon`, `ollama`, `llama-cpp`, `kv-cache`, `local-llm`,
  `llm-inference`, `macos`, `benchmark`, `long-context`, `mlx`, `gguf`,
  `ollama-tools`.

The same values are recorded in
[docs/GITHUB_REPOSITORY_METADATA.md](docs/GITHUB_REPOSITORY_METADATA.md). They
can be applied with `python scripts/sync_github_metadata.py --apply` when a
GitHub token is available in the environment.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The CI workflow validates packaging, unit tests, and non-executing CLI smoke
paths on Ubuntu and macOS. It does not download models or call Ollama inference.

## Roadmap

- Add a packaged installer and shell completions.
- Add richer prompt suites for local validation.
- Add clearer warning thresholds for swap, pageouts, and throughput collapse.
- Add optional adapters for direct llama.cpp execution.
- Improve automatic metadata extraction from Ollama and GGUF files.
