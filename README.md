# MacKV-Opt

MacKV-Opt is a local optimization helper for running large language models on
Apple Silicon Macs. It keeps the user's chosen model unchanged and plans safer
context, KV cache, and memory settings around the model instead.

The first release is an Ollama-compatible sidecar:

- profiles the Mac, available unified memory, memory pressure, and local Ollama
  models;
- estimates model memory, KV cache growth, runtime overhead, and concurrency
  risk;
- recommends `num_ctx`, llama.cpp KV cache types, KV offload, and a safe memory
  budget;
- prints an Ollama command or a llama.cpp argument set without rewriting model
  weights, quantizing user models, uploading prompts, or changing cloud state;
- creates benchmark matrices that can feed reproducible systems-paper
  experiments.

The practical target is simple: on the same Mac memory budget, make longer
contexts easier to run without falling into macOS memory pressure, swap storms,
or sudden throughput collapse.

## Contents

- [Why this exists](#why-this-exists)
- [Install](#install)
- [Quick start](#quick-start)
- [CLI](#cli)
- [What the planner does](#what-the-planner-does)
- [Benchmark artifacts](#benchmark-artifacts)
- [Safety and privacy](#safety-and-privacy)
- [Research basis](#research-basis)
- [Development](#development)

## Why This Exists

As of 2026-06-07, Mac local LLM users sit between two fast-moving realities:

- Apple Silicon unified memory lets desktop users run capable local models, but
  long context inference makes KV cache memory grow linearly with context length.
- Ollama has a friendly model UX, while the lower-level llama.cpp runtime exposes
  important controls such as `--ctx-size`, `--kv-offload`, `--cache-type-k`, and
  `--cache-type-v`.
- Real users report pain around automatic `num_ctx` sizing, tiered context memory
  exhaustion, and high Mac RAM usage.
- Research is rapidly improving KV cache quantization and compression, but most
  work is not packaged as a simple Mac/Ollama optimization tool.

MacKV-Opt starts with the systems layer that is immediately useful: model-aware
KV budget planning around existing runtimes. Later versions can add deeper
Metal/llama.cpp implementations of compressed KV attention.

## Install

Developer install from this repository:

```bash
python -m pip install -e .
python -m pip install -e ".[dev]"
```

Run without installing:

```bash
python -m mackv_opt.cli --help
```

MacKV-Opt has no runtime dependency beyond Python 3.10+ in the MVP. Ollama is
optional for planning with manual model metadata, but required for automatic
model discovery and `run --execute`.

## Quick Start

```bash
mackv-opt doctor
mackv-opt plan llama3.1:8b --target-context 64k --memory-budget 12GiB
mackv-opt baseline-template \
  --output-dir experiments/m2-16gb \
  --models llama3.1:8b \
  --contexts 8k,16k,32k \
  --memory-budget 12GiB
mackv-opt experiment llama3.1:8b \
  --contexts 8k,16k,32k \
  --memory-budget 12GiB \
  --dry-run --json
```

For executable paper runs on Apple Silicon, start with
[docs/REAL_MAC_EXPERIMENT_CHECKLIST.md](docs/REAL_MAC_EXPERIMENT_CHECKLIST.md).

## CLI

Profile local hardware and Ollama:

```bash
mackv-opt profile
mackv-opt profile --json
```

Run a read-only preflight check before experiments:

```bash
mackv-opt doctor
mackv-opt doctor --json
```

`doctor` combines hardware, Ollama, llama.cpp, memory pressure, swap sampling,
and local model checks into one readiness payload. It is intended for
experiment setup and paper artifact auditing; it does not run inference or
change local runtime configuration.

Collect a complete read-only preflight artifact bundle:

```bash
mackv-opt collect \
  --output-dir experiments/m2-16gb/preflight \
  --models llama3.1:8b,qwen2.5:7b \
  --model-metadata-overrides model-overrides.json \
  --json
```

`collect` writes `doctor.json`, `machine-profile.json`,
`runtime-capabilities.json`, `manifest.json`, `manifest.md`, and per-model raw
`ollama show --json` plus normalized planner metadata under `models/`. Each
model record includes a metadata audit for KV-budget-critical fields such as
model size, hidden size, layer count, attention heads, and KV heads.
On macOS, the machine profile also records macOS build, kernel version, power
source, power mode, and thermal state when the local OS exposes them.

Use `--model-metadata-overrides` when Ollama omits fields needed for KV budget
estimates. The override file fills metadata only; it does not alter the model
weights, quantization, prompt, or selected model:

```json
{
  "models": {
    "llama3.1:8b": {
      "size_bytes": 5100000000,
      "parameter_count": 8000000000,
      "hidden_size": 4096,
      "layer_count": 32,
      "attention_head_count": 32,
      "kv_head_count": 8,
      "max_context": 131072,
      "family": "llama",
      "architecture": "llama"
    }
  }
}
```

Audit a collection before executable paper runs:

```bash
mackv-opt audit experiments/m2-16gb/preflight/manifest.json \
  --fail-on-missing-metadata \
  --require-apple-silicon \
  --output experiments/m2-16gb/preflight/audit.json \
  --markdown-output experiments/m2-16gb/preflight/audit.md
```

`audit` returns a non-zero exit code when the collection is not ready. The
matrix script uses this as a gate for `MACKV_EXECUTE=1`, so failed doctor checks
unverified hardware, unavailable models, or missing KV metadata do not silently
produce paper-looking artifacts.

Check local runtime capability surfaces:

```bash
mackv-opt capabilities
mackv-opt capabilities --json
```

Plan a long-context run for an Ollama model:

```bash
mackv-opt plan llama3.1:8b \
  --target-context 64k \
  --memory-budget 20GiB
```

`plan`, `run`, and `experiment` perform a read-only capability check by default
and include `runtime_advice` in JSON output. Use `--skip-capability-check` when
you want deterministic offline planning without probing local commands.

Plan with manual metadata when Ollama is unavailable or model metadata is
incomplete:

```bash
mackv-opt plan llama3.1:8b \
  --target-context 64k \
  --memory-budget 12GiB \
  --model-size 4.8GiB \
  --parameters 8B \
  --hidden-size 4096 \
  --layers 32 \
  --heads 32 \
  --kv-heads 8 \
  --hardware-memory 16GiB
```

Print the Ollama command MacKV-Opt would run:

```bash
mackv-opt run llama3.1:8b \
  --target-context 64k \
  --memory-budget 12GiB \
  --prompt "Summarize this document"
```

Execute the command only when you are ready:

```bash
mackv-opt run llama3.1:8b --target-context 64k --execute
```

Build a benchmark matrix:

```bash
mackv-opt bench \
  --models llama3.1:8b,qwen2.5:7b \
  --contexts 8k,16k,32k,64k \
  --dry-run --json
```

Create paper baseline artifact directories before running comparisons:

```bash
mackv-opt baseline-template \
  --output-dir experiments/m2-16gb \
  --models llama3.1:8b,qwen2.5:7b \
  --contexts 8k,16k,32k \
  --memory-budget 12GiB \
  --json
```

This creates `default/`, `manual-num-ctx/`, and `mackv-opt/` directories under
each model folder. The default baseline uses
`bench --use-ollama-default-options`, so it records structured JSON without
sending `num_ctx`; the manual baseline sends an explicit context; the MacKV-Opt
baseline runs the planner-backed experiment.

Run the matrix through the local Ollama API and collect metrics:

```bash
mackv-opt bench \
  --models llama3.1:8b \
  --contexts 8k,16k,32k \
  --execute --json \
  --prompt "Write a concise summary of Apple Silicon local LLM inference." \
  --num-predict 128 \
  --repeats 3 \
  --max-swap 512MiB \
  --min-tokens-per-second 2.0 \
  --stable-context-policy all \
  --memory-sample-interval 0.5 \
  --include-memory-series \
  --output-dir experiments/m2-16gb \
  --output-prefix llama3-8b-longctx
```

With `--output-dir`, MacKV-Opt writes JSON, Markdown, and CSV artifacts by
default and includes the written paths in the command output. Use
`--save-formats json,csv` to limit formats.

Render a plan, benchmark matrix, experiment JSON file, or readiness artifact
bundle into a paper table:

```bash
mackv-opt report plan-llama3-64k.json --format markdown
mackv-opt report experiment-runs.json --format csv
mackv-opt report experiments/m2-16gb/preflight --table readiness-compact
mackv-opt report experiments/m2-16gb/preflight --table readiness --format markdown
mackv-opt report experiments/m2-16gb/preflight/manifest.json \
  experiments/m2-16gb/preflight/audit.json \
  --table readiness --format csv
mackv-opt report full-run.json --table context --format markdown
mackv-opt report full-run.json --table performance --format markdown
mackv-opt report full-run.json --table memory --format markdown
mackv-opt report full-run.json --table quality --format markdown
mackv-opt report full-run.json --table stability --format markdown
mackv-opt report full-run.json \
  --output-dir paper-tables \
  --output-prefix m2-16gb-llama3
mackv-opt plot-memory full-run.json --output paper-tables/memory-series.svg
```

Compare multiple artifacts as a baseline table:

```bash
mackv-opt compare \
  default=experiments/m2-16gb/llama3-default/full-run.json \
  manual-num-ctx=experiments/m2-16gb/llama3-manual-num-ctx/full-run.json \
  mackv-opt=experiments/m2-16gb/llama3-mackv-opt/full-run.json \
  --baseline-label default \
  --format markdown
```

Build the RQ1 paper table from a machine directory:

```bash
mackv-opt rq1-summary experiments/m2-16gb \
  --output experiments/m2-16gb/rq1-summary.md
```

`rq1-summary` scans each model directory for
`default/full-run.json`, `manual-num-ctx/full-run.json`, and
`mackv-opt/full-run.json`, then reports maximum stable context and MacKV-Opt's
ratio against both baselines. Use `--format csv` or `--format json` for
supplementary artifacts.

Run a Needle-in-a-Haystack long-context retrieval check:

```bash
mackv-opt needle \
  --models llama3.1:8b \
  --contexts 8k,16k,32k \
  --depths 10%,50%,90% \
  --execute --json \
  --repeats 3 \
  --output-dir experiments/m2-16gb \
  --output-prefix llama3-8b-needle
```

Run a LongBench-style long-document QA sanity check:

```bash
mackv-opt qa \
  --models llama3.1:8b \
  --contexts 8k,16k,32k \
  --execute --json \
  --repeats 3 \
  --output-dir experiments/m2-16gb \
  --output-prefix llama3-8b-qa
```

Use a custom JSONL dataset when you want fixed documents and questions:

```jsonl
{"id":"q1","document":"The answer is Alpine-42.","question":"What is the code?","expected_answer":"Alpine-42","source":"mini-qa"}
```

```bash
mackv-opt qa \
  --models llama3.1:8b \
  --contexts 8k,16k \
  --dataset mini-qa.jsonl \
  --dry-run --json
```

Run an end-to-end experiment that plans contexts, benchmarks Ollama timing, and
checks Needle retrieval plus QA quality:

```bash
mackv-opt experiment llama3.1:8b \
  --contexts 8k,16k,32k,64k \
  --memory-budget 12GiB \
  --execute --json \
  --depths 10%,50%,90% \
  --repeats 3 \
  --memory-sample-interval 0.5 \
  --output-dir experiments/m2-16gb/llama3-8b \
  --output-prefix mackv-opt-full
```

Use `experiment --dry-run` first to inspect planned contexts and generated jobs
without calling Ollama.

Run a multi-model macOS experiment matrix:

```bash
./scripts/run_macos_matrix.sh
MACKV_EXECUTE=1 MACKV_MEMORY_BUDGET=20GiB ./scripts/run_macos_matrix.sh
```

The script defaults to dry-run and writes per-model `full-run` artifacts plus
`paper-tables`; it also writes a top-level `collect/` preflight bundle for the
machine and requested models. Configure it with environment variables such as
`MACKV_MODELS`, `MACKV_CONTEXTS`, `MACKV_REPEATS`, `MACKV_MAX_SWAP`,
`MACKV_MIN_TOKENS_PER_SEC`, `MACKV_STABLE_POLICY`,
`MACKV_MIN_STABLE_FRACTION`, `MACKV_COMPARE_LABELS`,
`MACKV_COMPARE_BASELINE`, `MACKV_FAIL_ON_MISSING_METADATA`,
`MACKV_REQUIRE_APPLE_SILICON`, `MACKV_MODEL_METADATA_OVERRIDES`,
`MACKV_OUTPUT_ROOT`, and `MACKV_INCLUDE_SERIES`. When `MACKV_EXECUTE=1`, the
script refuses to run if `mackv-opt audit` fails; execute mode defaults to
requiring Apple Silicon and complete KV-budget metadata.
For each model, the script also copies its generated artifact to
`mackv-opt/full-run.json` and pre-creates sibling baseline directories such as
`default/` and `manual-num-ctx/`; when at least two labeled `full-run.json`
artifacts are present it writes per-model compare tables plus
`matrix-compare.md/csv`.

## What The Planner Does

The core planner estimates:

```text
estimated_total = model_weights + KV_cache(context, layers, hidden, GQA_ratio, KV_type) + runtime_overhead
```

It then searches context and KV cache strategies:

1. Try the requested target context.
2. Prefer higher-quality KV types when they fit.
3. Fall back to more compact KV types when memory is tight.
4. Reduce context by powers of two only when no KV strategy fits.
5. Warn when the target exceeds model metadata, the platform is not Apple
   Silicon, or even the minimum strategy exceeds the budget.

The first version emits two strategy surfaces:

- Ollama options: currently `num_ctx` and `num_gpu`.
- llama.cpp args: `--ctx-size`, `--cache-type-k`, `--cache-type-v`,
  `--kv-offload`.

Ollama does not expose every llama.cpp KV cache option through `ollama run` in a
stable user-facing way, so MacKV-Opt keeps the llama.cpp arguments explicit in
  the plan instead of pretending they are always applied by Ollama.
Use `mackv-opt capabilities --json` to record which local runtime commands and
flags were actually detected for a paper run.
The planner also embeds `runtime_advice` so artifacts can distinguish memory
planning success from locally verified runtime flag support.

## Benchmark Artifacts

`bench --execute` records Ollama API timing fields and derived metrics:

- `eval_count`, `eval_duration_ns`, and `tokens_per_second`;
- `prompt_eval_count`, `prompt_eval_duration_ns`, and
  `prompt_tokens_per_second`;
- `total_duration_ns`, `load_duration_ns`, and wall-clock time;
- macOS memory pressure and swap delta where available;
- macOS `vm_stat` page-in/pageout and swap-in/swapout deltas where available;
- best-effort time-series Ollama process RSS as `peak_memory_bytes`;
- `memory_samples` and `memory_sample_interval_seconds` for sampler auditing;
- optional `memory_series` samples when `--include-memory-series` is set;
- heuristic stability labels as `stable` and `stability_reason`;
- per-model `stability_summary.max_stable_context_by_model`;
- structured errors when the local Ollama API is unavailable or a run fails.

Use `--repeats N` on `bench`, `needle`, `qa`, or `experiment` to repeat every
model/context/task cell. Executed payloads include per-run `repeat_index` plus
`repeat_summaries` with means, sample standard deviations, min/max values,
success rate, and quality accuracy where applicable.

Process RSS is a practical approximation for early experiments. On Apple
Silicon, full unified-memory attribution still needs careful OS/runtime
cross-checking for final paper claims.
`--include-memory-series` is useful for plots and debugging but can make large
multi-model experiment JSON files much bigger.

By default, `bench --execute` and executable `experiment` runs mark a cell
unstable when the run status is not `ok`, macOS reports critical memory
pressure, or observed swap growth exceeds `512MiB`. Use `--max-swap` to tighten
or relax the swap threshold, and `--min-tokens-per-second` to mark throughput
collapse as unstable. These labels are designed for repeatable experiment logs;
final paper claims still need real Apple Silicon validation across machines.

For repeated runs, `--stable-context-policy` controls when a context contributes
to `max_stable_context`: `any` accepts at least one stable repeat, `all` requires
every repeat to be stable, and `fraction` uses `--min-stable-fraction`. The
default is `any` for exploratory sweeps; paper tables should normally use `all`
or an explicitly justified fraction such as `0.67`.

`needle --execute` generates synthetic long documents with one hidden key at a
chosen depth, asks the model to return the exact key, and records `found` plus
`quality_score`. This is a fast retrieval sanity check, not a replacement for
LongBench or task-specific quality evaluation.

`qa --execute` generates synthetic long-document QA probes, or reads JSONL
records with `document`, `question`, and `expected_answer` or `answer` fields.
It records exact-answer containment as `found` and `quality_score`. This is a
LongBench-style sanity check for regression tracking; final paper claims should
also include official or task-specific quality suites.

`report --table` provides fixed paper-table views:

- `readiness`: hardware, runtime capability, doctor/audit status, referenced
  artifact presence, model metadata audit, and missing KV-budget fields from
  `collect`, `audit`, `doctor`, `profile`, or `capabilities` artifacts;
- `readiness-compact`: one-row paper-readiness summary with Apple Silicon,
  macOS/power/thermal state, runtime support, model metadata completeness, and
  failed/warning checks;
- `context`: planned context, KV strategy, and per-model max stable context
  summary rows when executed benchmark artifacts contain `stability_summary`;
- `performance`: Ollama API throughput, latency, and repeat aggregate fields;
- `memory`: peak RSS, swap, pressure, and repeat aggregate fields;
- `quality`: Needle and QA exact-answer quality scores plus repeat accuracy.
- `stability`: per-context stable fraction, repeat policy, stable/unstable
  runs, and instability reason distribution.

With `report --output-dir`, MacKV-Opt writes all fixed tables by default.
Use `--tables readiness-compact,readiness,context,memory,stability` to write
only selected tables.

`plot-memory` renders the first available `memory_series` from a benchmark or
experiment JSON as a dependency-free SVG line chart.

`compare` renders multiple plan, bench, or experiment artifacts into one
baseline table with label, model, max stable context, best throughput, latency,
memory, quality, and stability policy fields. Use it for the paper's default
Ollama vs manual context vs MacKV-Opt comparison. The first input is the
baseline unless `--baseline-label` is provided; relative columns report context,
throughput, latency, and memory ratios plus quality accuracy deltas.

## Safety And Privacy

MacKV-Opt is local-only by design:

- it does not modify, replace, or re-quantize the user's selected model;
- it does not upload prompts, outputs, model metadata, benchmark logs, or
  hardware information;
- it does not change Ollama configuration unless the user explicitly executes a
  printed command;
- it treats planner output as a recommendation, not a guarantee, because memory
  pressure depends on background apps and runtime version.

## Research Basis

MacKV-Opt is positioned as a systems and open-source-tool contribution, not as a
claim of a new compression algorithm in the MVP.

Relevant runtime/project surface:

- [llama.cpp](https://github.com/ggml-org/llama.cpp): context size, KV offload,
  and cache type controls.
- [Ollama issue 12353](https://github.com/ollama/ollama/issues/12353): user pain
  around automatic `num_ctx` sizing.
- [Ollama issue 14116](https://github.com/ollama/ollama/issues/14116): tiered
  context length exhausting memory.
- [Ollama issue 4151](https://github.com/ollama/ollama/issues/4151): high RAM
  usage on Mac.
- [MLX-LM](https://github.com/ml-explore/mlx-lm), [vllm-mlx](https://github.com/waybarrios/vllm-mlx),
  and [NVIDIA kvpress](https://github.com/NVIDIA/kvpress): reference projects
  for local Apple inference and KV compression tooling.

Representative research threads:

- KVarN (2026-06-02), RedKnot (2026-06-04), and Open-TQ-Metal (2026-04-18) as
  recent 2026 evidence that KV/cache and Apple Silicon inference remain active
  research targets as of the 2026-06-07 evidence refresh.
- [ChunkKV](https://arxiv.org/abs/2502.00299), [KIVI](https://arxiv.org/abs/2402.02750),
  [KVQuant](https://arxiv.org/abs/2401.18079), and StreamingLLM as key context
  for KV cache quantization, eviction, streaming, and long-context serving.

See [docs/PAPER_PLAN.md](docs/PAPER_PLAN.md) and
[docs/BENCHMARK.md](docs/BENCHMARK.md) for the publishable experiment plan.

## Development

Run tests:

```bash
python -m pytest -q
```

The GitHub Actions workflow runs the test suite and non-executing CLI smoke
checks on Ubuntu and macOS across Python 3.10 and 3.12. It does not require
Ollama or any model downloads.

Useful smoke checks:

```bash
python -m mackv_opt.cli profile --json
python -m mackv_opt.cli doctor --json
python -m mackv_opt.cli collect --output-dir work/collect-smoke --models llama3.1:8b --json
python -m mackv_opt.cli audit work/collect-smoke/manifest.json --no-require-artifacts --json --output work/collect-smoke/audit.json --markdown-output work/collect-smoke/audit.md
python -m mackv_opt.cli plan llama3.1:8b --target-context 64k --memory-budget 12GiB --model-size 4.8GiB --hidden-size 4096 --layers 32 --heads 32 --kv-heads 8 --hardware-memory 16GiB
python -m mackv_opt.cli run llama3.1:8b --target-context 64k --memory-budget 12GiB --model-size 4.8GiB --hidden-size 4096 --layers 32 --heads 32 --kv-heads 8 --hardware-memory 16GiB --json
python -m mackv_opt.cli bench --models llama3.1:8b,qwen2.5:7b --contexts 8k,16k --dry-run --json
python -m mackv_opt.cli baseline-template --output-dir work/baseline-template-smoke --models llama3.1:8b --contexts 8k,16k --memory-budget 12GiB --json
python -m mackv_opt.cli rq1-summary work/baseline-template-smoke --format markdown
python -m mackv_opt.cli bench --models __missing_model__ --contexts 1k --execute --timeout 0.1 --num-predict 1 --repeats 2 --max-swap 512MiB --stable-context-policy all --json
python -m mackv_opt.cli compare default=work/stability-table-smoke/missing.json --format markdown
python -m mackv_opt.cli qa --models llama3.1:8b --contexts 8k --dry-run --json
```

## Roadmap

- Add richer benchmark prompt suites and process-level peak memory sampling.
- Add richer LongBench/task-specific adapters beyond the current synthetic QA
  and Needle-in-a-Haystack sanity checks.
- Add Ollama version detection and runtime capability probing for cache-type
  support.
- Validate 16GB, 32GB, 64GB, and 128GB Apple Silicon machines across M1-M4.
- Add quality checks: Needle-in-a-Haystack, LongBench-style retrieval, and
  perplexity deltas under KV compression.
- Explore deeper llama.cpp/Metal patches for fused compressed attention or
  algorithmic KV strategies inspired by KIVI, KVQuant, KVarN, ChunkKV, RedKnot,
  and related work.
