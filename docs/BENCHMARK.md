# Benchmark And Validation Guide

This guide explains how to check whether MacKV-Opt improves local Ollama runs on
your Mac. It focuses on practical outputs: max stable context, tokens/s, memory
pressure, swap/pageout deltas, and quality sanity checks.

## What Counts As An Improvement

MacKV-Opt is useful when it improves at least one of these outcomes without
changing the selected model:

- a longer stable context under the same memory budget;
- fewer failed or killed long-context runs;
- lower swap growth and fewer pageouts at the same context;
- less throughput collapse under memory pressure;
- similar Needle/QA correctness at the chosen context.

The most important comparison is:

```text
default Ollama vs manual num_ctx vs MacKV-Opt
```

All three should use the same model, prompt/task, repeat count, and machine
state.

## Collect Readiness Data

```bash
mackv-opt collect \
  --output-dir experiments/MACHINE/collect \
  --models llama3.1:8b,qwen2.5:7b,mistral:7b \
  --model-metadata-overrides model-overrides.json \
  --json > experiments/MACHINE/collect.stdout.json

mackv-opt audit experiments/MACHINE/collect/manifest.json \
  --fail-on-missing-metadata \
  --require-apple-silicon \
  --output experiments/MACHINE/collect-audit.json \
  --markdown-output experiments/MACHINE/collect-audit.md \
  --json > experiments/MACHINE/collect-audit.stdout.json
```

`collect` records:

- hardware profile;
- macOS version, power source, power mode, and thermal state when available;
- Ollama and llama.cpp capability checks;
- local Ollama model metadata;
- model metadata gaps that affect KV cache planning.

`audit` is a pass/warn/fail gate. Use it to catch missing model metadata,
missing files, unavailable models, and non-Apple-Silicon environments before
running long benchmarks.

Render readiness tables:

```bash
mackv-opt report experiments/MACHINE/collect --table readiness-compact
mackv-opt report experiments/MACHINE/collect --table readiness
```

## Validate A Single Model

Generate a plan:

```bash
mackv-opt plan MODEL \
  --target-context 64k \
  --memory-budget 20GiB \
  --json > plan-MODEL-64k-20GiB.json
```

Run the planner-backed flow:

```bash
mackv-opt experiment MODEL \
  --contexts 8k,16k,32k,64k \
  --memory-budget 20GiB \
  --execute --json \
  --depths 10%,50%,90% \
  --repeats 3 \
  --max-swap 512MiB \
  --min-tokens-per-second 2.0 \
  --stable-context-policy all \
  --memory-sample-interval 0.5 \
  --include-memory-series \
  --output-dir experiments/MACHINE/MODEL \
  --output-prefix full-run
```

Use `--dry-run` first to inspect commands without calling Ollama.

## Baseline Comparison

Create the standard baseline folders:

```bash
mackv-opt baseline-template \
  --output-dir experiments/MACHINE \
  --models llama3.1:8b,qwen2.5:7b \
  --contexts 8k,16k,32k,64k \
  --memory-budget 20GiB
```

This creates:

```text
experiments/MACHINE/MODEL/
  default/
  manual-num-ctx/
  mackv-opt/
  report-tables/
```

Each baseline folder contains a `run.sh`, `manifest.json`, and README. Run the
three baselines, then compare:

```bash
mackv-opt compare \
  default=experiments/MACHINE/MODEL/default/full-run.json \
  manual-num-ctx=experiments/MACHINE/MODEL/manual-num-ctx/full-run.json \
  mackv-opt=experiments/MACHINE/MODEL/mackv-opt/full-run.json \
  --baseline-label default \
  --format markdown
```

For a machine-level summary:

```bash
mackv-opt baseline-summary experiments/MACHINE \
  --output experiments/MACHINE/baseline-summary.md
mackv-opt baseline-summary experiments/MACHINE \
  --format csv \
  --output experiments/MACHINE/baseline-summary.csv
```

## Matrix Script

For repeated multi-model validation:

```bash
./scripts/run_macos_matrix.sh
MACKV_EXECUTE=1 MACKV_MEMORY_BUDGET=20GiB ./scripts/run_macos_matrix.sh
```

The script defaults to dry-run. In executable mode it calls local Ollama,
collects memory/timing/quality logs, writes fixed report tables, and generates
compare tables when enough baseline outputs are present.

Common environment variables:

- `MACKV_MODELS`: comma-separated model tags.
- `MACKV_CONTEXTS`: comma-separated contexts.
- `MACKV_MEMORY_BUDGET`: planner memory budget.
- `MACKV_MODEL_METADATA_OVERRIDES`: optional JSON file for missing model fields.
- `MACKV_REPEATS`: repeated runs per model/context/quality cell.
- `MACKV_MAX_SWAP`: swap growth threshold for unstable runs.
- `MACKV_MIN_TOKENS_PER_SEC`: optional output throughput floor.
- `MACKV_STABLE_POLICY`: max-stable-context policy: `any`, `all`, or `fraction`.
- `MACKV_INCLUDE_SERIES=1`: include full memory time-series samples.
- `MACKV_OUTPUT_ROOT`: output root.

See [MAC_VALIDATION_CHECKLIST.md](MAC_VALIDATION_CHECKLIST.md) for 16GB, 32GB,
and 64GB presets.

## Metrics

Executable benchmark logs can include:

- prompt tokens;
- generated tokens;
- first-token latency in milliseconds;
- total generation time;
- output tokens per second;
- peak process RSS;
- memory pressure state;
- swap bytes before and after;
- `vm_stat` pagein/pageout/swapin/swapout deltas on macOS;
- stability label and instability reason;
- max stable context per model;
- Needle and QA quality scores;
- repeat means, standard deviations, min/max, success rate, and accuracy.

Suggested macOS commands for manual cross-checking:

```bash
memory_pressure
vm_stat
ps -o pid,rss,command -p PID
```

Process RSS is a practical userspace proxy. For exact unified-memory attribution,
pair MacKV-Opt logs with Activity Monitor, `powermetrics`, or a dedicated local
sampler.

## Stability Policy

`bench --execute` and `experiment --execute` mark a run unstable when:

- the run status is not `ok`;
- macOS reports critical memory pressure;
- swap growth exceeds `--max-swap`;
- tokens/s falls below `--min-tokens-per-second`, when configured.

`--stable-context-policy` controls repeated runs:

- `any`: at least one stable repeat is enough;
- `all`: every repeat must be stable;
- `fraction`: stable repeat fraction must meet `--min-stable-fraction`.

For user-facing comparisons, `all` with at least three repeats is the most
conservative setting.

## Quality Checks

Needle-in-a-Haystack:

```bash
mackv-opt needle \
  --models llama3.1:8b,qwen2.5:7b \
  --contexts 8k,16k,32k,64k \
  --depths 10%,50%,90% \
  --execute --json \
  --repeats 3 \
  --output-dir experiments/m2-16gb \
  --output-prefix needle-retrieval
```

Long-document QA:

```bash
mackv-opt qa \
  --models llama3.1:8b,qwen2.5:7b \
  --contexts 8k,16k,32k,64k \
  --execute --json \
  --repeats 3 \
  --output-dir experiments/m2-16gb \
  --output-prefix qa-retrieval
```

Custom JSONL QA:

```jsonl
{"id":"q1","document":"The answer is Alpine-42.","question":"What is the code?","expected_answer":"Alpine-42","source":"mini-qa"}
```

```bash
mackv-opt qa \
  --models llama3.1:8b,qwen2.5:7b \
  --contexts 8k,16k \
  --dataset mini-qa.jsonl \
  --execute --json \
  --output-dir experiments/m2-16gb \
  --output-prefix qa-mini
```

These checks are sanity tests. They help catch obvious retrieval collapse when a
larger context or compact KV strategy is selected.

## No Mac Available

Without an Apple Silicon Mac, you can still verify:

- package import and CLI parser behavior;
- planner math with manual metadata;
- `doctor`, `collect`, `audit`, `report`, `compare`, and baseline generation;
- dry-run benchmark and quality matrices;
- GitHub Actions macOS smoke tests.

You cannot validate Apple unified-memory pressure, Metal execution, Ollama
tokens/s, or max stable context for a target Mac without Apple Silicon hardware.
Use a borrowed Mac, a remote Mac mini, or a self-hosted GitHub Actions runner
when executable validation is needed.
