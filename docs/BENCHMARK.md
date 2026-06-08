# Benchmark Protocol

This protocol is designed to turn MacKV-Opt from a useful CLI into a defensible
systems paper artifact.

Date baseline: 2026-06-07.

## Goals

Measure whether MacKV-Opt can run longer stable contexts under the same memory
budget while preserving usable latency, throughput, and long-context quality.

## Required Machine Metadata

Collect once per machine:

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

Also record:

- `collect/manifest.json` and `collect/manifest.md`;
- doctor status, failed checks, warnings, and recommended next steps;
- macOS version and kernel version.
- Apple chip and unified memory size.
- Ollama version.
- llama.cpp commit or release.
- Detected runtime support for `num_ctx`, `--ctx-size`, `--cache-type-k`,
  `--cache-type-v`, and `--kv-offload`.
- Power mode and whether the machine is plugged in.
- Thermal state.
- Background workload notes.

`doctor.json` is the preflight readiness artifact. It combines hardware,
runtime capability, memory pressure, swap sampler availability, Ollama model
count, and paper-readiness checks. A warning status can still be useful for
development smoke tests, but paper-grade executed runs should explain or
resolve every warning before claiming maximum stable context or memory-pressure
improvements.

`collect` writes the same `doctor.json`, `machine-profile.json`, and
`runtime-capabilities.json` files, then adds per-model raw `ollama show --json`
payloads, normalized planner profiles, and metadata audit results under
`collect/models/`. This is the preferred paper artifact because it preserves
both the original runtime metadata and MacKV-Opt's normalized interpretation.
`audit` reads `collect/manifest.json` and turns those checks into a pass/warn/fail
gate. Use `--fail-on-missing-metadata` for final paper runs so hidden-size,
layer-count, attention-head, or KV-head gaps cannot silently become planner
fallback estimates. Use `--require-apple-silicon` for final paper runs so
development machines cannot be confused with Apple Silicon evidence.
Render the combined readiness evidence before any executable matrix:

```bash
mackv-opt report experiments/MACHINE/collect --table readiness-compact > table-readiness-compact.md
mackv-opt report experiments/MACHINE/collect --table readiness > table-readiness.md
mackv-opt report experiments/MACHINE/collect/manifest.json \
  experiments/MACHINE/collect-audit.json \
  --table readiness --format csv > table-readiness.csv
```

The compact readiness table is the one-row paper gate. The full readiness table
keeps hardware, macOS build, power, thermal state, runtime capabilities,
doctor/audit status, referenced artifacts, model metadata audit status, manual
override use, and missing KV-budget-critical fields.
The equivalent manual commands are:

```bash
mackv-opt doctor --json > doctor.json
mackv-opt profile --json > machine-profile.json
mackv-opt capabilities --json > runtime-capabilities.json
ollama show MODEL --json > model-MODEL.json
mackv-opt audit collect/manifest.json --fail-on-missing-metadata --require-apple-silicon
```

## Required Model Metadata

For each Ollama model:

```bash
ollama show MODEL --json > model-MODEL.json
```

Record:

- model name and tag;
- GGUF quantization type;
- model size on disk;
- parameter count;
- hidden size, layer count, attention heads, KV heads;
- model maximum context.

When using `mackv-opt collect`, inspect each model's `metadata_audit`. Missing
`size_bytes`, `hidden_size`, `layer_count`, `attention_head_count`, or
`kv_head_count` means the planner may fall back to conservative estimates, and
the paper should either fill the fields manually or report the limitation.
Use a model metadata override file only to fill missing metadata; it must not
change the selected model, model file, quantization, or prompt.

Example override file:

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

## Planning Runs

Generate a plan for each model/context/budget pair:

```bash
mackv-opt plan MODEL \
  --target-context 64k \
  --memory-budget 20GiB \
  --json > plan-MODEL-64k-20GiB.json
```

Plan artifacts include `runtime_advice` by default. This field records whether
Ollama and llama.cpp command/flag support was locally confirmed. Use
`--skip-capability-check` only when producing offline estimates where local
runtime probing is intentionally disabled.

Recommended end-to-end path for a full MacKV-Opt run:

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
  --output-dir experiments/MACHINE/MODEL \
  --output-prefix full-run
```

This writes a single experiment JSON payload with:

- hardware and model metadata;
- planner outputs for every target context;
- runtime advice for Ollama and llama.cpp flag support;
- Ollama API timing benchmark runs;
- Needle-in-a-Haystack quality runs;
- LongBench-style QA quality runs;
- per-cell repeat summaries with mean, sample standard deviation, success rate,
  and quality accuracy;
- per-run stability labels plus per-model maximum stable context summaries;
- a summary with plan count, benchmark run count, repeat-summary counts,
  maximum planned context, maximum stable context by model, and Needle/QA
  accuracy.

Use `--dry-run` instead of `--execute` to inspect the experiment matrix before
calling Ollama.

For repeated multi-model runs on a Mac, use the repository script:

```bash
./scripts/run_macos_matrix.sh
MACKV_EXECUTE=1 MACKV_MEMORY_BUDGET=20GiB ./scripts/run_macos_matrix.sh
```

The script writes a top-level `collect/` preflight bundle, mirrors
`doctor.json`, `machine-profile.json`, and `runtime-capabilities.json` for
compatibility, runs `mackv-opt experiment` for every model in `MACKV_MODELS`,
and generates paper tables through `mackv-opt report`. It defaults to dry-run.
Set `MACKV_EXECUTE=1` only when the Mac is ready for real Ollama API runs. In
execute mode, the script refuses to continue when `mackv-opt audit` fails.
It also pre-creates default, manual `num_ctx`, and MacKV-Opt baseline artifact
directories by default. Set `MACKV_WRITE_BASELINE_TEMPLATES=0` to skip that
template step.

Real Mac execute presets:

```bash
# 16GB tier: conservative contexts and a 12GiB planner budget.
MACKV_MACHINE=m2-16gb \
MACKV_MEMORY_BUDGET=12GiB \
MACKV_CONTEXTS=8k,16k,32k \
MACKV_REPEATS=3 \
MACKV_STABLE_POLICY=all \
MACKV_INCLUDE_SERIES=1 \
MACKV_EXECUTE=1 \
./scripts/run_macos_matrix.sh

# 32GB tier: include 64k if the model metadata allows it.
MACKV_MACHINE=m3-32gb \
MACKV_MEMORY_BUDGET=24GiB \
MACKV_CONTEXTS=8k,16k,32k,64k \
MACKV_REPEATS=3 \
MACKV_STABLE_POLICY=all \
MACKV_INCLUDE_SERIES=1 \
MACKV_EXECUTE=1 \
./scripts/run_macos_matrix.sh

# 64GB tier: include 128k for long-context scaling and failure-boundary study.
MACKV_MACHINE=m4-64gb \
MACKV_MEMORY_BUDGET=48GiB \
MACKV_CONTEXTS=8k,16k,32k,64k,128k \
MACKV_REPEATS=3 \
MACKV_STABLE_POLICY=all \
MACKV_INCLUDE_SERIES=1 \
MACKV_EXECUTE=1 \
./scripts/run_macos_matrix.sh
```

Before running these presets, pull every model in `MACKV_MODELS`, start Ollama,
plug the Mac into AC power, disable Low Power Mode, let the machine cool down,
close heavy background apps for best-case runs, and provide
`MACKV_MODEL_METADATA_OVERRIDES` if `collect-audit` reports missing KV metadata.

Common environment variables:

- `MACKV_MODELS`: comma-separated model tags.
- `MACKV_CONTEXTS`: comma-separated contexts.
- `MACKV_MEMORY_BUDGET`: planner memory budget.
- `MACKV_MODEL_METADATA_OVERRIDES`: optional JSON file used by `collect` to
  fill missing model metadata fields.
- `MACKV_REPEATS`: repeated runs per model/context/quality cell.
- `MACKV_MAX_SWAP`: swap growth threshold for unstable runs.
- `MACKV_MIN_TOKENS_PER_SEC`: optional output throughput floor.
- `MACKV_STABLE_POLICY`: max-stable-context policy: `any`, `all`, or
  `fraction`.
- `MACKV_MIN_STABLE_FRACTION`: threshold for `fraction` policy.
- `MACKV_COMPARE_LABELS`: comma-separated baseline labels to compare inside each
  model directory. The script looks for `LABEL/full-run.json`.
- `MACKV_COMPARE_BASELINE`: label used for relative compare columns.
- `MACKV_COMPARE_CURRENT`: label assigned to the artifact written by the script.
  Default: `mackv-opt`.
- `MACKV_FAIL_ON_MISSING_METADATA`: fail audit when model KV metadata is
  incomplete. Default: `1` for execute mode, `0` for dry-run mode.
- `MACKV_REQUIRE_APPLE_SILICON`: fail audit unless collection verifies Apple
  Silicon hardware. Default: `1` for execute mode, `0` for dry-run mode.
- `MACKV_OUTPUT_ROOT`: experiment output root.
- `MACKV_INCLUDE_SERIES=1`: include full memory time-series samples.

For baseline comparisons, arrange per-model artifacts like this:

```text
experiments/MACHINE/MODEL/
  default/full-run.json
  manual-num-ctx/full-run.json
  mackv-opt/full-run.json
```

Generate this structure explicitly with:

```bash
mackv-opt baseline-template \
  --output-dir experiments/MACHINE \
  --models llama3.1:8b,qwen2.5:7b \
  --contexts 8k,16k,32k,64k \
  --memory-budget 20GiB \
  --json
```

The template writes `README.md`, `manifest.json`, and `run.sh` files for
`default`, `manual-num-ctx`, and `mackv-opt`. The default run uses
`bench --use-ollama-default-options`, which calls the Ollama API without sending
`num_ctx`; the manual run sends the selected manual context; the MacKV-Opt run
uses the planner-backed experiment. The matrix script writes the current run to
`mackv-opt/full-run.json` by default, then writes per-model
`paper-tables/MODEL-compare.md/csv` when at least two labeled artifacts exist.
It also writes `matrix-compare.md/csv` under the machine directory as a quick
cross-model summary of the current runs.

For dry-run matrix generation:

```bash
mackv-opt bench \
  --models llama3.1:8b,qwen2.5:7b,mistral:7b \
  --contexts 8k,16k,32k,64k,128k \
  --dry-run --json > bench-matrix.json
```

For executable Ollama API runs:

```bash
mackv-opt bench \
  --models llama3.1:8b,qwen2.5:7b,mistral:7b \
  --contexts 8k,16k,32k,64k \
  --execute --json \
  --prompt "Write a concise technical summary of local LLM inference." \
  --num-predict 128 \
  --repeats 3 \
  --memory-sample-interval 0.5 \
  --include-memory-series \
  --output-dir experiments/m2-16gb \
  --output-prefix ollama-long-context
```

The executable path calls the local Ollama API at
`http://localhost:11434/api/generate` with `stream=false`. It records
`prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`,
`total_duration`, and `load_duration` when Ollama returns them. MacKV-Opt derives
output tokens/s and prompt tokens/s from these fields.

When `--output-dir` is provided, MacKV-Opt writes:

- `PREFIX.json`: full machine-readable experiment payload;
- `PREFIX.md`: Markdown table for paper drafts;
- `PREFIX.csv`: spreadsheet-ready table.

Use `--save-formats json,markdown,csv` to control the generated artifacts.

## Runtime Conditions

For each run:

- Close heavy background apps when measuring best-case stable context.
- Run a second "typical desktop" profile with browser and editor open.
- Warm the model once before timing.
- Repeat each measurement at least three times.
- Treat a run as unstable if it crashes, is killed, enters sustained critical
  memory pressure, or swaps heavily enough that tokens/s collapses for more than
  one minute.

MacKV-Opt encodes this policy in executable benchmark logs. `bench --execute`
and executable `experiment` runs mark a run unstable when `status` is not `ok`,
`memory_pressure` matches a configured critical state, swap growth exceeds
`--max-swap` (`512MiB` by default), or output throughput is below
`--min-tokens-per-second` when that threshold is provided.

Each run receives `stable` and `stability_reason`. The benchmark payload also
includes `stability_summary.max_stable_context_by_model`. The
`--stable-context-policy` option controls how repeated runs count toward that
summary:

- `any`: at least one stable repeat is enough. This is useful for exploratory
  sweeps and is the default.
- `all`: every repeat for that model/context must be stable. This is the
  recommended first-paper setting when `--repeats` is at least 3.
- `fraction`: stable repeat fraction must be at least `--min-stable-fraction`.
  Use this only when the threshold is reported in the paper.

Artifacts record `stable_context_policy` and `min_stable_fraction` in both
`stability_config` and `stability_summary`.

## Metrics

Collect:

- prompt tokens;
- generated tokens;
- first-token latency in milliseconds;
- total generation time;
- output tokens per second;
- peak resident memory;
- peak unified memory pressure state;
- swap bytes before and after;
- stability label and instability reason;
- maximum stable context per model;
- stable-context policy and minimum stable-run fraction;
- exit status and error text;
- retrieval answer correctness for long-context probes.
- repeat means, sample standard deviations, min/max values, success rate, and
  quality accuracy for repeated cells.

Suggested macOS commands:

```bash
memory_pressure
vm_stat
ps -o pid,rss,command -p PID
```

Ollama and llama.cpp logs should be saved with timestamps.

MacKV-Opt samples the Ollama process RSS during each API run and stores the
largest observed value as `peak_memory_bytes`. The default interval is 0.5s and
can be changed with `--memory-sample-interval`. The JSON also records
`memory_samples` and `memory_sample_interval_seconds` so paper artifacts can
audit how dense the sampling was. This is still a best-effort userspace
approximation; final paper runs should pair it with Activity Monitor,
`powermetrics`, or a dedicated sampler when exact unified-memory attribution
matters.

On macOS, each sample also parses `vm_stat` when available. Executed benchmark
results include `pageins_delta`, `pageout_delta`, `pageout_bytes_delta`,
`swapins_delta`, and `swapouts_delta`, and full `--include-memory-series`
payloads keep the raw page counters per sample. These counters are important
for paper credibility because they distinguish harmless RSS growth from real VM
churn and pageout-driven latency collapse.

Use `--include-memory-series` when plotting memory curves or diagnosing swap
stutter. It embeds every sample as `memory_series`, including timestamp,
pressure, swap, and process RSS fields. Leave it off for large sweeps when only
summary tables are needed.

Use `--repeats N` on `bench`, `needle`, `qa`, or `experiment` to repeat each
cell. Executed JSON payloads keep every raw run with `repeat_index` and
`repeat_count`, then add `repeat_summaries` for paper tables and variance
checks. Bench payloads also add `stability_summary`, which the reporter can
render into the context and stability tables.

## Baseline Commands

Default Ollama:

```bash
ollama run MODEL "PROMPT"
```

Manual Ollama context:

```bash
ollama run MODEL --option num_ctx=65536 "PROMPT"
```

MacKV-Opt planned Ollama:

```bash
mackv-opt run MODEL --target-context 64k --memory-budget 20GiB --prompt "PROMPT"
```

llama.cpp planned args:

```bash
mackv-opt plan MODEL --target-context 64k --memory-budget 20GiB
```

Copy the emitted `llama.cpp args` into the local llama.cpp invocation for
cache-type experiments.

After each baseline produces a JSON artifact, render a direct comparison table:

```bash
mackv-opt compare \
  ollama-default=experiments/MACHINE/MODEL/default/full-run.json \
  manual-num-ctx=experiments/MACHINE/MODEL/manual-num-ctx/full-run.json \
  mackv-opt=experiments/MACHINE/MODEL/mackv-opt/full-run.json \
  --baseline-label ollama-default \
  --format markdown > table-baseline-compare.md
```

The comparison table is a compact view for RQ1/RQ4. It includes maximum stable
context, best observed throughput row, latency, memory, quality accuracy, and
the stable-context policy used by each artifact. It also reports relative
columns such as `max_stable_context_vs_baseline`,
`tokens_per_second_vs_baseline`, `peak_memory_bytes_vs_baseline`, and
`quality_accuracy_vs_baseline`. Use `--baseline-label` to make the comparison
explicit in saved paper artifacts.

After the three baseline directories are present for every model on a machine,
generate the RQ1 summary table:

```bash
mackv-opt rq1-summary experiments/MACHINE \
  --output experiments/MACHINE/rq1-summary.md
mackv-opt rq1-summary experiments/MACHINE \
  --format csv \
  --output experiments/MACHINE/rq1-summary.csv
```

The RQ1 table scans `default/full-run.json`,
`manual-num-ctx/full-run.json`, and `mackv-opt/full-run.json` under each model
directory. It reports maximum stable context for all three methods plus
MacKV-Opt's ratio against default Ollama and manual `num_ctx`.

## Quality Tasks

Use at least two classes of long-context tasks:

- Needle-in-a-Haystack: exact-key retrieval across synthetic long contexts.
- Long document QA or summarization: answer questions where the evidence appears
  near the beginning, middle, and end of the context.

MacKV-Opt includes a Needle-in-a-Haystack command:

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

It records `found`, `quality_score`, `response_excerpt`, and timing fields when
Ollama returns them. A `quality_score` of `1.0` means the exact hidden key
appeared in the model response; `0.0` means it did not.

MacKV-Opt also includes a LongBench-style QA command:

```bash
mackv-opt qa \
  --models llama3.1:8b,qwen2.5:7b \
  --contexts 8k,16k,32k,64k \
  --execute --json \
  --repeats 3 \
  --output-dir experiments/m2-16gb \
  --output-prefix qa-retrieval
```

By default it generates synthetic long documents with answer-bearing evidence
near the beginning, middle, and end of the context. To use fixed tasks, provide
a JSONL file:

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

The QA adapter records exact-answer containment as `found` and `quality_score`.
It is meant as a reproducible sanity check and regression guard, not a substitute
for official LongBench scoring.

Quality is acceptable only when compact KV strategies do not cause obvious
retrieval collapse relative to the baseline at the same context.

## Paper Tables

Recommended tables:

- Table 1: experiment readiness, hardware, runtime versions, artifact audit,
  and model metadata status (`report --table readiness-compact`, with
  `report --table readiness` as the detailed appendix table).
- Table 2: model metadata and model file sizes.
- Table 3: maximum stable context by method and memory tier
  (`rq1-summary` for the default/manual/MacKV-Opt comparison, with
  `report --table context` for the detailed appendix table).
- Table 4: latency/tokens-per-second by method
  (`report --table performance`).
- Table 5: memory pressure and swap by method (`report --table memory`).
- Table 6: quality metrics under KV strategy choices (`report --table quality`).
- Table 7: per-context stability fraction and failure reason distribution
  (`report --table stability`).

When `--repeats` is used, the fixed tables include aggregate columns such as
`tokens_per_second_mean`, `tokens_per_second_stdev`, `success_rate`,
`quality_score_mean`, and `accuracy`. Executed benchmark artifacts also expose
`stable`, `stability_reason`, `max_stable_context`, `stable_runs`, and
`unstable_runs` in the context/performance/stability table views where
applicable.

MacKV-Opt can render JSON logs into a table skeleton:

```bash
mackv-opt report plan-MODEL-64k-20GiB.json --format markdown > table-plan.md
mackv-opt report experiment-runs.json --format csv > table-runs.csv
mackv-opt report experiments/MACHINE/collect --table readiness-compact > table-readiness-compact.md
mackv-opt report experiments/MACHINE/collect --table readiness > table-readiness.md
mackv-opt report full-run.json --table context > table-context.md
mackv-opt report full-run.json --table performance > table-performance.md
mackv-opt report full-run.json --table memory > table-memory.md
mackv-opt report full-run.json --table quality > table-quality.md
mackv-opt report full-run.json --table stability > table-stability.md
mackv-opt compare default=default.json manual-num-ctx=manual-num-ctx.json mackv-opt=full-run.json \
  --baseline-label default \
  --format markdown > table-compare.md
mackv-opt rq1-summary experiments/MACHINE --output table-rq1.md
mackv-opt report full-run.json \
  --output-dir paper-tables \
  --output-prefix m2-16gb-llama3
mackv-opt plot-memory full-run.json --output paper-tables/memory-series.svg
```

The `--output-dir` form writes `PREFIX-readiness-compact.md`,
`PREFIX-readiness.md`, `PREFIX-context.md`, `PREFIX-performance.md`,
`PREFIX-memory.md`, `PREFIX-quality.md`, and `PREFIX-stability.md`. Add
`--format csv` to generate CSV files, or
`--tables readiness-compact,readiness,context,memory,stability` to generate
only selected tables.
Use `plot-memory` on JSON captured with `--include-memory-series` to create an
SVG memory curve for paper drafts and debugging.

## Minimum Acceptance For A First Paper Draft

- At least one 16GB, one 32GB, and one 64GB Apple Silicon machine.
- At least three models.
- At least four context targets.
- At least three repeated runs per cell for main performance tables.
- `collect/manifest.json`, `doctor.json`, `machine-profile.json`, and
  `runtime-capabilities.json` for every machine directory.
- `collect-audit.json` and `collect-audit.md` with passing status for every
  executed machine/model matrix, including Apple Silicon verification.
- At least one compare table per model that includes default Ollama, manual
  `num_ctx`, and MacKV-Opt artifacts.
- Full logs and scripts committed with anonymized local paths.

## CI Boundary

The repository CI validates Python packaging, unit tests, and CLI smoke commands
on Ubuntu and macOS. CI intentionally does not execute Ollama model inference;
real performance, memory, and quality claims require the Mac hardware runs
described above.
