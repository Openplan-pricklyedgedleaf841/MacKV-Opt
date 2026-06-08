# MacKV-Opt Paper Plan

Date baseline: 2026-06-07.

## Working Title

MacKV-Opt: Model-Independent KV Cache Budget Planning for Local LLM Inference on
Apple Silicon Unified Memory

## Thesis

Local Mac LLM users increasingly run Ollama or llama.cpp models with long
context windows, but context length is usually chosen manually or with coarse
defaults. Because KV cache memory grows linearly with context and interacts with
Apple Silicon unified memory pressure, users often discover limits only through
crashes, swap storms, or severe throughput drops.

MacKV-Opt argues that a model-independent planning layer can make existing
runtimes more usable without changing the selected model. The planner estimates
model memory, KV cache memory, runtime overhead, and memory safety margins, then
chooses context and KV cache strategies that maximize stable context under a
fixed memory budget.

## Contributions

1. A practical Apple Silicon unified-memory budget model for local LLM inference.
2. A model-independent strategy planner that maps user intent into concrete
   Ollama and llama.cpp settings.
3. An open-source CLI that preserves user model choice and does not require
   model re-quantization.
4. A reproducible benchmark harness for maximum stable context, latency,
   throughput, memory pressure, swap, repeated-run variance, and quality
   regression.
5. A stability-labeling and reporting layer that turns raw Ollama runs into
   maximum stable context tables for paper artifacts.
6. An empirical study across Apple Silicon memory tiers and current local LLM
   runtimes.

## Research Questions

RQ1. Under the same memory budget, how much does MacKV-Opt increase maximum
stable context compared with default Ollama settings?

RQ2. What is the throughput and first-token-latency cost of choosing compact KV
cache types?

RQ3. At what memory pressure thresholds do long-context Ollama runs become
unstable on 16GB, 32GB, 64GB, and 128GB Apple Silicon machines?

RQ4. How often can a model-independent planner match or improve hand-tuned
settings?

RQ5. What quality degradation appears under KV cache precision changes on
retrieval-style long-context tasks?

## Baselines

- Ollama default run settings.
- Ollama with manually chosen `num_ctx`.
- llama.cpp with `f16` KV cache.
- llama.cpp with hand-picked `q8_0`, `q5_1`, `q4_1`, or `q4_0` KV cache.
- MacKV-Opt automatic plan.

## Metrics

- Maximum stable context.
- First-token latency.
- Output tokens per second.
- Output tokens per second mean and sample standard deviation across repeats.
- Peak resident memory.
- macOS memory pressure state.
- Swap bytes and swap growth during generation.
- Generation failure rate.
- Per-cell success rate.
- Needle-in-a-Haystack retrieval accuracy.
- LongBench-style retrieval and summarization quality.
- Perplexity or log-probability delta when available.

## Experimental Matrix

Hardware:

- M1/M2/M3/M4 class machines.
- 16GB, 32GB, 64GB, and 128GB unified memory tiers.

Models:

- 7B/8B class: Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B.
- 13B/14B class where practical.
- 30B+ class on larger memory tiers.

Contexts:

- 8k, 16k, 32k, 64k, 128k, and model-specific maxima.

Prompt suites:

- Short generation sanity prompts.
- Long document summarization.
- Needle-in-a-Haystack retrieval.
- LongBench-derived retrieval and QA tasks.

## System Design

MacKV-Opt is a sidecar rather than a model runtime. It has these layers:

1. Profiler: hardware, memory, pressure, Ollama presence, model metadata.
2. Capability probe: Ollama and llama.cpp command/version/help-surface
   detection for runtime-specific experiment manifests.
3. Doctor/preflight: read-only readiness checks for Apple Silicon hardware,
   Ollama models, llama.cpp cache flags, memory pressure, swap sampling, and
   paper artifact completeness, including macOS version, power, and thermal
   state when available.
4. Collector: one-command preflight artifact capture for doctor/profile/runtime
   capabilities plus raw and normalized Ollama model metadata with
   KV-budget-field audit and optional manual metadata overrides for missing
   Ollama fields.
5. Audit gate: manifest-level readiness checks that fail executable experiment
   runs when doctor checks, artifact references, model availability, or
   KV-budget-critical metadata are insufficient, with an Apple Silicon hardware
   requirement for final paper runs.
6. Planner: KV memory estimator, memory budget search, context fallback, warning
   generation.
7. Runner: command generation for Ollama and llama.cpp-compatible args.
8. Reporter: reproducible JSON/CSV/Markdown logs for experiments plus fixed
   readiness tables from collection, audit, doctor, profile, and capability
   artifacts.
9. Benchmark executor: local Ollama API runs that collect token timing metrics
   plus sampled process RSS, optional memory time-series, and Mac memory
   pressure signals, including `vm_stat` pageout/swapout deltas on macOS.
10. Quality executor: Needle-in-a-Haystack retrieval and LongBench-style QA
   checks for quick long-context sanity testing.
11. Experiment orchestrator: one command to generate planner, performance, and
   quality artifacts for each model/context/memory-budget cell.
12. Repeat summarizer: per-cell aggregate statistics, success rates, and quality
   accuracy for paper tables.
13. Stability analyzer: per-run `stable`/`stability_reason` labels from status,
    memory pressure, swap growth, and optional throughput thresholds, plus
    per-model maximum stable context summaries with explicit repeat policy
    (`any`, `all`, or stable-run fraction).
14. Comparator: artifact-level baseline comparison for default Ollama, manual
    context settings, llama.cpp cache-type runs, and MacKV-Opt automatic plans,
    including context/throughput/latency/memory ratios and quality deltas
    relative to a named baseline.
15. Baseline template generator: one command to pre-create `default`,
    `manual-num-ctx`, and `mackv-opt` artifact directories with manifests and
    runnable scripts for reproducible paper comparisons.

The MVP does not implement new attention kernels. This is intentional: the first
paper contribution is the systems layer and user-facing reproducibility. A later
paper can add Metal fused compressed attention or algorithm-specific KV
compression.

## Related Work Map

Runtime and local deployment:

- llama.cpp: local GGUF runtime exposing context, offload, and KV cache options.
- Ollama: user-facing local model manager with real-world Mac memory pain
  points.
- MLX-LM and vllm-mlx: Apple-focused inference ecosystems.
- Open-TQ-Metal: Apple Silicon and Metal-oriented quantization/inference work.

KV cache compression and long context:

- KIVI and KVQuant: KV cache quantization.
- ChunkKV and StreamingLLM: long-context and streaming KV management.
- KVarN, RedKnot, MomentKV: recent KV/cache optimization directions.
- NVIDIA kvpress: open-source reference for KV cache compression tooling.

## Paper Claims To Validate

The paper should not claim speedups or context improvements until measured on
real Macs. The claims that are valid after the current MVP are:

- MacKV-Opt preserves model choice and generates concrete runtime settings.
- The planner can estimate KV cache budgets from model metadata.
- The capability probe can record Ollama and llama.cpp command availability,
  versions where exposed, and support for relevant context/KV flags.
- The doctor command can emit a single read-only preflight artifact that records
  hardware suitability, Ollama/model availability, llama.cpp KV flag support,
  memory-pressure sampler status, macOS power/thermal environment, and
  paper-readiness warnings.
- The collect command can capture a preflight artifact bundle with raw Ollama
  model metadata, normalized planner profiles, and explicit audits for
  KV-budget-critical missing fields, while recording any manual metadata
  overrides used to fill incomplete Ollama metadata.
- The audit command can turn a collection manifest into a pass/warn/fail
  readiness gate, and the macOS matrix runner can refuse executable runs when
  that gate fails, including when Apple Silicon hardware is required but not
  verified.
- The planner can embed runtime advice that separates memory-budget feasibility
  from locally verified command/flag support.
- The CLI can create reproducible benchmark matrices and collect Ollama API
  timing metrics into JSON experiment logs.
- The CLI can repeat benchmark and quality cells and emit aggregate means,
  sample standard deviations, success rates, and accuracy summaries.
- The CLI can write JSON, CSV, and Markdown experiment artifacts and capture
  best-effort Ollama process RSS as an early peak-memory proxy.
- On macOS, executable benchmark artifacts can include `vm_stat` page-in,
  pageout, pageout-byte, swap-in, and swapout deltas for stronger memory
  pressure evidence.
- The CLI can run synthetic Needle-in-a-Haystack checks and report exact-key
  retrieval accuracy as `quality_score`.
- The CLI can run synthetic or JSONL-backed LongBench-style QA checks and
  report exact-answer containment as `quality_score`.
- The CLI can orchestrate plan, performance, and quality runs into one nested
  experiment artifact for paper reproduction.
- The reporter can render fixed context, performance, memory, quality, and
  stability paper table views from the same artifact.
- The reporter can render a fixed readiness table from collection directories
  or multiple preflight artifacts, summarizing hardware, runtime capability,
  doctor/audit status, artifact presence, and model metadata completeness.
- The reporter can render a compact one-row paper-readiness table with an
  explicit `paper_ready` field for quick experiment gating.
- The reporter can write all fixed paper-table views in one command for a given
  machine/model experiment artifact.
- The CLI can assign heuristic stability labels to executed benchmark runs and
  summarize maximum stable context by model under an explicit repeated-run
  policy.
- The reporter can include maximum stable context summary rows in the context
  paper table.
- The reporter can render per-context stability rows with stable fraction,
  repeated-run policy, and instability reason distribution.
- The comparator can render multiple experiment artifacts into one baseline
  table for maximum stable context, throughput, latency, memory, quality, and
  stability policy, with relative improvement columns against a selected
  baseline.
- The CLI can generate a baseline artifact template with default Ollama,
  manual `num_ctx`, and MacKV-Opt comparison directories before real Mac runs.

Claims that require experiments:

- Maximum stable context improvement.
- Throughput improvement or degradation bounds.
- Memory pressure avoidance.
- Quality preservation under compact KV cache choices.

## Artifact Checklist

- Source code with CLI and tests.
- Version-pinned benchmark logs.
- Collection manifest JSON/Markdown for each hardware tier.
- Collection audit JSON/Markdown with pass/warn/fail checks and Apple Silicon
  verification for each executed matrix.
- Doctor/preflight JSON for each hardware tier.
- Machine profile JSON for each hardware tier.
- Runtime capability JSON for each machine/runtime setup.
- Model metadata JSON for each model.
- Metadata audit status for model size, hidden size, layers, attention heads,
  KV heads, and maximum context.
- Any model metadata override JSON files used for collection.
- Markdown/CSV readiness tables generated from collection and audit artifacts.
- Compact readiness tables with `paper_ready`, macOS build, power, thermal
  state, runtime support, and failed/warning checks.
- Per-run JSON metrics.
- Per-run macOS `vm_stat` pageout, pageout-byte, swap-in, and swapout deltas
  where available.
- Per-cell repeat summaries with mean, variance proxy, success rate, and
  quality accuracy.
- Per-run stability labels, stable-context policy metadata, and per-model
  maximum stable context summaries.
- Markdown/CSV tables for context, performance, memory, quality, and stability
  paper figures.
- Baseline comparison tables linking default, manual, llama.cpp, and MacKV-Opt
  artifacts.
- `baseline-template-manifest.json` plus per-model `default/`,
  `manual-num-ctx/`, and `mackv-opt/` template directories.
- Reproduction instructions for Ollama and llama.cpp.
