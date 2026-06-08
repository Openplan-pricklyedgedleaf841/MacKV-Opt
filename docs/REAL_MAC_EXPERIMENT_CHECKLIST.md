# Real Mac experiment checklist

This checklist is for paper runs on 16GB, 32GB, and 64GB Apple Silicon Macs.
It keeps the selected Ollama model unchanged and records enough evidence to
compare default Ollama, manual `num_ctx`, and MacKV-Opt.

## 1. Prepare each Mac

- Install Python 3.10 or newer.
- Install Ollama and start the Ollama service.
- Pull every model in the matrix:

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b
```

- Install MacKV-Opt from the repository:

```bash
python -m pip install -e ".[dev]"
mackv-opt --help
```

- Plug the Mac into AC power.
- Disable Low Power Mode.
- Close heavy background apps for best-case runs.
- Let the machine cool down before the first executable run.
- Keep a short note of background apps and room/thermal conditions.

## 2. Collect readiness evidence

Run this once per Mac:

```bash
MACKV_MACHINE=m2-16gb \
MACKV_MODELS=llama3.1:8b,qwen2.5:7b,mistral:7b \
MACKV_CONTEXTS=8k,16k,32k \
MACKV_MEMORY_BUDGET=12GiB \
./scripts/run_macos_matrix.sh
```

Inspect:

- `experiments/MACHINE/collect/manifest.json`
- `experiments/MACHINE/collect-audit.json`
- `experiments/MACHINE/paper-tables/readiness-readiness-compact.md`
- `experiments/MACHINE/paper-tables/readiness-readiness.md`

If the audit reports missing KV metadata, create an override file and rerun:

```bash
MACKV_MODEL_METADATA_OVERRIDES=model-overrides.json ./scripts/run_macos_matrix.sh
```

Do not run executable experiments until the audit is acceptable for the paper.

## 3. Generate baseline directories

The matrix script creates these directories by default:

```text
experiments/MACHINE/MODEL/
  default/
  manual-num-ctx/
  mackv-opt/
  paper-tables/
```

You can generate them explicitly:

```bash
mackv-opt baseline-template \
  --output-dir experiments/MACHINE \
  --models llama3.1:8b,qwen2.5:7b,mistral:7b \
  --contexts 8k,16k,32k,64k \
  --memory-budget 20GiB \
  --json
```

Run each `run.sh` from its own directory when collecting direct baseline
artifacts.

## 4. 16GB execute preset

Use 16GB machines for the first failure-boundary study:

```bash
MACKV_MACHINE=m2-16gb \
MACKV_MODELS=llama3.1:8b,qwen2.5:7b,mistral:7b \
MACKV_MEMORY_BUDGET=12GiB \
MACKV_CONTEXTS=8k,16k,32k \
MACKV_REPEATS=3 \
MACKV_STABLE_POLICY=all \
MACKV_INCLUDE_SERIES=1 \
MACKV_EXECUTE=1 \
./scripts/run_macos_matrix.sh
```

Expected outputs:

- `experiments/m2-16gb/MODEL/mackv-opt/full-run.json`
- `experiments/m2-16gb/MODEL/paper-tables/MODEL-compare.md`
- `experiments/m2-16gb/matrix-compare.md`

## 5. 32GB execute preset

Use 32GB machines for 64k context measurements:

```bash
MACKV_MACHINE=m3-32gb \
MACKV_MODELS=llama3.1:8b,qwen2.5:7b,mistral:7b \
MACKV_MEMORY_BUDGET=24GiB \
MACKV_CONTEXTS=8k,16k,32k,64k \
MACKV_REPEATS=3 \
MACKV_STABLE_POLICY=all \
MACKV_INCLUDE_SERIES=1 \
MACKV_EXECUTE=1 \
./scripts/run_macos_matrix.sh
```

## 6. 64GB execute preset

Use 64GB machines for 128k context scaling:

```bash
MACKV_MACHINE=m4-64gb \
MACKV_MODELS=llama3.1:8b,qwen2.5:7b,mistral:7b \
MACKV_MEMORY_BUDGET=48GiB \
MACKV_CONTEXTS=8k,16k,32k,64k,128k \
MACKV_REPEATS=3 \
MACKV_STABLE_POLICY=all \
MACKV_INCLUDE_SERIES=1 \
MACKV_EXECUTE=1 \
./scripts/run_macos_matrix.sh
```

## 7. Required artifacts per machine

- `collect/manifest.json`
- `collect/manifest.md`
- `collect-audit.json`
- `collect-audit.md`
- `doctor.json`
- `machine-profile.json`
- `runtime-capabilities.json`
- per-model `default/full-run.json`
- per-model `manual-num-ctx/full-run.json`
- per-model `mackv-opt/full-run.json`
- per-model compare table
- machine-level `matrix-compare.md`
- memory SVGs for runs captured with `--include-memory-series`

## 8. Minimum run notes

Record these in a lab notebook or machine README:

- Mac model and chip.
- Unified memory size.
- macOS version.
- Ollama version.
- llama.cpp version if used.
- Power source and power mode.
- Thermal state before the run.
- Background apps.
- Model tags and model file hashes if available.
- Any metadata override file used.

## 9. RQ1 table generation

After each model has all three baseline artifacts:

```bash
mackv-opt rq1-summary experiments/m2-16gb \
  --output experiments/m2-16gb/rq1-summary.md
```

Use the Markdown output in the paper draft and keep the JSON/CSV versions as
supplementary artifacts.
