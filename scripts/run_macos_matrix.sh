#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a reproducible MacKV-Opt experiment matrix on Apple Silicon Macs.

Default mode is dry-run. Set MACKV_EXECUTE=1 to call the local Ollama API.

Environment variables:
  MACKV_MODELS              Comma-separated model names.
                            Default: llama3.1:8b,qwen2.5:7b
  MACKV_CONTEXTS            Comma-separated contexts.
                            Default: 8k,16k,32k,64k
  MACKV_MEMORY_BUDGET       Memory budget passed to mackv-opt experiment.
                            Default: 12GiB
  MACKV_MODEL_METADATA_OVERRIDES
                            Optional JSON file passed to collect for manual
                            model metadata overrides.
  MACKV_MACHINE             Machine label for output path.
                            Default: current hostname
  MACKV_OUTPUT_ROOT         Root output directory.
                            Default: experiments
  MACKV_DEPTHS              Needle depths.
                            Default: 10%,50%,90%
  MACKV_TIMEOUT             Per-run timeout seconds.
                            Default: 300
  MACKV_BENCH_NUM_PREDICT   Benchmark generated token limit.
                            Default: 128
  MACKV_NEEDLE_NUM_PREDICT  Needle generated token limit.
                            Default: 64
  MACKV_QA_NUM_PREDICT      QA generated token limit.
                            Default: 96
  MACKV_REPEATS             Repeated runs per model/context/quality cell.
                            Default: 1
  MACKV_MEMORY_INTERVAL     Memory sampling interval seconds.
                            Default: 0.5
  MACKV_MAX_SWAP            Swap growth threshold for unstable runs.
                            Default: 512MiB
  MACKV_MIN_TOKENS_PER_SEC  Optional throughput floor for unstable runs.
                            Default: unset
  MACKV_STABLE_POLICY       Context stability policy: any, all, or fraction.
                            Default: any
  MACKV_MIN_STABLE_FRACTION Required stable-run fraction for fraction policy.
                            Default: 1.0
  MACKV_COMPARE_LABELS      Optional comma-separated baseline labels to compare
                            inside each model directory. Each label must map to
                            LABEL/full-run.json. Default: default,manual-num-ctx,mackv-opt
  MACKV_COMPARE_BASELINE    Label used as the relative-improvement baseline.
                            Default: first available compare label
  MACKV_COMPARE_CURRENT     Label assigned to this script's generated full-run.
                            Default: mackv-opt
  MACKV_WRITE_BASELINE_TEMPLATES
                            Set to 1 to pre-create default/manual-num-ctx/
                            mackv-opt output directories. Default: 1
  MACKV_FAIL_ON_MISSING_METADATA
                            Set to 1 to fail audit when model KV metadata is
                            incomplete. Default: 1 for execute, 0 for dry-run
  MACKV_REQUIRE_APPLE_SILICON
                            Set to 1 to fail audit unless collection verifies
                            Apple Silicon. Default: 1 for execute, 0 for dry-run
  MACKV_INCLUDE_SERIES      Set to 1 to embed memory_series in JSON.
                            Default: 0
  MACKV_EXECUTE             Set to 1 for executable runs. Otherwise dry-run.
                            Default: 0
  MACKV_EXTRA_ARGS          Extra args appended to every experiment command.

Examples:
  ./scripts/run_macos_matrix.sh
  MACKV_EXECUTE=1 MACKV_MEMORY_BUDGET=20GiB ./scripts/run_macos_matrix.sh
  MACKV_MODELS='llama3.1:8b,qwen2.5:7b' MACKV_CONTEXTS='8k,16k,32k' ./scripts/run_macos_matrix.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v mackv-opt >/dev/null 2>&1; then
  echo "mackv-opt not found. Install with: python -m pip install -e ." >&2
  exit 127
fi

MACKV_MODELS="${MACKV_MODELS:-llama3.1:8b,qwen2.5:7b}"
MACKV_CONTEXTS="${MACKV_CONTEXTS:-8k,16k,32k,64k}"
MACKV_MEMORY_BUDGET="${MACKV_MEMORY_BUDGET:-12GiB}"
MACKV_MODEL_METADATA_OVERRIDES="${MACKV_MODEL_METADATA_OVERRIDES:-}"
MACKV_MACHINE="${MACKV_MACHINE:-$(hostname | tr ' /:' '---')}"
MACKV_OUTPUT_ROOT="${MACKV_OUTPUT_ROOT:-experiments}"
MACKV_DEPTHS="${MACKV_DEPTHS:-10%,50%,90%}"
MACKV_TIMEOUT="${MACKV_TIMEOUT:-300}"
MACKV_BENCH_NUM_PREDICT="${MACKV_BENCH_NUM_PREDICT:-128}"
MACKV_NEEDLE_NUM_PREDICT="${MACKV_NEEDLE_NUM_PREDICT:-64}"
MACKV_QA_NUM_PREDICT="${MACKV_QA_NUM_PREDICT:-96}"
MACKV_REPEATS="${MACKV_REPEATS:-1}"
MACKV_MEMORY_INTERVAL="${MACKV_MEMORY_INTERVAL:-0.5}"
MACKV_MAX_SWAP="${MACKV_MAX_SWAP:-512MiB}"
MACKV_MIN_TOKENS_PER_SEC="${MACKV_MIN_TOKENS_PER_SEC:-}"
MACKV_STABLE_POLICY="${MACKV_STABLE_POLICY:-any}"
MACKV_MIN_STABLE_FRACTION="${MACKV_MIN_STABLE_FRACTION:-1.0}"
MACKV_COMPARE_LABELS="${MACKV_COMPARE_LABELS:-default,manual-num-ctx,mackv-opt}"
MACKV_COMPARE_BASELINE="${MACKV_COMPARE_BASELINE:-}"
MACKV_COMPARE_CURRENT="${MACKV_COMPARE_CURRENT:-mackv-opt}"
MACKV_WRITE_BASELINE_TEMPLATES="${MACKV_WRITE_BASELINE_TEMPLATES:-1}"
MACKV_FAIL_ON_MISSING_METADATA="${MACKV_FAIL_ON_MISSING_METADATA:-}"
MACKV_REQUIRE_APPLE_SILICON="${MACKV_REQUIRE_APPLE_SILICON:-}"
MACKV_INCLUDE_SERIES="${MACKV_INCLUDE_SERIES:-0}"
MACKV_EXECUTE="${MACKV_EXECUTE:-0}"
MACKV_EXTRA_ARGS="${MACKV_EXTRA_ARGS:-}"

IFS=',' read -r -a MODELS <<< "$MACKV_MODELS"
IFS=',' read -r -a COMPARE_LABELS <<< "$MACKV_COMPARE_LABELS"
RUN_MODE="--dry-run"
if [[ "$MACKV_EXECUTE" == "1" ]]; then
  RUN_MODE="--execute"
fi

SERIES_ARGS=()
if [[ "$MACKV_INCLUDE_SERIES" == "1" ]]; then
  SERIES_ARGS=(--include-memory-series)
fi

STABILITY_ARGS=(
  --max-swap "$MACKV_MAX_SWAP"
  --stable-context-policy "$MACKV_STABLE_POLICY"
  --min-stable-fraction "$MACKV_MIN_STABLE_FRACTION"
)
if [[ -n "$MACKV_MIN_TOKENS_PER_SEC" ]]; then
  STABILITY_ARGS+=(--min-tokens-per-second "$MACKV_MIN_TOKENS_PER_SEC")
fi

COLLECT_ARGS=()
if [[ -n "$MACKV_MODEL_METADATA_OVERRIDES" ]]; then
  COLLECT_ARGS+=(--model-metadata-overrides "$MACKV_MODEL_METADATA_OVERRIDES")
fi

BASE_DIR="$MACKV_OUTPUT_ROOT/$MACKV_MACHINE"
mkdir -p "$BASE_DIR"

echo "MacKV-Opt matrix"
echo "  machine: $MACKV_MACHINE"
echo "  models: $MACKV_MODELS"
echo "  contexts: $MACKV_CONTEXTS"
echo "  memory budget: $MACKV_MEMORY_BUDGET"
echo "  repeats: $MACKV_REPEATS"
echo "  stable policy: $MACKV_STABLE_POLICY"
echo "  mode: $RUN_MODE"
echo "  output: $BASE_DIR"

if [[ "$MACKV_WRITE_BASELINE_TEMPLATES" == "1" ]]; then
  mackv-opt baseline-template \
    --output-dir "$BASE_DIR" \
    --models "$MACKV_MODELS" \
    --contexts "$MACKV_CONTEXTS" \
    --manual-context "$(printf '%s' "$MACKV_CONTEXTS" | awk -F',' '{print $NF}')" \
    --memory-budget "$MACKV_MEMORY_BUDGET" \
    --repeats "$MACKV_REPEATS" \
    --json \
    > "$BASE_DIR/baseline-template.stdout.json"
fi

mackv-opt collect \
  --output-dir "$BASE_DIR/collect" \
  --models "$MACKV_MODELS" \
  "${COLLECT_ARGS[@]}" \
  --json \
  > "$BASE_DIR/collect.stdout.json"
cp "$BASE_DIR/collect/doctor.json" "$BASE_DIR/doctor.json"
cp "$BASE_DIR/collect/machine-profile.json" "$BASE_DIR/machine-profile.json"
cp "$BASE_DIR/collect/runtime-capabilities.json" "$BASE_DIR/runtime-capabilities.json"

AUDIT_ARGS=()
if [[ "$MACKV_FAIL_ON_MISSING_METADATA" == "1" || ( "$MACKV_EXECUTE" == "1" && "$MACKV_FAIL_ON_MISSING_METADATA" != "0" ) ]]; then
  AUDIT_ARGS+=(--fail-on-missing-metadata)
fi
if [[ "$MACKV_REQUIRE_APPLE_SILICON" == "1" || ( "$MACKV_EXECUTE" == "1" && "$MACKV_REQUIRE_APPLE_SILICON" != "0" ) ]]; then
  AUDIT_ARGS+=(--require-apple-silicon)
fi
if mackv-opt audit "$BASE_DIR/collect/manifest.json" "${AUDIT_ARGS[@]}" \
  --output "$BASE_DIR/collect-audit.json" \
  --markdown-output "$BASE_DIR/collect-audit.md" \
  --json > "$BASE_DIR/collect-audit.stdout.json"; then
  echo "Preflight audit passed."
else
  echo "Preflight audit failed. See $BASE_DIR/collect-audit.json" >&2
  if [[ "$MACKV_EXECUTE" == "1" ]]; then
    echo "Refusing executable runs because MACKV_EXECUTE=1 requires a passing audit." >&2
    exit 2
  fi
  echo "Continuing because default matrix mode is dry-run." >&2
fi

mackv-opt report "$BASE_DIR/collect" \
  --table readiness-compact \
  --output-dir "$BASE_DIR/report-tables" \
  --output-prefix readiness
mackv-opt report "$BASE_DIR/collect" \
  --table readiness \
  --output-dir "$BASE_DIR/report-tables" \
  --output-prefix readiness

for model in "${MODELS[@]}"; do
  model="$(printf '%s' "$model" | xargs)"
  [[ -z "$model" ]] && continue
  safe_model="$(printf '%s' "$model" | tr '/: ' '---')"
  out_dir="$BASE_DIR/$safe_model"
  mkdir -p "$out_dir"

  echo "Running $model -> $out_dir"
  # shellcheck disable=SC2086
  mackv-opt experiment "$model" \
    --contexts "$MACKV_CONTEXTS" \
    --memory-budget "$MACKV_MEMORY_BUDGET" \
    "$RUN_MODE" \
    --json \
    --depths "$MACKV_DEPTHS" \
    --timeout "$MACKV_TIMEOUT" \
    --bench-num-predict "$MACKV_BENCH_NUM_PREDICT" \
    --needle-num-predict "$MACKV_NEEDLE_NUM_PREDICT" \
    --qa-num-predict "$MACKV_QA_NUM_PREDICT" \
    --repeats "$MACKV_REPEATS" \
    --memory-sample-interval "$MACKV_MEMORY_INTERVAL" \
    "${STABILITY_ARGS[@]}" \
    "${SERIES_ARGS[@]}" \
    --output-dir "$out_dir" \
    --output-prefix full-run \
    $MACKV_EXTRA_ARGS \
    > "$out_dir/full-run.stdout.json"

  mackv-opt report "$out_dir/full-run.json" \
    --output-dir "$out_dir/report-tables" \
    --output-prefix "$safe_model"

  compare_current_dir="$out_dir/$MACKV_COMPARE_CURRENT"
  mkdir -p "$compare_current_dir"
  cp "$out_dir/full-run.json" "$compare_current_dir/full-run.json"

  compare_args=()
  for label in "${COMPARE_LABELS[@]}"; do
    label="$(printf '%s' "$label" | xargs)"
    [[ -z "$label" ]] && continue
    artifact="$out_dir/$label/full-run.json"
    if [[ -f "$artifact" ]]; then
      compare_args+=("$label=$artifact")
    fi
  done
  if (( ${#compare_args[@]} >= 2 )); then
    baseline_args=()
    if [[ -n "$MACKV_COMPARE_BASELINE" ]]; then
      baseline_args=(--baseline-label "$MACKV_COMPARE_BASELINE")
    fi
    echo "Writing compare table for $model"
    mackv-opt compare "${compare_args[@]}" "${baseline_args[@]}" \
      --format markdown \
      --output "$out_dir/report-tables/$safe_model-compare.md"
    mackv-opt compare "${compare_args[@]}" "${baseline_args[@]}" \
      --format csv \
      --output "$out_dir/report-tables/$safe_model-compare.csv"
  else
    echo "Skipping compare for $model; found ${#compare_args[@]} artifact(s)."
  fi
done

matrix_compare_args=()
for model in "${MODELS[@]}"; do
  model="$(printf '%s' "$model" | xargs)"
  [[ -z "$model" ]] && continue
  safe_model="$(printf '%s' "$model" | tr '/: ' '---')"
  artifact="$BASE_DIR/$safe_model/full-run.json"
  if [[ -f "$artifact" ]]; then
    matrix_compare_args+=("$safe_model=$artifact")
  fi
done
if (( ${#matrix_compare_args[@]} >= 1 )); then
  echo "Writing matrix compare summary"
  mackv-opt compare "${matrix_compare_args[@]}" \
    --format markdown \
    --output "$BASE_DIR/matrix-compare.md"
  mackv-opt compare "${matrix_compare_args[@]}" \
    --format csv \
    --output "$BASE_DIR/matrix-compare.csv"
fi

echo "Done. Results are under $BASE_DIR"
