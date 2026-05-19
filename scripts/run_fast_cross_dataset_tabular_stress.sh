#!/bin/bash
# Fast cross-dataset runner for all tabular models on stress_binary.
# FTTransformer is skipped because it was already run by run_fast_cross_dataset.sh.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR="${OUTPUT_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs/fast_cross_dataset}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

HPO_TRIALS="${HPO_TRIALS:-5}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-20}"
PATIENCE="${PATIENCE:-5}"
SEED="${SEED:-42}"
RUN_SETTING="${RUN_SETTING:-all}"
BACKBONE="${BACKBONE:-MLP}"
LABEL="stress_binary"

# All tabular models. AutoInt excluded (matches run_cross_dataset.sh).
# FTTransformer already produced results in the previous run.
MODELS=(
    "XGB"
    "LGB"
    "MLP"
    "ResNet"
    "TabNet"
    "SAINT"
    "TabTransformer"
    "DCN"
)

START_TS=$(date +%s)

for model in "${MODELS[@]}"; do
    output_csv="${OUTPUT_DIR}/cross_dataset_fast_${LABEL}_${model}_${BACKBONE}_${RUN_SETTING}.csv"
    log_file="${LOG_DIR}/${LABEL}_${model}.log"

    cmd=(
        "$PYTHON_BIN" execute_cross_dataset.py
        --label "$LABEL"
        --mode run
        --run_setting "$RUN_SETTING"
        --model "$model"
        --backbone "$BACKBONE"
        --hpo_trials "$HPO_TRIALS"
        --hpo_mode single_split
        --epochs_override "$EPOCHS_OVERRIDE"
        --patience "$PATIENCE"
        --seeds "$SEED"
        --output "$output_csv"
    )

    case "$model" in
        SAINT|TabTransformer|FTTransformer) cmd+=(--efficient_attention) ;;
    esac

    echo "========================================="
    echo "[RUN] label=$LABEL model=$model | hpo=$HPO_TRIALS epochs=$EPOCHS_OVERRIDE patience=$PATIENCE"
    echo "Output: $output_csv"
    echo "Log:    $log_file"
    echo "========================================="
    "${cmd[@]}" 2>&1 | tee "$log_file"
done

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
echo ""
echo "All tabular jobs done in ${ELAPSED}s. Results under ${OUTPUT_DIR}/, logs under ${LOG_DIR}/"
