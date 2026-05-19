#!/bin/bash
# Fast cross-dataset runner for CGDM (all labels), JAN/FTTransformer (disturbance, stress, valence).
# Uses aggressive low-HPO/low-epoch settings for quick turnaround.

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

# (model, label) pairs
JOBS=(
    "CGDM:arousal"
    "CGDM:disturbance"
    "CGDM:valence"
    "CGDM:stress_binary"
    "JAN:disturbance"
    "JAN:stress_binary"
    "JAN:valence"
    "FTTransformer:disturbance"
    "FTTransformer:stress_binary"
    "FTTransformer:valence"
)

START_TS=$(date +%s)

for job in "${JOBS[@]}"; do
    model="${job%%:*}"
    label="${job##*:}"
    output_csv="${OUTPUT_DIR}/cross_dataset_fast_${label}_${model}_${BACKBONE}_${RUN_SETTING}.csv"
    log_file="${LOG_DIR}/${label}_${model}.log"

    cmd=(
        "$PYTHON_BIN" execute_cross_dataset.py
        --label "$label"
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
        CGDM|JAN) cmd+=(--uda) ;;
    esac
    case "$model" in
        FTTransformer|SAINT|TabTransformer) cmd+=(--efficient_attention) ;;
    esac

    echo "========================================="
    echo "[RUN] label=$label model=$model | hpo=$HPO_TRIALS epochs=$EPOCHS_OVERRIDE patience=$PATIENCE"
    echo "Output: $output_csv"
    echo "Log:    $log_file"
    echo "========================================="
    "${cmd[@]}" 2>&1 | tee "$log_file"
done

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
echo ""
echo "All jobs done in ${ELAPSED}s. Results under ${OUTPUT_DIR}/, logs under ${LOG_DIR}/"
