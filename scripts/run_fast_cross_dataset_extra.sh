#!/bin/bash
# Fast cross-dataset runner for additional requested jobs:
#   stress_binary: DeepCORAL, MCC, MCD, SHOT, CBST
#   valence:       TabTransformer, DCN, SHOT, CBST

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

DA_MODELS=("DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM")

# (model, label) pairs
JOBS=(
    "DeepCORAL:stress_binary"
    "MCC:stress_binary"
    "MCD:stress_binary"
    "SHOT:stress_binary"
    "CBST:stress_binary"
    "TabTransformer:valence"
    "DCN:valence"
    "SHOT:valence"
    "CBST:valence"
)

is_da() {
    local needle="$1"
    local item
    for item in "${DA_MODELS[@]}"; do
        [ "$needle" = "$item" ] && return 0
    done
    return 1
}

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

    if is_da "$model"; then
        cmd+=(--uda)
    fi
    case "$model" in
        SAINT|TabTransformer|FTTransformer) cmd+=(--efficient_attention) ;;
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
echo "All extra jobs done in ${ELAPSED}s. Results under ${OUTPUT_DIR}/, logs under ${LOG_DIR}/"
