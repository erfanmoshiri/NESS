#!/bin/bash
# Parallel validation sweep: run ALL models on a dataset concurrently.
# Purpose: confirm no model yields F1≈0 due to logic/code errors on a bigger graph.
#
# Usage:
#   bash run_validation.sh            # default: amac
#   bash run_validation.sh amap       # a specific dataset
#   bash run_validation.sh amac 0.4   # dataset + miss_rate

cd "$(dirname "$0")"

DATASET=${1:-amac}
MISS_RATE=${2:-0.4}
MISSINGNESS=MCAR
EPOCHS=200
PATIENCE=20

LOGDIR="results/validation_${DATASET}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

MODELS=("NeighAggre" "KNN" "GraphSAGE" "FP" "PaGCN" "MATE" "NESS")

echo "================================================="
# CPU-only (non-parametric + LogisticRegression): safe to run in parallel.
# GPU models: run sequentially to avoid OOM on a shared/congested GPU.
CPU_MODELS=("NeighAggre" "KNN")
GPU_MODELS=("GraphSAGE" "FP" "PaGCN" "MATE" "NESS")

echo "Validation | dataset=$DATASET @ ${MISS_RATE}"
echo "CPU models parallel: ${CPU_MODELS[*]}"
echo "GPU models sequential: ${GPU_MODELS[*]}"
echo "Logs: $LOGDIR"
echo "================================================="

run_model() {
    local M=$1
    python main_benchmark_ogbn.py \
        --model "$M" \
        --dataset "$DATASET" \
        --missingness "$MISSINGNESS" \
        --miss_rate "$MISS_RATE" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        > "$LOGDIR/${M}.log" 2>&1
}

# Launch CPU models in parallel (they don't touch the GPU)
declare -A PIDS
for M in "${CPU_MODELS[@]}"; do
    run_model "$M" &
    PIDS[$M]=$!
    echo "  launched (CPU) $M (pid ${PIDS[$M]})"
done

# Run GPU models one at a time
for M in "${GPU_MODELS[@]}"; do
    echo "  running (GPU) $M ..."
    if run_model "$M"; then
        echo "    $M: ok"
    else
        echo "    $M: FAILED (exit $?)"
    fi
done

# Wait for the CPU models to finish
echo ""
echo "Waiting for CPU models to finish..."
for M in "${CPU_MODELS[@]}"; do
    if wait "${PIDS[$M]}"; then
        echo "  $M: ok"
    else
        echo "  $M: FAILED (exit $?)"
    fi
done

# ---- Summary: grep final F1 from each model's log ----
echo ""
echo "================================================="
echo "RESULTS SUMMARY | dataset=$DATASET"
echo "================================================="
printf "%-12s %-12s %-12s\n" "MODEL" "VAL_F1" "TEST_F1"
for M in "${MODELS[@]}"; do
    LOG="$LOGDIR/${M}.log"
    VAL=$(grep -oE "Val  Macro-F1: [0-9.]+" "$LOG" | grep -oE "[0-9.]+" | tail -1)
    TEST=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOG" | grep -oE "[0-9.]+" | tail -1)
    if [ -z "$TEST" ]; then
        ERR=$(grep -E "Error|Traceback|Exception" "$LOG" | head -1)
        printf "%-12s %-12s %-12s <- CHECK LOG: %s\n" "$M" "${VAL:-N/A}" "${TEST:-N/A}" "$ERR"
    else
        # Flag suspiciously low F1
        FLAG=""
        awk_check=$(awk -v t="$TEST" 'BEGIN{print (t < 0.05) ? "1" : "0"}')
        [ "$awk_check" = "1" ] && FLAG="  <- SUSPICIOUS (near 0)"
        printf "%-12s %-12s %-12s%s\n" "$M" "$VAL" "$TEST" "$FLAG"
    fi
done
echo "================================================="
echo "Full logs in: $LOGDIR"
