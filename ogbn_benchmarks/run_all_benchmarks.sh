#!/bin/bash
# Run all OGBN benchmark models one by one.
# If any model fails, log the error and continue to the next.

cd "$(dirname "$0")"

MISSINGNESS=${1:-MCAR}
MISS_RATE=${2:-0.4}
LOG_FILE="results/run_all_${MISSINGNESS}_${MISS_RATE}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p results

echo "=================================================" | tee -a "$LOG_FILE"
echo "OGBN Benchmark Run" | tee -a "$LOG_FILE"
echo "Missingness: $MISSINGNESS @ ${MISS_RATE}" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "=================================================" | tee -a "$LOG_FILE"

MODELS=("NeighAggre" "KNN" "GraphSAGE" "FP" "PaGCN" "MATE")
FAILED=()
PASSED=()

for MODEL in "${MODELS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "-------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Running: $MODEL  [$(date +%H:%M:%S)]" | tee -a "$LOG_FILE"
    echo "-------------------------------------------------" | tee -a "$LOG_FILE"

    python main_benchmark_ogbn.py \
        --model "$MODEL" \
        --missingness "$MISSINGNESS" \
        --miss_rate "$MISS_RATE" \
        2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ $MODEL finished successfully" | tee -a "$LOG_FILE"
        PASSED+=("$MODEL")
    else
        echo "✗ $MODEL FAILED (exit code $EXIT_CODE) — continuing..." | tee -a "$LOG_FILE"
        FAILED+=("$MODEL")
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "=================================================" | tee -a "$LOG_FILE"
echo "SUMMARY  [$(date)]" | tee -a "$LOG_FILE"
echo "Passed: ${PASSED[*]}" | tee -a "$LOG_FILE"
echo "Failed: ${FAILED[*]:-none}" | tee -a "$LOG_FILE"
echo "=================================================" | tee -a "$LOG_FILE"
