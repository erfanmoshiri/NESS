#!/bin/bash
# Prefill ablation: how to fill missing nodes for the main representation.
# Compare fp (Feature Propagation, default) vs zero (no fill) vs mean (observed-
# neighbor mean). Second view stays ppr throughout, so this isolates the prefill
# of view1. Locked config, arxiv MCAR @ 80%, 800 epochs.
#
# Run from ogbn_benchmarks/:  bash experiments/e_prefill.sh

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

DATASET=ogbn-arxiv
MISS_RATE=0.8
MISSINGNESS=MCAR
EPOCHS=800
PATIENCE=20
NUM_PARTS=20
PREFILLS=("fp" "zero" "mean")

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e_prefill_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

echo "=================================================" | tee -a "$SUMMARY"
echo "Prefill ablation | NESS | $DATASET | $MISSINGNESS @ $MISS_RATE | view2=ppr, 800 ep" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

for P in "${PREFILLS[@]}"; do
    echo "  running: prefill=$P ..." | tee -a "$SUMMARY"
    python main_benchmark_ogbn.py --model NESS --dataset "$DATASET" \
        --missingness "$MISSINGNESS" --miss_rate "$MISS_RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device gpu \
        --prefill "$P" --ssl_objective hist path \
        > "$LOGROOT/prefill_${P}.log" 2>&1 \
        && echo "    $P ok" || echo "    $P FAILED (exit $?) — continuing"
done

echo "" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
echo "PREFILL RESULTS | $DATASET @ $MISS_RATE" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
printf "%-10s %-10s %-10s %-10s\n" "PREFILL" "VAL_F1" "TEST_F1" "TEST_ACC" | tee -a "$SUMMARY"
for P in "${PREFILLS[@]}"; do
    LOG="$LOGROOT/prefill_${P}.log"
    VAL=$(grep -oE "Val  Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    F1=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    ACC=$(grep -oE "Test Accuracy: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    printf "%-10s %-10s %-10s %-10s\n" "$P" "${VAL:-FAIL}" "${F1:-FAIL}" "${ACC:-FAIL}" | tee -a "$SUMMARY"
done
echo "=================================================" | tee -a "$SUMMARY"
echo "Finished: $(date)" | tee -a "$SUMMARY"
echo "Logs + summary in: $LOGROOT"
