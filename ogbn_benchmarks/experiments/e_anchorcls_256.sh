#!/bin/bash
# Does anchorcls SSL help at HIGHER width (256)? At 128 it didn't (0.435 vs no_ssl 0.437).
# Test whether more encoder capacity gives SSL room to add signal.
# Locked base otherwise: SAGE-3L, edge_mask, w_cls=3, dropout 0.3, arxiv MCAR @ 80%, 800 ep.
#
# Run from ogbn_benchmarks/:
#   bash experiments/e_anchorcls_256.sh
#   bash experiments/e_anchorcls_256.sh 0.8 ogbn-arxiv

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

MISS_RATE=${1:-0.8}
DATASET=${2:-ogbn-arxiv}
MISSINGNESS=MCAR
EPOCHS=800
PATIENCE=20
NUM_PARTS=20
HIDDEN=256

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e_anchorcls256_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

echo "=================================================" | tee -a "$SUMMARY"
echo "anchorcls @ width 256 | NESS | $DATASET | $MISSINGNESS @ $MISS_RATE" | tee -a "$SUMMARY"
echo "base: SAGE-3L, hidden=256, edge_mask, w_cls=3, 800 ep" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

run() {
    local LABEL=$1; shift
    echo "  running: $LABEL ..." | tee -a "$SUMMARY"
    python main_benchmark_ogbn.py --model NESS --dataset "$DATASET" \
        --missingness "$MISSINGNESS" --miss_rate "$MISS_RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device gpu \
        --hidden "$HIDDEN" --view2 edge_mask \
        "$@" > "$LOGROOT/${LABEL}.log" 2>&1 \
        && echo "    $LABEL ok" || echo "    $LABEL FAILED (exit $?) — continuing"
}

LABELS=("no_ssl_256" "anchorcls_256")
run "no_ssl_256"     --ssl_objective
run "anchorcls_256"  --ssl_objective anchorcls --w_anchor 3

# ---- Summary ----
echo "" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
echo "RESULTS | anchorcls @256 | $DATASET @ $MISS_RATE" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
printf "%-16s %-10s %-10s %-10s\n" "CONFIG" "VAL_F1" "TEST_F1" "TEST_ACC" | tee -a "$SUMMARY"
for LABEL in "${LABELS[@]}"; do
    LOG="$LOGROOT/${LABEL}.log"
    VAL=$(grep -oE "Val  Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    F1=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    ACC=$(grep -oE "Test Accuracy: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    printf "%-16s %-10s %-10s %-10s\n" "$LABEL" "${VAL:-FAIL}" "${F1:-FAIL}" "${ACC:-FAIL}" | tee -a "$SUMMARY"
done
echo "=================================================" | tee -a "$SUMMARY"
echo "Finished: $(date)" | tee -a "$SUMMARY"
echo "Logs + summary in: $LOGROOT"
