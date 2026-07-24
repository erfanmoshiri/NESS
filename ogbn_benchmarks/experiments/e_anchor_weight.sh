#!/bin/bash
# Does anchorcls SSL actually help? Isolate anchorcls (no hist/path) and sweep its weight.
# E11 showed SSL only helped when UPWEIGHTED (w_ssl 1->5 gave +0.037). Locked config
# uses w_cls=3, which may drown SSL at w_anchor=1 — so sweep w_anchor up.
#
# Compare: no_anchor (no SSL at all) vs anchorcls @ {1, 3, 5, 8} (weight = --w_anchor).
# If anchor helps, higher weights should beat no_anchor. Locked base:
# SAGE-3layer-128, edge_mask, w_cls=3, dropout 0.3, arxiv MCAR @ 80%, 800 epochs.
#
# Run from ogbn_benchmarks/:
#   bash experiments/e_anchor_weight.sh
#   bash experiments/e_anchor_weight.sh 0.8 ogbn-arxiv

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

MISS_RATE=${1:-0.8}
DATASET=${2:-ogbn-arxiv}
MISSINGNESS=MCAR
EPOCHS=800
PATIENCE=20
NUM_PARTS=20

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e_anchorw_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

echo "=================================================" | tee -a "$SUMMARY"
echo "Anchor-weight sweep | NESS | $DATASET | $MISSINGNESS @ $MISS_RATE" | tee -a "$SUMMARY"
echo "base: SAGE-3L-128, edge_mask, w_cls=3, dropout 0.3, 800 epochs" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

run() {
    local LABEL=$1; shift
    echo "  running: $LABEL ..." | tee -a "$SUMMARY"
    python main_benchmark_ogbn.py --model NESS --dataset "$DATASET" \
        --missingness "$MISSINGNESS" --miss_rate "$MISS_RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device gpu \
        --view2 edge_mask \
        "$@" > "$LOGROOT/${LABEL}.log" 2>&1 \
        && echo "    $LABEL ok" || echo "    $LABEL FAILED (exit $?) — continuing"
}

LABELS=("no_anchor" "anchorcls_w1" "anchorcls_w3" "anchorcls_w5" "anchorcls_w8")

# baseline: no SSL objective at all (contrastive still on per locked config)
run "no_anchor"  --ssl_objective
# anchorcls (hop-bucket classification) only, sweeping its weight (uses w_anchor)
run "anchorcls_w1"  --ssl_objective anchorcls --w_anchor 1
run "anchorcls_w3"  --ssl_objective anchorcls --w_anchor 3
run "anchorcls_w5"  --ssl_objective anchorcls --w_anchor 5
run "anchorcls_w8"  --ssl_objective anchorcls --w_anchor 8

# ---- Summary ----
echo "" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
echo "RESULTS | anchor weight sweep | $DATASET @ $MISS_RATE (base=no_anchor)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
printf "%-12s %-10s %-10s %-10s\n" "CONFIG" "VAL_F1" "TEST_F1" "TEST_ACC" | tee -a "$SUMMARY"
for LABEL in "${LABELS[@]}"; do
    LOG="$LOGROOT/${LABEL}.log"
    VAL=$(grep -oE "Val  Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    F1=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    ACC=$(grep -oE "Test Accuracy: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    printf "%-12s %-10s %-10s %-10s\n" "$LABEL" "${VAL:-FAIL}" "${F1:-FAIL}" "${ACC:-FAIL}" | tee -a "$SUMMARY"
done
echo "=================================================" | tee -a "$SUMMARY"
echo "Finished: $(date)" | tee -a "$SUMMARY"
echo "Logs + summary in: $LOGROOT"
