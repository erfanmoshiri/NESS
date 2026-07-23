#!/bin/bash
# E7 — Second-view ablation (RQ4). NESS only, ogbn-arxiv, MCAR @ 60%.
# One run per view-creation method (view1 always edge-mask; view2 varies).
# SSL objectives fixed to hist + path; contrastive ON (default) throughout.
# Tests whether better views improve the contrastive/overall signal.
# Failure-tolerant; reports F1 + accuracy per view.
#
# Run from ogbn_benchmarks/:
#   bash experiments/e7_view_ablation.sh
#   bash experiments/e7_view_ablation.sh 0.6 ogbn-arxiv   # override rate / dataset

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

MISS_RATE=${1:-0.6}
DATASET=${2:-ogbn-arxiv}
MISSINGNESS=MCAR
EPOCHS=400
PATIENCE=20
NUM_PARTS=20

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e7_viewablation_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

VIEWS=("edge_mask" "dropout" "feat_mask" "ppr" "prefill_contrast" "deep")

echo "=================================================" | tee -a "$SUMMARY"
echo "E7 View Ablation | NESS | $DATASET | $MISSINGNESS @ $MISS_RATE" | tee -a "$SUMMARY"
echo "SSL = hist path (contrastive ON) | epochs=$EPOCHS patience=$PATIENCE" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

run() {
    local V=$1
    echo "  running: view2=$V ..." | tee -a "$SUMMARY"
    python main_benchmark_ogbn.py --model NESS --dataset "$DATASET" \
        --missingness "$MISSINGNESS" --miss_rate "$MISS_RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device gpu \
        --ssl_objective hist path --view2 "$V" \
        > "$LOGROOT/view_${V}.log" 2>&1 \
        && echo "    $V ok" || echo "    $V FAILED (exit $?) — continuing"
}

for V in "${VIEWS[@]}"; do
    run "$V"
done

# ---- Summary ----
echo "" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
echo "E7 RESULTS | NESS | $DATASET | $MISSINGNESS @ $MISS_RATE | SSL=hist+path" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
printf "%-18s %-10s %-10s %-10s\n" "VIEW2" "VAL_F1" "TEST_F1" "TEST_ACC" | tee -a "$SUMMARY"
for V in "${VIEWS[@]}"; do
    LOG="$LOGROOT/view_${V}.log"
    VAL=$(grep -oE "Val  Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    F1=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    ACC=$(grep -oE "Test Accuracy: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    printf "%-18s %-10s %-10s %-10s\n" "$V" "${VAL:-FAIL}" "${F1:-FAIL}" "${ACC:-FAIL}" | tee -a "$SUMMARY"
done
echo "=================================================" | tee -a "$SUMMARY"
echo "Finished: $(date)" | tee -a "$SUMMARY"
echo "Logs + summary in: $LOGROOT"
