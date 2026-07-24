#!/bin/bash
# E11-b — FP prefill-iterations sweep (the two rows that failed in e11_design_hp.sh
# due to the missing --fp_iterations CLI arg, now fixed). Locked base: SAGE-128,
# edge_mask, hist+path, arxiv MCAR @ 80%, 800 epochs. base (fp_iters=40) = 0.3924.
#
# Run from ogbn_benchmarks/:
#   bash experiments/e11b_fpiters.sh
#   bash experiments/e11b_fpiters.sh 0.8 ogbn-arxiv

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

MISS_RATE=${1:-0.8}
DATASET=${2:-ogbn-arxiv}
MISSINGNESS=MCAR
EPOCHS=800
PATIENCE=20
NUM_PARTS=20
ITERS=(10 20)

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e11b_fpiters_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

echo "=================================================" | tee -a "$SUMMARY"
echo "E11-b FP-iters sweep | NESS | $DATASET | $MISSINGNESS @ $MISS_RATE" | tee -a "$SUMMARY"
echo "base: SAGE-128, edge_mask, hist+path, 800 epochs | fp_iters: ${ITERS[*]} (base=40 -> 0.3924)" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

for IT in "${ITERS[@]}"; do
    echo "  running: fpiters_${IT} ..." | tee -a "$SUMMARY"
    python main_benchmark_ogbn.py --model NESS --dataset "$DATASET" \
        --missingness "$MISSINGNESS" --miss_rate "$MISS_RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device gpu \
        --ssl_objective hist path --view2 edge_mask --fp_iterations "$IT" \
        > "$LOGROOT/fpiters_${IT}.log" 2>&1 \
        && echo "    fpiters_${IT} ok" || echo "    fpiters_${IT} FAILED (exit $?)"
done

# ---- Summary ----
echo "" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
echo "E11-b RESULTS | fp_iters | $DATASET @ $MISS_RATE (base fp_iters=40 = 0.3924)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
printf "%-14s %-10s %-10s %-10s\n" "CONFIG" "VAL_F1" "TEST_F1" "TEST_ACC" | tee -a "$SUMMARY"
for IT in "${ITERS[@]}"; do
    LOG="$LOGROOT/fpiters_${IT}.log"
    VAL=$(grep -oE "Val  Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    F1=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    ACC=$(grep -oE "Test Accuracy: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    printf "%-14s %-10s %-10s %-10s\n" "fpiters_${IT}" "${VAL:-FAIL}" "${F1:-FAIL}" "${ACC:-FAIL}" | tee -a "$SUMMARY"
done
echo "=================================================" | tee -a "$SUMMARY"
echo "Finished: $(date)" | tee -a "$SUMMARY"
echo "Logs + summary in: $LOGROOT"
