#!/bin/bash
# E3 — Objective ablation (RQ3). NESS only, ogbn-arxiv, MCAR @ 60%.
# One run per SSL objective in isolation (contrastive kept ON in all of these),
# plus: a no-SSL run, and a no-contrastive run (SSL off + contrastive off).
# Failure-tolerant; reports F1 + accuracy per configuration.
#
# Second view fixed to ppr (E7 showed it is the best view); pass a 3rd arg to change.
#
# Run from ogbn_benchmarks/:
#   bash experiments/e3_objective_ablation.sh
#   bash experiments/e3_objective_ablation.sh 0.6 ogbn-arxiv        # override rate / dataset
#   bash experiments/e3_objective_ablation.sh 0.6 ogbn-arxiv ppr    # override view too

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

MISS_RATE=${1:-0.6}
DATASET=${2:-ogbn-arxiv}
VIEW2=${3:-ppr}   # E7 showed ppr is the best second view; judge objectives under it
MISSINGNESS=MCAR
EPOCHS=400
PATIENCE=20
NUM_PARTS=20

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e3_ablation_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

# SSL objectives to test one-at-a-time (contrastive stays on for these)
SSL_OBJS=("centroid" "stats" "recon" "hist" "path" "triplet" "anchor")

echo "=================================================" | tee -a "$SUMMARY"
echo "E3 Objective Ablation | NESS | $DATASET | $MISSINGNESS @ $MISS_RATE | view2=$VIEW2" | tee -a "$SUMMARY"
echo "epochs=$EPOCHS patience=$PATIENCE | contrastive ON unless noted" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

# base command (contrastive on = default w_con=1)
run() {
    local LABEL=$1; shift
    echo "  running: $LABEL ..." | tee -a "$SUMMARY"
    python main_benchmark_ogbn.py --model NESS --dataset "$DATASET" \
        --missingness "$MISSINGNESS" --miss_rate "$MISS_RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device gpu --view2 "$VIEW2" \
        "$@" > "$LOGROOT/${LABEL}.log" 2>&1 \
        && echo "    $LABEL ok" || echo "    $LABEL FAILED (exit $?) — continuing"
}

# 1) one run per SSL objective (contrastive ON)
for OBJ in "${SSL_OBJS[@]}"; do
    run "ssl_${OBJ}" --ssl_objective "$OBJ"
done

# 2) no SSL, contrastive still ON
run "no_ssl" --ssl_objective

# 3) no contrastive at all (SSL off AND contrastive off)
run "no_contrastive" --ssl_objective --w_con 0

# ---- Summary ----
echo "" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
echo "E3 RESULTS | NESS | $DATASET | $MISSINGNESS @ $MISS_RATE" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
printf "%-16s %-10s %-10s %-10s\n" "CONFIG" "VAL_F1" "TEST_F1" "TEST_ACC" | tee -a "$SUMMARY"
for LABEL in "${SSL_OBJS[@]/#/ssl_}" "no_ssl" "no_contrastive"; do
    LOG="$LOGROOT/${LABEL}.log"
    VAL=$(grep -oE "Val  Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    F1=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    ACC=$(grep -oE "Test Accuracy: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    printf "%-16s %-10s %-10s %-10s\n" "$LABEL" "${VAL:-FAIL}" "${F1:-FAIL}" "${ACC:-FAIL}" | tee -a "$SUMMARY"
done
echo "=================================================" | tee -a "$SUMMARY"
echo "Finished: $(date)" | tee -a "$SUMMARY"
echo "Logs + summary in: $LOGROOT"
