#!/bin/bash
# E7c — Contrastive regime evaluation. Does the contrastive term's value depend on
# missingness? Compare contrastive ON (locked default) vs OFF (single_view) across
# rates. At 80% single_view ~= no_ssl; test whether contrastive helps at higher rates
# (like hist+path did in E12). Characterizes contrastive's regime — no "redundant" claim.
#
# Locked config: SAGE-3L-128, edge_mask, hist+path, w_cls=3, dropout 0.3, arxiv, 800 ep.
#
# Run from ogbn_benchmarks/:
#   bash experiments/e7c_contrastive_regime.sh
#   bash experiments/e7c_contrastive_regime.sh ogbn-arxiv

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

DATASET=${1:-ogbn-arxiv}
MISSINGNESS=MCAR
EPOCHS=800
PATIENCE=20
NUM_PARTS=20
RATES=(0.6 0.8 0.9 0.95)

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e7c_contrastive_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

echo "=================================================" | tee -a "$SUMMARY"
echo "E7c Contrastive regime | NESS | $DATASET | $MISSINGNESS" | tee -a "$SUMMARY"
echo "contrastive ON vs OFF (single_view) | rates: ${RATES[*]} | 800 ep" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

run() {
    local LABEL=$1; local RATE=$2; shift 2
    echo "  running: $LABEL @ $RATE ..." | tee -a "$SUMMARY"
    python main_benchmark_ogbn.py --model NESS --dataset "$DATASET" \
        --missingness "$MISSINGNESS" --miss_rate "$RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device gpu \
        --view2 edge_mask --ssl_objective hist path \
        "$@" > "$LOGROOT/${LABEL}.log" 2>&1 \
        && echo "    $LABEL ok" || echo "    $LABEL FAILED (exit $?) — continuing"
}

for R in "${RATES[@]}"; do
    RT=${R/./}
    run "con_on_${RT}"  "$R"                 # contrastive on (default w_con)
    run "con_off_${RT}" "$R" --single_view   # contrastive off (single view)
done

# ---- Summary (rows = config, cols = rate) ----
echo "" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
echo "E7c RESULTS — Test Macro-F1 (does con_on - con_off grow with rate?)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
printf "%-10s" "config" | tee -a "$SUMMARY"; for R in "${RATES[@]}"; do printf "%-10s" "$R"; done | tee -a "$SUMMARY"; echo "" | tee -a "$SUMMARY"
for C in "con_on" "con_off"; do
    printf "%-10s" "$C" | tee -a "$SUMMARY"
    for R in "${RATES[@]}"; do
        RT=${R/./}
        F1=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOGROOT/${C}_${RT}.log" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
        printf "%-10s" "${F1:-FAIL}" | tee -a "$SUMMARY"
    done
    echo "" | tee -a "$SUMMARY"
done
echo "=================================================" | tee -a "$SUMMARY"
echo "Finished: $(date)" | tee -a "$SUMMARY"
echo "Logs + summary in: $LOGROOT"
