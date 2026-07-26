#!/bin/bash
# Probe: does SSL earn its place as features get scarcer?
# Compare no_ssl vs anchor vs hist+path across increasing missingness.
# Hypothesis: SSL headroom grows with missingness (encoder starves -> pretext task
# fills the gap). If the anchor/hist+path margin over no_ssl WIDENS at 0.9/0.95,
# that is the regime where SSL matters (strong RQ2+RQ3 story). If flat, the encoder
# dominates everywhere and we report that honestly.
#
# Locked base: SAGE-128, edge_mask, arxiv, 800 epochs. Single-seed.
#
# Run from ogbn_benchmarks/:
#   bash experiments/e_ssl_vs_missingness.sh

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

DATASET=${1:-ogbn-arxiv}
MISSINGNESS=MCAR
EPOCHS=800
PATIENCE=20
NUM_PARTS=20
RATES=(0.8 0.9 0.95)   # full curve on the locked config for a self-contained E12

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e_sslmiss_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

echo "=================================================" | tee -a "$SUMMARY"
echo "SSL-vs-missingness probe | NESS | $DATASET | $MISSINGNESS" | tee -a "$SUMMARY"
echo "base: SAGE-128, edge_mask, 800 epochs | rates: ${RATES[*]}" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

# run LABEL RATE <ssl args...>
run() {
    local LABEL=$1; local RATE=$2; shift 2
    echo "  running: $LABEL @ $RATE ..." | tee -a "$SUMMARY"
    python main_benchmark_ogbn.py --model NESS --dataset "$DATASET" \
        --missingness "$MISSINGNESS" --miss_rate "$RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device cpu \
        "$@" > "$LOGROOT/${LABEL}.log" 2>&1 \
        && echo "    $LABEL ok" || echo "    $LABEL FAILED (exit $?) — continuing"
}

CONFIGS=("no_ssl" "anchor" "hist_path")
for R in "${RATES[@]}"; do
    RT=${R/./}   # 0.8 -> 08 for label
    run "no_ssl_${RT}"    "$R" --ssl_objective
    run "anchor_${RT}"    "$R" --ssl_objective anchor
    run "hist_path_${RT}" "$R" --ssl_objective hist path
done

# ---- Summary (rows = config, cols = rate; shows if SSL margin widens) ----
echo "" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
echo "RESULTS — Test Macro-F1 (rows=config, cols=rate)" | tee -a "$SUMMARY"
echo "watch: does (anchor - no_ssl) / (hist_path - no_ssl) grow with rate?" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
printf "%-12s" "config" | tee -a "$SUMMARY"; for R in "${RATES[@]}"; do printf "%-10s" "$R"; done | tee -a "$SUMMARY"; echo "" | tee -a "$SUMMARY"
for C in "${CONFIGS[@]}"; do
    printf "%-12s" "$C" | tee -a "$SUMMARY"
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
