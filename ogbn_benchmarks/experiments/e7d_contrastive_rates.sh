#!/bin/bash
# E7d — Contrastive design x missing rate. Compare four contrastive configurations
# across missing rates, to see which contrastive helps and whether its value grows
# with missingness. All on locked config (SAGE-3L-128, w_cls=3, dropout 0.3), ppr
# view where a second view exists, ssl_objective = hist path, 800 epochs.
#
# Configs:
#   bt        : Barlow-Twins       (two views, ppr)
#   infonce   : node InfoNCE       (two views, ppr)  <- current main
#   proto     : class-aware InfoNCE (two views, ppr, semi-supervised)
#   single    : single view        (no second view, no contrastive)
#
# Rates: 0.6 0.8 0.9 0.95
#
# Run from ogbn_benchmarks/:
#   bash experiments/e7d_contrastive_rates.sh
#   bash experiments/e7d_contrastive_rates.sh ogbn-arxiv

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

DATASET=${1:-ogbn-arxiv}
MISSINGNESS=MCAR
EPOCHS=800
PATIENCE=20
NUM_PARTS=20
RATES=(0.6 0.8 0.9 0.95)
CONFIGS=("bt" "infonce" "proto" "single")

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e7d_conrates_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

echo "=================================================" | tee -a "$SUMMARY"
echo "E7d Contrastive x rate | NESS | $DATASET | $MISSINGNESS" | tee -a "$SUMMARY"
echo "configs: bt / infonce / proto / single | rates: ${RATES[*]} | 800 ep, ppr, hist+path" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

# run LABEL RATE <extra args>
run() {
    local LABEL=$1; local RATE=$2; shift 2
    echo "  running: $LABEL @ $RATE ..." | tee -a "$SUMMARY"
    python main_benchmark_ogbn.py --model NESS --dataset "$DATASET" \
        --missingness "$MISSINGNESS" --miss_rate "$RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device gpu \
        --view2 ppr --ssl_objective hist path \
        "$@" > "$LOGROOT/${LABEL}.log" 2>&1 \
        && echo "    $LABEL ok" || echo "    $LABEL FAILED (exit $?) — continuing"
}

for R in "${RATES[@]}"; do
    RT=${R/./}
    run "bt_${RT}"      "$R" --con_loss barlow
    run "infonce_${RT}" "$R" --con_loss infonce --w_con 0.1
    run "proto_${RT}"   "$R" --con_loss proto   --w_con 0.2
    run "single_${RT}"  "$R" --single_view
done

# ---- Summary (rows=config, cols=rate) ----
echo "" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
echo "E7d RESULTS — Test Macro-F1 (rows=contrastive, cols=rate)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
printf "%-10s" "config" | tee -a "$SUMMARY"; for R in "${RATES[@]}"; do printf "%-10s" "$R"; done | tee -a "$SUMMARY"; echo "" | tee -a "$SUMMARY"
for C in "${CONFIGS[@]}"; do
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
