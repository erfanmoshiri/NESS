#!/bin/bash
# E9 — Is SSL's benefit feature-type dependent? Run hist / anchorcls WITH vs WITHOUT
# on BINARY bag-of-words datasets, to compare the SSL margin against dense-embedding
# arxiv. Hypothesis: on binary BoW, neighborhood/structural targets are far less
# predictable (averaging 0/1 vectors ~ noise) -> SSL adds little; on dense embeddings
# (arxiv) targets are predictable -> SSL has learnable signal.
#
# For each dataset: no_ssl vs hist vs anchorcls (locked config otherwise).
# Compare (hist - no_ssl) and (anchorcls - no_ssl) across feature types.
#
# High missingness (0.8) — the regime where SSL matters most (cf. E12).
# Small datasets train full-batch automatically. Locked config: SAGE-3L, edge_mask,
# w_cls=3, dropout 0.3, 800 epochs.
#
# Run from ogbn_benchmarks/:
#   bash experiments/e9_ssl_by_feature.sh
#   bash experiments/e9_ssl_by_feature.sh 0.8

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

MISS_RATE=${1:-0.8}
MISSINGNESS=MCAR
EPOCHS=800
PATIENCE=20
NUM_PARTS=20

# binary bag-of-words datasets (arxiv = dense-embedding reference, run separately / from E3+anchorw)
DATASETS=("cora" "citeseer" "amac" "amap")
SSL_CONFIGS=("no_ssl" "hist" "anchorcls")

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e9_sslfeature_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

echo "=================================================" | tee -a "$SUMMARY"
echo "E9 SSL-by-feature-type | binary BoW datasets | $MISSINGNESS @ $MISS_RATE" | tee -a "$SUMMARY"
echo "locked config: SAGE-3L, edge_mask, w_cls=3, dropout 0.3, 800 ep" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

run() {
    local DS=$1; local LABEL=$2; shift 2
    echo "  running: ${DS}/${LABEL} ..." | tee -a "$SUMMARY"
    python main_benchmark_ogbn.py --model NESS --dataset "$DS" \
        --missingness "$MISSINGNESS" --miss_rate "$MISS_RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device gpu --view2 edge_mask \
        "$@" > "$LOGROOT/${DS}_${LABEL}.log" 2>&1 \
        && echo "    ${DS}/${LABEL} ok" || echo "    ${DS}/${LABEL} FAILED (exit $?) — continuing"
}

for DS in "${DATASETS[@]}"; do
    run "$DS" "no_ssl"     --ssl_objective
    run "$DS" "hist"       --ssl_objective hist
    run "$DS" "anchorcls"  --ssl_objective anchorcls --w_anchor 3
done

# ---- Summary (per dataset: no_ssl / hist / anchorcls + deltas) ----
echo "" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
echo "E9 RESULTS — Test Macro-F1 by dataset (binary BoW) @ $MISS_RATE" | tee -a "$SUMMARY"
echo "compare (hist - no_ssl) and (anchorcls - no_ssl) here vs arxiv (dense)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
printf "%-10s %-10s %-10s %-10s\n" "DATASET" "no_ssl" "hist" "anchorcls" | tee -a "$SUMMARY"
for DS in "${DATASETS[@]}"; do
    vals=""
    for LABEL in "${SSL_CONFIGS[@]}"; do
        F1=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOGROOT/${DS}_${LABEL}.log" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
        vals="$vals $(printf '%-10s' "${F1:-FAIL}")"
    done
    printf "%-10s%s\n" "$DS" "$vals" | tee -a "$SUMMARY"
done
echo "=================================================" | tee -a "$SUMMARY"
echo "Finished: $(date)" | tee -a "$SUMMARY"
echo "Logs + summary in: $LOGROOT"
