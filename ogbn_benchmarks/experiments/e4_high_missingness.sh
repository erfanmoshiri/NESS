#!/bin/bash
# E4 (partial) — high-missingness stress test (RQ2).
# Same as E1 but at 80% missingness: masked nodes have FEW observable neighbors,
# so trivial neighbor-averaging (KNN/NeighAggre) should collapse and structure/
# SSL-based methods should hold up better. Tests the "wins where it matters" story.
#
# Run from ogbn_benchmarks/:
#   bash experiments/e4_high_missingness.sh
#   bash experiments/e4_high_missingness.sh ogbn-arxiv 0.8

cd "$(dirname "$0")/.."

DATASET=${1:-ogbn-arxiv}
MISS_RATE=${2:-0.8}
MISSINGNESS=MCAR
EPOCHS=400
PATIENCE=20
NUM_PARTS=20

LOGDIR="results/e4_miss${MISS_RATE}_${DATASET}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

CPU_MODELS=("NeighAggre" "KNN")
GPU_MODELS=("GraphSAGE" "FP" "PaGCN" "MATE" "NESS")

echo "================================================="
echo "E4 High-Missingness | $DATASET | $MISSINGNESS @ $MISS_RATE"
echo "Budget: epochs=$EPOCHS patience=$PATIENCE"
echo "Logs: $LOGDIR"
echo "================================================="

run_model() {
    local M=$1
    python main_benchmark_ogbn.py \
        --model "$M" --dataset "$DATASET" \
        --missingness "$MISSINGNESS" --miss_rate "$MISS_RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device cpu \
        > "$LOGDIR/${M}.log" 2>&1
}

declare -A PIDS
for M in "${CPU_MODELS[@]}"; do
    run_model "$M" & PIDS[$M]=$!
    echo "  launched (CPU) $M"
done
for M in "${GPU_MODELS[@]}"; do
    echo "  running (GPU) $M ..."
    run_model "$M" && echo "    $M ok" || echo "    $M FAILED"
done
for M in "${CPU_MODELS[@]}"; do wait "${PIDS[$M]}"; done

echo ""
echo "================================================="
echo "E4 RESULTS | $DATASET | $MISSINGNESS @ $MISS_RATE"
echo "================================================="
printf "%-12s %-10s %-10s\n" "MODEL" "VAL_F1" "TEST_F1"
for M in "${CPU_MODELS[@]}" "${GPU_MODELS[@]}"; do
    LOG="$LOGDIR/${M}.log"
    VAL=$(grep -oE "Val  Macro-F1: [0-9.]+" "$LOG" | grep -oE "[0-9.]+" | tail -1)
    TEST=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOG" | grep -oE "[0-9.]+" | tail -1)
    printf "%-12s %-10s %-10s\n" "$M" "${VAL:-FAIL}" "${TEST:-FAIL}"
done | tee "$LOGDIR/summary.txt"
echo "================================================="
echo "Logs + summary in: $LOGDIR"
