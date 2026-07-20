#!/bin/bash
# E1 — Main comparison table (RQ1).
# All methods on ogbn-arxiv, MCAR @ 0.4, EQUAL training budget (fair comparison).
# GPU models run sequentially (shared-GPU safe); CPU non-parametric models parallel.
#
# Run from the ogbn_benchmarks/ directory:
#   bash experiments/e1_main_comparison.sh
#   bash experiments/e1_main_comparison.sh ogbn-arxiv 0.4   # dataset + rate override

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

DATASET=${1:-ogbn-arxiv}
MISS_RATE=${2:-0.4}
MISSINGNESS=MCAR
EPOCHS=400       # ceiling — NESS keeps improving past 200; early stopping trims the rest
PATIENCE=20      # stops after 20 evals (=100 epochs) of no val-F1 gain; plateaued models exit early
NUM_PARTS=20

LOGDIR="results/e1_main_${DATASET}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

CPU_MODELS=("NeighAggre" "KNN")
GPU_MODELS=("GraphSAGE" "FP" "PaGCN" "MATE" "NESS")

echo "================================================="
echo "E1 Main Comparison | $DATASET | $MISSINGNESS @ $MISS_RATE"
echo "Budget: epochs=$EPOCHS patience=$PATIENCE (equal for all learnable models)"
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

# CPU non-parametric models in parallel (no GPU contention)
declare -A PIDS
for M in "${CPU_MODELS[@]}"; do
    run_model "$M" & PIDS[$M]=$!
    echo "  launched (CPU) $M"
done

# GPU models one at a time
for M in "${GPU_MODELS[@]}"; do
    echo "  running (GPU) $M ..."
    run_model "$M" && echo "    $M ok" || echo "    $M FAILED"
done

for M in "${CPU_MODELS[@]}"; do wait "${PIDS[$M]}"; done

# ---- Summary ----
echo ""
echo "================================================="
echo "E1 RESULTS | $DATASET | $MISSINGNESS @ $MISS_RATE"
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
