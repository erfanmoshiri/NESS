#!/bin/bash
# E6 — Scalability. Time + peak GPU memory for every method on ogbn-arxiv (169K).
# One run per model; collects train_time_s and peak_gpu_mem_gb from final_results.json.
# Point: NESS scales at standard GNN cost (SSL + prefill add no scalability penalty).
# Locked config, MCAR @ 80%, 800 epochs (early stopping trims converged runs).
#
# Run from ogbn_benchmarks/:
#   bash experiments/e6_scalability.sh
#   bash experiments/e6_scalability.sh 0.8

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

MISS_RATE=${1:-0.8}
DATASET=ogbn-arxiv
MISSINGNESS=MCAR
EPOCHS=800
PATIENCE=20
NUM_PARTS=20

CPU_MODELS=("NeighAggre" "KNN")
GPU_MODELS=("GraphSAGE" "FP" "PaGCN" "MATE" "NESS")

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e6_scalability_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

echo "=================================================" | tee -a "$SUMMARY"
echo "E6 Scalability | $DATASET | $MISSINGNESS @ $MISS_RATE | 800 ep" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

run() {
    local M=$1
    echo "  running: $M ..." | tee -a "$SUMMARY"
    python main_benchmark_ogbn.py --model "$M" --dataset "$DATASET" \
        --missingness "$MISSINGNESS" --miss_rate "$MISS_RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device gpu \
        > "$LOGROOT/${M}.log" 2>&1 \
        && echo "    $M ok" || echo "    $M FAILED (exit $?) — continuing"
}

for M in "${CPU_MODELS[@]}" "${GPU_MODELS[@]}"; do run "$M"; done

# ---- Summary: total time + peak GPU mem (from each run's stdout) ----
echo "" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
echo "E6 RESULTS | $DATASET @ $MISS_RATE" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
printf "%-12s %-12s %-14s %-10s\n" "MODEL" "TIME(s)" "PEAK_MEM(GB)" "TEST_F1" | tee -a "$SUMMARY"
for M in "${CPU_MODELS[@]}" "${GPU_MODELS[@]}"; do
    LOG="$LOGROOT/${M}.log"
    T=$(grep -oE "Time: [0-9.]+s" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    MEM=$(grep -oE "Peak GPU mem: [0-9.]+ GB" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    F1=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    printf "%-12s %-12s %-14s %-10s\n" "$M" "${T:-FAIL}" "${MEM:-NA}" "${F1:-FAIL}" | tee -a "$SUMMARY"
done
echo "=================================================" | tee -a "$SUMMARY"
echo "Finished: $(date)" | tee -a "$SUMMARY"
echo "Logs + summary in: $LOGROOT"
