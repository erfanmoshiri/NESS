#!/bin/bash
# E1 — Main comparison table (RQ1).
# ALL methods × ALL datasets, MCAR @ fixed rate, EQUAL training budget (fair comparison).
# GPU models run sequentially (shared-GPU safe); CPU non-parametric models in parallel.
# Failure-tolerant: if a model crashes, log it and continue — the sweep never stops.
#
# Run from the ogbn_benchmarks/ directory:
#   bash experiments/e1_main_comparison.sh
#   bash experiments/e1_main_comparison.sh 0.4          # override miss_rate
#   bash experiments/e1_main_comparison.sh 0.4 "cora citeseer"   # override datasets

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

MISS_RATE=${1:-0.4}
DATASETS=(${2:-cora citeseer amac amap ogbn-arxiv})
MISSINGNESS=MCAR
EPOCHS=800       # match locked config (NESS still improves past 400)
PATIENCE=20
NUM_PARTS=20

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e1_main_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

CPU_MODELS=("NeighAggre" "KNN")
GPU_MODELS=("GraphSAGE" "FP" "PaGCN" "MATE" "NESS")

echo "=================================================" | tee -a "$SUMMARY"
echo "E1 Main Comparison | $MISSINGNESS @ $MISS_RATE | epochs=$EPOCHS patience=$PATIENCE" | tee -a "$SUMMARY"
echo "Datasets: ${DATASETS[*]}" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

run_model() {
    local M=$1 D=$2
    python main_benchmark_ogbn.py \
        --model "$M" --dataset "$D" \
        --missingness "$MISSINGNESS" --miss_rate "$MISS_RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device cpu \
        > "$LOGROOT/${D}_${M}.log" 2>&1
}

for D in "${DATASETS[@]}"; do
    echo "" | tee -a "$SUMMARY"
    echo ">>> DATASET: $D  [$(date +%H:%M:%S)]" | tee -a "$SUMMARY"

    # CPU non-parametric models in parallel
    declare -A PIDS=()
    for M in "${CPU_MODELS[@]}"; do
        run_model "$M" "$D" & PIDS[$M]=$!
    done

    # GPU models sequentially; never abort the loop on failure
    for M in "${GPU_MODELS[@]}"; do
        echo "  running (GPU) $M on $D ..."
        if run_model "$M" "$D"; then echo "    $M ok"; else echo "    $M FAILED (exit $?) — continuing"; fi
    done

    for M in "${CPU_MODELS[@]}"; do
        if wait "${PIDS[$M]}"; then echo "    $M ok"; else echo "    $M FAILED — continuing"; fi
    done
done

# ---- Summary table (F1 + Accuracy), per dataset ----
echo "" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
echo "E1 RESULTS | $MISSINGNESS @ $MISS_RATE" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
for D in "${DATASETS[@]}"; do
    echo "" | tee -a "$SUMMARY"
    echo "--- $D ---" | tee -a "$SUMMARY"
    printf "%-12s %-10s %-10s %-10s\n" "MODEL" "VAL_F1" "TEST_F1" "TEST_ACC" | tee -a "$SUMMARY"
    for M in "${CPU_MODELS[@]}" "${GPU_MODELS[@]}"; do
        LOG="$LOGROOT/${D}_${M}.log"
        VAL=$(grep -oE "Val  Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
        F1=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
        ACC=$(grep -oE "Test Accuracy: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
        printf "%-12s %-10s %-10s %-10s\n" "$M" "${VAL:-FAIL}" "${F1:-FAIL}" "${ACC:-FAIL}" | tee -a "$SUMMARY"
    done
done
echo "=================================================" | tee -a "$SUMMARY"
echo "Finished: $(date)" | tee -a "$SUMMARY"
echo "Logs + summary in: $LOGROOT"
