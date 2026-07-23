#!/bin/bash
# E4 — Missingness-rate sensitivity (RQ2, the "money-shot" degradation curve).
# Sweeps missing rates {0.2, 0.4, 0.6, 0.8, 0.9} and reports F1 + accuracy per model.
# Default dataset: ogbn-arxiv (the dense-embedding regime where SSL helps, so the
# rate effect is meaningful). Models trimmed to the curve essentials.
# GPU models sequential (shared-GPU safe); CPU models parallel; failure-tolerant.
#
# Run from ogbn_benchmarks/:
#   bash experiments/e4_rate_sweep.sh
#   bash experiments/e4_rate_sweep.sh "ogbn-arxiv citeseer"     # add a contrast dataset
#   bash experiments/e4_rate_sweep.sh ogbn-arxiv "0.2 0.5 0.8"  # custom rates

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

DATASETS=(${1:-ogbn-arxiv})
RATES=(${2:-0.2 0.4 0.6 0.8 0.9})
MISSINGNESS=MCAR
EPOCHS=400
PATIENCE=20
NUM_PARTS=20

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e4_ratesweep_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

CPU_MODELS=("KNN")                                  # strongest non-parametric baseline
GPU_MODELS=("FP" "PaGCN" "MATE" "NESS")             # NESS + key learnable baselines

echo "=================================================" | tee -a "$SUMMARY"
echo "E4 Rate Sweep | $MISSINGNESS | epochs=$EPOCHS patience=$PATIENCE" | tee -a "$SUMMARY"
echo "Datasets: ${DATASETS[*]} | Rates: ${RATES[*]}" | tee -a "$SUMMARY"
echo "Models: ${CPU_MODELS[*]} ${GPU_MODELS[*]}" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

run_model() {
    local M=$1 D=$2 R=$3
    python main_benchmark_ogbn.py \
        --model "$M" --dataset "$D" \
        --missingness "$MISSINGNESS" --miss_rate "$R" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device cpu \
        > "$LOGROOT/${D}_r${R}_${M}.log" 2>&1
}

for D in "${DATASETS[@]}"; do
    for R in "${RATES[@]}"; do
        echo "" | tee -a "$SUMMARY"
        echo ">>> $D @ rate $R  [$(date +%H:%M:%S)]" | tee -a "$SUMMARY"

        declare -A PIDS=()
        for M in "${CPU_MODELS[@]}"; do
            run_model "$M" "$D" "$R" & PIDS[$M]=$!
        done
        for M in "${GPU_MODELS[@]}"; do
            echo "  running (GPU) $M ..."
            if run_model "$M" "$D" "$R"; then echo "    $M ok"; else echo "    $M FAILED (exit $?) — continuing"; fi
        done
        for M in "${CPU_MODELS[@]}"; do
            if wait "${PIDS[$M]}"; then echo "    $M ok"; else echo "    $M FAILED — continuing"; fi
        done
    done
done

# ---- Summary: for each dataset, a rate × model table of Test F1 (and a second of Test Acc) ----
ALL_MODELS=("${CPU_MODELS[@]}" "${GPU_MODELS[@]}")
grab() { grep -oE "$2: [0-9.]+" "$1" 2>/dev/null | grep -oE "[0-9.]+" | tail -1; }

for METRIC in "Test Macro-F1" "Test Accuracy"; do
    echo "" | tee -a "$SUMMARY"
    echo "=================================================" | tee -a "$SUMMARY"
    echo "E4 RESULTS — ${METRIC} (rows=rate, cols=model)" | tee -a "$SUMMARY"
    echo "=================================================" | tee -a "$SUMMARY"
    for D in "${DATASETS[@]}"; do
        echo "" | tee -a "$SUMMARY"
        echo "--- $D ---" | tee -a "$SUMMARY"
        printf "%-6s" "rate" | tee -a "$SUMMARY"
        for M in "${ALL_MODELS[@]}"; do printf "%-11s" "$M" | tee -a "$SUMMARY"; done
        echo "" | tee -a "$SUMMARY"
        for R in "${RATES[@]}"; do
            printf "%-6s" "$R" | tee -a "$SUMMARY"
            for M in "${ALL_MODELS[@]}"; do
                V=$(grab "$LOGROOT/${D}_r${R}_${M}.log" "$METRIC")
                printf "%-11s" "${V:-FAIL}" | tee -a "$SUMMARY"
            done
            echo "" | tee -a "$SUMMARY"
        done
    done
done
echo "" | tee -a "$SUMMARY"
echo "Finished: $(date)" | tee -a "$SUMMARY"
echo "Logs + summary in: $LOGROOT"
