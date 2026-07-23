#!/bin/bash
# E11 — Design & hyperparameter sensitivity (supporting, not a contribution).
# NESS, ogbn-arxiv, MCAR @ 80%, locked base: SAGE encoder, hidden 128, ppr view,
# ssl=hist+path, fair lr=0.001, 2 layers, dropout 0.5, FP 40 iters.
# One knob swept at a time; everything else at the locked defaults.
#
# Encoder type (E11-a) and width (E11-b) are ALREADY DONE — recorded in
# research_plan/experiments results/E11.md. This script does the REMAINING knobs.
# 800 epochs + patience 20 (early stopping trims converged runs).
# Failure-tolerant; reports F1 + accuracy per config.
#
# Run from ogbn_benchmarks/:
#   bash experiments/e11_design_hp.sh
#   bash experiments/e11_design_hp.sh 0.8 ogbn-arxiv

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

MISS_RATE=${1:-0.8}
DATASET=${2:-ogbn-arxiv}
MISSINGNESS=MCAR
EPOCHS=800
PATIENCE=20
NUM_PARTS=20

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e11_designhp_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

echo "=================================================" | tee -a "$SUMMARY"
echo "E11 Design & HP | NESS | $DATASET | $MISSINGNESS @ $MISS_RATE" | tee -a "$SUMMARY"
echo "base: SAGE, hidden=128, ppr, ssl=hist+path, 2 layers | epochs=$EPOCHS" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

# run LABEL <extra args...>   (extra args override the locked base)
run() {
    local LABEL=$1; shift
    echo "  running: $LABEL ..." | tee -a "$SUMMARY"
    python main_benchmark_ogbn.py --model NESS --dataset "$DATASET" \
        --missingness "$MISSINGNESS" --miss_rate "$MISS_RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device gpu \
        --ssl_objective hist path --view2 ppr \
        "$@" > "$LOGROOT/${LABEL}.log" 2>&1 \
        && echo "    $LABEL ok" || echo "    $LABEL FAILED (exit $?) — continuing"
}

LABELS=()
add() { run "$1" "${@:2}"; LABELS+=("$1"); }

# ---- Baseline (all locked defaults) ----
add "base"                                   # w_*=1, 2 layers, dropout 0.5, lr default, FP 40

# ---- Loss-weight sweeps (one term relative to others=1) ----
# w_cls — skewed high: cls is the target task and gets outvoted by aux terms
add "wcls_0.5"   --w_cls 0.5
add "wcls_2"     --w_cls 2
add "wcls_3"     --w_cls 3
add "wcls_5"     --w_cls 5
# w_ssl = hist and path together (0 = SSL-off ablation)
add "wssl_0"     --w_hist 0   --w_path 0
add "wssl_0.5"   --w_hist 0.5 --w_path 0.5
add "wssl_2"     --w_hist 2   --w_path 2
add "wssl_5"     --w_hist 5   --w_path 5
# w_con (0 = contrastive-off ablation)
add "wcon_0"     --w_con 0
add "wcon_0.5"   --w_con 0.5
add "wcon_2"     --w_con 2
# w_edge (0 = edge-recon-off ablation)
add "wedge_0"    --w_edge 0
add "wedge_0.5"  --w_edge 0.5
add "wedge_2"    --w_edge 2

# ---- Architecture / optimization sensitivity ----
# encoder depth (over-smoothing vs reach)
add "layers_1"   --num_layers 1
add "layers_3"   --num_layers 3
# dropout
add "dropout_0.3" --dropout 0.3
add "dropout_0.7" --dropout 0.7
# learning rate (defend the per-dataset default is not a knife-edge)
add "lr_5e-4"    --lr 0.0005
add "lr_5e-3"    --lr 0.005
# FP prefill iterations (how much diffusion)
add "fpiters_10" --fp_iterations 10
add "fpiters_20" --fp_iterations 20

# ---- Summary ----
echo "" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
echo "E11 RESULTS | NESS | $DATASET | $MISSINGNESS @ $MISS_RATE | base SAGE-128 ppr hist+path" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
printf "%-16s %-10s %-10s %-10s\n" "CONFIG" "VAL_F1" "TEST_F1" "TEST_ACC" | tee -a "$SUMMARY"
for LABEL in "${LABELS[@]}"; do
    LOG="$LOGROOT/${LABEL}.log"
    VAL=$(grep -oE "Val  Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    F1=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    ACC=$(grep -oE "Test Accuracy: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    printf "%-16s %-10s %-10s %-10s\n" "$LABEL" "${VAL:-FAIL}" "${F1:-FAIL}" "${ACC:-FAIL}" | tee -a "$SUMMARY"
done
echo "=================================================" | tee -a "$SUMMARY"
echo "Finished: $(date)" | tee -a "$SUMMARY"
echo "Logs + summary in: $LOGROOT"
