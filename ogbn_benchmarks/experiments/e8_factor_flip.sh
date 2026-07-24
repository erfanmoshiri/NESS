#!/bin/bash
# E8 — Factor-flip study (RQ3 core). All on the ANCHOR objective; each run flips
# exactly ONE axis vs the shared base. Base = varying-input, resampled, global.
#   F1  anchor_fixed      : fixed input (per-node regression)   vs varying (pairwise diff)
#   F2  freeze_landmarks  : static references                    vs resampled
#   F3  anchor_local      : local (2-hop) reference reach        vs global (far landmarks)
# Effect of each factor = (flip - base). Locked NESS base: SAGE-128, edge_mask,
# arxiv MCAR @ 80%, 800 epochs. ssl_objective = anchor ONLY (isolate it).
# Single-seed (per current plan). Reports F1 + accuracy.
#
# Run from ogbn_benchmarks/:
#   bash experiments/e8_factor_flip.sh
#   bash experiments/e8_factor_flip.sh 0.8 ogbn-arxiv

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

MISS_RATE=${1:-0.8}
DATASET=${2:-ogbn-arxiv}
MISSINGNESS=MCAR
EPOCHS=800
PATIENCE=20
NUM_PARTS=20

STAMP=$(date +%Y%m%d_%H%M%S)
LOGROOT="results/e8_factorflip_${STAMP}"
mkdir -p "$LOGROOT"
SUMMARY="$LOGROOT/summary.txt"

echo "=================================================" | tee -a "$SUMMARY"
echo "E8 Factor-Flip | NESS anchor | $DATASET | $MISSINGNESS @ $MISS_RATE" | tee -a "$SUMMARY"
echo "base: SAGE-128, edge_mask, anchor only, 800 epochs" | tee -a "$SUMMARY"
echo "Started: $(date)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"

run() {
    local LABEL=$1; shift
    echo "  running: $LABEL ..." | tee -a "$SUMMARY"
    python main_benchmark_ogbn.py --model NESS --dataset "$DATASET" \
        --missingness "$MISSINGNESS" --miss_rate "$MISS_RATE" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --num_parts "$NUM_PARTS" --cache_device gpu \
        --ssl_objective anchor \
        "$@" > "$LOGROOT/${LABEL}.log" 2>&1 \
        && echo "    $LABEL ok" || echo "    $LABEL FAILED (exit $?) — continuing"
}

LABELS=("anchor_base" "F1_fixed" "F2_frozen" "F3_local")

# shared base: varying-input, resampled, global (no flags)
run "anchor_base"
# F1: flip input-varying -> fixed
run "F1_fixed"    --anchor_fixed
# F2: flip resampled -> static
run "F2_frozen"   --freeze_landmarks
# F3: flip global -> local reach
run "F3_local"    --anchor_local

# ---- Summary ----
echo "" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
echo "E8 RESULTS | anchor factor-flips | $DATASET @ $MISS_RATE" | tee -a "$SUMMARY"
echo "(effect of a factor = that row - anchor_base)" | tee -a "$SUMMARY"
echo "=================================================" | tee -a "$SUMMARY"
printf "%-14s %-10s %-10s %-10s\n" "CONFIG" "VAL_F1" "TEST_F1" "TEST_ACC" | tee -a "$SUMMARY"
for LABEL in "${LABELS[@]}"; do
    LOG="$LOGROOT/${LABEL}.log"
    VAL=$(grep -oE "Val  Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    F1=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    ACC=$(grep -oE "Test Accuracy: [0-9.]+" "$LOG" 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
    printf "%-14s %-10s %-10s %-10s\n" "$LABEL" "${VAL:-FAIL}" "${F1:-FAIL}" "${ACC:-FAIL}" | tee -a "$SUMMARY"
done
echo "=================================================" | tee -a "$SUMMARY"
echo "Finished: $(date)" | tee -a "$SUMMARY"
echo "Logs + summary in: $LOGROOT"
