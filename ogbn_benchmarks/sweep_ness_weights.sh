#!/bin/bash
# NESS loss-weight sweep on a small dataset (fast).
# Tests several (w_stats, w_residual) combos + the SSL-off ablation floor.
# GPU-sequential (safe on a shared GPU).

cd "$(dirname "$0")"

DATASET=${1:-amac}
EPOCHS=200
PATIENCE=20
W_CON=10.0

LOGDIR="results/sweep_ness_${DATASET}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

# (label, w_stats, w_residual)
CONFIGS=(
  "orig:100:50"
  "equal:1:1"
  "light:0.1:0.1"
  "mid:10:5"
  "ssl_off:0:0"
  "stats_only:1:0"
  "resid_only:0:1"
)

echo "================================================="
echo "NESS weight sweep | dataset=$DATASET | w_con=$W_CON"
echo "Logs: $LOGDIR"
echo "================================================="

for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r LABEL WS WR <<< "$CFG"
    echo "  running $LABEL (w_stats=$WS w_residual=$WR) ..."
    python main_benchmark_ogbn.py \
        --model NESS --dataset "$DATASET" \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --w_con "$W_CON" --w_stats "$WS" --w_residual "$WR" \
        > "$LOGDIR/${LABEL}.log" 2>&1
    TEST=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOGDIR/${LABEL}.log" | grep -oE "[0-9.]+" | tail -1)
    VAL=$(grep -oE "Best Val F1: [0-9.]+" "$LOGDIR/${LABEL}.log" | grep -oE "[0-9.]+" | tail -1)
    echo "    $LABEL -> Val $VAL | Test $TEST"
done

echo ""
echo "================================================="
echo "SWEEP SUMMARY | dataset=$DATASET"
echo "================================================="
printf "%-12s %-10s %-10s %-10s\n" "CONFIG" "w_stats" "w_resid" "TEST_F1"
for CFG in "${CONFIGS[@]}"; do
    IFS=':' read -r LABEL WS WR <<< "$CFG"
    TEST=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOGDIR/${LABEL}.log" | grep -oE "[0-9.]+" | tail -1)
    printf "%-12s %-10s %-10s %-10s\n" "$LABEL" "$WS" "$WR" "${TEST:-FAIL}"
done
echo "================================================="
echo "Logs: $LOGDIR"
