#!/bin/bash
# Run E3 -> E7 -> E11 sequentially (each starts only after the previous finishes).
# All on the locked base: SAGE-128, ppr, hist+path, arxiv @ 80%, 800 epochs.
#
# Run from ogbn_benchmarks/ (inside tmux — this is a multi-day batch):
#   bash experiments/run_e3_e7_e11.sh
#   bash experiments/run_e3_e7_e11.sh 0.8 ogbn-arxiv

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

RATE=${1:-0.8}
DATASET=${2:-ogbn-arxiv}

echo "############################################################"
echo "# BATCH: E3 -> E7 -> E11  |  $DATASET @ $RATE"
echo "# Started: $(date)"
echo "############################################################"

echo ">>> [1/3] E3 objective ablation  ($(date))"
bash experiments/e3_objective_ablation.sh "$RATE" "$DATASET" ppr
echo ">>> E3 done ($(date))"

echo ">>> [2/3] E7 view ablation  ($(date))"
bash experiments/e7_view_ablation.sh "$RATE" "$DATASET"
echo ">>> E7 done ($(date))"

echo ">>> [3/3] E11 design & HP  ($(date))"
bash experiments/e11_design_hp.sh "$RATE" "$DATASET"
echo ">>> E11 done ($(date))"

echo "############################################################"
echo "# BATCH COMPLETE: $(date)"
echo "# Summaries: results/e3_ablation_* , results/e7_viewablation_* , results/e11_designhp_*"
echo "############################################################"
