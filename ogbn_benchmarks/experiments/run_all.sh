#!/bin/bash
# Sequential batch: E9 -> E1 -> E4 -> E12 -> E7c. Each starts after the previous finishes.
# All on the locked config (SAGE-3L-128, edge_mask, hist+path, w_cls=3, dropout 0.3,
# fair per-dataset lr, 800 epochs) inherited from the runner defaults.
#
# Run inside tmux (long, multi-hour/overnight batch):
#   bash experiments/run_all.sh

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

echo "############################################################"
echo "# BATCH: E9 -> E1 -> E4 -> E12 -> E7c"
echo "# Started: $(date)"
echo "############################################################"

echo ">>> [1/5] E9 SSL-by-feature-type (binary datasets) @0.8  ($(date))"
bash experiments/e9_ssl_by_feature.sh 0.8
echo ">>> E9 done ($(date))"

echo ">>> [2/5] E1 main comparison @0.4 (all models x datasets)  ($(date))"
bash experiments/e1_main_comparison.sh 0.4
echo ">>> E1 done ($(date))"

echo ">>> [3/5] E4 rate sweep (arxiv, rates 0.2-0.9)  ($(date))"
bash experiments/e4_rate_sweep.sh ogbn-arxiv
echo ">>> E4 done ($(date))"

echo ">>> [4/5] E12 SSL-vs-missingness (arxiv, 0.9 & 0.95)  ($(date))"
bash experiments/e_ssl_vs_missingness.sh ogbn-arxiv
echo ">>> E12 done ($(date))"

echo ">>> [5/5] E7c contrastive regime (arxiv, 0.6-0.95)  ($(date))"
bash experiments/e7c_contrastive_regime.sh ogbn-arxiv
echo ">>> E7c done ($(date))"

echo "############################################################"
echo "# BATCH COMPLETE: $(date)"
echo "# Summaries:"
echo "#   E9  -> results/e9_sslfeature_*/summary.txt"
echo "#   E1  -> results/e1_main_*/summary.txt"
echo "#   E4  -> results/e4_ratesweep_*/summary.txt"
echo "#   E12 -> results/e_sslmiss_*/summary.txt"
echo "#   E7c -> results/e7c_contrastive_*/summary.txt"
echo "############################################################"
