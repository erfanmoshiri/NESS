#!/bin/bash
# Sequential batch: E6 (scalability) -> E7d (contrastive x rate) -> E7b (hist transfer).
# Locked config inherited from runner defaults (SAGE-3L-128, ppr view, barlow, w_cls=3).
# Run inside tmux:  bash experiments/run_e6_e7d_e7b.sh

cd "$(dirname "$0")/.."   # -> ogbn_benchmarks/

echo "############################################################"
echo "# BATCH: E6 -> E7d -> E7b   |  Started: $(date)"
echo "############################################################"

echo ">>> [1/3] E6 scalability  ($(date))"
bash experiments/e6_scalability.sh 0.8
echo ">>> E6 done ($(date))"

echo ">>> [2/3] E7d contrastive x rate  ($(date))"
bash experiments/e7d_contrastive_rates.sh ogbn-arxiv
echo ">>> E7d done ($(date))"

echo ">>> [3/3] E7b hist transfer (GraphSAGE + PaGCN, +hist) @0.8  ($(date))"
STAMP=$(date +%Y%m%d_%H%M%S); LOGROOT="results/e7b_transfer_${STAMP}"; mkdir -p "$LOGROOT"
for M in GraphSAGE PaGCN; do
    echo "  running: ${M} +hist ..."
    python main_benchmark_ogbn.py --model "$M" --dataset ogbn-arxiv \
        --missingness MCAR --miss_rate 0.8 --epochs 800 --patience 20 \
        --num_parts 20 --aux_hist \
        > "$LOGROOT/${M}_hist.log" 2>&1 \
        && echo "    ${M}+hist ok" || echo "    ${M}+hist FAILED"
    F1=$(grep -oE "Test Macro-F1: [0-9.]+" "$LOGROOT/${M}_hist.log" | grep -oE "[0-9.]+" | tail -1)
    echo "    ${M}+hist Test F1 = ${F1:-FAIL}"
done
echo ">>> E7b done ($(date)) | logs: $LOGROOT"
echo "    (compare to no-hist GraphSAGE/PaGCN from E1/E4 @0.8)"

echo ">>> [4/4] Prefill ablation (fp / zero / mean) @0.8  ($(date))"
bash experiments/e_prefill.sh
echo ">>> Prefill done ($(date))"

echo "############################################################"
echo "# BATCH COMPLETE: $(date)"
echo "############################################################"
