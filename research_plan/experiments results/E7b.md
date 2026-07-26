# E7b — Transferable enhancement: hist on other backbones (RQ5)

Add the hist (neighbor label-histogram, semi-supervised) objective to other GNN
backbones and measure the change in Test Macro-F1, to test whether the effective
objective transfers beyond NESS. ogbn-arxiv MCAR @ 80%, 800 epochs. No-hist baselines
taken from the E6 run at the same rate.

**Source:** `experiments/run_e6_e7d_e7b.sh` (E7b stage) →
`ogbn_benchmarks/results/e7b_transfer_20260725_155804/`; no-hist baselines from
`results/e6_scalability_20260725_091754/`.

**Status:** DONE.

## Results — Test Macro-F1 @ 80%

| Backbone | no hist | + hist | Δ |
|---|---|---|---|
| GraphSAGE | 0.2169 | 0.2291 | +0.012 |
| PaGCN | 0.2997 | 0.3179 | +0.018 |

## Reading

- Adding hist improves **both** backbones (+0.012 GraphSAGE, +0.018 PaGCN), so the
  label-histogram objective is a transferable enhancement, not specific to the NESS
  encoder.
- The gains are modest, consistent with hist being a semi-supervised signal that helps
  where the backbone is otherwise weak (these zero-fill baselines are low at 80%).

## Framing for the paper

Report as RQ5 evidence: the effective objective (hist) transfers as a drop-in
enhancement to other backbones, indicating that the benefit is driven by the objective
rather than by a specific architecture. Note hist is semi-supervised (uses observable
labels at training; none at inference).
