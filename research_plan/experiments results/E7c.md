# E7c — Contrastive design: InfoNCE + PPR view (main setup)

Our contrastive component uses an **InfoNCE** loss (GRACE-style, node-level, with
negatives) over a **PPR-diffused second view**, aligning the primary view toward it
(no view averaging). This is the main NESS contrastive configuration.

**Setup:** locked config (SAGE-3L-128, w_cls=3, dropout 0.3), arxiv MCAR @ 80%,
800 epochs, view2 = ppr, con\_loss = infonce.

**Source:** standalone runs under `ogbn_benchmarks/results/NESS_ogbn-arxiv_MCAR_0.8_*`
(config: con\_loss=infonce, view2=ppr).

## Results — Test Macro-F1 @ 80%

| Contrastive setup | objectives | Test F1 | Test Acc |
|---|---|---|---|
| **InfoNCE + PPR** (averaged views) | anchor | 0.4395 | 0.6577 |
| **InfoNCE + PPR** (primary-view, no avg) | hist+path | 0.4371 | 0.6552 |
| Barlow-Twins + PPR (matched config) | anchor / hist+path | *pending* | *pending* |

## Reading

- InfoNCE with a PPR view is the contrastive design we adopt.
- A matched Barlow-Twins control (same locked config, ppr view) is the direct
  comparison; run pending. Prior Barlow-Twins results on the earlier edge-mask
  configuration were weaker, motivating the switch to InfoNCE + PPR.

## To do
- Run Barlow-Twins + ppr on the locked config (anchor and hist+path) to complete the
  InfoNCE-vs-BT comparison and fill the pending row.
