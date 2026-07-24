# E7b — Transferable enhancement: hist (semi-SSL) on other backbones (RQ5)

**Reframed:** transfer the objective that actually helps — **hist** (neighbor
label-histogram, semi-supervised) — onto other GNN backbones, and measure Δ Macro-F1
(with-hist vs without) under missingness. Shows hist is a *general* semi-SSL enhancement,
not specific to NESS's architecture.

(Original plan transferred path/anchor; changed because structural SSL is only marginal at
moderate missingness — hist is the objective with a consistent, transferable benefit.)

**Backbones:** GraphSAGE, PaGCN (both zero/mask-based GNNs with a trainable encoder).
Optionally GAT. Non-parametric (NeighAggre/KNN) excluded — no encoder to attach to.

**Status:** PENDING — needs the hist objective factored out as a reusable auxiliary loss
that other backbones' training loops can call.

<!-- Table: backbone | F1 no-hist | F1 +hist | ΔF1 | (same for Acc), per missingness rate -->

**Note:** since hist is semi-supervised (uses observable-node labels at train, none at test —
see E3), the transfer claim is "semi-SSL enhancement," reported honestly as label-using.
Consider running at a high rate (0.8/0.9) where the enhancement is most likely to show.
