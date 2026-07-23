# E7b — Transferable Enhancement (RQ5, bonus)

Bolt the effective SSL objective (path) onto other backbones (GraphSAGE, GAT,
PaGCN) as an auxiliary loss; measure Δ Macro-F1 (with-SSL vs without) under
missingness. Shows it's a general technique, not a one-off architecture.

**Status:** PENDING — needs the SSL objective factored out to attach to other backbones.

<!-- Table: backbone | F1 no-SSL | F1 +path | ΔF1 | (same for Acc) -->
