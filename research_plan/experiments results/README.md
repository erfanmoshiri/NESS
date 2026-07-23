# Experiment Results — Final Tables for the Paper

One file per experiment (`E1.md`, `E2.md`, …). Only *clean, fair* numbers land
here (fair per-dataset lr for all models; final locked config unless noted). Raw
run logs live in `ogbn_benchmarks/results/`.

**Metrics:** Macro-F1 (primary) + Accuracy (secondary), on masked/unobserved nodes.
Single-seed unless a ± is shown (multi-seed = E2, pending).

**Standing caveats (apply to all files):**
- A learning-rate fairness bug (baselines ran lr=0.01 while NESS got 0.001)
  invalidated the *original* E1/E4 runs. All tables here use the fixed per-dataset
  lr (arxiv → 0.001) applied uniformly to every model.
- NESS's old default encoder (GCN + 128d) crippled it; the E11 encoder ablation
  corrected that. Numbers predating the fix are not paper-eligible.

## Index
| File | Experiment | Status |
|---|---|---|
| E1.md  | Main comparison (RQ1) | pending — rerun fair lr + locked config |
| E2.md  | Multi-seed rigor (RQ1) | pending |
| E3.md  | Objective ablation (RQ3) | pending — rerun post encoder-fix |
| E4.md  | Missingness-rate sweep (RQ2) | pending — rerun fair lr |
| E5.md  | Missingness mechanisms (RQ2) | pending (bonus) |
| E6.md  | Scalability | pending |
| E7.md  | View analysis (RQ4) | pending — rerun post encoder-fix |
| E7b.md | Transferable enhancement (RQ5) | pending |
| E7c.md | Contrastive-loss ablation (RQ4) | pending |
| E8.md  | Factor-flip study (RQ3 core) | pending |
| E9.md  | Regime analysis (dense vs sparse) | pending |
| E10.md | Convergence + representation viz | pending |
| E11.md | Design & HP choices | **partial — encoder + width DONE** |
