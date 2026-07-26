# Experiment Results — Final Tables for the Paper

One file per experiment (`E1.md`, `E2.md`, …). Only *clean, fair* numbers land
here (fair per-dataset lr for all models; final locked config unless noted). Raw
run logs live in `ogbn_benchmarks/results/`.

**Metrics:** Macro-F1 (primary) + Accuracy (secondary), on masked/unobserved nodes.
Test Macro-F1 and Accuracy on masked nodes.

**Standing caveats (apply to all files):**
- A learning-rate fairness bug (baselines ran lr=0.01 while NESS got 0.001)
  invalidated the *original* E1/E4 runs. All tables here use the fixed per-dataset
  lr (arxiv → 0.001) applied uniformly to every model.
- NESS's old default encoder (GCN + 128d) crippled it; the E11 encoder ablation
  corrected that. Numbers predating the fix are not paper-eligible.

## Index
| File | Experiment | Status |
|---|---|---|
| E1.md  | Main comparison (RQ1) | **DONE** |
| E2.md  | Multi-seed rigor (RQ1) | pending |
| E3.md  | Objective ablation (RQ3) | **DONE** |
| E4.md  | Missingness-rate sweep (RQ2) | **DONE** |
| E5.md  | Missingness mechanisms (RQ2) | pending (bonus) |
| E6.md  | Scalability | **DONE** |
| E7.md  | View analysis (RQ4) | **DONE** |
| E7b.md | Transferable enhancement (RQ5) | **DONE** |
| E7c.md | Contrastive: InfoNCE + PPR (main setup) | **DONE** |
| E8.md  | Factor-flip study (RQ3 core) | **DONE** |
| E9.md  | Regime analysis (dense vs sparse) | **DONE** |
| E10.md | Training-dynamics figure (loss curves) | **DONE** (t-SNE deferred) |
| E11.md | Design & HP choices | **DONE** — encoder, width, HP sweeps |
| E12.md | SSL benefit grows with missingness (RQ2×RQ3) | **DONE** — key finding |
