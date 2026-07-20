# Research Plan — Journal Extension

**Working title:** Learning Under Missing Node Features: Structure-Predicted Statistics vs. Per-Node Memorization

**Status:** Planning. Single-thesis journal extension of prior conference work on graph attribute imputation.

---

## 1. Framing

The conference paper framed the problem as **feature imputation** (reconstruct missing raw features; evaluate with Recall@K / NDCG).

This extension **reframes the problem as learning under missingness**: given a graph where a fraction of nodes have no observed features, learn node representations that support an accurate **downstream node-classification** task.

- Imputation is no longer the goal or the metric. It may exist only as an internal/auxiliary training signal.
- The unified evaluation metric across **all** datasets is **classification Macro-F1** on the masked nodes.
- Old imputation-quality results (Recall@K/NDCG) are **not reused**.

---

## 2. Research Questions

**RQ1 (mechanism / ablation):**
Do neighborhood-centric SSL objectives (predicting neighborhood embedding spread + centroid residual) improve downstream classification robustness under missing node features?

**RQ2 (headline / architectural):**
Does *predicting feature statistics from neighborhood structure* generalize better under missingness than *learning a per-node embedding for each missing node* (as MATE does)?

**Relationship:** RQ2 is the headline contribution. RQ1 is the supporting ablation that proves the SSL objectives are the *mechanism* behind RQ2's advantage, not an incidental factor.

---

## 3. Central Claim

> Predicting feature statistics from neighborhood structure generalizes better under missingness than memorizing per-node embeddings, because per-node embeddings for missing nodes receive no direct supervision and effectively memorize rather than generalize. The neighborhood-centric SSL objectives are the mechanism that delivers this robustness, and the approach scales to graphs (2.4M nodes) where prior imputation methods are infeasible.

**Three pillars of the contribution:**
1. New problem formulation — learning under missingness (not imputation).
2. New architectural claim — structure-prediction > per-node memorization (RQ2).
3. Scalability — works at 2.4M nodes where SVGA/SAT/original MATE cannot run.

---

## 4. Unified Protocol (all datasets)

```
missing features (masked nodes zeroed) → encoder (+ SSL objectives) → node embedding z → classification head → Macro-F1 on masked nodes
```

- **End-to-end**, not impute-then-classify.
- Train with cross-entropy on **observable-node** labels (no label leakage to masked nodes) + SSL objectives.
- On binary-feature datasets (Cora/CiteSeer), feature reconstruction *may* be kept as an auxiliary SSL loss; on OGBN embeddings it is not meaningful. This is the only cross-dataset difference — the evaluation protocol is identical.

---

## 5. Datasets

| Dataset | Nodes | Features | Seeds | Role |
|---|---|---|---|---|
| Cora | ~2.7k | binary BoW | 5 | statistical rigor |
| CiteSeer | ~3.3k | binary BoW | 5 | statistical rigor |
| AMAP | — | — | 5 | statistical rigor |
| AMAC | — | — | 5 | statistical rigor |
| OGBN-products | 2.4M | 100-d embeddings | 1 | scale; carries most experiments |

**Seed policy statement (for the paper):** OGBN-products is reported single-seed (fixed seed 72) due to compute cost; the four smaller benchmarks report mean ± std over 5 seeds to establish statistical significance. OGBN seed is fixed and documented for reproducibility.

**Missingness:** MCAR only for the main paper. MAR / MNAR deferred; test later if time permits (mechanism comparison would be a bonus contribution, not a requirement).

---

## 6. Methods Compared

Baselines are chosen to span **every category** of the missingness landscape, so the comparison can't be dismissed as "only imputation methods." Categorization:

| Category | Methods | Scales to 2.4M? | Notes |
|---|---|---|---|
| Non-parametric | NeighAggre, KNN | ✅ | Simple floors |
| Imputation — learnable | SVGA, SAT | ❌ | Reconstruct features first; OOM at scale (SAT O(N²) decoder ~20k cap, SVGA per-node params). Run at small scale; report as infeasible in E3. |
| Imputation — scalable | **FP** (Feature Propagation) | ✅ | De-facto standard, O(E)/iter, benchmarked on OGBN-products in its own paper. |
| Work-around (no reconstruction) | **PaGCN** | ✅ | Imputation-free masked aggregation, GCN-cost. Represents the non-imputation camp. |
| Per-node memorization | MATE | ✅ | Our conceptual foil (RQ2). |
| GNN (zero-fill) | GraphSAGE, GAT | ✅ | Standard GNNs on zero-filled features. GAT small-scale only (too slow on OGBN, ~similar F1 to GraphSAGE). |
| **Ours** | Full model (SSL: stats + residual) | ✅ | Structure-predicted statistics. |
| **Ablations** | Ours − stats, Ours − residual, Ours − both (no SSL) | ✅ | RQ1 support. |

**New baselines to implement:** FP (easy, ~15 lines) and PaGCN (easy, one modified GCN aggregation line). These two close the "you didn't test non-imputation methods" gap — FP covers scalable imputation, PaGCN covers the work-around camp.

**SVGA / SAT:** keep, but run only at small scale; at OGBN they are reported as OOM/infeasible — a favorable scalability contrast, not a gap.

**Cut for scope:**
- **ARWMF — dropped.** Weakest baseline: not truly an imputation method (repurposed embedding method), OOMs at 169k, only ever tested at ~7.6k nodes; awkward to justify and adds little.
- **GAT — small datasets only.** ~3.5× slower than GraphSAGE on OGBN for ~equal F1; GraphSAGE alone covers the "standard GNN" slot at scale.

---

## 7. Experiments

### Tier 1 — Core (required)

**E1 — Main comparison table.**
Methods × datasets, MCAR @ rate 0.4. Metric: Macro-F1 (Micro-F1 to appendix).
Proves: beats baselines + MATE across scales.

**E2 — Ablation.**
On ≥3 datasets incl. OGBN: full / −stats / −residual / −both.
Proves (RQ1): each SSL objective contributes; removing SSL collapses the advantage.

**E3 — Scalability table.**
Per-epoch + total wall-clock and peak GPU memory, all methods on OGBN. Explicitly mark methods infeasible at 2.4M (SVGA, SAT, original MATE formulation).
Proves: scales where prior work cannot (standalone contribution).

### Tier 2 — Strongly expected

**E4 — Missingness-rate sensitivity.**
Macro-F1 vs rate ∈ {0.2, 0.4, 0.6, 0.8} on 2–3 datasets. Plot degradation curves: Ours vs MATE vs best GNN.
Proves: graceful degradation (robustness money-shot).

**E5 — Hyperparameter sensitivity.**
Key knobs: SSL loss weights, num_parts (clusters), embedding dim. Show non-brittleness.
Proves: results not cherry-picked.

**E6 — Statistical rigor.**
Mean ± std over 5 seeds on the four small datasets. (OGBN single-seed per policy above.)
Proves: reported wins are real, not seed noise.

### Tier 3 — Strengtheners (elevate to strong accept)

**E7 — "Why MATE fails" diagnostic. [differentiator]**
Empirically show MATE's per-node embeddings for missing nodes don't learn (embeddings stay near init / negligible gradient signal / classification loss flat). This is the intellectual core of RQ2 — demonstrate it, don't just assert it.

**E8 — Representation analysis.**
t-SNE of missing-node embeddings: Ours (class-structured) vs MATE (collapsed/random). Visual evidence for E7.

**E9 — Convergence curves.**
F1 and losses over epochs, Ours vs MATE. Supports the observed "MATE classification loss never decreases" finding.

---

## 8. Execution Order (cheap validates before expensive)

1. Re-run small datasets (Cora/CiteSeer/AMAP/AMAC) under the classification-under-missingness protocol → validates protocol + E1/E2/E6 on cheap graphs.
2. Lock ablation harness (E2) on small datasets.
3. Run OGBN E1 + E2 (single seed).
4. E3 scalability measurements (piggyback on the OGBN runs).
5. E4 rate sensitivity (small datasets first, then OGBN).
6. E7 + E8 + E9 diagnostics (MATE vs Ours).
7. E5 hyperparameter sweeps (last; only the knobs that matter).

---

## 9. Deferred / Out of Scope

- **Pretrain / adapt strategies** — parked. Not part of this thesis; risks splitting the paper's focus. Mention as future work in the conclusion.
- **MAR / MNAR mechanisms** — deferred; optional bonus if time allows.
- **Imputation-quality metrics (Recall@K/NDCG)** — dropped; different task.

### Engineering / refactor TODOs (not experiments)
- **Port our model into `ogbn_benchmarks/`** as `Ours.py` so all methods share one runner (`main_benchmark_ogbn.py`), one protocol, and one logging format. Currently our model trains via `src/main_ogbn_clustered.py` (separate entry point + log format from the baselines), which makes the E1 comparison table harder to assemble consistently. Unifying this before the big experiment sweep will save effort and reduce protocol-mismatch risk.
- **Small-dataset protocol for our model** — `src/main_ogbn_clustered.py` is OGBN-specific; needs the same full-batch (<100k) path the baselines now have, or fold it into the unified runner above.

---

## 10. Open Risks

- Single-seed OGBN — mitigated by transparent seed-policy statement + 5-seed rigor on small datasets.
- MATE diagnostic (E7) must actually reproduce the failure cleanly — if MATE can be tuned to work, the RQ2 claim weakens. Verify early.
- Ensure no label leakage: classification loss trains only on observable nodes; masked nodes are evaluation-only.
