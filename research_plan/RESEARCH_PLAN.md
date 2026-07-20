# Research Plan — Journal Extension

**Working title:** Learning Under Missing Node Features via Neighborhood-Centric Self-Supervision

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

**RQ1 (headline — the method):**
Does a GNN trained with neighborhood-centric self-supervised objectives (predicting properties of a node's neighborhood) learn node representations that are robust to missing node features, outperforming existing paradigms (imputation-based and per-node-parameter methods) on downstream classification?

**RQ2 (robustness across conditions):**
Does the advantage hold — and grow — as conditions get harder: across missingness rates (0.2→0.8) and, later, across missingness mechanisms (MCAR/MAR/MNAR)?

**RQ3 (transferable finding — the strongest claim):**
Is neighborhood-centric SSL a *general enhancement* rather than a one-off architecture? When bolted onto other GNN backbones (GraphSAGE, GAT, PaGCN), does it improve their robustness to missingness too?

**Relationship:** RQ1 establishes the method wins. RQ2 shows the win is robust (not a single-rate artifact). RQ3 is the elevating finding — if neighborhood SSL improves *other* models too, the contribution is a transferable technique, not just "our model." A separate **analysis** answers *why* the effective objectives work (see §7, E-analysis) — this is a supporting section, not the headline.

---

## 3. Central Claim

> We introduce **NESS**, a GNN trained with neighborhood-centric self-supervised objectives (predicting a node's neighborhood centroid and spread from graph structure) that learns representations robust to missing node features. NESS outperforms imputation-based and per-node-parameter baselines on downstream node classification, and its advantage holds and grows as missingness increases. Moreover, neighborhood-centric SSL is a **transferable enhancement**: adding it to other GNN backbones improves their robustness to missingness as well.

**Pillars of the contribution:**
1. **Method (RQ1)** — NESS: neighborhood-centric SSL for learning under missing features, evaluated end-to-end by downstream classification (not imputation quality). Beats imputation-based and per-node-parameter paradigms.
2. **Robustness (RQ2)** — the advantage holds across missingness rates (and later mechanisms); the gap widens as features get scarcer.
3. **Transferable finding (RQ3)** — neighborhood-centric SSL improves *other* GNN backbones too, establishing it as a general technique rather than a single architecture. *This is what lifts the paper above "yet another model."*
4. **Scalability** — scales to large graphs (arxiv-scale now; products-scale as an optional bonus) where imputation baselines (SVGA/SAT) are infeasible.

**Analysis (supporting, not headline):** a dedicated section explains *why* neighborhood objectives work — the effective targets (centroid, spread) are predictable from graph structure (high probe R², SSL loss decreases, positive downstream lift), whereas the residual/deviation target is structure-orthogonal (low R², loss frozen, no lift). This grounds the method's design; it is presented as an explanatory analysis, not the primary contribution.

**Note on scope:** OGBN-products (2.4M) is currently deferred due to shared-GPU limits; **ogbn-arxiv (169k, dense embeddings) is the primary large dataset**. Products may be added later as a scale bonus — the contribution does not depend on it, since the effective regime is *dense-embedding* graphs (where neighborhood targets are predictable), not a specific node count.

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
| Cora | ~2.7k | binary BoW | 5 | small; statistical rigor |
| CiteSeer | ~3.3k | binary BoW | 5 | small; statistical rigor |
| AMAC | ~13.7k | binary BoW | 5 | small; statistical rigor |
| AMAP | ~7.6k | binary BoW | 5 | small; statistical rigor |
| **ogbn-arxiv** | **169k** | **128-d dense embeddings** | 3 | **primary large graph; dense-embedding regime** |
| OGBN-products | 2.4M | 100-d embeddings | 1 | *optional* scale bonus (deferred — shared-GPU limits) |

**Dataset-type note (important for the story):** the effective regime for neighborhood-centric SSL is **dense-embedding** graphs, where neighborhood targets (centroid, spread) are predictable from structure. On small binary bag-of-words graphs (Cora/CiteSeer/AMAC/AMAP) these targets are far less predictable, so SSL adds little there — this is itself evidence for RQ2 and should be reported, not hidden. ogbn-arxiv is the primary venue for demonstrating the contribution because it is dense-embedding *and* large.

**Seed policy statement (for the paper):** small datasets and ogbn-arxiv report mean ± std over multiple seeds (5 / 3 respectively). If OGBN-products is added later, it is reported single-seed (fixed seed 72) due to compute cost, with a transparent note.

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
| **Ours (NESS)** | Full model (SSL: stats + centroid) | ✅ | Neighborhood-centric SSL. |
| **Ablations** | Ours − stats, − centroid, − both (no SSL); + residual variant | ✅ | RQ1/RQ2 support. |

**New baselines to implement:** FP (easy, ~15 lines) and PaGCN (easy, one modified GCN aggregation line). These two close the "you didn't test non-imputation methods" gap — FP covers scalable imputation, PaGCN covers the work-around camp.

**SVGA / SAT:** keep, but run only at small scale; at OGBN they are reported as OOM/infeasible — a favorable scalability contrast, not a gap.

**Cut for scope:**
- **ARWMF — dropped.** Weakest baseline: not truly an imputation method (repurposed embedding method), OOMs at 169k, only ever tested at ~7.6k nodes; awkward to justify and adds little.
- **GAT — small datasets only.** ~3.5× slower than GraphSAGE on OGBN for ~equal F1; GraphSAGE alone covers the "standard GNN" slot at scale.

---

## 7. Experiments

### Tier 1 — Core (required — the method wins)

**E1 — Main comparison table. [RQ1]**
All methods × datasets, MCAR @ rate 0.4, **same training budget** (equal epochs/patience for all learnable models — critical for fairness). Metric: Macro-F1 (Micro-F1 to appendix).
Proves: NESS beats imputation-based and per-node-parameter baselines.

**E2 — Multi-seed rigor. [RQ1]**
Repeat E1 over multiple seeds (5 small / 3 arxiv); report mean ± std.
Proves: the win is real, not seed noise. *Non-negotiable for a journal.*

**E3 — Objective ablation. [RQ1, supports analysis]**
On small datasets + ogbn-arxiv: full (stats+centroid) / −stats / −centroid / no-SSL / +residual.
Proves: the SSL objectives cause the win; centroid+stats help, residual does not.

### Tier 2 — Robustness (the advantage is general — RQ2)

**E4 — Missingness-rate sensitivity. [RQ2, money-shot]**
Macro-F1 vs rate ∈ {0.2, 0.4, 0.6, 0.8}. Degradation curves: NESS vs best baselines.
Proves: NESS degrades gracefully and the *gap widens* as features get scarcer — the advantage matters most when the problem is hardest.

**E5 — Missingness mechanisms. [RQ2, bonus]**
Repeat key comparisons under MAR / MNAR (not just MCAR). Deferred; run if time allows.
Proves: robustness generalizes beyond random missingness.

**E6 — Scalability table.**
Per-epoch + total wall-clock and peak GPU memory, all methods on ogbn-arxiv (products if added). Mark methods infeasible at scale (SVGA, SAT).
Proves: scales where prior paradigms cannot.

### Tier 3 — The elevating finding (RQ3) + analysis

**E7 — Neighborhood-SSL as a transferable enhancement. [RQ3 — strongest claim]**
Add the neighborhood-centric SSL objectives (stats+centroid) as an auxiliary loss to *other* GNN backbones — GraphSAGE, GAT, PaGCN — and measure Δ Macro-F1 (with-SSL vs without) under missingness.
Proves: neighborhood SSL is a **general technique**, not a one-off architecture — it improves other models too. Report honestly even if uneven (e.g., helps message-passing GNNs more than imputation methods — that pattern is itself a finding).

**E8 — "Why the effective objectives work" analysis. [supporting]**
For each candidate target (centroid, stats, residual): probe **predictability-from-structure** (R²) + **training behavior** (does the SSL loss decrease?) + **downstream lift** (ΔF1 vs no-SSL). Through-line: structure-predictable targets (high R², loss drops, positive lift) help; residual (low R², frozen loss, ~zero lift) does not. Explains the method's design; presented as analysis, not the headline. The failed residual becomes evidence, not embarrassment.

**E9 — Regime analysis (dense vs sparse features). [supporting]**
Same R²/lift across binary-BoW (small) vs dense-embedding (arxiv) datasets. Explains why SSL helps on embedding graphs but little on binary BoW.

**E10 — Representation & convergence analysis.**
t-SNE of masked-node embeddings (NESS vs no-SSL); F1 and per-objective loss curves over epochs. Visual/temporal support for the above.

---

## 8. Execution Order (cheap validates before expensive)

1. ✅ Small datasets run under the classification-under-missingness protocol (validated).
2. ✅ Objective investigation: R² probes show centroid/stats predictable (~0.6 on dense embeddings), residual not (~0.1); model updated to stats+centroid.
3. ✅ NESS vs PaGCN on arxiv (NESS 0.29 vs 0.25 at 400 ep) — promising, but budget was unequal.
4. **Fair E1 core:** rerun all learnable baselines at the SAME budget on arxiv (fix the unequal-epoch issue). Then E3 ablation.
5. **E2 multi-seed** on arxiv + small datasets — confirm the win survives variance.
6. **E4 rate sensitivity** (0.2→0.8) — the robustness money-shot.
7. **E7 transfer experiment** — neighborhood SSL bolted onto GraphSAGE/GAT/PaGCN. (The elevating finding.)
8. E6 scalability table; E8/E9/E10 analysis sections.
9. E5 missingness mechanisms (MAR/MNAR) — bonus if time allows.

---

## 9. Deferred / Out of Scope

- **OGBN-products (2.4M)** — deferred due to shared-GPU limits. ogbn-arxiv is the primary large graph. Products may be added later as a scale bonus; the contribution does not depend on it (effective regime is dense-embedding graphs, not a specific node count). A working frugal path exists (`--cache_device cpu`) if a memory window opens.
- **Pretrain / adapt strategies** — parked. Not part of this thesis; risks splitting the paper's focus. Mention as future work in the conclusion.
- **MAR / MNAR mechanisms** — deferred; optional bonus if time allows.
- **Imputation-quality metrics (Recall@K/NDCG)** — dropped; different task.
- **Residual objective as a *live* method component** — dropped from the model (replaced by centroid). It survives only as a **studied negative result** in E7 (the "which objectives work and why" analysis).

### Engineering / refactor TODOs (not experiments)
- ✅ **DONE — Port our model into `ogbn_benchmarks/`** as `NESS_bench.py` (`--model NESS`). Runs through the unified runner with identical folder/log/save format and full-batch (<100k) + cluster paths. `src/main_ogbn_clustered.py` is now redundant for benchmarking (kept as reference; delete once ported version is validated against it).
- ✅ **DONE — Unified saving:** every run saves config.json, training_log.jsonl, final_results.json, model_weights.pt under `results/{model}_{dataset}_{missingness}_{rate}_{timestamp}/`.

---

## 10. Open Risks

- **Fair comparison.** The current NESS>PaGCN result used unequal epochs (NESS 400 vs PaGCN 200). Must rerun all learnable baselines at the same budget before claiming the win (E1). Highest-priority fix.
- **Win must survive seeds.** Single-seed gaps can shrink under variance. E2 (multi-seed) is required before the method claim is credible.
- **RQ3 transfer may be uneven.** Neighborhood SSL might help some backbones and not others. That is still a reportable finding — but don't assume a clean universal win; run it to learn the pattern, report honestly.
- **SSL benefit is regime-dependent.** Small binary-BoW datasets show little lift (targets not predictable there). Demonstrate the contribution on dense-embedding graphs (arxiv); frame small-dataset results as regime evidence (E9).
- **No giant-scale claim without products.** Scalability pillar softens to "arxiv-scale (169k)." Mitigated by paradigm-infeasibility (SVGA/SAT OOM even at 169k) and products as add-later bonus.
- **Positioning vs MATE.** MATE is one instance of the per-node-parameter paradigm, not the target-to-beat.
- **Label leakage.** Classification loss trains only on observable nodes; masked nodes are evaluation-only. (Already enforced in code.)
