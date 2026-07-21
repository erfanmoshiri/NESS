# Research Plan — Journal Extension

**Working title:** Learning Under Missing Node Features: What Makes a Self-Supervised Objective Help? (FP-prefill + stochastic structure SSL)

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

**RQ1 (the method):**
Does a GNN combining Feature-Propagation prefill with a stochastic structure-based SSL objective learn representations robust to missing node features, outperforming imputation-based and per-node-parameter paradigms on downstream classification — especially under severe missingness?

**RQ2 (robustness across conditions):**
Does the advantage hold — and grow — as missingness increases (0.2→0.8), and later across mechanisms (MCAR/MAR/MNAR)? (Early evidence: SSL adds most at 80% missing; simple neighbor-averaging dominates at low missingness.)

**RQ3 (the objective-design finding — the intellectual core):**
*Which* self-supervised objectives help a GNN learn under missingness, and *why*? We show predictability-from-structure is necessary but not sufficient: an objective helps only if it is also (a) **not redundant** with what message-passing already computes, and (b) **stochastic / input-varying** so it keeps teaching instead of saturating. Feature-space and global-position objectives fail one or both tests; a stochastic multi-hop structure objective passes both.

**RQ4 (view design — a second SSL-design axis):**
The contrastive objective is itself an SSL task, and the two views feed all objectives through the shared `z`. Currently the two views differ only by random edge-masking (weak diversity). Part of the work is to test a small set of view constructions (e.g. edge-mask, feature-mask, PPR/diffusion view, prefilled-vs-raw) and characterize *which views are best and why* — mirroring the objective analysis (RQ3) but on the view axis.

**RQ5 (transferable enhancement — bonus):**
Is the effective SSL objective a *general* enhancement — does bolting it onto other GNN backbones (GraphSAGE, GAT, PaGCN) improve their robustness too?

**Relationship:** RQ1 = the method wins. RQ2 = the win is robust and largest under severe missingness. **RQ3 is the paper's real novelty** — a principled account of what makes an SSL *objective* useful under missingness (systematic ablation over 5+ objectives). **RQ4 is the companion analysis on the *view* axis** — what makes a good contrastive view, since views feed all objectives. RQ5 elevates the finding to a transferable technique.

---

## 3. Central Claim

> We introduce **NESS**, a GNN for learning under missing node features that combines (i) **Feature-Propagation prefill** to give missing nodes long-range signal and (ii) a **stochastic, structure-based self-supervised objective** (multi-hop reachability / "path") that shapes representations to encode local graph structure. NESS outperforms imputation-based and per-node-parameter baselines on downstream node classification, with the advantage most pronounced under **severe missingness**. Beyond the method, we contribute an **empirical characterization of which self-supervised objectives help under missingness and why** — the key factor is not just predictability-from-structure but whether the objective is *stochastic and input-varying* (so it keeps teaching) and *non-redundant with message-passing aggregation*.

**Pillars of the contribution:**
1. **Method (RQ1)** — NESS: FP-prefill + stochastic structure-based SSL (path). Beats imputation-based and per-node-parameter paradigms, especially at high missingness.
2. **Robustness (RQ2)** — the advantage holds across missingness rates; the gap is largest when features are scarcest (e.g. 80% missing).
3. **The objective-design finding (the real intellectual contribution)** — a systematic study of *why* some neighborhood SSL objectives help and others are inert. Feature-space, deterministic objectives (centroid, spread, masked-feature recon) saturate quickly and add nothing; global-position objectives (PageRank) hurt; only **stochastic, input-varying, relational** objectives (multi-hop reachability) keep learning and improve downstream F1. See §7 analysis.
4. **Scalability** — scales to large graphs (arxiv-scale now; products-scale as an optional bonus) where imputation baselines (SVGA/SAT) are infeasible.

**Two components matter, and both are findings:**
- **FP-prefill is essential.** Without it, at 80% missing NESS collapses (~0.16 F1) because a 2-hop GNN starves when neighborhoods are ~80% empty. FP-prefill alone lifts it to ~0.34. This is a core component, not a preprocessing detail.
- **The SSL objective adds on top only if it is the right *kind*.** Feature/label objectives are redundant with prefill+classification; the stochastic structure objective (path) adds a further ~+2.9 F1 over no-SSL and, crucially, its loss keeps decreasing across all epochs (does not plateau).

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
| **Ours (NESS)** | FP-prefill + stochastic structure SSL (path) | ✅ | The method. |
| **Ablations** | no-SSL, FP-prefill on/off, and each objective (centroid/spread/recon/PageRank/path) | ✅ | RQ3 evidence. |

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

**E3 — Objective ablation (the systematic study). [RQ3, core]**
On ogbn-arxiv (+ small datasets), at high missingness (0.8, where SSL matters):
compare no-SSL vs each objective — **centroid, spread, masked-recon, PageRank, path** —
plus the FP-prefill on/off ablation.
Proves: FP-prefill is essential; among SSL objectives only the stochastic structure
one (path) helps; feature/global objectives are inert or harmful. This table IS the
RQ3 evidence.

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

**E7 — View analysis. [RQ4 — companion to the objective study]**
Systematically vary how the two contrastive views are built — edge-mask (current), feature-mask, PPR/diffusion view, prefilled-vs-raw — and measure downstream F1 + which view best supports the SSL objectives. Same style as E8 (objective analysis) but on the view axis: report *what works and why*. (Design note: current views differ only by random edge-mask → weak diversity; a PPR view gives local-vs-global contrast.)

**E7b — SSL as a transferable enhancement. [RQ5 — bonus]**
Add the effective SSL objective (path) as an auxiliary loss to *other* GNN backbones — GraphSAGE, GAT, PaGCN — and measure Δ Macro-F1 (with-SSL vs without) under missingness.
Proves: it's a **general technique**, not a one-off architecture. Report honestly even if uneven (that pattern is itself a finding).

**E8 — "Why objectives work/fail" analysis. [RQ3 — the paper's core insight]**
For each objective (centroid, spread, masked-recon, PageRank, path) report three signals:
**predictability-from-structure** (probe R²), **training behavior** (does the loss keep
decreasing or plateau early?), and **downstream lift** (ΔF1 vs no-SSL). The pattern:
- centroid/spread/recon: predictable but **redundant** with aggregation → loss plateaus by ~epoch 20 → ~0 lift
- PageRank (global position): weakly predictable, not class-correlated → **hurts**
- path (stochastic multi-hop reachability): loss **keeps decreasing** to the end → **+~2.9 F1**
Through-line: an objective helps iff it is *not redundant with message-passing* AND
*stochastic/input-varying* (so it stays hard). This is the paper's headline insight.

**E9 — Regime analysis (dense vs sparse features). [supporting]**
Same R²/lift across binary-BoW (small) vs dense-embedding (arxiv) datasets. Explains why SSL helps on embedding graphs but little on binary BoW.

**E10 — Representation & convergence analysis.**
t-SNE of masked-node embeddings (NESS vs no-SSL); F1 and per-objective loss curves over epochs. Visual/temporal support for the above.

---

## 8. Execution Order (cheap validates before expensive)

1. ✅ Small datasets validated under the classification-under-missingness protocol.
2. ✅ Objective investigation (extensive): centroid/spread/recon inert; PageRank harmful; **path (stochastic multi-hop reachability) works** (+2.9 F1 at 80% missing, label-free, loss keeps decreasing). FP-prefill found essential.
3. ✅ NESS(path)+FP-prefill = 0.366 at 80% missing vs no-SSL 0.338, PaGCN 0.267, KNN 0.299.
4. **Fair E1 core** at rate 0.4: all learnable baselines, same budget, on arxiv.
5. **E3 objective-ablation table** (the RQ3 evidence) — clean run of each objective at 0.8, + FP-prefill on/off.
6. **E4 rate sensitivity** (0.2→0.8) — show the gap grows with missingness (already know: SSL helps at 0.8, less at 0.4).
7. **E2 multi-seed** — confirm wins survive variance.
8. **E4/RQ4 transfer** — path bolted onto GraphSAGE/GAT/PaGCN.
9. E6 scalability; E9/E10 analysis; E5 mechanisms (bonus).

---

## 9. Deferred / Out of Scope

- **OGBN-products (2.4M)** — deferred due to shared-GPU limits. ogbn-arxiv is the primary large graph. Products may be added later as a scale bonus; the contribution does not depend on it (effective regime is dense-embedding graphs, not a specific node count). A working frugal path exists (`--cache_device cpu`) if a memory window opens.
- **Pretrain / adapt strategies** — parked. Not part of this thesis; risks splitting the paper's focus. Mention as future work in the conclusion.
- **MAR / MNAR mechanisms** — deferred; optional bonus if time allows.
- **Imputation-quality metrics (Recall@K/NDCG)** — dropped; different task.
- **Feature/global SSL objectives as *live* method components** — centroid, spread, masked-feature recon, and PageRank were all tested and **dropped** (inert or harmful). They survive only as **studied negative results** in the E3/E8 objective analysis, which is the core RQ3 evidence. The live SSL objective is `path` (stochastic multi-hop reachability).
- **`hist` (neighbor label-distribution)** — helps (~0.36) but is **semi-supervised** (uses labels) and **redundant** with `path`. Kept in the codebase as an option; reported as an analysis point (semi-supervised comparison), not part of the label-free method.

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
