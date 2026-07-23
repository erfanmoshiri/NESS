# SSL Objectives — Comprehensive Inventory (RQ3 / E3)

Master list of candidate self-supervised auxiliary objectives for NESS under
missingness. For each: tested?, result, static vs stochastic, a possible stochastic
reformulation if it's static, implementation difficulty, and intuition on how
class-correlated it is (i.e. likely to help the downstream task).

## The design rule (earned from experiments)
An objective helps only if it is:
1. **Predictable from structure** (so the loss can decrease), AND
2. **Not redundant** with what message-passing aggregation already computes, AND
3. **Class-correlated** (so shaping `z` transfers to classification).
Bonus factor: **stochastic / input-varying** (fresh target each step) sustains learning
and avoids the plateau that killed the deterministic feature objectives.

Key axes: *feature-space vs structural*, *local vs global*, *deterministic vs stochastic*,
*label-free vs label-based*.

Legend — Result: ✅ helps / ➖ inert / ❌ hurts / ❓ untested.
Class-corr and Difficulty are **intuition estimates**, not measured, unless a result exists.

---

## A. TESTED objectives

| Objective | Axis | Static/Stoch | Result | Class-corr | Notes |
|---|---|---|---|---|---|
| **centroid** (predict neighbor mean) | feature, local | static | ➖ inert (plateaus ~ep20) | low | redundant with aggregation (GCN already averages neighbors) |
| **stats** (predict neighbor std) | feature, local | static | ➖ inert | low | 2nd moment; still feature-space redundant |
| **recon** (masked-feature reconstruction) | feature, local | stochastic (random mask) | ➖ inert (frozen loss) | low–med | stochastic but *easy* → saturates; feature values redundant w/ prefill |
| **pagerank** (predict PageRank decile) | global importance | static | ❌ hurt | low | global importance not class-correlated on arxiv; per-node scalar (can't easily stochasticize) |
| **path** (multi-hop reachability: near ≤2 hop vs far ≥3 hop) | structural, local | **stochastic** (fresh walks + negatives) | ✅ **helps (+2.9 @80%), keeps learning** | **med–high** | THE winner. non-redundant, stochastic, input-varying |
| **hist** (neighbor label distribution) | label-based, local | static | ✅ helps (~0.36) | **high** (uses labels) | semi-supervised (uses labels); redundant w/ path; kept as analysis point |
| **triplet** (distance ranking: anchor closer to near than far) | structural, local | **stochastic** (fresh triplets) | ❓ implemented, not cleanly evaluated | med–high | conceptually distinct from path (ranking vs binary) — **finish this** |

**Takeaway:** feature-space + deterministic-easy = inert; global-importance = hurts;
**stochastic + structural + local (path) = works.** hist works but is label-based.

---

## B. UNTESTED — structural / positional (the promising camp)

| Objective | Axis | Static/Stoch | Stochastic version? | Difficulty | Class-corr (intuition) | Priority |
|---|---|---|---|---|---|---|
| **Anchor-distance PE** (hop-distance to K landmark nodes) | structural, **global** | naturally **stochastic** (resample landmarks each epoch) | already stochastic — new landmarks per step | Medium (BFS/APSP from K anchors, cheap for small K) | **med–high** — global position may track topic clusters | **HIGH** — best complement to path; fills the *global* slot pagerank failed, done the stochastic way |
| **Random-walk PE** (return prob after 1…T steps) | structural, local→mid | static per node, but **stochastic** if estimated by sampled walks | estimate via fresh sampled walks each epoch (Monte-Carlo) | Medium | med — long-step components capture cycles (non-redundant); short steps ≈ degree (redundant) | Medium — overlaps path; test as path-generalization |
| **Shortest-path distance** (predict hop distance between a sampled pair) | structural, local–global | **stochastic** (sample fresh pairs) | already stochastic | Medium (BFS per query, cap hops) | med | Medium — close to path/triplet; may be redundant |
| **Common-neighbor / Jaccard / Adamic-Adar** (predict overlap score for a pair) | structural, local | **stochastic** (sample fresh pairs) | already stochastic | Low–Med | med — link-likelihood proxy, homophily-linked | Medium — cheap, worth a shot |
| **Betweenness centrality** (predict bucket) | structural, **global** (role) | static (per-node scalar) | hard to stochasticize (scalar per node); could do *pairwise* "which node is more between?" | **High cost** (O(NM) exact; approx needed) | med — bridge/bottleneck role ≠ degree, genuinely new | Low–Med — interesting but expensive + non-stochastic (pagerank-fate risk) |
| **k-core number** (predict core index) | structural, global | static | pairwise "higher core?" ranking | Low (k-core is cheap) | low–med — coarse density measure, degree-correlated | Low |
| **Structural role / struc2vec-style** (predict role embedding) | structural, global | static | resample role-context | High (role embeddings expensive) | med | Low — heavy machinery |
| **Motif / triangle count** per node | structural, local (higher-order) | static | stochastic via sampled ego-subgraphs | Medium | low–med — higher-order but weakly class-linked | Low–Med |
| **Degree centrality** | structural, local | static | — | Trivial | **very low — redundant with aggregation** | **Skip** |
| **Closeness centrality** | structural, global | static | pairwise ranking | High cost (needs distances) | low — correlated w/ pagerank/degree | **Skip** (pagerank fate) |

---

## C. UNTESTED — contrastive family (label-free)

| Objective | Axis | Static/Stoch | Difficulty | Class-corr (intuition) | Priority |
|---|---|---|---|---|---|
| **DGI** (node vs graph-summary MI) | global contrast | stochastic (readout) | Low–Med | med | Medium — standard, easy to add |
| **GMI** (node vs its own neighborhood MI) | local contrast | stochastic | Medium | med | Medium |
| **GRACE / GCA** (node vs augmented view) | instance contrast | stochastic (augmentations) | Medium | med | Medium — depends on view (ties to RQ4) |
| **GCC-style subgraph instance discrimination** | subgraph contrast | stochastic (sampled subgraphs) | High | med | Low–Med |
| **BYOL / AFGRL** (no-negative bootstrap) | instance | stochastic | Medium (EMA target) | med | Low–Med — augmentation-free variant interesting |

Note: our current **Barlow-Twins contrastive** already occupies this camp (weak-to-neutral, untested in isolation — worth an on/off ablation with `--w_con 0`).

---

## D. UNTESTED — feature / generative (we found this camp weak)

| Objective | Axis | Static/Stoch | Difficulty | Class-corr | Priority |
|---|---|---|---|---|---|
| **Denoising** (add noise, reconstruct) | feature | stochastic (noise) | Low | low | Low — likely inert like recon |
| **Masked attribute modeling, varied ratio/schedule** | feature | stochastic | Low | low–med | Low — recon variant; test only to confirm regime |
| **Feature-channel prediction** (predict one channel from rest) | feature | stochastic (which channel) | Low | low | Low |

**Overall intuition:** feature-space objectives are redundant with FP-prefill + classification;
low priority, mostly for completeness/ablation to support the "feature objectives don't help" finding.

---

## E. Label-based / semi-supervised (report as semi-supervised, not core SSL)

| Objective | Static/Stoch | Difficulty | Class-corr | Notes |
|---|---|---|---|---|
| **hist** (neighbor label dist) | static | Low | high | TESTED ✅ — semi-supervised; stochastic version: resample which observable labels are visible each epoch |
| **pseudo-label consistency / propagation** | stochastic | Medium | high | risks label leakage; semi-supervised |
| **label smoothness regularizer** | static | Low | high | semi-supervised |

---

## Recommended next tests (spanning subset, not exhaustive)

1. **Finish `triplet`** — already implemented; closes the tested-objective set (ranking vs binary structural).
2. **Anchor-distance PE (stochastic)** — HIGH priority; the global, stochastic complement to path. Tests whether the "stochastic+input-varying" principle generalizes local→global.
3. *(optional)* **Random-walk PE** — as a path-generalization, to see if longer-range reachability adds over 2-hop path.
4. *(optional)* **Barlow-contrastive on/off** — isolate what the existing contrastive term contributes.
5. Keep degree / pagerank / closeness / denoising as **negative-result rows** (predicted inert/harmful) to support the RQ3 principle — run 1–2 to confirm, don't over-invest.

**Guiding principle for selection:** pick objectives that fill distinct *axis cells*
(local/global × static/stochastic × structural/feature/label) rather than many similar ones.
The story is "we mapped the space and found stochastic-structural is what works," not "we tried 20 losses."
