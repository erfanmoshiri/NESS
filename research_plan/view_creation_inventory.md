# Second-View Creation — Comprehensive Inventory (RQ4 / E7)

Master list of ways to build the **second contrastive view** in NESS. The two views
are fused (`z = (z1+z2)/2`) and contrasted (Barlow Twins); they must be *different
enough to be informative* but *consistent enough to represent the same node*.

**Current default:** two random **edge-masks** → views are nearly identical → weak
diversity. This is the axis we've barely explored (1 of ~15 options tested).

## What makes a good view HERE (under missingness)
- **Diverse** from the other view (informative contrast), but consistent.
- **Long-range / global reach** helps most — missing nodes starve locally, so a view
  that reaches farther (diffusion, deeper encoder) injects signal the local view lacks.
- **Missingness-relevance** is the discriminating factor (replaces "class-correlation"
  from the objectives doc): does this view specifically help when features are absent?

Legend — Tested: ✅ / ❓ untested. Priority ⭐ = recommended to test (top picks marked).
Static/Stoch = does the view change each epoch. Miss-rel = missingness relevance (intuition).

Axes: *what's perturbed* (structure / feature / encoder / adaptive) × *static vs stochastic*
× *local vs global receptive field* × *missingness-relevance*.

---

## A. Structure perturbation (change the graph the encoder sees)

| View | Axis | Static/Stoch | Difficulty | Miss-rel | Priority |
|---|---|---|---|---|---|
| **Edge masking** (drop random edges) — CURRENT | structure, local | stochastic | trivial (done) | low — both views ~1-hop, weak diversity | baseline |
| **PPR / diffusion view** (replace adj with personalized-PageRank / heat-kernel diffusion) | structure, **global** | static (diffusion precomputed) or stoch (resample teleport set) | Medium | **high** — local-vs-global contrast; reaches past missing neighbors | ⭐⭐ **TOP** |
| **Imbalanced depth** (one view 2-layer, other 3–4-layer encoder) | encoder→structure reach | static (or stoch depth) | Low | **high** — deep view reaches farther when neighbors missing (MOBA idea) | ⭐⭐ **TOP** |
| **k-hop subgraph view** (view2 = expanded k-hop graph) | structure, mid | static | Low–Med | med — wider receptive field | ⭐ |
| **Edge dropping + adding** (also insert random edges) | structure, local | stochastic | Low | low | — |
| **Node dropping** (remove random nodes) | structure, local | stochastic | Low | low–med | — |
| **Subgraph sampling** (each view a different sampled subgraph) | structure, local | stochastic | Med | med | — |
| **Random-walk-induced view** (build neighborhood from walks, node2vec-style) | structure, mid | stochastic | Med | med — ties to `path` objective | ⭐ |

---

## B. Feature perturbation (same graph, change inputs)

| View | Axis | Static/Stoch | Difficulty | Miss-rel | Priority |
|---|---|---|---|---|---|
| **Prefilled-vs-raw** (view1 FP-prefilled, view2 zero-filled) | feature | static | Low | **high** — directly exploits our prefill; missingness-specific | ⭐⭐ **TOP** |
| **Feature masking** (zero random feature dims in one view) | feature | stochastic | Low | low–med | ⭐ |
| **Feature-dim dropout / Gaussian noise** | feature | stochastic | Low | low | — |
| **Feature shuffling across nodes** (corruption view, DGI-style) | feature | stochastic | Low | low–med — enables MI/DGI contrast | ⭐ |

---

## C. Encoder-side (same graph+features, different network)

| View | Axis | Static/Stoch | Difficulty | Miss-rel | Priority |
|---|---|---|---|---|---|
| **Dropout-as-view** (two stochastic forward passes, different dropout masks) | encoder | stochastic | trivial | low–med — cheapest diversity source | ⭐ |
| **Different aggregators** (one GCN, one GAT/SAGE) | encoder | static | Med | med | — |
| **EMA target network** (BYOL-style slow copy) | encoder | static (EMA) | Med | med — no-negatives contrast | — |

---

## D. Adaptive / learned

| View | Axis | Static/Stoch | Difficulty | Miss-rel | Priority |
|---|---|---|---|---|---|
| **GCA-style adaptive augmentation** (drop unimportant edges/feats by importance) | adaptive | stochastic | Med–High | med | — |
| **Augmentation-free (AFGRL)** (view = kNN in embedding space, no perturbation) | adaptive | dynamic | High | med — augmentation-free stance | — |
| **Spectral views** (perturb in graph-frequency domain) | adaptive | static | High | low–med | — |

---

## ⭐ Top 5 to test (spanning the meaningful axes)

Chosen to (a) cover distinct perturbation types and (b) maximize missingness-relevance:

1. **PPR / diffusion view** ⭐⭐ — the standout. Local (1-hop) vs global (diffused) contrast;
   reaches past missing neighbors. What MATE originally used; MVGRL uses diffusion views.
2. **Imbalanced depth (shallow vs deep encoder)** ⭐⭐ — cheap; deep view reaches farther under
   missingness (MOBA's critique of shallow encoders). Different axis than #1 (encoder, not graph).
3. **Prefilled-vs-raw** ⭐⭐ — missingness-specific and nearly free; directly leverages our FP-prefill.
   Tests whether contrasting "imputed vs unimputed" is useful signal.
4. **Feature masking** ⭐ — the standard feature-perturbation baseline; needed to represent the
   feature-perturb cell and compare against structure-perturb views.
5. **Dropout-as-view** ⭐ — the cheapest possible diversity (no graph/feature change); a good
   floor to check whether view *diversity* alone helps, isolating it from *what* is perturbed.

**Rationale:** #1–#3 are the high-missingness-relevance bets (global reach / prefill exploitation);
#4–#5 are cheap controls that fill the feature-perturb and encoder-perturb cells so the study
spans all four axes rather than clustering on structure.

**Note:** the current edge-mask default is the baseline to beat / compare against for all of these.
