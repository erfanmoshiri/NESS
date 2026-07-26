# Related Work — Parts 2 and 3 (Draft)


## 2. Graph Self-Supervised and Multi-View Learning

Self-supervised learning (SSL) offers an appealing alternative to imputation for
missing-attribute graphs. Rather than reconstructing absent inputs, it learns
representations directly from the abundant unlabeled structure, and its quality is judged by
downstream utility rather than by reconstruction fidelity. Our work sits squarely in this
camp. We therefore review it in depth, separating two strands that are frequently conflated.
The first comprises *technique neighbors*, namely general graph SSL developed on fully
observed graphs, from which we inherit our objective and view vocabulary. The second
comprises *problem neighbors*, namely SSL methods engineered specifically for
attribute-missing graphs, against which we must differentiate most sharply.

### 2.1 Contrastive and multi-view self-supervision

The dominant paradigm learns node representations by maximizing agreement between different
views of a graph, an objective commonly cast as maximizing a mutual-information lower bound.
Existing methods differ chiefly in the granularity at which agreement is enforced and in how
views are produced. Deep Graph Infomax [16] learns node representations by maximizing the
mutual information between patch-level representations and a high-level graph summary,
without recourse to random-walk objectives, and applies to both transductive and inductive
settings. GMI [17] instead maximizes graphical mutual information between each node's hidden
representation and its neighborhood, measured over both node features and topology, so that
the signal is tied to local structure. MVGRL [15] contrasts representations learned from two
structural views, an adjacency view and a graph-diffusion view; notably, its authors report
that increasing the number of views beyond two or contrasting multi-scale encodings does not
help, and that the strongest results come from contrasting first-order neighbors against a
diffusion view. A large family instead manufactures views through stochastic input
perturbations: GRACE [18] generates two views by corrupting structure and attributes and
contrasts them at the node level, and its successor GCA [19] makes these augmentations
adaptive by perturbing unimportant edges and features more aggressively according to
centrality and attribute-importance priors. GraphCL [21] catalogs four families of graph
augmentation and studies their combinations, and subsequent work automates the choice of
augmentation because it otherwise has to be tuned per dataset. Because handcrafted
augmentation can distort a graph's semantics, a parallel line reduces or removes it: AFGRL
[25] builds an alternative view without augmentation by discovering nodes that share local
structure and global semantics with a target node, and MA-GCL [20] perturbs the view
encoders' architectures (through asymmetric, random, and shuffling tricks) rather than the
graph inputs. A further group forgoes negative pairs in favor of redundancy reduction,
decorrelating embedding dimensions in the spirit of Barlow Twins [24], whose graph
counterpart is realized by DCLN [26]. Collectively, these methods establish that the choice
of objective and view is consequential, and that generic augmentations are a fragile design
axis. Yet all are developed and validated on fully observed graphs, their views are drawn
from augmentations that are not tuned to the absence of features, and none asks which
objective remains informative once whole attribute vectors are missing, which is precisely
the question we take up.

### 2.2 Masked and reconstruction-based self-supervision

A second pretext family corrupts part of the input and trains the model to restore it, an
idea imported from masked modeling in language and vision. The variants differ in *what*
they mask. GraphMAE [22] focuses on feature reconstruction, masking node features and
restoring them with a scaled cosine error together with a re-masking strategy, and reports
that these choices let a simple autoencoder match or exceed contrastive baselines. MaskGAE
[29] instead masks graph *structure*, adopting masked edge modeling as its pretext task and
reconstructing the removed edges from the visible ones, with theory relating this objective
to contrastive learning. RARE [30] adds robustness by masking and predicting in the latent
space rather than the input space. These objectives are attractive because they avoid
negative sampling and transfer a recipe that has proven powerful elsewhere. However, masking
is a pretext imposed on otherwise complete data. When attributes are genuinely missing, the
feature-reconstruction signal in particular overlaps heavily with what neighborhood
aggregation already recovers, so the pretext adds little beyond what the encoder computes.
Consistent with this observation, we find that a feature-reconstruction objective plateaus
early and yields no downstream gain under missingness, indicating that faithful
reconstruction and useful representation are not the same goal in this regime.

### 2.3 Self-supervision for attribute-missing graphs

Closest to our work are SSL and multi-view methods built specifically for attribute-missing
graphs. AmGCL [14] handles missing attributes in two stages: a Dirichlet-energy-minimization
feature precoder encodes information into the missing entries, and a self-supervised graph
augmentation contrastive module then learns representations by maximizing a lower bound on
the mutual information of the latent representation, alongside a structure-attribute
energy-based feature reconstruction. It nonetheless commits to a single contrastive design,
offers no analysis of which objective or view is responsible for its gains, and is evaluated
only at small scale. MATE [7] pursues a multi-view route, constructing complementary views
of the graph and enforcing agreement across them for attribute imputation; because
supervision does not directly reach its per-node augmented quantities, such designs are
prone to memorization, and shallow encoders can be starved of signal under heavy missingness.
MOBA [8] follows the same multi-view philosophy but explicitly targets these weaknesses,
adopting a collaborative learning strategy that reduces redundant information between views
while maximizing their consistency. Even so, its view design is fixed a priori, it provides
no principled account of why a particular view or objective succeeds, and it is not
demonstrated at large scale. AIAE [9] takes a reconstruction-centric, autoencoder-based
route to attribute imputation and is framed around imputation quality rather than downstream
performance under severe missingness. Beyond the homogeneous setting, HGCA [11] unifies
attribute completion and representation learning for heterogeneous graphs through an
unsupervised contrastive objective, using an augmented network to complete missing
attributes at a fine granularity. Our own prior conference work [71] introduced a
neighborhood-centric self-supervised formulation, which the present study substantially
extends. Across all of these, the pattern is the same: each method fixes a single objective
and view and tunes it, and none systematically investigates which self-supervised objective
and which view are helpful under missingness, or why. Reported results are further confined
to small graphs and, typically, a single missing rate, leaving behavior across the
missingness spectrum and at scale unexamined.

### 2.4 Adjacent communities

Two neighboring literatures pair missingness with contrastive learning under a different
task or connectivity model. Incomplete multi-view clustering shares our combination of absent
information and contrastive objectives but targets clustering rather than node
classification: COMPLETER [28] treats the views of an incomplete multi-view dataset with a
contrastive-prediction objective, and MCGC [52] learns a consensus graph with a
contrastive regularizer precisely because the observed graph may be noisy or incomplete. On
the higher-order side, SGHFP [51] propagates features over hypergraphs to cope with missing
attributes. We note these connections but do not treat them as central, as they differ from
our problem in task or in graph model.

---

## 3. Positioning of Our Work

Viewed together, the prior art leaves a specific opening. Imputation methods, whether
statistical propagation or learned generative models, optimize reconstruction rather than
downstream utility, and their learned variants do not scale. Imputation-free architectural
and per-node approaches either add no self-supervisory signal or memorize node-specific
quantities that fail to generalize. Structure-leaning methods succeed by exploiting topology
but are designed for a broader joint-deficiency problem. General graph SSL is powerful yet
untested as a tool for missing-feature learning, and its own literature shows that view
construction is a fragile, largely handcrafted design axis. Finally, the SSL methods built
for attribute-missing graphs each commit to a single objective and view, without asking which
choices matter or why, and report results only at small scale and a single missing rate. Our
work addresses this gap on three fronts. First, we evaluate representations by downstream
node-classification accuracy under missingness, and do so at substantially larger scale than
prior attribute-missing studies and across a range of missing rates. Second, we conduct a
systematic study of which self-supervised objectives and which views are beneficial under
missingness and offer an explanation for the pattern we observe, namely that stochastic,
structure-based objectives whose signal is not redundant with neighborhood aggregation are
the effective ones, whereas feature-reconstruction objectives are not. Third, we show that
the effective objective is not idiosyncratic to our encoder, since it transfers to other
backbones, indicating that the principle, rather than a specific architecture, drives the
gains.

---
# Verification status

| Ref | Method | Status | Note on what the source supports |
|-----|--------|--------|----------------------------------|
| [16] | DGI | verified (abstract) | Patch-vs-summary mutual information; no random walks; transductive + inductive. |
| [17] | GMI | verified (abstract) | Graphical MI between node representation and neighborhood, over features + topology. |
| [15] | MVGRL | verified (abstract) | Adjacency + diffusion views; authors state >2 views / multi-scale does **not** help. Corrected from first draft. |
| [18] | GRACE | verified (abstract) | Two views by corruption; node-level contrast; structure + attribute corruption. |
| [19] | GCA | verified (abstract) | Adaptive augmentation via centrality (topology) and attribute-importance priors. |
| [21] | GraphCL | verified (abstract) | Four augmentation families + study of combinations. |
| [25] | AFGRL | verified (abstract) | Augmentation-free; alternative view from nodes sharing local structure + global semantics. |
| [20] | MA-GCL | verified (abstract) | Perturbs encoder architectures (asymmetric/random/shuffling), not inputs. |
| [24] | Barlow Twins | verified (abstract) | Redundancy reduction via cross-correlation to identity; a vision method. |
| [26] | DCLN | **needs full-text check** | Cited as the graph redundancy-reduction counterpart; not re-verified from abstract this pass. |
| [22] | GraphMAE | verified (abstract) | **Feature** reconstruction, masking + scaled cosine error + re-mask. |
| [29] | MaskGAE ("What's Behind the Mask") | verified (abstract) | Masks **edges/structure**, not features. Corrected from first draft. |
| [30] | RARE | **needs full-text check** | Described as latent-space masked prediction; confirm exact mechanism. |
| [14] | AmGCL | verified (abstract) | Dirichlet-energy precoder + contrastive module maximizing MI lower bound + energy-based feature recon. "BYOL-style" from your notes is **not** supported by the abstract; dropped. |
| [7] | MATE | **needs full-text check** (paywalled, no abstract) | Multi-view / attribute imputation described conservatively; verify the per-node-memorization and shallow-encoder critiques against the paper (they trace to MOBA's framing). |
| [8] | MOBA | verified (TLDR) | Multi-view collaborative learning: reduce redundancy + maximize consistency between two views. "Not at large scale" and "fixed view design" are your positioning claims — confirm. |
| [9] | AIAE | **needs full-text check** (paywalled, no abstract) | Autoencoder-based attribute imputation; verify the "imputation-quality framing" characterization. |
| [11] | HGCA | verified (abstract) | Heterogeneous; unifies attribute completion + representation learning via unsupervised contrastive; augmented network for fine-grained completion. |
| [28] | COMPLETER | **needs full-text check** | Incomplete multi-view clustering via contrastive prediction; abstract not re-fetched this pass. |
| [52] | MCGC | verified (abstract) | Learns a consensus graph with contrastive regularizer because the graph may be noisy/incomplete. |
| [51] | SGHFP | **needs full-text check** | From your notes only; no abstract fetched. |
| [71] | NCSSL (your prior work) | not in master list | Needs a bibliography entry; cite in third person if under double-blind review. |

**Two claims that are yours, not the cited papers', and must match your experiments:**
the statement in §2.2 and §3 that a feature-reconstruction objective "plateaus early and
yields no downstream gain under missingness," and the §3 claim that stochastic,
non-redundant structure objectives are the effective ones and transfer across backbones.
These should read exactly as your RQ3/RQ4 results report them.
