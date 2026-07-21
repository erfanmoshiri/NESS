# Related Work — Categories to Survey

Scope: **learning on graphs with missing node features** (and the less-studied
**graph feature imputation**). This document maps the landscape so a research
agent can survey each category, find representative + recent papers, and identify
where our work (NESS: FP-prefill + stochastic structure-based SSL) sits.

Our contribution's core is an **analysis of what makes a self-supervised auxiliary
objective (and view) useful under missingness** — so the SSL/contrastive corner is
the most important to survey exhaustively.

---

## Two classification axes (avoid conflating them)

Prior work should be classified on **two independent axes**, not one flat list:

- **Axis A — how missingness is handled** (imputation / work-around / per-node params / structure-based)
- **Axis B — what training signal is used** (supervised / self-supervised / contrastive / distillation)

A method has one label on each axis (e.g. T2-GNN = structure+feature completion [A] via distillation [B]; NESS = structure-based [A] via SSL [B]). A 2-axis taxonomy table is the target deliverable.

Cross-cut both axes by **what is missing** (features / structure / labels / joint) and
**missingness mechanism** (MCAR / MAR / MNAR — most papers only do MCAR; a gap to note).

---

## Axis A — How missingness is handled

### A1. Imputation-based — statistical / propagation
Fill missing features by diffusing/averaging observed ones, then run a standard GNN.
- Representatives: **Feature Propagation (FP)**, NeighAggre, KNN
- Traits: explicit imputed feature matrix; O(E) propagation; strong, cheap, scalable baselines
- Survey goal: latest propagation-based imputation; any that go beyond linear diffusion

### A2. Imputation-based — learned / generative
Autoencoders / VAEs / GANs that reconstruct missing features (often measured by
reconstruction quality: Recall@K / NDCG / RMSE).
- Representatives: **SAT**, **SVGA**, GCNMF-adjacent, GraphRNA, ARWMF
- Traits: heavier, often don't scale (dense N×N ops, per-node params); small-graph focus
- Survey goal: recent generative imputation; scalability claims

### A3. Work-around / imputation-free
Modify the architecture/aggregation to handle missing entries without ever filling them.
- Representatives: **PaGCN** (masked aggregation), **GCNMF** (Gaussian-mixture expected activation)
- Traits: no explicit imputation; GNN-level cost; scalable
- Survey goal: other imputation-free formulations; theoretical framings

### A4. Per-node parameterization
Give each missing node its own learnable embedding/parameters.
- Representatives: **MATE** (learnable per-node feature)
- Traits: risk of memorization (no supervision reaches missing-node params) — our conceptual foil
- Survey goal: other per-node/learnable-fill methods; critiques of memorization

### A5. Structure-based / propagation-heavy
Lean on topology and long-range propagation when features are weak/absent.
- Representatives: **D2PT** (dual-channel diffusion, weak-information GLWI), structure-only methods
- Traits: bypass features via structure; often handle joint missingness; efficiency-oriented
- Survey goal: methods that exploit multi-hop / global structure under missingness

### A6. Hypergraph / higher-order
Model richer connectivity than pairwise edges.
- Representatives: **SGHFP** (feature propagation on hypergraphs)
- Traits: higher-order structure; usually small-graph
- Survey goal: higher-order / motif-based missing-feature methods

---

## Axis B — Training signal (most important corner for us)

### B1. Self-supervised / contrastive for missing graphs  ← OUR CAMP (survey exhaustively)
SSL pretext tasks (contrastive / reconstruction / structure prediction) to learn
missingness-robust representations without relying on imputation quality.
- Representatives: **AmGCL** (contrastive SSL for attribute-missing graphs), AIAE, and few others
- Traits: sparsest category → room to contribute, but must differentiate from closest neighbors
- Survey goal (HIGH PRIORITY):
  - all SSL/contrastive methods specifically for attribute-missing graphs
  - what pretext tasks they use (contrastive? reconstruction? structure?)
  - whether any analyze *which* pretext task/view is best (directly overlaps our RQ3/RQ4)

### B2. Distillation / teacher-student
Teacher(s) provide a learned signal distilled into a student; often self-supervised-ish
and frequently include contrastive alignment (overlaps B1).
- Representatives: **T2-GNN** (feature-teacher + structure-teacher → student)
- Traits: handles feature AND structure missingness jointly; heavier
- Survey goal: other distillation approaches to incomplete graphs

---

## Out-of-graph threads (brief — ~1 paragraph each in the paper)

### O1. General self-supervised representation learning
Because our core finding is about **SSL objective/view design**, position against the
broader SSL literature:
- Masked autoencoders (MAE / BERT-style) — our `recon` objective mirrors this
- Contrastive methods (SimCLR, BYOL, **Barlow Twins** — we use it), and graph SSL (GraphMAE, GRACE, GCA)
- "Which pretext task is good?" analyses — **critical to find**: if anyone framed
  "which SSL objective helps GNNs," we must cite and differentiate (our angle: *under
  missingness*, plus the stochastic/non-redundant-with-aggregation principle)
- Survey goal: SSL pretext-task selection/analysis papers; graph-SSL view-design papers

### O2. Missing-data / imputation in classical ML
A brief nod to ground the terminology:
- MCAR / MAR / MNAR framework (statistics)
- General imputation (MICE, matrix completion, etc.)
- Survey goal: just enough to cite the roots of the missingness taxonomy; not a deep survey

---

## Deliverables wanted from the survey (per category)

For each category above:
1. 2-4 representative + recent papers (title, venue, year, link if possible)
2. One-line method summary each
3. Which axis-A / axis-B labels apply
4. What is missing (features/structure/labels) and which mechanism (MCAR/MAR/MNAR)
5. Max scale demonstrated (node count) — feeds our scalability positioning
6. For B1 especially: what pretext task/view, and do they analyze *which* is best?

**Highest priority:** B1 (SSL/contrastive for missing graphs) and O1 (SSL pretext-task
/ view-design analyses) — these are the closest neighbors to our RQ3/RQ4 contribution
and where we must differentiate most sharply.
