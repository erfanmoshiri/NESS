# Detailed Categories — Related-Work Structure & Positioning Guide

Purpose: a **writing scaffold** for the Related Work section. Not prose — item-wise
hints. For each (sub)category: *what it is → which papers to discuss (cited) → their
limitation → a one-line takeaway that funnels toward our contribution.*

**Funnel storyline (top level):**
1. Scope the problem (what/how-much is missing) → we address *attribute-missing, features-only*.
2. §1 — How prior work *handles* missing features (4 paradigms) → these are our **baselines**.
3. §2 — Graph SSL & multi-view learning (objective/view design) → this is our **camp**; ends on the gap.
4. The gap → nobody systematically studies *which SSL objective/view helps under missingness, and why* → our RQ3/RQ4.

Keep §1 efficient (establish paradigms, cite, move on). Spend depth budget on §2, especially
differentiating from our 3 nearest neighbors: **AmGCL, MATE, MOBA**.

**Citation rule:** each (sub)category should cite **as many relevant papers as apply** — never
limit to one per bucket. Group multiple works with a shared limitation into one sentence
(e.g. "generative imputation methods [SAT, SVGA, GAIN, GINN] overfit and do not scale"). The
paper lists below are minimums, not caps — add any additional relevant works you find.

---

## § 0 — Problem scoping (1 short paragraph, not a full category)

- **What:** set two axes — *what is missing* (features / structure / labels) and *how much*
  (attribute-**incomplete** = some elements per node; attribute-**missing** = whole vectors of some nodes).
- **Position us:** we address **attribute-missing, features-only**. Acknowledge and scope OUT the rest:
  joint feature+structure+label deficiency (GLWI) [D2PT], joint feature+structure [T2-GNN],
  label-scarcity, and structure-only missing (GSL) — one sentence each, cite, move on.
- **Takeaway line:** "We focus on node classification when entire node feature vectors are missing;
  structure and labels are assumed observed."

---

## § 1 — How prior work handles missing features (the paradigm axis = our baselines)

### §1.1 Imputation-based
- **What:** fill missing feature values, then run a standard GNN. Two flavors below.
- **§1.1a statistical / propagation:** discuss **FP** [FP], NeighAggre, KNN. Cheap, scalable, strong.
  - *Limitation:* purely local/linear; no learning of what to impute; can wash out at high missingness.
- **§1.1b generative / learned:** discuss **SAT** [SAT], **SVGA** [SVGA], **GAIN** [GAIN], **GINN** [GINN],
  and matrix-completion roots (**GC-MC** [GCMC], **IGMC** [IGMC]).
  - *Limitation:* high-capacity, overfit under heavy missingness; most **don't scale** (dense ops,
    per-node params) — SAT/SVGA OOM beyond ~10²–10⁵ nodes.
- **Takeaway line:** "Imputation quality ≠ downstream utility, and the learned variants do not scale;
  we evaluate downstream classification directly and at larger scale."

### §1.2 Imputation-free / architectural
- **What:** modify aggregation so the GNN tolerates missing entries without ever filling them.
- **Papers:** **GCNMF** [GCNMF] (Gaussian-mixture expected activation), **PaGCN** [PaGCN] (masked
  partial aggregation), **PCFI** [PCFI] (confidence-based).
- *Limitation:* GCNMF has K·N·D memory blow-up (doesn't scale); both are supervised-only —
  no self-supervision to exploit unlabeled structure.
- **Takeaway line:** "These handle missingness inside the forward pass but add no self-supervised
  signal; PaGCN is our strongest scalable baseline in this camp."

### §1.3 Per-node parameterization / learned per-node fill
- **What:** give each missing node its own learnable embedding/parameters, or learn a per-node
  fill via random-walk/refinement rather than from shared structure.
- **Papers:** **MATE** [MATE] (learnable augmented view), **Amer** [Amer] (per-node embedding),
  **ITR** [ITR]/**RITR** [RITR] (initialize-then-refine per node), random-walk fills
  **GraphRNA** [GraphRNA], **ARWMF** [ARWMF].
- *Limitation:* the per-node quantities receive **no direct supervision → memorize rather than
  generalize**; sensitive to noisy initialization (MOBA's critique of MATE); refinement/random-walk
  variants degrade in the fully-missing regime.
- **Takeaway line:** "Per-node fills don't generalize under missingness — motivating learning
  from shared structure instead of memorizing per node."

### §1.4 Structure-leaning / weak-information
- **What:** rely on topology (long-range propagation) when features are weak/absent.
- **Papers:** **D2PT** [D2PT] (weak info: features+structure+labels), GSL line (IDGL [IDGL],
  Pro-GNN [ProGNN]); feature-structure interference framing (AM-GCN [AMGCN], Yang22 [Yang22]).
- *Limitation:* target a broader/different problem (joint deficiency, structure learning); not
  specialized to attribute-missing feature learning.
- **Takeaway line:** "Structure can substitute for missing features — a principle we exploit via a
  structure-based SSL objective rather than whole-model redesign."

---

## § 2 — Graph SSL & multi-view learning (our camp — go deep, end on the gap)

> Note: split into (a) **technique neighbors** (general graph SSL, mostly on complete graphs — cited
> for objective/view lineage) and (b) **problem neighbors** (SSL specifically for missing graphs).
> Label the distinction explicitly so reviewers see we know both.

### §2.1 Contrastive / multi-view graph SSL (technique lineage)
- **What:** learn representations by agreement across views / mutual-information maximization.
- **Papers by objective level:** global-local **DGI** [DGI]; local-local **GMI** [GMI]; multi-view
  **MVGRL** [MVGRL]; augmentation-based **GRACE**/**GCA** [GCA]; augmentation-free **AFGRL** [AFGRL],
  **MA-GCL** [MAGCL]; redundancy-reduction **Barlow Twins** [BT] / **DCLN** [DCLN].
- *Limitation:* designed for **fully-observed** graphs; views come from generic augmentations, not
  tuned for missingness; no analysis of which objective helps when features are absent.
- **Takeaway line:** "Graph SSL objectives are well-studied on complete graphs but untested as tools
  for *missing-feature* learning — and their view construction is generic."

### §2.2 Masked / reconstruction SSL (technique lineage)
- **What:** artificially mask inputs and reconstruct them as a pretext task.
- **Papers:** **GraphMAE** [GraphMAE], masked GAE ("What's Behind the Mask") [WBM], RARE [RARE].
- *Limitation:* masking is a *pretext* on complete data; reconstruction of features is redundant with
  what message-passing already aggregates (our own finding: recon objective plateaus, no lift).
- **Takeaway line:** "Masked feature reconstruction is a natural pretext but, we show, largely
  redundant with aggregation under missingness."

### §2.3 SSL / multi-view for attribute-missing graphs (problem neighbors — CLOSEST)
- **What:** SSL/contrastive/multi-view methods built specifically for missing-attribute graphs.
- **Papers (differentiate hardest here):**
  - **AmGCL** [AmGCL] — first contrastive (BYOL-style) for attribute-missing + Dirichlet-energy precoder.
    *Limit:* single contrastive objective; no study of *which* objective/view; small-scale.
  - **MATE** [MATE] — multi-view via input-space augmentation (structure+attribute) + consistency.
    *Limit:* per-node/augmentation memorization; shallow encoders starve under heavy missingness.
  - **MOBA/MVCL** [MOBA] — critiques MATE; reliable high-order-neighbor augmentation + variable-depth
    (imbalanced) encoders + redundancy reduction. *Limit:* still fixed view design; no principled
    account of *why* a view/objective works; not evaluated at large scale.
  - **AIAE** [AIAE] — masked-GAE dual-encoder + distillation imputation. *Limit:* reconstruction-centric;
    imputation-quality framing, not downstream-under-severe-missingness.
  - **HGCA** [HGCA] (heterogeneous) — adjacent, mention briefly.
  - Our prior conference work **NCSSL** [NCSSL] — neighborhood-centric SSL; the base we extend.
- *Collective limitation:* each proposes **one** objective/view and tunes it; **none systematically
  studies which SSL objective and which view help under missingness, or why** — and results are at
  small scale, single missing-rate.
- **Takeaway line (the gap):** "Prior SSL-for-missing-graphs works each fix one objective/view;
  the field lacks a principled, empirical answer to *which* SSL objective and view help under
  missingness and *why* — which we provide, at scale, across missingness rates."

### §2.4 (Optional, brief) Adjacent communities — one sentence each
- Incomplete multi-view **clustering**: COMPLETER [COMPLETER], MCGC [MCGC] — same missingness+contrastive
  problem, different task.
- Hypergraph: SGHFP [SGHFP]. MRF inference: SVGA's graphical-model view. Fairness/heterogeneous:
  brief mentions. *Don't make these full categories — footnote-level.*

---

## § 3 — Positioning summary (the last paragraph of Related Work)

- One paragraph tying it together: imputation (§1.1) trades on reconstruction not utility and doesn't
  scale; architectural (§1.2) and per-node (§1.3) add no self-supervision or memorize; structure-leaning
  (§1.4) targets a different problem; graph SSL (§2.1–2.2) is untested under missingness; SSL-for-missing
  (§2.3) fixes a single objective/view without analysis.
- **Our contribution sentence:** we (i) evaluate downstream classification under missingness at scale,
  (ii) systematically study which SSL objectives and views help and why (RQ3/RQ4), finding stochastic,
  non-redundant structure objectives are the effective ones, and (iii) show the effective objective
  transfers to other backbones.

---

## References

> Verify venue/year/authors against originals before manuscript use; details reconstructed from
> mined papers may contain minor errors. Keys match the [BRACKET] tags above.

[FP] Rossi, E., et al. "On the Unreasonable Effectiveness of Feature Propagation in Learning on Graphs with Missing Node Features." *LoG*, 2022.
[SAT] Chen, X., et al. "Learning on Attribute-Missing Graphs." *IEEE TPAMI*, 2022.
[SVGA] Yoo, J., Jeon, H., Jung, J., Kang, U. "Accurate Node Feature Estimation with Structured Variational Graph Autoencoder." *KDD*, 2022.
[GAIN] Yoon, J., Jordon, J., van der Schaar, M. "GAIN: Missing Data Imputation using Generative Adversarial Nets." *ICML*, 2018.
[GINN] Spinelli, I., Scardapane, S., Uncini, A. "Missing Data Imputation with Adversarially-trained Graph Convolutional Networks." *Neural Networks*, 2020.
[GCMC] van den Berg, R., Kipf, T. N., Welling, M. "Graph Convolutional Matrix Completion." arXiv:1706.02263, 2017.
[IGMC] Zhang, M., Chen, Y. "Inductive Matrix Completion Based on Graph Neural Networks." *ICLR*, 2020.
[GCNMF] Taguchi, H., Liu, X., Murata, T. "Graph Convolutional Networks for Graphs Containing Missing Features." *Future Generation Computer Systems*, 2021.
[PaGCN] Zhang, ..., Jiang, ..., et al. "Incomplete Graph Learning via Partial Graph Convolutional Network." *IEEE Trans. Artificial Intelligence*, 2024.
[PCFI] Um, D., Park, J., Park, S., Choi, J. Y. "Confidence-Based Feature Imputation for Graphs with Partially Known Features." *ICLR*, 2023.
[MATE] Peng, X., et al. "Multi-view Graph Imputation Network." *Information Fusion* 102:102024, 2024.
[Amer] Jin, D., et al. "Amer: A New Attribute-Missing Network Embedding Approach." *IEEE Trans. Cybernetics*, 2022.
[ITR] Tu, W., et al. "Initializing Then Refining: A Simple Graph Attribute Imputation Network." *IJCAI*, 2022.
[RITR] Tu, W., et al. "Revisiting Initializing Then Refining." *IEEE TNNLS*, 2024.
[GraphRNA] Huang, X., Song, Q., Li, Y., Hu, X. "Graph Recurrent Networks with Attributed Random Walks." *KDD*, 2019.
[ARWMF] Chen, L., Gong, ..., Bruna, J., Bronstein, M. "Attributed Random Walk as Matrix Factorization." *NeurIPS GRL Workshop*, 2019.
[D2PT] Liu, Y., et al. "Learning Strong Graph Neural Networks with Weak Information." *KDD*, 2023.
[IDGL] Chen, Y., Wu, L., Zaki, M. "Iterative Deep Graph Learning for Graph Neural Networks." *NeurIPS*, 2020.
[ProGNN] Jin, W., et al. "Graph Structure Learning for Robust Graph Neural Networks." *KDD*, 2020.
[AMGCN] Wang, X., et al. "AM-GCN: Adaptive Multi-channel Graph Convolutional Networks." *KDD*, 2020.
[Yang22] Yang, L., et al. "Graph Neural Networks Beyond Compromise Between Attribute and Topology." *WWW*, 2022.
[DGI] Veličković, P., et al. "Deep Graph Infomax." *ICLR*, 2019.
[GMI] Peng, Z., et al. "Graph Representation Learning via Graphical Mutual Information Maximization." *WWW*, 2020.
[MVGRL] Hassani, K., Khasahmadi, A. H. "Contrastive Multi-View Representation Learning on Graphs." *ICML*, 2020.
[GCA] Zhu, Y., et al. "Graph Contrastive Learning with Adaptive Augmentation." *WWW*, 2021.
[AFGRL] Lee, N., Lee, J., Park, C. "Augmentation-Free Self-Supervised Learning on Graphs." *AAAI*, 2022.
[MAGCL] Gong, X., et al. "MA-GCL: Model Augmentation Tricks for Graph Contrastive Learning." *AAAI*, 2023.
[BT] Zbontar, J., et al. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction." *ICML*, 2021.
[DCLN] Peng, ..., et al. "Dual Contrastive Learning Network." *IEEE TNNLS*, 2023.
[GraphMAE] Hou, Z., et al. "GraphMAE: Self-Supervised Masked Graph Autoencoders." *KDD*, 2022.
[WBM] Li, Y., et al. "What's Behind the Mask: Understanding Masked Graph Modeling for Graph Autoencoders." *KDD*, 2023.
[RARE] Tu, W., et al. "RARE: Robust Masked Graph Autoencoder." arXiv, 2023.
[AmGCL] Zhang, W., et al. "Feature Imputation of Attribute-Missing Graph via Self-supervised Contrastive Learning." *ACM MM* / arXiv:2305.03741, 2023.
[MOBA] Yu, ..., Li, Yang, Zhang, Song. "Multi-view Collaborative Learning for Graph Attribute Imputation." *Int. J. Machine Learning & Cybernetics* 16(5):3777-3791, 2025.
[AIAE] Xia, J., et al. "Attribute Imputation Autoencoders for Attribute-Missing Graphs." *Knowledge-Based Systems* 291:111583, 2024.
[HGCA] He, D., et al. "Contrastive Attribute Completion for Heterogeneous Graphs." *IEEE TNNLS*, 2022.
[NCSSL] "Neighborhood-Centric Self-Supervised Graph Models for Imputation as a Service." (our prior conference work)
[COMPLETER] Lin, Y., et al. "COMPLETER: Incomplete Multi-view Clustering via Contrastive Prediction." *CVPR*, 2021.
[MCGC] Pan, E., Kang, Z. "Multi-view Contrastive Graph Clustering." *NeurIPS*, 2021.
[SGHFP] "Self-Supervised Guided Hypergraph Feature Propagation for Semi-Supervised Classification with Missing Node Features." *ICASSP*, 2023.
[T2GNN] Huo, C., et al. "T2-GNN: Graph Neural Networks for Graphs with Incomplete Features and Structure via Teacher-Student Distillation." *AAAI*, 2023.
[Survey] Xia, J., et al. "Incomplete Graph Learning: A Comprehensive Survey." *Neural Networks* 190:107682, 2025.
