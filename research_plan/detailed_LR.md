# Detailed Literature Review — Findings from Mining Related-Work Sections

Compiled from the related-work AND introduction sections of 14 PDFs in `papers/`.
Purpose: (1) surface categories we hadn't considered, (2) build a per-category
reading list of citations to chase, (3) identify our closest competitors.

Our contribution's core: an analysis of **which SSL auxiliary objectives (and which
contrastive views) help learning under missingness, and why** (RQ3/RQ4). So the
SSL/contrastive + view-design literature is the highest priority.

---

## PART 1 — NEW categories / framings we had NOT considered

Our original 8: statistical imputation, generative imputation, work-around/imputation-free,
per-node-parameterization, structure-based, hypergraph, SSL/contrastive, distillation.

Newly surfaced (add or at least acknowledge these):

1. **Attribute-*incomplete* vs. attribute-*missing* — the primary top-level split.**
   Incomplete = some feature *elements* missing per node (tabular-style). Missing =
   *entire* feature vectors of some nodes absent. Nearly every paper (AIAE, MOBA, ITR,
   SVGA) organizes around this axis. We handle attribute-**missing** (node-level). Adopt
   this split explicitly and scope our problem.

2. **Missing *labels* / label-efficient (few-label) learning.** A distinct missingness
   axis we ignored (we only consider missing features). [D2PT: IGCN, M3S, CGPN, Meta-PN]

3. **Missing *structure* → Graph Structure Learning (GSL).** Distinct from structure-based
   *feature* imputation. [D2PT, T2-GNN: IDGL, Pro-GNN, LDS, GEN]

4. **"Graph Learning with Weak Information" (GLWI) / joint multi-deficiency.** Structure +
   features + labels all missing at once (D2PT), or features + structure jointly (T2-GNN).
   We should scope that we address features-only.

5. **Feature–structure "mutual interference".** Completing features and structure can *hurt
   each other*; motivates decoupling them. [T2-GNN: AM-GCN (Wang 2020), Yang et al. 2022 (WWW)]

6. **Matrix completion / low-rank** as a distinct classical lineage. [AmGCL, GCNMF, SAT, ITR]

7. **Masked graph autoencoders (masked-SSL).** GraphMAE lineage — sits between generative and
   SSL; our `recon` objective mirrors this. [AIAE, MOBA, NCSSL: GraphMAE, RARE, "What's Behind the Mask"]

8. **Probabilistic-graphical-model / MRF inference.** Belief-propagation lineage generalized
   to feature estimation. [SVGA — its own framing]

9. **Structure-only feature construction for non-attributed graphs.** Build features from
   structure when there were never any. [SVGA: Derr signed-GCN, Cui, Duong "On Node Features", NENN]

10. **Graph / structure generation.** MolGAN, GraphRNN, NetGAN, JT-VAE — generate structure,
    not features. Adjacent neighbor. [SAT]

11. **Deep graph clustering.** DFCN, DCRN. [ITR — authors' own line]

12. **Incomplete/absent multi-view *clustering*.** COMPLETER (contrastive prediction on
    incomplete multi-view) — a different community with the same missingness+contrastive problem. [MCGC, MATE]

13. **Augmentation-free SSL.** AFGRL — directly relevant since our method is augmentation-free. [MATE intro]

14. **Contrastive-objective *level* taxonomy** (important for RQ4): global-local (DGI),
    local-local (GMI), instance vs cluster (Contrastive Clustering), graph-level (MCGC),
    redundancy-reduction/decorrelation (Barlow Twins / DCLN / MOBA). A cleaner way to organize
    SSL prior work than one flat "SSL/contrastive" bucket.

15. **Classical non-graph missing-data ML** (brief, for roots): MICE, MissForest, GMMC,
    soft-impute. [GCNMF]

16. **Adversarial-robustness GNNs (RGCN).** Adjacent robustness angle. [GCNMF]

---

## PART 2 — Our CLOSEST competitors (must differentiate sharply)

- **AmGCL (Zhang et al., ACM MM / arXiv 2023)** — *first* SSL contrastive method (BYOL-style)
  for attribute-missing graphs + a Dirichlet-energy feature precoder. **Our nearest neighbor.**
  Differentiator: we analyze *which objective/view* works and why; not just "apply contrastive."
- **MATE (Peng et al., Information Fusion 2024)** — multi-view imputation; builds a complete
  view in *input* space (dual augmentation: structure + attribute) then multi-view consistency.
  Our conceptual foil for per-node/augmentation approaches.
- **MOBA / MVCL (Yu et al., IJMLC 2025)** — explicitly critiques MATE: (1) noisy augmentation
  init, (2) insufficient neighbor extraction with shallow encoders. Proposes reliable
  high-order-neighbor augmentation + *imbalanced (variable-depth) encoders* + redundancy-reduction.
  **Directly overlaps our RQ4 (view design) and the FP-prefill/depth issues we hit.** Highest-priority read.
- **NCSSL (our own paper)** — neighborhood-centric SSL (predict neighborhood statistics +
  node deviations) + pretrain-and-adapt service setting. The journal extension builds on this.
- **AIAE (Xia et al., KBS 2024)** — masked-GAE dual-encoder + distillation imputation. Make sure
  it is in our baseline table (cited in NCSSL, absent from MATE/MOBA).

**Survey to read first:** *Incomplete Graph Learning: A Comprehensive Survey* — Xia et al.,
Neural Networks 190:107682, 2025. (Cited in NCSSL [1]; the field's map.)

---

## PART 3 — Reading list by category (deduplicated; ★ = recurring canonical anchor)

### Attribute-missing graphs (core competitors)
- ★ **SAT** — Chen et al., "Learning on Attribute-Missing Graphs," IEEE TPAMI 2022 (founding work; shared latent space + distribution matching)
- ★ **SVGA** — Yoo et al., "Accurate Node Feature Estimation with Structured VGAE," KDD 2022 (GMRF regularizer)
- ★ **ITR** — Tu et al., "Initializing Then Refining," IJCAI 2022 (prior-free; affinity refine)
- **RITR** — Tu et al., "Revisiting Initializing Then Refining," IEEE TNNLS 2024
- **Amer** — Jin et al., "A New Attribute-Missing Network Embedding Approach," IEEE Trans. Cybern. 2022
- **MATE** — Peng et al., "Multi-view Graph Imputation Network," Information Fusion 2024
- **MOBA/MVCL** — Yu et al., "Multi-view Collaborative Learning for Graph Attribute Imputation," IJMLC 2025
- **AIAE** — Xia et al., "Attribute Imputation Autoencoders," KBS 2024
- Heterogeneous: **HGNN-AC** (Jin et al., WWW 2021); **HGCA** (He et al., IEEE TNNLS 2022); HetReGAT-FC (Li et al., Inf. Sci. 2023)
- Fairness: **Fair attribute completion** (Guo, Chu, Li, ICLR 2023)

### SSL / contrastive for graphs (RQ3/RQ4 core — differentiate here)
- **AmGCL** — Zhang et al., ACM MM 2023 (contrastive, attribute-missing) ← nearest competitor
- ★ **MVGRL** — Hassani & Khasahmadi, ICML 2020 (diffusion second view; multi-view contrast)
- **DGI** — Veličković et al., 2018 (global-local mutual information)
- **GMI** — Peng et al., WWW 2020 (local-local mutual information)
- **GRACE** — Zhu et al., 2020; **GCA** — Zhu et al., WWW 2021 (adaptive augmentation)
- **MA-GCL** — Gong et al., AAAI 2023 (model augmentation, augmentation-free-ish)
- **GraphCL** — You et al., NeurIPS 2020 (augmentations)
- **GraphMAE** — Hou et al., KDD 2022 (masked autoencoder SSL; source of SCE loss)
- **BYOL** — Grill et al., NeurIPS 2020; **Barlow Twins** — Zbontar et al., 2021 (we use it)
- **AFGRL** — Lee et al., AAAI 2022 (augmentation-free SSL)
- **DCLN** — Peng et al., IEEE TNNLS 2023 (dual contrastive, redundancy reduction)
- **GCC** — Qiu et al., KDD 2020 (contrastive pre-training)
- **COMPLETER** — Lin et al., CVPR 2021 (contrastive prediction on *incomplete* multi-view)
- ITR's framing: generative/predictive vs contrastive GRL

### Imputation-free / work-around
- ★ **GCNMF** — Taguchi, Liu, Murata, FGCS 2021 (Gaussian mixture, expected activation)
- **PaGCN** — Zhang, Jiang et al., IEEE TAI 2024 (masked partial aggregation; energy-minimization derivation)
- **PCFI / confidence-based** — Um et al., ICLR 2023

### Statistical / propagation imputation
- ★ **FP (Feature Propagation)** — Rossi et al., LoG 2022 (the strong scalable baseline; our prefill)
- **NeighAggre**, KNN (classic profiling baselines)

### Generative imputation
- ★ **GAIN** — Yoon et al., ICML 2018 (GAN imputation)
- **GINN** — Spinelli, Scardapane, Uncini, Neural Networks 2020 (adversarial GCN imputation)
- **GRAPE** — You et al., NeurIPS 2020 (bipartite/edge-level; element-wise — boundary case)
- **IGRM** — Zhong et al., AAAI 2023

### Distillation
- **T2-GNN** — Huo et al., AAAI 2023 (feature-teacher + structure-teacher → student; handles joint missingness)
- KD roots: Hinton et al. 2015; attention transfer (Zagoruyko & Komodakis 2017)

### Structure-based / weak-information / GSL
- **D2PT** — Liu et al., KDD 2023 (weak information: structure+features+labels; dual-channel diffusion + kNN global graph)
- GSL: IDGL (Chen 2020), Pro-GNN (Jin 2020), LDS (Franceschi 2019), GEN (Wang 2021), SimP-GCN (Jin 2021)
- Feature-structure interference: AM-GCN (Wang 2020), Yang et al. 2022 (WWW)

### Random-walk feature completion (older baselines)
- **GraphRNA** — Huang et al., KDD 2019
- **ARWMF** — Chen, Gong, Bruna, Bronstein, NeurIPS-W 2019

### Hypergraph
- **SGHFP** — self-supervised guided hypergraph feature propagation, ICASSP 2023
  (NOTE: no other paper we mined cites hypergraph methods — this category is isolated)

### Matrix completion / classical
- GC-MC (van den Berg, Kipf, Welling 2017); IGMC (Zhang & Chen 2020); Monti geometric MC (2017);
  soft-impute/SVD (Mazumder 2010); MICE (van Buuren); MissForest (Stekhoven & Bühlmann); GMMC (Śmieja 2018)

### Out-of-graph roots (brief citations only)
- General SSL: SimCLR (Chen 2020), BYOL, Barlow Twins, MAE/BERT-style masking
- Missing-data theory: MCAR/MAR/MNAR framework; Marsden 1990 (network measurement error);
  You et al. 2020 NeurIPS (handling missing graph data)
- GNN foundations: GCN (Kipf & Welling 2017), GAT (Veličković 2018), GraphSAGE (Hamilton 2017),
  APPNP/PPR (Gasteiger 2019), DeepWalk (Perozzi 2014), node2vec (Grover & Leskovec 2016)

---

## PART 4 — Actionable takeaways

1. **Adopt the attribute-incomplete vs attribute-missing split** as the top-level framing;
   scope our work to attribute-missing, features-only (not structure/labels).
2. **Organize SSL prior work by objective *level*** (global-local / local-local / graph-level /
   redundancy-reduction) — this frames RQ3/RQ4 cleanly and shows where our stochastic
   structure objective (`path`) sits.
3. **Differentiate hardest from AmGCL, MATE, MOBA** — all are SSL/multi-view for attribute-missing.
   Our angle: systematic *objective + view* analysis (what works and why) under severe missingness,
   at arxiv scale, label-free.
4. **MOBA is the must-read** — its view-design and encoder-depth critiques directly parallel our
   FP-prefill/2-hop-starvation findings and RQ4 (view design). Cite and position carefully.
5. **Read the Xia et al. 2025 survey first** — it is the field map and will surface anything missed.
6. **Baseline-table additions to verify:** AIAE, PCFI, RITR, MOBA (beyond current NeighAggre/KNN/
   FP/PaGCN/SVGA/SAT/MATE/GraphSAGE).

---

## PART 5 — References

> Compiled from the mined papers' citations. Verify exact venue/year/page details against
> the original sources before using in the manuscript (details reconstructed from
> related-work text may contain minor errors).

### Attribute-missing / attribute-incomplete graph learning
[1] Xia, J., et al. "Incomplete Graph Learning: A Comprehensive Survey." *Neural Networks* 190:107682, 2025.
[2] Chen, X., et al. "Learning on Attribute-Missing Graphs (SAT)." *IEEE TPAMI*, 2022.
[3] Yoo, J., Jeon, H., Jung, J., Kang, U. "Accurate Node Feature Estimation with Structured Variational Graph Autoencoder (SVGA)." *KDD*, 2022.
[4] Tu, W., et al. "Initializing Then Refining: A Simple Graph Attribute Imputation Network (ITR)." *IJCAI*, 2022.
[5] Tu, W., et al. "Revisiting Initializing Then Refining (RITR)." *IEEE TNNLS*, 2024.
[6] Jin, D., et al. "Amer: A New Attribute-Missing Network Embedding Approach." *IEEE Trans. Cybernetics*, 2022.
[7] Peng, X., et al. "Multi-view Graph Imputation Network (MATE)." *Information Fusion* 102:102024, 2024.
[8] Yu, ..., Li, Yang, Zhang, Song. "Multi-view Collaborative Learning for Graph Attribute Imputation (MOBA/MVCL)." *Int. J. Machine Learning & Cybernetics* 16(5):3777-3791, 2025.
[9] Xia, J., et al. "Attribute Imputation Autoencoders for Attribute-Missing Graphs (AIAE)." *Knowledge-Based Systems* 291:111583, 2024.
[10] Jin, D., et al. "Heterogeneous Graph Neural Network via Attribute Completion (HGNN-AC)." *WWW*, 2021.
[11] He, D., et al. "Contrastive Attribute Completion for Heterogeneous Graphs (HGCA)." *IEEE TNNLS*, 2022.
[12] Li, ..., et al. "HetReGAT-FC: Heterogeneous Residual Graph Attention Network via Feature Completion." *Information Sciences*, 2023.
[13] Guo, D., Chu, Z., Li, S. "Fair Attribute Completion on Graph with Missing Attributes." *ICLR*, 2023.

### SSL / contrastive graph learning
[14] Zhang, W., Li, ..., Wang, Fei. "Feature Imputation of Attribute-Missing Graph via Self-supervised Contrastive Learning (AmGCL)." *ACM MM* / arXiv:2305.03741, 2023.
[15] Hassani, K., Khasahmadi, A. H. "Contrastive Multi-View Representation Learning on Graphs (MVGRL)." *ICML*, 2020.
[16] Veličković, P., et al. "Deep Graph Infomax (DGI)." *ICLR*, 2019.
[17] Peng, Z., et al. "Graph Representation Learning via Graphical Mutual Information Maximization (GMI)." *WWW*, 2020.
[18] Zhu, Y., et al. "Deep Graph Contrastive Representation Learning (GRACE)." arXiv:2006.04131, 2020.
[19] Zhu, Y., et al. "Graph Contrastive Learning with Adaptive Augmentation (GCA)." *WWW*, 2021.
[20] Gong, X., et al. "MA-GCL: Model Augmentation Tricks for Graph Contrastive Learning." *AAAI*, 2023.
[21] You, Y., et al. "Graph Contrastive Learning with Augmentations (GraphCL)." *NeurIPS*, 2020.
[22] Hou, Z., et al. "GraphMAE: Self-Supervised Masked Graph Autoencoders." *KDD*, 2022.
[23] Grill, J.-B., et al. "Bootstrap Your Own Latent (BYOL)." *NeurIPS*, 2020.
[24] Zbontar, J., et al. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction." *ICML*, 2021.
[25] Lee, N., Lee, J., Park, C. "Augmentation-Free Self-Supervised Learning on Graphs (AFGRL)." *AAAI*, 2022.
[26] Peng, ..., et al. "Dual Contrastive Learning Network (DCLN)." *IEEE TNNLS*, 2023.
[27] Qiu, J., et al. "GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training." *KDD*, 2020.
[28] Lin, Y., et al. "COMPLETER: Incomplete Multi-view Clustering via Contrastive Prediction." *CVPR*, 2021.
[29] Li, Y., et al. "What's Behind the Mask: Understanding Masked Graph Modeling for Graph Autoencoders." *KDD*, 2023.
[30] Tu, W., et al. "RARE: Robust Masked Graph Autoencoder." arXiv, 2023.

### Imputation-free / work-around
[31] Taguchi, H., Liu, X., Murata, T. "Graph Convolutional Networks for Graphs Containing Missing Features (GCNMF)." *Future Generation Computer Systems*, 2021.
[32] Zhang, ..., Jiang, ..., et al. "Incomplete Graph Learning via Partial Graph Convolutional Network (PaGCN)." *IEEE Trans. Artificial Intelligence*, 2024.
[33] Um, D., Park, J., Park, S., Choi, J. Y. "Confidence-Based Feature Imputation for Graphs with Partially Known Features (PCFI)." *ICLR*, 2023.

### Statistical / propagation / random-walk
[34] Rossi, E., et al. "On the Unreasonable Effectiveness of Feature Propagation in Learning on Graphs with Missing Node Features (FP)." *LoG*, 2022.
[35] Huang, X., Song, Q., Li, Y., Hu, X. "Graph Recurrent Networks with Attributed Random Walks (GraphRNA)." *KDD*, 2019.
[36] Chen, L., Gong, ..., Bruna, J., Bronstein, M. "Attributed Random Walk as Matrix Factorization (ARWMF)." *NeurIPS GRL Workshop*, 2019.

### Generative imputation
[37] Yoon, J., Jordon, J., van der Schaar, M. "GAIN: Missing Data Imputation using Generative Adversarial Nets." *ICML*, 2018.
[38] Spinelli, I., Scardapane, S., Uncini, A. "Missing Data Imputation with Adversarially-trained Graph Convolutional Networks (GINN)." *Neural Networks*, 2020.
[39] You, J., Ma, X., Ding, Y., Kochenderfer, M., Leskovec, J. "Handling Missing Data with Graph Representation Learning (GRAPE)." *NeurIPS*, 2020.
[40] Zhong, J., et al. "IGRM: Iterative Graph Representation Learning for Missing Data Imputation." *AAAI*, 2023.

### Distillation
[41] Huo, C., et al. "T2-GNN: Graph Neural Networks for Graphs with Incomplete Features and Structure via Teacher-Student Distillation." *AAAI*, 2023.
[42] Hinton, G., Vinyals, O., Dean, J. "Distilling the Knowledge in a Neural Network." arXiv:1503.02531, 2015.

### Structure-based / weak-information / GSL
[43] Liu, Y., et al. "Learning Strong Graph Neural Networks with Weak Information (D2PT)." *KDD*, 2023.
[44] Chen, Y., Wu, L., Zaki, M. "Iterative Deep Graph Learning for Graph Neural Networks (IDGL)." *NeurIPS*, 2020.
[45] Jin, W., et al. "Graph Structure Learning for Robust Graph Neural Networks (Pro-GNN)." *KDD*, 2020.
[46] Franceschi, L., et al. "Learning Discrete Structures for Graph Neural Networks (LDS)." *ICML*, 2019.
[47] Wang, R., et al. "Graph Structure Estimation Neural Networks (GEN)." *WWW*, 2021.
[48] Jin, W., et al. "Node Similarity Preserving Graph Convolutional Networks (SimP-GCN)." *WSDM*, 2021.
[49] Wang, X., et al. "AM-GCN: Adaptive Multi-channel Graph Convolutional Networks." *KDD*, 2020.
[50] Yang, L., et al. "Graph Neural Networks Beyond Compromise Between Attribute and Topology." *WWW*, 2022.

### Hypergraph
[51] "SGHFP: Self-Supervised Guided Hypergraph Feature Propagation for Semi-Supervised Classification with Missing Node Features." *ICASSP*, 2023.

### Multi-view contrastive clustering (near-neighbor community)
[52] Pan, E., Kang, Z. "Multi-view Contrastive Graph Clustering (MCGC)." *NeurIPS*, 2021.
[53] Li, Y., et al. "Contrastive Clustering." *AAAI*, 2021.

### Matrix completion / classical missing data
[54] van den Berg, R., Kipf, T. N., Welling, M. "Graph Convolutional Matrix Completion (GC-MC)." arXiv:1706.02263, 2017.
[55] Zhang, M., Chen, Y. "Inductive Matrix Completion Based on Graph Neural Networks (IGMC)." *ICLR*, 2020.
[56] Monti, F., Bronstein, M., Bresson, X. "Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks." *NeurIPS*, 2017.
[57] Mazumder, R., Hastie, T., Tibshirani, R. "Spectral Regularization Algorithms for Learning Large Incomplete Matrices (Soft-Impute)." *JMLR*, 2010.
[58] van Buuren, S., Groothuis-Oudshoorn, K. "MICE: Multivariate Imputation by Chained Equations." *J. Statistical Software*, 2011.
[59] Stekhoven, D. J., Bühlmann, P. "MissForest — Non-parametric Missing Value Imputation for Mixed-Type Data." *Bioinformatics*, 2012.
[60] Śmieja, M., et al. "Processing of Missing Data by Neural Networks (GMMC)." *NeurIPS*, 2018.

### GNN foundations & SSL roots (out-of-scope, brief citations)
[61] Kipf, T. N., Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks (GCN)." *ICLR*, 2017.
[62] Veličković, P., et al. "Graph Attention Networks (GAT)." *ICLR*, 2018.
[63] Hamilton, W., Ying, R., Leskovec, J. "Inductive Representation Learning on Large Graphs (GraphSAGE)." *NeurIPS*, 2017.
[64] Gasteiger (Klicpera), J., Bojchevski, A., Günnemann, S. "Predict Then Propagate: GNNs Meet Personalized PageRank (APPNP)." *ICLR*, 2019.
[65] Perozzi, B., Al-Rfou, R., Skiena, S. "DeepWalk: Online Learning of Social Representations." *KDD*, 2014.
[66] Grover, A., Leskovec, J. "node2vec: Scalable Feature Learning for Networks." *KDD*, 2016.
[67] Kipf, T. N., Welling, M. "Variational Graph Auto-Encoders (VGAE)." *NeurIPS Bayesian DL Workshop*, 2016.
[68] Kingma, D. P., Welling, M. "Auto-Encoding Variational Bayes (VAE)." *ICLR*, 2014.
[69] Chen, T., et al. "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)." *ICML*, 2020.
[70] Marsden, P. V. "Network Data and Measurement." *Annual Review of Sociology*, 1990.
