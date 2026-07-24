# 1. Introduction

Graph neural networks (GNNs) have become the standard tool for learning on
relational data, achieving strong results on node classification, link
prediction, and recommendation. Their effectiveness rests on a quiet assumption:
that every node carries an informative feature vector, which message passing then
refines by aggregating information across edges. In practice this assumption
often fails. Nodes join a network before a profile is filled in, newly listed
items lack descriptions, sensors drop out, and privacy constraints withhold
attributes. The result is a graph in which a substantial fraction of nodes have
*no observed features at all*. Under such **attribute-missing** conditions,
message passing degrades sharply: a node whose neighborhood is largely
uninformative has little to aggregate, and the deficiency compounds with each
additional missing hop.

We study this setting through the lens of the *downstream task* rather than
reconstruction. Much prior work treats missing attributes as an imputation
problem—fill in the missing features, then run a standard model—and measures
success by how faithfully the raw features are recovered. Yet accurate
reconstruction is neither necessary nor sufficient for an accurate decision
boundary. What ultimately matters is whether the learned representations remain
discriminative when features are scarce. We therefore adopt **learning under
missing node features**: given a graph where whole feature vectors are absent,
learn node representations that support accurate downstream node classification.

Approaches to missing node features fall into a few broad families, each with a
characteristic limitation. *Imputation-based* methods reconstruct features before
applying a GNN; propagation variants are simple and scalable, while learned
generative imputers are more expressive but rely on dense or per-node operations
that do not scale. *Imputation-free* methods modify the aggregation to tolerate
missing entries without filling them. *Per-node parameterization* assigns each
missing node its own learnable embedding, which risks memorization when no
supervision reaches those parameters. Most recently, *self-supervised and
contrastive* methods learn missingness-robust representations from pretext tasks
instead of relying on reconstruction quality. (We defer individual methods and a
full taxonomy to Section 2.) Across these families, self-supervision is the most
active direction—but it is applied largely by analogy: an auxiliary objective is
borrowed from the graph-SSL literature and added in the hope that it helps. What
is missing is a principled account of *which* self-supervised objectives help a
GNN learn under missing features, and *when*.

This question is a natural continuation of our own prior work, which introduced a
neighborhood-centric self-supervised framework for graph attribute imputation and
showed that neighborhood statistics can serve as useful auxiliary supervision.
That work established *that* such supervision can help in an imputation setting;
it left open a more fundamental question: **when, and why, does self-supervision
actually help learning under missingness?** In this paper we take up that
question directly. Shifting the goal from imputation to downstream classification,
we treat the effect of self-supervision as an object of study rather than an
assumption.

We introduce **NESS**, a GNN for learning under missing node features that couples
(i) a **Feature-Propagation prefill** that gives every missing node long-range
structural signal before training, and (ii) a set of **self-supervised objectives**
layered on a well-configured encoder. NESS outperforms imputation-based,
imputation-free, and per-node-parameter baselines on downstream classification,
with the advantage most pronounced under severe missingness and at a scale where
generative imputation is infeasible.

Beyond the method, our central contribution is an empirical characterization of
*when* self-supervision helps under missingness. We find that its benefit is
**regime-dependent**. Rather than helping uniformly, self-supervision is most
valuable when the encoder is information-starved—under severe missingness—where a
propagation-based encoder alone cannot recover the signal. We further find that a
**label-aware objective (a neighbor label-histogram)** provides a consistent
semi-supervised signal across regimes, and we show it transfers as an enhancement
to other backbones. This regime-centric view refines the earlier premise that
neighborhood self-supervision is broadly beneficial, replacing it with an account
of the conditions under which it pays off.

Finally, we scope the regime in which neighborhood-centric self-supervision is
meaningful. Its targets are predictable only when node features are
**dense and continuous** (as in citation-graph embeddings), where neighborhood
statistics carry structural signal; on sparse binary bag-of-words attributes these
targets are far less predictable. We make this boundary explicit, which also
motivates our focus on large dense-embedding graphs—precisely the regime where
prior small-scale, binary-attribute benchmarks and imputation baselines are least
representative.

**Contributions.**
- **A method (NESS)** combining Feature-Propagation prefill with self-supervised
  objectives on a well-configured encoder, outperforming imputation-based,
  imputation-free, and per-node-parameter baselines on downstream node
  classification under missingness, at scale.
- **A characterization of when self-supervision helps under missingness**: its
  benefit is regime-dependent—most pronounced under severe missingness—rather
  than uniform. This refines the common assumption that auxiliary neighborhood
  objectives are broadly beneficial.
- **A label-aware semi-supervised objective** (neighbor label-histogram) that
  helps consistently and transfers as an enhancement to other GNN backbones.
- **An explicit regime analysis** (dense-embedding vs. sparse binary attributes)
  delineating where neighborhood-centric self-supervision is effective, and
  motivating evaluation on large dense-embedding graphs.

The remainder of the paper is organized as follows. Section 2 reviews learning
under missing node features and self-supervised graph representation learning.
Section 3 presents NESS and its objectives. Section 4 reports the main comparison,
the analysis of when self-supervision helps, and the robustness and regime
studies. Section 5 discusses implications and limitations, and Section 6 concludes.
