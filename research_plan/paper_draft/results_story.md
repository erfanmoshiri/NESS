# Results story — the single reference for Experiments + Discussion

The complete narrative in three blocks. Each claim lists its supporting experiment.

## (A) The method wins
- **NESS beats baselines on dense/large graphs.** ogbn-arxiv @40%: NESS 0.454 vs
  FP 0.439, MATE 0.404, PaGCN 0.383. [E1]
- **The advantage grows as features get scarcer.** Across rates 0.2–0.9 NESS is best
  or tied and most stable; widest lead at 0.9 (0.415 vs MATE 0.378). [E4]
- On small binary graphs at low missingness, strong methods converge and NESS is
  mid-pack — expected, ties to (B). [E1]

## (B) SSL's value is regime-dependent (two axes)
- **Missingness axis:** SSL helps more at higher missing rates. hist+path lift over
  no-SSL grows +0.016 → +0.028 from rate 0.8 → 0.95. [E12]
  *Why (interpretive):* at high missingness the encoder is starved (empty
  neighborhoods, little for propagation to spread), leaving headroom for SSL; at low
  missingness the encoder already extracts the signal, so SSL is redundant.
- **Feature-type axis:** SSL helps on dense continuous features, is inert on sparse
  binary ones. hist gives +0.018 on arxiv but ~0 on cora/citeseer/amac/amap. [E9]
  *Why:* neighborhood targets require averaging neighbor features — meaningful for
  continuous attributes, near-noise for binary indicators.

## (C) What makes an SSL objective useful (the mechanism)
An objective helps only if it satisfies all three:
1. **Class-relevant** — the label-aware neighbor-histogram is the one consistent
   winner; feature-space objectives are not. [E3]
2. **Non-redundant with the encoder** — it must teach something the encoder does not
   already capture. Anchor-distance is predictable and class-relevant (probe 0.207)
   yet does not help, because a strong FP+SAGE encoder already extracts it; no
   weighting, projection head, or warmup rescues it. [E8]
3. **Non-trivial to fit** — blunt objectives with low-variance targets (centroid,
   spread, feature-recon) saturate almost immediately and add nothing. But hardness
   alone is not enough: path/triplet keep learning yet do not help, because they fail
   (1)/(2). [E3, E8 loss curves]
- **A well-configured encoder does the heavy lifting**, which shrinks SSL's marginal
  value. Encoder/config choices (SAGE, depth, w_cls) drove a larger gain than any SSL
  objective at moderate missingness. [E11]

## One-line summary
NESS wins on dense/large graphs and degrades gracefully; self-supervision is not a
uniform good but a regime-dependent one — valuable when features are scarce (starved
encoder) and dense (learnable neighborhood targets), and only when the objective is
class-relevant, non-redundant with the encoder, and non-trivial. The label-aware
histogram is the objective that consistently satisfies these conditions.

## Support gaps (be honest in the paper)
- The "starved encoder → headroom" mechanism for the missingness axis is interpretive
  (E12 shows the effect, not the cause).
- Contrastive: E7c pending re-evaluation (InfoNCE + ppr + no-view-avg runs). Do not
  finalize contrastive framing until those land.
