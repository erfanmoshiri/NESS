# Key claim: blunt vs. informative SSL objectives (reviewer-proof version)

This is the defensible version of "what makes an SSL objective useful," stated so
it survives our own results table. It replaces the earlier "input-varying /
stochastic is better" framing — which our controlled experiment (E8) refutes.

---

## The claim (what to write)

> A self-supervised objective helps a GNN learn under missing features only if its
> prediction target is **(a) hard / high-variance** (so the loss keeps providing
> gradient rather than saturating), **(b) class-relevant** (what it teaches
> transfers to the downstream boundary), and **(c) non-redundant** with what the
> encoder already computes via propagation and aggregation. **Blunt** objectives —
> those that predict a low-variance, trivially-fittable target such as the
> neighborhood mean, neighborhood spread, or masked-feature reconstruction —
> satisfy none of these: their loss saturates almost immediately and they add
> nothing downstream. Designing an objective that meets all three conditions is
> the central difficulty, and even a class-relevant target (e.g. anchor-distance)
> can fail condition (c) when the encoder is strong enough to already capture it.

Positive framing: we characterize *what a useful objective needs*, not "SSL fails."

---

## What we do NOT claim (and why)

- **NOT "input-varying beats fixed-input."** E8 factor-flip refutes it: the
  fixed-input per-node form (`F1_fixed`, 0.438) BEAT the varying-input pairwise
  form (`anchor_base`, 0.427). Do not use input-variation as the discriminator.
- **NOT "stochastic beats static."** `triplet` and `recon` are stochastic and
  among the worst (-0.011, -0.024); stochastic objectives span the full range.
  Stochasticity does not separate winners from losers.
- **NOT "harder is sufficient."** `path`/`triplet` have hard, high-variance losses
  that keep dropping, yet do not help (path +0.001, triplet -0.011). Hardness is
  necessary but not sufficient — hence conditions (b) and (c).

---

## Evidence (all from E3 / E8, ogbn-arxiv @80%, no_ssl baseline = 0.3748 F1)

### 1. Loss-magnitude / saturation (the "blunt" signature) — E3 loss curves
| objective | loss start -> end | behavior | Delta F1 |
|---|---|---|---|
| centroid (neighbor mean) | 0.027 -> 0.005 | tiny, saturates ~ep20 | +0.003 |
| stats (neighbor spread) | 0.006 -> 0.001 | tiny, saturates | -0.025 |
| recon (masked feature) | 0.042 -> 0.026 | tiny, flat | -0.024 |
| anchor (dist-diff) | 0.026 -> 0.019 | tiny, flat | +0.011 |
| path (reachability) | 0.69 -> 0.39 | large, keeps dropping | +0.001 |
| triplet (dist ranking) | 1.01 -> 0.38 | large, keeps dropping | -0.011 |
| hist (neighbor labels) | 2.76 -> 0.88 | large, keeps dropping | **+0.018** |

Reading: blunt objectives (centroid/stats/recon/anchor) have near-zero,
quickly-saturated losses -> no sustained teaching -> inert. Only `hist` has all
three properties (hard target that keeps dropping, class-relevant via labels,
non-redundant) and it is the one consistent winner.

### 2. Hard-but-useless counterexamples (prove hardness alone insufficient)
`path` and `triplet` have large, still-decreasing losses (condition a met) but do
NOT help — they fail (b)/(c): local reachability is not class-relevant on arxiv
and is largely captured by the encoder already.

### 3. Redundancy with a strong encoder (condition c) — E8-c
Anchor-distance IS class-relevant (probe: macro-F1 0.207 from 32 landmark
distances). Yet the anchorcls objective does not help at 80% at any weight
(w_anchor 1-8), with a projection head (loss floor unchanged 0.83->0.82 -> not a
capacity limit), or with SSL-first warmup (parity, not gain). The signal exists
but the FP-prefill + SAGE encoder already extracts it -> redundant.

### 4. Where useful objectives DO pay off (the regime, E12)
`hist+path` benefit over no-SSL GROWS with missingness: +0.016 @0.8 -> +0.028
@0.95. When the encoder is starved (severe missingness), a class-relevant
objective is no longer redundant and helps.

---

## The "challenges" to mention (open difficulties, honest)
- Meeting (a)+(b)+(c) simultaneously is hard: most structural targets are either
  blunt (a fails), position-only (b fails), or redundant with propagation (c fails).
- Class-relevant objectives (anchor) can still be redundant with a strong encoder
  at moderate missingness; their value emerges mainly under scarcity.
- Label-aware objectives (hist) meet the conditions but are semi-supervised
  (use observable labels at train; none at test — no leakage, see E3 note).

---

## Sources
- E3 table + loss curves: `results/e3_ablation_20260723_094037/`
- E8 factor-flips + anchorcls ladder: `results/e8_factorflip_20260724_011653/`,
  `results/e_anchorw_20260724_022015/`, standalone anchorcls/proj/warmup runs
- E12 missingness curve: `results/e_sslmiss_20260723_154349/`
- probe (0.207): rerun `probe_anchor_class.py` (stdout, not saved)
