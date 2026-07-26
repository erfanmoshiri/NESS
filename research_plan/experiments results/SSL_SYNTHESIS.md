# SSL Synthesis — What We Learned (RQ3/RQ4 core narrative)

The consolidated take from E3, E7, E8, E11, E12 and the full "make SSL help"
investigation. All ogbn-arxiv, MCAR.

---

## The headline finding (positive framing — "SSL works, where and when")

> **Self-supervision helps under missing features — the contribution is characterizing
> WHERE and WHEN. It is most valuable when the encoder is information-starved (severe
> missingness, weak supervision); at moderate missingness a well-configured encoder
> already captures much of the structural signal, so the *marginal* value of structural
> SSL is small there. Label-aware SSL (hist, semi-supervised) helps across regimes.**

Framing rule (agreed): we say **SSL works and give the regime + analysis** — we do NOT
say "SSL is redundant." The finding is regime-dependence, stated positively.

### Positioning decision: Option C (hybrid)
Method paper (NESS beats baselines, robust, scalable) + the SSL where/when analysis as
the intellectual depth. NESS keeps FP-prefill + SAGE encoder + hist+path + **contrastive**
as its components; the analysis explains each component's regime of usefulness.

### Component stances (agreed, for the writeup)
- **hist (semi-supervised, label-aware):** a KEPT, first-class objective. Helps across
  regimes. Pitched as semi-SSL, and E7b transfers it to other backbones (RQ5).
- **contrastive (Barlow-Twins):** RETAINED as a component. Its contribution is
  regime-dependent and NOT yet evaluated at extreme missingness (E12 used hist+path only).
  Do NOT call it redundant — flag the 90/95% contrastive run as the evaluation that
  characterizes its regime.
- **structural (path/anchor):** help most under scarcity (E12); marginal at 80% because the
  encoder already captures local structure. Stated as regime-dependence, not uselessness.

---

## The evidence chain (each step rules out an alternative explanation)

1. **The signal is real (not "no structural signal").** Probe: a node's distances to 32
   random landmarks predict its class at macro-F1 **0.207** (11× random; 72% of full-feature
   F1). Graph position IS class-relevant on arxiv. → We can NEVER tell a "topology is
   useless" story; the probe forbids it.

2. **A strong encoder already captures it.** Switching GCN→SAGE and 2→3 layers (a better
   structural encoder) is exactly what *killed* the SSL benefit (path went from +2.9 F1 on
   the broken GCN encoder to +0.001 on SAGE). → SSL was *substituting* for a weak encoder;
   once the encoder is good, SSL is redundant.

3. **It's not weighting.** Swept w_anchor 1→8: none beats no-SSL (best reaches parity).

4. **It's not objective design.** Tested centroid, stats, recon, path, triplet, anchor
   (regression), anchorcls (classification). Only **hist** (label-based) helps at 80%.

5. **It's not z-capacity.** Added a SimCLR-style projection head → SSL loss floor barely
   moved (0.83→0.82) and F1 didn't improve. The floor is *inherent unpredictability*, not a
   bottleneck.

6. **It's not training order.** SSL-first warmup (100 ep) → only reached parity (0.4381 vs
   0.4367), not a gain. If unique signal were being crowded out, warmup would have unlocked
   it. It didn't → the signal is redundant, not suppressed.

7. **But SSL DOES help where the encoder is starved:**
   - **E11:** at w_cls=1, raising SSL weight 0→5 gave **+0.037 F1**.
   - **E12:** at 95% missing, hist+path gave **+0.028** over no-SSL (margin GROWS with
     missingness: +0.016 @0.8 → +0.028 @0.95).

Conclusion: signal is real (1), encoder already has it at 80% (2), and no lever
(3-6) changes that — but starve the encoder (7) and SSL becomes essential.

---

## The mechanism (why the SSL loss drops but doesn't help)

The anchorcls loss falls 1.66→0.83 then floors. Two parts:
- **Reducible (the drop):** coarse structure FP-prefill + SAGE *already encode* → fitting it
  trains the read-out head, not the representation → redundant.
- **Irreducible (the floor):** fine global position a local 3-hop encoder *cannot compute* →
  unlearnable. (Projection head not lowering the floor confirms it's inherent.)
So loss ↓ a little, F1 unchanged — perfectly consistent.

The rule for a *useful* objective (from the loss-magnitude analysis): it must be
**(a) hard / high-variance (keeps teaching)** AND **(b) class-relevant** AND
**(c) non-redundant with the encoder.** hist has all three at 80%; structural objectives
fail (c) unless the encoder is starved.

---

## What this means for the paper

- **RQ3 becomes:** *"which SSL objectives help under missingness, and why"* → answer: only
  label-aware (hist) at moderate missingness; structural objectives help only when features
  are scarce. The redundancy-with-encoder interaction is the insight.
- **RQ4 (views):** largely negative — view choice barely matters at 80% (E7); the contrastive
  term is near-inert (single_view = no_ssl). Simplest view (edge_mask) is best.
- **Reframes prior work:** SSL-for-missing-graph methods (AmGCL etc.) likely reported gains
  because they used weaker encoders where SSL was not yet redundant.

## The locked model (what actually drove performance)
NESS = FP-prefill + SAGE-3L-128 + edge_mask + hist+path + w_cls=3 + dropout 0.3.
Test F1 @80% = **0.4366**, beating FP (0.388@128 / 0.397@256) by ~0.04. The *encoder
config* (E11) — not the SSL objective — is what produced the win at 80%.

## Optional loose end (not required)
Frozen-encoder probe (train anchorcls head on frozen z): if it reaches ~0.83 too, proves the
loss drop is pure read-out of existing z (redundancy) with zero encoder reshaping. One run;
would make the mechanism claim unarguable.

## Robustness of the conclusion
The headline claim rests on a consistent PATTERN, not one number: nothing beats no-SSL across
six independent levers (objective, weight, projection, warmup, width, view). That convergence
is what makes "structural SSL is redundant at moderate missingness" robust.
