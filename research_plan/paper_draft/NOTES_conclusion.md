# Locked conclusion / takeaway (single reference for all sections)

The reader should walk away with this. All sections build toward it; the abstract,
intro contributions, and conclusion all restate it in their own words.

> We propose NESS, a model for learning under missing node features, and conduct an
> extensive study of the characteristics and effectiveness of self-supervised objectives
> in this setting. Self-supervision generally helps under missingness, but how much it
> helps depends on the setting: (i) the severity of missingness, self-supervision
> contributing more when the encoder is information-starved; and (ii) the objective's
> characteristics, objectives whose targets are more class-relevant and non-trivial to
> fit yielding larger gains. The label-aware neighbor-histogram (hist) is the most
> reliable single objective; NESS pairs it with a structural reachability objective
> (path).

## The two things that modulate SSL's benefit
1. **Environment / missingness severity** — SSL helps more at high missingness (starved
   encoder). [E12, E4]
2. **Objective characteristics** — more class-relevant + harder (non-trivial) targets
   help more; low-variance/easy targets help little or not at all. [E3, Figure A, E10]

## Key supporting facts
- Most objectives give a positive lift on the locked config (E3): hist+path +0.022,
  hist +0.020, recon +0.017, path/centroid +0.007, anchor +0.004; only stats slightly
  negative. So the story is "SSL generally helps, modulated by setting", NOT "SSL is
  usually useless".
- Class-relevance of the target predicts lift (Figure A): hist highest, stats lowest.
- Loss dynamics (E10): class-relevant/structural objectives keep decreasing; low-variance
  feature objectives saturate early.
- The two-view / contrastive structure contributes (single_view is the weakest config).
- Feature reconstruction (recon) helps, echoing the prior imputation work [NCSSL].
- Feature-regime axis (E9): SSL helps on dense features, is inert on sparse binary.

## Accuracy guardrails (do not drift)
- "hist" = label-aware, the most reliable single objective. "path" = structural
  reachability complement. Do NOT call path label-correlated.
- Method's strength at moderate missingness comes largely from the encoder (FP-prefill +
  SAGE + config, E11); SSL is a setting-dependent enhancement on top.
- Framing: SSL GENERALLY HELPS, its magnitude modulated by (missingness severity) x
  (objective characteristics). Avoid the earlier over-negative "SSL is redundant" tone.
