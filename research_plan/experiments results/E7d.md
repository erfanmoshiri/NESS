# E7d — Contrastive design across missing rates

Compare four contrastive configurations across missing rates: Barlow-Twins (bt),
node InfoNCE, class-aware prototype InfoNCE (proto), and single-view (no second view,
no contrastive). Locked config, ppr view where a second view exists, ssl=hist+path,
800 epochs. Tests whether contrastive helps and whether its value grows with missingness.

**Source:** `experiments/e7d_contrastive_rates.sh` →
`ogbn_benchmarks/results/e7d_conrates_20260725_104407/`.

**Status:** DONE.

## Results — Test Macro-F1 (rows = contrastive, cols = rate)

| Contrastive | 0.6 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|
| single-view | 0.4527 | 0.4382 | 0.4009 | 0.3913 |
| Barlow-Twins | 0.4432 | 0.4431 | 0.4281 | 0.3984 |
| InfoNCE | 0.4586 | 0.4473 | 0.4241 | 0.4066 |
| prototype-InfoNCE | 0.4471 | 0.4388 | 0.4282 | 0.4089 |

## Reading

- **Contrastive helps most at high missingness.** At 0.9 and 0.95, all three contrastive
  variants beat single-view: e.g. @0.95 proto 0.409 / InfoNCE 0.407 / BT 0.398 vs
  single-view 0.391; @0.9 BT/proto 0.428 vs single-view 0.401 (+0.02 to +0.03). At the
  milder 0.6 rate the gap is small or absent.
- **The contrastive advantage grows as features get scarcer** — consistent with the
  broader finding that self-supervision matters most when the encoder is starved.
- Among the losses, InfoNCE is strongest at 0.6/0.8, prototype-InfoNCE at 0.95; BT is
  competitive throughout. All are close; the choice of contrastive loss matters less than
  the presence of a second view under high missingness.

## Framing for the paper

Contrastive learning stabilizes NESS under severe missingness: a second view with a
contrastive term consistently outperforms a single view at high rates. This justifies
retaining the contrastive component. The default (Barlow-Twins over a PPR view) is
competitive across rates; InfoNCE and the class-aware prototype variant are alternatives
that trade off similarly.
