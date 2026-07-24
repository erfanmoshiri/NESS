# E7c — Contrastive: regime evaluation (RQ4)

**Reframed.** We are NOT calling contrastive redundant. Instead we characterize its
regime, like the other SSL components. Current evidence is only at 80% (where
single_view ≈ no_ssl, and w_con=2 slightly hurt) — but contrastive has **not** been
evaluated at extreme missingness, where other SSL becomes valuable (cf. E12).

**Experiment:** contrastive on vs off (`--w_con` / `--single_view`) across missingness
{0.6, 0.8, 0.9, 0.95}, locked config. Does the contrastive term's value grow with
missingness (like hist+path did in E12)?

**Status:** PENDING — the run that fairly characterizes contrastive's regime before any
claim about it.

<!-- Table: rate | F1 (contrastive on) | F1 (contrastive off / single_view) | Δ -->

**Optional, later:** if contrastive helps in some regime, a loss-function ablation
(Barlow-Twins vs InfoNCE vs prototype-InfoNCE) becomes worthwhile. Deferred until the
regime evaluation shows contrastive matters somewhere.

**Stance:** contrastive is retained as a NESS component; its contribution is stated as
regime-dependent, pending this evaluation. No "redundant" claim.
