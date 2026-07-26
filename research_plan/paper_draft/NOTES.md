# Paper draft — notes (kept OUT of the .tex files)

LaTeX files are pure text, no comments. All authoring notes live here.

## Global
- Citations use `[Name]` placeholders in all `.tex` — replace with `\cite{key}` from `references.bib`.
- Style: short, factual, scientific sentences; no repetition or line-filling.
- SINGLE-COLUMN journal. Never use `table*`/`figure*`. Plain `table`/`figure`; wrap wide
  numeric tables in `\resizebox{\linewidth}{!}{...}`; figures use `width=\linewidth`.
- No em-dashes (`---`) in prose. One line per paragraph in the `.tex` files.
- Section labels used for cross-refs: `sec:intro`, `sec:related`, `sec:method`,
  `sec:preliminaries` (not written yet), `sec:experiments`, `sec:discussion`, `sec:conclusion`.

## 01_introduction.tex
- "NESS outperforms baselines" (¶ on the method) is an IOU — depends on E1, not yet run on locked config.
- Optional: add a headline number teaser after E1 finishes.

## 02_related_work.tex
- Positioning (Sec 2.2 masked, Sec 2.5 positioning) is aligned to actual results:
  regime-dependent SSL; label-histogram is the consistent helper; structural objectives
  help mainly under scarcity. **Do NOT reintroduce "stochastic structural SSL wins" — our
  experiments do not support it.**
- Full-text verification needed before submission (currently from abstracts/notes only):
  MATE and MOBA (per-node-memorization, "not at large scale"), RARE (latent mechanism),
  DCLN, COMPLETER, SGHFP.
- Two claims are OURS (must match RQ3/RQ4 results, not the cited papers'):
  (1) feature-reconstruction objective plateaus early / no downstream gain under missingness;
  (2) the effective objective transfers across backbones.
- `[NCSSL]` needs a bibliography entry; cite in third person if double-blind.
- Table `tab:relwork-refs` is a working reference table — decide whether to keep it in the
  final paper or move to appendix / delete.

## ncss_positioning.tex
- Reusable NCSSL-positioning paragraph. Rests only on the TASK axis (imputation vs.
  downstream classification) — no assumption about the previous paper's datasets/feature types.
- Framing rule: "extends/scopes," never "overturns" or "redundant" near NCSSL.
- This paragraph overlaps the NCSSL bridge already in the intro; pick one home for it
  (intro vs. related work) to avoid duplication.
