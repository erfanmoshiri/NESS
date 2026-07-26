# Results section order

The order in which experiment subsections appear in the paper (story arc:
method wins -> dissect when SSL helps -> conditional finding -> consolidate).
File names keep the original experiment numbers, prefixed with a running index.

| Order | File | Experiment(s) | Role |
|---|---|---|---|
| — | (roadmap paragraph) | — | results-section intro / arc |
| 06 | 06_exp_main.tex   | E1  | main comparison (method wins) |
| 07 | 07_exp_rate.tex   | E4  | robustness to missingness rate |
| 09 | 09_E3.tex         | E3  | which objectives help |
| 10 | 10_E8_E10.tex     | E8 + E10 | why: mechanism + training dynamics |
| 11 | 11_E12.tex        | E12 | when: missingness regime |
| 12 | 12_E9.tex         | E9  | when: feature regime |
| 13 | 13_E11.tex        | E11 (+ prefill, views E7, contrastive E7d) | design & ablations |
| 14 | 14_E7b.tex        | E7b | transferable enhancement |
| 15 | 15_E6.tex         | E6  | scalability |

Notes:
- 06/07 already written; 08_exp_dynamics.tex (E10 draft) exists but E10 is folded
  into 10_E8_E10.tex in this order — reconcile when writing 10.
- Roadmap paragraph goes at the very start of the Results/Experiments section,
  before 06.
