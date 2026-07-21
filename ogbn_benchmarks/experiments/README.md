# Experiments

Reproducible scripts for the paper's experiments. Each script is self-contained:
run it from the `ogbn_benchmarks/` directory, it launches the needed runs (GPU
models sequentially to respect the shared GPU), and prints a summary table.


Run from the parent directory:
```bash
cd /home/erfan/MATE/ogbn_benchmarks
bash experiments/e1_main_comparison.sh
```

Each model's full artifacts (config, per-epoch log, metrics, weights) land in
`results/<model>_<dataset>_<missingness>_<rate>_<timestamp>/` as usual; the
experiment scripts additionally collect stdout logs + a summary under
`results/<experiment>_<timestamp>/`.

## Index

| Script | Experiment | Proves |
|--------|-----------|--------|
| `e1_main_comparison.sh` | E1 — all methods on ogbn-arxiv, equal budget, MCAR@0.4 | RQ1: NESS beats baselines |
| _(more added as we go)_ | | |

## Shared defaults

- Dataset: ogbn-arxiv (primary large graph)
- Missingness: MCAR @ 0.4
- Budget: `--epochs 400 --patience 20` for all learnable models (400 = ceiling; patience trims plateaued models early — fair, same rule for all)
- num_parts: 20 (arxiv cluster path); cache_device: cpu (shared-GPU safe)
