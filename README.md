# NESS: Robust Graph Learning under Missingness

NESS is a graph neural network for node classification when whole node feature vectors are missing. It couples a Feature-Propagation prefill, which gives every missing node long-range structural signal, with a set of pluggable self-supervised and auxiliary objectives on a multi-view contrastive encoder. Beyond the method, this repository supports a systematic study of *which* auxiliary objectives help under missingness and *when*: we find that self-supervision generally helps, but its benefit depends on the severity of missingness and on how class-relevant the objective's target is.

## Pipeline

1. **Prefill** — fill missing feature rows via Feature Propagation (diffuse observed features over the graph).
2. **Encode** — a GraphSAGE encoder produces two edge-masked views, fused by averaging.
3. **Objectives** — jointly optimize classification, edge reconstruction, a Barlow-Twins contrastive term, and pluggable auxiliary objectives (default: neighbor label-histogram + multi-hop reachability).
4. **Evaluate** — macro-F1 and accuracy on the masked test nodes.

## How to Run

```bash
python ogbn_benchmarks/main_benchmark_ogbn.py \
    --model NESS --dataset ogbn-arxiv --miss_rate 0.8
```

- `--model`: `NESS`, `FP`, `PaGCN`, `MATE`, `GraphSAGE`, `NeighAggre`, `KNN`
- `--dataset`: `ogbn-arxiv`, `cora`, `citeseer`, `amac`, `amap`
- `--miss_rate`: fraction of nodes with removed features (e.g. `0.4`, `0.8`)

Locked NESS defaults (SAGE, 3 layers, hidden 128, `hist + path` objectives, per-dataset learning rate) reproduce the paper results.
