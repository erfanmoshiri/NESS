"""
Figure A support — how class-relevant is each objective's TARGET?

For each per-node-target SSL objective we build its target matrix (the quantity the
objective asks the encoder to predict) and fit a logistic-regression probe to predict
the node's CLASS from that target. Macro-F1 of the probe = how class-relevant the
target is. We pair this with the objective's downstream lift (Delta F1 from E3) to see
whether "class-relevant target" predicts "helps downstream".

Covered (per-node targets): recon (= FP-prefilled features), centroid (neighbor mean),
stats (neighbor std), anchor (landmark distances), hist (neighbor label histogram).
Relational objectives (path, triplet) have no per-node target vector and are excluded.

Usage:  python probe_objectives.py --dataset ogbn-arxiv --miss_rate 0.8
Outputs: prints probe macro-F1 per objective; writes results/probe_objectives.json
"""
import argparse, json, sys
import torch
import numpy as np

sys.path.insert(0, '..')
from data_loader import load_dataset
from src.utils_ogbn import simulate_node_missingness
from src.utils import set_random_seed
from NESS_bench import _compute_ssl_targets, _compute_label_histogram, _landmark_distances
from FP import feature_propagation


def probe(X, y, tr, te, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    clf = LogisticRegression(max_iter=300, n_jobs=-1, C=1.0)
    clf.fit(X[tr], y[tr])
    p = clf.predict(X[te])
    f1 = f1_score(y[te], p, average='macro'); acc = accuracy_score(y[te], p)
    print(f'  {tag:12s} dim={X.shape[1]:<5} probe macro-F1={f1:.4f}  acc={acc:.4f}')
    return round(f1, 4), round(acc, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ogbn-arxiv')
    ap.add_argument('--data_root', default='../data/ogbn_products')
    ap.add_argument('--miss_rate', type=float, default=0.8)
    ap.add_argument('--num_anchors', type=int, default=16)
    ap.add_argument('--ssl_hops', type=int, default=2)
    ap.add_argument('--seed', type=int, default=72)
    args = ap.parse_args()

    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph, adj, features, labels, C = load_dataset(args.dataset, args.data_root)
    N = features.size(0)
    y = (labels.squeeze() if labels.dim() > 1 else labels).cpu().numpy()

    alln = torch.arange(N)
    obs, miss = simulate_node_missingness(alln, adj, features, labels, 'MCAR', args.miss_rate, args.seed)
    obs_cpu = obs.cpu()

    # probe split: fit on observable nodes, test on masked nodes (the evaluated regime)
    tr, te = obs.cpu().numpy(), miss.cpu().numpy()

    masked = features.clone(); masked[miss] = 0.0
    obs_bool = torch.zeros(N, dtype=torch.bool); obs_bool[obs_cpu] = True

    results = {}
    print(f'\n=== Probe: predict CLASS from each objective target ({args.dataset} @ {args.miss_rate}) ===')

    # recon target = FP-prefilled features (what recon reconstructs / the feature signal)
    fp = feature_propagation(graph.edge_index, masked.clone(), obs_bool, N, num_iterations=40, device=device).cpu().numpy()
    results['recon'] = probe(fp, y, tr, te, 'recon')

    # centroid / stats = neighbor mean / std targets
    tgt_stats, tgt_centroid = _compute_ssl_targets(adj, features, masked, obs.to(device), device, ssl_hops=args.ssl_hops)
    results['centroid'] = probe(tgt_centroid.cpu().numpy(), y, tr, te, 'centroid')
    results['stats']    = probe(tgt_stats.cpu().numpy(),    y, tr, te, 'stats')

    # anchor = landmark hop-distances
    ei = graph.edge_index; vals = torch.ones(ei.size(1))
    sp = torch.sparse_coo_tensor(ei, vals, (N, N)).coalesce().to(device)
    lm = torch.randperm(N)[:args.num_anchors].to(device)
    adist = _landmark_distances(sp, lm, N, device).cpu().numpy()
    results['anchor'] = probe(adist, y, tr, te, 'anchor')

    # hist = neighbor label histogram (label-derived; expected high)
    hist = _compute_label_histogram(adj.cpu(), labels.cpu(), obs_cpu, C, k=args.ssl_hops).cpu().numpy()
    results['hist'] = probe(hist, y, tr, te, 'hist')

    with open('results/probe_objectives.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved results/probe_objectives.json')


if __name__ == '__main__':
    main()
