"""
Probe: are ANCHOR-DISTANCE features class-relevant on ogbn-arxiv?

Gate for the "make anchor harder" redesign. If a node's vector of distances-to-
landmarks cannot predict its class much better than chance, then NO reformulation
of the anchor objective (harder loss, ranking, etc.) will help downstream — the
signal simply isn't there. If it CAN predict class well, a better anchor design
is worth building.

We fit a simple logistic-regression probe on distance features and report macro-F1
+ accuracy on held-out nodes, against controls:
  - random baseline (majority / uniform)
  - FP-prefilled raw features (the signal the encoder already has) -> upper anchor-context ref

Usage:
  python probe_anchor_class.py --dataset ogbn-arxiv --num_anchors 32
  python probe_anchor_class.py --dataset ogbn-arxiv --num_anchors 32 --local_hops 2
"""
import argparse
import sys
import torch
import numpy as np

sys.path.insert(0, '..')
from data_loader import load_dataset
from NESS_bench import _landmark_distances
from FP import feature_propagation
from src.utils import set_random_seed


def probe(X, y, train_idx, test_idx, tag):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score
    Xtr, ytr = X[train_idx], y[train_idx]
    Xte, yte = X[test_idx], y[test_idx]
    clf = LogisticRegression(max_iter=300, n_jobs=-1, C=1.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    f1 = f1_score(yte, pred, average='macro')
    acc = accuracy_score(yte, pred)
    print(f'  {tag:<28} dim={X.shape[1]:<5} macro-F1={f1:.4f}  acc={acc:.4f}')
    return f1, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ogbn-arxiv')
    ap.add_argument('--data_root', default='../data/ogbn_products')
    ap.add_argument('--num_anchors', type=int, default=32)
    ap.add_argument('--local_hops', type=int, default=0, help='0 = global (full BFS); >0 = cap hops (local)')
    ap.add_argument('--seed', type=int, default=72)
    args = ap.parse_args()

    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading {args.dataset}...')
    graph, adj, features, labels, num_classes = load_dataset(args.dataset, args.data_root)
    N = features.size(0)
    y = (labels.squeeze() if labels.dim() > 1 else labels).cpu().numpy()
    print(f'  {N:,} nodes | {num_classes} classes')

    # train/test split for the probe (70/30, random)
    perm = torch.randperm(N)
    split = int(0.7 * N)
    train_idx, test_idx = perm[:split].numpy(), perm[split:].numpy()

    # BFS adjacency
    ei = graph.edge_index
    vals = torch.ones(ei.size(1))
    sp_adj = torch.sparse_coo_tensor(ei, vals, (N, N)).coalesce().to(device)

    hop_cap = args.local_hops if args.local_hops > 0 else None
    reach = 'local(<=%d hops)' % args.local_hops if hop_cap else 'global(full BFS)'
    print(f'Computing anchor distances: K={args.num_anchors}, reach={reach} ...')
    lm = torch.randperm(N)[:args.num_anchors].to(device)
    adist = _landmark_distances(sp_adj, lm, N, device, max_hops=hop_cap).cpu().numpy()

    print('\n=== PROBE: predict CLASS from features (macro-F1 higher = more class-relevant) ===')
    # 1) anchor-distance features
    probe(adist, y, train_idx, test_idx, f'anchor-dist ({reach})')

    # 2) control: FP-prefilled raw features (what the encoder starts from) — upper reference
    print('  (computing FP-prefill control...)')
    obs = torch.zeros(N, dtype=torch.bool); obs[torch.from_numpy(train_idx)] = True
    fp = feature_propagation(ei, features.clone(), obs, N, num_iterations=40, device=device).cpu().numpy()
    probe(fp, y, train_idx, test_idx, 'FP-prefilled raw feats')

    # 3) random baseline
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(args.seed)
    rand_pred = rng.integers(0, num_classes, size=len(test_idx))
    print(f'  {"random baseline":<28} dim=-     macro-F1={f1_score(y[test_idx], rand_pred, average="macro"):.4f}')

    print('\nInterpretation: if anchor-dist macro-F1 is near random and far below FP-feats,')
    print('graph position is NOT class-relevant -> a harder anchor objective will not help.')


if __name__ == '__main__':
    main()
