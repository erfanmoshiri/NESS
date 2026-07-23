"""
KNN baseline for graphs with node-level missing features.

A fully-missing node has NO features, so "nearest neighbors" cannot mean
feature-space similarity — it means **graph proximity**. For each missing node we
BFS outward and collect the first K *observable* nodes (nearest by hop distance),
then impute with the mean of those K nodes' features. Falls back to the global
observable mean if fewer than 1 observable node is found within `max_hops`.

This is a TRUE K-nearest-observable-neighbors imputation (distinct from Feature
Propagation, which diffuses all observable features over the whole graph).
"""

import torch
import time
from collections import deque


def train_KNN(adj, features, observable_id, masked_id, device, K=3, max_hops=10):
    """
    Args:
        adj: sparse adjacency [N, N]
        features: [N, D] ground-truth features
        observable_id: visible node indices
        masked_id: missing node indices to impute
        K: number of nearest observable neighbors to average
        max_hops: BFS depth cap (safety bound)

    Returns:
        imputed_features: [N, D]
    """
    start = time.time()
    num_nodes = features.size(0)
    feats_cpu = features.cpu()

    # Build adjacency list (CPU) for BFS
    adj = adj.coalesce().cpu()
    src, dst = adj.indices()
    src, dst = src.numpy(), dst.numpy()
    adj_list = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        adj_list[s].append(d)

    obs_mask = torch.zeros(num_nodes, dtype=torch.bool)
    obs_mask[observable_id.cpu()] = True
    obs_mask_np = obs_mask.numpy()

    global_mean = feats_cpu[observable_id.cpu()].mean(dim=0)
    imputed = feats_cpu.clone()

    masked_list = masked_id.cpu().tolist()
    total = len(masked_list)
    print(f'  True KNN (K={K}, max_hops={max_hops}) over {total:,} missing nodes...')

    fallback = 0
    for i, node in enumerate(masked_list):
        # BFS outward, collect first K observable nodes by hop distance
        visited = {node}
        frontier = deque([(node, 0)])
        found = []
        while frontier and len(found) < K:
            cur, hop = frontier.popleft()
            if hop >= max_hops:
                continue
            for nbr in adj_list[cur]:
                if nbr in visited:
                    continue
                visited.add(nbr)
                if obs_mask_np[nbr]:
                    found.append(nbr)
                    if len(found) >= K:
                        break
                frontier.append((nbr, hop + 1))

        if found:
            imputed[node] = feats_cpu[found].mean(dim=0)
        else:
            imputed[node] = global_mean
            fallback += 1

        if (i + 1) % 50000 == 0:
            print(f'    {i+1:,}/{total:,} ({time.time()-start:.0f}s)')

    elapsed = time.time() - start
    print(f'  KNN done in {elapsed:.1f}s | {fallback:,} nodes used global-mean fallback')
    return imputed.to(device)
