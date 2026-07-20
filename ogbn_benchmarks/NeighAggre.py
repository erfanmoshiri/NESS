"""
NeighAggre baseline for OGBN-products.

For each missing node: average features from observable neighbors (sparse ops).
Falls back to global mean if no observable neighbors exist.
Non-parametric — no training required.
"""

import torch
import time


def train_NeighAggre(adj, features, observable_id, masked_id, device):
    """
    Args:
        adj: Sparse adjacency [N, N]
        features: [N, 100] ground-truth embeddings
        observable_id: indices of visible nodes
        masked_id: indices of missing nodes (to impute)
        device: torch device

    Returns:
        imputed_features: [N, 100]
    """
    start = time.time()
    num_nodes = features.size(0)

    # Observable mask
    obs_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    obs_mask[observable_id] = True

    # Global mean fallback (from observable nodes only)
    global_mean = features[observable_id].mean(dim=0)

    # Sparse aggregation: for each node sum observable-neighbor features
    # adj is [N, N] sparse; zero out columns of non-observable nodes
    adj = adj.coalesce()
    src, dst = adj.indices()  # src -> dst edges

    # Keep only edges where src is observable
    obs_edge_mask = obs_mask[src]
    src_obs = src[obs_edge_mask]
    dst_obs = dst[obs_edge_mask]

    # Scatter-sum neighbor features onto destination nodes
    neighbor_sum = torch.zeros(num_nodes, features.size(1), device=device)
    neighbor_sum.scatter_add_(0, dst_obs.unsqueeze(1).expand(-1, features.size(1)), features[src_obs])

    neighbor_count = torch.zeros(num_nodes, 1, device=device)
    neighbor_count.scatter_add_(0, dst_obs.unsqueeze(1), torch.ones(dst_obs.size(0), 1, device=device))

    has_neighbors = (neighbor_count.squeeze(1) > 0)
    agg = neighbor_sum / neighbor_count.clamp(min=1)

    # Nodes with no observable neighbors → global mean
    agg[~has_neighbors] = global_mean

    imputed = features.clone()
    imputed[masked_id] = agg[masked_id]

    elapsed = time.time() - start
    print(f'  NeighAggre done in {elapsed:.1f}s — imputed {len(masked_id):,} nodes')
    return imputed
