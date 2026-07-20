"""
KNN baseline for OGBN-products.

Replaces the original O(N²) all-pairs shortest path AND the per-node Python
BFS loop with vectorized sparse matrix propagation:

  For each hop h=1..max_hops:
    propagate observable features across edges using sparse matmul
    accumulate weighted contributions (closer hops weighted more)

  For each missing node: use the accumulated neighbor mean as imputation.
  Falls back to global mean if no observable neighbors reached.

This is O(max_hops × E) — fully vectorized, runs on GPU.
"""

import torch
import time


def train_KNN(adj, features, observable_id, masked_id, device, K=3, max_hops=3):
    """
    Args:
        adj: Sparse adjacency [N, N] (on device)
        features: [N, 100] ground-truth embeddings (on device)
        observable_id: visible node indices
        masked_id: missing node indices to impute
        device: torch device
        K: kept for API compatibility (multi-hop effectively uses all neighbors)
        max_hops: number of propagation hops

    Returns:
        imputed_features: [N, 100]
    """
    start = time.time()
    num_nodes = features.size(0)

    # Only observable nodes contribute features; missing nodes are zero
    obs_features = torch.zeros_like(features)
    obs_features[observable_id] = features[observable_id]

    # obs_count[i] = number of observable nodes that have propagated to node i
    obs_count = torch.zeros(num_nodes, 1, device=device)
    obs_count[observable_id] = 1.0

    # Normalize adjacency row-wise for stable propagation
    adj = adj.coalesce()
    row_deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1)
    # Build row-normalized sparse adjacency
    src, dst = adj.indices()
    vals = adj.values() / row_deg[src]
    adj_norm = torch.sparse_coo_tensor(
        torch.stack([src, dst]), vals, adj.shape, device=device
    ).coalesce()

    # Accumulate multi-hop neighbor features with hop decay weighting
    accumulated_features = torch.zeros_like(features)
    accumulated_count = torch.zeros(num_nodes, 1, device=device)

    current_features = obs_features.clone()
    current_count = obs_count.clone()

    print(f'  Running {max_hops}-hop sparse propagation on {len(masked_id):,} missing nodes...')

    for hop in range(1, max_hops + 1):
        weight = 1.0 / hop  # closer hops weighted more
        # Propagate: each node gets weighted sum of normalized neighbor values
        current_features = torch.sparse.mm(adj_norm, current_features)
        current_count = torch.sparse.mm(adj_norm, current_count)

        accumulated_features += weight * current_features
        accumulated_count += weight * current_count

    # Normalize accumulated features by total weight received
    safe_count = accumulated_count.clamp(min=1e-9)
    neighbor_mean = accumulated_features / safe_count

    # Global mean fallback for nodes that received no signal
    global_mean = features[observable_id].mean(dim=0)
    no_signal = (accumulated_count.squeeze(1) < 1e-9)

    imputed = features.clone()
    imputed[masked_id] = neighbor_mean[masked_id]
    imputed[masked_id[no_signal[masked_id]]] = global_mean

    elapsed = time.time() - start
    no_signal_count = no_signal[masked_id].sum().item()
    print(f'  KNN done in {elapsed:.1f}s | {no_signal_count:,} nodes used global mean fallback')
    return imputed
