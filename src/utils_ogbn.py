"""
Data loading and missingness simulation utilities for OGBN-products dataset.

OGBN-products characteristics:
- 2.4M nodes (Amazon products)
- 124M edges (co-purchasing relationships)
- 100-dim pre-computed embeddings (NOT raw features!)
- 47 product categories

Key differences from Cora/CiteSeer:
- Mask ENTIRE nodes (all 100 dims), not individual features
- Features are embeddings, not categorical/numerical raw attributes
- Evaluate with classification F1, not Recall@K/NDCG
"""

import torch
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans


def load_ogbn_products(root='./data/ogbn_products'):
    """
    Load OGBN-products dataset in MATE-compatible format.

    Returns:
        graph: PyG Data object
        adj: Sparse adjacency matrix [N, N]
        features: [N, 100] node embeddings
        labels: [N] product categories (0-46)
        split_idx: {'train': tensor, 'valid': tensor, 'test': tensor}
    """
    print(f'Loading OGBN-products from {root}...')

    # Fix 1: torch.load weights_only issue (PyTorch 2.6)
    import torch as torch_temp
    original_load = torch_temp.load
    torch_temp.load = lambda *args, **kw: original_load(*args, **{**kw, 'weights_only': False})

    # Fix 2: Non-writable numpy arrays from pandas
    # Monkey-patch torch.from_numpy to handle non-writable arrays
    original_from_numpy = torch_temp.from_numpy
    def from_numpy_writable(ndarray):
        if not ndarray.flags.writeable:
            ndarray = ndarray.copy()  # Make writable copy
        return original_from_numpy(ndarray)
    torch_temp.from_numpy = from_numpy_writable

    try:
        from ogb.nodeproppred import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(name='ogbn-products', root=root)

        graph = dataset[0]
        split_idx_raw = dataset.get_idx_split()

        # PROPERLY FIX: Copy arrays to make them writable before converting to tensors
        # This fixes the "non-writable array" warning from PyTorch
        split_idx = {
            'train': split_idx_raw['train'].clone() if torch.is_tensor(split_idx_raw['train']) else torch.tensor(split_idx_raw['train'].copy(), dtype=torch.long),
            'valid': split_idx_raw['valid'].clone() if torch.is_tensor(split_idx_raw['valid']) else torch.tensor(split_idx_raw['valid'].copy(), dtype=torch.long),
            'test': split_idx_raw['test'].clone() if torch.is_tensor(split_idx_raw['test']) else torch.tensor(split_idx_raw['test'].copy(), dtype=torch.long)
        }

        print(f'  Nodes: {graph.num_nodes:,}')
        print(f'  Edges: {graph.edge_index.shape[1]:,}')
        print(f'  Feature dims: {graph.x.shape[1]}')
        print(f'  Classes: {dataset.num_classes}')

        # Convert to MATE format
        adj = edge_index_to_sparse_adj(graph.edge_index, graph.num_nodes)
        features = graph.x  # [2449029, 100] embeddings
        labels = graph.y.squeeze()  # [2449029] categories

        # Compute normalized adjacency (for diffusion view if needed)
        adj_norm = normalize_adj(adj)

        return graph, adj, adj_norm, features, labels, split_idx

    finally:
        # Restore original functions
        torch_temp.load = original_load
        torch_temp.from_numpy = original_from_numpy


def simulate_node_missingness(all_nodes, adj, features, labels,
                              missingness_type='MCAR', miss_rate=0.4, seed=72):
    """
    Simulate missing NODES (not features!) for OGBN-products.

    Different from Cora: mask entire 100-dim embeddings, not individual features.

    Args:
        all_nodes: All node indices (tensor)
        adj: Adjacency matrix (for MAR - degree-based)
        features: Node embeddings [N, D] (for MNAR - embedding-based)
        labels: Node labels [N] (for MAR - category-based)
        missingness_type: 'MCAR', 'MAR', or 'MNAR'
        miss_rate: Fraction of nodes to mask (0.2, 0.4, 0.6)
        seed: Random seed

    Returns:
        observable_id: Nodes with embeddings visible (tensor)
        masked_id: Nodes with embeddings hidden/to be imputed (tensor)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f'Simulating {missingness_type} missingness at {miss_rate*100:.0f}% rate...')

    num_nodes = len(all_nodes)
    num_to_mask = int(num_nodes * miss_rate)

    if missingness_type == 'MCAR':
        # Completely random - missing completely at random
        perm = torch.randperm(num_nodes)
        masked_id = all_nodes[perm[:num_to_mask]]
        observable_id = all_nodes[perm[num_to_mask:]]
        print('  MCAR: Uniform random selection')

    elif missingness_type == 'MAR':
        # Missingness depends on OBSERVABLE properties
        # Example: Low-degree nodes more likely missing (unpopular products)

        # Get node degrees
        degrees = torch.sparse.sum(adj, dim=1).to_dense()
        degree_scores = degrees[all_nodes]

        # Inverse probability: low degree → high prob of being missing
        prob_scores = 1.0 / (degree_scores + 1.0)
        prob_scores = prob_scores / prob_scores.sum()

        # Sample weighted
        masked_indices = np.random.choice(
            len(all_nodes),
            size=num_to_mask,
            replace=False,
            p=prob_scores.cpu().numpy()
        )
        masked_id = all_nodes[masked_indices]
        observable_id = torch.tensor(
            list(set(all_nodes.tolist()) - set(masked_id.tolist())),
            dtype=all_nodes.dtype
        )

        print(f'  MAR: Low-degree nodes preferentially masked')
        print(f'       Avg degree (masked): {degrees[masked_id].mean():.1f}')
        print(f'       Avg degree (observable): {degrees[observable_id].mean():.1f}')

    elif missingness_type == 'MNAR':
        # Missingness depends on UNOBSERVABLE (the embedding itself)
        # Example: Embeddings in certain regions more likely missing

        print('  MNAR: Clustering embeddings...')
        # Cluster embeddings using k-means
        kmeans = KMeans(n_clusters=10, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(features[all_nodes].cpu().numpy())

        # Make certain clusters more likely to be missing
        # Clusters 0, 3, 7 have 80% chance, others have 20%
        prob_per_cluster = np.array([0.8, 0.2, 0.2, 0.8, 0.2, 0.2, 0.2, 0.8, 0.2, 0.2])
        prob_scores = prob_per_cluster[cluster_labels]
        prob_scores = prob_scores / prob_scores.sum()

        masked_indices = np.random.choice(
            len(all_nodes),
            size=num_to_mask,
            replace=False,
            p=prob_scores
        )
        masked_id = all_nodes[masked_indices]
        observable_id = torch.tensor(
            list(set(all_nodes.tolist()) - set(masked_id.tolist())),
            dtype=all_nodes.dtype
        )

        masked_clusters = np.bincount(cluster_labels[masked_indices], minlength=10)
        print(f'       Cluster distribution (masked): {masked_clusters}')

    else:
        raise ValueError(f"Unknown missingness type: {missingness_type}")

    print(f'  Observable: {len(observable_id):,} ({len(observable_id)/len(all_nodes):.1%})')
    print(f'  Masked: {len(masked_id):,} ({len(masked_id)/len(all_nodes):.1%})')

    return observable_id, masked_id


def edge_index_to_sparse_adj(edge_index, num_nodes):
    """Convert PyG edge_index to scipy sparse adjacency matrix, then to torch sparse."""
    edge_index_np = edge_index.cpu().numpy()
    adj = sp.coo_matrix(
        (np.ones(edge_index_np.shape[1]), (edge_index_np[0], edge_index_np[1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32
    )
    return sparse_mx_to_torch_sparse_tensor(adj)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert scipy sparse matrix to torch sparse tensor (properly configured).

    Explicitly sets check_invariants to avoid warnings. For large graphs (OGBN-products),
    we disable invariant checks for performance since scipy.sparse guarantees valid COO format.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    # Always explicitly set check_invariants to avoid "implicitly disabled" warning
    # For large graphs, disable for performance (scipy guarantees validity)
    # For small graphs, still disable since scipy COO is always valid
    result = torch.sparse_coo_tensor(
        indices, values, shape,
        dtype=torch.float32,
        check_invariants=False  # Explicit opt-out (scipy guarantees valid COO)
    )

    return result


def normalize_adj(adj):
    """
    Normalize adjacency matrix: D^{-1/2} A D^{-1/2}

    Args:
        adj: Sparse adjacency matrix (torch sparse tensor)

    Returns:
        norm_adj: Normalized sparse adjacency matrix
    """
    # Convert torch sparse to scipy sparse (avoid dense conversion!)
    adj = adj.coalesce()  # Ensure sparse tensor is in coalesced form
    indices = adj.indices().cpu().numpy()
    values = adj.values().cpu().numpy()
    shape = adj.shape

    adj_sp = sp.coo_matrix(
        (values, (indices[0], indices[1])),
        shape=shape
    )

    # Add self-loops
    adj_sp = adj_sp + sp.eye(adj_sp.shape[0])

    # Compute degree matrix
    rowsum = np.array(adj_sp.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    # D^{-1/2} A D^{-1/2}
    adj_normalized = adj_sp.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def compute_diffusion_matrix(adj, alpha=0.15, max_iter=10):
    """
    Compute personalized PageRank diffusion matrix.
    PPR = alpha * (I - (1-alpha) * A_norm)^{-1}

    WARNING: This is expensive for large graphs (2.4M nodes).
    For OGBN-products, recommend skipping diffusion and using same adj for both views.

    Args:
        adj: Sparse adjacency matrix
        alpha: Teleport probability
        max_iter: Number of power iterations

    Returns:
        diff: Diffusion matrix (sparse)
    """
    print(f'Computing diffusion matrix (alpha={alpha}, iter={max_iter})...')
    print(f'WARNING: This is memory-intensive for large graphs!')

    num_nodes = adj.size(0)

    # For large graphs, just return normalized adjacency as approximation
    if num_nodes > 100000:
        print(f'  Graph too large ({num_nodes:,} nodes), returning normalized adj as diffusion approximation')
        return normalize_adj(adj)

    # Normalize adj
    adj_dense = adj.to_dense()
    deg = adj_dense.sum(dim=1, keepdim=True)
    deg[deg == 0] = 1
    adj_norm = adj_dense / deg

    # Power iteration: D = sum_{k=0}^{K} alpha * (1-alpha)^k * A^k
    diff = alpha * torch.eye(num_nodes, device=adj.device)
    a_power = torch.eye(num_nodes, device=adj.device)

    for k in range(1, max_iter + 1):
        a_power = torch.mm(a_power, adj_norm)
        diff = diff + alpha * ((1 - alpha) ** k) * a_power

    # Convert back to sparse
    indices = diff.nonzero(as_tuple=False).t()
    values = diff[indices[0], indices[1]]

    return torch.sparse.FloatTensor(indices, values, torch.Size([num_nodes, num_nodes]))
