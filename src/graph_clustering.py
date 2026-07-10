"""
Graph clustering using PyTorch Geometric's built-in ClusterData and ClusterLoader.

This is the RECOMMENDED approach - uses PyG's optimized ClusterGCN implementation
instead of manually implementing partitioning.

References:
- PyG ClusterData: https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.ClusterData
- ClusterGCN paper: https://arxiv.org/abs/1905.07953
"""

import torch
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.data import Data
import os


def create_cluster_loader(data, num_parts=20, batch_size=1, shuffle=True,
                          save_dir=None, **kwargs):
    """
    Create ClusterLoader using PyG's built-in ClusterGCN implementation.

    This is much simpler and more efficient than manual clustering!

    Args:
        data: PyG Data object (full graph)
        num_parts: Number of clusters/partitions (default: 20)
        batch_size: Number of clusters per training batch (default: 1)
        shuffle: Shuffle cluster order each epoch (default: True)
        save_dir: Directory to cache partitions (speeds up re-use)
        **kwargs: Additional arguments for ClusterData
            - recursive: Use recursive bisection (default: False)
            - keep_inter_cluster_edges: Keep edges between clusters (default: False)

    Returns:
        cluster_loader: ClusterLoader iterator

    Example:
        >>> from src.utils_ogbn import load_ogbn_products
        >>> graph, adj, _, features, labels, _ = load_ogbn_products()
        >>> loader = create_cluster_loader(graph, num_parts=20, save_dir='../data/ogbn_products')
        >>> for batch_data in loader:
        >>>     z = model.encoder(batch_data.x, batch_data.edge_index)
        >>>     # batch_data contains cluster subgraph
    """

    print(f"Creating ClusterData with {num_parts} partitions...")
    if save_dir:
        print(f"  Caching to: {save_dir}")

    # Create ClusterData (performs partitioning using METIS)
    cluster_data = ClusterData(
        data,
        num_parts=num_parts,
        save_dir=save_dir,
        log=True,  # Show progress
        **kwargs
    )

    print(f"  ✓ Partitioned into {len(cluster_data)} clusters")
    print(f"  ✓ Avg cluster size: ~{data.num_nodes // num_parts:,} nodes")

    # Create ClusterLoader (iterates over clusters)
    cluster_loader = ClusterLoader(
        cluster_data,
        batch_size=batch_size,  # Clusters per batch
        shuffle=shuffle,
        num_workers=0  # Set > 0 for multi-process loading
    )

    return cluster_loader


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')
    from src.utils_ogbn import load_ogbn_products

    print("="*70)
    print("GRAPH CLUSTERING WITH PyG ClusterData")
    print("="*70 + "\n")

    # Load OGBN-products
    print("[1/3] Loading OGBN-products...")
    graph, adj, adj_norm, features, labels, split_idx = load_ogbn_products(
        root='../data/ogbn_products'
    )
    print(f"  Loaded: {graph.num_nodes:,} nodes, {graph.edge_index.size(1):,} edges\n")

    # Create cluster loader (PyG will cache partitions automatically)
    print("[2/3] Creating ClusterLoader (PyG built-in)...")
    loader = create_cluster_loader(
        graph,
        num_parts=20,
        batch_size=1,  # Process 1 cluster at a time
        shuffle=True,
        save_dir='../data/ogbn_products'  # Cache partitions here
    )
    print()

    # Test iteration
    print("[3/3] Testing cluster iteration...")
    for i, batch_data in enumerate(loader):
        if i >= 3:
            break
        print(f"  Cluster {i}: {batch_data.num_nodes:,} nodes, "
              f"{batch_data.edge_index.size(1):,} edges")

    print("\n" + "="*70)
    print("SUCCESS! PyG ClusterLoader is much simpler ✓")
    print("="*70)
    print("\nBenefits of using PyG ClusterData:")
    print("  1. Uses optimized METIS partitioning (better quality)")
    print("  2. Automatically caches partitions (faster re-use)")
    print("  3. Handles edge cases and optimizations")
    print("  4. Well-tested and maintained by PyG team")
    print("  5. ~10 lines of code vs our ~300 lines!")
    print("="*70)
