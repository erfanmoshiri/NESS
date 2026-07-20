"""
Unified dataset loader for benchmark experiments.

Returns a consistent format for ALL datasets (OGBN-products + Cora/CiteSeer/AMAC/AMAP)
so model training functions never branch on dataset identity.

Unified return:
    graph:       PyG Data object with edge_index
    adj:         torch sparse adjacency [N, N]
    features:    [N, D] node features (binary BoW for small, embeddings for OGBN)
    labels:      [N] class labels (1-D long)
    num_classes: int

Small datasets (<100k nodes) are trained full-batch; OGBN uses cluster/neighbor
sampling. The FULL_BATCH_THRESHOLD constant defines the cutoff.
"""

import sys
import torch
from torch_geometric.data import Data

sys.path.insert(0, '..')

FULL_BATCH_THRESHOLD = 100_000  # graphs smaller than this train full-batch

SMALL_DATASETS = ['cora', 'citeseer', 'amac', 'amap']
LARGE_DATASETS = ['ogbn-products', 'ogbn-arxiv']


def load_dataset(name, data_root=None):
    """
    Load any supported dataset in unified format.

    Args:
        name: 'cora' | 'citeseer' | 'amac' | 'amap' | 'ogbn-products' | 'ogbn-arxiv'
        data_root: root dir (OGBN only; small datasets use ../data/<name>)

    Returns:
        graph, adj, features, labels, num_classes
    """
    if name == 'ogbn-products':
        return _load_ogbn(name, data_root)
    elif name == 'ogbn-arxiv':
        return _load_ogbn_arxiv(data_root)
    elif name in SMALL_DATASETS:
        return _load_small(name)
    else:
        raise ValueError(f'Unknown dataset: {name}')


def _load_ogbn(name, data_root):
    from src.utils_ogbn import load_ogbn_products
    root = data_root or '../data/ogbn_products'
    graph, adj, adj_norm, features, labels, split_idx = load_ogbn_products(root=root)
    labels = labels.squeeze() if labels.dim() > 1 else labels
    num_classes = int(labels.max().item()) + 1
    return graph, adj, features, labels, num_classes


def _load_ogbn_arxiv(data_root):
    """
    Load ogbn-arxiv (169k nodes, 128-d dense embeddings, 40 classes).

    Unlike products, arxiv is a DIRECTED citation graph. We symmetrize it
    (to_undirected) following the OGB reference implementations, so message
    passing and neighborhood-centric SSL see the full citation context — and
    for consistency with the other (undirected) benchmarks.
    """
    import torch as _torch
    from torch_geometric.utils import to_undirected

    root = data_root if (data_root and 'arxiv' in data_root) else '../data/ogbn_arxiv'

    # OGB stores some arrays non-writable / needs weights_only=False on load
    orig_load = _torch.load
    _torch.load = lambda *a, **k: orig_load(*a, **{**k, 'weights_only': False})
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
    finally:
        _torch.load = orig_load

    graph = dataset[0]
    features = graph.x
    labels = graph.y.squeeze() if graph.y.dim() > 1 else graph.y
    num_nodes = features.size(0)
    num_classes = int(labels.max().item()) + 1

    # Symmetrize the directed citation graph
    edge_index = to_undirected(graph.edge_index, num_nodes=num_nodes)
    graph.edge_index = edge_index
    graph.num_nodes = num_nodes

    # Sparse adjacency (needed by SSL targets, NeighAggre, KNN)
    vals = torch.ones(edge_index.size(1), dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_index, vals, (num_nodes, num_nodes)).coalesce()

    return graph, adj, features, labels, num_classes


def _load_small(name):
    """Wrap the legacy load_data() and normalize to the unified format."""
    from src.utils import load_data

    # load_data reads args.dataset and args.generative_flag
    class _Args:
        dataset = name
        generative_flag = True  # keep raw (unnormalized) features, like OGBN protocol

    adj, diff, adj_norm, features, labels, indices = load_data(_Args())

    # Build PyG Data with edge_index from the sparse adjacency
    adj = adj.coalesce()
    edge_index = adj.indices()
    labels = labels.squeeze() if labels.dim() > 1 else labels
    num_classes = int(labels.max().item()) + 1

    graph = Data(x=features, edge_index=edge_index, y=labels)
    graph.num_nodes = features.size(0)

    return graph, adj, features, labels, num_classes


def is_small(num_nodes):
    """True if the graph should be trained full-batch."""
    return num_nodes < FULL_BATCH_THRESHOLD
