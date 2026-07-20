"""
PaGCN (Partial Graph Convolutional Network) baseline for OGBN-products.

Zhang, Jiang et al., "Incomplete Graph Learning via Partial Graph Convolutional
Network", IEEE TAI 2024.

Method: imputation-free. Reinterprets GCN aggregation as energy minimization and
inserts a per-entry observed-mask so only OBSERVED neighbor features contribute:

    H' = ( Ã·(M ⊙ H) ) ⊘ ( Ã·M )        (element-wise divide, epsilon-guarded)

where M is the binary observed-mask (1 if feature entry observed, else 0), Ã is
symmetric-normalized adjacency. Reduces exactly to standard GCN when nothing is
missing. No GMM, no per-node parameters, GCN-level cost — scales to 2.4M nodes.

For OGBN node-level missingness, a masked node has ALL feature entries missing,
so M is a per-node 0/1 vector broadcast across feature dims.
"""

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.loader import ClusterData, ClusterLoader
import random


class PartialGCNLayer(nn.Module):
    """One partial graph convolution: masked aggregation + linear transform."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, norm_adj, mask):
        """
        x:        [N, D] features (missing entries are 0)
        norm_adj: sparse [N, N] symmetric-normalized adjacency
        mask:     [N, D] observed-entry mask (1 observed, 0 missing)
        """
        # Aggregate observed features and observed-mask separately
        agg_x = torch.sparse.mm(norm_adj, x * mask)      # Ã·(M⊙H)
        agg_m = torch.sparse.mm(norm_adj, mask)          # Ã·M
        # Normalize by observed-neighbor mass (epsilon guard for isolated/all-missing)
        h = agg_x / (agg_m + 1e-8)
        return self.lin(h)


class PaGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.pgc1 = PartialGCNLayer(in_channels, hidden_channels)
        self.pgc2 = PartialGCNLayer(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, norm_adj, mask):
        # First partial conv uses the feature mask
        h = F.relu(self.pgc1(x, norm_adj, mask))
        h = F.dropout(h, p=self.dropout, training=self.training)
        # After layer 1, features are dense (all entries meaningful) → mask = ones
        ones = torch.ones_like(h)
        h = F.relu(self.pgc2(h, norm_adj, ones))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)


def _normalize_adj(edge_index, num_nodes, device):
    """Build symmetric-normalized sparse adjacency with self-loops."""
    from torch_geometric.utils import to_scipy_sparse_matrix
    import scipy.sparse as sp
    import numpy as np

    adj = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=num_nodes).tocoo()
    adj = adj + sp.eye(num_nodes)  # self-loops
    deg = np.asarray(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, where=deg > 0, out=deg_inv_sqrt)
    deg_inv_sqrt[deg == 0] = 0.0
    D = sp.diags(deg_inv_sqrt)
    norm = (D @ adj @ D).tocoo()

    idx = torch.from_numpy(np.vstack((norm.row, norm.col)).astype(np.int64))
    val = torch.from_numpy(norm.data.astype('float32'))
    return torch.sparse_coo_tensor(idx, val, (num_nodes, num_nodes)).coalesce().to(device)


def train_PaGCN(graph, features, labels, observable_id, masked_id, vali_id, test_id,
                num_classes, device, hidden=256, dropout=0.5, lr=0.01,
                weight_decay=5e-4, epochs=200, patience=20, num_parts=50,
                log_path=None, weights_path=None):
    """
    Cluster-based training (same scalability approach as our MATE_ogbn).

    Returns:
        val_f1, test_f1
    """
    from sklearn.metrics import f1_score
    from data_loader import is_small

    num_nodes = features.size(0)

    # Node-level mask: masked nodes have all entries missing
    masked_features_cpu = features.cpu().clone()
    masked_features_cpu[masked_id.cpu()] = 0.0

    node_mask = torch.ones(num_nodes, dtype=torch.float32)
    node_mask[masked_id.cpu()] = 0.0  # 0 for missing nodes

    obs_mask_full = torch.zeros(num_nodes, dtype=torch.bool)
    obs_mask_full[observable_id.cpu()] = True

    model = PaGCN(features.size(1), hidden, num_classes, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Small graph: full-batch training ----
    if is_small(num_nodes):
        from fullbatch import train_full_batch
        x = masked_features_cpu.to(device)
        nrm = _normalize_adj(graph.edge_index, num_nodes, device)
        mask = node_mask.unsqueeze(1).expand(-1, features.size(1)).to(device)
        return train_full_batch(
            model, optimizer,
            logits_fn=lambda: model(x, nrm, mask),
            labels=labels.to(device),
            observable_id=observable_id.to(device),
            vali_id=vali_id.to(device),
            test_id=test_id.to(device),
            device=device, epochs=epochs, patience=patience,
            log_path=log_path, model_name='PaGCN',
            weights_path=weights_path,
        )

    # ---- Large graph: cluster-based training ----
    graph_cpu = graph.clone()
    graph_cpu.x = masked_features_cpu
    graph_cpu.y = labels.cpu()
    graph_cpu.node_mask = node_mask.unsqueeze(1)  # [N, 1] travels with clusters

    print(f'  Partitioning graph into {num_parts} clusters...')
    cluster_data = ClusterData(graph_cpu, num_parts=num_parts,
                               save_dir='../data/ogbn_products', log=False)

    print('  Pre-caching clusters (with normalized adjacency)...')
    cache_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False, num_workers=0)
    node_perm = cluster_data.partition.node_perm
    partptr = cluster_data.partition.partptr

    cluster_cache = []
    for i, batch in enumerate(cache_loader):
        batch = batch.to(device)
        nrm = _normalize_adj(batch.edge_index, batch.num_nodes, device)
        gidx = node_perm[partptr[i]:partptr[i+1]].to(device)
        cluster_cache.append({
            'x': batch.x,
            'y': batch.y,
            'norm_adj': nrm,
            'mask': batch.node_mask.expand(-1, batch.x.size(1)),  # [n, D]
            'obs_mask': obs_mask_full[gidx.cpu()].to(device),
            'global_indices': gidx,
        })
    print(f'  Cached {len(cluster_cache)} clusters')

    print(f'  Training PaGCN for up to {epochs} epochs (patience={patience})...')
    best_val_f1 = 0.0
    best_state = None
    epochs_no_improve = 0
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        random.shuffle(cluster_cache)

        for c in cluster_cache:
            optimizer.zero_grad()
            logits = model(c['x'], c['norm_adj'], c['mask'])
            y = c['y'].squeeze() if c['y'].dim() > 1 else c['y']
            obs = c['obs_mask']
            loss = F.cross_entropy(logits[obs], y[obs])  # observable nodes only
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(cluster_cache)
        epoch_time = time.time() - epoch_start
        log_entry = {'epoch': epoch, 'loss': avg_loss, 'epoch_time_s': round(epoch_time, 2)}

        if epoch % 5 == 0:
            val_f1 = _eval(model, cluster_cache, labels, vali_id, device)
            log_entry['val_f1'] = val_f1
            print(f'  Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} | {epoch_time:.1f}s')
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'  Early stopping at epoch {epoch}')
                    if log_path:
                        with open(log_path, 'a') as f:
                            f.write(json.dumps(log_entry) + '\n')
                    break

        if log_path:
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

    total_train_time = time.time() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    if weights_path is not None:
        torch.save(model.state_dict(), weights_path)

    test_f1 = _eval(model, cluster_cache, labels, test_id, device)
    print(f'  Best Val F1: {best_val_f1:.4f} | Test F1: {test_f1:.4f}')
    print(f'  Total training time: {total_train_time:.1f}s ({total_train_time/60:.1f} min)')
    return best_val_f1, test_f1


def _eval(model, cluster_cache, labels, eval_id, device):
    from sklearn.metrics import f1_score
    model.eval()
    eval_id_set = set(eval_id.cpu().tolist())

    # Rebuild global index per cluster from cached obs_mask ordering is not stored;
    # instead we re-derive predictions per cluster and gather by matching.
    # We stored obs_mask but need global indices — recompute below.
    all_preds = {}
    with torch.no_grad():
        for c in cluster_cache:
            logits = model(c['x'], c['norm_adj'], c['mask'])
            preds = logits.argmax(dim=1)
            gidx = c['global_indices']
            for local_i, g in enumerate(gidx.tolist()):
                if g in eval_id_set:
                    all_preds[g] = preds[local_i].item()

    eval_list = [n for n in eval_id.cpu().tolist() if n in all_preds]
    y_pred = [all_preds[n] for n in eval_list]
    y_true = labels[torch.tensor(eval_list)].cpu().tolist()
    return f1_score(y_true, y_pred, average='macro')
