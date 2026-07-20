"""
MATE baseline for OGBN-products.

Key difference from our model (NESS): learns a per-node embedding
for each missing node via nn.Embedding(N, F) instead of predicting from structure.

Changes from original MATE:
- Feature_learner: nn.Embedding(N, F) replaces torch.eye(N) approach (~1GB vs 46TB)
- Diffusion view: replaced with second edge-masked view (diff matrix is N×N dense)
- Training: ClusterLoader mini-batches instead of full-graph
- Loss: cross-entropy classification + Barlow Twins contrastive + edge
- Evaluation: macro-F1 instead of Recall@K
"""

import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.utils import add_self_loops, negative_sampling
from torch_geometric.nn import GCNConv


# ============================================================================
# Model
# ============================================================================

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            ic = in_channels if i == 0 else hidden_channels
            oc = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GCNConv(ic, oc))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(conv(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.elu(self.convs[-1](x, edge_index))


class EdgeDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, z_1, z_2, edge, sigmoid=False):
        x = z_1[edge[0]] * z_2[edge[1]]
        out = self.mlp(x)
        return out.sigmoid() if sigmoid else out


class MATEModel(nn.Module):
    def __init__(self, num_nodes, feature_dim, hidden_dim, num_classes,
                 encoder_channels=256, decoder_channels=64, dropout=0.5):
        super().__init__()
        # Learnable embeddings for ALL nodes — missing nodes will be optimized
        # ~1GB for 2.4M × 100 (feasible on 49GB GPU)
        self.node_embeddings = nn.Embedding(num_nodes, feature_dim)
        nn.init.zeros_(self.node_embeddings.weight)  # Start at zero

        self.encoder = GNNEncoder(feature_dim, encoder_channels, hidden_dim,
                                  num_layers=2, dropout=dropout)
        self.edge_decoder = EdgeDecoder(hidden_dim, decoder_channels, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def get_node_features(self, cluster_x, global_indices, observable_mask):
        """
        For observable nodes: use true features (cluster_x).
        For missing nodes: use learned embedding.
        """
        x = cluster_x.clone()
        missing_local = ~observable_mask
        if missing_local.any():
            missing_global = global_indices[missing_local]
            x[missing_local] = self.node_embeddings(missing_global)
        return x

    def forward_classifier(self, z):
        return self.classifier(z)


def _mask_edge(edge_index, p=0.7):
    e_ids = torch.arange(edge_index.size(1), device=edge_index.device)
    mask = torch.bernoulli(torch.full_like(e_ids, p, dtype=torch.float32)).bool()
    return edge_index[:, ~mask], edge_index[:, mask]


def _barlow_twins(z1, z2, lambda_param=0.005):
    N = z1.size(0)
    D = z1.size(1)
    z1_n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2_n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    c = torch.mm(z1_n.T, z2_n) / N
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.fill_diagonal_(0).pow_(2).sum()
    # Normalize by dimension so the loss scale is comparable to cross-entropy
    return (on_diag + lambda_param * off_diag) / D


# ============================================================================
# Training
# ============================================================================

def train_MATE_ogbn(graph, features, labels, observable_id, masked_id,
                    vali_id, test_id, num_classes, device,
                    hidden=128, encoder_channels=256, decoder_channels=64,
                    dropout=0.5, lr=0.001, weight_decay=5e-5,
                    epochs=200, patience=20, num_parts=50, p=0.7,
                    log_path=None):
    """
    Args:
        graph: PyG Data with edge_index
        features: [N, 100] ground-truth embeddings
        labels: [N] class labels
        observable_id: visible node indices
        masked_id: all missing node indices
        vali_id: validation split
        test_id: test split
        num_classes: 47

    Returns:
        val_f1, test_f1
    """
    from sklearn.metrics import f1_score

    num_nodes = features.size(0)
    feature_dim = features.size(1)

    # Zero out missing nodes in graph features — keep on CPU for ClusterData
    masked_features_cpu = features.cpu().clone()
    masked_features_cpu[masked_id.cpu()] = 0.0

    graph_cpu = graph.clone()
    graph_cpu.x = masked_features_cpu
    graph_cpu.y = labels.cpu()

    # Observable mask for full graph
    obs_mask_full = torch.zeros(num_nodes, dtype=torch.bool)
    obs_mask_full[observable_id.cpu()] = True

    from data_loader import is_small

    if is_small(num_nodes):
        # ---- Small graph: single full-graph "cluster" (no partitioning) ----
        print('  Small graph: full-batch (single cluster, no partitioning)')
        masked_features = masked_features_cpu.to(device)
        ei = graph.edge_index.to(device)
        aug_ei, _ = add_self_loops(ei)
        global_idx = torch.arange(num_nodes, device=device)
        cluster_cache = [{
            'x': masked_features,
            'y': labels.to(device),
            'edge_index': ei,
            'aug_edge_index': aug_ei,
            'global_indices': global_idx,
            'obs_mask': obs_mask_full.to(device),
            'num_nodes': num_nodes,
        }]
    else:
        # ---- Large graph: ClusterLoader partitioning ----
        print(f'  Partitioning graph into {num_parts} clusters...')
        cluster_data = ClusterData(graph_cpu, num_parts=num_parts,
                                   save_dir='../data/ogbn_products', log=False)

        print('  Pre-caching clusters...')
        cache_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False, num_workers=0)
        masked_features = masked_features_cpu.to(device)
        node_perm = cluster_data.partition.node_perm
        partptr = cluster_data.partition.partptr

        cluster_cache = []
        for i, batch in enumerate(cache_loader):
            batch = batch.to(device)
            global_idx = node_perm[partptr[i]:partptr[i+1]].to(device)
            aug_ei, _ = add_self_loops(batch.edge_index)
            obs_local = obs_mask_full[global_idx.cpu()].to(device)
            cluster_cache.append({
                'x': batch.x,
                'y': batch.y,
                'edge_index': batch.edge_index,
                'aug_edge_index': aug_ei,
                'global_indices': global_idx,
                'obs_mask': obs_local,
                'num_nodes': batch.num_nodes,
            })
        print(f'  Cached {len(cluster_cache)} clusters')

    # Build model
    model = MATEModel(num_nodes, feature_dim, hidden, num_classes,
                      encoder_channels, decoder_channels, dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f'  Training MATE for up to {epochs} epochs (patience={patience})...')
    best_val_f1 = 0.0
    best_state = None
    epochs_no_improve = 0
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        epoch_losses = {'total': 0.0, 'edge': 0.0, 'con': 0.0, 'cls': 0.0}
        random.shuffle(cluster_cache)

        for cluster in cluster_cache:
            edge_index = cluster['edge_index']
            global_indices = cluster['global_indices']
            obs_mask = cluster['obs_mask']

            # Get input features: true for observable, learned embedding for missing
            x = model.get_node_features(cluster['x'], global_indices, obs_mask)

            # Two views with different edge masks
            rem1, masked_edges = _mask_edge(edge_index, p)
            rem2, _ = _mask_edge(edge_index, p)

            z1 = model.encoder(x, rem1)
            z2 = model.encoder(x, rem2)
            z = (z1 + z2) * 0.5

            # Edge loss
            max_edges = 50000
            if masked_edges.size(1) > max_edges:
                perm = torch.randperm(masked_edges.size(1), device=device)[:max_edges]
                pos_edges = masked_edges[:, perm]
            else:
                pos_edges = masked_edges

            num_neg = min(pos_edges.size(1), 50000)
            neg_edges = negative_sampling(cluster['aug_edge_index'],
                                          num_nodes=cluster['num_nodes'],
                                          num_neg_samples=num_neg, method='sparse')

            pos1 = model.edge_decoder(z1, z2, pos_edges)
            pos2 = model.edge_decoder(z2, z1, pos_edges)
            neg1 = model.edge_decoder(z1, z2, neg_edges)
            neg2 = model.edge_decoder(z2, z1, neg_edges)

            loss_edge = (
                F.binary_cross_entropy_with_logits(pos1, torch.ones_like(pos1)) +
                F.binary_cross_entropy_with_logits(pos2, torch.ones_like(pos2)) +
                F.binary_cross_entropy_with_logits(neg1, torch.zeros_like(neg1)) +
                F.binary_cross_entropy_with_logits(neg2, torch.zeros_like(neg2))
            ) / 4

            # Barlow Twins contrastive loss
            loss_con = _barlow_twins(z1, z2)

            # Classification loss — only on OBSERVABLE nodes (avoid label leakage)
            y = cluster['y'].squeeze() if cluster['y'].dim() > 1 else cluster['y']
            obs_mask = cluster['obs_mask']
            logits = model.forward_classifier(z)
            loss_cls = F.cross_entropy(logits[obs_mask], y[obs_mask])

            # Weight down contrastive so classification signal is not drowned
            loss_total = loss_edge + loss_con + loss_cls

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            epoch_losses['total'] += loss_total.detach().item()
            epoch_losses['edge'] += loss_edge.detach().item()
            epoch_losses['con'] += loss_con.detach().item()
            epoch_losses['cls'] += loss_cls.detach().item()

        n = len(cluster_cache)
        epoch_time = time.time() - epoch_start
        log_entry = {
            'epoch': epoch,
            'loss_total': epoch_losses['total'] / n,
            'loss_edge': epoch_losses['edge'] / n,
            'loss_con': epoch_losses['con'] / n,
            'loss_cls': epoch_losses['cls'] / n,
            'epoch_time_s': round(epoch_time, 2),
        }

        if epoch % 5 == 0:
            val_f1 = _eval(model, cluster_cache, labels, vali_id, device, obs_mask_full)
            log_entry['val_f1'] = val_f1
            print(f'  Epoch {epoch}/{epochs} | '
                  f'Total: {epoch_losses["total"]/n:.3f} '
                  f'Edge: {epoch_losses["edge"]/n:.3f} '
                  f'Con: {epoch_losses["con"]/n:.3f} '
                  f'Cls: {epoch_losses["cls"]/n:.3f} | '
                  f'Val F1: {val_f1:.4f} | {epoch_time:.1f}s')
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

    test_f1 = _eval(model, cluster_cache, labels, test_id, device, obs_mask_full)
    print(f'  Best Val F1: {best_val_f1:.4f} | Test F1: {test_f1:.4f}')
    print(f'  Total training time: {total_train_time:.1f}s '
          f'({total_train_time/60:.1f} min) over {epoch} epochs')
    return best_val_f1, test_f1


def _eval(model, cluster_cache, labels, eval_id, device, obs_mask_full):
    from sklearn.metrics import f1_score

    model.eval()
    eval_id_set = set(eval_id.cpu().tolist())

    all_preds = {}
    with torch.no_grad():
        for cluster in cluster_cache:
            global_indices = cluster['global_indices']
            obs_mask = cluster['obs_mask']

            # Check if any eval nodes are in this cluster
            local_eval_mask = torch.tensor(
                [g.item() in eval_id_set for g in global_indices], device=device
            )
            if not local_eval_mask.any():
                continue

            x = model.get_node_features(cluster['x'], global_indices, obs_mask)
            # Use full edges for eval (no masking)
            z = model.encoder(x, cluster['edge_index'])
            logits = model.forward_classifier(z)
            preds = logits.argmax(dim=1)

            for local_i, global_i in enumerate(global_indices.tolist()):
                if global_i in eval_id_set:
                    all_preds[global_i] = preds[local_i].item()

    eval_list = eval_id.cpu().tolist()
    y_pred = [all_preds[n] for n in eval_list if n in all_preds]
    y_true = labels[eval_id[:len(y_pred)]].cpu().tolist()

    return f1_score(y_true, y_pred, average='macro')
