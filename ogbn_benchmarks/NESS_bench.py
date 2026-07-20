"""
NESS — our model — as a unified benchmark entry.

NEighborhood Statistics Self-supervision. Predicts neighborhood feature
statistics from graph structure (spread + centroid residual) as SSL objectives,
alongside edge reconstruction, Barlow-Twins contrastive, and classification.

Wraps the model components from models/NESS.py so it runs through the SAME
benchmark runner / logging / saving as the baselines. Small graphs (<100k) train
full-batch; OGBN uses cluster-based training. SSL targets are computed once on
the full graph and indexed per cluster.
"""

import sys
import json
import time
import random
import torch
import torch.nn.functional as F
from torch import optim
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.utils import add_self_loops, negative_sampling

sys.path.insert(0, '..')
from models.NESS import (
    GNNEncoder, EdgeDecoder, Projector, Con_Projector, Model,
)
from src.utils import MaskEdge


def _compute_ssl_targets(adj, features, masked_features, observable_id, device):
    """Compute neighborhood-stats + centroid-residual SSL targets on the full graph."""
    try:
        from src.fast_ssl_compute import (
            compute_neighborhood_embedding_stats as stats_fn,
            compute_neighborhood_centroid_residual as resid_fn,
        )
    except ImportError:
        from models.NESS import (
            compute_neighborhood_embedding_stats as stats_fn,
            compute_neighborhood_centroid_residual as resid_fn,
        )
    target_stats = stats_fn(adj, features, observable_id)
    target_residual = resid_fn(adj, masked_features, observable_id)
    return target_stats.to(device), target_residual.to(device)


def _barlow_twins(z1, z2, lambda_param=0.005):
    N = z1.size(0)
    D = z1.size(1)
    z1n = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
    z2n = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
    c = torch.mm(z1n.T, z2n) / N
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.fill_diagonal_(0).pow_(2).sum()
    return (on_diag + lambda_param * off_diag) / D


def _build_model(num_features, num_classes, encoder_channels, hidden, decoder_channels,
                 dropout, p, device):
    encoder = GNNEncoder(num_features, encoder_channels, hidden,
                         num_layers=2, dropout=dropout, layer='gcn', activation='elu')
    edge_decoder = EdgeDecoder(hidden, decoder_channels, num_layers=2, dropout=0.3)
    projector = Projector(hidden, encoder_channels, out_channels=num_features,
                          num_layers=2, dropout=0.3)
    con_projector = Con_Projector(hidden, encoder_channels, out_channels=num_features,
                                  num_layers=2, dropout=0.3)
    mask_edge = MaskEdge(p=p)
    model = Model(encoder, edge_decoder, projector, con_projector,
                  temp=0.2, pos_weight_tensor=None, neg_weight_tensor=None,
                  mask=mask_edge, feature_dim=num_features, hidden_dim=hidden,
                  num_classes=num_classes)
    return model.to(device)


def train_NESS(graph, features, labels, observable_id, masked_id, vali_id, test_id,
               num_classes, device, adj=None, hidden=128, encoder_channels=256,
               decoder_channels=64, dropout=0.5, lr=0.001, weight_decay=5e-5,
               epochs=200, patience=20, num_parts=50, p=0.7,
               w_con=10.0, w_stats=100.0, w_residual=50.0,
               log_path=None, weights_path=None):
    """
    Args:
        graph: PyG Data with edge_index
        features: [N, D] ground-truth features
        labels: [N] labels
        observable_id, masked_id, vali_id, test_id: node splits
        num_classes: number of classes
        adj: sparse adjacency (needed for SSL target computation)

    Returns:
        val_f1, test_f1
    """
    from sklearn.metrics import f1_score
    from data_loader import is_small

    num_nodes = features.size(0)
    num_features = features.size(1)

    # Zero out missing nodes
    masked_features = features.clone()
    masked_features[masked_id] = 0.0

    # SSL targets (once, on full graph)
    print('  Computing SSL targets (neighborhood stats + centroid residual)...')
    target_stats, target_residual = _compute_ssl_targets(
        adj, features, masked_features, observable_id, device)

    obs_mask_full = torch.zeros(num_nodes, dtype=torch.bool)
    obs_mask_full[observable_id.cpu()] = True

    mask_edge = MaskEdge(p=p)
    model = _build_model(num_features, num_classes, encoder_channels, hidden,
                         decoder_channels, dropout, p, device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Build cluster cache (single full-graph cluster if small) ----
    if is_small(num_nodes):
        print('  Small graph: full-batch (single cluster)')
        mf = masked_features.to(device)
        ei = graph.edge_index.to(device)
        aug_ei, _ = add_self_loops(ei)
        gidx = torch.arange(num_nodes, device=device)
        cluster_cache = [{
            'x': mf, 'y': labels.to(device),
            'edge_index': ei, 'aug_edge_index': aug_ei,
            'global_indices': gidx,
            'target_stats': target_stats, 'target_residual': target_residual,
            'obs_mask': obs_mask_full.to(device),
            'num_nodes': num_nodes,
        }]
    else:
        graph_cpu = graph.clone()
        graph_cpu.x = masked_features.cpu()
        graph_cpu.y = labels.cpu()
        print(f'  Partitioning graph into {num_parts} clusters...')
        cluster_data = ClusterData(graph_cpu, num_parts=num_parts,
                                   save_dir='../data/ogbn_products', log=False)
        cache_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False, num_workers=0)
        node_perm = cluster_data.partition.node_perm
        partptr = cluster_data.partition.partptr
        cluster_cache = []
        for i, batch in enumerate(cache_loader):
            batch = batch.to(device)
            gidx = node_perm[partptr[i]:partptr[i+1]].to(device)
            aug_ei, _ = add_self_loops(batch.edge_index)
            cluster_cache.append({
                'x': batch.x, 'y': batch.y,
                'edge_index': batch.edge_index, 'aug_edge_index': aug_ei,
                'global_indices': gidx,
                'target_stats': target_stats[gidx], 'target_residual': target_residual[gidx],
                'obs_mask': obs_mask_full[gidx.cpu()].to(device),
                'num_nodes': batch.num_nodes,
            })
        print(f'  Cached {len(cluster_cache)} clusters')

    encoder = model.encoder
    edge_decoder = model.edge_decoder

    print(f'  Training NESS for up to {epochs} epochs (patience={patience})...')
    best_val_f1 = 0.0
    best_state = None
    epochs_no_improve = 0
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        el = {'total': 0.0, 'edge': 0.0, 'con': 0.0, 'cls': 0.0, 'stats': 0.0, 'residual': 0.0}
        random.shuffle(cluster_cache)

        for c in cluster_cache:
            edge_index = c['edge_index']
            rem1, masked_edges = mask_edge(edge_index)
            rem2, _ = mask_edge(edge_index)

            num_neg = min(masked_edges.size(1), 50000)
            neg_edges = negative_sampling(c['aug_edge_index'], num_nodes=c['num_nodes'],
                                          num_neg_samples=num_neg, method='sparse')

            z1 = encoder(c['x'], rem1)
            z2 = encoder(c['x'], rem2)
            z = (z1 + z2) * 0.5

            # Edge loss (subsample edges)
            max_edges = 100000
            if masked_edges.size(1) > max_edges:
                perm = torch.randperm(masked_edges.size(1), device=device)[:max_edges]
                masked_edges = masked_edges[:, perm]
            if neg_edges.size(1) > max_edges:
                perm = torch.randperm(neg_edges.size(1), device=device)[:max_edges]
                neg_edges = neg_edges[:, perm]

            pos1 = edge_decoder(z1, z2, masked_edges, sigmoid=False)
            pos2 = edge_decoder(z2, z1, masked_edges, sigmoid=False)
            neg1 = edge_decoder(z1, z2, neg_edges, sigmoid=False)
            neg2 = edge_decoder(z2, z1, neg_edges, sigmoid=False)
            loss_edge = (
                F.binary_cross_entropy_with_logits(pos1, torch.ones_like(pos1)) +
                F.binary_cross_entropy_with_logits(pos2, torch.ones_like(pos2)) +
                F.binary_cross_entropy_with_logits(neg1, torch.zeros_like(neg1)) +
                F.binary_cross_entropy_with_logits(neg2, torch.zeros_like(neg2))
            ) / 4

            loss_con = _barlow_twins(z1, z2)

            # Classification — observable nodes only (no label leakage)
            y = c['y'].squeeze() if c['y'].dim() > 1 else c['y']
            obs = c['obs_mask']
            logits = model.forward_classifier(z)
            loss_cls = F.cross_entropy(logits[obs], y[obs])

            # SSL objectives (our contribution)
            loss_stats = F.mse_loss(model.stats_predictor(z1), c['target_stats'])
            loss_residual = F.mse_loss(model.residual_predictor(z2), c['target_residual'])

            loss_total = (loss_edge + w_con * loss_con + loss_cls
                          + w_stats * loss_stats + w_residual * loss_residual)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            el['total'] += loss_total.detach().item()
            el['edge'] += loss_edge.detach().item()
            el['con'] += loss_con.detach().item()
            el['cls'] += loss_cls.detach().item()
            el['stats'] += loss_stats.detach().item()
            el['residual'] += loss_residual.detach().item()

        n = len(cluster_cache)
        epoch_time = time.time() - epoch_start
        log_entry = {
            'epoch': epoch,
            'loss_total': el['total'] / n, 'loss_edge': el['edge'] / n,
            'loss_con': el['con'] / n, 'loss_cls': el['cls'] / n,
            'loss_stats': el['stats'] / n, 'loss_residual': el['residual'] / n,
            'epoch_time_s': round(epoch_time, 2),
        }

        if epoch % 5 == 0:
            val_f1 = _eval(model, cluster_cache, labels, vali_id, device)
            log_entry['val_f1'] = val_f1
            print(f'  Epoch {epoch}/{epochs} | Total: {el["total"]/n:.3f} '
                  f'Edge: {el["edge"]/n:.3f} Con: {el["con"]/n:.3f} '
                  f'Cls: {el["cls"]/n:.3f} Stats: {el["stats"]/n:.3f} '
                  f'Res: {el["residual"]/n:.3f} | Val F1: {val_f1:.4f} | {epoch_time:.1f}s')
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
    all_preds = {}
    with torch.no_grad():
        for c in cluster_cache:
            z = model.encoder(c['x'], c['edge_index'])
            preds = model.forward_classifier(z).argmax(dim=1)
            for local_i, g in enumerate(c['global_indices'].tolist()):
                if g in eval_id_set:
                    all_preds[g] = preds[local_i].item()
    eval_list = [n for n in eval_id.cpu().tolist() if n in all_preds]
    y_pred = [all_preds[n] for n in eval_list]
    y_true = labels[torch.tensor(eval_list)].cpu().tolist()
    return f1_score(y_true, y_pred, average='macro')
