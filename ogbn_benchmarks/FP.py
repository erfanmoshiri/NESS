"""
Feature Propagation (FP) baseline for OGBN-products.

Rossi et al., "On the Unreasonable Effectiveness of Feature Propagation in
Learning on Graphs with Missing Node Features", LoG 2022.

Method: reconstruct missing features by diffusing known features over the graph
(Dirichlet energy minimization = heat diffusion with known values as boundary
conditions). Iterate: x = Ã·x; then reset known entries to their true values.
Then train a standard GNN classifier on the reconstructed features.

Fully scalable: O(E) per iteration, no dense N×N ops. This is the de-facto
standard baseline for learning under missing node features.
"""

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader


def feature_propagation(edge_index, features, observable_mask, num_nodes,
                        num_iterations=40, device='cpu'):
    """
    Reconstruct missing features via iterative diffusion.

    Args:
        edge_index: [2, E] graph edges
        features: [N, D] features (missing rows already zeroed)
        observable_mask: [N] bool, True for observable nodes
        num_iterations: diffusion steps (paper uses ~40)

    Returns:
        propagated: [N, D] reconstructed features
    """
    from torch_geometric.utils import to_scipy_sparse_matrix
    import scipy.sparse as sp
    import numpy as np

    # Build symmetric-normalized adjacency Ã = D^{-1/2} (A) D^{-1/2}
    adj = to_scipy_sparse_matrix(edge_index.cpu(), num_nodes=num_nodes).tocoo()
    deg = np.asarray(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, where=deg > 0, out=deg_inv_sqrt)
    deg_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    norm_adj = (D_inv_sqrt @ adj @ D_inv_sqrt).tocoo()

    # torch sparse normalized adjacency
    idx = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    val = torch.from_numpy(norm_adj.data.astype('float32'))
    A = torch.sparse_coo_tensor(idx, val, (num_nodes, num_nodes)).coalesce().to(device)

    x = features.clone().to(device)
    known = features[observable_mask].clone().to(device)
    obs_idx = observable_mask.nonzero(as_tuple=True)[0].to(device)

    for _ in range(num_iterations):
        x = torch.sparse.mm(A, x)
        x[obs_idx] = known  # reset known entries (boundary condition)

    return x


class SAGEClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return self.classifier(x)


def train_FP(graph, features, labels, observable_id, masked_id, vali_id, test_id,
             num_classes, device, hidden=256, dropout=0.5, lr=0.01,
             weight_decay=5e-4, epochs=200, patience=20, batch_size=512,
             fp_iterations=40, log_path=None):
    """
    Args:
        graph: PyG Data with edge_index
        features: [N, D] ground-truth features
        labels: [N] labels
        observable_id, masked_id, vali_id, test_id: node splits
        num_classes: number of classes

    Returns:
        val_f1, test_f1
    """
    from sklearn.metrics import f1_score

    num_nodes = features.size(0)

    # Zero missing features
    masked_features = features.clone()
    masked_features[masked_id] = 0.0

    obs_mask = torch.zeros(num_nodes, dtype=torch.bool)
    obs_mask[observable_id.cpu()] = True

    # --- Stage 1: reconstruct features via propagation (once, preprocessing) ---
    print(f'  Feature propagation ({fp_iterations} iterations)...')
    fp_start = time.time()
    propagated = feature_propagation(
        graph.edge_index, masked_features, obs_mask, num_nodes,
        num_iterations=fp_iterations, device=device
    ).cpu()
    print(f'    Propagation done in {time.time()-fp_start:.1f}s')

    # --- Stage 2: train GNN classifier on reconstructed features ---
    graph = graph.clone()
    graph.x = propagated
    graph.y = labels

    model = SAGEClassifier(features.size(1), hidden, num_classes, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Small graph: full-batch classifier training ----
    from data_loader import is_small
    if is_small(features.size(0)):
        from fullbatch import train_full_batch
        x = propagated.to(device)
        ei = graph.edge_index.to(device)
        return train_full_batch(
            model, optimizer,
            logits_fn=lambda: model(x, ei),
            labels=labels.to(device),
            observable_id=observable_id.to(device),
            vali_id=vali_id.to(device),
            test_id=test_id.to(device),
            device=device, epochs=epochs, patience=patience,
            log_path=log_path, model_name='FP',
        )

    # ---- Large graph: NeighborLoader mini-batches ----
    train_loader = NeighborLoader(
        graph, num_neighbors=[10, 5], batch_size=batch_size,
        input_nodes=observable_id, shuffle=True, num_workers=0,
    )

    print(f'  Training classifier for up to {epochs} epochs (patience={patience})...')
    best_val_f1 = 0.0
    best_state = None
    epochs_no_improve = 0
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            seed_labels = batch.y[:batch.batch_size].squeeze()
            loss = F.cross_entropy(logits[:batch.batch_size], seed_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        log_entry = {'epoch': epoch, 'loss': avg_loss, 'epoch_time_s': round(epoch_time, 2)}

        if epoch % 5 == 0:
            val_f1 = _eval(model, graph, propagated, labels, vali_id, device)
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

    test_f1 = _eval(model, graph, propagated, labels, test_id, device)
    print(f'  Best Val F1: {best_val_f1:.4f} | Test F1: {test_f1:.4f}')
    print(f'  Total training time: {total_train_time:.1f}s ({total_train_time/60:.1f} min)')
    return best_val_f1, test_f1


def _eval(model, graph, propagated, labels, eval_id, device, batch_size=1024):
    from sklearn.metrics import f1_score
    model.eval()

    eval_graph = graph.clone()
    eval_graph.x = propagated

    loader = NeighborLoader(
        eval_graph, num_neighbors=[10, 5], batch_size=batch_size,
        input_nodes=eval_id.cpu(), shuffle=False, num_workers=0,
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            all_preds.append(logits[:batch.batch_size].argmax(dim=1).cpu())
            all_labels.append(batch.y[:batch.batch_size].squeeze().cpu())

    return f1_score(torch.cat(all_labels).numpy(),
                    torch.cat(all_preds).numpy(), average='macro')
