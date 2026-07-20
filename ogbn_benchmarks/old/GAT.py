"""
GAT baseline for OGBN-products.

Zero-fills missing nodes, trains with cross-entropy on observable nodes via
NeighborLoader mini-batches, evaluates macro-F1 on masked nodes.
"""

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes,
                 heads=4, dropout=0.6, alpha=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                             dropout=dropout, negative_slope=alpha)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1,
                             concat=False, dropout=dropout, negative_slope=alpha)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def predict(self, x, edge_index):
        return self.classifier(self.forward(x, edge_index))


def train_GAT(graph, features, labels, observable_id, masked_id, vali_id, test_id,
              num_classes, device, hidden=256, heads=2, dropout=0.3, alpha=0.2,
              lr=0.005, weight_decay=5e-4, epochs=200, patience=20, batch_size=1024,
              log_path=None):
    """
    Args:
        graph: PyG Data object with edge_index
        features: [N, 100] ground-truth embeddings
        labels: [N] class labels
        observable_id: visible node indices (train)
        masked_id: all masked node indices
        vali_id: validation split of masked nodes
        test_id: test split of masked nodes
        num_classes: 47 for OGBN-products

    Returns:
        val_f1, test_f1
    """
    from sklearn.metrics import f1_score

    masked_features = features.clone()
    masked_features[masked_id] = 0.0

    graph = graph.clone()
    graph.x = masked_features
    graph.y = labels

    model = GAT(features.size(1), hidden, num_classes, heads, dropout, alpha).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = NeighborLoader(
        graph,
        num_neighbors=[10, 5],
        batch_size=batch_size,
        input_nodes=observable_id,
        shuffle=True,
        num_workers=0,
    )

    print(f'  Training GAT for up to {epochs} epochs (heads={heads}, patience={patience})...')
    best_val_f1 = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model.predict(batch.x, batch.edge_index)
            seed_labels = batch.y[:batch.batch_size].squeeze()
            loss = F.cross_entropy(logits[:batch.batch_size], seed_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        log_entry = {'epoch': epoch, 'loss': avg_loss, 'epoch_time_s': round(epoch_time, 2)}

        if epoch % 5 == 0:
            val_f1 = _eval(model, graph, masked_features, labels, vali_id, device, batch_size)
            log_entry['val_f1'] = val_f1
            print(f'  Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}')
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

    if best_state is not None:
        model.load_state_dict(best_state)

    test_f1 = _eval(model, graph, masked_features, labels, test_id, device, batch_size)
    print(f'  Best Val F1: {best_val_f1:.4f} | Test F1: {test_f1:.4f}')
    return best_val_f1, test_f1


def _eval(model, graph, masked_features, labels, eval_id, device, batch_size=1024):
    from sklearn.metrics import f1_score
    model.eval()

    eval_graph = graph.clone()
    eval_graph.x = masked_features

    loader = NeighborLoader(
        eval_graph,
        num_neighbors=[10, 5],
        batch_size=batch_size,
        input_nodes=eval_id.cpu(),
        shuffle=False,
        num_workers=0,
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model.predict(batch.x, batch.edge_index)
            preds = logits[:batch.batch_size].argmax(dim=1).cpu()
            true = batch.y[:batch.batch_size].squeeze().cpu()
            all_preds.append(preds)
            all_labels.append(true)

    return f1_score(
        torch.cat(all_labels).numpy(),
        torch.cat(all_preds).numpy(),
        average='macro'
    )
