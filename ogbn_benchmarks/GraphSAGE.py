"""
GraphSAGE baseline for OGBN-products.

Zero-fills missing node embeddings, trains with cross-entropy on observable nodes,
evaluates classification F1 on masked nodes. Uses NeighborLoader for scalability.
"""

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader


class GraphSAGE(nn.Module):
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
        return x

    def predict(self, x, edge_index):
        z = self.forward(x, edge_index)
        return self.classifier(z)


def train_GraphSAGE(graph, features, labels, observable_id, masked_id, vali_id, test_id,
                    num_classes, device, hidden=256, dropout=0.5, lr=0.01,
                    weight_decay=5e-4, epochs=200, patience=20, neighbors=[10, 5],
                    batch_size=512, log_path=None):
    """
    Args:
        graph: PyG Data object with edge_index
        features: [N, 100] ground-truth embeddings
        labels: [N] class labels
        observable_id: indices of visible nodes (training nodes)
        masked_id: all masked node indices
        vali_id: validation split of masked nodes
        test_id: test split of masked nodes
        num_classes: 47 for OGBN-products
        device: torch device

    Returns:
        val_f1, test_f1
    """
    from sklearn.metrics import f1_score
    from data_loader import is_small

    # Zero-fill missing nodes
    masked_features = features.clone()
    masked_features[masked_id] = 0.0

    # Attach to graph for NeighborLoader
    graph = graph.clone()
    graph.x = masked_features
    graph.y = labels

    model = GraphSAGE(features.size(1), hidden, num_classes, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Small graph: full-batch training ----
    if is_small(features.size(0)):
        from fullbatch import train_full_batch
        x = masked_features.to(device)
        ei = graph.edge_index.to(device)
        return train_full_batch(
            model, optimizer,
            logits_fn=lambda: model(x, ei),
            labels=labels.to(device),
            observable_id=observable_id.to(device),
            vali_id=vali_id.to(device),
            test_id=test_id.to(device),
            device=device, epochs=epochs, patience=patience,
            log_path=log_path, model_name='GraphSAGE',
        )

    # ---- Large graph: NeighborLoader mini-batches ----
    train_loader = NeighborLoader(
        graph,
        num_neighbors=neighbors,
        batch_size=batch_size,
        input_nodes=observable_id,
        shuffle=True,
        num_workers=0,
    )

    print(f'  Training GraphSAGE for up to {epochs} epochs (patience={patience})...')
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
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        log_entry = {'epoch': epoch, 'loss': avg_loss, 'epoch_time_s': round(epoch_time, 2)}

        if epoch % 5 == 0:
            val_f1 = _eval(model, graph, masked_features, labels, vali_id, device)
            log_entry['val_f1'] = val_f1
            print(f'  Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}')
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'  Early stopping at epoch {epoch} (no improvement for {patience} evals)')
                    if log_path:
                        with open(log_path, 'a') as f:
                            f.write(json.dumps(log_entry) + '\n')
                    break

        if log_path:
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

    if best_state is not None:
        model.load_state_dict(best_state)

    test_f1 = _eval(model, graph, masked_features, labels, test_id, device)
    print(f'  Best Val F1: {best_val_f1:.4f} | Test F1: {test_f1:.4f}')
    return best_val_f1, test_f1


def _eval(model, graph, masked_features, labels, eval_id, device):
    from sklearn.metrics import f1_score
    model.eval()

    # Eval loader around eval nodes
    eval_graph = graph.clone()
    eval_graph.x = masked_features

    loader = NeighborLoader(
        eval_graph,
        num_neighbors=[10, 5],
        batch_size=512,
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

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return f1_score(all_labels, all_preds, average='macro')
