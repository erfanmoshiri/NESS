"""
Shared full-batch training loop for small graphs (<100k nodes).

Small graphs don't need NeighborLoader/ClusterLoader — the whole graph fits in
memory and one forward pass per epoch is faster and more stable. This helper
runs that loop for any model exposing a `logits_fn(...) -> [N, C]` forward.

Used by GraphSAGE, FP, PaGCN, MATE small-graph paths so the protocol
(observable-only CE loss, val-F1 early stopping, best-state restore) is identical.
"""

import json
import time
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def train_full_batch(model, optimizer, logits_fn, labels,
                     observable_id, vali_id, test_id, device,
                     epochs=200, patience=20, log_path=None, model_name='model'):
    """
    Args:
        model: nn.Module (already on device)
        optimizer: optimizer over model params
        logits_fn: callable() -> [N, C] logits for ALL nodes (closure over graph)
        labels: [N] labels on device
        observable_id, vali_id, test_id: node splits on device
        device: torch device

    Returns:
        val_f1, test_f1
    """
    labels = labels.squeeze() if labels.dim() > 1 else labels

    print(f'  Training {model_name} (full-batch) for up to {epochs} epochs (patience={patience})...')
    best_val_f1 = 0.0
    best_state = None
    epochs_no_improve = 0
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()

        optimizer.zero_grad()
        logits = logits_fn()
        loss = F.cross_entropy(logits[observable_id], labels[observable_id])
        loss.backward()
        optimizer.step()

        epoch_time = time.time() - epoch_start
        log_entry = {'epoch': epoch, 'loss': loss.item(), 'epoch_time_s': round(epoch_time, 3)}

        if epoch % 5 == 0:
            val_f1 = _eval_full(model, logits_fn, labels, vali_id)
            log_entry['val_f1'] = val_f1
            print(f'  Epoch {epoch}/{epochs} | Loss: {loss.item():.4f} | Val F1: {val_f1:.4f} | {epoch_time:.2f}s')
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

    test_f1 = _eval_full(model, logits_fn, labels, test_id)
    print(f'  Best Val F1: {best_val_f1:.4f} | Test F1: {test_f1:.4f}')
    print(f'  Total training time: {total_train_time:.1f}s')
    return best_val_f1, test_f1


def _eval_full(model, logits_fn, labels, eval_id):
    model.eval()
    with torch.no_grad():
        logits = logits_fn()
        preds = logits[eval_id].argmax(dim=1).cpu().numpy()
        true = labels[eval_id].cpu().numpy()
    return f1_score(true, preds, average='macro')
