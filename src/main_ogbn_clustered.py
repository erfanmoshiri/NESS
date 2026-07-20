"""
Training script for OGBN-products using PyG's ClusterLoader (RECOMMENDED).

This version uses PyTorch Geometric's built-in ClusterData/ClusterLoader
instead of manual clustering. Much simpler and better optimized!

Benefits over manual clustering:
- Uses optimized METIS partitioning
- Automatically caches partitions
- Handles edge cases properly
- ~10 lines vs ~300 lines of clustering code
"""

import argparse
from torch import optim
import sys
import os
import json
import datetime
sys.path.insert(0, '..')
from models.NESS import *
from src.utils_ogbn import *
from src.utils import set_random_seed, MaskEdge
from src.cache_manager import get_or_create_masks, get_or_create_ssl_targets
from torch_geometric.loader import ClusterData, ClusterLoader
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ogbn-products')
parser.add_argument('--seed', type=int, default=72)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=5e-5)

# Clustering
parser.add_argument('--num_parts', type=int, default=50,
                    help='Number of graph partitions (default: 50, ~49k nodes/partition)')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Number of clusters per batch (default: 1)')

# Missingness
parser.add_argument('--missingness', type=str, default='MCAR',
                    choices=['MCAR', 'MAR', 'MNAR'])
parser.add_argument('--miss_rate', type=float, default=0.4)

# Model
parser.add_argument('--encoder_channels', type=int, default=256)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--decoder_channels', type=int, default=64)
parser.add_argument('--encoder_dropout', type=float, default=0.5)
parser.add_argument('--decoder_dropout', type=float, default=0.3)
parser.add_argument('--p', type=float, default=0.7)
parser.add_argument('--temp', type=float, default=0.2)

parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--data_root', type=str, default='../data/ogbn_products')


def setup_run_dir(args):
    """Create a timestamped run directory under ../runs/."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"ogbn_clustered_{args.missingness}_{args.miss_rate}_{timestamp}"
    run_dir = os.path.join('..', 'runs', run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    config = vars(args)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    return run_dir


def main(args):
    print("="*70)
    print("MATE EMBEDDINGS - OGBN-PRODUCTS (PyG ClusterLoader)")
    print("="*70)
    print(f"Partitions: {args.num_parts} (~{2449029//args.num_parts:,} nodes/partition)")
    print(f"Missingness: {args.missingness} @ {args.miss_rate*100:.0f}%")
    print(f"Device: cuda:{args.device}" if args.cuda else "Device: cpu")
    print("="*70 + "\n")

    run_dir = setup_run_dir(args)
    log_path = os.path.join(run_dir, 'training_log.jsonl')
    print(f"Run directory: {run_dir}\n")

    set_random_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if args.cuda else 'cpu')

    # ==================== LOAD DATA ====================
    print('[1/4] Loading OGBN-products...')
    graph, adj, adj_norm, features, labels, split_idx = load_ogbn_products(root=args.data_root)
    num_nodes = features.size(0)
    num_features = features.size(1)
    num_classes = labels.max().item() + 1
    print(f'  Loaded: {num_nodes:,} nodes, {num_features} dims, {num_classes} classes\n')

    # ==================== SIMULATE MISSINGNESS (WITH CACHE) ====================
    print(f'[2/4] Loading/Computing {args.missingness} missingness masks...')
    all_nodes = torch.arange(num_nodes, dtype=torch.long)

    # Try to load from cache, or create new
    observable_id, masked_id = get_or_create_masks(
        args.data_root, all_nodes, adj, features, labels,
        args.missingness, args.miss_rate, args.seed
    )

    split_point = len(masked_id) // 2
    vali_id = masked_id[:split_point]
    test_id = masked_id[split_point:]
    print(f'  Observable: {len(observable_id):,}, Val: {len(vali_id):,}, Test: {len(test_id):,}\n')

    # Prepare masked features
    masked_features = features.clone()
    masked_features[masked_id] = 0.0

    # ==================== COMPUTE SSL TARGETS (WITH CACHE) ====================
    print('[3/4] Loading/Computing SSL objectives (YOUR MAIN CONTRIBUTION!)...')

    # Try to load from cache, or create new
    target_stats, target_residual = get_or_create_ssl_targets(
        args.data_root, adj, features, masked_features, observable_id,
        args.missingness, args.miss_rate, args.seed
    )
    print(f'  ✓ SSL targets ready (~2GB, loaded once for all clusters)\n')

    # Update graph with masked features
    graph.x = masked_features

    # ==================== CREATE CLUSTER LOADER (PyG) ====================
    print('[4/4] Creating ClusterLoader (PyG built-in)...')
    print(f'  Using PyG ClusterData with METIS partitioning...')

    cluster_data = ClusterData(
        graph,
        num_parts=args.num_parts,
        save_dir=args.data_root,  # Cache partitions
        log=True
    )

    cluster_loader = ClusterLoader(
        cluster_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    print(f'  ✓ Created loader with {len(cluster_data)} partitions\n')

    # ==================== BUILD MODEL ====================
    print('[5/5] Building model...')
    encoder = GNNEncoder(
        in_channels=num_features,
        hidden_channels=args.encoder_channels,
        out_channels=args.hidden_channels,
        num_layers=2,
        dropout=args.encoder_dropout,
        layer='gcn',
        activation='elu'
    )

    edge_decoder = EdgeDecoder(
        in_channels=args.hidden_channels,
        hidden_channels=args.decoder_channels,
        num_layers=2,
        dropout=args.decoder_dropout
    )

    projector = Projector(
        in_channels=args.hidden_channels,
        hidden_channels=args.encoder_channels,
        out_channels=num_features,
        num_layers=2,
        dropout=args.decoder_dropout
    )

    con_projector = Con_Projector(
        in_channels=args.hidden_channels,
        hidden_channels=args.encoder_channels,
        out_channels=num_features,
        num_layers=2,
        dropout=args.decoder_dropout
    )

    mask_edge = MaskEdge(p=args.p)

    model = Model(
        encoder, edge_decoder, projector, con_projector,
        temp=args.temp,
        pos_weight_tensor=None,
        neg_weight_tensor=None,
        mask=mask_edge,
        feature_dim=num_features,
        hidden_dim=args.hidden_channels,
        num_classes=num_classes
    )

    print(f'  ✓ Model created\n')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)

    # ==================== TRAINING LOOP ====================
    print("\n" + "="*70)
    print("STARTING TRAINING (PyG ClusterLoader)")
    print("="*70)

    best_val_f1 = 0.0
    best_model_path = os.path.join(run_dir, 'best_model.pt')
    loss_history = {'total': [], 'classification': []}

    import time
    from torch_geometric.utils import add_self_loops, negative_sampling

    # Move SSL targets to device (only 2GB, stays in memory)
    target_stats = target_stats.to(device)
    target_residual = target_residual.to(device)

    # Pre-cache per-cluster data that is fixed across all epochs
    # Use a non-shuffled loader so cluster index i matches partptr[i]
    print("Pre-caching cluster data...")
    cache_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False, num_workers=0)
    node_perm = cluster_data.partition.node_perm
    partptr = cluster_data.partition.partptr
    cluster_cache = []
    for i, batch_data in enumerate(cache_loader):
        batch_data = batch_data.to(device)
        edge_index = batch_data.edge_index
        global_indices = node_perm[partptr[i]:partptr[i+1]].to(device)
        aug_edge_index, _ = add_self_loops(edge_index)
        cluster_cache.append({
            'x': batch_data.x,
            'y': batch_data.y,
            'edge_index': edge_index,
            'aug_edge_index': aug_edge_index,
            'global_indices': global_indices,
            'target_stats': target_stats[global_indices],
            'target_residual': target_residual[global_indices],
            'num_nodes': batch_data.num_nodes,
        })
    print(f"  Cached {len(cluster_cache)} clusters\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_losses = {'total': 0.0, 'edge': 0.0, 'contrastive': 0.0, 'classification': 0.0, 'stats': 0.0, 'residual': 0.0}
        num_batches = 0

        # Iterate over pre-cached clusters (shuffle each epoch)
        import random
        random.shuffle(cluster_cache)
        for cluster in cluster_cache:
            edge_index = cluster['edge_index']
            global_indices = cluster['global_indices']

            # Two views via different edge masks (proper contrastive learning)
            remaining_edges_1, masked_edges = mask_edge(edge_index)
            remaining_edges_2, _ = mask_edge(edge_index)

            # Generate negative edges on GPU
            num_neg_samples = min(masked_edges.size(1), 50000)
            neg_edges = negative_sampling(
                cluster['aug_edge_index'],
                num_nodes=cluster['num_nodes'],
                num_neg_samples=num_neg_samples,
                method='sparse'
            )

            # Two views with different edge masks
            z_1 = encoder(cluster['x'], remaining_edges_1)
            z_2 = encoder(cluster['x'], remaining_edges_2)
            z = (z_1 + z_2) * 0.5

            # 1. Edge reconstruction loss (single GPU call, no inner loop)
            max_edges = 100000
            if masked_edges.size(1) > max_edges:
                perm = torch.randperm(masked_edges.size(1), device=device)[:max_edges]
                masked_edges_subset = masked_edges[:, perm]
            else:
                masked_edges_subset = masked_edges

            if neg_edges.size(1) > max_edges:
                perm = torch.randperm(neg_edges.size(1), device=device)[:max_edges]
                neg_edges_subset = neg_edges[:, perm]
            else:
                neg_edges_subset = neg_edges

            pos_out_1 = edge_decoder(z_1, z_2, masked_edges_subset, sigmoid=False)
            pos_out_2 = edge_decoder(z_2, z_1, masked_edges_subset, sigmoid=False)
            neg_out_1 = edge_decoder(z_1, z_2, neg_edges_subset, sigmoid=False)
            neg_out_2 = edge_decoder(z_2, z_1, neg_edges_subset, sigmoid=False)

            loss_edge = (
                F.binary_cross_entropy_with_logits(pos_out_1, torch.ones_like(pos_out_1)) +
                F.binary_cross_entropy_with_logits(pos_out_2, torch.ones_like(pos_out_2)) +
                F.binary_cross_entropy_with_logits(neg_out_1, torch.zeros_like(neg_out_1)) +
                F.binary_cross_entropy_with_logits(neg_out_2, torch.zeros_like(neg_out_2))
            ) / 4

            # 2. Barlow Twins contrastive loss (memory efficient + prevents redundancy)
            # Cross-correlation between views: diagonal → 1, off-diagonal → 0
            # Memory: O(D²) where D=128 - very efficient!

            # Normalize features batch-wise (z-score normalization)
            z_1_norm = (z_1 - z_1.mean(0)) / (z_1.std(0) + 1e-6)
            z_2_norm = (z_2 - z_2.mean(0)) / (z_2.std(0) + 1e-6)

            # Cross-correlation matrix [D, D]
            N = z_1.size(0)
            c = torch.mm(z_1_norm.T, z_2_norm) / N  # [128, 128]

            # Loss: on-diagonal → 1, off-diagonal → 0
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()  # Want diagonal = 1
            off_diag = c.fill_diagonal_(0).pow_(2).sum()         # Want off-diagonal = 0

            lambda_param = 0.005  # Weight for off-diagonal term (standard value)
            loss_con = on_diag + lambda_param * off_diag

            # 3. Classification loss
            logits = model.forward_classifier(z)
            labels_batch = cluster['y'].squeeze() if cluster['y'].dim() > 1 else cluster['y']
            loss_cls = F.cross_entropy(logits, labels_batch)

            # SKIP: Feature reconstruction loss (as requested)

            # 4. SSL Stats loss (YOUR MAIN CONTRIBUTION!)
            # Predict neighborhood embedding spread from view 1
            if model.stats_predictor is not None:
                pred_stats = model.stats_predictor(z_1)
                loss_stats = F.mse_loss(pred_stats, cluster['target_stats'])
            else:
                loss_stats = torch.tensor(0.0, device=device)

            if model.residual_predictor is not None:
                pred_residual = model.residual_predictor(z_2)
                loss_residual = F.mse_loss(pred_residual, cluster['target_residual'])
            else:
                loss_residual = torch.tensor(0.0, device=device)

            # Total loss with SSL (same weighting as original: 100x stats, 50x residual)
            loss_total = loss_edge + 10 * loss_con + loss_cls + (100 * loss_stats) + (50 * loss_residual)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # Track losses (detach to avoid keeping computation graph)
            epoch_losses['total'] += loss_total.detach().item()
            epoch_losses['edge'] += loss_edge if isinstance(loss_edge, float) else loss_edge.detach().item()
            epoch_losses['contrastive'] += loss_con.detach().item()
            epoch_losses['classification'] += loss_cls.detach().item()
            epoch_losses['stats'] += loss_stats.detach().item()
            epoch_losses['residual'] += loss_residual.detach().item()
            num_batches += 1

            del z_1, z_2, z, logits
            del loss_total, loss_edge, loss_con, loss_cls, loss_stats, loss_residual

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        loss_history['total'].append(epoch_losses['total'])
        loss_history['classification'].append(epoch_losses['classification'])

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:3d}/{args.epochs} | Total: {epoch_losses['total']:.4f} | "
              f"Edge: {epoch_losses['edge']:.4f} Con: {epoch_losses['contrastive']:.4f} "
              f"Cls: {epoch_losses['classification']:.4f} Stats: {epoch_losses['stats']:.4f} "
              f"Res: {epoch_losses['residual']:.4f} | {epoch_time:.1f}s")

        # Write epoch log
        log_entry = {
            'epoch': epoch,
            'loss_total': epoch_losses['total'],
            'loss_edge': epoch_losses['edge'],
            'loss_contrastive': epoch_losses['contrastive'],
            'loss_classification': epoch_losses['classification'],
            'loss_stats': epoch_losses['stats'],
            'loss_residual': epoch_losses['residual'],
            'epoch_time_s': round(epoch_time, 2),
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Validation every 20 epochs
        if epoch % 20 == 0:
            print(f"\nEpoch {epoch}: Evaluating...")
            model.eval()

            # Full graph inference for evaluation
            with torch.no_grad():
                # Process in large batches to avoid OOM
                batch_size_eval = 100000
                all_logits = []

                for start_idx in range(0, num_nodes, batch_size_eval):
                    end_idx = min(start_idx + batch_size_eval, num_nodes)
                    batch_nodes = torch.arange(start_idx, end_idx)  # Keep on CPU for indexing

                    # Get subgraph for this batch
                    batch_edge_mask = (graph.edge_index[0] >= start_idx) & \
                                     (graph.edge_index[0] < end_idx) & \
                                     (graph.edge_index[1] >= start_idx) & \
                                     (graph.edge_index[1] < end_idx)
                    batch_edge_index = graph.edge_index[:, batch_edge_mask].to(device) - start_idx

                    batch_features = masked_features[batch_nodes].to(device)  # Index on CPU, then move to GPU

                    z_batch = encoder(batch_features, batch_edge_index)
                    logits_batch = model.forward_classifier(z_batch)
                    all_logits.append(logits_batch.cpu())

                # Concatenate all logits
                all_logits = torch.cat(all_logits, dim=0)

                # Evaluate on validation set
                preds_val = all_logits[vali_id].argmax(dim=1)
                true_val = labels[vali_id]

                val_f1 = f1_score(true_val.cpu().numpy(), preds_val.numpy(), average='macro')

                print(f'  Val F1: {val_f1:.4f}')

                # Append val_f1 to last log entry
                with open(log_path, 'a') as f:
                    f.write(json.dumps({'epoch': epoch, 'val_f1': val_f1}) + '\n')

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(model.state_dict(), best_model_path)
                    print(f'  ✓ Saved best model (val F1: {val_f1:.4f})')

    # ==================== FINAL EVALUATION ====================
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from run dir")
    else:
        print("No saved model found. Using final model state.")

    model.eval()

    with torch.no_grad():
        batch_size_eval = 100000
        all_logits = []

        for start_idx in range(0, num_nodes, batch_size_eval):
            end_idx = min(start_idx + batch_size_eval, num_nodes)
            batch_nodes = torch.arange(start_idx, end_idx)

            batch_edge_mask = (graph.edge_index[0] >= start_idx) & \
                             (graph.edge_index[0] < end_idx) & \
                             (graph.edge_index[1] >= start_idx) & \
                             (graph.edge_index[1] < end_idx)
            batch_edge_index = graph.edge_index[:, batch_edge_mask].to(device) - start_idx
            batch_features = masked_features[batch_nodes].to(device)

            z_batch = encoder(batch_features, batch_edge_index)
            logits_batch = model.forward_classifier(z_batch)
            all_logits.append(logits_batch.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        preds_test = all_logits[test_id].argmax(dim=1)
        true_test = labels[test_id]

    test_f1_macro = f1_score(true_test.cpu().numpy(), preds_test.numpy(), average='macro')
    test_f1_micro = f1_score(true_test.cpu().numpy(), preds_test.numpy(), average='micro')

    print(f'\nTest Macro-F1: {test_f1_macro:.4f}')
    print(f'Test Micro-F1: {test_f1_micro:.4f}')
    print(f'Best Val F1:   {best_val_f1:.4f}')
    print("="*70)

    final_results = {
        'dataset': args.dataset,
        'missingness': args.missingness,
        'miss_rate': args.miss_rate,
        'num_parts': args.num_parts,
        'seed': args.seed,
        'epochs': args.epochs,
        'test_f1_macro': test_f1_macro,
        'test_f1_micro': test_f1_micro,
        'best_val_f1': best_val_f1,
    }
    with open(os.path.join(run_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f'\n✓ Results saved to: {run_dir}')


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    if args.cuda:
        torch.cuda.set_device(f'cuda:{args.device}')
    main(args)
