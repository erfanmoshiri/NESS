"""
Training script for OGBN-products dataset with MATE_embeddings model.

Differences from main_fixed.py (Cora/CiteSeer):
- Uses MATE_embeddings.py model (adapted SSL for embeddings)
- Node-level masking (entire 100-dim embeddings) instead of feature-level
- Classification evaluation (F1-score) instead of Recall@K/NDCG
- Supports MCAR/MAR/MNAR missingness simulation
- Larger scale: 2.4M nodes vs 2.7k for Cora

Memory requirements: ~12GB GPU
For lower memory (3-4GB GPU), use: main_ogbn_clustered.py (PyG ClusterLoader)
"""

import argparse
from torch import optim
import sys
sys.path.insert(0, '..')
from models.MATE_embeddings import *
from src.utils_ogbn import *
from src.utils import set_random_seed
from tqdm import tqdm
from torch_geometric.data import Data
import torch
import os
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ogbn-products')
parser.add_argument('--method_name', type=str, default='MATE_Embeddings')
parser.add_argument('--seed', type=int, default=72)
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of training epochs (default: 200)')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=5e-5)

# Missingness simulation
parser.add_argument('--missingness', type=str, default='MCAR',
                    choices=['MCAR', 'MAR', 'MNAR'],
                    help='Missingness mechanism')
parser.add_argument('--miss_rate', type=float, default=0.4,
                    help='Fraction of nodes to mask (0.2, 0.4, 0.6)')

# Model architecture
parser.add_argument("--layer", nargs="?", default="gcn",
                    help="GNN layer, (default: gcn)")
parser.add_argument("--encoder_activation", nargs="?", default="elu",
                    help="Activation function for GNN encoder")
parser.add_argument('--encoder_channels', type=int, default=256,
                    help='Channels of GNN encoder layers (default: 256)')
parser.add_argument('--hidden_channels', type=int, default=128,
                    help='Channels of hidden representation (default: 128)')
parser.add_argument('--decoder_channels', type=int, default=64,
                    help='Channels of decoder layers (default: 64)')
parser.add_argument('--encoder_layers', type=int, default=2,
                    help='Number of layers for encoder')
parser.add_argument('--decoder_layers', type=int, default=2,
                    help='Number of layers for decoders')
parser.add_argument('--encoder_dropout', type=float, default=0.5,
                    help='Dropout probability of encoder')
parser.add_argument('--decoder_dropout', type=float, default=0.3,
                    help='Dropout probability of decoder')
parser.add_argument('--bn', type=bool, default=False)

# Training
parser.add_argument('--p', type=float, default=0.7,
                    help='Edge masking probability for contrastive learning')
parser.add_argument('--temp', type=float, default=0.2,
                    help='Temperature for contrastive loss')
parser.add_argument('--cuda', action='store_true',
                    default=torch.cuda.is_available())
parser.add_argument('--device', type=int, default=0)

# Data
parser.add_argument('--data_root', type=str, default='../data/ogbn_products',
                    help='Root directory for OGBN-products dataset')


def main(args):
    print("="*70)
    print("MATE EMBEDDINGS - OGBN-PRODUCTS TRAINING")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Missingness: {args.missingness} @ {args.miss_rate*100:.0f}% rate")
    print(f"Epochs: {args.epochs}")
    print(f"Device: cuda:{args.device}" if args.cuda else "Device: cpu")
    print("="*70 + "\n")

    set_random_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if args.cuda else 'cpu')

    # ==================== LOAD DATA ====================
    print('[1/6] Loading OGBN-products dataset...')
    graph, adj, adj_norm, features, labels, split_idx = load_ogbn_products(root=args.data_root)

    num_nodes = features.size(0)
    num_features = features.size(1)  # 100 embedding dims
    num_classes = labels.max().item() + 1  # 47 categories

    print(f'  Loaded: {num_nodes:,} nodes, {num_features} dims, {num_classes} classes')

    # ==================== SIMULATE MISSINGNESS ====================
    print(f'\n[2/6] Simulating {args.missingness} missingness...')

    # Use all nodes for missingness simulation (not just official splits)
    all_nodes = torch.arange(num_nodes, dtype=torch.long)

    observable_id, masked_id = simulate_node_missingness(
        all_nodes, adj, features, labels,
        missingness_type=args.missingness,
        miss_rate=args.miss_rate,
        seed=args.seed
    )

    print(f'  Observable: {len(observable_id):,} ({len(observable_id)/num_nodes:.1%})')
    print(f'  Masked: {len(masked_id):,} ({len(masked_id)/num_nodes:.1%})')

    # Further split masked nodes into validation and test
    split_point = len(masked_id) // 2
    vali_id = masked_id[:split_point]
    test_id = masked_id[split_point:]
    print(f'  Validation: {len(vali_id):,}')
    print(f'  Test: {len(test_id):,}')

    # ==================== PREPARE MASKED FEATURES ====================
    print('\n[3/6] Preparing masked features...')

    # Zero out masked nodes (entire 100-dim embeddings)
    masked_features = features.clone()
    masked_features[masked_id] = 0.0
    print(f'  Masked {len(masked_id):,} node embeddings (set to zero)')

    # ==================== COMPUTE FIXED FEATURES ====================
    print('\n[4/6] Computing fixed features via neighbor averaging...')
    fixed_features = compute_fixed_features(adj, masked_features, observable_id, masked_id)
    print(f'  Fixed features computed for {len(masked_id):,} missing nodes')

    # ==================== COMPUTE SSL TARGETS ====================
    print('\n[5/6] Computing SSL objectives (embedding-adapted)...')

    # Stats: neighborhood embedding spread (std)
    target_stats = compute_neighborhood_embedding_stats(adj, features, observable_id)
    print(f'  ✓ Neighborhood stats (std) computed')

    # Residual: deviation from neighborhood centroid
    target_residual = compute_neighborhood_centroid_residual(adj, masked_features, observable_id)
    print(f'  ✓ Centroid residuals computed')

    # ==================== CREATE DATA OBJECTS ====================
    edge_index = graph.edge_index

    # View 1: Masked features
    data_1 = Data(x=masked_features, y=labels, edge_index=edge_index)

    # View 2: Same as View 1 (simplified, could use diffusion if needed)
    data_2 = Data(x=masked_features, y=labels, edge_index=edge_index)

    # ==================== BUILD MODEL ====================
    print('\n[6/6] Building MATE_embeddings model...')

    encoder = GNNEncoder(
        in_channels=num_features,  # 100
        hidden_channels=args.encoder_channels,
        out_channels=args.hidden_channels,
        num_layers=args.encoder_layers,
        dropout=args.encoder_dropout,
        bn=args.bn,
        layer=args.layer,
        activation=args.encoder_activation
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
        out_channels=num_features,  # Reconstruct 100-dim embeddings
        num_layers=args.decoder_layers,
        dropout=args.decoder_dropout
    )

    con_projector = Con_Projector(
        in_channels=args.hidden_channels,
        hidden_channels=args.encoder_channels,
        out_channels=num_features,
        num_layers=2,
        dropout=args.decoder_dropout
    )

    mask = MaskEdge(p=args.p)

    model = Model(
        encoder, edge_decoder, projector, con_projector,
        temp=args.temp,
        pos_weight_tensor=None,  # No reweighting for embeddings
        neg_weight_tensor=None,
        mask=mask,
        feature_dim=num_features,  # 100
        hidden_dim=args.hidden_channels,
        num_classes=num_classes  # 47 (NEW: classification head)
    )

    print(f'  ✓ Model created with {num_classes} classes')
    print(f'  ✓ SSL predictors: stats + residual')
    print(f'  ✓ Classification head: {args.hidden_channels} → {num_classes}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def scheduler_fn(epoch):
        return (1 + np.cos((epoch) * np.pi / args.epochs)) * 0.5

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fn)

    # ==================== MOVE TO DEVICE ====================
    if args.cuda:
        data_1 = data_1.to(device)
        data_2 = data_2.to(device)
        model = model.to(device)
        fixed_features = fixed_features.to(device)
        target_stats = target_stats.to(device)
        target_residual = target_residual.to(device)
        adj_norm = adj_norm.to(device)
        observable_id = observable_id.to(device)
        masked_id = masked_id.to(device)
        vali_id = vali_id.to(device)
        test_id = test_id.to(device)

    # ==================== TRAINING LOOP ====================
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    best_val_f1 = 0.0
    loss_history = {
        'total': [],
        'edge': [],
        'contrastive': [],
        'reconstruction': [],
        'classification': [],
        'stats': [],
        'residual': []
    }

    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()

        loss_total, loss_edge, loss_con, loss_recon, loss_cls, loss_stats, loss_res = \
            model.train_one_epoch(
                data_1, data_2, adj_norm, fixed_features,
                observable_id, masked_id,
                target_stats=target_stats,
                target_residual=target_residual,
                labels=labels,
                use_classification=True
            )

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        # Track losses
        loss_history['total'].append(loss_total.item())
        loss_history['edge'].append(loss_edge.item())
        loss_history['contrastive'].append(loss_con.item())
        loss_history['reconstruction'].append(loss_recon.item())
        loss_history['classification'].append(loss_cls.item())
        loss_history['stats'].append(loss_stats.item())
        loss_history['residual'].append(loss_res.item())

        # Validation every 20 epochs
        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                # Get embeddings (fused from both views)
                x_1_ = data_1.x
                zero_ = torch.zeros_like(fixed_features)
                zero = torch.zeros_like(fixed_features)
                zero[masked_id] = zero_[masked_id] + fixed_features[masked_id]
                x_1__ = x_1_ + zero
                x_1 = torch.mm(adj_norm, x_1__)

                z_1 = model.encoder(x_1, edge_index)
                z_2 = model.encoder(data_2.x, edge_index)
                z = (z_1 + z_2) * 0.5

                # Get class logits
                logits = model.forward_classifier(z)

                # Evaluate on validation set
                preds = logits[vali_id].argmax(dim=1)
                true_labels = labels[vali_id]

                val_f1 = f1_score(
                    true_labels.cpu().numpy(),
                    preds.cpu().numpy(),
                    average='macro'
                )

                print(f'\nEpoch {epoch:3d} | F1: {val_f1:.4f} | '
                      f'Loss: {loss_total.item():.4f} '
                      f'(edge={loss_edge.item():.3f}, con={loss_con.item():.3f}, '
                      f'recon={loss_recon.item():.3f}, cls={loss_cls.item():.3f}, '
                      f'stats={loss_stats.item():.3f}, res={loss_res.item():.3f})')

                # Save best model
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    os.makedirs('../best_model', exist_ok=True)
                    torch.save(
                        model.state_dict(),
                        os.path.join('../best_model',
                                    f'ogbn_{args.missingness}_{args.miss_rate}.pt')
                    )
                    torch.save(
                        loss_history,
                        os.path.join('../best_model',
                                    f'ogbn_loss_{args.missingness}_{args.miss_rate}.pt')
                    )

    # ==================== FINAL EVALUATION ====================
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)

    # Load best model
    model.load_state_dict(
        torch.load(os.path.join('../best_model',
                                f'ogbn_{args.missingness}_{args.miss_rate}.pt'))
    )
    model.eval()

    with torch.no_grad():
        # Get embeddings
        x_1_ = data_1.x
        zero_ = torch.zeros_like(fixed_features)
        zero = torch.zeros_like(fixed_features)
        zero[masked_id] = zero_[masked_id] + fixed_features[masked_id]
        x_1__ = x_1_ + zero
        x_1 = torch.mm(adj_norm, x_1__)

        z_1 = model.encoder(x_1, edge_index)
        z_2 = model.encoder(data_2.x, edge_index)
        z = (z_1 + z_2) * 0.5

        # Get predictions
        logits = model.forward_classifier(z)
        preds_test = logits[test_id].argmax(dim=1)
        true_test = labels[test_id]

    # Compute metrics
    test_f1_macro = f1_score(true_test.cpu().numpy(), preds_test.cpu().numpy(), average='macro')
    test_f1_micro = f1_score(true_test.cpu().numpy(), preds_test.cpu().numpy(), average='micro')
    precision, recall, f1_weighted, _ = precision_recall_fscore_support(
        true_test.cpu().numpy(),
        preds_test.cpu().numpy(),
        average='weighted'
    )

    print(f'\nTest Results:')
    print(f'  Macro-F1:    {test_f1_macro:.4f}')
    print(f'  Micro-F1:    {test_f1_micro:.4f}')
    print(f'  Weighted-F1: {f1_weighted:.4f}')
    print(f'  Precision:   {precision:.4f}')
    print(f'  Recall:      {recall:.4f}')

    # Detailed classification report
    print(f'\nClassification Report (first 10 classes):')
    report = classification_report(
        true_test.cpu().numpy(),
        preds_test.cpu().numpy(),
        target_names=[f'Class_{i}' for i in range(num_classes)],
        digits=4
    )
    # Print first 15 lines (header + 10 classes)
    print('\n'.join(report.split('\n')[:15]))
    print('...')

    # ==================== LOSS SUMMARY ====================
    print("\n" + "="*70)
    print("LOSS SUMMARY")
    print("="*70)
    print(f"Final Total Loss:          {loss_history['total'][-1]:.4f}")
    print(f"Final Edge Loss:           {loss_history['edge'][-1]:.4f}")
    print(f"Final Contrastive Loss:    {loss_history['contrastive'][-1]:.4f}")
    print(f"Final Reconstruction Loss: {loss_history['reconstruction'][-1]:.4f}")
    print(f"Final Classification Loss: {loss_history['classification'][-1]:.4f}")
    print(f"Final Stats Loss:          {loss_history['stats'][-1]:.4f}")
    print(f"Final Residual Loss:       {loss_history['residual'][-1]:.4f}")
    print("="*70)

    # Save final results
    results = {
        'dataset': args.dataset,
        'missingness': args.missingness,
        'miss_rate': args.miss_rate,
        'test_f1_macro': test_f1_macro,
        'test_f1_micro': test_f1_micro,
        'test_f1_weighted': f1_weighted,
        'test_precision': precision,
        'test_recall': recall,
        'best_val_f1': best_val_f1,
        'num_epochs': args.epochs,
        'seed': args.seed
    }

    torch.save(results, os.path.join('../best_model',
                                     f'ogbn_results_{args.missingness}_{args.miss_rate}.pt'))

    print(f'\n✓ Results saved to: ../best_model/ogbn_results_{args.missingness}_{args.miss_rate}.pt')
    print("="*70)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    if args.cuda:
        torch.cuda.set_device(f'cuda:{args.device}')
    main(args)
