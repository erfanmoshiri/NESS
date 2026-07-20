"""
Benchmark runner for OGBN-products under node missingness.

Usage:
    python main_benchmark_ogbn.py --model NeighAggre --missingness MCAR --miss_rate 0.4
    python main_benchmark_ogbn.py --model GraphSAGE --missingness MAR --miss_rate 0.4
"""

import argparse
import sys
import os
import json
import time
import datetime
import torch

sys.path.insert(0, '..')
from src.utils_ogbn import simulate_node_missingness
from src.utils import set_random_seed
from data_loader import load_dataset, SMALL_DATASETS, LARGE_DATASETS

from NeighAggre import train_NeighAggre
from GraphSAGE import train_GraphSAGE
from KNN import train_KNN
from MATE_ogbn import train_MATE_ogbn
from FP import train_FP
from PaGCN import train_PaGCN

MODELS = ['NeighAggre', 'GraphSAGE', 'KNN', 'MATE', 'FP', 'PaGCN']


def get_run_dir(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f"{args.model}_{args.dataset}_{args.missingness}_{args.miss_rate}_{timestamp}"
    run_dir = os.path.join('results', name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=MODELS)
    parser.add_argument('--dataset', type=str, default='ogbn-products',
                        choices=SMALL_DATASETS + LARGE_DATASETS)
    parser.add_argument('--missingness', type=str, default='MCAR', choices=['MCAR', 'MAR', 'MNAR'])
    parser.add_argument('--miss_rate', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--data_root', type=str, default='../data/ogbn_products')
    parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available())
    parser.add_argument('--device', type=int, default=0)
    # GNN args (GraphSAGE / FP / PaGCN)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (in eval intervals of 5 epochs)')
    parser.add_argument('--batch_size', type=int, default=512)
    # KNN specific
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--max_hops', type=int, default=3)
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if args.cuda else 'cpu')

    print('=' * 70)
    print(f'  {args.model} | {args.dataset} | {args.missingness} @ {args.miss_rate*100:.0f}%')
    print('=' * 70)

    # Load data (unified loader handles all datasets)
    print(f'[1/3] Loading {args.dataset}...')
    graph, adj, features, labels, num_classes = load_dataset(args.dataset, args.data_root)
    num_nodes = features.size(0)
    print(f'  {num_nodes:,} nodes | {features.size(1)} dims | {num_classes} classes')

    # Simulate missingness
    print(f'[2/3] Simulating {args.missingness} missingness...')
    all_nodes = torch.arange(num_nodes, dtype=torch.long)
    observable_id, masked_id = simulate_node_missingness(
        all_nodes, adj, features, labels,
        missingness_type=args.missingness,
        miss_rate=args.miss_rate,
        seed=args.seed
    )
    split_point = len(masked_id) // 2
    vali_id = masked_id[:split_point]
    test_id = masked_id[split_point:]
    print(f'  Observable: {len(observable_id):,} | Val: {len(vali_id):,} | Test: {len(test_id):,}')

    features = features.to(device)
    labels = labels.to(device)
    adj = adj.to(device)
    observable_id = observable_id.to(device)
    masked_id = masked_id.to(device)
    vali_id = vali_id.to(device)
    test_id = test_id.to(device)

    run_dir = get_run_dir(args)
    log_path = os.path.join(run_dir, 'training_log.jsonl')

    # Run model
    print(f'[3/3] Running {args.model}...')
    start = time.time()

    if args.model == 'NeighAggre':
        imputed = train_NeighAggre(adj, features, observable_id, masked_id, device)

        # Evaluate: train a simple linear classifier on imputed features
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        import numpy as np

        print('  Training linear classifier on imputed features...')
        X_train = imputed[observable_id].cpu().numpy()
        y_train = labels[observable_id].cpu().numpy()
        X_val = imputed[vali_id].cpu().numpy()
        y_val = labels[vali_id].cpu().numpy()
        X_test = imputed[test_id].cpu().numpy()
        y_test = labels[test_id].cpu().numpy()

        clf = LogisticRegression(max_iter=1000, C=1.0, solver='saga')
        clf.fit(X_train, y_train)
        val_f1 = f1_score(y_val, clf.predict(X_val), average='macro')
        test_f1 = f1_score(y_test, clf.predict(X_test), average='macro')

    elif args.model == 'GraphSAGE':
        val_f1, test_f1 = train_GraphSAGE(
            graph, features, labels,
            observable_id, masked_id, vali_id, test_id,
            num_classes=num_classes,
            device=device,
            hidden=args.hidden,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            log_path=log_path,
        )

    elif args.model == 'FP':
        val_f1, test_f1 = train_FP(
            graph, features, labels,
            observable_id, masked_id, vali_id, test_id,
            num_classes=num_classes,
            device=device,
            hidden=args.hidden,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            log_path=log_path,
        )

    elif args.model == 'PaGCN':
        val_f1, test_f1 = train_PaGCN(
            graph, features, labels,
            observable_id, masked_id, vali_id, test_id,
            num_classes=num_classes,
            device=device,
            hidden=args.hidden,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            log_path=log_path,
        )

    elif args.model == 'MATE':
        val_f1, test_f1 = train_MATE_ogbn(
            graph, features, labels,
            observable_id, masked_id, vali_id, test_id,
            num_classes=num_classes,
            device=device,
            hidden=args.hidden,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            log_path=log_path,
        )

    elif args.model == 'KNN':
        imputed = train_KNN(
            adj, features, observable_id, masked_id, device,
            K=args.K, max_hops=args.max_hops,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score

        print('  Training linear classifier on imputed features...')
        X_train = imputed[observable_id].cpu().numpy()
        y_train = labels[observable_id].cpu().numpy()
        X_val = imputed[vali_id].cpu().numpy()
        y_val = labels[vali_id].cpu().numpy()
        X_test = imputed[test_id].cpu().numpy()
        y_test = labels[test_id].cpu().numpy()

        clf = LogisticRegression(max_iter=1000, C=1.0, solver='saga', n_jobs=-1)
        clf.fit(X_train, y_train)
        val_f1 = f1_score(y_val, clf.predict(X_val), average='macro')
        test_f1 = f1_score(y_test, clf.predict(X_test), average='macro')

    elapsed = time.time() - start

    print('\n' + '=' * 70)
    print(f'  Val  Macro-F1: {val_f1:.4f}')
    print(f'  Test Macro-F1: {test_f1:.4f}')
    print(f'  Time: {elapsed:.1f}s')
    print('=' * 70)

    results = {
        'model': args.model,
        'dataset': args.dataset,
        'missingness': args.missingness,
        'miss_rate': args.miss_rate,
        'seed': args.seed,
        'val_f1_macro': val_f1,
        'test_f1_macro': test_f1,
        'train_time_s': round(elapsed, 1),
    }
    with open(os.path.join(run_dir, 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved to: {run_dir}')


if __name__ == '__main__':
    main()
