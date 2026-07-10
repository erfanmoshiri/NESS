"""
Cache manager for OGBN-products preprocessing.

Caches:
1. Missingness masks (observable_id, masked_id)
2. SSL targets (target_stats, target_residual)

Cache structure:
    data/ogbn_products/cache/
    ├── MCAR_0.4/
    │   ├── masks.pt              # observable_id, masked_id
    │   ├── ssl_targets.pt        # target_stats, target_residual
    ├── MAR_0.4/
    │   └── ...
    └── MNAR_0.4/
        └── ...
"""

import os
import torch
import hashlib


def get_cache_dir(data_root, missingness_type, miss_rate, seed=72):
    """
    Get cache directory for given missingness configuration.

    Args:
        data_root: Root data directory
        missingness_type: 'MCAR', 'MAR', or 'MNAR'
        miss_rate: Missingness rate (0.2, 0.4, 0.6)
        seed: Random seed

    Returns:
        cache_dir: Path to cache directory
    """
    cache_dir = os.path.join(data_root, 'cache', f'{missingness_type}_{miss_rate}_seed{seed}')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_cached_masks(data_root, missingness_type, miss_rate, seed=72):
    """
    Load cached missingness masks if they exist.

    Returns:
        (observable_id, masked_id) if cached, else None
    """
    cache_dir = get_cache_dir(data_root, missingness_type, miss_rate, seed)
    masks_path = os.path.join(cache_dir, 'masks.pt')

    if os.path.exists(masks_path):
        print(f'  Loading cached masks from: {masks_path}')
        data = torch.load(masks_path, weights_only=True)
        return data['observable_id'], data['masked_id']

    return None


def save_masks(data_root, missingness_type, miss_rate, observable_id, masked_id, seed=72):
    """
    Save missingness masks to cache.
    """
    cache_dir = get_cache_dir(data_root, missingness_type, miss_rate, seed)
    masks_path = os.path.join(cache_dir, 'masks.pt')

    torch.save({
        'observable_id': observable_id,
        'masked_id': masked_id,
        'missingness_type': missingness_type,
        'miss_rate': miss_rate,
        'seed': seed
    }, masks_path)

    print(f'  ✓ Saved masks to: {masks_path}')


def load_cached_ssl_targets(data_root, missingness_type, miss_rate, seed=72):
    """
    Load cached SSL targets if they exist.

    Returns:
        (target_stats, target_residual) if cached, else None
    """
    cache_dir = get_cache_dir(data_root, missingness_type, miss_rate, seed)
    ssl_path = os.path.join(cache_dir, 'ssl_targets.pt')

    if os.path.exists(ssl_path):
        print(f'  Loading cached SSL targets from: {ssl_path}')
        data = torch.load(ssl_path, weights_only=True)
        return data['target_stats'], data['target_residual']

    return None


def save_ssl_targets(data_root, missingness_type, miss_rate, target_stats, target_residual, seed=72):
    """
    Save SSL targets to cache.
    """
    cache_dir = get_cache_dir(data_root, missingness_type, miss_rate, seed)
    ssl_path = os.path.join(cache_dir, 'ssl_targets.pt')

    torch.save({
        'target_stats': target_stats,
        'target_residual': target_residual,
        'missingness_type': missingness_type,
        'miss_rate': miss_rate,
        'seed': seed
    }, ssl_path)

    print(f'  ✓ Saved SSL targets to: {ssl_path}')


def get_or_create_masks(data_root, all_nodes, adj, features, labels,
                        missingness_type, miss_rate, seed=72):
    """
    Load cached masks or create new ones.
    """
    from src.utils_ogbn import simulate_node_missingness

    # Try to load from cache
    cached = load_cached_masks(data_root, missingness_type, miss_rate, seed)
    if cached is not None:
        return cached

    # Create new masks
    print(f'  Cache miss - computing {missingness_type} missingness...')
    observable_id, masked_id = simulate_node_missingness(
        all_nodes, adj, features, labels,
        missingness_type=missingness_type,
        miss_rate=miss_rate,
        seed=seed
    )

    # Save to cache
    save_masks(data_root, missingness_type, miss_rate, observable_id, masked_id, seed)

    return observable_id, masked_id


def get_or_create_ssl_targets(data_root, adj, features, masked_features, observable_id,
                               missingness_type, miss_rate, seed=72):
    """
    Load cached SSL targets or create new ones.
    Uses FAST vectorized computation if torch_scatter available (10-30x speedup).
    """
    # Try to load from cache
    cached = load_cached_ssl_targets(data_root, missingness_type, miss_rate, seed)
    if cached is not None:
        return cached

    # Create new SSL targets
    print(f'  Cache miss - computing SSL targets...')

    # Try to use fast vectorized version
    try:
        from src.fast_ssl_compute import compute_neighborhood_embedding_stats, compute_neighborhood_centroid_residual
        print(f'    Using FAST vectorized computation (~2-5 minutes)...')
    except ImportError:
        from models.MATE_embeddings import compute_neighborhood_embedding_stats, compute_neighborhood_centroid_residual
        print(f'    Using slow loop-based computation (~30-60 minutes)...')
        print(f'    Install torch_scatter for 10-30x speedup!')

    import time
    start = time.time()

    print('    Computing neighborhood stats...')
    target_stats = compute_neighborhood_embedding_stats(adj, features, observable_id)

    print('    Computing centroid residuals...')
    target_residual = compute_neighborhood_centroid_residual(adj, masked_features, observable_id)

    elapsed = time.time() - start
    print(f'    ✓ SSL targets computed in {elapsed:.1f}s ({elapsed/60:.1f} min)')

    # Save to cache
    save_ssl_targets(data_root, missingness_type, miss_rate, target_stats, target_residual, seed)

    return target_stats, target_residual


def clear_cache(data_root, missingness_type=None, miss_rate=None):
    """
    Clear cached data.

    Args:
        data_root: Root data directory
        missingness_type: If specified, only clear this type
        miss_rate: If specified, only clear this rate
    """
    cache_root = os.path.join(data_root, 'cache')

    if not os.path.exists(cache_root):
        print("No cache found")
        return

    if missingness_type is not None and miss_rate is not None:
        # Clear specific cache
        cache_dir = os.path.join(cache_root, f'{missingness_type}_{miss_rate}')
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print(f"Cleared cache: {cache_dir}")
        else:
            print(f"Cache not found: {cache_dir}")
    else:
        # Clear all cache
        import shutil
        shutil.rmtree(cache_root)
        print(f"Cleared all cache: {cache_root}")


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, '..')

    print("Cache Manager Test")
    print("="*70)

    # You can manually clear cache if needed:
    # clear_cache('../data/ogbn_products', 'MCAR', 0.4)

    print("Cache structure:")
    print("  data/ogbn_products/cache/")
    print("  ├── MCAR_0.4_seed72/")
    print("  │   ├── masks.pt")
    print("  │   └── ssl_targets.pt")
    print("  ├── MAR_0.4_seed72/")
    print("  └── MNAR_0.4_seed72/")
