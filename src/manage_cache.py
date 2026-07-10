#!/usr/bin/env python
"""
Utility to manage OGBN-products cache.

Usage:
    python manage_cache.py list              # List all cached data
    python manage_cache.py clear MCAR 0.4    # Clear specific cache
    python manage_cache.py clear_all         # Clear all cache
"""

import sys
import os
sys.path.insert(0, '..')
from src.cache_manager import clear_cache


def list_cache(data_root='../data/ogbn_products'):
    """List all cached data."""
    cache_root = os.path.join(data_root, 'cache')

    if not os.path.exists(cache_root):
        print("No cache found")
        return

    print("="*70)
    print("CACHED DATA")
    print("="*70)

    total_size = 0
    for item in sorted(os.listdir(cache_root)):
        item_path = os.path.join(cache_root, item)
        if os.path.isdir(item_path):
            # Get size
            size = sum(os.path.getsize(os.path.join(item_path, f))
                      for f in os.listdir(item_path)
                      if os.path.isfile(os.path.join(item_path, f)))
            size_mb = size / (1024 * 1024)
            total_size += size

            # List files
            files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]

            print(f"\n{item}/ ({size_mb:.1f} MB)")
            for f in sorted(files):
                f_path = os.path.join(item_path, f)
                f_size = os.path.getsize(f_path) / (1024 * 1024)
                print(f"  ├─ {f} ({f_size:.1f} MB)")

    print(f"\nTotal cache size: {total_size / (1024 * 1024):.1f} MB")
    print("="*70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "list":
        list_cache()

    elif command == "clear" and len(sys.argv) == 4:
        missingness_type = sys.argv[2]
        miss_rate = float(sys.argv[3])
        clear_cache('../data/ogbn_products', missingness_type, miss_rate)

    elif command == "clear_all":
        confirm = input("Clear ALL cache? (yes/no): ")
        if confirm.lower() == 'yes':
            clear_cache('../data/ogbn_products')
        else:
            print("Cancelled")

    else:
        print(__doc__)
        sys.exit(1)
