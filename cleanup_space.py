#!/usr/bin/env python3
"""
CLEANUP UNNECESSARY FILES
=========================
Removes temporary files, caches, and old data to free up disk space.

Usage:
    python cleanup_space.py --dry-run  # See what would be deleted
    python cleanup_space.py            # Actually delete files
"""

import argparse
import shutil
from pathlib import Path


def get_dir_size(path):
    """Get total size of directory in MB."""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except:
        pass
    return total / (1024 * 1024)


def cleanup(dry_run=True):
    """
    Clean up unnecessary files.

    Args:
        dry_run: If True, just show what would be deleted
    """
    to_delete = []

    # Find directories to delete
    dirs_to_check = [
        'raw_data_cache',  # Old download cache
        'temp_download',   # Temporary downloads
        '__pycache__',     # Python cache
        '.pytest_cache',   # Pytest cache
        'backtest_results',  # Old backtest results (keep only latest)
    ]

    # Find files to delete
    patterns_to_delete = [
        '*.pyc',           # Python compiled
        '*.pyo',           # Python optimized
        '.DS_Store',       # Mac files
        '*.log',           # Log files
        '*checkpoint.parquet',  # Checkpoint files
    ]

    print("\n" + "="*80)
    print("CLEANUP ANALYSIS")
    print("="*80 + "\n")

    total_size = 0

    # Check directories
    for dir_name in dirs_to_check:
        dir_path = Path(dir_name)
        if dir_path.exists():
            size = get_dir_size(dir_path)
            if size > 0:
                to_delete.append(('dir', dir_path, size))
                total_size += size
                print(f"📁 {dir_name}: {size:.1f} MB")

    # Check patterns
    for pattern in patterns_to_delete:
        for file_path in Path('.').rglob(pattern):
            if file_path.is_file():
                size = file_path.stat().st_size / (1024 * 1024)
                to_delete.append(('file', file_path, size))
                total_size += size
                if size > 0.1:  # Only print files > 100KB
                    print(f"📄 {file_path}: {size:.1f} MB")

    print(f"\n{'='*80}")
    print(f"TOTAL: {total_size:.1f} MB can be freed")
    print(f"{'='*80}\n")

    if not to_delete:
        print("✨ No unnecessary files found. Already clean!")
        return

    if dry_run:
        print("⚠️  DRY RUN - Nothing deleted")
        print("\nTo actually delete these files, run:")
        print("  python cleanup_space.py")
        return

    # Actually delete
    print("🗑️  Deleting files...")

    deleted_count = 0
    for item_type, item_path, size in to_delete:
        try:
            if item_type == 'dir':
                shutil.rmtree(item_path)
                print(f"  ✓ Deleted {item_path}/ ({size:.1f} MB)")
            else:
                item_path.unlink()
                print(f"  ✓ Deleted {item_path} ({size:.1f} MB)")
            deleted_count += 1
        except Exception as e:
            print(f"  ✗ Error deleting {item_path}: {e}")

    print(f"\n✅ Freed {total_size:.1f} MB ({deleted_count} items)")


def main():
    parser = argparse.ArgumentParser(description='Clean up unnecessary files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without deleting')

    args = parser.parse_args()

    cleanup(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
