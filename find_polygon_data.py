#!/usr/bin/env python3
"""
Diagnostic tool to find your Polygon data structure.

Usage:
    python find_polygon_data.py
    python find_polygon_data.py --search-path /path/to/search
"""

import argparse
from pathlib import Path
from typing import List, Dict
import os


def find_parquet_files(search_path: Path, max_depth: int = 5) -> List[Path]:
    """Find all parquet files recursively."""
    parquet_files = []
    
    def search_recursive(path: Path, depth: int):
        if depth > max_depth:
            return
        
        try:
            for item in path.iterdir():
                if item.is_file() and (item.suffix == '.parquet' or item.suffix == '.gz' or item.suffix == '.csv'):
                    parquet_files.append(item)
                elif item.is_dir():
                    search_recursive(item, depth + 1)
        except PermissionError:
            pass
    
    search_recursive(search_path, 0)
    return parquet_files


def find_polygon_directories(search_path: Path) -> List[Path]:
    """Find directories that look like Polygon data."""
    candidates = []
    
    # Common patterns
    patterns = [
        '*polygon*',
        '*s3*',
        '*data*',
        '*trading*',
        '*forex*',
        '*XAUUSD*',
        '*XAU*',
        '*gold*',
        '*C:XAU-USD*'
    ]
    
    for pattern in patterns:
        matches = list(search_path.glob(f'**/{pattern}'))
        for match in matches:
            if match.is_dir():
                # Check if it contains data files
                has_data = any(f.suffix in ['.parquet', '.csv', '.gz'] 
                              for f in match.rglob('*') if f.is_file())
                if has_data:
                    candidates.append(match)
    
    return candidates


def analyze_directory_structure(directory: Path, max_depth: int = 3):
    """Analyze and print directory structure."""
    
    print(f"\n📁 Analyzing: {directory}")
    print("="*80)
    
    # Count files by type
    file_counts = {}
    sample_files = []
    
    for item in directory.rglob('*'):
        if item.is_file():
            suffix = item.suffix if item.suffix else 'no_extension'
            file_counts[suffix] = file_counts.get(suffix, 0) + 1
            
            if len(sample_files) < 5:
                sample_files.append(item)
    
    print(f"\n📊 File type distribution:")
    for suffix, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {suffix}: {count:,} files")
    
    print(f"\n📝 Sample files:")
    for f in sample_files[:5]:
        rel_path = f.relative_to(directory)
        print(f"   {rel_path}")
        
    # Check for XAUUSD/Gold data
    gold_files = []
    for pattern in ['*XAU*', '*gold*', '*GOLD*', '*2600*']:  # 2600 is approx gold price
        gold_files.extend(list(directory.glob(f'**/{pattern}')))
    
    if gold_files:
        print(f"\n🥇 Found potential XAUUSD/Gold files:")
        for f in gold_files[:10]:
            rel_path = f.relative_to(directory)
            print(f"   {rel_path}")
    
    # Analyze directory depth/structure
    print(f"\n🗂️  Directory structure sample:")
    
    def print_tree(path: Path, prefix: str = "", depth: int = 0, max_depth: int = 3):
        if depth > max_depth:
            return
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))[:10]
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                
                if item.is_dir():
                    print(f"{prefix}{current_prefix}{item.name}/")
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    print_tree(item, next_prefix, depth + 1, max_depth)
                else:
                    size = item.stat().st_size / 1024 / 1024  # MB
                    print(f"{prefix}{current_prefix}{item.name} ({size:.2f} MB)")
        except PermissionError:
            print(f"{prefix}[Permission Denied]")
    
    print_tree(directory, max_depth=2)


def check_existing_feature_store():
    """Check if feature store already exists."""
    
    feature_store_path = Path("ML_model/ML_model/feature_store")
    
    if feature_store_path.exists():
        print(f"\n✅ Found existing feature store: {feature_store_path.absolute()}")
        print("="*80)
        
        # Check for XAUUSD
        xauusd_path = feature_store_path / "XAUUSD"
        if xauusd_path.exists():
            print(f"\n📂 XAUUSD feature files:")
            for f in sorted(xauusd_path.iterdir()):
                if f.is_file():
                    size = f.stat().st_size / 1024 / 1024
                    print(f"   {f.name} ({size:.2f} MB)")
            
            # Check if we can use existing files
            parquet_files = list(xauusd_path.glob("*.parquet"))
            if parquet_files:
                print(f"\n💡 You already have {len(parquet_files)} feature files!")
                print(f"   You may not need to extract from Polygon S3 again.")
                return True
        else:
            print(f"\n⚠️  Feature store exists but no XAUUSD directory")
    else:
        print(f"\n❌ No feature store found at: {feature_store_path.absolute()}")
    
    return False


def main():
    parser = argparse.ArgumentParser(description='Find Polygon data structure')
    parser.add_argument('--search-path', type=str, default='~', 
                       help='Base path to search (default: home directory)')
    parser.add_argument('--check-desktop', action='store_true',
                       help='Check Desktop and Downloads folders')
    
    args = parser.parse_args()
    
    print("="*80)
    print("POLYGON DATA STRUCTURE FINDER")
    print("="*80)
    
    # First check if feature store already exists
    if check_existing_feature_store():
        print("\n" + "="*80)
        print("RECOMMENDATION:")
        print("="*80)
        print("You already have processed feature files.")
        print("You can skip the extraction step and go directly to:")
        print("  python production_ml_system_fixed.py --symbol XAUUSD --all-timeframes")
        print()
        return
    
    # Search for Polygon data
    print(f"\n🔍 Searching for Polygon data...")
    
    search_paths = [Path(args.search_path).expanduser()]
    
    if args.check_desktop:
        home = Path.home()
        search_paths.extend([
            home / "Desktop",
            home / "Downloads",
            home / "Documents"
        ])
    
    # Common locations
    common_paths = [
        Path.home() / "polygon_s3_data",
        Path.home() / "polygon_data",
        Path.home() / "data",
        Path.home() / "trading_data",
        Path("/Users/omar/polygon_s3_data"),  # From error message
        Path("/Users/omar/Desktop"),
        Path.cwd() / "data",
        Path.cwd() / "polygon_data"
    ]
    
    print(f"\n📍 Checking common locations:")
    found_any = False
    
    for path in common_paths:
        if path.exists():
            print(f"   ✓ {path}")
            
            # Quick check for data files
            data_files = list(path.rglob('*.parquet'))[:5]
            data_files.extend(list(path.rglob('*.csv.gz'))[:5])
            
            if data_files:
                print(f"     Found {len(data_files)} data files")
                found_any = True
                analyze_directory_structure(path)
        else:
            print(f"   ✗ {path} (not found)")
    
    if not found_any:
        print(f"\n⚠️  No Polygon data found in common locations")
        print(f"\n💡 Options:")
        print(f"   1. If you have Polygon data elsewhere, run:")
        print(f"      python find_polygon_data.py --search-path /your/data/path")
        print(f"   2. If you don't have Polygon data:")
        print(f"      - Download from Polygon.io (requires API key)")
        print(f"      - Or use existing feature files if available")
        print(f"   3. Check if you have minute data in a different format")
    
    # Try to find ANY trading data
    print(f"\n🔍 Searching entire home directory for trading data (this may take a minute)...")
    
    home = Path.home()
    candidates = find_polygon_directories(home)
    
    if candidates:
        print(f"\n✅ Found {len(candidates)} potential data directories:")
        for i, candidate in enumerate(candidates[:10], 1):
            rel_path = candidate.relative_to(home) if candidate.is_relative_to(home) else candidate
            print(f"   {i}. {rel_path}")
        
        if candidates:
            print(f"\n💡 To analyze a directory, run:")
            print(f"   python find_polygon_data.py --search-path '{candidates[0]}'")


if __name__ == '__main__':
    main()