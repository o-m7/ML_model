#!/usr/bin/env python3
"""
Find Polygon Data Files
Scans your directory to find actual file structure.

Usage:
    python find_polygon_files.py
    python find_polygon_files.py --path ~/polygon_s3_data
"""

import argparse
from pathlib import Path
from collections import defaultdict


def scan_directory(base_path: Path):
    """Scan directory and show structure."""
    
    print(f"\n{'='*80}")
    print(f"SCANNING: {base_path}")
    print(f"{'='*80}\n")
    
    if not base_path.exists():
        print(f"❌ Path does not exist: {base_path}")
        return
        
    print(f"✅ Path exists\n")
    
    # Find all CSV/CSV.GZ files
    csv_files = list(base_path.glob('**/*.csv'))
    csv_gz_files = list(base_path.glob('**/*.csv.gz'))
    
    all_files = csv_files + csv_gz_files
    
    print(f"📊 Found {len(all_files)} data files\n")
    
    if len(all_files) == 0:
        print(f"❌ No CSV files found in {base_path}")
        print(f"\n🔍 Directory contents:")
        for item in sorted(base_path.iterdir())[:20]:
            print(f"   {item.name}")
        return
    
    # Group by symbol
    symbol_files = defaultdict(list)
    for f in all_files[:100]:  # Sample first 100
        # Extract symbol from path or filename
        parts = f.parts
        for part in parts:
            if 'XAU' in part or 'XAG' in part or 'EUR' in part or 'GBP' in part or 'C:' in part:
                symbol_files[part].append(f)
                break
                
    print(f"📂 Symbol directories found:")
    for symbol, files in sorted(symbol_files.items()):
        print(f"\n   {symbol}:")
        print(f"      Files: {len(files)}")
        print(f"      Sample: {files[0].name}")
        if len(files) > 1:
            print(f"              {files[1].name if len(files) > 1 else ''}")
        
        # Show directory structure
        rel_path = files[0].relative_to(base_path)
        print(f"      Structure: {base_path} / {' / '.join(rel_path.parts[:-1])} / <files>")
        
    # Sample first 10 files
    print(f"\n📄 Sample file paths:")
    for f in all_files[:10]:
        rel = f.relative_to(base_path)
        print(f"   {rel}")
        
    if len(all_files) > 10:
        print(f"   ... and {len(all_files) - 10} more")
        
    # Detect pattern
    print(f"\n🔍 Detected patterns:")
    
    sample_file = all_files[0]
    rel = sample_file.relative_to(base_path)
    parts = rel.parts
    
    print(f"\n   Example: {base_path} / {' / '.join(parts)}")
    print(f"\n   Pattern breakdown:")
    for i, part in enumerate(parts):
        print(f"      Level {i}: {part}")
        
    # Check for date structure
    has_year_dirs = any('202' in p for f in all_files[:20] for p in f.parts)
    has_month_dirs = any(p.isdigit() and len(p) == 2 for f in all_files[:20] for p in f.parts)
    
    print(f"\n   Year directories: {'✅' if has_year_dirs else '❌'}")
    print(f"   Month directories: {'✅' if has_month_dirs else '❌'}")
    
    # Suggest command
    print(f"\n{'='*80}")
    print(f"SUGGESTED COMMAND")
    print(f"{'='*80}\n")
    
    if 'C:XAU-USD' in symbol_files or any('C:' in s for s in symbol_files):
        print(f"✅ Found Polygon format (C:XAU-USD)")
        print(f"\nYour data is already in the correct format!")
        print(f"\nRun extraction with:")
        print(f"   python extract_features_from_s3.py --symbol XAUUSD --start 2020-01-01 --end 2025-11-17")
    else:
        print(f"⚠️  Non-standard format detected")
        print(f"\nFirst, tell me the symbol directory name:")
        if symbol_files:
            first_symbol = list(symbol_files.keys())[0]
            print(f"   Example: {first_symbol}")


def main():
    parser = argparse.ArgumentParser(description='Find Polygon data files')
    parser.add_argument('--path', type=str, default='~/polygon_s3_data', 
                       help='Base path to scan')
    
    args = parser.parse_args()
    
    base_path = Path(args.path).expanduser()
    scan_directory(base_path)


if __name__ == '__main__':
    main()