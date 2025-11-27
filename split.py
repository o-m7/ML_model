import pandas as pd
import numpy as np
from pathlib import Path
import os

BASE_DIR = Path("/Users/omar/Desktop/ML_model/ML_model/feature_store/XAUUSD")

# Automatically compress all yearly split files
FILES = [
    "XAUUSD_5T_2023.parquet",
    "XAUUSD_5T_2024.parquet",
    "XAUUSD_5T_2025.parquet",
]

def optimize_and_compress(src_path: Path):
    print(f"\n📦 Compressing: {src_path.name}")

    df = pd.read_parquet(src_path)

    # Downcast float64 → float32 (50% smaller, no meaningful loss)
    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        df[col] = df[col].astype(np.float32)

    # Downcast int64 → int32
    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        df[col] = df[col].astype(np.int32)

    # Output filename
    dst_path = src_path.with_name(src_path.stem + "_small.parquet")

    # High-level zstd compression
    df.to_parquet(
        dst_path,
        compression="zstd",
        compression_level=18   # HIGH compression (max reasonable)
    )

    size_mb = os.path.getsize(dst_path) / 1024 / 1024
    print(f"   ✅ Saved: {dst_path.name} ({size_mb:.2f} MB)")
    print(f"   📉 Columns downcasted: {len(float_cols) + len(int_cols)}")


def main():
    for fname in FILES:
        fpath = BASE_DIR / fname
        if fpath.exists():
            optimize_and_compress(fpath)
        else:
            print(f"⚠️ File missing: {fpath}")

if __name__ == "__main__":
    main()
