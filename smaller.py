import pandas as pd
import os

src = "/Users/omar/Desktop/ML_model/ML_model/feature_store/XAUUSD/XAUUSD_15T.parquet"
dst = "/Users/omar/Desktop/ML_model/ML_model/feature_store/XAUUSD/XAUUSD_15T_small.parquet"

df = pd.read_parquet(src)

df.to_parquet(
    dst,
    compression="zstd",
    compression_level=9
)

print("Done! Small file saved to:", dst)
print("Size (MB):", os.path.getsize(dst) / 1024 / 1024)
