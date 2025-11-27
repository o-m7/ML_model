import pandas as pd

# Load your data
df = pd.read_parquet("feature_store/XAUUSD/XAUUSD_1H.parquet")

# Check if 4H_returns exists
if '4H_returns' in df.columns:
    print("4H_returns exists - this is the problem!")
    print("Higher timeframe returns need extra lagging")