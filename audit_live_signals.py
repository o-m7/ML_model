#!/usr/bin/env python3
"""
Audit live signals stored in Supabase to catch stale prices, low confidence,
or excessive fallback usage before they impact trading.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("❌ SUPABASE_URL or SUPABASE_KEY not set in environment.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

NOW = datetime.now(timezone.utc)
LOOKBACK_HOURS = 6
STALE_THRESHOLD_MINUTES = 10  # flags if latest timestamp older than 10 minutes
LOW_CONFIDENCE_THRESHOLD = 0.40
LOW_EDGE_THRESHOLD = 0.02


def fetch_recent_signals() -> pd.DataFrame:
    since = (NOW - timedelta(hours=LOOKBACK_HOURS)).isoformat()
    response = supabase.table("live_signals").select("*").gte("timestamp", since).execute()
    data = response.data or []
    if not data:
        print(f"⚠️  No live_signals rows found in the last {LOOKBACK_HOURS} hours.")
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = df.groupby(["symbol", "timeframe"]).agg(
        rows=("signal_type", "size"),
        avg_confidence=("confidence", "mean"),
        avg_edge=("edge", "mean"),
        last_timestamp=("timestamp", "max"),
    ).reset_index()
    return metrics


def flag_anomalies(metrics: pd.DataFrame) -> List[str]:
    issues = []
    for _, row in metrics.iterrows():
        staleness = (NOW - row["last_timestamp"]).total_seconds() / 60.0
        if staleness > STALE_THRESHOLD_MINUTES:
            issues.append(f"{row['symbol']} {row['timeframe']}: stale ({staleness:.1f} min since last signal)")
        if row["avg_confidence"] < LOW_CONFIDENCE_THRESHOLD:
            issues.append(f"{row['symbol']} {row['timeframe']}: low confidence ({row['avg_confidence']:.2f})")
        if row["avg_edge"] < LOW_EDGE_THRESHOLD:
            issues.append(f"{row['symbol']} {row['timeframe']}: low edge ({row['avg_edge']:.3f})")
    return issues


def main():
    df = fetch_recent_signals()
    if df.empty:
        return

    metrics = summarize(df)
    print("\n=== Live Signal Summary (last 6 hours) ===")
    print(metrics.sort_values(["symbol", "timeframe"]).to_string(index=False, float_format="%.3f"))

    issues = flag_anomalies(metrics)
    if issues:
        print("\n⚠️  Anomalies detected:")
        for issue in issues:
            print(f"  - {issue}")
        # Exit with non-zero status so workflow surfaces the failure
        raise SystemExit(1)

    print("\n✅ No issues detected with live signals.")


if __name__ == "__main__":
    main()

