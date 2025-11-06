"""Test for data leakage prevention."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from intraday_system.features.utils import check_leakage, align_timeframes
from intraday_system.labels.triple_barrier import TripleBarrierLabeler


def test_no_future_leakage_in_features():
    """Test that features don't contain future information."""
    # Create toy data
    dates = pd.date_range('2020-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': np.cumsum(np.random.randn(100)) + 100,
        'atr14': np.random.uniform(1, 3, 100),
        'target': np.random.randint(0, 3, 100)
    })
    
    # Add a clean feature
    df['clean_feature'] = df['close'].shift(1)  # Properly shifted
    
    # Check leakage
    results = check_leakage(df, target_col='target')
    
    # Should have no suspicious correlations for clean feature
    assert len(results['future_peeking']) == 0 or all(
        feat['feature'] != 'clean_feature' for feat in results['future_peeking']
    )


def test_htf_alignment_no_leakage():
    """Test that HTF alignment doesn't leak future data."""
    # Low TF data (hourly)
    dates_low = pd.date_range('2020-01-01', periods=24, freq='1H')
    df_low = pd.DataFrame({
        'timestamp': dates_low,
        'value': range(24)
    })
    
    # High TF data (daily)
    dates_high = pd.date_range('2020-01-01', periods=2, freq='1D')
    df_high = pd.DataFrame({
        'timestamp': dates_high,
        'daily_value': [100, 200]
    })
    
    # Align
    merged = align_timeframes(df_low, df_high)
    
    # HTF values should only use past data
    # First 24 hours should use day 1 value
    assert all(merged.iloc[:24]['daily_value_htf'] == 100)


def test_triple_barrier_uses_future_correctly():
    """Test that triple-barrier only looks forward within horizon."""
    dates = pd.date_range('2020-01-01', periods=100, freq='15T')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 100.0,
        'high': np.random.uniform(100, 102, 100),
        'low': np.random.uniform(98, 100, 100),
        'close': np.random.uniform(99, 101, 100),
        'atr14': 1.0
    })
    
    labeler = TripleBarrierLabeler(
        horizon_bars=10,
        tp_atr_mult=2.0,
        sl_atr_mult=1.0
    )
    
    df_labeled = labeler.create_labels(df)
    
    # Should have removed horizon bars from end
    assert len(df_labeled) == len(df) - 10
    
    # All rows should have labels
    assert df_labeled['target'].notna().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

