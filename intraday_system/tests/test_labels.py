"""Test label generation."""

import pytest
import pandas as pd
import numpy as np

from intraday_system.labels.triple_barrier import TripleBarrierLabeler
from intraday_system.labels.horizons import get_horizon_config


def test_triple_barrier_basic():
    """Test basic triple-barrier labeling."""
    # Create toy data with clear TP/SL scenarios
    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=50, freq='1H'),
        'open': 100.0,
        'high': [101, 102, 103, 102, 101] * 10,  # Upward movement
        'low': [99, 98, 97, 98, 99] * 10,
        'close': [100, 101, 102, 101, 100] * 10,
        'atr14': 1.0
    })
    
    labeler = TripleBarrierLabeler(
        horizon_bars=5,
        tp_atr_mult=2.0,
        sl_atr_mult=1.0
    )
    
    df_labeled = labeler.create_labels(df)
    
    # Should have labels
    assert 'target' in df_labeled.columns
    assert 'expected_return' in df_labeled.columns
    
    # Labels should be 0, 1, or 2
    assert df_labeled['target'].isin([0, 1, 2]).all()
    
    # Should have distribution info
    dist = labeler.get_label_distribution(df_labeled)
    assert dist['total'] == len(df_labeled)
    assert dist['flat_pct'] + dist['up_pct'] + dist['down_pct'] == 100.0


def test_horizon_configs():
    """Test that horizon configs exist for all timeframes."""
    timeframes = ['5T', '15T', '30T', '1H', '2H', '4H']
    
    for tf in timeframes:
        config = get_horizon_config(tf)
        assert 'horizon_bars' in config
        assert 'tp_atr_mult' in config
        assert 'sl_atr_mult' in config
        assert config['tp_atr_mult'] > config['sl_atr_mult']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

