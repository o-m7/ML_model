"""Horizon configurations per timeframe and strategy."""

from typing import Dict


def get_horizon_config(timeframe: str, strategy: str = None) -> Dict:
    """
    Get label horizon configuration for a timeframe/strategy.
    
    Args:
        timeframe: Timeframe string (5T, 15T, 30T, 1H, 2H, 4H)
        strategy: Optional strategy name (S1-S6)
        
    Returns:
        Dictionary with horizon_bars, tp_atr_mult, sl_atr_mult
    """
    # Default configurations
    configs = {
        "5T": {
            "horizon_bars": 10,
            "tp_atr_mult": 1.5,
            "sl_atr_mult": 1.0
        },
        "15T": {
            "horizon_bars": 8,
            "tp_atr_mult": 1.5,
            "sl_atr_mult": 1.0
        },
        "30T": {
            "horizon_bars": 6,
            "tp_atr_mult": 1.5,
            "sl_atr_mult": 1.0
        },
        "1H": {
            "horizon_bars": 6,
            "tp_atr_mult": 2.0,
            "sl_atr_mult": 1.0
        },
        "2H": {
            "horizon_bars": 4,
            "tp_atr_mult": 2.0,
            "sl_atr_mult": 1.0
        },
        "4H": {
            "horizon_bars": 3,
            "tp_atr_mult": 2.5,
            "sl_atr_mult": 1.0
        }
    }
    
    if timeframe not in configs:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    
    return configs[timeframe]

