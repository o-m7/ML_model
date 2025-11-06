"""Post-processing filters for live signals."""

from typing import Dict, Optional
import pandas as pd


def apply_filters(
    signal: Dict,
    current_spread: float,
    atr: float,
    last_trade_bar: Optional[int] = None,
    current_bar: int = 0,
    cooldown_bars: int = 5,
    spread_atr_threshold: float = 0.5
) -> Dict:
    """
    Apply post-processing filters to signal.
    
    Args:
        signal: Raw signal from prediction
        current_spread: Current bid-ask spread
        atr: Current ATR value
        last_trade_bar: Bar index of last trade exit
        current_bar: Current bar index
        cooldown_bars: Minimum bars between trades
        spread_atr_threshold: Max spread as fraction of ATR
        
    Returns:
        Filtered signal (may change to HOLD)
    """
    filtered_signal = signal.copy()
    reasons = []
    
    # 1. Cooldown Filter
    if last_trade_bar is not None:
        bars_since_last_trade = current_bar - last_trade_bar
        if bars_since_last_trade < cooldown_bars:
            filtered_signal['signal'] = 'HOLD'
            reasons.append(f"Cooldown active ({bars_since_last_trade}/{cooldown_bars})")
    
    # 2. Spread Filter
    max_spread = atr * spread_atr_threshold
    if current_spread > max_spread:
        filtered_signal['signal'] = 'HOLD'
        reasons.append(f"Spread too wide ({current_spread:.5f} > {max_spread:.5f})")
    
    # 3. Confidence Filter (already applied in strategy config)
    # Additional check if needed
    min_confidence = 0.40
    if signal['confidence'] < min_confidence and signal['signal'] != 'HOLD':
        filtered_signal['signal'] = 'HOLD'
        reasons.append(f"Low confidence ({signal['confidence']:.2f} < {min_confidence})")
    
    # 4. Expiry Check (signal too old)
    # This would require timestamp comparison in live system
    
    # Add filter reasons
    if reasons:
        filtered_signal['filtered'] = True
        filtered_signal['filter_reasons'] = reasons
        filtered_signal['original_signal'] = signal['signal']
    else:
        filtered_signal['filtered'] = False
    
    return filtered_signal


def check_expiry(signal: Dict, current_bar_index: int) -> bool:
    """Check if signal has expired."""
    expiry_bar = signal.get('expiry_bar_index', float('inf'))
    return current_bar_index > expiry_bar


def calculate_position_size(
    signal: Dict,
    account_equity: float,
    risk_per_trade_pct: float = 0.01,
    max_position_pct: float = 0.10
) -> Dict:
    """
    Calculate position size based on risk parameters.
    
    Args:
        signal: Trading signal
        account_equity: Current account equity
        risk_per_trade_pct: Risk per trade as percentage
        max_position_pct: Max position as percentage of equity
        
    Returns:
        Position sizing information
    """
    if signal['signal'] == 'HOLD':
        return {
            'position_size': 0,
            'position_value': 0,
            'risk_amount': 0
        }
    
    # Risk amount
    risk_amount = account_equity * risk_per_trade_pct
    
    # Stop distance
    entry_price = signal['entry_ref']
    stop_loss = signal['stop_loss']
    stop_distance = abs(entry_price - stop_loss)
    
    if stop_distance == 0:
        return {
            'position_size': 0,
            'position_value': 0,
            'risk_amount': 0,
            'error': 'Invalid stop distance'
        }
    
    # Calculate position size
    position_size = risk_amount / stop_distance
    position_value = position_size * entry_price
    
    # Cap to max position
    max_position_value = account_equity * max_position_pct
    if position_value > max_position_value:
        position_size = max_position_value / entry_price
        position_value = max_position_value
    
    return {
        'position_size': float(position_size),
        'position_value': float(position_value),
        'risk_amount': float(risk_amount),
        'stop_distance': float(stop_distance),
        'risk_reward_ratio': float(signal['expected_R'])
    }

