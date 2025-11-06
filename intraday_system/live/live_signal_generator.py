"""Complete live signal generation with Polygon API integration."""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timezone

from .data_fetcher import get_live_data
from .runner import predict
from .postprocess import apply_filters, calculate_position_size


class LiveSignalGenerator:
    """
    Complete live signal generation pipeline.
    
    Fetches data from Polygon -> Generates signal -> Applies filters -> Calculates position size
    """
    
    def __init__(
        self,
        models_dir: str = "models_intraday",
        config_path: str = "intraday_system/config/settings.yaml",
        polygon_api_key: Optional[str] = None
    ):
        """
        Initialize live signal generator.
        
        Args:
            models_dir: Directory containing trained models
            config_path: Path to system configuration
            polygon_api_key: Optional Polygon API key (uses .env if not provided)
        """
        self.models_dir = models_dir
        self.config_path = config_path
        self.polygon_api_key = polygon_api_key
        
        # Trading state
        self.last_trade_bars = {}  # Track last trade per symbol/TF
        self.current_positions = {}  # Track open positions
    
    def generate_signal(
        self,
        symbol: str,
        timeframe: str,
        account_equity: float = 100000,
        apply_post_filters: bool = True,
        current_spread: Optional[float] = None
    ) -> Dict:
        """
        Generate complete trading signal with all filters.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            account_equity: Current account equity for position sizing
            apply_post_filters: Whether to apply cooldown/spread filters
            current_spread: Current bid-ask spread (fetched if not provided)
            
        Returns:
            Complete signal dictionary
        """
        print(f"\n{'='*60}")
        print(f"Generating signal: {symbol} {timeframe}")
        print(f"{'='*60}")
        
        # Step 1: Fetch live data from Polygon
        print("\n[1/5] Fetching live data from Polygon API...")
        try:
            latest_bars = get_live_data(
                symbol=symbol,
                timeframe=timeframe,
                n_bars=200,
                api_key=self.polygon_api_key
            )
        except Exception as e:
            return {
                'error': f"Failed to fetch data: {e}",
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': 'ERROR',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        # Step 2: Generate ML prediction
        print("\n[2/5] Generating ML prediction...")
        try:
            signal = predict(
                symbol=symbol,
                timeframe=timeframe,
                latest_bars=latest_bars,
                models_dir=self.models_dir,
                config_path=self.config_path
            )
        except Exception as e:
            return {
                'error': f"Prediction failed: {e}",
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': 'ERROR',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        # Step 3: Apply post-processing filters
        if apply_post_filters:
            print("\n[3/5] Applying post-processing filters...")
            
            # Get current spread (estimate from data if not provided)
            if current_spread is None:
                # Estimate spread from bid-ask (use ATR as proxy)
                atr = signal.get('atr', latest_bars['close'].iloc[-1] * 0.001)
                current_spread = atr * 0.1  # Conservative estimate
            
            # Get last trade bar for cooldown
            key = f"{symbol}_{timeframe}"
            last_trade_bar = self.last_trade_bars.get(key)
            current_bar = len(latest_bars) - 1
            
            signal = apply_filters(
                signal=signal,
                current_spread=current_spread,
                atr=signal.get('atr', 1.0),
                last_trade_bar=last_trade_bar,
                current_bar=current_bar,
                cooldown_bars=5,
                spread_atr_threshold=0.5
            )
            
            if signal.get('filtered'):
                print(f"  ⚠️  Signal filtered: {signal['filter_reasons']}")
        
        # Step 4: Calculate position size
        if signal['signal'] != 'HOLD':
            print("\n[4/5] Calculating position size...")
            position_info = calculate_position_size(
                signal=signal,
                account_equity=account_equity,
                risk_per_trade_pct=0.01,
                max_position_pct=0.10
            )
            signal['position_sizing'] = position_info
            
            if position_info['position_size'] > 0:
                print(f"  Position size: {position_info['position_size']:.2f}")
                print(f"  Position value: ${position_info['position_value']:.2f}")
                print(f"  Risk amount: ${position_info['risk_amount']:.2f}")
        
        # Step 5: Add metadata
        print("\n[5/5] Finalizing signal...")
        signal['generated_at'] = datetime.now(timezone.utc).isoformat()
        signal['data_source'] = 'Polygon API'
        signal['bars_used'] = len(latest_bars)
        signal['latest_price'] = float(latest_bars['close'].iloc[-1])
        signal['latest_timestamp'] = str(latest_bars['timestamp'].iloc[-1])
        
        # Print summary
        self._print_signal_summary(signal)
        
        return signal
    
    def _print_signal_summary(self, signal: Dict):
        """Print formatted signal summary."""
        print(f"\n{'='*60}")
        print(f"SIGNAL SUMMARY")
        print(f"{'='*60}")
        print(f"Signal:      {signal['signal']}")
        print(f"Confidence:  {signal.get('confidence', 0):.1%}")
        
        if signal['signal'] in ['BUY', 'SELL']:
            print(f"Entry:       {signal['entry_ref']:.4f}")
            print(f"Stop Loss:   {signal['stop_loss']:.4f}")
            print(f"Take Profit: {signal['take_profit']:.4f}")
            print(f"Risk/Reward: {signal['expected_R']:.2f}R")
            
            if 'position_sizing' in signal:
                ps = signal['position_sizing']
                print(f"Position:    {ps['position_size']:.2f} units (${ps['position_value']:.2f})")
                print(f"Risk:        ${ps['risk_amount']:.2f}")
        
        if signal.get('filtered'):
            print(f"\n⚠️  FILTERED: {', '.join(signal['filter_reasons'])}")
        
        print(f"{'='*60}\n")
    
    def update_trade_state(self, symbol: str, timeframe: str, bar_index: int):
        """Update last trade bar for cooldown tracking."""
        key = f"{symbol}_{timeframe}"
        self.last_trade_bars[key] = bar_index


# Convenience function
def generate_live_signal(
    symbol: str,
    timeframe: str,
    account_equity: float = 100000,
    polygon_api_key: Optional[str] = None
) -> Dict:
    """
    Quick function to generate a live signal.
    
    Args:
        symbol: Trading symbol (e.g., 'XAUUSD')
        timeframe: Timeframe (e.g., '15T')
        account_equity: Account equity for position sizing
        polygon_api_key: Optional Polygon API key
        
    Returns:
        Signal dictionary
    """
    generator = LiveSignalGenerator(polygon_api_key=polygon_api_key)
    return generator.generate_signal(symbol, timeframe, account_equity)


# Example usage
if __name__ == '__main__':
    # Generate signal for XAUUSD 15-minute
    signal = generate_live_signal(
        symbol='XAUUSD',
        timeframe='15T',
        account_equity=100000
    )
    
    # Act on signal
    if signal['signal'] == 'BUY':
        print(f"\n🟢 BUY SIGNAL")
        print(f"Entry: {signal['entry_ref']}")
        print(f"Stop: {signal['stop_loss']}")
        print(f"Target: {signal['take_profit']}")
    elif signal['signal'] == 'SELL':
        print(f"\n🔴 SELL SIGNAL")
        print(f"Entry: {signal['entry_ref']}")
        print(f"Stop: {signal['stop_loss']}")
        print(f"Target: {signal['take_profit']}")
    else:
        print(f"\n⚪ HOLD - No trade")

