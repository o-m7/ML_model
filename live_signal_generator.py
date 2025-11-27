"""
Live Trading Signal Generator
==============================

Main orchestrator that:
1. Loads live market data (quote + OHLCV)
2. Computes required features in real-time
3. Generates ensemble signals from all 8 models
4. Exports signals for trading/alerts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
import json
from pathlib import Path

from signal_generator_ensemble import (
    EnsembleSignalGenerator,
    display_signal_summary,
    save_signal,
    export_signal_csv
)
from feature_computer import FeatureComputer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class LiveSignalGenerator:
    """
    Live trading signal generator orchestrator.
    
    Workflow:
    1. Load/stream market data
    2. Compute features in real-time
    3. Generate ensemble signals
    4. Export for trading platform
    """
    
    def __init__(self, artifacts_dir: str = "artifacts", signal_dir: str = "signals"):
        """
        Initialize live signal generator.
        
        Args:
            artifacts_dir: Directory with trained models
            signal_dir: Output directory for signals
        """
        self.artifacts_dir = artifacts_dir
        self.signal_dir = signal_dir
        self.ensemble_generator = EnsembleSignalGenerator(artifacts_dir)
        self.feature_computer = FeatureComputer()
        
        # Create signal directory if needed
        Path(signal_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ Live Signal Generator initialized")
        logger.info(f"  Models: {len(self.ensemble_generator.models)} loaded")
        logger.info(f"  Signal output: {signal_dir}/")
    
    def generate_signal_from_market_data(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        quote_data: Optional[pd.DataFrame] = None,
        use_timeframe: str = "1T"
    ) -> Dict:
        """
        Generate ensemble signal from raw market data.
        
        Args:
            ohlcv_data: Dict of DataFrames by timeframe: {"1T": df, "5T": df, ...}
            quote_data: Optional bid/ask data for quote models
            use_timeframe: Which timeframe to use for primary signal
            
        Returns:
            Dictionary with signal info
        """
        try:
            logger.info("=" * 80)
            logger.info("GENERATING LIVE TRADING SIGNAL")
            logger.info("=" * 80)
            
            # Compute OHLCV features for primary timeframe
            logger.info(f"\n1. Computing OHLCV features ({use_timeframe})...")
            if use_timeframe not in ohlcv_data:
                raise ValueError(f"Timeframe {use_timeframe} not in data")
            
            ohlcv_df = ohlcv_data[use_timeframe]
            features_df = self.feature_computer.compute_ohlcv_features(
                ohlcv_df, 
                timeframe=use_timeframe
            )
            
            logger.info(f"   ✓ Features: {len(features_df.columns)} computed")
            logger.info(f"   Latest bar: {ohlcv_df.index[-1]}")
            
            # Get latest row for prediction
            latest_features = features_df.iloc[-1:].copy()
            
            # Generate ensemble signal
            logger.info("\n2. Generating ensemble signal...")
            ensemble_signal = self.ensemble_generator.generate_ensemble_signal(
                latest_features
            )
            
            # Display results
            logger.info("\n3. Signal Results:")
            display_signal_summary(ensemble_signal)
            
            # Export signals
            logger.info("\n4. Exporting signals...")
            json_path = save_signal(ensemble_signal, self.signal_dir)
            csv_path = export_signal_csv(ensemble_signal, self.signal_dir)
            
            logger.info(f"   ✓ JSON: {json_path}")
            logger.info(f"   ✓ CSV: {csv_path}")
            
            result = {
                'timestamp': datetime.now(),
                'signal': ensemble_signal.signal_strength,
                'agreement': ensemble_signal.agreement_level,
                'long_votes': ensemble_signal.long_votes,
                'short_votes': ensemble_signal.short_votes,
                'neutral_votes': ensemble_signal.neutral_votes,
                'consensus_strength': ensemble_signal.consensus_strength,
                'model_count': len(ensemble_signal.individual_signals),
                'json_export': json_path,
                'csv_export': csv_path,
            }
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ SIGNAL GENERATION COMPLETE")
            logger.info("=" * 80 + "\n")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}", exc_info=True)
            return None
    
    def generate_multi_timeframe_signals(
        self,
        ohlcv_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        Generate signals for multiple timeframes and create composite view.
        
        Args:
            ohlcv_data: Dict of DataFrames by timeframe
            
        Returns:
            Dictionary with signals per timeframe + composite
        """
        results = {}
        
        logger.info("=" * 80)
        logger.info("MULTI-TIMEFRAME SIGNAL ANALYSIS")
        logger.info("=" * 80)
        
        for timeframe in ['1T', '5T', '15T', '30T']:
            if timeframe not in ohlcv_data:
                logger.warning(f"⚠ Timeframe {timeframe} not available")
                continue
            
            logger.info(f"\n→ Processing {timeframe}...")
            ohlcv_df = ohlcv_data[timeframe]
            features_df = self.feature_computer.compute_ohlcv_features(
                ohlcv_df, 
                timeframe=timeframe
            )
            latest_features = features_df.iloc[-1:].copy()
            
            # Generate signal
            try:
                signal = self.ensemble_generator.generate_ensemble_signal(latest_features)
                results[timeframe] = {
                    'signal': signal.signal_strength,
                    'long_votes': signal.long_votes,
                    'short_votes': signal.short_votes,
                    'consensus': signal.consensus_strength,
                    'agreement': signal.agreement_level,
                }
                logger.info(f"  {signal.signal_strength} | {signal.agreement_level}")
            except Exception as e:
                logger.warning(f"  Failed: {e}")
        
        # Create composite signal (voting across timeframes)
        if results:
            long_count = sum(1 for r in results.values() if 'LONG' in r['signal'])
            short_count = sum(1 for r in results.values() if 'SHORT' in r['signal'])
            
            composite_signal = "NEUTRAL"
            if long_count > short_count:
                composite_signal = "LONG"
            elif short_count > long_count:
                composite_signal = "SHORT"
            
            results['_composite'] = {
                'signal': composite_signal,
                'long_timeframes': long_count,
                'short_timeframes': short_count,
                'neutral_timeframes': len(results) - long_count - short_count,
            }
            
            logger.info("\n" + "-" * 80)
            logger.info(f"COMPOSITE SIGNAL: {composite_signal}")
            logger.info(f"  Bullish timeframes: {long_count}")
            logger.info(f"  Bearish timeframes: {short_count}")
            logger.info("=" * 80 + "\n")
        
        return results


# ============================================================================
# EXAMPLE USAGE WITH SAMPLE DATA
# ============================================================================

def demo_signal_generation():
    """Demo: Generate signals with sample market data."""
    
    logger.info("\n" + "=" * 80)
    logger.info("DEMO: LIVE SIGNAL GENERATION")
    logger.info("=" * 80 + "\n")
    
    # Create sample market data for all timeframes
    dates_1t = pd.date_range(start='2024-11-20 00:00', periods=100, freq='1T')
    dates_5t = pd.date_range(start='2024-11-20 00:00', periods=100, freq='5T')
    dates_15t = pd.date_range(start='2024-11-20 00:00', periods=100, freq='15T')
    dates_30t = pd.date_range(start='2024-11-20 00:00', periods=100, freq='30T')
    
    np.random.seed(42)
    
    ohlcv_data = {
        '1T': pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 3980,
            'high': np.random.randn(100).cumsum() + 3985,
            'low': np.random.randn(100).cumsum() + 3975,
            'close': np.random.randn(100).cumsum() + 3980,
            'volume': np.random.randint(1000, 5000, 100),
            'timestamp': dates_1t,
        }).set_index('timestamp'),
        '5T': pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 3980,
            'high': np.random.randn(100).cumsum() + 3985,
            'low': np.random.randn(100).cumsum() + 3975,
            'close': np.random.randn(100).cumsum() + 3980,
            'volume': np.random.randint(5000, 20000, 100),
            'timestamp': dates_5t,
        }).set_index('timestamp'),
        '15T': pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 3980,
            'high': np.random.randn(100).cumsum() + 3985,
            'low': np.random.randn(100).cumsum() + 3975,
            'close': np.random.randn(100).cumsum() + 3980,
            'volume': np.random.randint(10000, 50000, 100),
            'timestamp': dates_15t,
        }).set_index('timestamp'),
        '30T': pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 3980,
            'high': np.random.randn(100).cumsum() + 3985,
            'low': np.random.randn(100).cumsum() + 3975,
            'close': np.random.randn(100).cumsum() + 3980,
            'volume': np.random.randint(20000, 100000, 100),
            'timestamp': dates_30t,
        }).set_index('timestamp'),
    }
    
    # Initialize generator
    generator = LiveSignalGenerator()
    
    # Generate signals for primary timeframe
    logger.info("SINGLE TIMEFRAME SIGNAL (1T)")
    result = generator.generate_signal_from_market_data(ohlcv_data, use_timeframe='1T')
    
    # Generate multi-timeframe analysis
    logger.info("\nMULTI-TIMEFRAME ANALYSIS")
    multi_results = generator.generate_multi_timeframe_signals(ohlcv_data)


if __name__ == "__main__":
    demo_signal_generation()
