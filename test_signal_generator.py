"""
Signal Generator Test with Mock Data
====================================

Demonstrates the ensemble signal generator with sample feature data.
Shows how to generate, display, and export trading signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from signal_generator_ensemble import (
    EnsembleSignalGenerator,
    display_signal_summary,
    save_signal,
    export_signal_csv,
    ARTIFACTS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_features() -> pd.DataFrame:
    """
    Create sample feature data for testing.
    
    In production, this would come from:
    - Live quote data (bid/ask spreads, microprice, etc.)
    - Live OHLCV data (candlestick features, regimes, etc.)
    """
    # This is a minimal example - in reality you'd have 50+ features per model
    sample_data = {
        # Basic OHLCV
        'open': [3980.5],
        'high': [3982.0],
        'low': [3979.0],
        'close': [3981.5],
        'volume': [1500],
        
        # Mock additional features (normally computed from live data)
        'returns': [0.002],
        'volatility': [0.015],
        'atr': [1.5],
        'rsi': [55.0],
        'macd': [0.5],
        'ema_10': [3980.0],
        'ema_20': [3979.5],
    }
    
    df = pd.DataFrame(sample_data)
    return df


def test_generator():
    """Test the ensemble signal generator with mock data."""
    logger.info("\n" + "=" * 100)
    logger.info("ENSEMBLE SIGNAL GENERATOR - TEST WITH MOCK DATA")
    logger.info("=" * 100)
    
    try:
        # Initialize generator
        logger.info("\n1. Initializing generator...")
        generator = EnsembleSignalGenerator(ARTIFACTS_DIR)
        
        # Create sample features
        logger.info("2. Creating sample feature data...")
        features_df = create_sample_features()
        logger.info(f"   Features shape: {features_df.shape}")
        logger.info(f"   Features: {list(features_df.columns)[:5]}... ({len(features_df.columns)} total)")
        
        # Generate single model signals
        logger.info("\n3. Testing individual model signals...")
        logger.info("-" * 100)
        
        sample_models = list(generator.models.keys())[:4]  # Test first 4 models
        
        for model_name in sample_models:
            try:
                signal = generator.generate_model_signal(model_name, features_df)
                if signal:
                    direction_str = "LONG" if signal.direction == 1 else "SHORT" if signal.direction == -1 else "NEUTRAL"
                    logger.info(
                        f"   {model_name:<30} | {direction_str:<8} | "
                        f"Prob: {signal.probability:.4f} | Conf: {signal.confidence:.4f}"
                    )
                else:
                    logger.warning(f"   {model_name:<30} | No signal generated")
            except Exception as e:
                logger.warning(f"   {model_name:<30} | Error: {e}")
        
        # Generate ensemble signal
        logger.info("\n4. Generating ensemble signal...")
        ensemble_signal = generator.generate_ensemble_signal(features_df)
        
        # Display results
        logger.info("\n5. Display results:")
        display_signal_summary(ensemble_signal)
        
        # Export signals
        logger.info("6. Exporting signals...")
        json_path = save_signal(ensemble_signal)
        csv_path = export_signal_csv(ensemble_signal)
        
        logger.info(f"\n✓ JSON export: {json_path}")
        logger.info(f"✓ CSV export: {csv_path}")
        
        # Summary stats
        logger.info("\n7. Summary Statistics:")
        logger.info("-" * 100)
        logger.info(f"   Total Models Voted: {ensemble_signal.total_votes}")
        logger.info(f"   Long Votes: {ensemble_signal.long_votes} ({ensemble_signal.long_votes/ensemble_signal.total_votes*100:.1f}%)")
        logger.info(f"   Short Votes: {ensemble_signal.short_votes} ({ensemble_signal.short_votes/ensemble_signal.total_votes*100:.1f}%)")
        logger.info(f"   Neutral Votes: {ensemble_signal.neutral_votes} ({ensemble_signal.neutral_votes/ensemble_signal.total_votes*100:.1f}%)")
        logger.info(f"   Consensus Strength: {ensemble_signal.consensus_strength}")
        logger.info(f"   Agreement Level: {ensemble_signal.agreement_level}")
        
        logger.info("\n" + "=" * 100)
        logger.info("✅ TEST COMPLETE - SIGNAL GENERATOR WORKING")
        logger.info("=" * 100 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    test_generator()
