#!/usr/bin/env python3
"""
Quick Start - Live Signal Generation System
===========================================

Initializes and tests the complete signal generation pipeline.
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check required dependencies."""
    logger.info("Checking dependencies...")
    
    required = [
        'numpy',
        'pandas',
        'xgboost',
        'onnx',
        'onnxruntime',
        'websockets',
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            logger.info(f"  ✓ {package}")
        except ImportError:
            logger.error(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error("Install with: pip install -r requirements_live_signals.txt")
        return False
    
    return True


def check_environment():
    """Check environment variables."""
    logger.info("\nChecking environment variables...")
    
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    required_vars = [
        'POLYGON_API_KEY',
        'SUPABASE_URL',
        'SUPABASE_KEY',
    ]
    
    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask value for security
            masked = f"{value[:6]}...{value[-4:]}" if len(value) > 10 else "***"
            logger.info(f"  ✓ {var} = {masked}")
        else:
            logger.warning(f"  ✗ {var} (missing)")
            missing.append(var)
    
    if missing:
        logger.warning(f"\nMissing environment variables: {', '.join(missing)}")
        logger.warning("Add to .env file or export as environment variables")
        return len(missing) == 0
    
    return True


def check_models():
    """Check ONNX models."""
    logger.info("\nChecking ONNX models...")
    
    models_dir = Path("artifacts/onnx_models")
    
    if not models_dir.exists():
        logger.warning(f"  ONNX models directory not found: {models_dir}")
        logger.info("  Run: python model_converter.py")
        return False
    
    models = list(models_dir.glob("*.onnx"))
    
    if not models:
        logger.warning("  No ONNX models found")
        logger.info("  Run: python model_converter.py")
        return False
    
    for model in models:
        logger.info(f"  ✓ {model.name}")
    
    return True


def run_test():
    """Run test mode signal generation."""
    logger.info("\nRunning test mode...")
    logger.info("=" * 80)
    
    try:
        from run_live_signals import standalone_test
        standalone_test()
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


def print_next_steps():
    """Print next steps."""
    logger.info("\n" + "=" * 80)
    logger.info("NEXT STEPS")
    logger.info("=" * 80)
    logger.info("""
1. Test mode completed successfully!

2. To run live signal generation:
   python run_live_signals.py

3. To deploy on GitHub:
   - Add POLYGON_API_KEY secret to GitHub repo
   - Add SUPABASE_URL secret
   - Add SUPABASE_KEY secret
   - Workflow will run automatically on schedule

4. Monitor signals in Supabase:
   Dashboard → signals table

5. Connect frontend to Supabase realtime:
   Supabase JS client → subscribe to 'signals' table

6. Documentation:
   cat LIVE_SIGNALS_DEPLOYMENT.md
    """)


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("LIVE SIGNAL GENERATOR - QUICK START")
    logger.info("=" * 80)
    
    # Checks
    if not check_dependencies():
        return 1
    
    if not check_environment():
        logger.warning("Proceeding with missing environment variables...")
    
    if not check_models():
        logger.info("Converting models...")
        try:
            from model_converter import batch_convert_models
            batch_convert_models()
        except Exception as e:
            logger.error(f"Model conversion failed: {e}")
            return 1
    
    # Run test
    if not run_test():
        return 1
    
    # Success
    print_next_steps()
    
    logger.info("\n✓ Quick start completed successfully!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
