#!/usr/bin/env python3
"""
Worker process to continuously generate trading signals
"""
import time
from live_trading_engine import run_once

if __name__ == "__main__":
    print("🚀 Starting continuous signal generation worker...")
    
    while True:
        try:
            print("\n" + "="*80)
            print("Running signal generation cycle...")
            print("="*80)
            
            run_once()
            
            # Wait 5 minutes before next cycle
            print("\n⏳ Waiting 5 minutes until next cycle...")
            time.sleep(300)  # 5 minutes
            
        except KeyboardInterrupt:
            print("\n👋 Stopping worker...")
            break
        except Exception as e:
            print(f"\n❌ Error in worker: {e}")
            print("⏳ Waiting 1 minute before retry...")
            time.sleep(60)  # Wait 1 minute on error