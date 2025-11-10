#!/usr/bin/env python3
"""
Continuous worker that runs signal generation every N minutes.
Perfect for Railway, Render, or any cloud platform.
"""
import time
import sys
from datetime import datetime
from signal_generator import main as generate_signals

INTERVAL_MINUTES = 3  # Run every 3 minutes

if __name__ == "__main__":
    print(f"🚀 Starting continuous signal generator (interval: {INTERVAL_MINUTES} minutes)")
    print(f"⏰ Started at: {datetime.now()}\n")
    
    iteration = 0
    while True:
        try:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"🔄 ITERATION #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")
            
            generate_signals()
            
            print(f"\n✅ Iteration #{iteration} complete. Sleeping for {INTERVAL_MINUTES} minutes...")
            time.sleep(INTERVAL_MINUTES * 60)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Worker stopped by user")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error in iteration #{iteration}: {e}")
            print(f"⏳ Retrying in {INTERVAL_MINUTES} minutes...")
            time.sleep(INTERVAL_MINUTES * 60)

