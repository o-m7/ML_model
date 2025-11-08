#!/usr/bin/env python3
"""
Trade Collector - Fetch and store all executed trades for learning
Converts live trading results into training data
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client
import json

load_dotenv()

# Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


class TradeCollector:
    """Collects and analyzes live trades for continuous learning"""
    
    def __init__(self):
        self.trades_dir = Path("live_trades")
        self.trades_dir.mkdir(exist_ok=True)
        
        self.analysis_dir = Path("trade_analysis")
        self.analysis_dir.mkdir(exist_ok=True)
    
    def fetch_all_trades(self, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch all trades from Supabase
        
        Args:
            days_back: How many days of history to fetch
        
        Returns:
            DataFrame with all trade data
        """
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        
        print(f"📊 Fetching trades from last {days_back} days...")
        
        try:
            response = supabase.table("trades")\
                .select("*")\
                .gte("created_at", cutoff_date)\
                .order("created_at", desc=False)\
                .execute()
            
            if not response.data:
                print("⚠️  No trades found")
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            print(f"✅ Fetched {len(df)} trades")
            
            # Convert types
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            df['entry_price'] = pd.to_numeric(df['entry_price'])
            df['exit_price'] = pd.to_numeric(df['exit_price'])
            df['pnl'] = pd.to_numeric(df['pnl'])
            df['pnl_pct'] = pd.to_numeric(df['pnl_pct'])
            df['r_multiple'] = pd.to_numeric(df['r_multiple'])
            df['confidence'] = pd.to_numeric(df['confidence'])
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching trades: {e}")
            return pd.DataFrame()
    
    def analyze_losing_trades(self, df: pd.DataFrame) -> dict:
        """
        Analyze patterns in losing trades
        
        Returns:
            Dictionary with insights about losing trades
        """
        if df.empty:
            return {}
        
        # Separate winners and losers
        losers = df[df['pnl'] < 0].copy()
        winners = df[df['pnl'] > 0].copy()
        
        if losers.empty:
            print("🎉 No losing trades found!")
            return {"message": "No losses to analyze"}
        
        print(f"\n{'='*80}")
        print(f"📉 ANALYZING {len(losers)} LOSING TRADES")
        print(f"{'='*80}\n")
        
        analysis = {}
        
        # 1. Loss by symbol
        loss_by_symbol = losers.groupby('symbol').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_pct': 'mean',
            'confidence': 'mean'
        }).round(4)
        loss_by_symbol.columns = ['_'.join(col).strip() for col in loss_by_symbol.columns.values]
        analysis['loss_by_symbol'] = loss_by_symbol.to_dict()
        
        print("Loss by Symbol:")
        print(loss_by_symbol)
        print()
        
        # 2. Loss by timeframe
        loss_by_tf = losers.groupby('timeframe').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_pct': 'mean',
            'confidence': 'mean'
        }).round(4)
        loss_by_tf.columns = ['_'.join(col).strip() for col in loss_by_tf.columns.values]
        analysis['loss_by_timeframe'] = loss_by_tf.to_dict()
        
        print("Loss by Timeframe:")
        print(loss_by_tf)
        print()
        
        # 3. Loss by direction
        loss_by_direction = losers.groupby('direction').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_pct': 'mean',
            'confidence': 'mean'
        }).round(4)
        loss_by_direction.columns = ['_'.join(col).strip() for col in loss_by_direction.columns.values]
        analysis['loss_by_direction'] = loss_by_direction.to_dict()
        
        print("Loss by Direction:")
        print(loss_by_direction)
        print()
        
        # 4. Loss by confidence level
        losers['confidence_bucket'] = pd.cut(losers['confidence'], 
                                              bins=[0, 0.4, 0.5, 0.6, 1.0],
                                              labels=['Low', 'Medium', 'High', 'Very High'])
        
        loss_by_conf = losers.groupby('confidence_bucket').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_pct': 'mean'
        }).round(4)
        loss_by_conf.columns = ['_'.join(col).strip() for col in loss_by_conf.columns.values]
        analysis['loss_by_confidence'] = loss_by_conf.to_dict()
        
        print("Loss by Confidence Level:")
        print(loss_by_conf)
        print()
        
        # 5. Identify worst performing combinations
        losers['combo'] = losers['symbol'] + '_' + losers['timeframe'] + '_' + losers['direction']
        worst_combos = losers.groupby('combo').agg({
            'pnl': ['count', 'sum', 'mean'],
            'confidence': 'mean'
        }).round(4)
        worst_combos.columns = ['_'.join(col).strip() for col in worst_combos.columns.values]
        worst_combos = worst_combos.sort_values('pnl_sum')
        analysis['worst_combinations'] = worst_combos.head(10).to_dict()
        
        print("Top 10 Worst Combinations:")
        print(worst_combos.head(10))
        print()
        
        # 6. Exit reason analysis
        if 'exit_reason' in losers.columns:
            loss_by_exit = losers.groupby('exit_reason').agg({
                'pnl': ['count', 'sum', 'mean'],
                'pnl_pct': 'mean'
            }).round(4)
            loss_by_exit.columns = ['_'.join(col).strip() for col in loss_by_exit.columns.values]
            analysis['loss_by_exit_reason'] = loss_by_exit.to_dict()
            
            print("Loss by Exit Reason:")
            print(loss_by_exit)
            print()
        
        # 7. Compare winners vs losers
        comparison = pd.DataFrame({
            'Winners': [
                len(winners),
                winners['confidence'].mean() if not winners.empty else 0,
                winners['bars_held'].mean() if not winners.empty else 0,
                winners['pnl_pct'].mean() if not winners.empty else 0
            ],
            'Losers': [
                len(losers),
                losers['confidence'].mean(),
                losers['bars_held'].mean() if 'bars_held' in losers.columns else 0,
                losers['pnl_pct'].mean()
            ]
        }, index=['Count', 'Avg Confidence', 'Avg Bars Held', 'Avg PnL %'])
        
        analysis['winner_loser_comparison'] = comparison.to_dict()
        
        print("Winners vs Losers:")
        print(comparison)
        print()
        
        # 8. Recommendations
        recommendations = []
        
        # Check if confidence is too low
        if losers['confidence'].mean() < 0.45:
            recommendations.append("⚠️  Losing trades have LOW confidence. Increase MIN_CONFIDENCE threshold.")
        
        # Check if specific symbols are problematic
        if 'loss_by_symbol' in analysis:
            for symbol in loss_by_symbol.index:
                loss_count = loss_by_symbol.loc[symbol, 'pnl_count']
                if loss_count > 5:
                    avg_loss = loss_by_symbol.loc[symbol, 'pnl_mean']
                    if avg_loss < -50:
                        recommendations.append(f"⚠️  {symbol} has {loss_count} losses averaging ${avg_loss:.2f}. Consider excluding or retraining.")
        
        # Check if direction bias
        if not loss_by_direction.empty:
            if 'long' in loss_by_direction.index and 'short' in loss_by_direction.index:
                long_losses = loss_by_direction.loc['long', 'pnl_count']
                short_losses = loss_by_direction.loc['short', 'pnl_count']
                if long_losses > short_losses * 2:
                    recommendations.append("⚠️  Long trades losing 2x more than shorts. Model may have directional bias.")
                elif short_losses > long_losses * 2:
                    recommendations.append("⚠️  Short trades losing 2x more than longs. Model may have directional bias.")
        
        analysis['recommendations'] = recommendations
        
        if recommendations:
            print("🔍 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"  {rec}")
            print()
        
        return analysis
    
    def save_trades_for_retraining(self, df: pd.DataFrame) -> None:
        """
        Save trades in a format suitable for retraining
        Creates labeled data from actual trading results
        """
        if df.empty:
            return
        
        print(f"\n{'='*80}")
        print(f"💾 SAVING TRADES FOR RETRAINING")
        print(f"{'='*80}\n")
        
        for symbol in df['symbol'].unique():
            symbol_trades = df[df['symbol'] == symbol].copy()
            
            for tf in symbol_trades['timeframe'].unique():
                tf_trades = symbol_trades[symbol_trades['timeframe'] == tf].copy()
                
                # Create training labels from actual outcomes
                # 1 = winning trade, 0 = losing trade
                tf_trades['actual_outcome'] = (tf_trades['pnl'] > 0).astype(int)
                
                # Save to file
                filename = self.trades_dir / f"{symbol}_{tf}_live_trades.csv"
                tf_trades.to_csv(filename, index=False)
                print(f"  ✅ Saved {len(tf_trades)} trades: {filename}")
        
        print(f"\n✅ Live trades saved to {self.trades_dir}/")
    
    def generate_training_report(self, df: pd.DataFrame, analysis: dict) -> None:
        """Generate a comprehensive report for model retraining"""
        
        if df.empty:
            return
        
        report_path = self.analysis_dir / f"trade_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'total_trades': len(df),
            'winners': len(df[df['pnl'] > 0]),
            'losers': len(df[df['pnl'] < 0]),
            'win_rate': (len(df[df['pnl'] > 0]) / len(df) * 100) if len(df) > 0 else 0,
            'total_pnl': float(df['pnl'].sum()),
            'avg_win': float(df[df['pnl'] > 0]['pnl'].mean()) if len(df[df['pnl'] > 0]) > 0 else 0,
            'avg_loss': float(df[df['pnl'] < 0]['pnl'].mean()) if len(df[df['pnl'] < 0]) > 0 else 0,
            'analysis': analysis
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Report saved: {report_path}")


def main():
    """Main execution"""
    
    print(f"\n{'='*80}")
    print(f"🧠 LIVE TRADE LEARNING SYSTEM")
    print(f"{'='*80}\n")
    
    collector = TradeCollector()
    
    # Fetch all trades
    trades_df = collector.fetch_all_trades(days_back=30)
    
    if trades_df.empty:
        print("⚠️  No trades to analyze yet. Start trading first!")
        return
    
    # Analyze losing trades
    analysis = collector.analyze_losing_trades(trades_df)
    
    # Save trades for retraining
    collector.save_trades_for_retraining(trades_df)
    
    # Generate report
    collector.generate_training_report(trades_df, analysis)
    
    print(f"\n{'='*80}")
    print(f"✅ TRADE COLLECTION COMPLETE")
    print(f"{'='*80}\n")
    print(f"📁 Live trades saved to: live_trades/")
    print(f"📊 Analysis saved to: trade_analysis/")
    print(f"\nNext: Run 'python3 retrain_from_live_trades.py' to improve models\n")


if __name__ == "__main__":
    main()

