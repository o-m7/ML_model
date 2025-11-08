#!/usr/bin/env python3
"""
Trade Learning Dashboard - Visualize what the model is learning from live trades
Shows patterns in winning/losing trades and improvement over time
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import json
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 10)


class TradeLearningDashboard:
    """Dashboard for trade learning analysis"""
    
    def __init__(self):
        self.trades_dir = Path("live_trades")
        self.analysis_dir = Path("trade_analysis")
        self.output_dir = Path("dashboard_output")
        self.output_dir.mkdir(exist_ok=True)
    
    def load_all_analyses(self) -> List[Dict]:
        """Load all historical trade analyses"""
        
        if not self.analysis_dir.exists():
            return []
        
        analyses = []
        for file in sorted(self.analysis_dir.glob("trade_analysis_*.json")):
            with open(file, 'r') as f:
                data = json.load(f)
                data['filename'] = file.name
                analyses.append(data)
        
        return analyses
    
    def load_all_trades(self) -> pd.DataFrame:
        """Load all live trades from all symbol/timeframe files"""
        
        if not self.trades_dir.exists():
            return pd.DataFrame()
        
        all_trades = []
        for file in self.trades_dir.glob("*_live_trades.csv"):
            df = pd.read_csv(file)
            all_trades.append(df)
        
        if not all_trades:
            return pd.DataFrame()
        
        combined = pd.concat(all_trades, ignore_index=True)
        
        # Convert timestamps
        if 'entry_time' in combined.columns:
            combined['entry_time'] = pd.to_datetime(combined['entry_time'])
        if 'exit_time' in combined.columns:
            combined['exit_time'] = pd.to_datetime(combined['exit_time'])
        
        return combined
    
    def plot_performance_over_time(self, trades_df: pd.DataFrame) -> None:
        """Plot cumulative performance and win rate over time"""
        
        if trades_df.empty or 'exit_time' not in trades_df.columns:
            print("⚠️  No time-series data available")
            return
        
        # Sort by exit time
        trades_df = trades_df.sort_values('exit_time')
        
        # Calculate cumulative metrics
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['rolling_win_rate'] = trades_df['pnl'].rolling(window=20, min_periods=1).apply(
            lambda x: (x > 0).sum() / len(x) * 100
        )
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Cumulative P&L
        ax1 = axes[0]
        ax1.plot(trades_df['exit_time'], trades_df['cumulative_pnl'], 
                linewidth=2, color='#2196F3')
        ax1.fill_between(trades_df['exit_time'], 0, trades_df['cumulative_pnl'], 
                         alpha=0.3, color='#2196F3')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_title('Cumulative P&L Over Time', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Cumulative P&L ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rolling Win Rate
        ax2 = axes[1]
        ax2.plot(trades_df['exit_time'], trades_df['rolling_win_rate'], 
                linewidth=2, color='#4CAF50')
        ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% Breakeven')
        ax2.fill_between(trades_df['exit_time'], 50, trades_df['rolling_win_rate'], 
                         alpha=0.3, color='#4CAF50')
        ax2.set_title('Rolling Win Rate (20-trade window)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Win Rate (%)', fontsize=12)
        ax2.set_ylim([0, 100])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "performance_over_time.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved: {output_file}")
        plt.close()
    
    def plot_winner_vs_loser_analysis(self, trades_df: pd.DataFrame) -> None:
        """Compare characteristics of winning vs losing trades"""
        
        if trades_df.empty:
            return
        
        winners = trades_df[trades_df['pnl'] > 0]
        losers = trades_df[trades_df['pnl'] < 0]
        
        if winners.empty or losers.empty:
            print("⚠️  Not enough data for winner/loser comparison")
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Confidence distribution
        ax1 = axes[0, 0]
        ax1.hist([winners['confidence'], losers['confidence']], 
                bins=20, label=['Winners', 'Losers'], 
                color=['#4CAF50', '#F44336'], alpha=0.7)
        ax1.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: P&L distribution
        ax2 = axes[0, 1]
        ax2.hist([winners['pnl'], losers['pnl']], 
                bins=30, label=['Winners', 'Losers'],
                color=['#4CAF50', '#F44336'], alpha=0.7)
        ax2.set_title('P&L Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('P&L ($)')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Bars held comparison
        if 'bars_held' in trades_df.columns:
            ax3 = axes[1, 0]
            data = [
                winners['bars_held'].dropna(),
                losers['bars_held'].dropna()
            ]
            ax3.boxplot(data, labels=['Winners', 'Losers'],
                       patch_artist=True,
                       boxprops=dict(facecolor='#4CAF50', alpha=0.7))
            ax3.set_title('Bars Held Comparison', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Bars Held')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Win/Loss by Symbol
        ax4 = axes[1, 1]
        symbol_stats = trades_df.groupby('symbol').agg({
            'pnl': lambda x: (x > 0).sum() / len(x) * 100
        }).sort_values('pnl', ascending=False)
        
        colors = ['#4CAF50' if x >= 50 else '#F44336' for x in symbol_stats['pnl']]
        symbol_stats['pnl'].plot(kind='barh', ax=ax4, color=colors, alpha=0.7)
        ax4.axvline(x=50, color='orange', linestyle='--', alpha=0.5)
        ax4.set_title('Win Rate by Symbol', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Win Rate (%)')
        ax4.set_ylabel('Symbol')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "winner_vs_loser_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved: {output_file}")
        plt.close()
    
    def plot_learning_improvements(self, analyses: List[Dict]) -> None:
        """Plot how the model is improving over time"""
        
        if len(analyses) < 2:
            print("⚠️  Need at least 2 analysis points to show improvement")
            return
        
        # Extract metrics over time
        timestamps = []
        win_rates = []
        total_pnls = []
        total_trades = []
        
        for analysis in analyses:
            if 'generated_at' in analysis:
                timestamps.append(pd.to_datetime(analysis['generated_at']))
                win_rates.append(analysis.get('win_rate', 0))
                total_pnls.append(analysis.get('total_pnl', 0))
                total_trades.append(analysis.get('total_trades', 0))
        
        if not timestamps:
            return
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Win Rate Improvement
        ax1 = axes[0]
        ax1.plot(timestamps, win_rates, marker='o', linewidth=2, 
                markersize=8, color='#2196F3')
        ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% Target')
        ax1.set_title('Win Rate Improvement Over Time', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Win Rate (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Total P&L Growth
        ax2 = axes[1]
        ax2.plot(timestamps, total_pnls, marker='s', linewidth=2,
                markersize=8, color='#4CAF50')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Total P&L Growth', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Total P&L ($)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Trading Volume
        ax3 = axes[2]
        ax3.bar(timestamps, total_trades, color='#9C27B0', alpha=0.7)
        ax3.set_title('Trading Volume Over Time', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Number of Trades')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "learning_improvements.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved: {output_file}")
        plt.close()
    
    def generate_text_report(self, trades_df: pd.DataFrame, analyses: List[Dict]) -> None:
        """Generate a comprehensive text report"""
        
        report_file = self.output_dir / f"learning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(" " * 20 + "TRADE LEARNING REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if not trades_df.empty:
                f.write("OVERALL STATISTICS:\n")
                f.write("-"*80 + "\n")
                f.write(f"Total Trades: {len(trades_df)}\n")
                f.write(f"Winners: {len(trades_df[trades_df['pnl'] > 0])}\n")
                f.write(f"Losers: {len(trades_df[trades_df['pnl'] < 0])}\n")
                f.write(f"Win Rate: {len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100:.2f}%\n")
                f.write(f"Total P&L: ${trades_df['pnl'].sum():.2f}\n")
                f.write(f"Avg Win: ${trades_df[trades_df['pnl'] > 0]['pnl'].mean():.2f}\n")
                f.write(f"Avg Loss: ${trades_df[trades_df['pnl'] < 0]['pnl'].mean():.2f}\n\n")
                
                f.write("PERFORMANCE BY SYMBOL:\n")
                f.write("-"*80 + "\n")
                symbol_stats = trades_df.groupby('symbol').agg({
                    'pnl': ['count', 'sum', lambda x: (x > 0).sum() / len(x) * 100]
                }).round(2)
                f.write(symbol_stats.to_string() + "\n\n")
                
                f.write("PERFORMANCE BY TIMEFRAME:\n")
                f.write("-"*80 + "\n")
                tf_stats = trades_df.groupby('timeframe').agg({
                    'pnl': ['count', 'sum', lambda x: (x > 0).sum() / len(x) * 100]
                }).round(2)
                f.write(tf_stats.to_string() + "\n\n")
            
            if analyses:
                f.write("LEARNING HISTORY:\n")
                f.write("-"*80 + "\n")
                f.write(f"Number of learning cycles: {len(analyses)}\n")
                f.write(f"Latest analysis: {analyses[-1].get('generated_at', 'N/A')}\n\n")
                
                if len(analyses) >= 2:
                    f.write("IMPROVEMENT METRICS:\n")
                    f.write("-"*80 + "\n")
                    first = analyses[0]
                    latest = analyses[-1]
                    
                    wr_change = latest.get('win_rate', 0) - first.get('win_rate', 0)
                    pnl_change = latest.get('total_pnl', 0) - first.get('total_pnl', 0)
                    
                    f.write(f"Win Rate Change: {wr_change:+.2f}%\n")
                    f.write(f"P&L Change: ${pnl_change:+.2f}\n")
                    f.write(f"Total Trades Growth: {latest.get('total_trades', 0) - first.get('total_trades', 0):+d}\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"  📄 Saved: {report_file}")
    
    def run(self) -> None:
        """Generate all dashboard outputs"""
        
        print(f"\n{'='*80}")
        print(f"📊 TRADE LEARNING DASHBOARD")
        print(f"{'='*80}\n")
        
        # Load data
        print("Loading data...")
        trades_df = self.load_all_trades()
        analyses = self.load_all_analyses()
        
        if trades_df.empty:
            print("⚠️  No trade data available yet. Start trading first!")
            return
        
        print(f"  ✅ Loaded {len(trades_df)} trades")
        print(f"  ✅ Loaded {len(analyses)} analysis reports\n")
        
        # Generate visualizations
        print("Generating visualizations...")
        self.plot_performance_over_time(trades_df)
        self.plot_winner_vs_loser_analysis(trades_df)
        
        if len(analyses) >= 2:
            self.plot_learning_improvements(analyses)
        
        # Generate text report
        self.generate_text_report(trades_df, analyses)
        
        print(f"\n{'='*80}")
        print(f"✅ DASHBOARD COMPLETE")
        print(f"{'='*80}")
        print(f"📁 Output saved to: {self.output_dir}/")
        print(f"\nGenerated files:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"  - {file.name}")
        print()


def main():
    """Main execution"""
    
    # Check for matplotlib
    try:
        import matplotlib
        import seaborn
    except ImportError:
        print("❌ Missing dependencies. Install with:")
        print("   pip install matplotlib seaborn")
        sys.exit(1)
    
    dashboard = TradeLearningDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()

