#!/usr/bin/env python3
"""
MODEL LEARNING DIAGNOSTIC
=========================

Win rate < 40% = Model is NOT learning properly.

This script investigates:
1. Are features actually predictive of future price?
2. Is the model just predicting "Flat" all the time?
3. Are labels correctly aligned with outcomes?
4. Is there a fundamental flaw in the approach?
5. What trades is the model ACTUALLY taking?
"""

import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class DiagnosticConfig:
    FEATURE_STORE = Path("feature_store")
    TRAIN_START = "2019-01-01"
    TRAIN_END = "2025-10-22"
    OOS_MONTHS = 12


def analyze_labels(df: pd.DataFrame, symbol: str, tf: str):
    """Analyze if labels make sense."""
    
    print(f"\n{'='*80}")
    print(f"LABEL ANALYSIS: {symbol} {tf}")
    print(f"{'='*80}\n")
    
    # Check label distribution
    counts = df['target'].value_counts()
    total = len(df)
    
    print("Label Distribution:")
    print(f"  Flat (0): {counts.get(0, 0):,} ({counts.get(0, 0)/total*100:.1f}%)")
    print(f"  Up (1):   {counts.get(1, 0):,} ({counts.get(1, 0)/total*100:.1f}%)")
    print(f"  Down (2): {counts.get(2, 0):,} ({counts.get(2, 0)/total*100:.1f}%)")
    
    # Check actual outcomes for each label
    print("\nActual Outcomes by Label (next 10 bars):")
    for label in [0, 1, 2]:
        label_data = df[df['target'] == label].copy()
        if len(label_data) > 0:
            future_returns = []
            for idx in label_data.index:
                if idx + 10 < len(df):
                    future_ret = (df.iloc[idx+10]['close'] - df.iloc[idx]['close']) / df.iloc[idx]['close']
                    future_returns.append(future_ret)
            
            if future_returns:
                avg_ret = np.mean(future_returns) * 100
                win_rate = sum(1 for r in future_returns if (label == 1 and r > 0) or (label == 2 and r < 0) or (label == 0 and abs(r) < 0.002)) / len(future_returns) * 100
                
                label_name = ['Flat', 'Up', 'Down'][label]
                print(f"  {label_name}: Avg future return = {avg_ret:.3f}%, Correct = {win_rate:.1f}%")
    
    print()


def analyze_features(df: pd.DataFrame, features: list, symbol: str, tf: str):
    """Analyze if features are predictive."""
    
    print(f"\n{'='*80}")
    print(f"FEATURE PREDICTIVENESS: {symbol} {tf}")
    print(f"{'='*80}\n")
    
    # Calculate future returns
    df['future_ret_5'] = df['close'].pct_change(5).shift(-5)
    df['future_ret_10'] = df['close'].pct_change(10).shift(-10)
    df['future_ret_20'] = df['close'].pct_change(20).shift(-20)
    
    # Find most predictive features
    correlations = []
    for feat in features:
        if feat in df.columns:
            corr_5 = abs(df[feat].corr(df['future_ret_5']))
            corr_10 = abs(df[feat].corr(df['future_ret_10']))
            corr_20 = abs(df[feat].corr(df['future_ret_20']))
            avg_corr = (corr_5 + corr_10 + corr_20) / 3
            correlations.append((feat, avg_corr, corr_5, corr_10, corr_20))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 15 Most Predictive Features:")
    for i, (feat, avg_corr, c5, c10, c20) in enumerate(correlations[:15], 1):
        print(f"  {i:2d}. {feat:30s} | Avg: {avg_corr:.4f} | 5bar: {c5:.4f} | 10bar: {c10:.4f} | 20bar: {c20:.4f}")
    
    print("\nBottom 10 Least Predictive Features:")
    for i, (feat, avg_corr, c5, c10, c20) in enumerate(correlations[-10:], 1):
        print(f"  {i:2d}. {feat:30s} | Avg: {avg_corr:.4f}")
    
    # Overall assessment
    avg_predictiveness = np.mean([c[1] for c in correlations])
    print(f"\n📊 Average Feature Predictiveness: {avg_predictiveness:.4f}")
    
    if avg_predictiveness < 0.01:
        print("  ❌ CRITICAL: Features are NOT predictive!")
    elif avg_predictiveness < 0.02:
        print("  ⚠️  WARNING: Features have weak predictive power")
    elif avg_predictiveness < 0.03:
        print("  ✅ OK: Features have moderate predictive power")
    else:
        print("  ✅ GOOD: Features have strong predictive power")
    
    print()


def analyze_model_predictions(model, X_test, y_test, df_test, symbol: str, tf: str):
    """Analyze what the model is actually predicting."""
    
    print(f"\n{'='*80}")
    print(f"MODEL PREDICTION ANALYSIS: {symbol} {tf}")
    print(f"{'='*80}\n")
    
    # Get predictions
    probs = model.predict_proba(X_test)
    predictions = np.argmax(probs, axis=1)
    
    # Prediction distribution
    pred_counts = pd.Series(predictions).value_counts()
    total = len(predictions)
    
    print("Model Prediction Distribution:")
    print(f"  Predicts Flat (0): {pred_counts.get(0, 0):,} ({pred_counts.get(0, 0)/total*100:.1f}%)")
    print(f"  Predicts Up (1):   {pred_counts.get(1, 0):,} ({pred_counts.get(1, 0)/total*100:.1f}%)")
    print(f"  Predicts Down (2): {pred_counts.get(2, 0):,} ({pred_counts.get(2, 0)/total*100:.1f}%)")
    
    # Check if model is just predicting one class
    if pred_counts.get(0, 0) / total > 0.95:
        print("\n  ❌ PROBLEM: Model is predicting Flat 95%+ of the time!")
        print("     → Model learned to always predict Flat (safest but useless)")
    
    # Accuracy by class
    print("\nModel Accuracy by Class:")
    for label in [0, 1, 2]:
        mask = y_test == label
        if mask.sum() > 0:
            correct = (predictions[mask] == label).sum()
            accuracy = correct / mask.sum() * 100
            label_name = ['Flat', 'Up', 'Down'][label]
            print(f"  {label_name}: {accuracy:.1f}% ({correct}/{mask.sum()})")
    
    # Overall accuracy
    overall_acc = (predictions == y_test).sum() / len(y_test) * 100
    print(f"\nOverall Accuracy: {overall_acc:.1f}%")
    
    # Confidence analysis
    print("\nPrediction Confidence:")
    max_probs = np.max(probs, axis=1)
    print(f"  Average confidence: {max_probs.mean():.3f}")
    print(f"  High confidence (>0.5): {(max_probs > 0.5).sum()/len(max_probs)*100:.1f}%")
    print(f"  Very high confidence (>0.7): {(max_probs > 0.7).sum()/len(max_probs)*100:.1f}%")
    
    if max_probs.mean() < 0.4:
        print("  ⚠️  Model is NOT confident in its predictions")
    
    # Actual outcomes of model predictions
    print("\nActual Outcomes When Model Predicts:")
    df_test_copy = df_test.copy().reset_index(drop=True)
    
    for pred_class in [0, 1, 2]:
        pred_mask = predictions == pred_class
        if pred_mask.sum() > 0:
            indices = np.where(pred_mask)[0]
            future_returns = []
            
            for idx in indices:
                if idx + 10 < len(df_test_copy):
                    future_ret = (df_test_copy.iloc[idx+10]['close'] - df_test_copy.iloc[idx]['close']) / df_test_copy.iloc[idx]['close']
                    future_returns.append(future_ret)
            
            if future_returns:
                avg_ret = np.mean(future_returns) * 100
                if pred_class == 0:
                    correct = sum(1 for r in future_returns if abs(r) < 0.002)
                elif pred_class == 1:
                    correct = sum(1 for r in future_returns if r > 0)
                else:
                    correct = sum(1 for r in future_returns if r < 0)
                
                accuracy = correct / len(future_returns) * 100
                label_name = ['Flat', 'Up', 'Down'][pred_class]
                
                print(f"  {label_name}: Avg return = {avg_ret:.3f}%, Correct direction = {accuracy:.1f}%")
                
                if pred_class in [1, 2] and accuracy < 45:
                    print(f"    ❌ Model is WRONG more than 55% of the time when predicting {label_name}!")
    
    print()


def analyze_signal_quality(model, X_test, df_test, symbol: str, tf: str, 
                           min_conf: float = 0.35, min_edge: float = 0.08):
    """Analyze the quality of trading signals."""
    
    print(f"\n{'='*80}")
    print(f"SIGNAL QUALITY ANALYSIS: {symbol} {tf}")
    print(f"{'='*80}\n")
    
    print(f"Signal Thresholds: Min Conf = {min_conf}, Min Edge = {min_edge}")
    
    probs = model.predict_proba(X_test)
    
    # Generate signals
    signals_long = []
    signals_short = []
    signal_confidences_long = []
    signal_confidences_short = []
    
    for i in range(len(probs)):
        max_prob = np.max(probs[i])
        sorted_probs = sorted(probs[i], reverse=True)
        edge = sorted_probs[0] - sorted_probs[1]
        
        if probs[i][1] == max_prob and probs[i][1] >= min_conf and edge >= min_edge:
            signals_long.append(i)
            signal_confidences_long.append(probs[i][1])
        elif probs[i][2] == max_prob and probs[i][2] >= min_conf and edge >= min_edge:
            signals_short.append(i)
            signal_confidences_short.append(probs[i][2])
    
    print(f"\nSignals Generated:")
    print(f"  Long signals:  {len(signals_long)} ({len(signals_long)/len(probs)*100:.1f}%)")
    print(f"  Short signals: {len(signals_short)} ({len(signals_short)/len(probs)*100:.1f}%)")
    print(f"  Total signals: {len(signals_long) + len(signals_short)}")
    
    if len(signals_long) + len(signals_short) == 0:
        print("\n  ❌ CRITICAL: NO SIGNALS GENERATED!")
        print("     → Thresholds are too strict OR model has no confidence")
        return
    
    # Analyze signal outcomes
    df_test_copy = df_test.copy().reset_index(drop=True)
    
    print("\nSignal Outcome Analysis (next 20 bars):")
    
    for signal_type, signals, confidences in [('LONG', signals_long, signal_confidences_long), 
                                               ('SHORT', signals_short, signal_confidences_short)]:
        if len(signals) > 0:
            wins = 0
            losses = 0
            total_return = 0
            
            for idx, conf in zip(signals, confidences):
                if idx + 20 < len(df_test_copy):
                    entry = df_test_copy.iloc[idx]['close']
                    
                    # Check best and worst case in next 20 bars
                    future_highs = df_test_copy.iloc[idx+1:idx+21]['high'].values
                    future_lows = df_test_copy.iloc[idx+1:idx+21]['low'].values
                    
                    atr = df_test_copy.iloc[idx].get('atr14', entry * 0.02)
                    
                    if signal_type == 'LONG':
                        tp = entry + (atr * 1.5)
                        sl = entry - (atr * 1.0)
                        
                        # Check if TP hit
                        if (future_highs >= tp).any():
                            tp_idx = np.where(future_highs >= tp)[0][0]
                            sl_idx = np.where(future_lows <= sl)[0]
                            
                            if len(sl_idx) == 0 or tp_idx < sl_idx[0]:
                                wins += 1
                                total_return += 1.5
                            else:
                                losses += 1
                                total_return -= 1.0
                        elif (future_lows <= sl).any():
                            losses += 1
                            total_return -= 1.0
                    else:  # SHORT
                        tp = entry - (atr * 1.5)
                        sl = entry + (atr * 1.0)
                        
                        # Check if TP hit
                        if (future_lows <= tp).any():
                            tp_idx = np.where(future_lows <= tp)[0][0]
                            sl_idx = np.where(future_highs >= sl)[0]
                            
                            if len(sl_idx) == 0 or tp_idx < sl_idx[0]:
                                wins += 1
                                total_return += 1.5
                            else:
                                losses += 1
                                total_return -= 1.0
                        elif (future_highs >= sl).any():
                            losses += 1
                            total_return -= 1.0
            
            total_trades = wins + losses
            if total_trades > 0:
                win_rate = wins / total_trades * 100
                avg_return = total_return / total_trades
                avg_conf = np.mean(confidences)
                
                print(f"\n  {signal_type}:")
                print(f"    Trades: {total_trades} ({wins}W / {losses}L)")
                print(f"    Win Rate: {win_rate:.1f}%")
                print(f"    Avg R-Multiple: {avg_return:.2f}R")
                print(f"    Avg Confidence: {avg_conf:.3f}")
                
                if win_rate < 40:
                    print(f"    ❌ PROBLEM: {signal_type} win rate is terrible!")
                elif win_rate < 48:
                    print(f"    ⚠️  WARNING: {signal_type} win rate is below target")
                else:
                    print(f"    ✅ {signal_type} signals are good")
    
    print()


def diagnose_model(symbol: str, timeframe: str):
    """Full diagnostic of a model."""
    
    print(f"\n{'#'*80}")
    print(f"# DEEP DIAGNOSTIC: {symbol} {timeframe}")
    print(f"{'#'*80}\n")
    
    # Load data
    print(f"[1/6] Loading data...")
    path = DiagnosticConfig.FEATURE_STORE / symbol / f"{symbol}_{timeframe}.parquet"
    df = pd.read_parquet(path)
    
    if 'timestamp' not in df.columns:
        df = df.reset_index()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    train_start = pd.to_datetime(DiagnosticConfig.TRAIN_START, utc=True)
    train_end = pd.to_datetime(DiagnosticConfig.TRAIN_END, utc=True)
    df = df[(df['timestamp'] >= train_start) & (df['timestamp'] <= train_end)]
    
    print(f"  Loaded {len(df):,} bars\n")
    
    # Add basic features
    print(f"[2/6] Adding features...")
    for p in [5, 10, 20]:
        df[f'mom_{p}'] = df['close'].pct_change(p)
    df['vol_10'] = df['close'].pct_change().rolling(10).std()
    df['vol_20'] = df['close'].pct_change().rolling(20).std()
    
    if 'ema20' in df.columns and 'ema50' in df.columns:
        df['trend'] = ((df['ema20'] > df['ema50']).astype(int) * 2 - 1)
        atr = df.get('atr14', df['close'] * 0.02)
        df['dist_ema50'] = (df['close'] - df['ema50']) / atr
    
    if 'rsi14' in df.columns:
        df['rsi_norm'] = (df['rsi14'] - 50) / 50
    
    # Create labels
    print(f"[3/6] Creating labels...")
    n = len(df)
    horizon = 40
    atr = df.get('atr14', df['close'] * 0.02).values
    entry = df['close'].values
    
    tp_mult = 1.5
    sl_mult = 1.0
    
    tp_long = entry + (atr * tp_mult)
    sl_long = entry - (atr * sl_mult)
    tp_short = entry - (atr * tp_mult)
    sl_short = entry + (atr * sl_mult)
    
    labels = np.zeros(n, dtype=int)
    highs = df['high'].values
    lows = df['low'].values
    
    for i in range(n - horizon):
        end = min(i + 1 + horizon, n)
        future_highs = highs[i+1:end]
        future_lows = lows[i+1:end]
        
        if len(future_highs) == 0:
            continue
        
        tp_long_hits = np.where(future_highs >= tp_long[i])[0]
        sl_long_hits = np.where(future_lows <= sl_long[i])[0]
        tp_short_hits = np.where(future_lows <= tp_short[i])[0]
        sl_short_hits = np.where(future_highs >= sl_short[i])[0]
        
        if len(tp_long_hits) > 0 and (len(sl_long_hits) == 0 or tp_long_hits[0] < sl_long_hits[0]):
            labels[i] = 1
        elif len(tp_short_hits) > 0 and (len(sl_short_hits) == 0 or tp_short_hits[0] < sl_short_hits[0]):
            labels[i] = 2
    
    df['target'] = labels
    df = df.iloc[:-horizon]
    
    # Select features
    exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    features = [col for col in df.columns 
                if col not in exclude 
                and pd.api.types.is_numeric_dtype(df[col])][:30]
    
    print(f"  Using {len(features)} features\n")
    
    # Analyze labels
    print(f"[4/6] Analyzing labels...")
    analyze_labels(df, symbol, timeframe)
    
    # Analyze features
    print(f"[5/6] Analyzing features...")
    analyze_features(df, features, symbol, timeframe)
    
    # Train model
    print(f"[6/6] Training and analyzing model...")
    oos_start = pd.to_datetime(DiagnosticConfig.TRAIN_END, utc=True) - timedelta(days=DiagnosticConfig.OOS_MONTHS * 30)
    train_df = df[df['timestamp'] < oos_start].copy()
    test_df = df[df['timestamp'] >= oos_start].copy()
    
    X_train = train_df[features].fillna(0).values
    y_train = train_df['target'].values
    X_test = test_df[features].fillna(0).values
    y_test = test_df['target'].values
    
    # Train model
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    counts = np.bincount(y_train)
    weights = len(y_train) / (len(counts) * counts)
    weights[0] *= 1.5
    sample_weight = weights[y_train]
    
    model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.04,
        num_leaves=12,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=3.0,
        reg_lambda=4.0,
        min_child_samples=40,
        random_state=42,
        verbosity=-1,
        force_row_wise=True
    )
    
    model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    
    # Create wrapper for predictions
    class ModelWrapper:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler
        
        def predict_proba(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
    
    model_wrapper = ModelWrapper(model, scaler)
    
    # Analyze predictions
    analyze_model_predictions(model_wrapper, X_test, y_test, test_df, symbol, timeframe)
    
    # Analyze signals
    analyze_signal_quality(model_wrapper, X_test, test_df, symbol, timeframe)
    
    # Final diagnosis
    print(f"\n{'='*80}")
    print(f"DIAGNOSIS SUMMARY: {symbol} {timeframe}")
    print(f"{'='*80}\n")
    
    probs = model_wrapper.predict_proba(X_test)
    predictions = np.argmax(probs, axis=1)
    pred_counts = pd.Series(predictions).value_counts()
    
    issues = []
    
    # Check 1: Is model just predicting Flat?
    if pred_counts.get(0, 0) / len(predictions) > 0.90:
        issues.append("❌ Model predicts Flat >90% of time (learned to play it safe)")
    
    # Check 2: Are features predictive?
    avg_feat_corr = np.mean([abs(df[f].corr(df['close'].pct_change(10).shift(-10))) for f in features if f in df.columns])
    if avg_feat_corr < 0.015:
        issues.append("❌ Features are NOT predictive of future price")
    
    # Check 3: Overall accuracy
    overall_acc = (predictions == y_test).sum() / len(y_test) * 100
    if overall_acc < 40:
        issues.append(f"❌ Model accuracy is terrible ({overall_acc:.1f}%)")
    
    if len(issues) == 0:
        print("✅ No critical issues found")
        print("   → Problem is likely in signal filtering or risk management")
    else:
        print("CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\nRECOMMENDED FIXES:")
        if pred_counts.get(0, 0) / len(predictions) > 0.90:
            print("  1. Reduce Flat class weight (currently boosted)")
            print("  2. Increase TP/SL ratio to create more directional labels")
        
        if avg_feat_corr < 0.015:
            print("  3. Add better features (momentum, trend strength, volatility)")
            print("  4. Use longer timeframes (more signal, less noise)")
        
        if overall_acc < 40:
            print("  5. Increase labeling threshold (only label clear moves)")
            print("  6. Use ensemble of multiple models")
    
    print(f"\n{'='*80}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--tf', type=str, required=True)
    
    args = parser.parse_args()
    
    diagnose_model(args.symbol, args.tf)


if __name__ == '__main__':
    main()

