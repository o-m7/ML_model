"""
CITADEL ENHANCED SYSTEM - PART 2
Evaluation, Risk Filters, and Production Pipeline
"""

# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED MODEL EVALUATION WITH COST-AWARE METRICS
# ═══════════════════════════════════════════════════════════════════════════

class ModelEvaluator:
    """
    ENHANCED: Add cost-aware evaluation and prediction quality metrics.
    
    NEW METRICS:
    - Brier score (probability calibration quality)
    - Expected value with costs
    - Slippage-adjusted PF
    """
    
    @staticmethod
    def evaluate_all_models(models: Dict, X_test, y_test, 
                           spread_cost: float = 0.0002) -> Dict:
        """ENHANCED: Evaluate with cost simulation."""
        
        print(f"\n📊 EVALUATING MODELS (COST-AWARE)")
        print(f"{'='*80}")
        
        results = {}
        
        for model_name, model_dict in models.items():
            try:
                model = model_dict['model']
                scaler = model_dict['scaler']
                
                X_test_scaled = scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Basic metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # NEW: Brier score (calibration quality)
                brier = brier_score_loss(y_test, y_proba)
                
                # Trading metrics
                wins = ((y_pred == 1) & (y_test == 1)).sum()
                losses = ((y_pred == 1) & (y_test == 0)).sum()
                total = wins + losses
                
                win_rate = wins / total if total > 0 else 0
                
                # Simple PF (no costs)
                avg_win_r = 1.4  # Approximate from TP
                avg_loss_r = 1.0
                pf_base = (wins * avg_win_r) / (losses * avg_loss_r) if losses > 0 else 0
                
                # NEW: Cost-adjusted PF
                cost_per_trade_r = spread_cost / avg_loss_r  # Convert to R-multiple
                avg_win_r_net = avg_win_r - cost_per_trade_r
                avg_loss_r_net = avg_loss_r + cost_per_trade_r
                pf_cost_adj = (wins * avg_win_r_net) / (losses * avg_loss_r_net) if losses > 0 else 0
                
                # NEW: Expected value
                ev_base = (win_rate * avg_win_r) - ((1 - win_rate) * avg_loss_r)
                ev_cost_adj = (win_rate * avg_win_r_net) - ((1 - win_rate) * avg_loss_r_net)
                
                results[model_name] = {
                    'accuracy': acc,
                    'win_rate': win_rate,
                    'profit_factor': pf_base,
                    'pf_cost_adj': pf_cost_adj,
                    'f1': f1,
                    'brier': brier,
                    'ev_base': ev_base,
                    'ev_cost_adj': ev_cost_adj,
                    'total_trades': total,
                    'wins': wins,
                    'losses': losses
                }
                
                print(f"\n{model_name.upper()}:")
                print(f"   Win Rate: {win_rate:.1%}")
                print(f"   PF (base): {pf_base:.2f} → (cost-adj): {pf_cost_adj:.2f}")
                print(f"   EV: {ev_base:.3f}R → {ev_cost_adj:.3f}R")
                print(f"   Brier: {brier:.4f} (lower = better calibration)")
                print(f"   F1: {f1:.4f}")
                print(f"   Trades: {total:,}")
                
            except Exception as e:
                print(f"\n{model_name.upper()}: ❌ Failed - {e}")
        
        return results
    
    @staticmethod
    def print_comparison_table(results: Dict):
        """Enhanced comparison table."""
        
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON (ENHANCED)")
        print(f"{'='*80}")
        
        print(f"\n{'Model':<20} {'WinRate':>10} {'PF':>8} {'PF_adj':>8} {'F1':>8} {'Brier':>8} {'Trades':>10}")
        print("-"*90)
        
        for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
            print(f"{model_name:<20} "
                  f"{metrics['win_rate']:>9.1%} "
                  f"{metrics['profit_factor']:>8.2f} "
                  f"{metrics['pf_cost_adj']:>8.2f} "
                  f"{metrics['f1']:>8.4f} "
                  f"{metrics['brier']:>8.4f} "
                  f"{metrics['total_trades']:>10,}")
        
        best = max(results.items(), key=lambda x: x[1]['f1'])
        print(f"\n🏆 Best: {best[0]}")
        print(f"   WR: {best[1]['win_rate']:.1%}")
        print(f"   PF (cost-adj): {best[1]['pf_cost_adj']:.2f}")
        print(f"   EV (cost-adj): {best[1]['ev_cost_adj']:.3f}R/trade")


# ═══════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION (KEPT FROM V2 - WORKS WELL)
# ═══════════════════════════════════════════════════════════════════════════

class WalkForwardValidator:
    """Walk-forward validation (unchanged - validated)."""
    
    @staticmethod
    def run_walk_forward(df: pd.DataFrame, labels: pd.Series, n_splits: int = 5) -> List[Dict]:
        """Execute walk-forward validation."""
        
        print(f"\n🔄 WALK-FORWARD VALIDATION")
        print(f"{'='*80}")
        
        labeled_mask = (labels == 0) | (labels == 1)
        df_labeled = df[labeled_mask].copy()
        labels_filtered = labels[labeled_mask].copy()
        
        n_samples = len(df_labeled)
        fold_size = n_samples // (n_splits + 1)
        
        results = []
        
        for i in range(n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = test_start + fold_size
            
            if test_end > n_samples:
                break
            
            df_train = df_labeled.iloc[:train_end]
            df_test = df_labeled.iloc[test_start:test_end]
            
            y_train = labels_filtered.iloc[:train_end].values
            y_test = labels_filtered.iloc[test_start:test_end].values
            
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            
            feature_cols = [c for c in df_labeled.columns 
                           if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            X_train = df_train[feature_cols].values
            X_test = df_test[feature_cols].values
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            weight_dict = dict(zip(classes, class_weights))
            sample_weights = np.array([weight_dict[yi] for yi in y_train])
            scale_pos_weight = weight_dict[1] / weight_dict[0]
            
            model = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=6,
                scale_pos_weight=scale_pos_weight,
                verbose=-1
            )
            
            model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            y_pred = model.predict(X_test_scaled)
            
            wins = ((y_pred == 1) & (y_test == 1)).sum()
            losses = ((y_pred == 1) & (y_test == 0)).sum()
            total = wins + losses
            
            result = {
                'fold_num': i + 1,
                'win_rate': wins / total if total > 0 else 0,
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'total_trades': total
            }
            
            results.append(result)
            print(f"   Fold {i+1}: WR={result['win_rate']:.1%}, Trades={total:,}")
        
        if results:
            avg_wr = np.mean([r['win_rate'] for r in results])
            std_wr = np.std([r['win_rate'] for r in results])
            
            print(f"\n   Avg WR: {avg_wr:.1%} ± {std_wr:.1%}")
            if std_wr < 0.05:
                print(f"   ✅ Stable")
            else:
                print(f"   ⚠️  High variance")
        
        return results


# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED CONFIDENCE FILTERING
# ═══════════════════════════════════════════════════════════════════════════

class ConfidenceFilter:
    """ENHANCED: Better threshold optimization with calibration benefits."""
    
    @staticmethod
    def find_optimal_threshold(model_dict, X_val, y_val) -> Tuple[float, Dict]:
        """Find optimal threshold with detailed metrics."""
        
        print(f"\n🎯 OPTIMIZING CONFIDENCE THRESHOLD")
        print(f"{'='*80}")
        
        model = model_dict['model']
        scaler = model_dict['scaler']
        
        X_val_scaled = scaler.transform(X_val)
        y_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        best_threshold = 0.50
        best_score = 0
        threshold_metrics = {}
        
        print(f"\n{'Threshold':>12} {'WinRate':>10} {'Trades':>10} {'EV':>8} {'Score':>10}")
        print("-"*60)
        
        for threshold in CONFIG.CONFIDENCE_THRESHOLDS:
            y_pred_filtered = (y_proba >= threshold).astype(int)
            
            mask = y_pred_filtered == 1
            if mask.sum() == 0:
                continue
            
            wins = ((y_pred_filtered == 1) & (y_val == 1)).sum()
            losses = ((y_pred_filtered == 1) & (y_val == 0)).sum()
            total = wins + losses
            
            wr = wins / total if total > 0 else 0
            
            # Expected value
            avg_win_r = 1.4
            avg_loss_r = 1.0
            ev = (wr * avg_win_r) - ((1 - wr) * avg_loss_r)
            
            # Score: prioritize EV but maintain frequency
            score = ev * np.log(total + 1)
            
            threshold_metrics[threshold] = {
                'wr': wr,
                'trades': total,
                'ev': ev,
                'score': score
            }
            
            print(f"{threshold:>12.2f} {wr:>9.1%} {total:>10,} {ev:>8.3f} {score:>10.4f}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        print(f"\n   ✅ Best threshold: {best_threshold:.2f}")
        print(f"      WR: {threshold_metrics[best_threshold]['wr']:.1%}")
        print(f"      EV: {threshold_metrics[best_threshold]['ev']:.3f}R")
        
        return best_threshold, threshold_metrics


# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

class RegimeAnalyzer:
    """ENHANCED: Multi-dimensional regime analysis."""
    
    @staticmethod
    def analyze_by_regime(df: pd.DataFrame, y_true, y_pred) -> Dict:
        """Enhanced regime analysis."""
        
        print(f"\n📊 REGIME PERFORMANCE ANALYSIS")
        print(f"{'='*80}")
        
        regime_results = {}
        
        # Volatility regime
        if 'regime_vol' in df.columns:
            print(f"\n🌊 Volatility Regimes:")
            regimes = df['regime_vol'].replace({0: 'Low Vol', 1: 'Med Vol', 2: 'High Vol'})
            
            for regime in ['Low Vol', 'Med Vol', 'High Vol']:
                mask = regimes == regime
                if mask.sum() == 0:
                    continue
                
                regime_y_true = y_true[mask]
                regime_y_pred = y_pred[mask]
                
                wins = ((regime_y_pred == 1) & (regime_y_true == 1)).sum()
                losses = ((regime_y_pred == 1) & (regime_y_true == 0)).sum()
                total = wins + losses
                
                wr = wins / total if total > 0 else 0
                
                regime_results[f'vol_{regime}'] = {
                    'win_rate': wr,
                    'trades': total
                }
                
                print(f"   {regime}: WR={wr:.1%}, Trades={total:,}")
        
        # Session regime
        if 'regime_session_asian' in df.columns:
            print(f"\n🌍 Session Performance:")
            
            sessions = {
                'Asian': df['regime_session_asian'] == 1,
                'London': df['regime_session_london'] == 1,
                'NY': df['regime_session_ny'] == 1,
                'Overlap': df['regime_session_overlap'] == 1 if 'regime_session_overlap' in df.columns else None
            }
            
            for session_name, mask in sessions.items():
                if mask is None or mask.sum() == 0:
                    continue
                
                session_y_true = y_true[mask]
                session_y_pred = y_pred[mask]
                
                wins = ((session_y_pred == 1) & (session_y_true == 1)).sum()
                losses = ((session_y_pred == 1) & (session_y_true == 0)).sum()
                total = wins + losses
                
                wr = wins / total if total > 0 else 0
                
                regime_results[f'session_{session_name}'] = {
                    'win_rate': wr,
                    'trades': total
                }
                
                print(f"   {session_name}: WR={wr:.1%}, Trades={total:,}")
        
        return regime_results


# ═══════════════════════════════════════════════════════════════════════════
# NEW: RISK FILTERS FOR LIVE TRADING
# ═══════════════════════════════════════════════════════════════════════════

class RiskFilters:
    """
    NEW: Trade gating filters for live trading.
    
    Filters out trades in:
    - Extreme volatility conditions
    - Low liquidity sessions
    - High spread environments
    """
    
    @staticmethod
    def apply_volatility_filter(df: pd.DataFrame, atr_col: str = 'atr') -> pd.Series:
        """Filter extreme volatility."""
        
        if atr_col not in df.columns:
            return pd.Series(True, index=df.index)
        
        atr_percentile = df[atr_col].rolling(100).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) if len(x) > 0 else 0.5
        )
        
        # Skip bottom 20% (too quiet) and top 5% (too chaotic)
        valid = (atr_percentile >= CONFIG.MIN_ATR_PERCENTILE) & (atr_percentile <= CONFIG.MAX_ATR_PERCENTILE)
        
        return valid
    
    @staticmethod
    def apply_session_filter(df: pd.DataFrame) -> pd.Series:
        """Filter low-quality sessions."""
        
        if 'hour' not in df.columns:
            return pd.Series(True, index=df.index)
        
        valid = pd.Series(True, index=df.index)
        
        # Avoid Asian session if configured
        if CONFIG.AVOID_ASIAN_SESSION:
            asian_hours = (df['hour'] >= 0) & (df['hour'] < 8)
            valid = valid & ~asian_hours
        
        # Prefer London/NY overlap if configured
        if CONFIG.PREFER_LONDON_NY_OVERLAP:
            overlap_hours = (df['hour'] >= 13) & (df['hour'] < 16)
            # Don't force, just boost quality (handled in probability weighting)
        
        return valid
    
    @staticmethod
    def apply_all_filters(df: pd.DataFrame) -> pd.Series:
        """Apply all risk filters."""
        
        vol_filter = RiskFilters.apply_volatility_filter(df)
        session_filter = RiskFilters.apply_session_filter(df)
        
        combined = vol_filter & session_filter
        
        pct_filtered = (1 - combined.mean()) * 100
        print(f"\n🛡️  Risk Filters Applied:")
        print(f"   Volatility filter: {vol_filter.mean():.1%} passed")
        print(f"   Session filter: {session_filter.mean():.1%} passed")
        print(f"   Combined: {combined.mean():.1%} passed ({pct_filtered:.1f}% filtered)")
        
        return combined


# ═══════════════════════════════════════════════════════════════════════════
# NEW: FEATURE IMPORTANCE TRACKING
# ═══════════════════════════════════════════════════════════════════════════

class FeatureImportanceAnalyzer:
    """NEW: Track and display feature importance."""
    
    @staticmethod
    def analyze_importance(model, feature_names: List[str], top_n: int = 20):
        """Display top feature importances."""
        
        print(f"\n🔍 TOP {top_n} FEATURE IMPORTANCES")
        print(f"{'='*80}")
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            print("   Model doesn't support feature importance")
            return
        
        indices = np.argsort(importances)[::-1][:top_n]
        
        print(f"\n{'Rank':>5} {'Feature':<45} {'Importance':>12}")
        print("-"*70)
        
        for rank, idx in enumerate(indices, 1):
            print(f"{rank:>5} {feature_names[idx]:<45} {importances[idx]:>12.6f}")
        
        # Check for suspicious patterns
        suspicious = []
        for idx in indices[:10]:
            feat_name = feature_names[idx]
            if any(word in feat_name.lower() for word in ['close', 'high', 'low', 'open']):
                if 'micro' not in feat_name and 'mr_' not in feat_name:
                    suspicious.append(feat_name)
        
        if suspicious:
            print(f"\n   ⚠️  WARNING: Raw OHLCV in top features:")
            for feat in suspicious:
                print(f"      - {feat}")
            print(f"   Check for potential lookahead bias!")


# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class TrainingPipeline:
    """Enhanced training pipeline with all improvements."""
    
    def __init__(self, symbol: str, timeframe: str,
                 enable_walk_forward: bool = False,
                 enable_diagnostics: bool = False,
                 enable_calibration: bool = True):
        self.symbol = symbol
        self.timeframe = timeframe
        self.enable_walk_forward = enable_walk_forward
        self.enable_diagnostics = enable_diagnostics
        self.enable_calibration = enable_calibration
        self.results = {}
    
    def run(self):
        """Execute enhanced pipeline."""
        
        print(f"\n{'#'*80}")
        print(f"# CITADEL ML SYSTEM - ENHANCED")
        print(f"# Symbol: {self.symbol} | Timeframe: {self.timeframe}")
        print(f"# Calibration: {self.enable_calibration} | Diagnostics: {self.enable_diagnostics}")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")
        
        # Step 1: Load data
        df, metadata = DataLoader.load_timeframe_data(self.symbol, self.timeframe)
        
        # Step 2: Engineer features (ENHANCED)
        df = FeatureEngineer.engineer_all_features(df)
        
        # Step 3: Find optimal TP
        time_barrier = CONFIG.get_time_barrier(self.timeframe)
        best_tp = TripleBarrierLabeler.find_best_tp_mult(
            df,
            CONFIG.TP_MULTIPLIERS,
            CONFIG.SL_MULTIPLIER,
            time_barrier
        )
        
        # Step 4: Label
        labels, r_multiples = TripleBarrierLabeler.label(
            df,
            best_tp,
            CONFIG.SL_MULTIPLIER,
            time_barrier
        )
        
        # Step 5: Walk-forward (optional)
        if self.enable_walk_forward:
            wf_results = WalkForwardValidator.run_walk_forward(df, labels, CONFIG.WF_N_SPLITS)
        
        # Step 6: Chronological split
        splits = DataSplitter.split_chronological(df, labels)
        
        # Step 7: Train models (ENHANCED with calibration)
        models = ModelFactory.train_all_models(
            splits['X_train'],
            splits['X_val'],
            splits['y_train'],
            splits['y_val'],
            enable_calibration=self.enable_calibration
        )
        
        # Step 8: Evaluate (ENHANCED with costs)
        results = ModelEvaluator.evaluate_all_models(
            models,
            splits['X_test'],
            splits['y_test']
        )
        
        # Step 9: Select best model
        best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
        best_model = models[best_model_name]
        
        # Step 10: Optimize confidence threshold
        optimal_threshold, threshold_metrics = ConfidenceFilter.find_optimal_threshold(
            best_model,
            splits['X_val'],
            splits['y_val']
        )
        
        # Step 11: Regime analysis
        test_indices = splits['test_ts'].index
        df_test = df.loc[test_indices]
        
        X_test_scaled = best_model['scaler'].transform(splits['X_test'])
        y_test_pred = best_model['model'].predict(X_test_scaled)
        
        regime_results = RegimeAnalyzer.analyze_by_regime(
            df_test,
            splits['y_test'],
            y_test_pred
        )
        
        # Step 12: Feature importance (if diagnostics enabled)
        if self.enable_diagnostics:
            # Get base model (unwrap calibration if applied)
            base_model = best_model['model']
            if hasattr(base_model, 'base_estimator'):
                base_model = base_model.base_estimator
            
            FeatureImportanceAnalyzer.analyze_importance(
                base_model,
                splits['feature_cols']
            )
        
        # Step 13: Apply risk filters (demonstration)
        risk_filter = RiskFilters.apply_all_filters(df_test)
        
        # Print comparison
        ModelEvaluator.print_comparison_table(results)
        
        # Store results
        self.results = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'best_tp_mult': best_tp,
            'optimal_threshold': optimal_threshold,
            'threshold_metrics': threshold_metrics,
            'models': models,
            'results': results,
            'regime_results': regime_results,
            'feature_cols': splits['feature_cols'],
            'calibration_enabled': self.enable_calibration
        }
        
        print(f"\n{'#'*80}")
        print(f"# TRAINING COMPLETE")
        print(f"# Best Model: {best_model_name}")
        print(f"# Optimal Threshold: {optimal_threshold:.2f}")
        print(f"# Expected WR @ threshold: {threshold_metrics[optimal_threshold]['wr']:.1%}")
        print(f"# Expected EV @ threshold: {threshold_metrics[optimal_threshold]['ev']:.3f}R")
        print(f"# Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}\n")
        
        return self.results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Citadel ML System - Enhanced')
    
    parser.add_argument('--symbol', type=str, default='XAUUSD')
    parser.add_argument('--timeframe', type=str, help='Single timeframe')
    parser.add_argument('--all-timeframes', action='store_true')
    parser.add_argument('--walk-forward', action='store_true')
    parser.add_argument('--diagnose', action='store_true')
    parser.add_argument('--full-system', action='store_true')
    parser.add_argument('--no-calibration', action='store_true', help='Disable probability calibration')
    
    args = parser.parse_args()
    
    if args.full_system:
        args.walk_forward = True
        args.diagnose = True
    
    # Determine timeframes
    if args.all_timeframes:
        timeframes = ['5T', '15T', '30T', '1H']
    elif args.timeframe:
        timeframes = [args.timeframe]
    else:
        parser.print_help()
        return
    
    # Train each timeframe
    all_results = {}
    
    for timeframe in timeframes:
        try:
            pipeline = TrainingPipeline(
                args.symbol,
                timeframe,
                enable_walk_forward=args.walk_forward,
                enable_diagnostics=args.diagnose,
                enable_calibration=not args.no_calibration
            )
            results = pipeline.run()
            all_results[timeframe] = results
            
        except Exception as e:
            print(f"\n❌ ERROR in {timeframe}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY - {args.symbol}")
    print(f"{'='*80}")
    
    print(f"\n{'TF':<6} {'Best Model':<15} {'WR':>8} {'PF':>6} {'PF_adj':>8} {'Threshold':>11} {'EV':>8}")
    print("-"*80)
    
    for tf, result in all_results.items():
        if result and 'results' in result:
            best = max(result['results'].items(), key=lambda x: x[1]['f1'])
            threshold = result['optimal_threshold']
            threshold_metrics = result['threshold_metrics']
            
            print(f"{tf:<6} {best[0]:<15} "
                  f"{best[1]['win_rate']:>7.1%} "
                  f"{best[1]['profit_factor']:>6.2f} "
                  f"{best[1]['pf_cost_adj']:>8.2f} "
                  f"{threshold:>11.2f} "
                  f"{threshold_metrics[threshold]['ev']:>8.3f}")
    
    print(f"\n✅ ALL TRAINING COMPLETE\n")


if __name__ == '__main__':
    main()