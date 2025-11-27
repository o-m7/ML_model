import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import torch

logger = logging.getLogger(__name__)


class PerformanceCalculator:
    """Calculate trading performance metrics from predictions and returns."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         returns: np.ndarray, spread_cost: float = 0.0002) -> Dict:
        """
        Calculate comprehensive trading metrics.
        
        Args:
            y_true: Actual labels {0, 1}
            y_pred: Predicted labels {0, 1}
            returns: Actual returns per trade
            spread_cost: Spread as fraction (e.g., 0.0002 = 2 pips)
        
        Returns:
            Dictionary with all performance metrics
        """
        # Apply spread cost
        returns_net = returns - spread_cost
        
        # Filter to only predicted trades
        trade_mask = y_pred == 1
        traded_returns = returns_net[trade_mask]
        traded_true = y_true[trade_mask]
        traded_pred = y_pred[trade_mask]
        
        if len(traded_returns) == 0:
            logger.warning("No trades predicted")
            return {
                'profit_factor': 0,
                'sharpe': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'r_multiple': 0,
                'expected_value_r': 0,
                'expected_value_dollar': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            }
        
        # Basic metrics
        wins = traded_returns[traded_returns > 0]
        losses = traded_returns[traded_returns <= 0]
        
        total_trades = len(traded_returns)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
        
        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-10
        profit_factor = gross_profit / gross_loss
        
        # Sharpe ratio (per trade)
        sharpe = traded_returns.mean() / traded_returns.std() if traded_returns.std() > 0 else 0
        
        # Maximum drawdown
        cumulative = np.cumsum(traded_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = drawdown.max()
        max_drawdown_pct = (max_drawdown / (1 + running_max.max())) * 100 if running_max.max() > 0 else 0
        
        # R-multiple (avg win / avg loss)
        r_multiple = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Expected value
        ev_r = win_rate * avg_win - (1 - win_rate) * avg_loss
        ev_dollar = traded_returns.mean()
        
        # Classification metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(traded_true, traded_pred)
        precision = precision_score(traded_true, traded_pred, zero_division=0)
        recall = recall_score(traded_true, traded_pred, zero_division=0)
        f1 = f1_score(traded_true, traded_pred, zero_division=0)
        
        return {
            'profit_factor': profit_factor,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'r_multiple': r_multiple,
            'expected_value_r': ev_r,
            'expected_value_dollar': ev_dollar,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': wins.max() if len(wins) > 0 else 0,
            'largest_loss': losses.min() if len(losses) > 0 else 0,
            'cumulative_return': cumulative[-1] if len(cumulative) > 0 else 0,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }


class WalkForwardValidator:
    """
    Implement walk-forward validation with expanding or rolling window.
    """
    
    def __init__(self, config: Dict):
        self.train_months = config['validation']['train_months']
        self.val_months = config['validation']['val_months']
        self.test_months = config['validation']['test_months']
        self.roll_months = config['validation']['roll_forward_months']
        self.min_trades = config['validation']['min_trades_per_fold']
        self.perf_calc = PerformanceCalculator()
        
    def create_folds(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Create walk-forward folds.
        
        Returns:
            List of (train_df, val_df, test_df) tuples
        """
        folds = []
        
        # Total months needed
        total_months_per_fold = self.train_months + self.val_months + self.test_months
        total_days = len(df)
        
        # Approximate bars per month (assuming H1 = ~22 trading days * 24 hours)
        bars_per_month = 22 * 24
        
        start_idx = 0
        fold_num = 0
        
        while True:
            train_end = start_idx + self.train_months * bars_per_month
            val_end = train_end + self.val_months * bars_per_month
            test_end = val_end + self.test_months * bars_per_month
            
            if test_end > len(df):
                break
            
            train_df = df.iloc[start_idx:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:test_end]
            
            # Check minimum trades in validation
            val_trades = (val_df['label'] != 0).sum()
            test_trades = (test_df['label'] != 0).sum()
            
            if val_trades < self.min_trades or test_trades < self.min_trades:
                logger.warning(f"Fold {fold_num}: Insufficient trades (val={val_trades}, test={test_trades})")
                start_idx += self.roll_months * bars_per_month
                fold_num += 1
                continue
            
            folds.append((train_df, val_df, test_df))
            logger.info(f"Fold {fold_num}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            
            # Roll forward
            start_idx += self.roll_months * bars_per_month
            fold_num += 1
        
        logger.info(f"Created {len(folds)} walk-forward folds")
        return folds
    
    def predict_model(self, model, model_name: str, X: np.ndarray, scaler=None) -> np.ndarray:
        """Make predictions with different model types."""
        if model_name == 'neural_net':
            if scaler is None:
                raise ValueError("Scaler required for neural_net")
            X_scaled = scaler.transform(X)
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                outputs = model(X_tensor)
                proba = torch.softmax(outputs, dim=1).numpy()[:, 1]
            return proba
        
        elif model_name == 'lightgbm':
            return model.predict(X)
        
        elif model_name == 'xgboost':
            dmatrix = xgb.DMatrix(X)
            return model.predict(dmatrix)
        
        elif model_name in ['catboost', 'random_forest']:
            return model.predict_proba(X)[:, 1]
        
        else:
            raise ValueError(f"Unknown model type: {model_name}")
    
    def validate_fold(self, model, model_name: str, fold_data: Tuple, 
                     confidence_threshold: float = 0.45, scaler=None) -> Dict:
        """
        Validate model on a single fold.
        
        Returns:
            Dictionary with train, val, test metrics
        """
        train_df, val_df, test_df = fold_data
        
        from model_trainer import ModelTrainer
        trainer = ModelTrainer({})
        
        results = {}
        
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            X, y_true, _ = trainer.prepare_data(split_df)
            
            # Get predictions
            y_pred_proba = self.predict_model(model, model_name, X, scaler)
            
            # Log prediction statistics
            logger.info(f"  {split_name.capitalize()} predictions: "
                       f"min={y_pred_proba.min():.3f}, "
                       f"max={y_pred_proba.max():.3f}, "
                       f"mean={y_pred_proba.mean():.3f}")
            
            y_pred = (y_pred_proba >= confidence_threshold).astype(int)
            
            # Log how many trades would be taken
            n_trades = y_pred.sum()
            logger.info(f"  {split_name.capitalize()}: {n_trades}/{len(y_pred)} signals above threshold {confidence_threshold}")
            
            # Get actual returns
            returns = split_df[split_df['label'] != 0]['return'].values
            
            # Calculate metrics
            metrics = self.perf_calc.calculate_metrics(y_true, y_pred, returns)
            results[split_name] = metrics
        
        return results
    
    def run_walk_forward(self, models: Dict, df: pd.DataFrame, 
                        feature_names: List[str], scalers: Dict = None) -> pd.DataFrame:
        """
        Run walk-forward validation for all models.
        
        Returns:
            DataFrame with performance metrics per model per fold
        """
        folds = self.create_folds(df)
        
        results_list = []
        
        for fold_idx, fold_data in enumerate(folds):
            logger.info(f"\n{'='*50}")
            logger.info(f"FOLD {fold_idx + 1}/{len(folds)}")
            logger.info(f"{'='*50}")
            
            for model_name, model in models.items():
                logger.info(f"Validating {model_name}...")
                
                scaler = scalers.get(model_name) if scalers else None
                fold_results = self.validate_fold(model, model_name, fold_data, scaler=scaler)
                
                # Add metadata
                for split_name, metrics in fold_results.items():
                    result_row = {
                        'fold': fold_idx,
                        'model': model_name,
                        'split': split_name,
                        **metrics
                    }
                    results_list.append(result_row)
                
                # Log validation results
                val_metrics = fold_results['val']
                logger.info(f"  Val: PF={val_metrics.get('profit_factor', 0):.2f}, "
                          f"WR={val_metrics.get('win_rate', 0):.2%}, "
                          f"DD={val_metrics.get('max_drawdown_pct', 0):.2f}%, "
                          f"Trades={val_metrics.get('total_trades', 0)}")
        
        results_df = pd.DataFrame(results_list)
        
        # Summary statistics
        summary = results_df.groupby(['model', 'split']).agg({
            'profit_factor': ['mean', 'std'],
            'win_rate': ['mean', 'std'],
            'max_drawdown_pct': ['mean', 'std'],
            'sharpe': ['mean', 'std'],
            'total_trades': 'sum'
        }).round(4)
        
        logger.info("\n" + "="*50)
        logger.info("WALK-FORWARD SUMMARY")
        logger.info("="*50)
        logger.info("\n" + str(summary))
        
        return results_df
    
    def check_deployment_ready(self, results_df: pd.DataFrame, 
                              targets: Dict) -> pd.DataFrame:
        """
        Check which models meet deployment criteria.
        
        Returns:
            DataFrame with deployment readiness flags
        """
        # Get test set results only
        test_results = results_df[results_df['split'] == 'test'].copy()
        
        # Check criteria
        test_results['pf_pass'] = test_results['profit_factor'] >= targets['profit_factor_min']
        test_results['wr_pass'] = (test_results['win_rate'] >= targets['win_rate_min']) & \
                                  (test_results['win_rate'] <= targets['win_rate_max'])
        test_results['dd_pass'] = test_results['max_drawdown_pct'] <= targets['max_drawdown_pct']
        test_results['sharpe_pass'] = test_results['sharpe'] >= targets['sharpe_min']
        test_results['r_pass'] = test_results['r_multiple'] >= targets['r_multiple_min']
        test_results['trades_pass'] = test_results['total_trades'] >= targets['min_trades_test']
        
        # Overall deployment ready
        test_results['deployment_ready'] = (
            test_results['pf_pass'] &
            test_results['wr_pass'] &
            test_results['dd_pass'] &
            test_results['sharpe_pass'] &
            test_results['r_pass'] &
            test_results['trades_pass']
        )
        
        # Aggregate by model
        deployment_summary = test_results.groupby('model').agg({
            'deployment_ready': 'mean',  # % of folds passing
            'profit_factor': 'mean',
            'win_rate': 'mean',
            'max_drawdown_pct': 'mean',
            'sharpe': 'mean',
            'r_multiple': 'mean',
            'total_trades': 'mean'
        }).round(4)
        
        deployment_summary = deployment_summary.sort_values('deployment_ready', ascending=False)
        
        logger.info("\n" + "="*50)
        logger.info("DEPLOYMENT READINESS")
        logger.info("="*50)
        logger.info("\n" + str(deployment_summary))
        
        return deployment_summary