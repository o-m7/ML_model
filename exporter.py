import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import onnx
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
import torch

logger = logging.getLogger(__name__)


class ModelExporter:
    """
    Export trained models, performance reports, and deployment configs.
    """
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_path / 'models').mkdir(exist_ok=True)
        (self.output_path / 'reports').mkdir(exist_ok=True)
        (self.output_path / 'plots').mkdir(exist_ok=True)
        (self.output_path / 'configs').mkdir(exist_ok=True)
    
    def save_model(self, model, model_name: str, scaler=None):
        """Save model to pickle and optionally ONNX."""
        model_path = self.output_path / 'models' / f'{model_name}.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved {model_name} to {model_path}")
        
        # Save scaler if exists
        if scaler is not None:
            scaler_path = self.output_path / 'models' / f'{model_name}_scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Saved {model_name} scaler to {scaler_path}")
    
    def export_to_onnx(self, model, model_name: str, input_dim: int):
        """Export model to ONNX format for production deployment."""
        onnx_path = self.output_path / 'models' / f'{model_name}.onnx'
        
        try:
            if model_name == 'neural_net':
                # PyTorch to ONNX
                dummy_input = torch.randn(1, input_dim)
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}}
                )
            
            elif model_name in ['random_forest', 'lightgbm', 'xgboost', 'catboost']:
                # Sklearn-like models to ONNX
                initial_type = [('float_input', FloatTensorType([None, input_dim]))]
                onnx_model = to_onnx(model, initial_types=initial_type)
                
                with open(onnx_path, 'wb') as f:
                    f.write(onnx_model.SerializeToString())
            
            else:
                logger.warning(f"ONNX export not supported for {model_name}")
                return
            
            logger.info(f"Exported {model_name} to ONNX: {onnx_path}")
        
        except Exception as e:
            logger.error(f"Failed to export {model_name} to ONNX: {str(e)}")
    
    def generate_performance_report(self, results_df: pd.DataFrame, 
                                   deployment_summary: pd.DataFrame) -> str:
        """Generate markdown performance report."""
        report = []
        
        report.append("# XAUUSD Model Training - Performance Report\n")
        report.append(f"Generated: {pd.Timestamp.now()}\n")
        
        report.append("## Executive Summary\n")
        report.append(deployment_summary.to_markdown())
        report.append("\n")
        
        report.append("## Walk-Forward Validation Results\n")
        
        # Summary by model and split
        summary = results_df.groupby(['model', 'split']).agg({
            'profit_factor': ['mean', 'std', 'min', 'max'],
            'win_rate': ['mean', 'std'],
            'max_drawdown_pct': ['mean', 'max'],
            'sharpe': ['mean', 'std'],
            'total_trades': 'sum'
        }).round(4)
        
        report.append(summary.to_markdown())
        report.append("\n")
        
        report.append("## Model Comparison\n")
        test_results = results_df[results_df['split'] == 'test']
        model_comparison = test_results.groupby('model').agg({
            'profit_factor': 'mean',
            'win_rate': 'mean',
            'max_drawdown_pct': 'mean',
            'sharpe': 'mean',
            'r_multiple': 'mean',
            'expected_value_dollar': 'mean',
            'total_trades': 'sum'
        }).round(4)
        
        model_comparison = model_comparison.sort_values('profit_factor', ascending=False)
        report.append(model_comparison.to_markdown())
        report.append("\n")
        
        report.append("## Deployment Recommendations\n")
        
        # Find best models
        best_models = deployment_summary[deployment_summary['deployment_ready'] >= 0.5].index.tolist()
        
        if best_models:
            report.append(f"**Deployment-Ready Models**: {', '.join(best_models)}\n\n")
            report.append("These models meet all deployment criteria:\n")
            report.append("- Profit Factor ≥ 1.6\n")
            report.append("- Win Rate: 50-60%\n")
            report.append("- Max Drawdown ≤ 6%\n")
            report.append("- Sharpe ≥ 0.25\n")
            report.append("- R-multiple > 1.2\n")
        else:
            report.append("**WARNING**: No models currently meet all deployment criteria.\n")
            report.append("Review training data, features, or labeling strategy.\n")
        
        report.append("\n## Next Steps\n")
        report.append("1. Review feature importance plots\n")
        report.append("2. Examine equity curves for stability\n")
        report.append("3. Check prediction calibration\n")
        report.append("4. Test on live paper trading\n")
        report.append("5. Monitor for data drift\n")
        
        report_text = '\n'.join(report)
        
        # Save to file
        report_path = self.output_path / 'reports' / 'performance_report.md'
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Performance report saved to {report_path}")
        return report_text
    
    def plot_feature_importance(self, feature_importance: Dict[str, Dict], top_n: int = 20):
        """Plot feature importance for each model."""
        n_models = len(feature_importance)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4*n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, importances) in enumerate(feature_importance.items()):
            # Sort and get top N
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features, values = zip(*sorted_features)
            
            axes[idx].barh(range(len(features)), values)
            axes[idx].set_yticks(range(len(features)))
            axes[idx].set_yticklabels(features)
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{model_name.upper()} - Top {top_n} Features')
            axes[idx].invert_yaxis()
        
        plt.tight_layout()
        
        plot_path = self.output_path / 'plots' / 'feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {plot_path}")
    
    def plot_equity_curves(self, results_df: pd.DataFrame):
        """Plot equity curves for all models."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot by split (train/val/test)
        for split in ['train', 'val', 'test']:
            split_data = results_df[results_df['split'] == split]
            for model in split_data['model'].unique():
                model_data = split_data[split_data['model'] == model]
                cumulative_return = model_data['expected_value_dollar'].cumsum()
                axes[0].plot(cumulative_return, label=f'{model} ({split})', alpha=0.7)
        
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].set_title('Equity Curves by Model and Split')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Plot test only
        test_data = results_df[results_df['split'] == 'test']
        for model in test_data['model'].unique():
            model_data = test_data[test_data['model'] == model]
            cumulative_return = model_data['expected_value_dollar'].cumsum()
            axes[1].plot(cumulative_return, marker='o', label=model)
        
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('Cumulative Return')
        axes[1].set_title('Test Set Equity Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_path / 'plots' / 'equity_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Equity curves saved to {plot_path}")
    
    def plot_return_distribution(self, results_df: pd.DataFrame):
        """Plot return distributions."""
        test_data = results_df[results_df['split'] == 'test']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        for model in test_data['model'].unique():
            model_returns = test_data[test_data['model'] == model]['expected_value_dollar']
            axes[0].hist(model_returns, alpha=0.6, label=model, bins=20)
        
        axes[0].set_xlabel('Expected Value per Trade')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Return Distribution (Test Set)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        test_data.boxplot(column='expected_value_dollar', by='model', ax=axes[1])
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Expected Value per Trade')
        axes[1].set_title('Return Distribution by Model')
        axes[1].get_figure().suptitle('')  # Remove default title
        
        plt.tight_layout()
        
        plot_path = self.output_path / 'plots' / 'return_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Return distribution plot saved to {plot_path}")
    
    def create_deployment_config(self, best_model_name: str, feature_names: List[str],
                                config: Dict) -> Dict:
        """Create deployment configuration JSON."""
        deployment_config = {
            'model_version': config['deployment']['model_version'],
            'timestamp': pd.Timestamp.now().isoformat(),
            'model': {
                'name': best_model_name,
                'file': f'{best_model_name}.pkl',
                'onnx_file': f'{best_model_name}.onnx',
                'scaler_file': f'{best_model_name}_scaler.pkl' if best_model_name == 'neural_net' else None
            },
            'features': feature_names,
            'risk_management': config['risk_management'],
            'labeling': config['labeling'],
            'performance_targets': config['performance_targets']
        }
        
        config_path = self.output_path / 'configs' / 'deployment_config.json'
        with open(config_path, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        logger.info(f"Deployment config saved to {config_path}")
        return deployment_config