import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class NeuralNetClassifier(nn.Module):
    """Simple feedforward neural network for classification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], 
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ModelTrainer:
    """
    Train multiple model architectures with hyperparameter optimization.
    Supports LightGBM, XGBoost, CatBoost, Random Forest, Neural Network.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.random_seed = config.get('random_seed', 42)
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract features and labels from dataframe.
        Converts multi-class labels {-1, 0, 1} to binary {0, 1} by filtering out timeouts.
        """
        exclude_cols = ['label', 'return', 'bars_held', 'barrier_hit']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Filter out timeouts (label=0), keep only wins and losses
        df_trades = df[df['label'] != 0].copy()
        
        # Convert {-1, 1} to {0, 1}
        y = (df_trades['label'].values == 1).astype(int)
        X = df_trades[feature_cols].values
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class balance: {np.bincount(y)}")
        
        return X, y, feature_cols
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       params: Dict = None) -> lgb.Booster:
        """Train LightGBM model."""
        if params is None:
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': self.random_seed,
                'verbose': -1
            }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )
        
        return model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      params: Dict = None) -> xgb.Booster:
        """Train XGBoost model."""
        if params is None:
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': self.random_seed
            }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return model
    
    def train_catboost(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       params: Dict = None) -> cb.CatBoost:
        """Train CatBoost model."""
        if params is None:
            params = {
                'loss_function': 'Logloss',
                'eval_metric': 'Logloss',
                'depth': 6,
                'learning_rate': 0.05,
                'l2_leaf_reg': 3,
                'random_seed': self.random_seed,
                'verbose': 100
            }
        
        train_pool = cb.Pool(X_train, y_train)
        val_pool = cb.Pool(X_val, y_val)
        
        model = cb.CatBoostClassifier(**params, iterations=1000, early_stopping_rounds=50)
        model.fit(train_pool, eval_set=val_pool)
        
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           params: Dict = None) -> RandomForestClassifier:
        """Train Random Forest model."""
        if params is None:
            params = {
                'n_estimators': 500,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 'sqrt',
                'random_state': self.random_seed,
                'n_jobs': -1
            }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def train_neural_net(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        params: Dict = None) -> Tuple[NeuralNetClassifier, StandardScaler]:
        """Train PyTorch neural network."""
        if params is None:
            params = {
                'hidden_dims': [128, 64, 32],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 512,
                'epochs': 100,
                'patience': 10
            }
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NeuralNetClassifier(
            input_dim=X_train.shape[1],
            hidden_dims=params['hidden_dims'],
            dropout=params['dropout']
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Training loop with early stopping
        best_val_loss = np.inf
        patience_counter = 0
        
        for epoch in range(params['epochs']):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= params['patience']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best weights
        model.load_state_dict(best_state)
        
        return model, scaler
    
    def optimize_hyperparameters(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Use Optuna to optimize hyperparameters with custom objective.
        Objective: weighted sum of PF, Sharpe, and (1-DD)
        """
        def objective(trial):
            if model_name == 'lightgbm':
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                    'random_state': self.random_seed,
                    'verbose': -1
                }
                
                model = self.train_lightgbm(X_train, y_train, X_val, y_val, params)
                y_pred_proba = model.predict(X_val)
            
            elif model_name == 'xgboost':
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                    'random_state': self.random_seed
                }
                
                model = self.train_xgboost(X_train, y_train, X_val, y_val, params)
                dval = xgb.DMatrix(X_val)
                y_pred_proba = model.predict(dval)
            
            else:
                raise ValueError(f"Hyperparameter optimization not implemented for {model_name}")
            
            # Compute custom objective (requires actual returns for PF/Sharpe)
            # For now, use AUC as proxy
            from sklearn.metrics import roc_auc_score
            try:
                score = roc_auc_score(y_val, y_pred_proba)
            except:
                score = 0.5
            
            return score
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=self.random_seed))
        study.optimize(objective, n_trials=self.config['hyperparameter_tuning']['n_trials'], 
                      show_progress_bar=True)
        
        logger.info(f"Best {model_name} params: {study.best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")
        
        return study.best_params
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        feature_names: List[str],
                        optimize_hp: bool = False) -> Dict:
        """Train all configured models."""
        models = {}
        
        model_list = self.config['models']['train']
        
        for model_name in model_list:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name.upper()}")
            logger.info(f"{'='*50}")
            
            try:
                if optimize_hp and model_name in ['lightgbm', 'xgboost']:
                    best_params = self.optimize_hyperparameters(model_name, X_train, y_train, X_val, y_val)
                else:
                    best_params = None
                
                if model_name == 'lightgbm':
                    model = self.train_lightgbm(X_train, y_train, X_val, y_val, best_params)
                    self.feature_importance[model_name] = dict(zip(feature_names, model.feature_importance()))
                
                elif model_name == 'xgboost':
                    model = self.train_xgboost(X_train, y_train, X_val, y_val, best_params)
                    self.feature_importance[model_name] = model.get_score(importance_type='gain')
                
                elif model_name == 'catboost':
                    model = self.train_catboost(X_train, y_train, X_val, y_val)
                    self.feature_importance[model_name] = dict(zip(feature_names, model.get_feature_importance()))
                
                elif model_name == 'random_forest':
                    model = self.train_random_forest(X_train, y_train)
                    self.feature_importance[model_name] = dict(zip(feature_names, model.feature_importances_))
                
                elif model_name == 'neural_net':
                    model, scaler = self.train_neural_net(X_train, y_train, X_val, y_val)
                    self.scalers[model_name] = scaler
                
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                models[model_name] = model
                logger.info(f"{model_name} training complete")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
        
        return models