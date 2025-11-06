#!/usr/bin/env python3
"""
Production Signal Generation API for WebApp Integration
FastAPI endpoint that loads models and generates signals on demand.

Usage:
    uvicorn signal_api:app --host 0.0.0.0 --port 8000 --reload
"""

import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==================== MODELS ====================

class SignalRequest(BaseModel):
    symbol: str
    timeframe: str
    current_data: Dict[str, float]  # Latest OHLCV data

class SignalResponse(BaseModel):
    symbol: str
    timeframe: str
    signal: str  # "BUY", "SELL", "NEUTRAL"
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    expected_return_r: float
    timestamp: str
    latency_ms: float

class ModelInfo(BaseModel):
    symbol: str
    timeframe: str
    avg_win_rate: float
    avg_profit_factor: float
    avg_max_dd: float
    avg_sharpe: float
    trained_at: str
    model_path: str

# ==================== APP SETUP ====================

app = FastAPI(
    title="ML Trading Signal API",
    description="Production-grade trading signal generation",
    version="1.0.0"
)

# CORS middleware for webapp
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODEL CACHE ====================

class ModelCache:
    """Cache loaded models for fast inference."""
    
    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        # Auto-detect path
        if Path("models").exists():
            self.model_dir = Path("models")
        else:
            self.model_dir = Path("ML_Trading/models")
    
    def get_model_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}_{timeframe}"
    
    def load_model(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Load the latest model for symbol/timeframe."""
        key = self.get_model_key(symbol, timeframe)
        
        # Check cache
        if key in self.cache:
            return self.cache[key]
        
        # Find latest model file
        pattern = f"{symbol}_{timeframe}_*.pkl"
        model_files = sorted(self.model_dir.glob(pattern))
        
        if not model_files:
            return None
        
        latest_model_file = model_files[-1]
        
        try:
            with open(latest_model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.cache[key] = model_data
            return model_data
        except Exception as e:
            print(f"Error loading model {latest_model_file}: {e}")
            return None
    
    def list_available_models(self) -> List[ModelInfo]:
        """List all available trained models."""
        models = []
        
        for model_file in self.model_dir.glob("*.pkl"):
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                models.append(ModelInfo(
                    symbol=model_data['symbol'],
                    timeframe=model_data['timeframe'],
                    avg_win_rate=model_data['metrics']['avg_win_rate'],
                    avg_profit_factor=model_data['metrics']['avg_profit_factor'],
                    avg_max_dd=model_data['metrics']['avg_max_dd'],
                    avg_sharpe=model_data['metrics']['avg_sharpe'],
                    trained_at=model_data['trained_at'],
                    model_path=str(model_file)
                ))
            except Exception as e:
                print(f"Error reading {model_file}: {e}")
                continue
        
        return models

model_cache = ModelCache()

# ==================== FEATURE ENGINEERING ====================

class FeatureBuilder:
    """Build features from current market data."""
    
    @staticmethod
    def build_features(data: Dict[str, float], feature_cols: List[str]) -> np.ndarray:
        """
        Build feature vector from current data.
        Note: In production, you'd need historical data to compute indicators.
        This is a simplified version - you should pass recent bars, not just current.
        """
        # This is placeholder - in reality, you need recent historical data
        # to compute moving averages, RSI, etc.
        features = {}
        
        # Price-based features (simplified)
        close = data.get('close', 0)
        open_price = data.get('open', close)
        high = data.get('high', close)
        low = data.get('low', close)
        
        features['returns'] = (close - open_price) / open_price if open_price != 0 else 0
        features['high_low_ratio'] = (high - low) / close if close != 0 else 0
        features['close_open_ratio'] = (close - open_price) / open_price if open_price != 0 else 0
        
        # For indicators, you'd compute from historical data
        # This is simplified - extract from your feature_cols
        feature_vector = []
        for col in feature_cols:
            if col in features:
                feature_vector.append(features[col])
            else:
                # Use data if available, otherwise 0
                feature_vector.append(data.get(col, 0))
        
        return np.array([feature_vector])

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "ML Trading Signal API",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available trained models."""
    return model_cache.list_available_models()

@app.post("/signal", response_model=SignalResponse)
async def generate_signal(request: SignalRequest):
    """
    Generate trading signal for given symbol/timeframe.
    
    NOTE: This simplified version uses current bar data only.
    In production, you should pass recent historical bars to compute proper indicators.
    """
    start_time = time.time()
    
    # Load model
    model_data = model_cache.load_model(request.symbol, request.timeframe)
    
    if model_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No model found for {request.symbol} @ {request.timeframe}"
        )
    
    try:
        # Extract model components
        model = model_data['model']
        threshold = model_data['threshold']
        feature_cols = model_data['feature_cols']
        risk_pct = model_data['risk_pct']
        
        # Build features (simplified - in production, compute from historical data)
        X = FeatureBuilder.build_features(request.current_data, feature_cols)
        
        # Generate prediction
        proba = model.predict_proba(X)[0, 1]  # Probability of win
        confidence = proba * 100
        
        # Determine signal
        if proba >= threshold:
            signal = "BUY"
        else:
            signal = "NEUTRAL"
        
        # Calculate position sizing
        entry_price = request.current_data.get('close', 0)
        atr = request.current_data.get('atr', entry_price * 0.01)  # Fallback to 1% if no ATR
        
        sl_distance = atr * 1.0  # 1R
        tp_distance = atr * 1.5  # 1.5R
        
        stop_loss = entry_price - sl_distance
        take_profit = entry_price + tp_distance
        
        # Position sizing (assuming $100k account)
        account_equity = 100000
        risk_amount = account_equity * (risk_pct / 100)
        position_size = risk_amount / sl_distance if sl_distance > 0 else 0
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        return SignalResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            signal=signal,
            confidence=round(confidence, 2),
            entry_price=round(entry_price, 5),
            stop_loss=round(stop_loss, 5),
            take_profit=round(take_profit, 5),
            position_size=round(position_size, 4),
            risk_amount=round(risk_amount, 2),
            expected_return_r=1.5 if signal == "BUY" else 0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=round(latency_ms, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating signal: {str(e)}"
        )

@app.post("/batch_signals")
async def generate_batch_signals(symbols: List[str] = Query(...), 
                                 timeframe: str = Query(...)):
    """Generate signals for multiple symbols at once."""
    signals = []
    
    for symbol in symbols:
        try:
            # You'd need to fetch current data for each symbol
            # This is placeholder
            current_data = {
                'open': 2000.0,
                'high': 2010.0,
                'low': 1995.0,
                'close': 2005.0,
                'atr': 15.0
            }
            
            request = SignalRequest(
                symbol=symbol,
                timeframe=timeframe,
                current_data=current_data
            )
            
            signal = await generate_signal(request)
            signals.append(signal)
            
        except Exception as e:
            print(f"Error generating signal for {symbol}: {e}")
            continue
    
    return {"signals": signals}

@app.post("/reload_models")
async def reload_models():
    """Clear model cache and reload from disk."""
    model_cache.cache.clear()
    return {
        "status": "success",
        "message": "Model cache cleared. Models will be reloaded on next request."
    }

@app.get("/model_info/{symbol}/{timeframe}")
async def get_model_info(symbol: str, timeframe: str):
    """Get detailed information about a specific model."""
    model_data = model_cache.load_model(symbol, timeframe)
    
    if model_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"No model found for {symbol} @ {timeframe}"
        )
    
    return {
        "symbol": model_data['symbol'],
        "timeframe": model_data['timeframe'],
        "threshold": model_data['threshold'],
        "num_features": len(model_data['feature_cols']),
        "metrics": model_data['metrics'],
        "trained_at": model_data['trained_at'],
        "fold_results": model_data['fold_results']
    }

# ==================== RUN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
