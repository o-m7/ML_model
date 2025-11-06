#!/usr/bin/env python3
"""
FastAPI server for serving predictions to web app
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Optional
from supabase import create_client
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to import onnxruntime (may not be available on Python 3.14)
try:
    import onnxruntime as rt
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False
    print("⚠️  onnxruntime not available - predictions will fail")

app = FastAPI(
    title="ML Trading API",
    description="Renaissance Technologies ML Trading System API",
    version="1.0.0"
)

# CORS for web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Lovable domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Connected to Supabase")
else:
    supabase = None
    print("⚠️  Supabase not configured - running in local mode only")

# Load all ONNX models at startup
models_cache = {}


def load_onnx_model(symbol: str, timeframe: str):
    """Load ONNX model and metadata."""
    if not ONNX_RUNTIME_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="onnxruntime not available. Install with: pip install onnxruntime"
        )
    
    key = f"{symbol}_{timeframe}"
    
    if key in models_cache:
        return models_cache[key]
    
    onnx_path = Path(f"models_onnx/{symbol}/{symbol}_{timeframe}.onnx")
    metadata_path = onnx_path.with_suffix('.json')
    
    if not onnx_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {key}")
    
    sess = rt.InferenceSession(str(onnx_path))
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    models_cache[key] = {
        'session': sess,
        'metadata': metadata,
        'input_name': sess.get_inputs()[0].name
    }
    
    print(f"✅ Loaded model: {key}")
    
    return models_cache[key]


class PredictionRequest(BaseModel):
    symbol: str
    timeframe: str
    features: List[float]


class PredictionResponse(BaseModel):
    symbol: str
    timeframe: str
    signal: str  # 'long', 'short', 'flat' - what model predicts
    directional_signal: str  # 'long' or 'short' - always directional
    confidence: float
    probabilities: Dict[str, float]
    edge: float
    should_trade: bool
    signal_quality: str  # 'high', 'medium', 'low'
    backtest_metrics: Optional[Dict] = None


class ModelInfo(BaseModel):
    symbol: str
    timeframe: str
    num_features: int
    features: List[str]
    backtest_results: Dict
    parameters: Dict


@app.get("/")
def root():
    """API health check."""
    return {
        "status": "online",
        "service": "ML Trading API",
        "models_loaded": len(models_cache),
        "supabase_connected": supabase is not None
    }


@app.get("/health")
def health_check():
    """Detailed health check."""
    onnx_dir = Path('models_onnx')
    available_models = []
    
    if onnx_dir.exists():
        for symbol_dir in onnx_dir.iterdir():
            if symbol_dir.is_dir():
                for onnx_file in symbol_dir.glob('*.onnx'):
                    symbol = symbol_dir.name
                    timeframe = onnx_file.stem.split('_')[1]
                    available_models.append(f"{symbol}_{timeframe}")
    
    return {
        "status": "healthy",
        "models_available": len(available_models),
        "models_cached": len(models_cache),
        "available_models": available_models,
        "supabase": "connected" if supabase else "not configured"
    }


@app.get("/models", response_model=List[str])
def list_models():
    """List all available models."""
    onnx_dir = Path('models_onnx')
    models = []
    
    if onnx_dir.exists():
        for symbol_dir in onnx_dir.iterdir():
            if symbol_dir.is_dir():
                for onnx_file in symbol_dir.glob('*.onnx'):
                    symbol = symbol_dir.name
                    timeframe = onnx_file.stem.split('_')[1]
                    models.append(f"{symbol}_{timeframe}")
    
    return sorted(models)


@app.get("/models/{symbol}/{timeframe}", response_model=ModelInfo)
def get_model_info(symbol: str, timeframe: str):
    """Get detailed model metadata."""
    model = load_onnx_model(symbol, timeframe)
    metadata = model['metadata']
    
    return ModelInfo(
        symbol=metadata['symbol'],
        timeframe=metadata['timeframe'],
        num_features=metadata['num_features'],
        features=metadata['features'],
        backtest_results=metadata['backtest_results'],
        parameters=metadata['params']
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """Get prediction from model."""
    
    # Load model
    model = load_onnx_model(req.symbol, req.timeframe)
    sess = model['session']
    metadata = model['metadata']
    input_name = model['input_name']
    
    # Validate features
    if len(req.features) != metadata['num_features']:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {metadata['num_features']} features, got {len(req.features)}"
        )
    
    # Prepare input
    X = np.array(req.features).reshape(1, -1).astype(np.float32)
    
    # Get prediction
    pred = sess.run(None, {input_name: X})
    probabilities = pred[1][0]  # [flat, long, short]
    
    # Parse probabilities
    probs_dict = {
        'flat': float(probabilities[0]),
        'long': float(probabilities[1]),
        'short': float(probabilities[2])
    }
    
    # Determine signal (what model actually predicts)
    max_prob = max(probs_dict.values())
    signal = max(probs_dict, key=probs_dict.get)
    
    # ALWAYS provide a directional signal (ignore flat)
    directional_signal = 'long' if probs_dict['long'] > probs_dict['short'] else 'short'
    directional_confidence = max(probs_dict['long'], probs_dict['short'])
    
    # Calculate edge (difference between top 2 probabilities)
    sorted_probs = sorted(probs_dict.values(), reverse=True)
    edge = sorted_probs[0] - sorted_probs[1]
    
    # Check if should trade (strict criteria)
    params = metadata['params']
    should_trade = (
        signal != 'flat' and
        max_prob >= params['min_conf'] and
        edge >= params['min_edge']
    )
    
    # Determine signal quality
    if should_trade:
        signal_quality = 'high'
    elif directional_confidence >= 0.40 and edge >= 0.05:
        signal_quality = 'medium'
    else:
        signal_quality = 'low'
    
    return PredictionResponse(
        symbol=req.symbol,
        timeframe=req.timeframe,
        signal=signal,
        directional_signal=directional_signal,
        confidence=max_prob,
        probabilities=probs_dict,
        edge=edge,
        should_trade=should_trade,
        signal_quality=signal_quality,
        backtest_metrics={
            'win_rate': metadata['backtest_results']['win_rate'],
            'profit_factor': metadata['backtest_results']['profit_factor'],
            'sharpe_ratio': metadata['backtest_results']['sharpe_ratio'],
            'max_drawdown': metadata['backtest_results']['max_drawdown_pct']
        }
    )


@app.get("/signals/active")
def get_active_signals():
    """Get all active signals from Supabase."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    
    try:
        response = supabase.table('live_signals').select('*').eq('status', 'active').execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trades/recent")
def get_recent_trades(limit: int = 50):
    """Get recent trades from Supabase."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    
    try:
        response = supabase.table('trades').select('*').order('entry_time', desc=True).limit(limit).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/summary")
def get_performance_summary():
    """Get overall performance summary."""
    onnx_dir = Path('models_onnx')
    summary = {
        'total_models': 0,
        'symbols': {},
        'timeframes': {},
        'aggregate_metrics': {
            'avg_win_rate': 0,
            'avg_profit_factor': 0,
            'avg_sharpe': 0,
            'avg_max_drawdown': 0
        }
    }
    
    if not onnx_dir.exists():
        return summary
    
    all_metrics = []
    
    for symbol_dir in onnx_dir.iterdir():
        if not symbol_dir.is_dir():
            continue
        
        symbol = symbol_dir.name
        summary['symbols'][symbol] = {'count': 0, 'timeframes': []}
        
        for metadata_file in symbol_dir.glob('*.json'):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            tf = metadata['timeframe']
            results = metadata['backtest_results']
            
            summary['total_models'] += 1
            summary['symbols'][symbol]['count'] += 1
            summary['symbols'][symbol]['timeframes'].append(tf)
            
            if tf not in summary['timeframes']:
                summary['timeframes'][tf] = 0
            summary['timeframes'][tf] += 1
            
            all_metrics.append(results)
    
    # Calculate aggregate metrics
    if all_metrics:
        summary['aggregate_metrics'] = {
            'avg_win_rate': sum(m['win_rate'] for m in all_metrics) / len(all_metrics),
            'avg_profit_factor': sum(m['profit_factor'] for m in all_metrics) / len(all_metrics),
            'avg_sharpe': sum(m['sharpe_ratio'] for m in all_metrics) / len(all_metrics),
            'avg_max_drawdown': sum(m['max_drawdown_pct'] for m in all_metrics) / len(all_metrics)
        }
    
    return summary


if __name__ == '__main__':
    import uvicorn
    print("\n" + "="*80)
    print("Starting ML Trading API Server")
    print("="*80 + "\n")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

