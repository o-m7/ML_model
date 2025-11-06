"""Model registry for managing trained models."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone


class ModelRegistry:
    """Registry for managing trained models and their metadata."""
    
    def __init__(self, models_dir: str = "models_intraday"):
        """Initialize registry."""
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.models_dir / "manifest.json"
    
    def register_model(
        self,
        symbol: str,
        timeframe: str,
        strategy: str,
        model_path: Path,
        model_card: dict,
        status: str = "READY"
    ):
        """
        Register a trained model.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            strategy: Strategy name
            model_path: Path to saved model
            model_card: Model metadata
            status: READY or FAILED
        """
        # Create symbol/timeframe directory
        model_dir = self.models_dir / symbol / timeframe
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model card
        card_path = model_dir / "model_card.json"
        with open(card_path, 'w') as f:
            json.dump(model_card, f, indent=2)
        
        # Update manifest
        self._update_manifest(symbol, timeframe, strategy, status, model_card)
    
    def _update_manifest(
        self,
        symbol: str,
        timeframe: str,
        strategy: str,
        status: str,
        model_card: dict
    ):
        """Update the global manifest."""
        # Load existing manifest
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {
                'created_at': datetime.now(timezone.utc).isoformat(),
                'models': []
            }
        
        # Add/update entry
        entry = {
            'symbol': symbol,
            'timeframe': timeframe,
            'strategy': strategy,
            'status': status,
            'model_path': str(self.models_dir / symbol / timeframe / "model.pkl"),
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'oos_metrics': model_card.get('oos_metrics', {}),
            'benchmarks_passed': status == 'READY'
        }
        
        # Remove old entry if exists
        manifest['models'] = [
            m for m in manifest['models']
            if not (m['symbol'] == symbol and m['timeframe'] == timeframe)
        ]
        
        manifest['models'].append(entry)
        manifest['updated_at'] = datetime.now(timezone.utc).isoformat()
        
        # Calculate summary
        total = len(manifest['models'])
        passed = sum(1 for m in manifest['models'] if m['status'] == 'READY')
        manifest['summary'] = {
            'total_models': total,
            'ready_models': passed,
            'failed_models': total - passed
        }
        
        # Save
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def get_model_path(self, symbol: str, timeframe: str) -> Optional[Path]:
        """Get path to a trained model."""
        model_path = self.models_dir / symbol / timeframe / "model.pkl"
        return model_path if model_path.exists() else None
    
    def list_ready_models(self) -> List[Dict]:
        """List all production-ready models."""
        if not self.manifest_path.exists():
            return []
        
        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)
        
        return [m for m in manifest['models'] if m['status'] == 'READY']
    
    def get_manifest(self) -> dict:
        """Get full manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {'models': [], 'summary': {}}

