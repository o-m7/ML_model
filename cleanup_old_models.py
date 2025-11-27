"""
Cleanup Old/Incomplete Models

Removes models that are missing feature_cols or other required fields.
These are likely from older training runs with different save formats.
"""

import joblib
from pathlib import Path
import argparse
import shutil


class ModelCleanup:
    """Clean up incomplete model files."""
    
    def __init__(self, symbol: str, backup: bool = True):
        self.symbol = symbol
        self.models_dir = Path("ML_model/ML_model/models") / symbol
        self.backup_dir = Path("ML_model/ML_model/models/backup") / symbol if backup else None
    
    def analyze_all_models(self):
        """Analyze all models and identify issues."""
        if not self.models_dir.exists():
            print(f"❌ Model directory not found: {self.models_dir}")
            return
        
        model_files = list(self.models_dir.glob(f"{self.symbol}_*.pkl"))
        
        if not model_files:
            print(f"❌ No models found")
            return
        
        print(f"\n{'='*80}")
        print(f"ANALYZING MODELS")
        print(f"{'='*80}")
        print(f"Found {len(model_files)} models\n")
        
        good_models = []
        incomplete_models = []
        
        for model_file in model_files:
            try:
                model_data = joblib.load(model_file)
                
                # Check required fields
                has_model = 'model' in model_data
                has_scaler = 'scaler' in model_data
                has_features = 'feature_cols' in model_data
                
                if has_model and has_features:
                    good_models.append(model_file)
                    print(f"✅ {model_file.name}")
                    print(f"   Keys: {list(model_data.keys())}")
                else:
                    incomplete_models.append(model_file)
                    print(f"❌ {model_file.name}")
                    print(f"   Keys: {list(model_data.keys())}")
                    print(f"   Missing: {[k for k in ['model', 'feature_cols'] if k not in model_data]}")
                
            except Exception as e:
                print(f"⚠️  {model_file.name}: {e}")
                incomplete_models.append(model_file)
        
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Complete models: {len(good_models)}")
        print(f"❌ Incomplete models: {len(incomplete_models)}")
        
        if incomplete_models:
            print(f"\nIncomplete models:")
            for m in incomplete_models:
                print(f"   {m.name}")
        
        return good_models, incomplete_models
    
    def cleanup_incomplete_models(self, dry_run: bool = False):
        """Remove or backup incomplete models."""
        good_models, incomplete_models = self.analyze_all_models()
        
        if not incomplete_models:
            print(f"\n✅ All models are complete!")
            return
        
        print(f"\n{'='*80}")
        print(f"CLEANUP")
        print(f"{'='*80}")
        
        if dry_run:
            print("DRY RUN - No files will be modified")
            print(f"\nWould remove {len(incomplete_models)} models:")
            for m in incomplete_models:
                print(f"   {m.name}")
            print("\nRun without --dry-run to actually remove files")
            return
        
        # Create backup if needed
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            print(f"Backup directory: {self.backup_dir}")
        
        removed_count = 0
        backed_up_count = 0
        
        for model_file in incomplete_models:
            try:
                # Backup if requested
                if self.backup_dir:
                    backup_file = self.backup_dir / model_file.name
                    shutil.copy2(model_file, backup_file)
                    backed_up_count += 1
                    print(f"📦 Backed up: {model_file.name}")
                
                # Remove original
                model_file.unlink()
                removed_count += 1
                print(f"🗑️  Removed: {model_file.name}")
                
                # Also remove metadata if it exists
                meta_file = model_file.parent / f"{model_file.stem}_meta.json"
                if meta_file.exists():
                    if self.backup_dir:
                        backup_meta = self.backup_dir / meta_file.name
                        shutil.copy2(meta_file, backup_meta)
                    meta_file.unlink()
                    print(f"🗑️  Removed metadata: {meta_file.name}")
                
            except Exception as e:
                print(f"⚠️  Error removing {model_file.name}: {e}")
        
        print(f"\n{'='*80}")
        print(f"CLEANUP COMPLETE")
        print(f"{'='*80}")
        print(f"📦 Backed up: {backed_up_count}")
        print(f"🗑️  Removed: {removed_count}")
        print(f"✅ Remaining: {len(good_models)}")
        
        if self.backup_dir and backed_up_count > 0:
            print(f"\n💾 Backups saved to: {self.backup_dir}")
        
        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Clean up incomplete models')
    parser.add_argument('--symbol', type=str, default='XAUUSD', help='Symbol to clean')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup before removal')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be removed without actually removing')
    
    args = parser.parse_args()
    
    cleanup = ModelCleanup(args.symbol, backup=not args.no_backup)
    
    if args.dry_run:
        cleanup.cleanup_incomplete_models(dry_run=True)
    else:
        print("\n⚠️  This will remove incomplete model files!")
        if not args.no_backup:
            print("📦 Backups will be created before removal")
        
        response = input("\nContinue? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            cleanup.cleanup_incomplete_models(dry_run=False)
        else:
            print("Cancelled")


if __name__ == '__main__':
    main()