# Simple model registry - tracks model versions and metrics
import joblib
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    # Simple model registry
    
    def __init__(self, registry_path: str = "models/"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(
        self,
        model_name: str,
        model_path: str,
        version: str,
        metrics: Dict,
        metadata: Dict = None
    ):
        if model_name not in self.registry:
            self.registry[model_name] = {}
        
        self.registry[model_name][version] = {
            'model_path': model_path,
            'metrics': metrics,
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat()
        }
        
        self._save_registry()
        logger.info(f"Registered {model_name} version {version}")
    
    def get_model(self, model_name: str, version: str = "latest") -> Optional[Dict]:
        if model_name not in self.registry:
            return None
        
        if version == "latest":
            versions = list(self.registry[model_name].keys())
            if not versions:
                return None
            version = max(versions)
        
        if version not in self.registry[model_name]:
            return None
        
        return self.registry[model_name][version]
    
    def load_model(self, model_name: str, version: str = "latest"):
        model_info = self.get_model(model_name, version)
        if model_info is None:
            raise ValueError(f"Model {model_name} version {version} not found")
        
        model_path = model_info['model_path']
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return joblib.load(model_path)
    
    def list_models(self) -> Dict:
        return self.registry
    
    def get_best_model(self, metric: str = "pr_auc") -> Optional[Dict]:
        best_score = -1
        best_model = None
        
        for model_name, versions in self.registry.items():
            for version, info in versions.items():
                metrics = info.get('metrics', {})
                if metric in metrics:
                    score = metrics[metric]
                    if score > best_score:
                        best_score = score
                        best_model = {
                            'model_name': model_name,
                            'version': version,
                            'metrics': metrics,
                            'model_path': info['model_path']
                        }
        
        return best_model
