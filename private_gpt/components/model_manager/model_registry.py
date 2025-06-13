import json
import logging
from pathlib import Path
from typing import Dict, Optional
from private_gpt.paths import local_data_path
from private_gpt.components.model_manager.models import ModelInfo

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for tracking available and loaded models."""
    
    def __init__(self):
        self.registry_path = local_data_path / "model_registry.json"
        self.models: Dict[str, ModelInfo] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load model registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    for model_id, model_data in data.items():
                        self.models[model_id] = ModelInfo(**model_data)
            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")
    
    def _save_registry(self) -> None:
        """Save model registry to disk."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, 'w') as f:
                data = {k: v.dict() for k, v in self.models.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def register_model(self, model_info: ModelInfo) -> None:
        """Register a new model."""
        self.models[model_info.model_id] = model_info
        self._save_registry()
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info by ID."""
        return self.models.get(model_id)
    
    def get_all_models(self) -> Dict[str, ModelInfo]:
        """Get all registered models."""
        return self.models.copy()
    
    def update_model(self, model_info: ModelInfo) -> None:
        """Update model information."""
        self.models[model_info.model_id] = model_info
        self._save_registry()
    
    def remove_model(self, model_id: str) -> bool:
        """Remove a model from registry."""
        if model_id in self.models:
            del self.models[model_id]
            self._save_registry()
            return True
        return False