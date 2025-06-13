import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional
from injector import inject, singleton
from llama_index.core.llms import LLM

from private_gpt.paths import models_path
from private_gpt.settings.settings import Settings
from private_gpt.components.model_manager.models import ModelInfo
from private_gpt.components.model_manager.model_loader import ModelLoader
from private_gpt.components.model_manager.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

@singleton
class ModelManagerComponent:
    """Manages multiple AI models with hot-swapping capabilities."""
    
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.models_path = models_path
        self.loaded_models: Dict[str, LLM] = {}
        self.model_registry = ModelRegistry()
        self.model_loader = ModelLoader(settings)
        self._lock = threading.RLock()
        
        # Initialize with current models
        self._initialize_current_models()
    
    def _initialize_current_models(self) -> None:
        """Initialize registry with currently configured models."""
        # Register current LLM
        if self.settings.llm.mode != "mock":
            current_llm = ModelInfo(
                model_id="current_llm",
                model_type="llm",
                model_name=f"{self.settings.llm.mode}_current",
                is_loaded=True,
                capabilities=["chat", "completion"]
            )
            self.model_registry.register_model(current_llm)
    
    def list_available_models(self) -> Dict[str, ModelInfo]:
        """List all available models."""
        return self.model_registry.get_all_models()
    
    def list_loaded_models(self) -> Dict[str, ModelInfo]:
        """List currently loaded models."""
        with self._lock:
            loaded = {}
            for model_id, model_info in self.model_registry.get_all_models().items():
                if model_info.is_loaded:
                    loaded[model_id] = model_info
            return loaded
    
    def download_model(self, model_config: Dict[str, Any]) -> str:
        """Download a model and return download ID."""
        model_id = model_config.get("model_id")
        if not model_id:
            raise ValueError("model_id is required")
        
        # Register model as downloading
        model_info = ModelInfo(
            model_id=model_id,
            model_type=model_config.get("model_type", "llm"),
            model_name=model_config.get("model_name", model_id),
            is_downloading=True,
            capabilities=model_config.get("capabilities", [])
        )
        
        self.model_registry.register_model(model_info)
        
        # Start download in background
        download_id = self.model_loader.start_download(model_config)
        return download_id
    
    def load_model(self, model_id: str) -> bool:
        """Load a model into memory."""
        with self._lock:
            model_info = self.model_registry.get_model(model_id)
            if not model_info:
                raise ValueError(f"Model {model_id} not found in registry")
            
            if model_info.is_loaded:
                logger.info(f"Model {model_id} already loaded")
                return True
            
            # Load the model
            model = self.model_loader.load_model(model_info)
            if model:
                self.loaded_models[model_id] = model
                model_info.is_loaded = True
                self.model_registry.update_model(model_info)
                logger.info(f"Successfully loaded model {model_id}")
                return True
            
            return False
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        with self._lock:
            if model_id in self.loaded_models:
                del self.loaded_models[model_id]
                
                model_info = self.model_registry.get_model(model_id)
                if model_info:
                    model_info.is_loaded = False
                    self.model_registry.update_model(model_info)
                
                logger.info(f"Unloaded model {model_id}")
                return True
            return False
    
    def get_model(self, model_id: str) -> Optional[LLM]:
        """Get a loaded model."""
        return self.loaded_models.get(model_id)
    
    def switch_active_model(self, model_id: str, model_type: str = "llm") -> bool:
        """Switch the active model for a specific type."""
        if not self.get_model(model_id):
            # Try to load the model first
            if not self.load_model(model_id):
                return False
        
        # Update active model reference
        setattr(self, f"active_{model_type}_id", model_id)
        logger.info(f"Switched active {model_type} to {model_id}")
        return True
    
    def get_active_model(self, model_type: str = "llm") -> Optional[LLM]:
        """Get the currently active model of specified type."""
        active_id = getattr(self, f"active_{model_type}_id", None)
        if active_id:
            return self.get_model(active_id)
        return None