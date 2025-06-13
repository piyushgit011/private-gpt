import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from huggingface_hub import hf_hub_download, snapshot_download
from llama_index.core.llms import LLM

from private_gpt.paths import models_path, models_cache_path
from private_gpt.settings.settings import Settings
from private_gpt.components.model_manager.models import ModelInfo

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles downloading and loading of models."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models_path = models_path
        self.cache_path = models_cache_path
        self.download_tasks: Dict[str, Dict] = {}
    
    def start_download(self, model_config: Dict[str, Any]) -> str:
        """Start downloading a model in background."""
        download_id = str(uuid.uuid4())
        
        self.download_tasks[download_id] = {
            "model_config": model_config,
            "status": "downloading",
            "progress": 0.0,
            "error": None
        }
        
        # Start download in background thread
        thread = threading.Thread(
            target=self._download_model,
            args=(download_id, model_config),
            daemon=True
        )
        thread.start()
        
        return download_id
    
    def _download_model(self, download_id: str, model_config: Dict[str, Any]) -> None:
        """Download model implementation."""
        try:
            model_type = model_config.get("model_type", "llm")
            repo_id = model_config.get("repo_id")
            filename = model_config.get("filename")
            
            if not repo_id:
                raise ValueError("repo_id is required")
            
            # Update progress
            self.download_tasks[download_id]["status"] = "downloading"
            self.download_tasks[download_id]["progress"] = 0.1
            
            if filename:
                # Download specific file
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(self.cache_path),
                    local_dir=self.models_path / model_config["model_id"],
                    token=getattr(self.settings.huggingface, 'access_token', None),
                )
            else:
                # Download entire repository
                local_path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=str(self.cache_path),
                    local_dir=self.models_path / model_config["model_id"],
                    token=getattr(self.settings.huggingface, 'access_token', None),
                )
            
            # Update task status
            self.download_tasks[download_id]["status"] = "completed"
            self.download_tasks[download_id]["progress"] = 1.0
            self.download_tasks[download_id]["local_path"] = str(local_path)
            
            logger.info(f"Model download completed: {model_config['model_id']}")
            
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            self.download_tasks[download_id]["status"] = "failed"
            self.download_tasks[download_id]["error"] = str(e)
    
    def get_download_status(self, download_id: str) -> Optional[Dict]:
        """Get download task status."""
        return self.download_tasks.get(download_id)
    
    def load_model(self, model_info: ModelInfo) -> Optional[LLM]:
        """Load a model into memory."""
        try:
            model_type = model_info.model_type
            model_path = model_info.model_path or (self.models_path / model_info.model_id)
            
            if model_type == "llm":
                return self._load_llm_model(model_info, Path(model_path))
            elif model_type == "embedding":
                return self._load_embedding_model(model_info, Path(model_path))
            else:
                logger.warning(f"Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load model {model_info.model_id}: {e}")
            return None
    
    def _load_llm_model(self, model_info: ModelInfo, model_path: Path) -> Optional[LLM]:
        """Load an LLM model."""
        try:
            from llama_index.llms.llama_cpp import LlamaCPP
            
            # Find model file
            model_files = list(model_path.glob("*.gguf"))
            if not model_files:
                model_files = list(model_path.glob("*.bin"))
            
            if not model_files:
                raise FileNotFoundError(f"No model files found in {model_path}")
            
            model_file = model_files[0]  # Use first found model file
            
            llm = LlamaCPP(
                model_path=str(model_file),
                temperature=0.1,
                max_new_tokens=512,
                context_window=3900,
                n_gpu_layers=-1,
                verbose=True,
            )
            
            return llm
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            return None
    
    def _load_embedding_model(self, model_info: ModelInfo, model_path: Path) -> Optional[Any]:
        """Load an embedding model."""
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            embedding = HuggingFaceEmbedding(
                model_name=str(model_path),
                cache_folder=str(self.cache_path),
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None