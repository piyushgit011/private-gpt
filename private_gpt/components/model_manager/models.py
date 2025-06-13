from typing import Optional, List
from pydantic import BaseModel, ConfigDict

class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: str
    model_type: str  # 'llm', 'embedding', 'analysis', 'summarization'
    model_name: str
    model_path: Optional[str] = None
    is_loaded: bool = False
    is_downloading: bool = False
    size_gb: Optional[float] = None
    capabilities: List[str] = []