from typing import Dict, Any, List
from fastapi import APIRouter, Depends, Request, HTTPException
from pydantic import BaseModel, ConfigDict

from private_gpt.components.model_manager.model_manager_component import ModelManagerComponent
from private_gpt.components.model_manager.models import ModelInfo
from private_gpt.server.utils.auth import authenticated

models_router = APIRouter(prefix="/v1/models", dependencies=[Depends(authenticated)])

class ModelConfigBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: str
    model_type: str = "llm"  # llm, embedding, analysis, summarization
    model_name: str
    repo_id: str
    filename: str = None
    capabilities: List[str] = []

class ModelListResponse(BaseModel):
    models: Dict[str, ModelInfo]

class DownloadResponse(BaseModel):
    download_id: str
    message: str

class StatusResponse(BaseModel):
    status: str
    message: str = None

@models_router.get("/list", response_model=ModelListResponse, tags=["Model Management"])
def list_models(request: Request) -> ModelListResponse:
    """List all available models."""
    model_manager = request.state.injector.get(ModelManagerComponent)
    models = model_manager.list_available_models()
    return ModelListResponse(models=models)

@models_router.get("/loaded", response_model=ModelListResponse, tags=["Model Management"])
def list_loaded_models(request: Request) -> ModelListResponse:
    """List currently loaded models."""
    model_manager = request.state.injector.get(ModelManagerComponent)
    models = model_manager.list_loaded_models()
    return ModelListResponse(models=models)

@models_router.post("/download", response_model=DownloadResponse, tags=["Model Management"])
def download_model(request: Request, body: ModelConfigBody) -> DownloadResponse:
    """Download a model for local use."""
    model_manager = request.state.injector.get(ModelManagerComponent)
    
    model_config = {
        "model_id": body.model_id,
        "model_type": body.model_type,
        "model_name": body.model_name,
        "repo_id": body.repo_id,
        "filename": body.filename,
        "capabilities": body.capabilities
    }
    
    download_id = model_manager.download_model(model_config)
    return DownloadResponse(
        download_id=download_id,
        message=f"Started downloading model {body.model_id}"
    )

@models_router.get("/download/{download_id}/status", tags=["Model Management"])
def get_download_status(request: Request, download_id: str) -> Dict[str, Any]:
    """Get download status."""
    model_manager = request.state.injector.get(ModelManagerComponent)
    status = model_manager.model_loader.get_download_status(download_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Download task not found")
    
    return status

@models_router.post("/load/{model_id}", response_model=StatusResponse, tags=["Model Management"])
def load_model(request: Request, model_id: str) -> StatusResponse:
    """Load a model into memory."""
    model_manager = request.state.injector.get(ModelManagerComponent)
    
    success = model_manager.load_model(model_id)
    if success:
        return StatusResponse(status="success", message=f"Model {model_id} loaded successfully")
    else:
        raise HTTPException(status_code=400, detail=f"Failed to load model {model_id}")

@models_router.post("/unload/{model_id}", response_model=StatusResponse, tags=["Model Management"])
def unload_model(request: Request, model_id: str) -> StatusResponse:
    """Unload a model from memory."""
    model_manager = request.state.injector.get(ModelManagerComponent)
    
    success = model_manager.unload_model(model_id)
    if success:
        return StatusResponse(status="success", message=f"Model {model_id} unloaded successfully")
    else:
        raise HTTPException(status_code=400, detail=f"Failed to unload model {model_id}")

@models_router.post("/switch/{model_id}", response_model=StatusResponse, tags=["Model Management"])
def switch_active_model(request: Request, model_id: str, model_type: str = "llm") -> StatusResponse:
    """Switch the active model for a specific type."""
    model_manager = request.state.injector.get(ModelManagerComponent)
    
    success = model_manager.switch_active_model(model_id, model_type)
    if success:
        return StatusResponse(
            status="success", 
            message=f"Switched active {model_type} to {model_id}"
        )
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to switch active {model_type} to {model_id}"
        )

@models_router.delete("/{model_id}", response_model=StatusResponse, tags=["Model Management"])
def delete_model(request: Request, model_id: str) -> StatusResponse:
    """Delete a model and its files."""
    model_manager = request.state.injector.get(ModelManagerComponent)
    
    # First unload if loaded
    model_manager.unload_model(model_id)
    
    # Remove from registry
    success = model_manager.model_registry.remove_model(model_id)
    
    # TODO: Also delete model files from disk
    
    if success:
        return StatusResponse(status="success", message=f"Model {model_id} deleted successfully")
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")