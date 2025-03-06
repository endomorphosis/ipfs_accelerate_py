#!/usr/bin/env python3
"""
Hardware Selection API

This module implements a RESTful API for the Hardware Selector system, providing endpoints
for hardware recommendations, performance predictions, and configuration optimization.

The API provides dynamic hardware selection based on model characteristics and available hardware,
with comprehensive documentation and client libraries for easy integration.

Usage:
  # Start the API server
  python hardware_selection_api.py --host 0.0.0.0 --port 8000 --debug

  # In production with gunicorn
  gunicorn hardware_selection_api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator

# Import hardware selector
from hardware_selector import HardwareSelector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hardware_selection_api")

# Check if model_performance_predictor is available
try:
    import model_performance_predictor
    HAS_PERFORMANCE_PREDICTOR = True
    logger.info("model_performance_predictor is available, enabling advanced prediction features")
except ImportError:
    HAS_PERFORMANCE_PREDICTOR = False
    logger.warning("model_performance_predictor not available, some prediction features will be limited")

# Create FastAPI app
app = FastAPI(
    title="Hardware Selection API",
    description="API for hardware recommendations and performance predictions",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define API models
class ModelInfo(BaseModel):
    """Model information."""
    name: str = Field(..., description="Model name", example="bert-base-uncased")
    family: str = Field(..., description="Model family", example="embedding")
    size: Optional[int] = Field(None, description="Model size in parameters", example=110000000)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "bert-base-uncased",
                "family": "embedding",
                "size": 110000000
            }
        }

class HardwareInfo(BaseModel):
    """Hardware information."""
    type: str = Field(..., description="Hardware type", example="cuda")
    name: Optional[str] = Field(None, description="Hardware name", example="NVIDIA RTX 3090")
    memory_gb: Optional[float] = Field(None, description="Memory capacity in GB", example=24.0)
    
    class Config:
        schema_extra = {
            "example": {
                "type": "cuda",
                "name": "NVIDIA RTX 3090",
                "memory_gb": 24.0
            }
        }

class SelectionRequest(BaseModel):
    """Hardware selection request."""
    model: ModelInfo = Field(..., description="Model information")
    batch_size: int = Field(1, description="Batch size", example=1)
    sequence_length: Optional[int] = Field(128, description="Sequence length", example=128)
    mode: str = Field("inference", description="Mode (inference or training)", example="inference")
    available_hardware: Optional[List[str]] = Field(None, description="Available hardware types")
    precision: str = Field("fp32", description="Precision (fp32, fp16, int8)", example="fp32")
    
    @validator('mode')
    def mode_must_be_valid(cls, v):
        if v not in ["inference", "training"]:
            raise ValueError('mode must be either "inference" or "training"')
        return v
    
    @validator('precision')
    def precision_must_be_valid(cls, v):
        if v not in ["fp32", "fp16", "int8"]:
            raise ValueError('precision must be one of "fp32", "fp16", "int8"')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model": {
                    "name": "bert-base-uncased",
                    "family": "embedding",
                    "size": 110000000
                },
                "batch_size": 1,
                "sequence_length": 128,
                "mode": "inference",
                "available_hardware": ["cpu", "cuda", "mps"],
                "precision": "fp32"
            }
        }

class PredictionRequest(BaseModel):
    """Performance prediction request."""
    model: ModelInfo = Field(..., description="Model information")
    hardware: HardwareInfo = Field(..., description="Hardware information")
    batch_size: int = Field(1, description="Batch size", example=1)
    sequence_length: Optional[int] = Field(128, description="Sequence length", example=128)
    precision: str = Field("fp32", description="Precision (fp32, fp16, int8)", example="fp32")
    mode: str = Field("inference", description="Mode (inference or training)", example="inference")
    
    @validator('mode')
    def mode_must_be_valid(cls, v):
        if v not in ["inference", "training"]:
            raise ValueError('mode must be either "inference" or "training"')
        return v
    
    @validator('precision')
    def precision_must_be_valid(cls, v):
        if v not in ["fp32", "fp16", "int8"]:
            raise ValueError('precision must be one of "fp32", "fp16", "int8"')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model": {
                    "name": "bert-base-uncased",
                    "family": "embedding"
                },
                "hardware": {
                    "type": "cuda",
                    "name": "NVIDIA RTX 3090"
                },
                "batch_size": 1,
                "sequence_length": 128,
                "precision": "fp32",
                "mode": "inference"
            }
        }

class DistributedTrainingRequest(BaseModel):
    """Distributed training configuration request."""
    model: ModelInfo = Field(..., description="Model information")
    gpu_count: int = Field(..., description="Number of GPUs", example=4)
    batch_size: int = Field(8, description="Per-GPU batch size", example=8)
    max_memory_gb: Optional[int] = Field(None, description="Maximum GPU memory in GB", example=24)
    strategy: Optional[str] = Field(None, description="Distributed strategy (DDP, FSDP, DeepSpeed)")
    
    @validator('gpu_count')
    def gpu_count_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('gpu_count must be positive')
        return v
    
    @validator('batch_size')
    def batch_size_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('batch_size must be positive')
        return v
    
    @validator('strategy')
    def strategy_must_be_valid(cls, v):
        if v is not None and v not in ["DDP", "FSDP", "DeepSpeed"]:
            raise ValueError('strategy must be one of "DDP", "FSDP", "DeepSpeed"')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model": {
                    "name": "llama-7b",
                    "family": "text_generation"
                },
                "gpu_count": 4,
                "batch_size": 8,
                "max_memory_gb": 24,
                "strategy": "FSDP"
            }
        }

class MatrixGenerationRequest(BaseModel):
    """Hardware selection matrix generation request."""
    model_families: Optional[List[str]] = Field(None, description="Model families to include in the matrix")
    
    class Config:
        schema_extra = {
            "example": {
                "model_families": ["embedding", "text_generation", "vision"]
            }
        }

class TaskRequest(BaseModel):
    """Task-specific hardware selection request."""
    model: ModelInfo = Field(..., description="Model information")
    task_type: str = Field(..., description="Task type", example="generation")
    batch_size: int = Field(1, description="Batch size", example=1)
    sequence_length: Optional[int] = Field(128, description="Sequence length", example=128)
    available_hardware: Optional[List[str]] = Field(None, description="Available hardware types")
    distributed: bool = Field(False, description="Whether to use distributed training")
    gpu_count: int = Field(1, description="Number of GPUs for distributed training")
    
    class Config:
        schema_extra = {
            "example": {
                "model": {
                    "name": "gpt2",
                    "family": "text_generation"
                },
                "task_type": "generation",
                "batch_size": 1,
                "sequence_length": 128,
                "available_hardware": ["cpu", "cuda", "mps"],
                "distributed": False,
                "gpu_count": 1
            }
        }

# Hardware selection dependency
def get_hardware_selector():
    """Create and return a HardwareSelector instance."""
    # You can customize the database path here or use environment variables
    database_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_results")
    config_path = os.environ.get("HARDWARE_SELECTOR_CONFIG", None)
    
    return HardwareSelector(
        database_path=database_path,
        config_path=config_path
    )

# API endpoints
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint providing API information."""
    return {
        "name": "Hardware Selection API",
        "version": "1.0.0",
        "description": "API for hardware recommendations and performance predictions",
        "documentation": "/docs"
    }

@app.get("/health", include_in_schema=False)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/select", response_model=Dict[str, Any], tags=["Selection"])
async def select_hardware(
    request: SelectionRequest,
    selector: HardwareSelector = Depends(get_hardware_selector)
):
    """
    Select optimal hardware for a given model and configuration.
    
    This endpoint recommends the most suitable hardware platform based on the model
    characteristics, batch size, and available hardware.
    """
    try:
        result = selector.select_hardware(
            model_family=request.model.family,
            model_name=request.model.name,
            batch_size=request.batch_size,
            sequence_length=request.sequence_length,
            mode=request.mode,
            available_hardware=request.available_hardware,
            precision=request.precision
        )
        
        return result
    except Exception as e:
        logger.error(f"Error selecting hardware: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error selecting hardware: {str(e)}"
        )

@app.post("/task", response_model=Dict[str, Any], tags=["Selection"])
async def select_hardware_for_task(
    request: TaskRequest,
    selector: HardwareSelector = Depends(get_hardware_selector)
):
    """
    Select optimal hardware for a specific task.
    
    This endpoint recommends hardware based on the specific task requirements,
    including task type and distributed training configuration.
    """
    try:
        # For distributed training, we may need a training config
        training_config = None
        if request.distributed and request.gpu_count > 1:
            training_config = {
                "mixed_precision": True  # Default to mixed precision for distributed training
            }
            
        result = selector.select_hardware_for_task(
            model_family=request.model.family,
            model_name=request.model.name,
            task_type=request.task_type,
            batch_size=request.batch_size,
            sequence_length=request.sequence_length,
            available_hardware=request.available_hardware,
            distributed=request.distributed,
            gpu_count=request.gpu_count,
            training_config=training_config
        )
        
        return result
    except Exception as e:
        logger.error(f"Error selecting hardware for task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error selecting hardware for task: {str(e)}"
        )

@app.post("/predict", response_model=Dict[str, Any], tags=["Prediction"])
async def predict_performance(
    request: PredictionRequest,
    selector: HardwareSelector = Depends(get_hardware_selector)
):
    """
    Predict performance for a model-hardware configuration.
    
    This endpoint provides predictions for latency, throughput, and memory usage
    based on the model and hardware characteristics.
    """
    try:
        # First try to use model_performance_predictor if available
        if HAS_PERFORMANCE_PREDICTOR:
            # Load models for prediction
            models = model_performance_predictor.load_prediction_models()
            
            # Use the prediction function
            prediction = model_performance_predictor.predict_performance(
                models=models,
                model_name=request.model.name,
                model_category=request.model.family,
                hardware=request.hardware.type,
                batch_size=request.batch_size,
                precision=request.precision,
                mode=request.mode
            )
            
            if prediction:
                return prediction
        
        # Fall back to hardware selector's prediction capabilities
        # This is a workaround since the hardware selector doesn't have a direct
        # predict_performance method, we'll use select_hardware and extract predictions
        result = selector.select_hardware(
            model_family=request.model.family,
            model_name=request.model.name,
            batch_size=request.batch_size,
            sequence_length=request.sequence_length,
            mode=request.mode,
            available_hardware=[request.hardware.type],
            precision=request.precision
        )
        
        # Extract predictions
        predictions = {}
        
        # Try to get predictions from all_scores if available
        if "all_scores" in result and request.hardware.type in result["all_scores"]:
            hw_scores = result["all_scores"][request.hardware.type]
            
            if "predictions" in hw_scores:
                predictions = hw_scores["predictions"]
            else:
                # Build predictions from scores
                predictions = {
                    "latency": 1.0 / (hw_scores["latency_score"] * 100) if hw_scores["latency_score"] > 0 else 0,
                    "throughput": hw_scores["throughput_score"] * 100,
                    "memory_usage": (1.0 - hw_scores["memory_factor"]) * result["model_size"] * 0.01
                }
        
        # If no predictions available, create a basic response
        if not predictions:
            predictions = {
                "model": request.model.name,
                "hardware": request.hardware.type,
                "batch_size": request.batch_size,
                "precision": request.precision,
                "mode": request.mode,
                "is_predicted": True,
                "timestamp": datetime.now().isoformat()
            }
        
        return predictions
    except Exception as e:
        logger.error(f"Error predicting performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error predicting performance: {str(e)}"
        )

@app.post("/distributed", response_model=Dict[str, Any], tags=["Training"])
async def get_distributed_training_config(
    request: DistributedTrainingRequest,
    selector: HardwareSelector = Depends(get_hardware_selector)
):
    """
    Generate an optimal distributed training configuration.
    
    This endpoint provides hardware and configuration recommendations for
    distributed training based on the model and available GPUs.
    """
    try:
        config = selector.get_distributed_training_config(
            model_family=request.model.family,
            model_name=request.model.name,
            gpu_count=request.gpu_count,
            batch_size=request.batch_size,
            strategy=request.strategy,
            max_memory_gb=request.max_memory_gb
        )
        
        return config
    except Exception as e:
        logger.error(f"Error generating distributed training configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating distributed training configuration: {str(e)}"
        )

@app.post("/matrix", response_model=Dict[str, Any], tags=["Selection"])
async def create_selection_matrix(
    request: MatrixGenerationRequest,
    selector: HardwareSelector = Depends(get_hardware_selector)
):
    """
    Generate a hardware selection matrix.
    
    This endpoint creates a comprehensive matrix of hardware recommendations for
    various model families, sizes, and batch sizes.
    """
    try:
        matrix = selector.create_hardware_selection_map(
            model_families=request.model_families
        )
        
        return matrix
    except Exception as e:
        logger.error(f"Error creating selection matrix: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating selection matrix: {str(e)}"
        )

@app.get("/hardware/available", response_model=Dict[str, bool], tags=["Hardware"])
async def get_available_hardware():
    """
    Get available hardware platforms.
    
    This endpoint returns a list of available hardware platforms based on
    the runtime detection of the system.
    """
    try:
        # Basic hardware detection
        available_hw = {
            "cpu": True,  # CPU is always available
        }
        
        # Try to detect CUDA
        try:
            import torch
            available_hw["cuda"] = torch.cuda.is_available()
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, "mps"):
                available_hw["mps"] = torch.backends.mps.is_available()
        except ImportError:
            available_hw["cuda"] = False
            available_hw["mps"] = False
        
        # Try to detect ROCm through PyTorch
        try:
            import torch
            if torch.cuda.is_available() and "rocm" in torch.__version__.lower():
                available_hw["rocm"] = True
            else:
                available_hw["rocm"] = False
        except (ImportError, AttributeError):
            available_hw["rocm"] = False
        
        # Try to detect OpenVINO
        try:
            import openvino
            available_hw["openvino"] = True
        except ImportError:
            available_hw["openvino"] = False
            
        # We can't detect WebNN or WebGPU from the server side
        # as they're browser-specific technologies
        available_hw["webnn"] = False
        available_hw["webgpu"] = False
        
        # Try to detect Qualcomm AI Engine
        try:
            import qnn
            available_hw["qualcomm"] = True
        except ImportError:
            available_hw["qualcomm"] = False
        
        return available_hw
    except Exception as e:
        logger.error(f"Error detecting available hardware: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error detecting available hardware: {str(e)}"
        )

@app.get("/models/families", response_model=List[str], tags=["Models"])
async def get_model_families(
    selector: HardwareSelector = Depends(get_hardware_selector)
):
    """
    Get available model families.
    
    This endpoint returns a list of supported model families.
    """
    try:
        # Extract model families from the compatibility matrix
        model_families = list(selector.compatibility_matrix.get("model_families", {}).keys())
        
        return model_families
    except Exception as e:
        logger.error(f"Error getting model families: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model families: {str(e)}"
        )

@app.get("/compatibility/matrix", response_model=Dict[str, Any], tags=["Compatibility"])
async def get_compatibility_matrix(
    selector: HardwareSelector = Depends(get_hardware_selector)
):
    """
    Get the hardware compatibility matrix.
    
    This endpoint returns the complete compatibility matrix for all model families
    and hardware platforms.
    """
    try:
        return selector.compatibility_matrix
    except Exception as e:
        logger.error(f"Error getting compatibility matrix: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting compatibility matrix: {str(e)}"
        )

@app.get("/compatibility/{model_family}", response_model=Dict[str, Any], tags=["Compatibility"])
async def get_model_family_compatibility(
    model_family: str,
    selector: HardwareSelector = Depends(get_hardware_selector)
):
    """
    Get compatibility information for a specific model family.
    
    This endpoint returns the compatibility information for a specific model family
    across all hardware platforms.
    """
    try:
        if model_family not in selector.compatibility_matrix.get("model_families", {}):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model family '{model_family}' not found in compatibility matrix"
            )
            
        return selector.compatibility_matrix["model_families"][model_family]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting compatibility for model family {model_family}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting compatibility for model family {model_family}: {str(e)}"
        )

@app.get("/docs/openapi.json", include_in_schema=False)
async def get_openapi_json():
    """
    Get OpenAPI schema for API documentation.
    
    This endpoint returns a customized OpenAPI schema.
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Hardware Selection API",
        version="1.0.0",
        description="API for hardware recommendations and performance predictions",
        routes=app.routes,
    )
    
    # Add additional metadata
    openapi_schema["info"]["x-logo"] = {
        "url": "https://raw.githubusercontent.com/anthropics/claude-code/master/logo.png"
    }
    
    # Add servers information
    openapi_schema["servers"] = [
        {"url": "/", "description": "Current server"},
    ]
    
    # Add contact information
    openapi_schema["info"]["contact"] = {
        "name": "IPFS Accelerate Team",
        "email": "ipfs-accelerate@example.com",
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Main entry point
def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(description="Hardware Selection API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--database", help="Path to benchmark database")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (for production)")
    
    args = parser.parse_args()
    
    # Configure environment variables
    if args.database:
        os.environ["BENCHMARK_DB_PATH"] = args.database
    if args.config:
        os.environ["HARDWARE_SELECTOR_CONFIG"] = args.config
    
    # Configure logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.getLogger("hardware_selection_api").setLevel(log_level)
    
    print(f"Starting Hardware Selection API on {args.host}:{args.port}")
    print(f"API documentation available at http://{args.host}:{args.port}/docs")
    
    # Start the server
    uvicorn.run(
        "hardware_selection_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug" if args.debug else "info",
        workers=args.workers if not args.reload else 1
    )

if __name__ == "__main__":
    main()