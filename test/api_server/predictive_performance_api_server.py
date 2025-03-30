#!/usr/bin/env python3
"""
FastAPI server for the Predictive Performance Modeling System.

This module provides a FastAPI server with RESTful endpoints and WebSocket support
for accessing the Predictive Performance Modeling System functionality, including
hardware recommendations, performance predictions, and measurement tracking.
"""

import os
import sys
import time
import uuid
import json
import logging
import argparse
import asyncio
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("predictive_performance_api")

# Import FastAPI components
try:
    from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, Depends, BackgroundTasks, Query, Path, Body, Header
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
except ImportError:
    logger.error("FastAPI or related packages not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

# Import DuckDB components
try:
    from duckdb_api.predictive_performance.predictor_repository import DuckDBPredictorRepository
    from duckdb_api.predictive_performance.repository_adapter import (
        HardwareModelPredictorDuckDBAdapter,
        ModelPerformancePredictorDuckDBAdapter
    )
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB predictive performance components not available")
    DUCKDB_AVAILABLE = False

# Try to import hardware model predictor
try:
    from predictive_performance.hardware_model_predictor import HardwareModelPredictor
    HARDWARE_MODEL_PREDICTOR_AVAILABLE = True
except ImportError:
    logger.warning("HardwareModelPredictor not available")
    HARDWARE_MODEL_PREDICTOR_AVAILABLE = False

# Define Pydantic models for API endpoints
class HardwarePlatform(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    OPENVINO = "openvino"
    WEBGPU = "webgpu"
    WEBNN = "webnn"
    QNN = "qnn"

class PrecisionType(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"

class ModelMode(str, Enum):
    INFERENCE = "inference"
    TRAINING = "training"

class HardwareRequest(BaseModel):
    model_name: str
    model_family: Optional[str] = None
    batch_size: int = 1
    sequence_length: int = 128
    mode: ModelMode = ModelMode.INFERENCE
    precision: PrecisionType = PrecisionType.FP32
    available_hardware: Optional[List[HardwarePlatform]] = None
    predict_performance: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "bert-base-uncased",
                "model_family": "embedding",
                "batch_size": 8,
                "sequence_length": 128,
                "mode": "inference",
                "precision": "fp16",
                "available_hardware": ["cuda", "cpu", "rocm"],
                "predict_performance": True
            }
        }

class PerformanceRequest(BaseModel):
    model_name: str
    model_family: Optional[str] = None
    hardware: Union[HardwarePlatform, List[HardwarePlatform]]
    batch_size: int = 1
    sequence_length: int = 128
    mode: ModelMode = ModelMode.INFERENCE
    precision: PrecisionType = PrecisionType.FP32
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "bert-base-uncased",
                "model_family": "embedding",
                "hardware": ["cuda", "cpu"],
                "batch_size": 8,
                "sequence_length": 128,
                "mode": "inference",
                "precision": "fp16"
            }
        }

class MeasurementRequest(BaseModel):
    model_name: str
    model_family: Optional[str] = None
    hardware_platform: HardwarePlatform
    batch_size: int = 1
    sequence_length: int = 128
    precision: PrecisionType = PrecisionType.FP32
    mode: ModelMode = ModelMode.INFERENCE
    throughput: Optional[float] = None
    latency: Optional[float] = None
    memory_usage: Optional[float] = None
    prediction_id: Optional[str] = None
    source: str = "api"
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "bert-base-uncased",
                "model_family": "embedding",
                "hardware_platform": "cuda",
                "batch_size": 8,
                "sequence_length": 128,
                "precision": "fp16",
                "mode": "inference",
                "throughput": 123.45,
                "latency": 7.89,
                "memory_usage": 1024.5,
                "source": "benchmark"
            }
        }

class AnalysisRequest(BaseModel):
    model_name: Optional[str] = None
    hardware_platform: Optional[HardwarePlatform] = None
    metric: Optional[str] = None
    days: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "bert-base-uncased",
                "hardware_platform": "cuda",
                "metric": "throughput",
                "days": 30
            }
        }

class FeedbackRequest(BaseModel):
    recommendation_id: str
    accepted: bool
    feedback: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "recommendation_id": "rec-1234567890",
                "accepted": True,
                "feedback": "Works great on our production environment"
            }
        }

class MLModelRequest(BaseModel):
    model_type: str
    target_metric: str
    hardware_platform: HardwarePlatform
    model_family: Optional[str] = None
    features: List[str]
    training_score: Optional[float] = None
    validation_score: Optional[float] = None
    test_score: Optional[float] = None
    serialized_model_path: str
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "RandomForestRegressor",
                "target_metric": "throughput",
                "hardware_platform": "cuda",
                "model_family": "embedding",
                "features": ["batch_size", "sequence_length", "precision_numeric"],
                "training_score": 0.95,
                "validation_score": 0.92,
                "test_score": 0.90,
                "serialized_model_path": "/path/to/model.joblib"
            }
        }

class MatrixGenerationRequest(BaseModel):
    model_configs: List[Dict[str, str]]
    hardware_platforms: List[HardwarePlatform]
    batch_sizes: List[int]
    precision_options: List[PrecisionType] = [PrecisionType.FP32]
    mode: ModelMode = ModelMode.INFERENCE
    
    class Config:
        schema_extra = {
            "example": {
                "model_configs": [
                    {"name": "bert-base-uncased", "category": "embedding"},
                    {"name": "gpt2", "category": "text_generation"}
                ],
                "hardware_platforms": ["cuda", "cpu"],
                "batch_sizes": [1, 8, 32],
                "precision_options": ["fp32", "fp16"],
                "mode": "inference"
            }
        }

class SampleDataRequest(BaseModel):
    num_models: int = 5
    
    class Config:
        schema_extra = {
            "example": {
                "num_models": 10
            }
        }

# Background task tracking
task_status = {}
task_results = {}

# Create FastAPI app
app = FastAPI(
    title="Predictive Performance API",
    description="API for Predictive Performance Modeling System",
    version="1.0.0"
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router
router = APIRouter(prefix="/api/predictive-performance", tags=["predictive-performance"])

# WebSocket connections
websocket_connections = {}

# Repository and adapter instances
repository = None
hardware_adapter = None
ml_adapter = None

def get_repository():
    """Get or create repository instance."""
    global repository
    if repository is None and DUCKDB_AVAILABLE:
        try:
            db_path = os.environ.get("PREDICTIVE_PERFORMANCE_DB", "predictive_performance.duckdb")
            repository = DuckDBPredictorRepository(db_path=db_path, create_if_missing=True)
            logger.info(f"Created DuckDB repository at {db_path}")
        except Exception as e:
            logger.error(f"Failed to create repository: {e}")
            raise
    return repository

def get_hardware_adapter():
    """Get or create hardware model predictor adapter."""
    global hardware_adapter
    if hardware_adapter is None and DUCKDB_AVAILABLE:
        try:
            # Create predictor if available
            predictor = None
            if HARDWARE_MODEL_PREDICTOR_AVAILABLE:
                benchmark_dir = os.environ.get("BENCHMARK_DIR", "./benchmark_results")
                benchmark_db = os.environ.get("BENCHMARK_DB", None)
                predictor = HardwareModelPredictor(
                    benchmark_dir=benchmark_dir,
                    database_path=benchmark_db
                )
            
            # Create adapter
            repo = get_repository()
            hardware_adapter = HardwareModelPredictorDuckDBAdapter(
                predictor=predictor,
                repository=repo,
                user_id="api-server"
            )
            logger.info("Created hardware model predictor adapter")
        except Exception as e:
            logger.error(f"Failed to create hardware adapter: {e}")
            raise
    return hardware_adapter

def get_ml_adapter():
    """Get or create ML model adapter."""
    global ml_adapter
    if ml_adapter is None and DUCKDB_AVAILABLE:
        try:
            repo = get_repository()
            ml_adapter = ModelPerformancePredictorDuckDBAdapter(
                repository=repo
            )
            logger.info("Created ML model adapter")
        except Exception as e:
            logger.error(f"Failed to create ML adapter: {e}")
            raise
    return ml_adapter

async def update_task_status(task_id: str, status: str, progress: float = 0, message: str = "", result: Any = None):
    """Update task status and notify connected clients."""
    task_status[task_id] = {
        "status": status,
        "progress": progress,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    if result is not None:
        task_results[task_id] = result
    
    # Notify connected WebSocket clients
    if task_id in websocket_connections:
        for ws in websocket_connections[task_id]:
            try:
                await ws.send_json({
                    "task_id": task_id,
                    "status": status,
                    "progress": progress,
                    "message": message,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to send WebSocket update: {e}")

# Background tasks
async def predict_hardware_task(task_id: str, request: HardwareRequest):
    """Background task for hardware prediction."""
    try:
        # Create progress updates
        await update_task_status(task_id, "running", 0.1, "Initializing hardware prediction")
        
        # Get adapter
        adapter = get_hardware_adapter()
        if adapter is None:
            await update_task_status(task_id, "failed", 0, "Hardware adapter not available")
            return
        
        # Convert available hardware from enum to string if provided
        available_hardware = None
        if request.available_hardware:
            available_hardware = [hw.value for hw in request.available_hardware]
        
        # Predict optimal hardware
        await update_task_status(task_id, "running", 0.3, "Predicting optimal hardware")
        result = adapter.predict_optimal_hardware(
            model_name=request.model_name,
            model_family=request.model_family,
            batch_size=request.batch_size,
            sequence_length=request.sequence_length,
            mode=request.mode.value,
            precision=request.precision.value,
            available_hardware=available_hardware
        )
        
        await update_task_status(task_id, "running", 0.6, "Hardware prediction completed")
        
        # Predict performance if requested
        if request.predict_performance:
            await update_task_status(task_id, "running", 0.7, "Predicting performance")
            
            primary_hw = result.get("primary_recommendation")
            if primary_hw:
                performance = adapter.predict_performance(
                    model_name=request.model_name,
                    model_family=result.get("model_family", request.model_family),
                    hardware=primary_hw,
                    batch_size=request.batch_size,
                    sequence_length=request.sequence_length,
                    mode=request.mode.value,
                    precision=request.precision.value
                )
                
                # Add performance to result
                if primary_hw in performance.get("predictions", {}):
                    result["performance"] = performance["predictions"][primary_hw]
        
        await update_task_status(task_id, "completed", 1.0, "Task completed successfully", result)
    
    except Exception as e:
        logger.error(f"Error in hardware prediction task: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await update_task_status(task_id, "failed", 0, f"Task failed: {str(e)}")

async def predict_performance_task(task_id: str, request: PerformanceRequest):
    """Background task for performance prediction."""
    try:
        # Create progress updates
        await update_task_status(task_id, "running", 0.1, "Initializing performance prediction")
        
        # Get adapter
        adapter = get_hardware_adapter()
        if adapter is None:
            await update_task_status(task_id, "failed", 0, "Hardware adapter not available")
            return
        
        # Convert hardware to list of strings if needed
        if isinstance(request.hardware, list):
            hardware = [hw.value for hw in request.hardware]
        else:
            hardware = request.hardware.value
        
        # Predict performance
        await update_task_status(task_id, "running", 0.5, "Predicting performance")
        result = adapter.predict_performance(
            model_name=request.model_name,
            model_family=request.model_family,
            hardware=hardware,
            batch_size=request.batch_size,
            sequence_length=request.sequence_length,
            mode=request.mode.value,
            precision=request.precision.value
        )
        
        await update_task_status(task_id, "completed", 1.0, "Task completed successfully", result)
    
    except Exception as e:
        logger.error(f"Error in performance prediction task: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await update_task_status(task_id, "failed", 0, f"Task failed: {str(e)}")

async def record_measurement_task(task_id: str, request: MeasurementRequest):
    """Background task for recording measurements."""
    try:
        # Create progress updates
        await update_task_status(task_id, "running", 0.1, "Initializing measurement recording")
        
        # Get adapter
        adapter = get_hardware_adapter()
        if adapter is None:
            await update_task_status(task_id, "failed", 0, "Hardware adapter not available")
            return
        
        # Record measurement
        await update_task_status(task_id, "running", 0.5, "Recording measurement")
        result = adapter.record_actual_performance(
            model_name=request.model_name,
            model_family=request.model_family,
            hardware_platform=request.hardware_platform.value,
            batch_size=request.batch_size,
            sequence_length=request.sequence_length,
            precision=request.precision.value,
            mode=request.mode.value,
            throughput=request.throughput,
            latency=request.latency,
            memory_usage=request.memory_usage,
            prediction_id=request.prediction_id,
            measurement_source=request.source
        )
        
        await update_task_status(task_id, "completed", 1.0, "Task completed successfully", result)
    
    except Exception as e:
        logger.error(f"Error in measurement recording task: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await update_task_status(task_id, "failed", 0, f"Task failed: {str(e)}")

async def analyze_predictions_task(task_id: str, request: AnalysisRequest):
    """Background task for analyzing predictions."""
    try:
        # Create progress updates
        await update_task_status(task_id, "running", 0.1, "Initializing prediction analysis")
        
        # Get repository
        repo = get_repository()
        if repo is None:
            await update_task_status(task_id, "failed", 0, "Repository not available")
            return
        
        # Parse time range if provided
        start_time = None
        end_time = None
        
        if request.days:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=request.days)
        
        # Convert hardware platform to string if provided
        hardware_platform = None
        if request.hardware_platform:
            hardware_platform = request.hardware_platform.value
        
        # Get accuracy stats
        await update_task_status(task_id, "running", 0.5, "Analyzing predictions")
        stats = repo.get_prediction_accuracy_stats(
            model_name=request.model_name,
            hardware_platform=hardware_platform,
            metric=request.metric,
            start_time=start_time,
            end_time=end_time
        )
        
        await update_task_status(task_id, "completed", 1.0, "Task completed successfully", stats)
    
    except Exception as e:
        logger.error(f"Error in prediction analysis task: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await update_task_status(task_id, "failed", 0, f"Task failed: {str(e)}")

async def record_feedback_task(task_id: str, request: FeedbackRequest):
    """Background task for recording feedback."""
    try:
        # Create progress updates
        await update_task_status(task_id, "running", 0.1, "Initializing feedback recording")
        
        # Get adapter
        adapter = get_hardware_adapter()
        if adapter is None:
            await update_task_status(task_id, "failed", 0, "Hardware adapter not available")
            return
        
        # Record feedback
        await update_task_status(task_id, "running", 0.5, "Recording feedback")
        result = adapter.record_recommendation_feedback(
            recommendation_id=request.recommendation_id,
            was_accepted=request.accepted,
            user_feedback=request.feedback
        )
        
        await update_task_status(task_id, "completed", 1.0, "Task completed successfully", {
            "success": result,
            "recommendation_id": request.recommendation_id
        })
    
    except Exception as e:
        logger.error(f"Error in feedback recording task: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await update_task_status(task_id, "failed", 0, f"Task failed: {str(e)}")

async def store_ml_model_task(task_id: str, request: MLModelRequest):
    """Background task for storing ML model."""
    try:
        # Create progress updates
        await update_task_status(task_id, "running", 0.1, "Initializing ML model storage")
        
        # Get adapter
        adapter = get_ml_adapter()
        if adapter is None:
            await update_task_status(task_id, "failed", 0, "ML adapter not available")
            return
        
        # Load serialized model
        await update_task_status(task_id, "running", 0.3, "Loading serialized model")
        import joblib
        model = joblib.load(request.serialized_model_path)
        
        # Store model
        await update_task_status(task_id, "running", 0.6, "Storing ML model")
        model_id = adapter.store_model(
            model=model,
            model_type=request.model_type,
            target_metric=request.target_metric,
            hardware_platform=request.hardware_platform.value,
            model_family=request.model_family,
            features=request.features,
            training_score=request.training_score,
            validation_score=request.validation_score,
            test_score=request.test_score
        )
        
        await update_task_status(task_id, "completed", 1.0, "Task completed successfully", {
            "model_id": model_id
        })
    
    except Exception as e:
        logger.error(f"Error in ML model storage task: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await update_task_status(task_id, "failed", 0, f"Task failed: {str(e)}")

async def generate_matrix_task(task_id: str, request: MatrixGenerationRequest):
    """Background task for matrix generation."""
    try:
        # Create progress updates
        await update_task_status(task_id, "running", 0.1, "Initializing matrix generation")
        
        # Get adapter
        adapter = get_ml_adapter()
        if adapter is None:
            await update_task_status(task_id, "failed", 0, "ML adapter not available")
            return
        
        # Convert hardware platforms to strings
        hardware_platforms = [hw.value for hw in request.hardware_platforms]
        
        # Convert precision options to strings
        precision_options = [p.value for p in request.precision_options]
        
        # Generate matrix
        await update_task_status(task_id, "running", 0.5, "Generating prediction matrix")
        matrix = adapter.generate_prediction_matrix(
            model_configs=request.model_configs,
            hardware_platforms=hardware_platforms,
            batch_sizes=request.batch_sizes,
            precision_options=precision_options,
            mode=request.mode.value
        )
        
        await update_task_status(task_id, "completed", 1.0, "Task completed successfully", matrix)
    
    except Exception as e:
        logger.error(f"Error in matrix generation task: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await update_task_status(task_id, "failed", 0, f"Task failed: {str(e)}")

async def generate_sample_data_task(task_id: str, request: SampleDataRequest):
    """Background task for generating sample data."""
    try:
        # Create progress updates
        await update_task_status(task_id, "running", 0.1, "Initializing sample data generation")
        
        # Get repository
        repo = get_repository()
        if repo is None:
            await update_task_status(task_id, "failed", 0, "Repository not available")
            return
        
        # Generate sample data
        await update_task_status(task_id, "running", 0.3, "Generating sample data")
        repo.generate_sample_data(num_models=request.num_models)
        
        await update_task_status(task_id, "completed", 1.0, "Task completed successfully", {
            "num_models": request.num_models
        })
    
    except Exception as e:
        logger.error(f"Error in sample data generation task: {e}")
        import traceback
        logger.error(traceback.format_exc())
        await update_task_status(task_id, "failed", 0, f"Task failed: {str(e)}")

# API endpoints
@router.post("/predict-hardware", summary="Predict optimal hardware for a model")
async def predict_hardware(request: HardwareRequest, background_tasks: BackgroundTasks):
    """
    Predict the optimal hardware for a given model and configuration.
    Optionally predicts performance metrics on the recommended hardware.
    Returns a task ID that can be used to track the prediction progress.
    """
    if not DUCKDB_AVAILABLE:
        raise HTTPException(status_code=500, detail="DuckDB components not available")
    
    # Generate task ID
    task_id = f"hardware-{uuid.uuid4()}"
    
    # Initialize task status
    task_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Task created",
        "timestamp": datetime.now().isoformat()
    }
    
    # Start background task
    background_tasks.add_task(predict_hardware_task, task_id, request)
    
    return {
        "task_id": task_id,
        "status": "pending"
    }

@router.post("/predict-performance", summary="Predict performance for a model on specified hardware")
async def predict_performance(request: PerformanceRequest, background_tasks: BackgroundTasks):
    """
    Predict performance metrics for a model on specified hardware.
    Returns a task ID that can be used to track the prediction progress.
    """
    if not DUCKDB_AVAILABLE:
        raise HTTPException(status_code=500, detail="DuckDB components not available")
    
    # Generate task ID
    task_id = f"performance-{uuid.uuid4()}"
    
    # Initialize task status
    task_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Task created",
        "timestamp": datetime.now().isoformat()
    }
    
    # Start background task
    background_tasks.add_task(predict_performance_task, task_id, request)
    
    return {
        "task_id": task_id,
        "status": "pending"
    }

@router.post("/record-measurement", summary="Record an actual performance measurement")
async def record_measurement(request: MeasurementRequest, background_tasks: BackgroundTasks):
    """
    Record an actual performance measurement and compare with predictions if available.
    Returns a task ID that can be used to track the recording progress.
    """
    if not DUCKDB_AVAILABLE:
        raise HTTPException(status_code=500, detail="DuckDB components not available")
    
    # Generate task ID
    task_id = f"measurement-{uuid.uuid4()}"
    
    # Initialize task status
    task_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Task created",
        "timestamp": datetime.now().isoformat()
    }
    
    # Start background task
    background_tasks.add_task(record_measurement_task, task_id, request)
    
    return {
        "task_id": task_id,
        "status": "pending"
    }

@router.post("/analyze-predictions", summary="Analyze prediction accuracy")
async def analyze_predictions(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze the accuracy of performance predictions compared to actual measurements.
    Returns a task ID that can be used to track the analysis progress.
    """
    if not DUCKDB_AVAILABLE:
        raise HTTPException(status_code=500, detail="DuckDB components not available")
    
    # Generate task ID
    task_id = f"analysis-{uuid.uuid4()}"
    
    # Initialize task status
    task_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Task created",
        "timestamp": datetime.now().isoformat()
    }
    
    # Start background task
    background_tasks.add_task(analyze_predictions_task, task_id, request)
    
    return {
        "task_id": task_id,
        "status": "pending"
    }

@router.post("/record-feedback", summary="Record feedback for a hardware recommendation")
async def record_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """
    Record feedback for a hardware recommendation, indicating whether it was accepted.
    Returns a task ID that can be used to track the recording progress.
    """
    if not DUCKDB_AVAILABLE:
        raise HTTPException(status_code=500, detail="DuckDB components not available")
    
    # Generate task ID
    task_id = f"feedback-{uuid.uuid4()}"
    
    # Initialize task status
    task_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Task created",
        "timestamp": datetime.now().isoformat()
    }
    
    # Start background task
    background_tasks.add_task(record_feedback_task, task_id, request)
    
    return {
        "task_id": task_id,
        "status": "pending"
    }

@router.post("/store-model", summary="Store a machine learning model for performance prediction")
async def store_model(request: MLModelRequest, background_tasks: BackgroundTasks):
    """
    Store a trained machine learning model for performance prediction.
    Returns a task ID that can be used to track the storage progress.
    """
    if not DUCKDB_AVAILABLE:
        raise HTTPException(status_code=500, detail="DuckDB components not available")
    
    # Generate task ID
    task_id = f"model-{uuid.uuid4()}"
    
    # Initialize task status
    task_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Task created",
        "timestamp": datetime.now().isoformat()
    }
    
    # Start background task
    background_tasks.add_task(store_ml_model_task, task_id, request)
    
    return {
        "task_id": task_id,
        "status": "pending"
    }

@router.post("/generate-matrix", summary="Generate a prediction matrix for various models and hardware")
async def generate_matrix(request: MatrixGenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate a prediction matrix for various models and hardware configurations.
    Returns a task ID that can be used to track the generation progress.
    """
    if not DUCKDB_AVAILABLE:
        raise HTTPException(status_code=500, detail="DuckDB components not available")
    
    # Generate task ID
    task_id = f"matrix-{uuid.uuid4()}"
    
    # Initialize task status
    task_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Task created",
        "timestamp": datetime.now().isoformat()
    }
    
    # Start background task
    background_tasks.add_task(generate_matrix_task, task_id, request)
    
    return {
        "task_id": task_id,
        "status": "pending"
    }

@router.post("/generate-sample-data", summary="Generate sample data for testing")
async def generate_sample_data(request: SampleDataRequest, background_tasks: BackgroundTasks):
    """
    Generate sample data for testing the Predictive Performance Modeling System.
    Returns a task ID that can be used to track the generation progress.
    """
    if not DUCKDB_AVAILABLE:
        raise HTTPException(status_code=500, detail="DuckDB components not available")
    
    # Generate task ID
    task_id = f"sample-{uuid.uuid4()}"
    
    # Initialize task status
    task_status[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Task created",
        "timestamp": datetime.now().isoformat()
    }
    
    # Start background task
    background_tasks.add_task(generate_sample_data_task, task_id, request)
    
    return {
        "task_id": task_id,
        "status": "pending"
    }

@router.get("/task/{task_id}", summary="Get task status")
async def get_task_status(task_id: str = Path(..., description="The task ID")):
    """
    Get the status of a task by ID.
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = task_status[task_id]
    
    # Include result if task is completed
    if status["status"] == "completed" and task_id in task_results:
        status["result"] = task_results[task_id]
    
    return status

@router.get("/recommendations", summary="List hardware recommendations")
async def list_recommendations(
    model_name: Optional[str] = None,
    model_family: Optional[str] = None,
    hardware: Optional[str] = None,
    accepted: Optional[bool] = None,
    days: Optional[int] = None,
    limit: int = 10
):
    """
    List hardware recommendations based on filters.
    """
    if not DUCKDB_AVAILABLE:
        raise HTTPException(status_code=500, detail="DuckDB components not available")
    
    repo = get_repository()
    
    # Parse time range if provided
    start_time = None
    end_time = None
    
    if days:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
    
    # Get recommendations
    recommendations = repo.get_recommendations(
        model_name=model_name,
        model_family=model_family,
        primary_recommendation=hardware,
        was_accepted=accepted,
        start_time=start_time,
        end_time=end_time,
        limit=limit
    )
    
    return {
        "count": len(recommendations),
        "recommendations": recommendations
    }

@router.get("/models", summary="List ML models")
async def list_models(
    model_type: Optional[str] = None,
    target_metric: Optional[str] = None,
    hardware: Optional[str] = None,
    model_family: Optional[str] = None,
    limit: int = 10
):
    """
    List ML models based on filters.
    """
    if not DUCKDB_AVAILABLE:
        raise HTTPException(status_code=500, detail="DuckDB components not available")
    
    repo = get_repository()
    
    # Get models
    models = repo.get_prediction_models(
        model_type=model_type,
        target_metric=target_metric,
        hardware_platform=hardware,
        model_family=model_family,
        limit=limit
    )
    
    # Remove serialized model from response (too large)
    for model in models:
        if "serialized_model" in model:
            model["serialized_model"] = "<binary data>"
    
    return {
        "count": len(models),
        "models": models
    }

@router.get("/predictions", summary="List performance predictions")
async def list_predictions(
    model_name: Optional[str] = None,
    model_family: Optional[str] = None,
    hardware: Optional[str] = None,
    batch_size: Optional[int] = None,
    limit: int = 10
):
    """
    List performance predictions based on filters.
    """
    if not DUCKDB_AVAILABLE:
        raise HTTPException(status_code=500, detail="DuckDB components not available")
    
    repo = get_repository()
    
    # Get predictions
    predictions = repo.get_predictions(
        model_name=model_name,
        model_family=model_family,
        hardware_platform=hardware,
        batch_size=batch_size,
        limit=limit
    )
    
    return {
        "count": len(predictions),
        "predictions": predictions
    }

@router.get("/measurements", summary="List performance measurements")
async def list_measurements(
    model_name: Optional[str] = None,
    model_family: Optional[str] = None,
    hardware: Optional[str] = None,
    batch_size: Optional[int] = None,
    days: Optional[int] = None,
    limit: int = 10
):
    """
    List performance measurements based on filters.
    """
    if not DUCKDB_AVAILABLE:
        raise HTTPException(status_code=500, detail="DuckDB components not available")
    
    repo = get_repository()
    
    # Parse time range if provided
    start_time = None
    end_time = None
    
    if days:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
    
    # Get measurements
    measurements = repo.get_measurements(
        model_name=model_name,
        model_family=model_family,
        hardware_platform=hardware,
        batch_size=batch_size,
        start_time=start_time,
        end_time=end_time,
        limit=limit
    )
    
    return {
        "count": len(measurements),
        "measurements": measurements
    }

@router.get("/feature-importance", summary="List feature importance for prediction models")
async def list_feature_importance(
    model_id: Optional[str] = None,
    feature_name: Optional[str] = None,
    method: Optional[str] = None,
    limit: int = 100
):
    """
    List feature importance records based on filters.
    """
    if not DUCKDB_AVAILABLE:
        raise HTTPException(status_code=500, detail="DuckDB components not available")
    
    repo = get_repository()
    
    # Get feature importance
    importance = repo.get_feature_importance(
        model_id=model_id,
        feature_name=feature_name,
        method=method,
        limit=limit
    )
    
    return {
        "count": len(importance),
        "feature_importance": importance
    }

@router.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for tracking task progress in real-time.
    """
    await websocket.accept()
    
    # Register connection
    if task_id not in websocket_connections:
        websocket_connections[task_id] = []
    
    websocket_connections[task_id].append(websocket)
    
    try:
        # Send initial status if available
        if task_id in task_status:
            await websocket.send_json({
                "task_id": task_id,
                **task_status[task_id]
            })
        
        # Keep connection open for updates
        while True:
            await asyncio.sleep(1)
    
    except Exception as e:
        logger.warning(f"WebSocket connection closed: {e}")
    
    finally:
        # Remove connection
        if task_id in websocket_connections and websocket in websocket_connections[task_id]:
            websocket_connections[task_id].remove(websocket)

# Add router to app
app.include_router(router)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    try:
        # Initialize repository
        get_repository()
        
        # Initialize adapters
        get_hardware_adapter()
        get_ml_adapter()
        
        logger.info("API server initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing API server: {e}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global repository
    if repository:
        try:
            repository.close()
            logger.info("Repository connection closed")
        except Exception as e:
            logger.error(f"Error closing repository: {e}")

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    status = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "repository": repository is not None,
            "hardware_adapter": hardware_adapter is not None,
            "ml_adapter": ml_adapter is not None
        },
        "duckdb_available": DUCKDB_AVAILABLE,
        "hardware_model_predictor_available": HARDWARE_MODEL_PREDICTOR_AVAILABLE
    }
    
    return status

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Predictive Performance API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind server")
    parser.add_argument("--port", type=int, default=8500, help="Port to bind server")
    parser.add_argument("--db", type=str, help="Path to DuckDB database")
    parser.add_argument("--benchmark-dir", type=str, help="Path to benchmark results directory")
    parser.add_argument("--benchmark-db", type=str, help="Path to benchmark database")
    
    args = parser.parse_args()
    
    # Set environment variables for component configuration
    if args.db:
        os.environ["PREDICTIVE_PERFORMANCE_DB"] = args.db
    
    if args.benchmark_dir:
        os.environ["BENCHMARK_DIR"] = args.benchmark_dir
    
    if args.benchmark_db:
        os.environ["BENCHMARK_DB"] = args.benchmark_db
    
    # Start server
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()