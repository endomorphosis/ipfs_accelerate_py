#!/usr/bin/env python3
"""
Hardware Optimization Integration for Unified API Server

This module integrates the Hardware Optimization Recommendation system with the Unified API Server.
It provides REST API endpoints for accessing hardware-specific optimization recommendations.
"""

import os
import sys
import json
import logging
import asyncio
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hardware_optimization_integration")

# Import optimization components
try:
    from test.optimization_recommendation.optimization_client import OptimizationClient
    from test.optimization_recommendation.hardware_optimization_analyzer import HardwareOptimizationAnalyzer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Hardware Optimization components not available")
    OPTIMIZATION_AVAILABLE = False

class HardwareOptimizationIntegration:
    """Integration class for Hardware Optimization with Unified API Server."""
    
    def __init__(
        self,
        benchmark_db_path: str = "benchmark_db.duckdb",
        api_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the hardware optimization integration.
        
        Args:
            benchmark_db_path: Path to benchmark database
            api_url: API base URL
            api_key: Optional API key
            config: Optional configuration
        """
        self.benchmark_db_path = benchmark_db_path
        self.api_url = api_url
        self.api_key = api_key
        self.config = config or {}
        
        # Initialize optimization client
        self.client = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.client = OptimizationClient(
                    benchmark_db_path=benchmark_db_path,
                    predictive_api_url=api_url,
                    api_key=api_key,
                    config=config
                )
                logger.info("Hardware Optimization client initialized")
            except Exception as e:
                logger.error(f"Error initializing Hardware Optimization client: {e}")
        
        # Task storage
        self.tasks = {}
    
    def register_routes(self, app):
        """
        Register API routes with the FastAPI app.
        
        Args:
            app: FastAPI application
        """
        if not OPTIMIZATION_AVAILABLE:
            logger.warning("Hardware Optimization components not available, skipping route registration")
            return
        
        try:
            from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
            from pydantic import BaseModel, Field
            
            # Create router
            router = APIRouter(prefix="/api/hardware-optimization", tags=["Hardware Optimization"])
            
            # Define models
            class HardwareOptimizationRequest(BaseModel):
                model_name: str = Field(..., description="Name of the model")
                hardware_platform: str = Field(..., description="Hardware platform")
                model_family: Optional[str] = Field(None, description="Optional model family")
                batch_size: Optional[int] = Field(None, description="Optional batch size")
                current_precision: Optional[str] = Field(None, description="Optional current precision")
            
            class PerformanceAnalysisRequest(BaseModel):
                model_name: str = Field(..., description="Name of the model")
                hardware_platform: str = Field(..., description="Hardware platform")
                batch_size: Optional[int] = Field(None, description="Optional batch size filter")
                days: int = Field(90, description="Number of days to look back")
                limit: int = Field(100, description="Maximum number of records to analyze")
            
            class ReportRequest(BaseModel):
                model_names: List[str] = Field(..., description="List of model names")
                hardware_platforms: List[str] = Field(..., description="List of hardware platforms")
                batch_size: Optional[int] = Field(None, description="Optional batch size filter")
                current_precision: Optional[str] = Field(None, description="Optional current precision")
            
            class TaskResponse(BaseModel):
                task_id: str = Field(..., description="Task ID")
                status: str = Field("pending", description="Task status")
                created_at: str = Field(..., description="Task creation timestamp")
                
            # Background task handler
            async def run_in_background(func, task_id, *args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    self.tasks[task_id] = {
                        "status": "completed",
                        "result": result,
                        "completed_at": datetime.datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.error(f"Error in background task {task_id}: {e}")
                    self.tasks[task_id] = {
                        "status": "failed",
                        "error": str(e),
                        "completed_at": datetime.datetime.now().isoformat()
                    }
            
            # Routes
            @router.post("/recommendations", response_model=TaskResponse, summary="Get hardware optimization recommendations")
            async def get_recommendations(
                request: HardwareOptimizationRequest,
                background_tasks: BackgroundTasks
            ):
                """
                Get optimization recommendations for a specific model and hardware platform.
                
                This endpoint analyzes historical performance data and generates
                hardware-specific optimization recommendations.
                """
                # Create task
                task_id = f"optimization-{datetime.datetime.now().timestamp()}"
                self.tasks[task_id] = {
                    "status": "pending",
                    "created_at": datetime.datetime.now().isoformat()
                }
                
                # Schedule background task
                background_tasks.add_task(
                    run_in_background,
                    self.client.get_recommendations,
                    task_id,
                    model_name=request.model_name,
                    hardware_platform=request.hardware_platform,
                    model_family=request.model_family,
                    batch_size=request.batch_size,
                    current_precision=request.current_precision
                )
                
                return {
                    "task_id": task_id,
                    "status": "pending",
                    "created_at": self.tasks[task_id]["created_at"]
                }
            
            @router.post("/analyze-performance", response_model=TaskResponse, summary="Analyze performance data")
            async def analyze_performance(
                request: PerformanceAnalysisRequest,
                background_tasks: BackgroundTasks
            ):
                """
                Analyze performance data for a specific model and hardware platform.
                
                This endpoint processes historical performance data to identify patterns
                and optimization opportunities.
                """
                # Create task
                task_id = f"analysis-{datetime.datetime.now().timestamp()}"
                self.tasks[task_id] = {
                    "status": "pending",
                    "created_at": datetime.datetime.now().isoformat()
                }
                
                # Schedule background task
                background_tasks.add_task(
                    run_in_background,
                    self.client.analyze_performance,
                    task_id,
                    model_name=request.model_name,
                    hardware_platform=request.hardware_platform,
                    batch_size=request.batch_size,
                    days=request.days,
                    limit=request.limit
                )
                
                return {
                    "task_id": task_id,
                    "status": "pending",
                    "created_at": self.tasks[task_id]["created_at"]
                }
            
            @router.post("/generate-report", response_model=TaskResponse, summary="Generate comprehensive report")
            async def generate_report(
                request: ReportRequest,
                background_tasks: BackgroundTasks
            ):
                """
                Generate a comprehensive optimization report for multiple models and hardware platforms.
                
                This endpoint analyzes multiple model-hardware combinations and provides
                prioritized optimization recommendations.
                """
                # Create task
                task_id = f"report-{datetime.datetime.now().timestamp()}"
                self.tasks[task_id] = {
                    "status": "pending",
                    "created_at": datetime.datetime.now().isoformat()
                }
                
                # Schedule background task
                background_tasks.add_task(
                    run_in_background,
                    self.client.generate_report,
                    task_id,
                    model_names=request.model_names,
                    hardware_platforms=request.hardware_platforms,
                    batch_size=request.batch_size,
                    current_precision=request.current_precision
                )
                
                return {
                    "task_id": task_id,
                    "status": "pending",
                    "created_at": self.tasks[task_id]["created_at"]
                }
            
            @router.get("/strategies/{hardware_platform}", summary="Get available optimization strategies")
            def get_strategies(hardware_platform: str):
                """
                Get available optimization strategies for a specific hardware platform.
                
                This endpoint returns hardware-specific optimization strategies
                that can be applied to models.
                """
                strategies = self.client.get_available_strategies(hardware_platform)
                return {"hardware_platform": hardware_platform, "strategies": strategies}
            
            @router.get("/task/{task_id}", summary="Get task status and result")
            def get_task_status(task_id: str):
                """
                Get the status and result of a task.
                
                This endpoint returns the current status of a task and its result
                if completed.
                """
                if task_id not in self.tasks:
                    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
                
                return self.tasks[task_id]
            
            # Register router with app
            app.include_router(router)
            logger.info("Hardware Optimization API routes registered")
            
        except ImportError as e:
            logger.error(f"Error importing FastAPI components: {e}")
        except Exception as e:
            logger.error(f"Error registering routes: {e}")
    
    def close(self):
        """Close connections."""
        if self.client:
            self.client.close()

# Factory function for integration
def create_integration(
    benchmark_db_path: str = "benchmark_db.duckdb",
    api_url: str = "http://localhost:8080",
    api_key: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> HardwareOptimizationIntegration:
    """
    Create a hardware optimization integration instance.
    
    Args:
        benchmark_db_path: Path to benchmark database
        api_url: API base URL
        api_key: Optional API key
        config: Optional configuration
        
    Returns:
        Hardware optimization integration instance
    """
    return HardwareOptimizationIntegration(
        benchmark_db_path=benchmark_db_path,
        api_url=api_url,
        api_key=api_key,
        config=config
    )