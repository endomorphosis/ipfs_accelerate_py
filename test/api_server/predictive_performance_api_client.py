#!/usr/bin/env python3
"""
Client for the Predictive Performance API.

This module provides a client for interacting with the Predictive Performance API,
including both synchronous and asynchronous implementations.
"""

import os
import sys
import json
import time
import logging
import argparse
import anyio
import websockets
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Add parent directory to path to allow importing project modules
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("predictive_performance_client")

# Try to import requests
try:
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    REQUESTS_AVAILABLE = True
except ImportError:
    logger.warning("requests package not available, synchronous client will not work")
    REQUESTS_AVAILABLE = False

# Try to import aiohttp
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    logger.warning("aiohttp package not available, asynchronous client will not work")
    AIOHTTP_AVAILABLE = False

class ApiClient:
    """
    Synchronous client for the Predictive Performance API.
    """
    
    def __init__(self, base_url: str, verify_ssl: bool = True, timeout: int = 30):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL for the API
            verify_ssl: Whether to verify SSL certificates
            timeout: Request timeout in seconds
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests package is required for synchronous client")
        
        self.base_url = base_url.rstrip("/")
        self.verify_ssl = verify_ssl
        self.timeout = timeout
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data for POST/PUT
            
        Returns:
            Response data as a dictionary
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.lower() == "get":
                response = requests.get(url, params=data, verify=self.verify_ssl, timeout=self.timeout)
            elif method.lower() == "post":
                response = requests.post(url, json=data, verify=self.verify_ssl, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_data = e.response.json()
                    logger.error(f"Response error: {error_data}")
                except:
                    logger.error(f"Response text: {e.response.text}")
            raise
    
    def predict_hardware(self, 
                       model_name: str,
                       model_family: Optional[str] = None,
                       batch_size: int = 1,
                       sequence_length: int = 128,
                       mode: str = "inference",
                       precision: str = "fp32",
                       available_hardware: Optional[List[str]] = None,
                       predict_performance: bool = False) -> Dict[str, Any]:
        """
        Predict optimal hardware for a model.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family/category
            batch_size: Batch size
            sequence_length: Sequence length
            mode: "inference" or "training"
            precision: Precision to use (fp32, fp16, int8)
            available_hardware: Optional list of available hardware types
            predict_performance: Whether to predict performance on recommended hardware
            
        Returns:
            Dictionary with task ID and status
        """
        data = {
            "model_name": model_name,
            "model_family": model_family,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mode": mode,
            "precision": precision,
            "available_hardware": available_hardware,
            "predict_performance": predict_performance
        }
        
        return self._make_request("post", "/api/predictive-performance/predict-hardware", data)
    
    def predict_performance(self,
                          model_name: str,
                          hardware: Union[str, List[str]],
                          model_family: Optional[str] = None,
                          batch_size: int = 1,
                          sequence_length: int = 128,
                          mode: str = "inference",
                          precision: str = "fp32") -> Dict[str, Any]:
        """
        Predict performance for a model on specified hardware.
        
        Args:
            model_name: Name of the model
            hardware: Hardware type or list of hardware types
            model_family: Optional model family/category
            batch_size: Batch size
            sequence_length: Sequence length
            mode: "inference" or "training"
            precision: Precision to use (fp32, fp16, int8)
            
        Returns:
            Dictionary with task ID and status
        """
        data = {
            "model_name": model_name,
            "model_family": model_family,
            "hardware": hardware,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mode": mode,
            "precision": precision
        }
        
        return self._make_request("post", "/api/predictive-performance/predict-performance", data)
    
    def record_measurement(self,
                         model_name: str,
                         hardware_platform: str,
                         model_family: Optional[str] = None,
                         batch_size: int = 1,
                         sequence_length: int = 128,
                         precision: str = "fp32",
                         mode: str = "inference",
                         throughput: Optional[float] = None,
                         latency: Optional[float] = None,
                         memory_usage: Optional[float] = None,
                         prediction_id: Optional[str] = None,
                         source: str = "api") -> Dict[str, Any]:
        """
        Record an actual performance measurement.
        
        Args:
            model_name: Name of the model
            hardware_platform: Hardware platform
            model_family: Optional model family/category
            batch_size: Batch size
            sequence_length: Sequence length
            precision: Precision (fp32, fp16, int8)
            mode: "inference" or "training"
            throughput: Optional throughput measurement
            latency: Optional latency measurement
            memory_usage: Optional memory usage measurement
            prediction_id: Optional ID of a previous prediction to compare with
            source: Source of the measurement
            
        Returns:
            Dictionary with task ID and status
        """
        data = {
            "model_name": model_name,
            "model_family": model_family,
            "hardware_platform": hardware_platform,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "precision": precision,
            "mode": mode,
            "throughput": throughput,
            "latency": latency,
            "memory_usage": memory_usage,
            "prediction_id": prediction_id,
            "source": source
        }
        
        return self._make_request("post", "/api/predictive-performance/record-measurement", data)
    
    def analyze_predictions(self,
                          model_name: Optional[str] = None,
                          hardware_platform: Optional[str] = None,
                          metric: Optional[str] = None,
                          days: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze prediction accuracy.
        
        Args:
            model_name: Optional model name filter
            hardware_platform: Optional hardware platform filter
            metric: Optional metric filter
            days: Optional number of days to analyze
            
        Returns:
            Dictionary with task ID and status
        """
        data = {
            "model_name": model_name,
            "hardware_platform": hardware_platform,
            "metric": metric,
            "days": days
        }
        
        return self._make_request("post", "/api/predictive-performance/analyze-predictions", data)
    
    def record_feedback(self,
                      recommendation_id: str,
                      accepted: bool,
                      feedback: Optional[str] = None) -> Dict[str, Any]:
        """
        Record feedback for a hardware recommendation.
        
        Args:
            recommendation_id: ID of the recommendation
            accepted: Whether the recommendation was accepted
            feedback: Optional feedback text
            
        Returns:
            Dictionary with task ID and status
        """
        data = {
            "recommendation_id": recommendation_id,
            "accepted": accepted,
            "feedback": feedback
        }
        
        return self._make_request("post", "/api/predictive-performance/record-feedback", data)
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary with task status
        """
        return self._make_request("get", f"/api/predictive-performance/task/{task_id}")
    
    def wait_for_task(self, task_id: str, poll_interval: float = 0.5, timeout: float = 60) -> Dict[str, Any]:
        """
        Wait for a task to complete.
        
        Args:
            task_id: ID of the task
            poll_interval: Polling interval in seconds
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with task result
        """
        start_time = time.time()
        
        while True:
            status = self.get_task_status(task_id)
            
            if status["status"] == "completed":
                return status
            
            if status["status"] == "failed":
                raise RuntimeError(f"Task failed: {status.get('message', 'Unknown error')}")
            
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task timed out after {timeout} seconds")
            
            time.sleep(poll_interval)
    
    def list_recommendations(self,
                           model_name: Optional[str] = None,
                           model_family: Optional[str] = None,
                           hardware: Optional[str] = None,
                           accepted: Optional[bool] = None,
                           days: Optional[int] = None,
                           limit: int = 10) -> Dict[str, Any]:
        """
        List hardware recommendations.
        
        Args:
            model_name: Optional model name filter
            model_family: Optional model family filter
            hardware: Optional hardware filter
            accepted: Optional filter for accepted recommendations
            days: Optional number of days to include
            limit: Maximum number of results
            
        Returns:
            Dictionary with recommendations
        """
        params = {
            "model_name": model_name,
            "model_family": model_family,
            "hardware": hardware,
            "accepted": accepted,
            "days": days,
            "limit": limit
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        return self._make_request("get", "/api/predictive-performance/recommendations", params)
    
    def list_models(self,
                  model_type: Optional[str] = None,
                  target_metric: Optional[str] = None,
                  hardware: Optional[str] = None,
                  model_family: Optional[str] = None,
                  limit: int = 10) -> Dict[str, Any]:
        """
        List ML models.
        
        Args:
            model_type: Optional model type filter
            target_metric: Optional target metric filter
            hardware: Optional hardware filter
            model_family: Optional model family filter
            limit: Maximum number of results
            
        Returns:
            Dictionary with models
        """
        params = {
            "model_type": model_type,
            "target_metric": target_metric,
            "hardware": hardware,
            "model_family": model_family,
            "limit": limit
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        return self._make_request("get", "/api/predictive-performance/models", params)
    
    def list_predictions(self,
                       model_name: Optional[str] = None,
                       model_family: Optional[str] = None,
                       hardware: Optional[str] = None,
                       batch_size: Optional[int] = None,
                       limit: int = 10) -> Dict[str, Any]:
        """
        List performance predictions.
        
        Args:
            model_name: Optional model name filter
            model_family: Optional model family filter
            hardware: Optional hardware filter
            batch_size: Optional batch size filter
            limit: Maximum number of results
            
        Returns:
            Dictionary with predictions
        """
        params = {
            "model_name": model_name,
            "model_family": model_family,
            "hardware": hardware,
            "batch_size": batch_size,
            "limit": limit
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        return self._make_request("get", "/api/predictive-performance/predictions", params)
    
    def list_measurements(self,
                        model_name: Optional[str] = None,
                        model_family: Optional[str] = None,
                        hardware: Optional[str] = None,
                        batch_size: Optional[int] = None,
                        days: Optional[int] = None,
                        limit: int = 10) -> Dict[str, Any]:
        """
        List performance measurements.
        
        Args:
            model_name: Optional model name filter
            model_family: Optional model family filter
            hardware: Optional hardware filter
            batch_size: Optional batch size filter
            days: Optional number of days to include
            limit: Maximum number of results
            
        Returns:
            Dictionary with measurements
        """
        params = {
            "model_name": model_name,
            "model_family": model_family,
            "hardware": hardware,
            "batch_size": batch_size,
            "days": days,
            "limit": limit
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        return self._make_request("get", "/api/predictive-performance/measurements", params)

class AsyncApiClient:
    """
    Asynchronous client for the Predictive Performance API.
    """
    
    def __init__(self, base_url: str, verify_ssl: bool = True, timeout: int = 30):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL for the API
            verify_ssl: Whether to verify SSL certificates
            timeout: Request timeout in seconds
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp package is required for asynchronous client")
        
        self.base_url = base_url.rstrip("/")
        self.verify_ssl = verify_ssl
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None
    
    async def _ensure_session(self):
        """Ensure we have an aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data for POST/PUT or query params for GET
            
        Returns:
            Response data as a dictionary
        """
        await self._ensure_session()
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.lower() == "get":
                async with self.session.get(url, params=data, ssl=self.verify_ssl) as response:
                    response.raise_for_status()
                    return await response.json()
            elif method.lower() == "post":
                async with self.session.post(url, json=data, ssl=self.verify_ssl) as response:
                    response.raise_for_status()
                    return await response.json()
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        
        except aiohttp.ClientError as e:
            logger.error(f"Request error: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_data = await e.response.json()
                    logger.error(f"Response error: {error_data}")
                except:
                    text = await e.response.text()
                    logger.error(f"Response text: {text}")
            raise
    
    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def __aenter__(self):
        """Enter context manager."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        await self.close()
    
    async def predict_hardware(self, 
                             model_name: str,
                             model_family: Optional[str] = None,
                             batch_size: int = 1,
                             sequence_length: int = 128,
                             mode: str = "inference",
                             precision: str = "fp32",
                             available_hardware: Optional[List[str]] = None,
                             predict_performance: bool = False) -> Dict[str, Any]:
        """
        Predict optimal hardware for a model.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family/category
            batch_size: Batch size
            sequence_length: Sequence length
            mode: "inference" or "training"
            precision: Precision to use (fp32, fp16, int8)
            available_hardware: Optional list of available hardware types
            predict_performance: Whether to predict performance on recommended hardware
            
        Returns:
            Dictionary with task ID and status
        """
        data = {
            "model_name": model_name,
            "model_family": model_family,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mode": mode,
            "precision": precision,
            "available_hardware": available_hardware,
            "predict_performance": predict_performance
        }
        
        return await self._make_request("post", "/api/predictive-performance/predict-hardware", data)
    
    async def predict_performance(self,
                                model_name: str,
                                hardware: Union[str, List[str]],
                                model_family: Optional[str] = None,
                                batch_size: int = 1,
                                sequence_length: int = 128,
                                mode: str = "inference",
                                precision: str = "fp32") -> Dict[str, Any]:
        """
        Predict performance for a model on specified hardware.
        
        Args:
            model_name: Name of the model
            hardware: Hardware type or list of hardware types
            model_family: Optional model family/category
            batch_size: Batch size
            sequence_length: Sequence length
            mode: "inference" or "training"
            precision: Precision to use (fp32, fp16, int8)
            
        Returns:
            Dictionary with task ID and status
        """
        data = {
            "model_name": model_name,
            "model_family": model_family,
            "hardware": hardware,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mode": mode,
            "precision": precision
        }
        
        return await self._make_request("post", "/api/predictive-performance/predict-performance", data)
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dictionary with task status
        """
        return await self._make_request("get", f"/api/predictive-performance/task/{task_id}")
    
    async def wait_for_task(self, task_id: str, poll_interval: float = 0.5, timeout: float = 60) -> Dict[str, Any]:
        """
        Wait for a task to complete.
        
        Args:
            task_id: ID of the task
            poll_interval: Polling interval in seconds
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with task result
        """
        start_time = time.time()
        
        while True:
            status = await self.get_task_status(task_id)
            
            if status["status"] == "completed":
                return status
            
            if status["status"] == "failed":
                raise RuntimeError(f"Task failed: {status.get('message', 'Unknown error')}")
            
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task timed out after {timeout} seconds")
            
            await anyio.sleep(poll_interval)
    
    async def monitor_task_ws(self, task_id: str, timeout: float = 60) -> Dict[str, Any]:
        """
        Monitor a task via WebSocket.
        
        Args:
            task_id: ID of the task
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with task result
        """
        ws_url = f"{self.base_url.replace('http://', 'ws://').replace('https://', 'wss://')}/api/predictive-performance/ws/{task_id}"
        
        start_time = time.time()
        result = None
        
        async with websockets.connect(ws_url) as ws:
            while True:
                try:
                    # Set a timeout for receiving messages
                    message = await # TODO: Replace with anyio.fail_after - asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(message)
                    
                    # Log progress
                    if "progress" in data:
                        logger.info(f"Task progress: {data['progress']:.0%} - {data.get('message', '')}")
                    
                    # Check for completion
                    if data.get("status") == "completed":
                        result = data.get("result")
                        break
                    
                    # Check for failure
                    if data.get("status") == "failed":
                        raise RuntimeError(f"Task failed: {data.get('message', 'Unknown error')}")
                
                except asyncio.TimeoutError:
                    # Check overall timeout
                    if time.time() - start_time > timeout:
                        raise TimeoutError(f"Task timed out after {timeout} seconds")
                    
                    # Continue waiting
                    continue
        
        if result is None:
            # Get final result via API
            status = await self.get_task_status(task_id)
            if status.get("status") == "completed":
                result = status.get("result")
            else:
                raise RuntimeError(f"Task did not complete successfully: {status}")
        
        return result

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Predictive Performance API Client")
    parser.add_argument("--url", type=str, default="http://localhost:8500", help="API server URL")
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # predict-hardware command
    predict_hw_parser = subparsers.add_parser("predict-hardware", help="Predict optimal hardware for a model")
    predict_hw_parser.add_argument("--model", type=str, required=True, help="Model name")
    predict_hw_parser.add_argument("--family", type=str, help="Model family/category")
    predict_hw_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    predict_hw_parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    predict_hw_parser.add_argument("--mode", choices=["inference", "training"], default="inference", help="Mode")
    predict_hw_parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32", help="Precision")
    predict_hw_parser.add_argument("--hardware", type=str, help="Comma-separated list of available hardware")
    predict_hw_parser.add_argument("--predict-performance", action="store_true", help="Also predict performance")
    
    # predict-performance command
    predict_perf_parser = subparsers.add_parser("predict-performance", help="Predict performance for a model")
    predict_perf_parser.add_argument("--model", type=str, required=True, help="Model name")
    predict_perf_parser.add_argument("--family", type=str, help="Model family/category")
    predict_perf_parser.add_argument("--hardware", type=str, required=True, help="Comma-separated list of hardware")
    predict_perf_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    predict_perf_parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    predict_perf_parser.add_argument("--mode", choices=["inference", "training"], default="inference", help="Mode")
    predict_perf_parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32", help="Precision")
    
    # record-measurement command
    record_meas_parser = subparsers.add_parser("record-measurement", help="Record a performance measurement")
    record_meas_parser.add_argument("--model", type=str, required=True, help="Model name")
    record_meas_parser.add_argument("--family", type=str, help="Model family/category")
    record_meas_parser.add_argument("--hardware", type=str, required=True, help="Hardware platform")
    record_meas_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    record_meas_parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    record_meas_parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp32", help="Precision")
    record_meas_parser.add_argument("--mode", choices=["inference", "training"], default="inference", help="Mode")
    record_meas_parser.add_argument("--throughput", type=float, help="Throughput measurement")
    record_meas_parser.add_argument("--latency", type=float, help="Latency measurement")
    record_meas_parser.add_argument("--memory", type=float, help="Memory usage measurement")
    record_meas_parser.add_argument("--prediction-id", type=str, help="ID of a previous prediction")
    record_meas_parser.add_argument("--source", type=str, default="cli", help="Source of the measurement")
    
    # analyze-predictions command
    analyze_parser = subparsers.add_parser("analyze-predictions", help="Analyze prediction accuracy")
    analyze_parser.add_argument("--model", type=str, help="Model name")
    analyze_parser.add_argument("--hardware", type=str, help="Hardware platform")
    analyze_parser.add_argument("--metric", choices=["throughput", "latency", "memory_usage"], help="Metric")
    analyze_parser.add_argument("--days", type=int, help="Number of days to analyze")
    
    # record-feedback command
    feedback_parser = subparsers.add_parser("record-feedback", help="Record feedback for a recommendation")
    feedback_parser.add_argument("--recommendation-id", type=str, required=True, help="Recommendation ID")
    feedback_parser.add_argument("--accepted", choices=["yes", "no"], required=True, help="Whether recommendation was accepted")
    feedback_parser.add_argument("--feedback", type=str, help="Feedback text")
    
    # list-recommendations command
    list_rec_parser = subparsers.add_parser("list-recommendations", help="List hardware recommendations")
    list_rec_parser.add_argument("--model", type=str, help="Model name")
    list_rec_parser.add_argument("--family", type=str, help="Model family")
    list_rec_parser.add_argument("--hardware", type=str, help="Hardware platform")
    list_rec_parser.add_argument("--accepted", choices=["yes", "no"], help="Filter by acceptance")
    list_rec_parser.add_argument("--days", type=int, help="Number of days to include")
    list_rec_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results")
    
    # list-predictions command
    list_pred_parser = subparsers.add_parser("list-predictions", help="List performance predictions")
    list_pred_parser.add_argument("--model", type=str, help="Model name")
    list_pred_parser.add_argument("--family", type=str, help="Model family")
    list_pred_parser.add_argument("--hardware", type=str, help="Hardware platform")
    list_pred_parser.add_argument("--batch-size", type=int, help="Batch size")
    list_pred_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results")
    
    # list-measurements command
    list_meas_parser = subparsers.add_parser("list-measurements", help="List performance measurements")
    list_meas_parser.add_argument("--model", type=str, help="Model name")
    list_meas_parser.add_argument("--family", type=str, help="Model family")
    list_meas_parser.add_argument("--hardware", type=str, help="Hardware platform")
    list_meas_parser.add_argument("--batch-size", type=int, help="Batch size")
    list_meas_parser.add_argument("--days", type=int, help="Number of days to include")
    list_meas_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results")
    
    # Generate sample data command
    sample_parser = subparsers.add_parser("generate-sample-data", help="Generate sample data for testing")
    sample_parser.add_argument("--num-models", type=int, default=5, help="Number of sample models to generate")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create client
    client = ApiClient(base_url=args.url, verify_ssl=False)
    
    # Execute command
    try:
        if args.command == "predict-hardware":
            # Parse hardware list if provided
            available_hardware = None
            if args.hardware:
                available_hardware = [hw.strip() for hw in args.hardware.split(",")]
            
            # Call API
            response = client.predict_hardware(
                model_name=args.model,
                model_family=args.family,
                batch_size=args.batch_size,
                sequence_length=args.seq_length,
                mode=args.mode,
                precision=args.precision,
                available_hardware=available_hardware,
                predict_performance=args.predict_performance
            )
            
            # Wait for task to complete
            task_id = response["task_id"]
            print(f"Task ID: {task_id}")
            
            result = client.wait_for_task(task_id)
            
            # Print result
            if "result" in result:
                result_data = result["result"]
                
                print(f"\nHardware Recommendation for {args.model}:")
                print(f"  Primary Recommendation: {result_data.get('primary_recommendation')}")
                print(f"  Fallback Options: {', '.join(result_data.get('fallback_options', []))}")
                print(f"  Compatible Hardware: {', '.join(result_data.get('compatible_hardware', []))}")
                print(f"  Model Family: {result_data.get('model_family')}")
                print(f"  Model Size: {result_data.get('model_size_category', 'unknown')} ({result_data.get('model_size', 'unknown')} parameters)")
                print(f"  Explanation: {result_data.get('explanation')}")
                
                # Print performance if available
                if "performance" in result_data:
                    perf = result_data["performance"]
                    print(f"\nPredicted Performance:")
                    print(f"  Throughput: {perf.get('throughput', 'N/A'):.2f} items/sec")
                    print(f"  Latency: {perf.get('latency', 'N/A'):.2f} ms")
                    print(f"  Memory Usage: {perf.get('memory_usage', 'N/A'):.2f} MB")
            else:
                print(f"\nError: No result found in task response")
        
        elif args.command == "predict-performance":
            # Parse hardware list
            hardware = [hw.strip() for hw in args.hardware.split(",")]
            
            # Call API
            response = client.predict_performance(
                model_name=args.model,
                model_family=args.family,
                hardware=hardware,
                batch_size=args.batch_size,
                sequence_length=args.seq_length,
                mode=args.mode,
                precision=args.precision
            )
            
            # Wait for task to complete
            task_id = response["task_id"]
            print(f"Task ID: {task_id}")
            
            result = client.wait_for_task(task_id)
            
            # Print result
            if "result" in result:
                result_data = result["result"]
                
                print(f"\nPerformance Predictions for {args.model}:")
                print(f"  Model Family: {result_data.get('model_family')}")
                print(f"  Batch Size: {result_data.get('batch_size')}")
                print(f"  Sequence Length: {result_data.get('sequence_length')}")
                print(f"  Mode: {result_data.get('mode')}")
                print(f"  Precision: {result_data.get('precision')}")
                
                print("\nPredicted Metrics by Hardware Platform:")
                for hw, pred in result_data.get('predictions', {}).items():
                    print(f"\n  {hw.upper()}:")
                    print(f"    Throughput: {pred.get('throughput', 'N/A'):.2f} items/sec")
                    print(f"    Latency: {pred.get('latency', 'N/A'):.2f} ms")
                    print(f"    Memory Usage: {pred.get('memory_usage', 'N/A'):.2f} MB")
                    print(f"    Prediction Source: {pred.get('source', 'N/A')}")
            else:
                print(f"\nError: No result found in task response")
        
        elif args.command == "record-measurement":
            # Call API
            response = client.record_measurement(
                model_name=args.model,
                model_family=args.family,
                hardware_platform=args.hardware,
                batch_size=args.batch_size,
                sequence_length=args.seq_length,
                precision=args.precision,
                mode=args.mode,
                throughput=args.throughput,
                latency=args.latency,
                memory_usage=args.memory,
                prediction_id=args.prediction_id,
                source=args.source
            )
            
            # Wait for task to complete
            task_id = response["task_id"]
            print(f"Task ID: {task_id}")
            
            result = client.wait_for_task(task_id)
            
            # Print result
            if "result" in result:
                result_data = result["result"]
                
                print(f"\nRecorded Measurement for {args.model} on {args.hardware}:")
                print(f"  Measurement ID: {result_data.get('measurement_id')}")
                
                # Print comparison with prediction if available
                if result_data.get('prediction'):
                    print("\nComparison with Prediction:")
                    print(f"  Prediction ID: {result_data.get('prediction_id')}")
                    
                    for error in result_data.get('errors', []):
                        metric = error.get('metric')
                        predicted = error.get('predicted_value')
                        actual = error.get('actual_value')
                        rel_error = error.get('relative_error', 0) * 100  # Convert to percentage
                        
                        print(f"\n  {metric.capitalize()}:")
                        print(f"    Predicted: {predicted:.2f}")
                        print(f"    Actual: {actual:.2f}")
                        print(f"    Relative Error: {rel_error:.2f}%")
            else:
                print(f"\nError: No result found in task response")
        
        elif args.command == "analyze-predictions":
            # Call API
            response = client.analyze_predictions(
                model_name=args.model,
                hardware_platform=args.hardware,
                metric=args.metric,
                days=args.days
            )
            
            # Wait for task to complete
            task_id = response["task_id"]
            print(f"Task ID: {task_id}")
            
            result = client.wait_for_task(task_id)
            
            # Print result
            if "result" in result:
                stats = result["result"]
                
                print("\nPrediction Accuracy Statistics:")
                
                if not stats:
                    print("  No prediction error data found for the specified filters")
                    return
                
                # Print overall stats if available
                if 'overall' in stats:
                    overall = stats['overall']
                    print(f"\nOverall Statistics:")
                    print(f"  Total Predictions: {overall.get('count')}")
                    print(f"  Metrics Analyzed: {', '.join(overall.get('metrics', []))}")
                    print(f"  Overall Mean Relative Error: {overall.get('overall_mean_relative_error', 0) * 100:.2f}%")
                
                # Print stats by metric
                for metric, metric_stats in stats.items():
                    if metric == 'overall':
                        continue
                        
                    print(f"\n{metric.capitalize()} Prediction Stats:")
                    print(f"  Count: {metric_stats.get('count')}")
                    print(f"  Mean Absolute Error: {metric_stats.get('mean_absolute_error'):.2f}")
                    print(f"  Mean Relative Error: {metric_stats.get('mean_relative_error', 0) * 100:.2f}%")
                    print(f"  Standard Deviation: {metric_stats.get('std_absolute_error', 0):.2f}")
                    
                    if metric_stats.get('r_squared') is not None:
                        print(f"  R²: {metric_stats.get('r_squared'):.4f}")
                    
                    print(f"  Bias: {metric_stats.get('bias', 0):.2f}")
            else:
                print(f"\nError: No result found in task response")
        
        elif args.command == "record-feedback":
            # Parse accepted flag
            accepted = args.accepted.lower() == "yes"
            
            # Call API
            response = client.record_feedback(
                recommendation_id=args.recommendation_id,
                accepted=accepted,
                feedback=args.feedback
            )
            
            # Wait for task to complete
            task_id = response["task_id"]
            print(f"Task ID: {task_id}")
            
            result = client.wait_for_task(task_id)
            
            # Print result
            if "result" in result:
                result_data = result["result"]
                
                if result_data.get("success"):
                    print(f"\nFeedback recorded successfully for recommendation {args.recommendation_id}")
                else:
                    print(f"\nFailed to record feedback for recommendation {args.recommendation_id}")
            else:
                print(f"\nError: No result found in task response")
        
        elif args.command == "list-recommendations":
            # Parse accepted flag if provided
            accepted = None
            if args.accepted:
                accepted = args.accepted.lower() == "yes"
            
            # Call API
            result = client.list_recommendations(
                model_name=args.model,
                model_family=args.family,
                hardware=args.hardware,
                accepted=accepted,
                days=args.days,
                limit=args.limit
            )
            
            # Print result
            recommendations = result.get("recommendations", [])
            print(f"\nHardware Recommendations ({len(recommendations)} found):")
            
            if not recommendations:
                print("  No recommendations found for the specified filters")
                return
            
            for rec in recommendations:
                print(f"\nRecommendation ID: {rec.get('recommendation_id')}")
                print(f"  Model: {rec.get('model_name')}")
                print(f"  Primary Recommendation: {rec.get('primary_recommendation')}")
                print(f"  Fallback Options: {', '.join(rec.get('fallback_options', []))}")
                print(f"  Configuration:")
                print(f"    Batch Size: {rec.get('batch_size')}")
                print(f"    Mode: {rec.get('mode')}")
                print(f"    Precision: {rec.get('precision')}")
                print(f"  User: {rec.get('user_id')}")
                print(f"  Timestamp: {rec.get('timestamp')}")
                
                if rec.get('was_accepted') is not None:
                    status = "Accepted" if rec.get('was_accepted') else "Rejected"
                    print(f"  Status: {status}")
                    if rec.get('user_feedback'):
                        print(f"  Feedback: {rec.get('user_feedback')}")
        
        elif args.command == "list-predictions":
            # Call API
            result = client.list_predictions(
                model_name=args.model,
                model_family=args.family,
                hardware=args.hardware,
                batch_size=args.batch_size,
                limit=args.limit
            )
            
            # Print result
            predictions = result.get("predictions", [])
            print(f"\nPerformance Predictions ({len(predictions)} found):")
            
            if not predictions:
                print("  No predictions found for the specified filters")
                return
            
            for pred in predictions:
                print(f"\nPrediction ID: {pred.get('prediction_id')}")
                print(f"  Model: {pred.get('model_name')}")
                print(f"  Hardware: {pred.get('hardware_platform')}")
                print(f"  Configuration:")
                print(f"    Batch Size: {pred.get('batch_size')}")
                print(f"    Mode: {pred.get('mode')}")
                print(f"    Precision: {pred.get('precision')}")
                
                # Performance metrics
                print(f"  Performance Metrics:")
                if pred.get('throughput') is not None:
                    print(f"    Throughput: {pred.get('throughput'):.2f} items/sec")
                if pred.get('latency') is not None:
                    print(f"    Latency: {pred.get('latency'):.2f} ms")
                if pred.get('memory_usage') is not None:
                    print(f"    Memory Usage: {pred.get('memory_usage'):.2f} MB")
                
                print(f"  Confidence: {pred.get('confidence_score', 0):.2f}")
                print(f"  Source: {pred.get('prediction_source')}")
                print(f"  Timestamp: {pred.get('timestamp')}")
        
        elif args.command == "list-measurements":
            # Call API
            result = client.list_measurements(
                model_name=args.model,
                model_family=args.family,
                hardware=args.hardware,
                batch_size=args.batch_size,
                days=args.days,
                limit=args.limit
            )
            
            # Print result
            measurements = result.get("measurements", [])
            print(f"\nPerformance Measurements ({len(measurements)} found):")
            
            if not measurements:
                print("  No measurements found for the specified filters")
                return
            
            for meas in measurements:
                print(f"\nMeasurement ID: {meas.get('measurement_id')}")
                print(f"  Model: {meas.get('model_name')}")
                print(f"  Hardware: {meas.get('hardware_platform')}")
                print(f"  Configuration:")
                print(f"    Batch Size: {meas.get('batch_size')}")
                print(f"    Mode: {meas.get('mode')}")
                print(f"    Precision: {meas.get('precision')}")
                
                # Performance metrics
                print(f"  Performance Metrics:")
                if meas.get('throughput') is not None:
                    print(f"    Throughput: {meas.get('throughput'):.2f} items/sec")
                if meas.get('latency') is not None:
                    print(f"    Latency: {meas.get('latency'):.2f} ms")
                if meas.get('memory_usage') is not None:
                    print(f"    Memory Usage: {meas.get('memory_usage'):.2f} MB")
                
                print(f"  Source: {meas.get('measurement_source')}")
                print(f"  Timestamp: {meas.get('timestamp')}")
        
        elif args.command == "generate-sample-data":
            # Call API
            data = {
                "num_models": args.num_models
            }
            
            response = client._make_request("post", "/api/predictive-performance/generate-sample-data", data)
            
            # Wait for task to complete
            task_id = response["task_id"]
            print(f"Task ID: {task_id}")
            
            result = client.wait_for_task(task_id)
            
            # Print result
            if "result" in result:
                result_data = result["result"]
                print(f"\nGenerated sample data for {result_data.get('num_models')} models")
            else:
                print(f"\nError: No result found in task response")
        
        else:
            print("Unknown command. Use --help to see available commands.")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()