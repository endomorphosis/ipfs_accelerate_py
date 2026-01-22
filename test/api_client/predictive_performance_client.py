#!/usr/bin/env python3
"""
Predictive Performance API Client

This module provides a client for interacting with the Predictive Performance API
through the Unified API Server.
"""

import os
import sys
import json
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pathlib import Path

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("predictive_performance_client")

# Try to import request libraries
try:
    import requests
    import websockets
except ImportError:
    logger.error("Required libraries not installed. Run: pip install requests websockets")
    sys.exit(1)

# Define Enums for client
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

class PredictivePerformanceClient:
    """Client for interacting with the Predictive Performance API."""
    
    def __init__(self, base_url="http://localhost:8080", api_key=None):
        """Initialize the client.
        
        Args:
            base_url: Base URL of the Unified API Server
            api_key: Optional API key for authenticated endpoints
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_prefix = "/api/predictive-performance"
        
        # Session for connection pooling
        self.session = requests.Session()
        
        # Set up headers
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def _get_url(self, endpoint):
        """Get the full URL for an endpoint.
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            Full URL for the endpoint
        """
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]
        return f"{self.base_url}{self.api_prefix}/{endpoint}"
    
    def predict_hardware(self, model_name, model_family=None, batch_size=1, sequence_length=128,
                        mode=ModelMode.INFERENCE, precision=PrecisionType.FP32,
                        available_hardware=None, predict_performance=False, wait=False, timeout=60):
        """Predict optimal hardware for a model.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family/category
            batch_size: Batch size for prediction
            sequence_length: Sequence length for prediction
            mode: Inference or training mode
            precision: Precision type (fp32, fp16, int8, int4)
            available_hardware: List of available hardware platforms
            predict_performance: Whether to also predict performance metrics
            wait: Whether to wait for the prediction to complete
            timeout: Timeout in seconds when waiting
            
        Returns:
            Dict with task ID and status, or prediction results if wait=True
        """
        # Convert enums
        if isinstance(mode, ModelMode):
            mode = mode.value
        
        if isinstance(precision, PrecisionType):
            precision = precision.value
        
        # Convert available hardware
        if available_hardware is not None:
            available_hardware = [hw.value if isinstance(hw, HardwarePlatform) else hw for hw in available_hardware]
        
        # Prepare request
        payload = {
            "model_name": model_name,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mode": mode,
            "precision": precision,
            "predict_performance": predict_performance
        }
        
        if model_family:
            payload["model_family"] = model_family
        
        if available_hardware:
            payload["available_hardware"] = available_hardware
        
        # Send request
        response = self.session.post(
            self._get_url("predict-hardware"),
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Error predicting hardware: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
        
        result = response.json()
        
        # Wait for completion if requested
        if wait and "task_id" in result:
            return self.wait_for_task(result["task_id"], timeout)
        
        return result
    
    def predict_performance(self, model_name, hardware, model_family=None, batch_size=1, sequence_length=128,
                           mode=ModelMode.INFERENCE, precision=PrecisionType.FP32, wait=False, timeout=60):
        """Predict performance for a model on specified hardware.
        
        Args:
            model_name: Name of the model
            hardware: Hardware platform or list of platforms
            model_family: Optional model family/category
            batch_size: Batch size for prediction
            sequence_length: Sequence length for prediction
            mode: Inference or training mode
            precision: Precision type (fp32, fp16, int8, int4)
            wait: Whether to wait for the prediction to complete
            timeout: Timeout in seconds when waiting
            
        Returns:
            Dict with task ID and status, or prediction results if wait=True
        """
        # Convert enums
        if isinstance(mode, ModelMode):
            mode = mode.value
        
        if isinstance(precision, PrecisionType):
            precision = precision.value
        
        # Convert hardware
        if isinstance(hardware, list):
            hardware = [hw.value if isinstance(hw, HardwarePlatform) else hw for hw in hardware]
        elif isinstance(hardware, HardwarePlatform):
            hardware = hardware.value
        
        # Prepare request
        payload = {
            "model_name": model_name,
            "hardware": hardware,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mode": mode,
            "precision": precision
        }
        
        if model_family:
            payload["model_family"] = model_family
        
        # Send request
        response = self.session.post(
            self._get_url("predict-performance"),
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Error predicting performance: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
        
        result = response.json()
        
        # Wait for completion if requested
        if wait and "task_id" in result:
            return self.wait_for_task(result["task_id"], timeout)
        
        return result
    
    def record_measurement(self, model_name, hardware_platform, batch_size=1, sequence_length=128,
                          precision=PrecisionType.FP32, mode=ModelMode.INFERENCE, throughput=None,
                          latency=None, memory_usage=None, prediction_id=None, source="api",
                          model_family=None, wait=False, timeout=60):
        """Record an actual performance measurement.
        
        Args:
            model_name: Name of the model
            hardware_platform: Hardware platform
            batch_size: Batch size for the measurement
            sequence_length: Sequence length for the measurement
            precision: Precision type (fp32, fp16, int8, int4)
            mode: Inference or training mode
            throughput: Optional throughput measurement (samples/second)
            latency: Optional latency measurement (milliseconds)
            memory_usage: Optional memory usage measurement (MB)
            prediction_id: Optional ID of a previous prediction to compare with
            source: Source of the measurement (e.g., "api", "benchmark")
            model_family: Optional model family/category
            wait: Whether to wait for the recording to complete
            timeout: Timeout in seconds when waiting
            
        Returns:
            Dict with task ID and status, or recording results if wait=True
        """
        # Convert enums
        if isinstance(mode, ModelMode):
            mode = mode.value
        
        if isinstance(precision, PrecisionType):
            precision = precision.value
        
        if isinstance(hardware_platform, HardwarePlatform):
            hardware_platform = hardware_platform.value
        
        # Prepare request
        payload = {
            "model_name": model_name,
            "hardware_platform": hardware_platform,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "precision": precision,
            "mode": mode,
            "source": source
        }
        
        if model_family:
            payload["model_family"] = model_family
        
        if throughput is not None:
            payload["throughput"] = throughput
        
        if latency is not None:
            payload["latency"] = latency
        
        if memory_usage is not None:
            payload["memory_usage"] = memory_usage
        
        if prediction_id:
            payload["prediction_id"] = prediction_id
        
        # Send request
        response = self.session.post(
            self._get_url("record-measurement"),
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Error recording measurement: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
        
        result = response.json()
        
        # Wait for completion if requested
        if wait and "task_id" in result:
            return self.wait_for_task(result["task_id"], timeout)
        
        return result
    
    def analyze_predictions(self, model_name=None, hardware_platform=None, metric=None, days=None, wait=False, timeout=60):
        """Analyze prediction accuracy.
        
        Args:
            model_name: Optional model name filter
            hardware_platform: Optional hardware platform filter
            metric: Optional metric filter (e.g., "throughput", "latency")
            days: Optional number of days to look back
            wait: Whether to wait for the analysis to complete
            timeout: Timeout in seconds when waiting
            
        Returns:
            Dict with task ID and status, or analysis results if wait=True
        """
        # Convert enums
        if isinstance(hardware_platform, HardwarePlatform):
            hardware_platform = hardware_platform.value
        
        # Prepare request
        payload = {}
        
        if model_name:
            payload["model_name"] = model_name
        
        if hardware_platform:
            payload["hardware_platform"] = hardware_platform
        
        if metric:
            payload["metric"] = metric
        
        if days:
            payload["days"] = days
        
        # Send request
        response = self.session.post(
            self._get_url("analyze-predictions"),
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Error analyzing predictions: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
        
        result = response.json()
        
        # Wait for completion if requested
        if wait and "task_id" in result:
            return self.wait_for_task(result["task_id"], timeout)
        
        return result
    
    def record_feedback(self, recommendation_id, accepted, feedback=None, wait=False, timeout=60):
        """Record feedback for a hardware recommendation.
        
        Args:
            recommendation_id: ID of the recommendation
            accepted: Whether the recommendation was accepted
            feedback: Optional feedback text
            wait: Whether to wait for the recording to complete
            timeout: Timeout in seconds when waiting
            
        Returns:
            Dict with task ID and status, or recording results if wait=True
        """
        # Prepare request
        payload = {
            "recommendation_id": recommendation_id,
            "accepted": accepted
        }
        
        if feedback:
            payload["feedback"] = feedback
        
        # Send request
        response = self.session.post(
            self._get_url("record-feedback"),
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Error recording feedback: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
        
        result = response.json()
        
        # Wait for completion if requested
        if wait and "task_id" in result:
            return self.wait_for_task(result["task_id"], timeout)
        
        return result
    
    def generate_sample_data(self, num_models=5, wait=False, timeout=60):
        """Generate sample data for testing.
        
        Args:
            num_models: Number of models to generate
            wait: Whether to wait for the generation to complete
            timeout: Timeout in seconds when waiting
            
        Returns:
            Dict with task ID and status, or generation results if wait=True
        """
        # Prepare request
        payload = {"num_models": num_models}
        
        # Send request
        response = self.session.post(
            self._get_url("generate-sample-data"),
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Error generating sample data: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
        
        result = response.json()
        
        # Wait for completion if requested
        if wait and "task_id" in result:
            return self.wait_for_task(result["task_id"], timeout)
        
        return result
    
    def get_task_status(self, task_id):
        """Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dict with task status information
        """
        response = self.session.get(
            self._get_url(f"task/{task_id}"),
            headers=self.headers
        )
        
        if response.status_code != 200:
            logger.error(f"Error getting task status: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
        
        return response.json()
    
    def wait_for_task(self, task_id, timeout=60):
        """Wait for a task to complete.
        
        Args:
            task_id: ID of the task
            timeout: Timeout in seconds
            
        Returns:
            Dict with task status information, including result if completed
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)
            
            if "error" in status:
                return status
            
            if status.get("status") == "completed":
                return status
            
            if status.get("status") == "failed":
                logger.error(f"Task failed: {status.get('message')}")
                return status
            
            # Sleep before checking again
            time.sleep(1)
        
        logger.warning(f"Task {task_id} did not complete within timeout")
        return {"error": "Timeout waiting for task completion", "task_id": task_id}
    
    def list_recommendations(self, model_name=None, model_family=None, hardware=None, accepted=None, days=None, limit=10):
        """List hardware recommendations.
        
        Args:
            model_name: Optional model name filter
            model_family: Optional model family filter
            hardware: Optional hardware platform filter
            accepted: Optional filter for whether the recommendation was accepted
            days: Optional number of days to look back
            limit: Maximum number of results to return
            
        Returns:
            Dict with list of recommendations
        """
        # Build query parameters
        params = {"limit": limit}
        
        if model_name:
            params["model_name"] = model_name
        
        if model_family:
            params["model_family"] = model_family
        
        if hardware:
            params["hardware"] = hardware.value if isinstance(hardware, HardwarePlatform) else hardware
        
        if accepted is not None:
            params["accepted"] = str(accepted).lower()
        
        if days:
            params["days"] = days
        
        # Send request
        response = self.session.get(
            self._get_url("recommendations"),
            headers=self.headers,
            params=params
        )
        
        if response.status_code != 200:
            logger.error(f"Error listing recommendations: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
        
        return response.json()
    
    def list_measurements(self, model_name=None, model_family=None, hardware=None, batch_size=None, days=None, limit=10):
        """List performance measurements.
        
        Args:
            model_name: Optional model name filter
            model_family: Optional model family filter
            hardware: Optional hardware platform filter
            batch_size: Optional batch size filter
            days: Optional number of days to look back
            limit: Maximum number of results to return
            
        Returns:
            Dict with list of measurements
        """
        # Build query parameters
        params = {"limit": limit}
        
        if model_name:
            params["model_name"] = model_name
        
        if model_family:
            params["model_family"] = model_family
        
        if hardware:
            params["hardware"] = hardware.value if isinstance(hardware, HardwarePlatform) else hardware
        
        if batch_size:
            params["batch_size"] = batch_size
        
        if days:
            params["days"] = days
        
        # Send request
        response = self.session.get(
            self._get_url("measurements"),
            headers=self.headers,
            params=params
        )
        
        if response.status_code != 200:
            logger.error(f"Error listing measurements: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
        
        return response.json()
    
    def list_predictions(self, model_name=None, model_family=None, hardware=None, batch_size=None, limit=10):
        """List performance predictions.
        
        Args:
            model_name: Optional model name filter
            model_family: Optional model family filter
            hardware: Optional hardware platform filter
            batch_size: Optional batch size filter
            limit: Maximum number of results to return
            
        Returns:
            Dict with list of predictions
        """
        # Build query parameters
        params = {"limit": limit}
        
        if model_name:
            params["model_name"] = model_name
        
        if model_family:
            params["model_family"] = model_family
        
        if hardware:
            params["hardware"] = hardware.value if isinstance(hardware, HardwarePlatform) else hardware
        
        if batch_size:
            params["batch_size"] = batch_size
        
        # Send request
        response = self.session.get(
            self._get_url("predictions"),
            headers=self.headers,
            params=params
        )
        
        if response.status_code != 200:
            logger.error(f"Error listing predictions: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
        
        return response.json()
    
    async def monitor_task(self, task_id, callback=None):
        """Monitor a task via WebSocket.
        
        Args:
            task_id: ID of the task to monitor
            callback: Optional callback function for updates
            
        Returns:
            Final task result
        """
        ws_url = f"{self.base_url.replace('http://', 'ws://').replace('https://', 'wss://')}{self.api_prefix}/ws/{task_id}"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Call the callback if provided
                    if callback:
                        callback(data)
                    
                    # Return the result if the task is completed or failed
                    if data.get("status") in ["completed", "failed"]:
                        return data
                    
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            return {"error": str(e), "task_id": task_id}

class AsyncPredictivePerformanceClient:
    """Asynchronous client for interacting with the Predictive Performance API."""
    
    def __init__(self, base_url="http://localhost:8080", api_key=None):
        """Initialize the client.
        
        Args:
            base_url: Base URL of the Unified API Server
            api_key: Optional API key for authenticated endpoints
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_prefix = "/api/predictive-performance"
        
        # Set up headers
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
        
        # Session will be created on demand
        self.session = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self.session is None:
            try:
                import aiohttp
            except ImportError:
                logger.error("aiohttp library not installed. Run: pip install aiohttp")
                raise ImportError("aiohttp library not installed")
            
            self.session = aiohttp.ClientSession(headers=self.headers)
        
        return self.session
    
    def _get_url(self, endpoint):
        """Get the full URL for an endpoint.
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            Full URL for the endpoint
        """
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]
        return f"{self.base_url}{self.api_prefix}/{endpoint}"
    
    async def predict_hardware(self, model_name, model_family=None, batch_size=1, sequence_length=128,
                              mode=ModelMode.INFERENCE, precision=PrecisionType.FP32,
                              available_hardware=None, predict_performance=False, wait=False, timeout=60):
        """Predict optimal hardware for a model.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family/category
            batch_size: Batch size for prediction
            sequence_length: Sequence length for prediction
            mode: Inference or training mode
            precision: Precision type (fp32, fp16, int8, int4)
            available_hardware: List of available hardware platforms
            predict_performance: Whether to also predict performance metrics
            wait: Whether to wait for the prediction to complete
            timeout: Timeout in seconds when waiting
            
        Returns:
            Dict with task ID and status, or prediction results if wait=True
        """
        # Convert enums
        if isinstance(mode, ModelMode):
            mode = mode.value
        
        if isinstance(precision, PrecisionType):
            precision = precision.value
        
        # Convert available hardware
        if available_hardware is not None:
            available_hardware = [hw.value if isinstance(hw, HardwarePlatform) else hw for hw in available_hardware]
        
        # Prepare request
        payload = {
            "model_name": model_name,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mode": mode,
            "precision": precision,
            "predict_performance": predict_performance
        }
        
        if model_family:
            payload["model_family"] = model_family
        
        if available_hardware:
            payload["available_hardware"] = available_hardware
        
        # Get session
        session = await self._ensure_session()
        
        # Send request
        async with session.post(self._get_url("predict-hardware"), json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Error predicting hardware: {error_text}")
                return {"error": error_text, "status_code": response.status}
            
            result = await response.json()
        
        # Wait for completion if requested
        if wait and "task_id" in result:
            return await self.wait_for_task(result["task_id"], timeout)
        
        return result
    
    async def get_task_status(self, task_id):
        """Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Dict with task status information
        """
        # Get session
        session = await self._ensure_session()
        
        # Send request
        async with session.get(self._get_url(f"task/{task_id}")) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Error getting task status: {error_text}")
                return {"error": error_text, "status_code": response.status}
            
            return await response.json()
    
    async def wait_for_task(self, task_id, timeout=60):
        """Wait for a task to complete.
        
        Args:
            task_id: ID of the task
            timeout: Timeout in seconds
            
        Returns:
            Dict with task status information, including result if completed
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = await self.get_task_status(task_id)
            
            if "error" in status:
                return status
            
            if status.get("status") == "completed":
                return status
            
            if status.get("status") == "failed":
                logger.error(f"Task failed: {status.get('message')}")
                return status
            
            # Sleep before checking again
            await asyncio.sleep(1)
        
        logger.warning(f"Task {task_id} did not complete within timeout")
        return {"error": "Timeout waiting for task completion", "task_id": task_id}
    
    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None

# Example usage
def demo_usage():
    """Demonstrate client usage."""
    client = PredictivePerformanceClient()
    
    # Generate sample data
    print("Generating sample data...")
    result = client.generate_sample_data(num_models=3, wait=True)
    print(f"Sample data generation result: {json.dumps(result, indent=2)}")
    
    # Predict hardware
    print("\nPredicting hardware...")
    result = client.predict_hardware(
        model_name="bert-base-uncased",
        batch_size=8,
        available_hardware=[HardwarePlatform.CPU, HardwarePlatform.CUDA],
        predict_performance=True,
        wait=True
    )
    print(f"Hardware prediction result: {json.dumps(result, indent=2)}")
    
    # Record measurement
    print("\nRecording measurement...")
    result = client.record_measurement(
        model_name="bert-base-uncased",
        hardware_platform=HardwarePlatform.CUDA,
        batch_size=8,
        throughput=120.5,
        latency=8.3,
        memory_usage=1024.0,
        wait=True
    )
    print(f"Measurement recording result: {json.dumps(result, indent=2)}")
    
    # List recommendations
    print("\nListing recommendations...")
    result = client.list_recommendations(limit=3)
    print(f"Recommendations: {json.dumps(result, indent=2)}")

async def demo_async_usage():
    """Demonstrate async client usage."""
    client = AsyncPredictivePerformanceClient()
    
    try:
        # Generate sample data
        print("Generating sample data...")
        result = await client.predict_hardware(
            model_name="bert-base-uncased",
            batch_size=8,
            available_hardware=[HardwarePlatform.CPU, HardwarePlatform.CUDA],
            predict_performance=True,
            wait=True
        )
        print(f"Hardware prediction result: {json.dumps(result, indent=2)}")
    
    finally:
        # Close the client
        await client.close()

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Predictive Performance API Client")
    parser.add_argument("--url", type=str, default="http://localhost:8080", help="Unified API Server URL")
    parser.add_argument("--key", type=str, help="API key for authenticated endpoints")
    parser.add_argument("--demo", action="store_true", help="Run demo usage")
    parser.add_argument("--async-demo", action="store_true", help="Run async demo usage")
    
    args = parser.parse_args()
    
    if args.demo:
        client = PredictivePerformanceClient(base_url=args.url, api_key=args.key)
        demo_usage()
    elif args.async_demo:
        asyncio.run(demo_async_usage())
    else:
        print("Use --demo or --async-demo to run example usage")
        print("For programmatic usage, import the client classes from this module")

if __name__ == "__main__":
    main()