#!/usr/bin/env python
"""
Client library for the Hardware-Aware Model Selection API.

This module provides a Python client for interacting with the Hardware-Aware Model
Selection API, making it easy to integrate hardware selection and performance prediction
capabilities into applications.

Part of the Phase 16 implementation.
"""

import json
import logging
import os
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urljoin

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("WARNING: requests module not available. Please install with: pip install requests")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HardwareSelectionClient:
    """Client for the Hardware-Aware Model Selection API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
            timeout: Timeout for API requests in seconds
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("The requests module is required. Please install with: pip install requests")
            
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        
        # Test the connection
        try:
            self.health()
            logger.info(f"Connected to Hardware Selection API at {base_url}")
        except Exception as e:
            logger.warning(f"Failed to connect to API at {base_url}: {e}")
            
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        return headers
        
    def _request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data for POST requests
            
        Returns:
            Dict with response data
            
        Raises:
            ValueError: If the request fails
        """
        url = urljoin(self.base_url, endpoint)
        headers = self._get_headers()
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=self.timeout)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Try to extract error message from response
            error_message = str(e)
            try:
                if hasattr(e, "response") and e.response is not None:
                    error_data = e.response.json()
                    if "message" in error_data:
                        error_message = error_data["message"]
            except (ValueError, AttributeError):
                pass
                
            raise ValueError(f"API request failed: {error_message}")
            
    def health(self) -> Dict[str, Any]:
        """
        Check the health status of the API.
        
        Returns:
            Dict with health status information
        """
        return self._request("GET", "/health")
        
    def detect_hardware(self) -> Dict[str, bool]:
        """
        Detect available hardware platforms.
        
        Returns:
            Dict mapping hardware names to availability
        """
        response = self._request("GET", "/hardware/detect")
        return response["hardware"]
        
    def select_hardware(self, 
                       model_name: str,
                       model_family: Optional[str] = None,
                       batch_size: int = 1,
                       sequence_length: int = 128,
                       mode: str = "inference",
                       precision: str = "fp32",
                       available_hardware: Optional[List[str]] = None,
                       task_type: Optional[str] = None,
                       distributed: bool = False,
                       gpu_count: int = 1) -> Dict[str, Any]:
        """
        Select optimal hardware for a model.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family (if not provided, will be inferred)
            batch_size: Batch size to use
            sequence_length: Sequence length for the model
            mode: "inference" or "training"
            precision: Precision to use (fp32, fp16, int8)
            available_hardware: List of available hardware platforms
            task_type: Specific task type
            distributed: Whether to consider distributed training
            gpu_count: Number of GPUs for distributed training
            
        Returns:
            Dict with hardware selection results
        """
        data = {
            "model_name": model_name,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mode": mode,
            "precision": precision,
            "distributed": distributed,
            "gpu_count": gpu_count
        }
        
        # Add optional parameters if provided
        if model_family:
            data["model_family"] = model_family
        if available_hardware:
            data["available_hardware"] = available_hardware
        if task_type:
            data["task_type"] = task_type
            
        return self._request("POST", "/hardware/select", data)
        
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
            model_family: Optional model family (if not provided, will be inferred)
            batch_size: Batch size
            sequence_length: Sequence length
            mode: "inference" or "training"
            precision: Precision to use
            
        Returns:
            Dict with performance predictions
        """
        data = {
            "model_name": model_name,
            "hardware": hardware,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "mode": mode,
            "precision": precision
        }
        
        # Add optional parameters if provided
        if model_family:
            data["model_family"] = model_family
            
        return self._request("POST", "/performance/predict", data)
        
    def get_distributed_training_config(self,
                                      model_name: str,
                                      model_family: Optional[str] = None,
                                      gpu_count: int = 8,
                                      batch_size: int = 8,
                                      max_memory_gb: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a distributed training configuration for a model.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family
            gpu_count: Number of GPUs
            batch_size: Per-GPU batch size
            max_memory_gb: Maximum GPU memory in GB
            
        Returns:
            Dict with distributed training configuration
        """
        data = {
            "model_name": model_name,
            "gpu_count": gpu_count,
            "batch_size": batch_size
        }
        
        # Add optional parameters if provided
        if model_family:
            data["model_family"] = model_family
        if max_memory_gb:
            data["max_memory_gb"] = max_memory_gb
            
        return self._request("POST", "/training/distributed", data)
        
    def select_hardware_for_models(self,
                                 models: List[Dict[str, str]],
                                 batch_size: int = 1,
                                 mode: str = "inference") -> Dict[str, Dict[str, Any]]:
        """
        Select hardware for multiple models in a batch.
        
        Args:
            models: List of model dictionaries with 'name' and 'family' keys
            batch_size: Batch size to use
            mode: "inference" or "training"
            
        Returns:
            Dict mapping model names to hardware recommendations
        """
        data = {
            "models": models,
            "batch_size": batch_size,
            "mode": mode
        }
        
        return self._request("POST", "/hardware/batch", data)
        
    def analyze_model_performance(self,
                                model_name: str,
                                model_family: Optional[str] = None,
                                batch_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Analyze model performance across hardware platforms.
        
        Args:
            model_name: Name of the model
            model_family: Optional model family
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dict with performance analysis
        """
        data = {
            "model_name": model_name
        }
        
        # Add optional parameters if provided
        if model_family:
            data["model_family"] = model_family
        if batch_sizes:
            data["batch_sizes"] = batch_sizes
            
        return self._request("POST", "/analysis/model", data)
        
    def create_hardware_map(self,
                          model_families: Optional[List[str]] = None,
                          batch_sizes: Optional[List[int]] = None,
                          hardware_platforms: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a hardware selection map for multiple model families.
        
        Args:
            model_families: List of model families to include
            batch_sizes: List of batch sizes to test
            hardware_platforms: List of hardware platforms to test
            
        Returns:
            Dict with hardware selection map
        """
        data = {}
        
        # Add optional parameters if provided
        if model_families:
            data["model_families"] = model_families
        if batch_sizes:
            data["batch_sizes"] = batch_sizes
        if hardware_platforms:
            data["hardware_platforms"] = hardware_platforms
            
        return self._request("POST", "/hardware/map", data)
        
    def generate_pytorch_device_str(self, recommendation: Dict[str, Any]) -> str:
        """
        Generate a PyTorch device string from a hardware recommendation.
        
        Args:
            recommendation: Hardware recommendation from select_hardware()
            
        Returns:
            PyTorch device string (e.g., 'cuda:0', 'cpu', etc.)
        """
        hardware = recommendation.get("primary_recommendation", "cpu")
        
        if hardware == "cuda":
            return "cuda:0"
        elif hardware == "mps":
            return "mps"
        elif hardware == "cpu":
            return "cpu"
        elif hardware == "rocm":
            return "cuda:0"  # PyTorch uses CUDA API for ROCm
        else:
            return "cpu"  # Default fallback
            
    def get_optimal_device(self, model_name: str, batch_size: int = 1, mode: str = "inference") -> str:
        """
        Get the optimal PyTorch device for a model.
        
        Args:
            model_name: Name of the model
            batch_size: Batch size to use
            mode: "inference" or "training"
            
        Returns:
            PyTorch device string (e.g., 'cuda:0', 'cpu', etc.)
        """
        try:
            recommendation = self.select_hardware(
                model_name=model_name,
                batch_size=batch_size,
                mode=mode
            )
            return self.generate_pytorch_device_str(recommendation)
        except Exception as e:
            logger.warning(f"Failed to get optimal device: {e}")
            return "cpu"  # Default fallback

# Simple usage example
def example_usage():
    """Example usage of the client library."""
    # Create client
    client = HardwareSelectionClient()
    
    # Check health
    health = client.health()
    print(f"API Status: {health['status']}")
    
    # Detect hardware
    hardware = client.detect_hardware()
    print("\nDetected Hardware:")
    for hw_type, available in hardware.items():
        status = "✅ Available" if available else "❌ Not available"
        print(f"  - {hw_type}: {status}")
    
    # Select hardware for a model
    recommendation = client.select_hardware(
        model_name="bert-base-uncased",
        batch_size=16,
        mode="inference"
    )
    print(f"\nRecommended Hardware: {recommendation['primary_recommendation']}")
    print(f"Fallback Options: {', '.join(recommendation['fallback_options'])}")
    
    # Get PyTorch device string
    device = client.get_optimal_device("bert-base-uncased")
    print(f"\nPyTorch Device: {device}")
    
    # Predict performance
    performance = client.predict_performance(
        model_name="bert-base-uncased",
        hardware=["cuda", "cpu"],
        batch_size=16
    )
    print("\nPerformance Predictions:")
    for hw, pred in performance["predictions"].items():
        print(f"  {hw.upper()}:")
        if "throughput" in pred:
            print(f"    Throughput: {pred['throughput']:.2f} items/sec")
        if "latency" in pred:
            print(f"    Latency: {pred['latency']:.2f} ms")
        if "memory_usage" in pred:
            print(f"    Memory Usage: {pred['memory_usage']:.2f} MB")

if __name__ == "__main__":
    if REQUESTS_AVAILABLE:
        example_usage()
    else:
        print("Please install the requests module to run the example.")
        print("pip install requests")