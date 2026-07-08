#!/usr/bin/env python3
"""
Unified Testing Interface for API Backends

This module provides a comprehensive interface for testing and benchmarking
different API providers (OpenAI, Claude, Groq, etc.) within the distributed
testing framework. It provides a unified interface, performance metrics collection,
and integration with anomaly detection and predictive analytics.
"""

import os
import sys
import json
import time
import uuid
import logging
import datetime
import threading
import functools
import traceback
from enum import Enum
from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable, Type, TypeVar

# Import API-specific modules and distributed testing framework components
try:
    from test.distributed_testing.coordinator import CoordinatorClient
    from test.distributed_testing.task import Task, TaskResult, TaskStatus
    from test.distributed_testing.worker import WorkerNode
    from api_anomaly_detection import AnomalyDetector
    from api_predictive_analytics import TimeSeriesPredictor, AnomalyPredictor
    from api_monitoring_dashboard import APIMonitoringDashboard
    from api_notification_manager import NotificationManager, NotificationRule, AnomalySeverity
except ImportError:
    # Add parent directory to path for local development
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Try import again
    from test.distributed_testing.coordinator import CoordinatorClient
    from test.distributed_testing.task import Task, TaskResult, TaskStatus
    from test.distributed_testing.worker import WorkerNode
    from test.api_anomaly_detection import AnomalyDetector
    from test.api_predictive_analytics import TimeSeriesPredictor, AnomalyPredictor
    from test.api_monitoring_dashboard import APIMonitoringDashboard
    from test.api_notification_manager import NotificationManager, NotificationRule, AnomalySeverity


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APITestType(Enum):
    """Enum for different API test types."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RELIABILITY = "reliability"
    CONCURRENCY = "concurrency"
    COST_EFFICIENCY = "cost_efficiency"
    TOKEN_COUNTING = "token_counting"
    STREAMING = "streaming"
    CONTEXT_WINDOW = "context_window"
    ERROR_HANDLING = "error_handling"
    RATE_LIMIT = "rate_limit"


class APIProvider(Enum):
    """Enum for different API providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    ANTHROPIC = "anthropic"  # Alternative name for Claude
    GROQ = "groq"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"
    TOGETHER = "together"
    ANYSCALE = "anyscale"
    BEDROCK = "bedrock"
    LOCAL = "local"
    CUSTOM = "custom"


class APICapability(Enum):
    """Enum for different API capabilities."""
    CHAT_COMPLETION = "chat_completion"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDIT = "image_edit"
    IMAGE_VARIATION = "image_variation"
    VISION = "vision"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    FINE_TUNING = "fine_tuning"
    MODERATION = "moderation"
    FILE_UPLOAD = "file_upload"
    VECTOR_SEARCH = "vector_search"
    FUNCTION_CALLING = "function_calling"
    STREAMING = "streaming"


class APIBackend:
    """Base class for API backend implementations."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize API backend.
        
        Args:
            api_key: API key for authentication
            model: Default model to use
        """
        self.api_key = api_key or self._get_default_api_key()
        self.model = model or self._get_default_model()
        self.metadata = self._get_metadata()
        
    def _get_default_api_key(self) -> Optional[str]:
        """Get default API key from environment variables."""
        return None
    
    def _get_default_model(self) -> str:
        """Get default model for this backend."""
        return "default"
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get metadata for this backend."""
        return {
            "name": "base",
            "version": "0.1.0",
            "capabilities": [],
            "models": [],
            "rate_limits": {}
        }
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            Response from the API
        """
        raise NotImplementedError("Chat completion not implemented for this backend")
    
    def embedding(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Get embeddings for text.
        
        Args:
            text: Text to embed
            **kwargs: Additional parameters
            
        Returns:
            Response from the API
        """
        raise NotImplementedError("Embedding not implemented for this backend")
    
    def check_capabilities(self) -> List[APICapability]:
        """
        Check which capabilities are supported by this backend.
        
        Returns:
            List of supported capabilities
        """
        capabilities = []
        for capability in APICapability:
            try:
                method_name = capability.value
                if hasattr(self, method_name) and callable(getattr(self, method_name)):
                    # Try to get the method's signature to confirm it's properly implemented
                    capabilities.append(capability)
            except:
                pass
        return capabilities
    
    def get_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models.
        
        Returns:
            List of model information dictionaries
        """
        return self.metadata.get("models", [])
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get rate limit information.
        
        Returns:
            Dictionary with rate limit information
        """
        return self.metadata.get("rate_limits", {})


class OpenAIBackend(APIBackend):
    """OpenAI API backend implementation."""
    
    def _get_default_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment variables."""
        return os.environ.get("OPENAI_API_KEY")
    
    def _get_default_model(self) -> str:
        """Get default model for OpenAI."""
        return "gpt-3.5-turbo"
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get metadata for OpenAI backend."""
        return {
            "name": "openai",
            "version": "1.0.0",
            "capabilities": [
                APICapability.CHAT_COMPLETION.value,
                APICapability.COMPLETION.value,
                APICapability.EMBEDDING.value,
                APICapability.IMAGE_GENERATION.value,
                APICapability.IMAGE_EDIT.value,
                APICapability.IMAGE_VARIATION.value,
                APICapability.VISION.value,
                APICapability.SPEECH_TO_TEXT.value,
                APICapability.TEXT_TO_SPEECH.value,
                APICapability.FINE_TUNING.value,
                APICapability.MODERATION.value,
                APICapability.FUNCTION_CALLING.value,
                APICapability.STREAMING.value,
            ],
            "models": [
                {
                    "id": "gpt-4o",
                    "capabilities": ["chat_completion", "vision", "function_calling"],
                    "context_window": 128000,
                    "cost_per_1k_input_tokens": 0.005,
                    "cost_per_1k_output_tokens": 0.015
                },
                {
                    "id": "gpt-4-turbo",
                    "capabilities": ["chat_completion", "vision", "function_calling"],
                    "context_window": 128000,
                    "cost_per_1k_input_tokens": 0.01,
                    "cost_per_1k_output_tokens": 0.03
                },
                {
                    "id": "gpt-4",
                    "capabilities": ["chat_completion", "vision", "function_calling"],
                    "context_window": 8192,
                    "cost_per_1k_input_tokens": 0.03,
                    "cost_per_1k_output_tokens": 0.06
                },
                {
                    "id": "gpt-3.5-turbo",
                    "capabilities": ["chat_completion", "function_calling"],
                    "context_window": 16385,
                    "cost_per_1k_input_tokens": 0.0005,
                    "cost_per_1k_output_tokens": 0.0015
                },
                {
                    "id": "text-embedding-3-small",
                    "capabilities": ["embedding"],
                    "cost_per_1k_input_tokens": 0.00002
                }
            ],
            "rate_limits": {
                "requests_per_minute": 60,
                "tokens_per_minute": 90000,
                "concurrent_requests": 50
            }
        }
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Send a chat completion request to OpenAI.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            Response from the API
        """
        try:
            import openai
            
            if self.api_key:
                openai.api_key = self.api_key
            
            model = kwargs.pop("model", self.model)
            
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            return response.model_dump()
        except ImportError:
            logger.error("OpenAI Python library not installed")
            raise
        except Exception as e:
            logger.error(f"Error in OpenAI chat completion: {e}")
            raise
    
    def embedding(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Get embeddings from OpenAI.
        
        Args:
            text: Text to embed
            **kwargs: Additional parameters
            
        Returns:
            Response from the API
        """
        try:
            import openai
            
            if self.api_key:
                openai.api_key = self.api_key
            
            model = kwargs.pop("model", "text-embedding-3-small")
            
            response = openai.embeddings.create(
                model=model,
                input=text,
                **kwargs
            )
            
            return response.model_dump()
        except ImportError:
            logger.error("OpenAI Python library not installed")
            raise
        except Exception as e:
            logger.error(f"Error in OpenAI embedding: {e}")
            raise


class ClaudeBackend(APIBackend):
    """Claude (Anthropic) API backend implementation."""
    
    def _get_default_api_key(self) -> Optional[str]:
        """Get Claude API key from environment variables."""
        return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    
    def _get_default_model(self) -> str:
        """Get default model for Claude."""
        return "claude-3-opus-20240229"
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get metadata for Claude backend."""
        return {
            "name": "claude",
            "version": "1.0.0",
            "capabilities": [
                APICapability.CHAT_COMPLETION.value,
                APICapability.VISION.value,
                APICapability.STREAMING.value,
            ],
            "models": [
                {
                    "id": "claude-3-opus-20240229",
                    "capabilities": ["chat_completion", "vision"],
                    "context_window": 200000,
                    "cost_per_1k_input_tokens": 0.015,
                    "cost_per_1k_output_tokens": 0.075
                },
                {
                    "id": "claude-3-sonnet-20240229",
                    "capabilities": ["chat_completion", "vision"],
                    "context_window": 200000,
                    "cost_per_1k_input_tokens": 0.003,
                    "cost_per_1k_output_tokens": 0.015
                },
                {
                    "id": "claude-3-haiku-20240307",
                    "capabilities": ["chat_completion", "vision"],
                    "context_window": 200000,
                    "cost_per_1k_input_tokens": 0.00025,
                    "cost_per_1k_output_tokens": 0.00125
                }
            ],
            "rate_limits": {
                "requests_per_minute": 40,
                "tokens_per_minute": 100000,
                "concurrent_requests": 30
            }
        }
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Send a chat completion request to Claude.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            Response from the API
        """
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            model = kwargs.pop("model", self.model)
            
            # Convert from OpenAI format to Anthropic format if needed
            if messages and isinstance(messages[0], dict) and "role" in messages[0]:
                anthropic_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        # System message needs special handling in Claude
                        if not any(m.get("role") == "system" for m in anthropic_messages):
                            anthropic_messages.append({
                                "role": "system",
                                "content": msg["content"]
                            })
                    elif msg["role"] == "user":
                        anthropic_messages.append({
                            "role": "user",
                            "content": msg["content"]
                        })
                    elif msg["role"] == "assistant":
                        anthropic_messages.append({
                            "role": "assistant",
                            "content": msg["content"]
                        })
                messages = anthropic_messages
            
            response = client.messages.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            return response.model_dump()
        except ImportError:
            logger.error("Anthropic Python library not installed")
            raise
        except Exception as e:
            logger.error(f"Error in Claude chat completion: {e}")
            raise


class GroqBackend(APIBackend):
    """Groq API backend implementation."""
    
    def _get_default_api_key(self) -> Optional[str]:
        """Get Groq API key from environment variables."""
        return os.environ.get("GROQ_API_KEY")
    
    def _get_default_model(self) -> str:
        """Get default model for Groq."""
        return "llama3-8b-8192"
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get metadata for Groq backend."""
        return {
            "name": "groq",
            "version": "1.0.0",
            "capabilities": [
                APICapability.CHAT_COMPLETION.value,
                APICapability.STREAMING.value,
            ],
            "models": [
                {
                    "id": "llama3-8b-8192",
                    "capabilities": ["chat_completion"],
                    "context_window": 8192,
                    "cost_per_1k_input_tokens": 0.0001,
                    "cost_per_1k_output_tokens": 0.0002
                },
                {
                    "id": "llama3-70b-8192",
                    "capabilities": ["chat_completion"],
                    "context_window": 8192,
                    "cost_per_1k_input_tokens": 0.0007,
                    "cost_per_1k_output_tokens": 0.0009
                },
                {
                    "id": "mixtral-8x7b-32768",
                    "capabilities": ["chat_completion"],
                    "context_window": 32768,
                    "cost_per_1k_input_tokens": 0.0006,
                    "cost_per_1k_output_tokens": 0.0006
                }
            ],
            "rate_limits": {
                "requests_per_minute": 100,
                "concurrent_requests": 100
            }
        }
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Send a chat completion request to Groq.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            Response from the API
        """
        try:
            import groq
            
            client = groq.Groq(api_key=self.api_key)
            
            model = kwargs.pop("model", self.model)
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            return response.model_dump()
        except ImportError:
            logger.error("Groq Python library not installed")
            raise
        except Exception as e:
            logger.error(f"Error in Groq chat completion: {e}")
            raise


class APIBackendFactory:
    """Factory class for creating API backends."""
    
    @staticmethod
    def create_backend(provider: Union[str, APIProvider], **kwargs) -> APIBackend:
        """
        Create an API backend instance.
        
        Args:
            provider: API provider name or enum
            **kwargs: Additional parameters for the backend
            
        Returns:
            APIBackend instance
        """
        if isinstance(provider, APIProvider):
            provider = provider.value
        
        provider = provider.lower()
        
        if provider in ("openai",):
            return OpenAIBackend(**kwargs)
        elif provider in ("claude", "anthropic"):
            return ClaudeBackend(**kwargs)
        elif provider in ("groq",):
            return GroqBackend(**kwargs)
        else:
            raise ValueError(f"Unsupported API provider: {provider}")


class APITester:
    """Class for testing API backends."""
    
    def __init__(self, 
                 backend: Optional[APIBackend] = None,
                 provider: Optional[Union[str, APIProvider]] = None,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 coordinator_url: Optional[str] = None,
                 enable_anomaly_detection: bool = False,
                 enable_predictive_analytics: bool = False,
                 anomaly_sensitivity: float = 1.0,
                 results_dir: str = "./api_test_results"):
        """
        Initialize API tester.
        
        Args:
            backend: API backend instance
            provider: API provider name or enum
            api_key: API key for authentication
            model: Model to use for testing
            coordinator_url: URL of the distributed testing coordinator
            enable_anomaly_detection: Whether to enable anomaly detection
            enable_predictive_analytics: Whether to enable predictive analytics
            anomaly_sensitivity: Sensitivity for anomaly detection
            results_dir: Directory for storing test results
        """
        # Initialize backend
        if backend:
            self.backend = backend
        elif provider:
            self.backend = APIBackendFactory.create_backend(
                provider, api_key=api_key, model=model)
        else:
            raise ValueError("Either backend or provider must be specified")
        
        # Initialize coordinator client
        self.coordinator_client = None
        if coordinator_url:
            try:
                self.coordinator_client = CoordinatorClient(coordinator_url)
            except Exception as e:
                logger.warning(f"Failed to connect to coordinator: {e}")
        
        # Initialize directories
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize test ID
        self.test_id = str(uuid.uuid4())
        
        # Initialize anomaly detection
        self.enable_anomaly_detection = enable_anomaly_detection
        if enable_anomaly_detection:
            self.anomaly_detector = AnomalyDetector(sensitivity=anomaly_sensitivity)
        else:
            self.anomaly_detector = None
        
        # Initialize predictive analytics
        self.enable_predictive_analytics = enable_predictive_analytics
        if enable_predictive_analytics:
            self.predictor = TimeSeriesPredictor()
            self.anomaly_predictor = AnomalyPredictor()
        else:
            self.predictor = None
            self.anomaly_predictor = None
        
        # Initialize results
        self.results = {
            "test_id": self.test_id,
            "provider": self.backend.metadata["name"],
            "model": self.backend.model,
            "timestamp": datetime.datetime.now().isoformat(),
            "tests": {}
        }
    
    def run_latency_test(self, 
                          messages: List[Dict[str, str]],
                          iterations: int = 5, 
                          percentiles: List[float] = [50, 90, 95, 99],
                          **kwargs) -> Dict[str, Any]:
        """
        Run latency test.
        
        Args:
            messages: List of message dictionaries
            iterations: Number of test iterations
            percentiles: Percentiles to calculate
            **kwargs: Additional parameters for chat completion
            
        Returns:
            Test results
        """
        latencies = []
        errors = []
        
        for i in range(iterations):
            try:
                start_time = time.time()
                response = self.backend.chat_completion(messages, **kwargs)
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
                
                logger.debug(f"Latency test iteration {i+1}/{iterations}: {latency:.2f}ms")
            except Exception as e:
                logger.error(f"Error in latency test iteration {i+1}/{iterations}: {e}")
                errors.append(str(e))
        
        # Calculate statistics
        results = {
            "test_type": APITestType.LATENCY.value,
            "iterations": iterations,
            "successful_iterations": len(latencies),
            "failed_iterations": len(errors),
            "errors": errors,
            "latencies_ms": latencies,
        }
        
        if latencies:
            import numpy as np
            
            results.update({
                "min_latency_ms": float(np.min(latencies)),
                "max_latency_ms": float(np.max(latencies)),
                "mean_latency_ms": float(np.mean(latencies)),
                "median_latency_ms": float(np.median(latencies)),
                "stddev_latency_ms": float(np.std(latencies)),
                "percentiles_ms": {
                    str(int(p)): float(np.percentile(latencies, p)) 
                    for p in percentiles
                }
            })
        
        # Store results
        self.results["tests"][f"latency_{int(time.time())}"] = results
        
        # Detect anomalies
        if self.enable_anomaly_detection and latencies:
            anomalies = self.anomaly_detector.detect_latency_anomalies(
                self.backend.metadata["name"], 
                latencies
            )
            if anomalies:
                results["anomalies"] = anomalies
        
        # Save results
        self._save_results()
        
        # Submit results to coordinator
        if self.coordinator_client:
            self._submit_to_coordinator("latency", results)
        
        return results
    
    def run_throughput_test(self,
                            messages: List[Dict[str, str]],
                            duration: int = 10,
                            concurrent_requests: int = 5,
                            **kwargs) -> Dict[str, Any]:
        """
        Run throughput test.
        
        Args:
            messages: List of message dictionaries
            duration: Test duration in seconds
            concurrent_requests: Number of concurrent requests
            **kwargs: Additional parameters for chat completion
            
        Returns:
            Test results
        """
        start_time = time.time()
        end_time = start_time + duration
        
        requests_completed = 0
        errors = []
        latencies = []
        
        # Create a lock for thread-safe incrementing
        lock = threading.Lock()
        
        def worker():
            nonlocal requests_completed
            
            while time.time() < end_time:
                try:
                    req_start = time.time()
                    response = self.backend.chat_completion(messages, **kwargs)
                    req_end = time.time()
                    
                    latency = (req_end - req_start) * 1000  # Convert to ms
                    
                    with lock:
                        requests_completed += 1
                        latencies.append(latency)
                        
                except Exception as e:
                    with lock:
                        errors.append(str(e))
        
        # Start worker threads
        threads = []
        for _ in range(concurrent_requests):
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Wait for test duration
        remaining = end_time - time.time()
        while remaining > 0:
            time.sleep(min(1, remaining))
            remaining = end_time - time.time()
        
        # Wait for threads to complete current requests
        for thread in threads:
            thread.join(timeout=5)
        
        actual_duration = time.time() - start_time
        
        # Calculate statistics
        results = {
            "test_type": APITestType.THROUGHPUT.value,
            "duration_seconds": actual_duration,
            "concurrent_requests": concurrent_requests,
            "requests_completed": requests_completed,
            "errors": len(errors),
            "error_messages": errors[:10],  # Limit to first 10 errors
            "throughput_rps": requests_completed / actual_duration,
        }
        
        if latencies:
            import numpy as np
            
            results.update({
                "min_latency_ms": float(np.min(latencies)),
                "max_latency_ms": float(np.max(latencies)),
                "mean_latency_ms": float(np.mean(latencies)),
                "median_latency_ms": float(np.median(latencies)),
                "stddev_latency_ms": float(np.std(latencies)),
                "latency_percentiles_ms": {
                    "50": float(np.percentile(latencies, 50)),
                    "90": float(np.percentile(latencies, 90)),
                    "95": float(np.percentile(latencies, 95)),
                    "99": float(np.percentile(latencies, 99)),
                }
            })
        
        # Store results
        self.results["tests"][f"throughput_{int(time.time())}"] = results
        
        # Detect anomalies
        if self.enable_anomaly_detection and latencies:
            throughput_anomalies = self.anomaly_detector.detect_throughput_anomalies(
                self.backend.metadata["name"], 
                results["throughput_rps"]
            )
            if throughput_anomalies:
                results["throughput_anomalies"] = throughput_anomalies
                
            latency_anomalies = self.anomaly_detector.detect_latency_anomalies(
                self.backend.metadata["name"], 
                latencies
            )
            if latency_anomalies:
                results["latency_anomalies"] = latency_anomalies
        
        # Save results
        self._save_results()
        
        # Submit results to coordinator
        if self.coordinator_client:
            self._submit_to_coordinator("throughput", results)
        
        return results
    
    def run_reliability_test(self,
                             messages: List[Dict[str, str]],
                             iterations: int = 50,
                             **kwargs) -> Dict[str, Any]:
        """
        Run reliability test.
        
        Args:
            messages: List of message dictionaries
            iterations: Number of test iterations
            **kwargs: Additional parameters for chat completion
            
        Returns:
            Test results
        """
        successes = 0
        failures = 0
        errors = []
        
        for i in range(iterations):
            try:
                response = self.backend.chat_completion(messages, **kwargs)
                successes += 1
            except Exception as e:
                failures += 1
                errors.append(str(e))
                logger.error(f"Error in reliability test iteration {i+1}/{iterations}: {e}")
        
        # Calculate statistics
        results = {
            "test_type": APITestType.RELIABILITY.value,
            "iterations": iterations,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / iterations if iterations > 0 else 0,
            "error_rate": failures / iterations if iterations > 0 else 0,
            "errors": errors[:10],  # Limit to first 10 errors
        }
        
        # Store results
        self.results["tests"][f"reliability_{int(time.time())}"] = results
        
        # Detect anomalies
        if self.enable_anomaly_detection:
            reliability_anomalies = self.anomaly_detector.detect_reliability_anomalies(
                self.backend.metadata["name"], 
                results["success_rate"]
            )
            if reliability_anomalies:
                results["anomalies"] = reliability_anomalies
        
        # Save results
        self._save_results()
        
        # Submit results to coordinator
        if self.coordinator_client:
            self._submit_to_coordinator("reliability", results)
        
        return results
    
    def run_cost_efficiency_test(self,
                                 messages: List[Dict[str, str]],
                                 iterations: int = 5,
                                 **kwargs) -> Dict[str, Any]:
        """
        Run cost efficiency test.
        
        Args:
            messages: List of message dictionaries
            iterations: Number of test iterations
            **kwargs: Additional parameters for chat completion
            
        Returns:
            Test results
        """
        # Get model cost information
        model = kwargs.get("model", self.backend.model)
        model_info = next((m for m in self.backend.metadata.get("models", []) if m["id"] == model), None)
        
        if not model_info:
            return {
                "test_type": APITestType.COST_EFFICIENCY.value,
                "error": f"Model information not found for {model}"
            }
        
        input_cost_per_1k = model_info.get("cost_per_1k_input_tokens", 0)
        output_cost_per_1k = model_info.get("cost_per_1k_output_tokens", 0)
        
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0
        latencies = []
        responses = []
        
        for i in range(iterations):
            try:
                start_time = time.time()
                response = self.backend.chat_completion(messages, **kwargs)
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(latency)
                
                # Extract token counts
                usage = response.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # Calculate cost
                input_cost = (input_tokens / 1000) * input_cost_per_1k
                output_cost = (output_tokens / 1000) * output_cost_per_1k
                request_cost = input_cost + output_cost
                total_cost += request_cost
                
                responses.append({
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "request_cost": request_cost,
                    "latency_ms": latency
                })
                
            except Exception as e:
                logger.error(f"Error in cost efficiency test iteration {i+1}/{iterations}: {e}")
        
        # Calculate statistics
        if responses:
            avg_input_tokens = total_input_tokens / len(responses)
            avg_output_tokens = total_output_tokens / len(responses)
            avg_cost = total_cost / len(responses)
            avg_latency = sum(latencies) / len(latencies)
            
            # Cost efficiency metrics
            cost_per_second = avg_cost / (avg_latency / 1000)
            tokens_per_dollar = (avg_input_tokens + avg_output_tokens) / avg_cost if avg_cost > 0 else 0
            output_tokens_per_dollar = avg_output_tokens / avg_cost if avg_cost > 0 else 0
            tokens_per_second = (avg_input_tokens + avg_output_tokens) / (avg_latency / 1000)
            output_tokens_per_second = avg_output_tokens / (avg_latency / 1000)
            
            results = {
                "test_type": APITestType.COST_EFFICIENCY.value,
                "model": model,
                "iterations": iterations,
                "successful_iterations": len(responses),
                "avg_input_tokens": avg_input_tokens,
                "avg_output_tokens": avg_output_tokens,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "input_cost_per_1k": input_cost_per_1k,
                "output_cost_per_1k": output_cost_per_1k,
                "total_cost": total_cost,
                "avg_cost": avg_cost,
                "avg_latency_ms": avg_latency,
                "cost_efficiency_metrics": {
                    "cost_per_second": cost_per_second,
                    "tokens_per_dollar": tokens_per_dollar,
                    "output_tokens_per_dollar": output_tokens_per_dollar,
                    "tokens_per_second": tokens_per_second,
                    "output_tokens_per_second": output_tokens_per_second
                },
                "detailed_responses": responses
            }
        else:
            results = {
                "test_type": APITestType.COST_EFFICIENCY.value,
                "model": model,
                "iterations": iterations,
                "successful_iterations": 0,
                "error": "No successful responses"
            }
        
        # Store results
        self.results["tests"][f"cost_efficiency_{int(time.time())}"] = results
        
        # Save results
        self._save_results()
        
        # Submit results to coordinator
        if self.coordinator_client:
            self._submit_to_coordinator("cost_efficiency", results)
        
        return results
    
    def run_all_tests(self, 
                       messages: List[Dict[str, str]], 
                       **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Run all available tests.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with test results
        """
        results = {}
        
        # Extract test-specific parameters
        latency_kwargs = kwargs.pop("latency", {})
        throughput_kwargs = kwargs.pop("throughput", {})
        reliability_kwargs = kwargs.pop("reliability", {})
        cost_efficiency_kwargs = kwargs.pop("cost_efficiency", {})
        
        # Run tests
        results["latency"] = self.run_latency_test(
            messages, 
            **{**kwargs, **latency_kwargs}
        )
        
        results["throughput"] = self.run_throughput_test(
            messages, 
            **{**kwargs, **throughput_kwargs}
        )
        
        results["reliability"] = self.run_reliability_test(
            messages, 
            **{**kwargs, **reliability_kwargs}
        )
        
        results["cost_efficiency"] = self.run_cost_efficiency_test(
            messages, 
            **{**kwargs, **cost_efficiency_kwargs}
        )
        
        return results
    
    def compare_apis(self, 
                     providers: List[str], 
                     messages: List[Dict[str, str]], 
                     test_type: str = "latency",
                     **kwargs) -> Dict[str, Any]:
        """
        Compare multiple API providers.
        
        Args:
            providers: List of API provider names
            messages: List of message dictionaries
            test_type: Type of test to run
            **kwargs: Additional parameters
            
        Returns:
            Comparison results
        """
        results = {}
        
        for provider in providers:
            try:
                backend = APIBackendFactory.create_backend(provider)
                tester = APITester(
                    backend=backend,
                    enable_anomaly_detection=self.enable_anomaly_detection,
                    enable_predictive_analytics=self.enable_predictive_analytics,
                    results_dir=self.results_dir
                )
                
                if test_type == "latency":
                    results[provider] = tester.run_latency_test(messages, **kwargs)
                elif test_type == "throughput":
                    results[provider] = tester.run_throughput_test(messages, **kwargs)
                elif test_type == "reliability":
                    results[provider] = tester.run_reliability_test(messages, **kwargs)
                elif test_type == "cost_efficiency":
                    results[provider] = tester.run_cost_efficiency_test(messages, **kwargs)
                elif test_type == "all":
                    results[provider] = tester.run_all_tests(messages, **kwargs)
                else:
                    results[provider] = {"error": f"Unsupported test type: {test_type}"}
            except Exception as e:
                logger.error(f"Error testing provider {provider}: {e}")
                results[provider] = {"error": str(e)}
        
        # Store comparison results
        comparison_id = f"comparison_{int(time.time())}"
        self.results["comparisons"] = self.results.get("comparisons", {})
        self.results["comparisons"][comparison_id] = {
            "providers": providers,
            "test_type": test_type,
            "results": results
        }
        
        # Save results
        self._save_results()
        
        return results
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get test results.
        
        Returns:
            Test results
        """
        return self.results
    
    def _save_results(self) -> None:
        """Save test results to file."""
        file_path = os.path.join(self.results_dir, f"{self.test_id}.json")
        
        with open(file_path, "w") as f:
            json.dump(self.results, f, indent=2)
    
    def _submit_to_coordinator(self, test_type: str, results: Dict[str, Any]) -> None:
        """
        Submit test results to distributed testing coordinator.
        
        Args:
            test_type: Type of test
            results: Test results
        """
        if not self.coordinator_client:
            return
        
        try:
            task_result = TaskResult(
                task_id=f"{self.test_id}_{test_type}",
                status=TaskStatus.COMPLETED,
                result=results,
                metadata={
                    "provider": self.backend.metadata["name"],
                    "model": self.backend.model,
                    "test_type": test_type
                }
            )
            
            self.coordinator_client.submit_result(task_result)
        except Exception as e:
            logger.error(f"Error submitting results to coordinator: {e}")


class APIDistributedTesting:
    """
    API Distributed Testing framework for integrating API testing
    with the distributed testing infrastructure.
    """
    
    def __init__(self, coordinator_url: Optional[str] = None):
        """
        Initialize the API Distributed Testing framework.
        
        Args:
            coordinator_url: URL of the distributed testing coordinator
        """
        self.coordinator_url = coordinator_url
        self.coordinator_client = None
        
        if coordinator_url:
            try:
                self.coordinator_client = CoordinatorClient(coordinator_url)
                logger.info(f"Connected to coordinator at {coordinator_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to coordinator: {e}")
        
        self.registered_backends = {}
        self.active_tests = {}
        
        # Initialize anomaly detection and predictive analytics
        self.anomaly_detector = AnomalyDetector()
        self.predictor = TimeSeriesPredictor()
    
    def register_api_backend(self, 
                              provider: Union[str, APIProvider], 
                              api_key: Optional[str] = None,
                              model: Optional[str] = None,
                              **kwargs) -> str:
        """
        Register an API backend for testing.
        
        Args:
            provider: API provider name or enum
            api_key: API key for authentication
            model: Model to use for testing
            **kwargs: Additional parameters for the backend
            
        Returns:
            Backend ID
        """
        backend = APIBackendFactory.create_backend(provider, api_key=api_key, model=model, **kwargs)
        backend_id = f"{backend.metadata['name']}_{str(uuid.uuid4())[:8]}"
        
        self.registered_backends[backend_id] = backend
        
        logger.info(f"Registered API backend: {backend_id} ({backend.metadata['name']})")
        
        return backend_id
    
    def run_distributed_test(self,
                              api_type: Union[str, APIProvider],
                              test_type: Union[str, APITestType],
                              parameters: Dict[str, Any],
                              num_workers: int = 1,
                              worker_tags: Optional[List[str]] = None) -> str:
        """
        Run a distributed test using the distributed testing framework.
        
        Args:
            api_type: API provider name or enum
            test_type: Test type name or enum
            parameters: Test parameters
            num_workers: Number of workers to allocate
            worker_tags: Tags for worker selection
            
        Returns:
            Test ID
        """
        if not self.coordinator_client:
            raise ValueError("Coordinator client not initialized. Cannot run distributed test.")
        
        # Convert enum values to strings if needed
        if isinstance(api_type, APIProvider):
            api_type = api_type.value
        
        if isinstance(test_type, APITestType):
            test_type = test_type.value
        
        # Create test ID
        test_id = f"api_test_{api_type}_{test_type}_{str(uuid.uuid4())[:8]}"
        
        # Prepare task
        task = Task(
            task_id=test_id,
            task_type="api_test",
            parameters={
                "api_type": api_type,
                "test_type": test_type,
                "test_parameters": parameters
            },
            priority=10,
            tags=worker_tags or [],
            metadata={
                "api_type": api_type,
                "test_type": test_type,
                "num_workers": num_workers
            }
        )
        
        # Submit task to coordinator
        try:
            self.coordinator_client.submit_task(task)
            self.active_tests[test_id] = task
            
            logger.info(f"Submitted distributed test: {test_id}")
            
            return test_id
        except Exception as e:
            logger.error(f"Error submitting task to coordinator: {e}")
            raise
    
    def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the results of a distributed test.
        
        Args:
            test_id: Test ID
            
        Returns:
            Test results or None if not found
        """
        if not self.coordinator_client:
            raise ValueError("Coordinator client not initialized. Cannot get test results.")
        
        try:
            result = self.coordinator_client.get_task_result(test_id)
            
            if not result:
                return None
            
            return result.result
        except Exception as e:
            logger.error(f"Error getting test results: {e}")
            return None
    
    def compare_apis(self,
                     api_types: List[Union[str, APIProvider]],
                     test_type: Union[str, APITestType],
                     parameters: Dict[str, Any],
                     num_workers_per_api: int = 1) -> str:
        """
        Compare multiple API providers using distributed testing.
        
        Args:
            api_types: List of API provider names or enums
            test_type: Test type name or enum
            parameters: Test parameters
            num_workers_per_api: Number of workers to allocate per API
            
        Returns:
            Comparison ID
        """
        comparison_id = f"api_comparison_{str(uuid.uuid4())[:8]}"
        test_ids = {}
        
        for api_type in api_types:
            test_id = self.run_distributed_test(
                api_type=api_type,
                test_type=test_type,
                parameters=parameters,
                num_workers=num_workers_per_api
            )
            test_ids[api_type.value if isinstance(api_type, APIProvider) else api_type] = test_id
        
        # Store comparison metadata
        if self.coordinator_client:
            try:
                self.coordinator_client.set_metadata(
                    f"comparison:{comparison_id}",
                    {
                        "api_types": [api.value if isinstance(api, APIProvider) else api for api in api_types],
                        "test_type": test_type.value if isinstance(test_type, APITestType) else test_type,
                        "test_ids": test_ids,
                        "created_at": datetime.datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"Error storing comparison metadata: {e}")
        
        return comparison_id
    
    def get_comparison_results(self, comparison_id: str) -> Dict[str, Any]:
        """
        Get the results of an API comparison.
        
        Args:
            comparison_id: Comparison ID
            
        Returns:
            Comparison results
        """
        if not self.coordinator_client:
            raise ValueError("Coordinator client not initialized. Cannot get comparison results.")
        
        try:
            metadata = self.coordinator_client.get_metadata(f"comparison:{comparison_id}")
            
            if not metadata:
                return {"error": f"Comparison not found: {comparison_id}"}
            
            results = {}
            
            for api_type, test_id in metadata.get("test_ids", {}).items():
                test_result = self.get_test_results(test_id)
                results[api_type] = test_result or {"status": "pending"}
            
            return {
                "comparison_id": comparison_id,
                "api_types": metadata.get("api_types", []),
                "test_type": metadata.get("test_type"),
                "created_at": metadata.get("created_at"),
                "results": results
            }
        except Exception as e:
            logger.error(f"Error getting comparison results: {e}")
            return {"error": str(e)}
    
    def get_metrics(self, 
                     api_type: Optional[Union[str, APIProvider]] = None,
                     metric_type: Optional[str] = None,
                     days: int = 7) -> Dict[str, Any]:
        """
        Get performance metrics for API providers.
        
        Args:
            api_type: API provider name or enum
            metric_type: Metric type (latency, throughput, reliability, cost_efficiency)
            days: Number of days to include
            
        Returns:
            Metrics data
        """
        if not self.coordinator_client:
            raise ValueError("Coordinator client not initialized. Cannot get metrics.")
        
        # Convert enum value to string if needed
        if isinstance(api_type, APIProvider):
            api_type = api_type.value
        
        try:
            query = {}
            
            if api_type:
                query["provider"] = api_type
            
            if metric_type:
                query["metric_type"] = metric_type
            
            # Add time range
            start_time = datetime.datetime.now() - datetime.timedelta(days=days)
            query["created_at"] = {"$gte": start_time.isoformat()}
            
            metrics = self.coordinator_client.query_results(query)
            
            # Process metrics for easier consumption
            processed_metrics = self._process_metrics(metrics)
            
            # Add predictions if enabled
            if hasattr(self, 'predictor') and self.predictor:
                processed_metrics["predictions"] = self._generate_predictions(processed_metrics)
            
            return processed_metrics
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"error": str(e)}
    
    def _process_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process raw metrics data into a structured format.
        
        Args:
            metrics: Raw metrics data
            
        Returns:
            Processed metrics
        """
        result = {
            "api_metrics": {},
            "global_metrics": {
                "total_tests": len(metrics),
                "providers": set(),
                "test_types": set()
            }
        }
        
        for metric in metrics:
            provider = metric.get("metadata", {}).get("provider")
            test_type = metric.get("metadata", {}).get("test_type")
            created_at = metric.get("created_at")
            test_result = metric.get("result", {})
            
            if not provider or not test_type:
                continue
            
            # Update global metrics
            result["global_metrics"]["providers"].add(provider)
            result["global_metrics"]["test_types"].add(test_type)
            
            # Initialize provider metrics if needed
            if provider not in result["api_metrics"]:
                result["api_metrics"][provider] = {}
            
            # Initialize test type metrics if needed
            if test_type not in result["api_metrics"][provider]:
                result["api_metrics"][provider][test_type] = []
            
            # Add test result
            result["api_metrics"][provider][test_type].append({
                "created_at": created_at,
                "data": test_result
            })
        
        # Convert sets to lists
        result["global_metrics"]["providers"] = list(result["global_metrics"]["providers"])
        result["global_metrics"]["test_types"] = list(result["global_metrics"]["test_types"])
        
        return result
    
    def _generate_predictions(self, processed_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate predictions for API metrics.
        
        Args:
            processed_metrics: Processed metrics data
            
        Returns:
            Predictions
        """
        predictions = {}
        
        for provider, provider_metrics in processed_metrics.get("api_metrics", {}).items():
            predictions[provider] = {}
            
            for test_type, test_results in provider_metrics.items():
                # Extract time series data
                time_points = []
                values = []
                
                for result in sorted(test_results, key=lambda x: x.get("created_at", "")):
                    created_at = result.get("created_at")
                    data = result.get("data", {})
                    
                    if not created_at:
                        continue
                    
                    # Extract relevant value based on test type
                    value = None
                    if test_type == "latency" and "mean_latency_ms" in data:
                        value = data["mean_latency_ms"]
                    elif test_type == "throughput" and "throughput_rps" in data:
                        value = data["throughput_rps"]
                    elif test_type == "reliability" and "success_rate" in data:
                        value = data["success_rate"]
                    elif test_type == "cost_efficiency" and "cost_efficiency_metrics" in data:
                        value = data.get("cost_efficiency_metrics", {}).get("tokens_per_dollar")
                    
                    if value is not None:
                        time_points.append(created_at)
                        values.append(value)
                
                # Generate predictions if enough data points
                if len(values) >= 5:
                    pred_result = self.predictor.predict_timeseries(provider, test_type, time_points, values)
                    predictions[provider][test_type] = pred_result
        
        return predictions


def main():
    """Command-line interface for API Distributed Testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='API Distributed Testing Framework')
    
    # API provider selection
    parser.add_argument('--api-type', type=str, help='API provider type')
    parser.add_argument('--model', type=str, help='Model to use')
    
    # Test selection
    parser.add_argument('--test-type', type=str, choices=[t.value for t in APITestType], help='Test type')
    
    # Comparison mode
    parser.add_argument('--compare', type=str, help='Compare multiple APIs (comma-separated)')
    
    # Distributed execution
    parser.add_argument('--coordinator', type=str, help='Coordinator URL')
    parser.add_argument('--distributed', action='store_true', help='Run in distributed mode')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    
    # Advanced features
    parser.add_argument('--enable-anomaly-detection', action='store_true', help='Enable anomaly detection')
    parser.add_argument('--enable-predictive-analytics', action='store_true', help='Enable predictive analytics')
    
    # Test parameters
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations for latency/reliability tests')
    parser.add_argument('--duration', type=int, default=10, help='Duration in seconds for throughput test')
    parser.add_argument('--concurrency', type=int, default=5, help='Concurrency for throughput test')
    
    # Input
    parser.add_argument('--prompt', type=str, help='Text prompt to use for testing')
    parser.add_argument('--prompt-file', type=str, help='File containing prompt to use for testing')
    
    # Result handling
    parser.add_argument('--output-dir', type=str, default='./api_test_results', help='Directory for test results')
    parser.add_argument('--report', action='store_true', help='Generate HTML report')
    
    # Listing
    parser.add_argument('--list-providers', action='store_true', help='List available API providers')
    parser.add_argument('--list-models', action='store_true', help='List available models for provider')
    parser.add_argument('--list-tests', action='store_true', help='List available test types')
    
    args = parser.parse_args()
    
    # Handle listing requests
    if args.list_providers:
        print("Available API providers:")
        for provider in APIProvider:
            print(f"  {provider.value}")
        return
    
    if args.list_tests:
        print("Available test types:")
        for test_type in APITestType:
            print(f"  {test_type.value}")
        return
    
    if args.list_models and args.api_type:
        try:
            backend = APIBackendFactory.create_backend(args.api_type)
            models = backend.get_models()
            
            print(f"Available models for {args.api_type}:")
            for model in models:
                print(f"  {model['id']}")
                
                if "capabilities" in model:
                    print(f"    Capabilities: {', '.join(model['capabilities'])}")
                
                if "context_window" in model:
                    print(f"    Context window: {model['context_window']} tokens")
                
                if "cost_per_1k_input_tokens" in model:
                    print(f"    Cost per 1K input tokens: ${model['cost_per_1k_input_tokens']:.6f}")
                
                if "cost_per_1k_output_tokens" in model:
                    print(f"    Cost per 1K output tokens: ${model['cost_per_1k_output_tokens']:.6f}")
                
                print()
        except Exception as e:
            print(f"Error listing models: {e}")
        return
    
    # Get prompt
    prompt = args.prompt
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r') as f:
                prompt = f.read().strip()
        except Exception as e:
            print(f"Error reading prompt file: {e}")
            return
    
    if not prompt:
        prompt = "Explain quantum computing in simple terms."
    
    messages = [{"role": "user", "content": prompt}]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Comparison mode
    if args.compare:
        providers = args.compare.split(',')
        
        if args.distributed and args.coordinator:
            # Distributed comparison
            api_testing = APIDistributedTesting(coordinator_url=args.coordinator)
            
            comparison_id = api_testing.compare_apis(
                api_types=providers,
                test_type=args.test_type or "latency",
                parameters={
                    "messages": messages,
                    "iterations": args.iterations,
                    "duration": args.duration,
                    "concurrent_requests": args.concurrency
                },
                num_workers_per_api=args.workers
            )
            
            print(f"Distributed comparison started with ID: {comparison_id}")
            print(f"Check results with: --get-comparison {comparison_id}")
        else:
            # Local comparison
            tester = APITester(
                enable_anomaly_detection=args.enable_anomaly_detection,
                enable_predictive_analytics=args.enable_predictive_analytics,
                results_dir=args.output_dir
            )
            
            results = tester.compare_apis(
                providers=providers,
                messages=messages,
                test_type=args.test_type or "latency",
                iterations=args.iterations,
                duration=args.duration,
                concurrent_requests=args.concurrency
            )
            
            print("Comparison results:")
            for provider, result in results.items():
                print(f"\n{provider}:")
                if "error" in result:
                    print(f"  Error: {result['error']}")
                else:
                    _print_test_summary(result)
    
    # Single API test
    elif args.api_type:
        if args.distributed and args.coordinator:
            # Distributed test
            api_testing = APIDistributedTesting(coordinator_url=args.coordinator)
            
            test_id = api_testing.run_distributed_test(
                api_type=args.api_type,
                test_type=args.test_type or "latency",
                parameters={
                    "messages": messages,
                    "iterations": args.iterations,
                    "duration": args.duration,
                    "concurrent_requests": args.concurrency,
                    "model": args.model
                },
                num_workers=args.workers
            )
            
            print(f"Distributed test started with ID: {test_id}")
            print(f"Check results with: --get-test {test_id}")
        else:
            # Local test
            try:
                backend = APIBackendFactory.create_backend(args.api_type, model=args.model)
                tester = APITester(
                    backend=backend,
                    enable_anomaly_detection=args.enable_anomaly_detection,
                    enable_predictive_analytics=args.enable_predictive_analytics,
                    results_dir=args.output_dir
                )
                
                if args.test_type == "latency":
                    result = tester.run_latency_test(messages, iterations=args.iterations)
                elif args.test_type == "throughput":
                    result = tester.run_throughput_test(messages, duration=args.duration, concurrent_requests=args.concurrency)
                elif args.test_type == "reliability":
                    result = tester.run_reliability_test(messages, iterations=args.iterations)
                elif args.test_type == "cost_efficiency":
                    result = tester.run_cost_efficiency_test(messages, iterations=args.iterations)
                elif args.test_type == "all" or not args.test_type:
                    result = tester.run_all_tests(messages, 
                                                 latency={"iterations": args.iterations},
                                                 throughput={"duration": args.duration, "concurrent_requests": args.concurrency},
                                                 reliability={"iterations": args.iterations},
                                                 cost_efficiency={"iterations": args.iterations})
                else:
                    print(f"Unsupported test type: {args.test_type}")
                    return
                
                print("Test results:")
                if isinstance(result, dict) and args.test_type == "all":
                    for test_name, test_result in result.items():
                        print(f"\n{test_name.capitalize()}:")
                        _print_test_summary(test_result)
                else:
                    _print_test_summary(result)
                
                # Print save location
                print(f"\nResults saved to: {os.path.join(args.output_dir, tester.test_id + '.json')}")
                
            except Exception as e:
                print(f"Error running test: {e}")
                traceback.print_exc()
    else:
        parser.print_help()


def _print_test_summary(result):
    """Print a summary of test results."""
    if "error" in result:
        print(f"  Error: {result['error']}")
        return
    
    test_type = result.get("test_type")
    
    if test_type == "latency":
        print(f"  Iterations: {result.get('iterations', 0)}")
        print(f"  Successful: {result.get('successful_iterations', 0)}")
        print(f"  Failed: {result.get('failed_iterations', 0)}")
        
        if "mean_latency_ms" in result:
            print(f"  Mean latency: {result['mean_latency_ms']:.2f} ms")
        
        if "median_latency_ms" in result:
            print(f"  Median latency: {result['median_latency_ms']:.2f} ms")
        
        if "percentiles_ms" in result:
            print(f"  95th percentile: {result['percentiles_ms'].get('95', 0):.2f} ms")
            print(f"  99th percentile: {result['percentiles_ms'].get('99', 0):.2f} ms")
    
    elif test_type == "throughput":
        print(f"  Duration: {result.get('duration_seconds', 0):.2f} seconds")
        print(f"  Concurrent requests: {result.get('concurrent_requests', 0)}")
        print(f"  Requests completed: {result.get('requests_completed', 0)}")
        print(f"  Errors: {result.get('errors', 0)}")
        
        if "throughput_rps" in result:
            print(f"  Throughput: {result['throughput_rps']:.2f} requests/second")
        
        if "mean_latency_ms" in result:
            print(f"  Mean latency: {result['mean_latency_ms']:.2f} ms")
        
        if "latency_percentiles_ms" in result:
            print(f"  95th percentile latency: {result['latency_percentiles_ms'].get('95', 0):.2f} ms")
    
    elif test_type == "reliability":
        print(f"  Iterations: {result.get('iterations', 0)}")
        print(f"  Successes: {result.get('successes', 0)}")
        print(f"  Failures: {result.get('failures', 0)}")
        
        if "success_rate" in result:
            print(f"  Success rate: {result['success_rate']:.1%}")
        
        if "error_rate" in result:
            print(f"  Error rate: {result['error_rate']:.1%}")
    
    elif test_type == "cost_efficiency":
        print(f"  Model: {result.get('model', 'unknown')}")
        print(f"  Iterations: {result.get('iterations', 0)}")
        print(f"  Successful iterations: {result.get('successful_iterations', 0)}")
        
        if "avg_input_tokens" in result:
            print(f"  Avg input tokens: {result['avg_input_tokens']:.1f}")
        
        if "avg_output_tokens" in result:
            print(f"  Avg output tokens: {result['avg_output_tokens']:.1f}")
        
        if "avg_cost" in result:
            print(f"  Avg cost per request: ${result['avg_cost']:.6f}")
        
        if "cost_efficiency_metrics" in result:
            metrics = result["cost_efficiency_metrics"]
            if "tokens_per_dollar" in metrics:
                print(f"  Tokens per dollar: {metrics['tokens_per_dollar']:.1f}")
            if "output_tokens_per_dollar" in metrics:
                print(f"  Output tokens per dollar: {metrics['output_tokens_per_dollar']:.1f}")
            if "tokens_per_second" in metrics:
                print(f"  Tokens per second: {metrics['tokens_per_second']:.1f}")
    
    # Print anomalies if detected
    if "anomalies" in result:
        print("  Anomalies detected:")
        for anomaly in result["anomalies"]:
            print(f"    - {anomaly.get('type', 'unknown')} (confidence: {anomaly.get('confidence', 0):.2f})")
    
    if "throughput_anomalies" in result:
        print("  Throughput anomalies detected:")
        for anomaly in result["throughput_anomalies"]:
            print(f"    - {anomaly.get('type', 'unknown')} (confidence: {anomaly.get('confidence', 0):.2f})")
    
    if "latency_anomalies" in result:
        print("  Latency anomalies detected:")
        for anomaly in result["latency_anomalies"]:
            print(f"    - {anomaly.get('type', 'unknown')} (confidence: {anomaly.get('confidence', 0):.2f})")


if __name__ == "__main__":
    main()