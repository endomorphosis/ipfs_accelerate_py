"""
Prometheus metrics integration.
"""

import time
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Prometheus metrics for model server."""
    
    def __init__(self):
        """Initialize Prometheus metrics."""
        # Request metrics
        self.requests_total = Counter(
            "hf_server_requests_total",
            "Total number of requests",
            ["model", "endpoint", "status"]
        )
        
        self.request_duration = Histogram(
            "hf_server_request_duration_seconds",
            "Request duration in seconds",
            ["model", "endpoint"]
        )
        
        # Model metrics
        self.models_loaded = Gauge(
            "hf_server_models_loaded",
            "Number of models currently loaded"
        )
        
        self.model_load_duration = Histogram(
            "hf_server_model_load_duration_seconds",
            "Model loading duration in seconds",
            ["model"]
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            "hf_server_cache_hits_total",
            "Total cache hits",
            ["cache_type"]
        )
        
        self.cache_misses = Counter(
            "hf_server_cache_misses_total",
            "Total cache misses",
            ["cache_type"]
        )
        
        # Error metrics
        self.errors_total = Counter(
            "hf_server_errors_total",
            "Total number of errors",
            ["model", "error_type"]
        )
        
        # Hardware metrics
        self.hardware_utilization = Gauge(
            "hf_server_hardware_utilization",
            "Hardware utilization percentage",
            ["hardware", "device"]
        )
        
        # IPFS metrics
        self.ipfs_operations_total = Counter(
            "hf_server_ipfs_operations_total",
            "Total IPFS operations",
            ["operation", "backend", "status"]
        )
        
        self.ipfs_operation_duration = Histogram(
            "hf_server_ipfs_operation_duration_seconds",
            "IPFS operation duration in seconds",
            ["operation", "backend"]
        )
        
        # Memory metrics
        self.memory_used_mb = Gauge(
            "hf_server_memory_used_mb",
            "Memory used by loaded models in MB"
        )
        
        self.memory_limit_mb = Gauge(
            "hf_server_memory_limit_mb",
            "Memory limit for model cache in MB"
        )
    
    def record_request(self, model: str, endpoint: str, duration: float, status: str):
        """Record a request."""
        self.requests_total.labels(model=model, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(model=model, endpoint=endpoint).observe(duration)
    
    def record_model_load(self, model: str, duration: float):
        """Record model loading."""
        self.model_load_duration.labels(model=model).observe(duration)
    
    def update_models_loaded(self, count: int):
        """Update loaded models count."""
        self.models_loaded.set(count)
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_error(self, model: str, error_type: str):
        """Record an error."""
        self.errors_total.labels(model=model, error_type=error_type).inc()
    
    def update_hardware_utilization(self, hardware: str, device: str, utilization: float):
        """Update hardware utilization."""
        self.hardware_utilization.labels(hardware=hardware, device=device).set(utilization)
    
    def record_ipfs_operation(self, operation: str, backend: str, duration: float, status: str):
        """Record IPFS operation."""
        self.ipfs_operations_total.labels(operation=operation, backend=backend, status=status).inc()
        self.ipfs_operation_duration.labels(operation=operation, backend=backend).observe(duration)
    
    def update_memory_metrics(self, used_mb: float, limit_mb: float):
        """Update memory metrics."""
        self.memory_used_mb.set(used_mb)
        self.memory_limit_mb.set(limit_mb)
    
    def generate_metrics(self) -> bytes:
        """Generate Prometheus metrics output."""
        return generate_latest()
    
    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST
