"""
Health check system.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health checker for model server."""
    
    def __init__(self, model_loader=None, hardware_detector=None):
        """
        Initialize health checker.
        
        Args:
            model_loader: ModelLoader instance
            hardware_detector: HardwareDetector instance
        """
        self.model_loader = model_loader
        self.hardware_detector = hardware_detector
        self.start_time = datetime.utcnow()
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Perform basic health check.
        
        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
        }
    
    async def check_readiness(self) -> Dict[str, Any]:
        """
        Perform readiness check (can serve requests).
        
        Returns:
            Readiness status dictionary
        """
        checks = {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
        }
        
        # Check if any models loaded
        if self.model_loader:
            try:
                loaded_models = await self.model_loader.get_loaded_models()
                checks["checks"]["models"] = {
                    "status": "ok",
                    "loaded_count": len(loaded_models),
                }
            except Exception as e:
                checks["checks"]["models"] = {
                    "status": "error",
                    "error": str(e),
                }
                checks["status"] = "not_ready"
        
        # Check hardware
        if self.hardware_detector:
            try:
                available = self.hardware_detector.get_available_hardware()
                checks["checks"]["hardware"] = {
                    "status": "ok",
                    "available": available,
                }
            except Exception as e:
                checks["checks"]["hardware"] = {
                    "status": "error",
                    "error": str(e),
                }
        
        return checks
    
    async def check_detailed(self) -> Dict[str, Any]:
        """
        Perform detailed health check.
        
        Returns:
            Detailed health status
        """
        detailed = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "components": {},
        }
        
        # Model loader status
        if self.model_loader:
            try:
                cache_stats = self.model_loader.get_cache_stats()
                loaded_models = await self.model_loader.get_loaded_models()
                detailed["components"]["model_loader"] = {
                    "status": "ok",
                    "cache_stats": cache_stats,
                    "loaded_models": len(loaded_models),
                }
            except Exception as e:
                detailed["components"]["model_loader"] = {
                    "status": "error",
                    "error": str(e),
                }
                detailed["status"] = "degraded"
        
        # Hardware status
        if self.hardware_detector:
            try:
                available = self.hardware_detector.get_available_hardware()
                detailed["components"]["hardware"] = {
                    "status": "ok",
                    "available_platforms": available,
                }
            except Exception as e:
                detailed["components"]["hardware"] = {
                    "status": "error",
                    "error": str(e),
                }
        
        return detailed
