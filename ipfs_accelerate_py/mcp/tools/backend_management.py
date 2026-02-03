"""
Enhanced MCP Tools for Unified Inference Backend

Provides MCP tool integration with the unified inference backend manager:
- Backend discovery and status via MCP
- Multi-backend inference routing
- Real-time monitoring
- Load balancing control
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Import backend manager
try:
    from ...inference_backend_manager import get_backend_manager, BackendType, BackendStatus
    HAVE_BACKEND_MANAGER = True
except ImportError:
    logger.warning("Backend manager not available for MCP integration")
    HAVE_BACKEND_MANAGER = False


def list_inference_backends(
    backend_type: Optional[str] = None,
    status: Optional[str] = None,
    task: Optional[str] = None
) -> Dict[str, Any]:
    """
    List available inference backends
    
    MCP Tool: Lists all registered inference backends with optional filtering
    
    Args:
        backend_type: Filter by backend type (gpu, api, cli, p2p, websocket, mcp)
        status: Filter by status (healthy, degraded, unhealthy, offline)
        task: Filter by supported task (e.g., text-generation, text-embedding)
        
    Returns:
        Dictionary with list of backends and their details
    """
    if not HAVE_BACKEND_MANAGER:
        return {
            "error": "Backend manager not available",
            "backends": []
        }
    
    try:
        manager = get_backend_manager()
        
        # Convert string filters to enums
        backend_type_filter = None
        if backend_type:
            try:
                backend_type_filter = BackendType(backend_type.lower())
            except ValueError:
                return {
                    "error": f"Invalid backend type: {backend_type}",
                    "valid_types": [t.value for t in BackendType]
                }
        
        status_filter = None
        if status:
            try:
                status_filter = BackendStatus(status.lower())
            except ValueError:
                return {
                    "error": f"Invalid status: {status}",
                    "valid_statuses": [s.value for s in BackendStatus]
                }
        
        # Get filtered backends
        backends = manager.list_backends(
            backend_type=backend_type_filter,
            status=status_filter,
            task=task
        )
        
        # Format response
        result = {
            "total_backends": len(backends),
            "backends": [
                {
                    "id": b.backend_id,
                    "name": b.name,
                    "type": b.backend_type.value,
                    "status": b.status.value,
                    "endpoint": b.endpoint,
                    "supported_tasks": list(b.capabilities.supported_tasks),
                    "supported_models": list(b.capabilities.supported_models) if b.capabilities.supported_models else None,
                    "protocols": list(b.capabilities.protocols),
                    "hardware_types": list(b.capabilities.hardware_types) if b.capabilities.hardware_types else None,
                    "metrics": {
                        "total_requests": b.metrics.total_requests,
                        "successful_requests": b.metrics.successful_requests,
                        "failed_requests": b.metrics.failed_requests,
                        "average_latency_ms": round(b.metrics.average_latency_ms, 2),
                        "queue_size": b.metrics.current_queue_size,
                        "models_loaded": b.metrics.models_loaded
                    }
                }
                for b in backends
            ],
            "timestamp": time.time()
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error listing backends: {e}")
        return {
            "error": str(e),
            "backends": []
        }


def get_backend_status() -> Dict[str, Any]:
    """
    Get comprehensive status of all inference backends
    
    MCP Tool: Returns detailed status report including metrics and health
    
    Returns:
        Comprehensive status dictionary
    """
    if not HAVE_BACKEND_MANAGER:
        return {
            "error": "Backend manager not available",
            "status": "unavailable"
        }
    
    try:
        manager = get_backend_manager()
        return manager.get_backend_status_report()
    
    except Exception as e:
        logger.error(f"Error getting backend status: {e}")
        return {
            "error": str(e),
            "status": "error"
        }


def select_backend_for_inference(
    task: str,
    model: Optional[str] = None,
    preferred_types: Optional[List[str]] = None,
    required_protocols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Select the best backend for an inference task
    
    MCP Tool: Uses intelligent routing to select optimal backend
    
    Args:
        task: The inference task type (e.g., text-generation)
        model: Optional specific model requirement
        preferred_types: Preferred backend types in priority order
        required_protocols: Required protocol support (e.g., websocket)
        
    Returns:
        Selected backend information or error
    """
    if not HAVE_BACKEND_MANAGER:
        return {
            "error": "Backend manager not available"
        }
    
    try:
        manager = get_backend_manager()
        
        # Convert preferred types to BackendType enums
        preferred_backend_types = None
        if preferred_types:
            try:
                preferred_backend_types = [
                    BackendType(t.lower()) for t in preferred_types
                ]
            except ValueError as e:
                return {
                    "error": f"Invalid backend type: {e}",
                    "valid_types": [t.value for t in BackendType]
                }
        
        # Select backend
        backend = manager.select_backend_for_task(
            task=task,
            model=model,
            preferred_types=preferred_backend_types,
            required_protocols=required_protocols
        )
        
        if not backend:
            return {
                "error": f"No suitable backend found for task: {task}",
                "task": task,
                "model": model,
                "available_backends": len(manager.list_backends(task=task))
            }
        
        # Return backend info
        return {
            "backend_id": backend.backend_id,
            "name": backend.name,
            "type": backend.backend_type.value,
            "status": backend.status.value,
            "endpoint": backend.endpoint,
            "capabilities": {
                "tasks": list(backend.capabilities.supported_tasks),
                "models": list(backend.capabilities.supported_models) if backend.capabilities.supported_models else None,
                "protocols": list(backend.capabilities.protocols),
                "streaming": backend.capabilities.supports_streaming,
                "batching": backend.capabilities.supports_batching
            },
            "current_metrics": {
                "queue_size": backend.metrics.current_queue_size,
                "average_latency_ms": round(backend.metrics.average_latency_ms, 2)
            }
        }
    
    except Exception as e:
        logger.error(f"Error selecting backend: {e}")
        return {
            "error": str(e)
        }


def route_inference_request(
    task: str,
    model: str,
    inputs: Any,
    parameters: Optional[Dict[str, Any]] = None,
    backend_id: Optional[str] = None,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """
    Route an inference request to the appropriate backend
    
    MCP Tool: Submits inference request with automatic backend selection
    
    Args:
        task: Inference task type
        model: Model to use
        inputs: Input data
        parameters: Optional inference parameters
        backend_id: Optional specific backend to use
        timeout: Request timeout in seconds
        
    Returns:
        Inference result or error
    """
    if not HAVE_BACKEND_MANAGER:
        return {
            "error": "Backend manager not available"
        }
    
    try:
        manager = get_backend_manager()
        
        # Select backend if not specified
        if backend_id:
            backend = manager.get_backend(backend_id)
            if not backend:
                return {
                    "error": f"Backend not found: {backend_id}"
                }
        else:
            backend = manager.select_backend_for_task(task, model)
            if not backend:
                return {
                    "error": f"No suitable backend found for task: {task}"
                }
        
        # Record start time
        start_time = time.time()
        
        # TODO: Actually route the request to the backend
        # For now, return a mock response
        result = {
            "status": "pending",
            "message": f"Request routed to backend: {backend.name}",
            "backend_id": backend.backend_id,
            "backend_name": backend.name,
            "backend_type": backend.backend_type.value,
            "task": task,
            "model": model,
            "note": "Actual inference execution not yet implemented in MCP integration"
        }
        
        # Record metrics
        latency_ms = (time.time() - start_time) * 1000
        manager.record_request(backend.backend_id, True, latency_ms)
        
        return result
    
    except Exception as e:
        logger.error(f"Error routing inference request: {e}")
        return {
            "error": str(e)
        }


def get_supported_tasks() -> Dict[str, Any]:
    """
    Get list of all supported inference tasks
    
    MCP Tool: Returns tasks that can be handled by available backends
    
    Returns:
        Dictionary with supported tasks and backend counts
    """
    if not HAVE_BACKEND_MANAGER:
        return {
            "error": "Backend manager not available",
            "tasks": []
        }
    
    try:
        manager = get_backend_manager()
        status = manager.get_backend_status_report()
        
        # Count backends per task
        task_backends = {}
        for backend in status["backends"]:
            for task in backend["tasks"]:
                if task not in task_backends:
                    task_backends[task] = {
                        "backend_count": 0,
                        "backends": []
                    }
                task_backends[task]["backend_count"] += 1
                task_backends[task]["backends"].append({
                    "id": backend["id"],
                    "name": backend["name"],
                    "type": backend["type"]
                })
        
        return {
            "total_tasks": len(task_backends),
            "tasks": task_backends,
            "timestamp": time.time()
        }
    
    except Exception as e:
        logger.error(f"Error getting supported tasks: {e}")
        return {
            "error": str(e),
            "tasks": []
        }


# Register MCP tools if FastMCP is available
try:
    from mcp.server import FastMCP
    
    # These will be registered by the MCP server initialization
    MCP_TOOLS = {
        "list_inference_backends": list_inference_backends,
        "get_backend_status": get_backend_status,
        "select_backend_for_inference": select_backend_for_inference,
        "route_inference_request": route_inference_request,
        "get_supported_tasks": get_supported_tasks
    }
    
except ImportError:
    logger.debug("FastMCP not available for tool registration")
    MCP_TOOLS = {}
