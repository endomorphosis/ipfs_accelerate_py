"""
IPFS Accelerate MCP Model Tools

This module provides MCP tools for model search, recommendation, and management.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import asdict, is_dataclass

# Set up logging
logger = logging.getLogger("ipfs_accelerate_mcp.tools.models")

# Shared scanner instance
_scanner = None
_model_manager = None
_recommender = None

def _get_scanner():
    """Get or create shared HuggingFaceHubScanner instance."""
    global _scanner
    if _scanner is None:
        from ipfs_accelerate_py.huggingface_hub_scanner import HuggingFaceHubScanner
        _scanner = HuggingFaceHubScanner()
    return _scanner

def _get_model_manager():
    """Get or create shared ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        from ipfs_accelerate_py.model_manager import get_default_model_manager
        _model_manager = get_default_model_manager()
    return _model_manager

def _get_recommender():
    """Get or create shared BanditModelRecommender instance."""
    global _recommender
    if _recommender is None:
        from ipfs_accelerate_py.model_manager import BanditModelRecommender
        _recommender = BanditModelRecommender(model_manager=_get_model_manager())
    return _recommender


# Tool: Search Models
def search_models_tool(query: str, task_filter: Optional[str] = None, 
                       hardware_filter: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    """
    Search for models on HuggingFace Hub
    
    Args:
        query: Search query string
        task_filter: Optional task type filter (e.g., 'text-generation', 'image-classification')
        hardware_filter: Optional hardware filter (e.g., 'cpu', 'gpu')
        limit: Maximum number of results to return (default: 20)
    
    Returns:
        Dictionary with search results and metadata
    """
    try:
        logger.info(f"Searching models: query='{query}', task={task_filter}, hardware={hardware_filter}, limit={limit}")
        
        hub_scanner = _get_scanner()
        results = hub_scanner.search_models(
            query=query,
            task_filter=task_filter,
            hardware_filter=hardware_filter,
            limit=limit
        )
        
        return {
            'status': 'success',
            'results': results,
            'total': len(results),
            'query': query
        }
    except Exception as e:
        logger.error(f"Error searching models: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'results': []
        }


# Tool: Get Model Recommendations
def recommend_models_tool(task_type: str, hardware: str = 'cpu', 
                         performance: str = 'balanced', limit: int = 5) -> Dict[str, Any]:
    """
    Get AI-powered model recommendations using bandit algorithm
    
    Args:
        task_type: Type of task (e.g., 'text-generation', 'image-classification')
        hardware: Target hardware ('cpu' or 'gpu')
        performance: Performance preference ('speed', 'balanced', 'quality')
        limit: Maximum number of recommendations (default: 5)
    
    Returns:
        Dictionary with recommended models and confidence scores
    """
    try:
        logger.info(f"Getting recommendations: task={task_type}, hardware={hardware}, performance={performance}")
        
        hub_scanner = _get_scanner()
        recommendations = hub_scanner.recommend_models(
            task_type=task_type,
            hardware=hardware,
            performance_preference=performance,
            limit=limit
        )
        
        return {
            'status': 'success',
            'recommendations': recommendations,
            'context': {
                'task_type': task_type,
                'hardware': hardware,
                'performance': performance
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'recommendations': []
        }


# Tool: Get Model Details
def get_model_details_tool(model_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model
    
    Args:
        model_id: HuggingFace model ID (e.g., 'bert-base-uncased')
    
    Returns:
        Dictionary with comprehensive model information including model card
    """
    try:
        logger.info(f"Getting model details: model_id='{model_id}'")
        
        hub_scanner = _get_scanner()
        
        # Check if model exists in cache
        if hasattr(hub_scanner, 'model_cache') and model_id in hub_scanner.model_cache:
            model_data = hub_scanner.model_cache[model_id]
            
            # Convert dataclass to dict if needed
            if is_dataclass(model_data) and not isinstance(model_data, type):
                model_info = asdict(model_data)
            elif isinstance(model_data, dict):
                model_info = model_data.get('model_info', model_data)
            else:
                model_info = {'model_id': model_id}
            
            # Get and convert performance data
            performance_data = getattr(hub_scanner, 'performance_cache', {}).get(model_id, {})
            if is_dataclass(performance_data) and not isinstance(performance_data, type):
                performance = asdict(performance_data)
            else:
                performance = performance_data if isinstance(performance_data, dict) else {}
            
            # Get and convert compatibility data
            compatibility_data = getattr(hub_scanner, 'compatibility_cache', {}).get(model_id, {})
            if is_dataclass(compatibility_data) and not isinstance(compatibility_data, type):
                compatibility = asdict(compatibility_data)
            else:
                compatibility = compatibility_data if isinstance(compatibility_data, dict) else {}
            
            return {
                'status': 'success',
                'model_id': model_id,
                'model_info': model_info,
                'performance': performance,
                'compatibility': compatibility
            }
        
        # If not in cache, try to fetch from search
        search_results = hub_scanner.search_models(model_id, limit=1)
        
        if search_results and len(search_results) > 0:
            result = search_results[0]
            
            if isinstance(result, dict):
                model_info = result.get('model_info', {})
                performance = result.get('performance', {})
                compatibility = result.get('compatibility', {})
            else:
                model_info = {'model_id': model_id}
                performance = {}
                compatibility = {}
            
            return {
                'status': 'success',
                'model_id': model_id,
                'model_info': model_info,
                'performance': performance,
                'compatibility': compatibility
            }
        else:
            return {
                'status': 'error',
                'error': f'Model {model_id} not found',
                'model_id': model_id
            }
    except Exception as e:
        logger.error(f"Error getting model details: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'model_id': model_id
        }


# Tool: Get Model Stats
def get_model_stats_tool() -> Dict[str, Any]:
    """
    Get statistics about cached models
    
    Returns:
        Dictionary with model statistics
    """
    try:
        logger.info("Getting model stats")
        
        hub_scanner = _get_scanner()
        
        stats = {
            'total_models': len(getattr(hub_scanner, 'model_cache', {})),
            'scan_stats': getattr(hub_scanner, 'scan_stats', {})
        }
        
        return {
            'status': 'success',
            'stats': stats
        }
    except Exception as e:
        logger.error(f"Error getting model stats: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


# Register model tools with the MCP server
def register_model_tools(mcp) -> None:
    """Register model management tools with the MCP server"""
    try:
        # Register tools based on MCP server type
        if hasattr(mcp, 'register_tool'):
            # Standalone MCP style
            mcp.register_tool(
                name="search_models",
                function=search_models_tool,
                description="Search for models on HuggingFace Hub",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "task_filter": {"type": "string", "description": "Task type filter"},
                        "hardware_filter": {"type": "string", "description": "Hardware filter"},
                        "limit": {"type": "integer", "description": "Max results", "default": 20}
                    },
                    "required": ["query"]
                },
                execution_context="server",
            )
            
            mcp.register_tool(
                name="recommend_models",
                function=recommend_models_tool,
                description="Get AI-powered model recommendations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_type": {"type": "string", "description": "Task type"},
                        "hardware": {"type": "string", "description": "Target hardware", "default": "cpu"},
                        "performance": {"type": "string", "description": "Performance preference", "default": "balanced"},
                        "limit": {"type": "integer", "description": "Max recommendations", "default": 5}
                    },
                    "required": ["task_type"]
                },
                execution_context="server",
            )
            
            mcp.register_tool(
                name="get_model_details",
                function=get_model_details_tool,
                description="Get detailed information about a specific model",
                input_schema={
                    "type": "object",
                    "properties": {
                        "model_id": {"type": "string", "description": "HuggingFace model ID"}
                    },
                    "required": ["model_id"]
                },
                execution_context="server",
            )
            
            mcp.register_tool(
                name="get_model_stats",
                function=get_model_stats_tool,
                description="Get statistics about cached models",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                execution_context="server",
            )
        elif hasattr(mcp, 'tool'):
            # FastMCP decorator style
            @mcp.tool()
            def search_models(query: str, task_filter: Optional[str] = None, 
                            hardware_filter: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
                """Search for models on HuggingFace Hub"""
                return search_models_tool(query, task_filter, hardware_filter, limit)
            
            @mcp.tool()
            def recommend_models(task_type: str, hardware: str = 'cpu', 
                               performance: str = 'balanced', limit: int = 5) -> Dict[str, Any]:
                """Get AI-powered model recommendations"""
                return recommend_models_tool(task_type, hardware, performance, limit)
            
            @mcp.tool()
            def get_model_details(model_id: str) -> Dict[str, Any]:
                """Get detailed information about a specific model"""
                return get_model_details_tool(model_id)
            
            @mcp.tool()
            def get_model_stats() -> Dict[str, Any]:
                """Get statistics about cached models"""
                return get_model_stats_tool()
        
        logger.info("Model tools registered successfully")
    except Exception as e:
        logger.error(f"Error registering model tools: {e}")
        raise
