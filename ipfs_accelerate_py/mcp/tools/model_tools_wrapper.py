"""
Simple wrapper to expose MCP model tools for the dashboard JSON-RPC endpoint.
"""

from ipfs_accelerate_py.huggingface_hub_scanner import HuggingFaceHubScanner
from ipfs_accelerate_py.model_manager import get_default_model_manager, BanditModelRecommender
from dataclasses import asdict, is_dataclass
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Shared instances
_scanner = None
_model_manager = None


def get_scanner():
    """Get or create shared HuggingFaceHubScanner instance."""
    global _scanner
    if _scanner is None:
        _scanner = HuggingFaceHubScanner()
    return _scanner


def get_model_manager():
    """Get or create shared ModelManager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = get_default_model_manager()
    return _model_manager


def search_models_tool(query: str, task_filter: Optional[str] = None, 
                       hardware_filter: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    """Search for models on HuggingFace Hub."""
    try:
        logger.info(f"Searching models: query='{query}', task={task_filter}, hardware={hardware_filter}, limit={limit}")
        
        hub_scanner = get_scanner()
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


def recommend_models_tool(task_type: str, hardware: str = 'cpu', 
                         performance: str = 'balanced', limit: int = 5) -> Dict[str, Any]:
    """Get AI-powered model recommendations."""
    try:
        logger.info(f"Getting recommendations: task={task_type}, hardware={hardware}, performance={performance}")
        
        hub_scanner = get_scanner()
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


def get_model_details_tool(model_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    try:
        logger.info(f"Getting model details: model_id='{model_id}'")
        
        hub_scanner = get_scanner()
        
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


def get_model_stats_tool() -> Dict[str, Any]:
    """Get statistics about cached models."""
    try:
        logger.info("Getting model stats")
        
        hub_scanner = get_scanner()
        
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
