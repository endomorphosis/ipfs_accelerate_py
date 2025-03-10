#!/usr/bin/env python3
"""
Resource Pool Bridge Extensions for WebNN/WebGPU

This module extends the ResourcePoolBridgeIntegration with additional methods to support
cross-browser model sharding and advanced resource management.

Key features:
- Optimal browser connection selection for model components
- Enhanced model type detection and classification
- Model component balancing across browser instances
- Advanced model metrics collection and analysis

Usage:
    from resource_pool_bridge_extensions import extend_resource_pool_bridge
    
    # Extend existing resource pool bridge
    extend_resource_pool_bridge()
    
    # Now use get_optimal_browser_connection in ResourcePoolBridgeIntegration
    connection_id, connection_info = integration.get_optimal_browser_connection(
        model_type='text', 
        platform='webgpu'
    )
"""

import os
import sys
import logging
import functools
from typing import Dict, List, Any, Optional, Tuple

# Import resource pool bridge
try:
    from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
except ImportError:
    # Try to import from parent directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from fixed_web_platform.resource_pool_bridge import ResourcePoolBridgeIntegration
    except ImportError:
        print("Error: Could not import ResourcePoolBridgeIntegration")
        ResourcePoolBridgeIntegration = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_optimal_browser_connection(self, model_type: str, platform: str = 'webgpu', 
                                  model_family: str = None, priority: int = 0) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Get the optimal browser connection for a model with advanced load balancing.
    
    This method implements sophisticated load balancing across available browser connections:
    1. First prioritizes browser type based on model type/family optimizations
    2. Then considers current load and connection health 
    3. Applies weighted scoring for optimal connection selection
    4. Supports priority levels for critical vs. non-critical models
    
    Args:
        model_type: Type of model ('text', 'vision', 'audio', etc.)
        platform: Platform to use ('webgpu' or 'webnn')
        model_family: Optional model family for more specific optimization
        priority: Priority level (0-10, higher numbers = higher priority)
        
    Returns:
        Tuple of (connection_id, connection_info) or (None, None) if no connection available
    """
    # Use model_family if provided, otherwise fall back to model_type
    model_category = model_family or model_type
    
    # Determine preferred browser for this model type
    preferred_browser = self.browser_preferences.get(model_category, self.browser_preferences.get(model_type, 'chrome'))
    
    # Score each connection based on multiple factors
    connection_scores = []
    
    for conn_id, conn_info in self.browser_connections.items():
        # Skip connections that don't match the platform
        if conn_info['platform'] != platform:
            continue
            
        # Skip connections that are unhealthy
        if ('connection' in conn_info and 
            hasattr(conn_info['connection'], 'is_healthy') and 
            not conn_info['connection'].is_healthy()):
            continue
        
        # Skip connections that are known to be busy
        if ('connection' in conn_info and 
            hasattr(conn_info['connection'], 'is_busy') and 
            conn_info['connection'].is_busy()):
            continue
        
        # Base score starts at 100
        score = 100
        
        # Browser match adds a significant boost (most important factor)
        if conn_info['browser_name'] == preferred_browser:
            score += 50
        
        # Adjust score based on existing models on this connection
        if 'connection' in conn_info and hasattr(conn_info['connection'], 'loaded_models'):
            # Each loaded model reduces score slightly (we prefer less loaded connections)
            model_count = len(conn_info['connection'].loaded_models)
            score -= min(40, model_count * 5)  # Cap penalty at 40 points
            
            # Bigger penalty if already processing models of different types (avoid mixing)
            if model_count > 0:
                loaded_model_types = set()
                for model_id in conn_info['connection'].loaded_models:
                    if ':' in model_id:
                        loaded_type = model_id.split(':', 1)[0]
                        loaded_model_types.add(loaded_type)
                
                # If this connection has models of different types, apply penalty
                if loaded_model_types and model_type not in loaded_model_types:
                    score -= 20
        
        # Adjust based on browser-specific optimizations
        if model_category == 'audio' and conn_info['browser_name'] == 'firefox':
            # Firefox is optimized for audio models
            score += 20
        elif model_category == 'text_embedding' and conn_info['browser_name'] == 'edge':
            # Edge is optimized for text embeddings with WebNN
            score += 20
        elif model_category == 'vision' and conn_info['browser_name'] == 'chrome':
            # Chrome is generally good for vision models
            score += 15
        
        # More recent connections are slightly preferred (better cache utilization)
        if 'last_used' in conn_info:
            recency_factor = min(10, max(0, (time.time() - conn_info['last_used']) / 60))
            score -= recency_factor  # Newer connections score higher
        
        # Add the connection and its score
        connection_scores.append((conn_id, conn_info, score))
    
    # If we have connection options, select the best one
    if connection_scores:
        # Sort by score (highest first)
        connection_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Log scoring at debug level for monitoring
        if logger.isEnabledFor(logging.DEBUG):
            score_details = [f"{conn_id} ({score})" for conn_id, _, score in connection_scores[:3]]
            logger.debug(f"Top connections for {model_category}: {', '.join(score_details)}")
        
        # Return the highest-scoring connection
        best_conn_id, best_conn_info, best_score = connection_scores[0]
        return best_conn_id, best_conn_info
    
    # No suitable connection found
    return None, None

def detect_model_family(self, model_name: str) -> str:
    """
    Detect model family from model name with enhanced detection.
    
    This method implements a comprehensive model family detection system that
    recognizes a wide range of model architectures and categories based on
    model name patterns.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model family identifier
    """
    model_name_lower = model_name.lower()
    
    # Text models
    if any(name in model_name_lower for name in ['bert', 'roberta', 'distilbert', 'albert']):
        return 'text_embedding'
    elif any(name in model_name_lower for name in ['t5', 'mt5', 'bart', 'pegasus']):
        return 'text_generation'
    elif any(name in model_name_lower for name in ['gpt', 'opt', 'bloom', 'llama', 'mistral', 'falcon']):
        return 'text_generation'
    elif any(name in model_name_lower for name in ['qlora', 'qwen', 'grok']):
        return 'text_generation'
    
    # Vision models
    elif any(name in model_name_lower for name in ['vit', 'deit', 'beit', 'swin']):
        return 'vision'
    elif any(name in model_name_lower for name in ['resnet', 'efficientnet', 'convnext']):
        return 'vision'
    elif any(name in model_name_lower for name in ['yolo', 'detr', 'maskrcnn', 'fasterrcnn']):
        return 'vision_detection'
    
    # Audio models
    elif any(name in model_name_lower for name in ['wav2vec', 'hubert', 'whisper']):
        return 'audio'
    elif any(name in model_name_lower for name in ['musicgen', 'audiogen', 'melgan']):
        return 'audio_generation'
    elif any(name in model_name_lower for name in ['clap', 'wav2clip']):
        return 'audio_embedding'
    
    # Multimodal models
    elif any(name in model_name_lower for name in ['clip', 'blip', 'flava']):
        return 'multimodal'
    elif any(name in model_name_lower for name in ['llava', 'flamingo', 'fuyu']):
        return 'multimodal'
    elif any(name in model_name_lower for name in ['videomae', 'videomaev2', 'videoclip']):
        return 'multimodal_video'
    
    # Default to text
    return 'text'

def balance_model_components(self, model_name: str, component_types: List[str], 
                            platform: str = 'webgpu') -> Dict[str, str]:
    """
    Balance model components across browser instances for optimal performance.
    
    This method distributes different model components across browser instances
    based on browser-specific optimizations and current load.
    
    Args:
        model_name: Name of the model
        component_types: List of component types (e.g., ['vision', 'text', 'fusion'])
        platform: Platform to use ('webgpu' or 'webnn')
        
    Returns:
        Dictionary mapping component types to browser connection IDs
    """
    component_allocations = {}
    
    # Define preferred browsers for each component type
    browser_preferences = {
        'vision': 'chrome',
        'text': 'edge',
        'audio': 'firefox',
        'fusion': 'chrome',
        'attention': 'firefox',
        'feedforward': 'chrome'
    }
    
    # Allocate each component to the most suitable browser
    for component in component_types:
        preferred_browser = browser_preferences.get(component, 'chrome')
        
        # Get optimal connection for this component
        connection_id, _ = self.get_optimal_browser_connection(
            model_type=component,
            platform=platform,
            model_family=component
        )
        
        if connection_id:
            component_allocations[component] = connection_id
        else:
            # No suitable connection found, create a new one
            logger.info(f"No suitable connection found for {component}, creating a new one")
            
            # This would involve creating a new browser connection
            # For now, just mark as unallocated
            component_allocations[component] = None
    
    return component_allocations

def collect_enhanced_metrics(self) -> Dict[str, Any]:
    """
    Collect enhanced metrics about browser connections and model performance.
    
    This method gathers comprehensive metrics about browser usage, connection
    efficiency, and model performance across different browser types.
    
    Returns:
        Dictionary with detailed metrics
    """
    metrics = {
        'browser_metrics': {},
        'platform_metrics': {},
        'model_type_metrics': {},
        'connection_efficiency': {},
        'overall': {}
    }
    
    # Collect browser-specific metrics
    browser_counts = {}
    browser_models = {}
    browser_memory = {}
    
    for conn_id, conn_info in self.browser_connections.items():
        browser = conn_info.get('browser_name', 'unknown')
        
        # Count browsers
        if browser not in browser_counts:
            browser_counts[browser] = 0
            browser_models[browser] = 0
            browser_memory[browser] = 0
        
        browser_counts[browser] += 1
        
        # Count models per browser
        if 'connection' in conn_info and hasattr(conn_info['connection'], 'loaded_models'):
            browser_models[browser] += len(conn_info['connection'].loaded_models)
        
        # Estimate memory usage (if available)
        if 'connection' in conn_info and hasattr(conn_info['connection'], 'memory_usage'):
            browser_memory[browser] += conn_info['connection'].get('memory_usage', 0)
    
    # Add browser metrics
    metrics['browser_metrics'] = {
        'counts': browser_counts,
        'models': browser_models,
        'memory': browser_memory,
        'models_per_browser': {
            browser: (models / count if count > 0 else 0)
            for browser, count in browser_counts.items()
            for models in [browser_models.get(browser, 0)]
        }
    }
    
    # Collect platform metrics
    platform_counts = {'webgpu': 0, 'webnn': 0, 'cpu': 0}
    platform_models = {'webgpu': 0, 'webnn': 0, 'cpu': 0}
    
    for conn_id, conn_info in self.browser_connections.items():
        platform = conn_info.get('platform', 'unknown')
        if platform in platform_counts:
            platform_counts[platform] += 1
            
            # Count models per platform
            if 'connection' in conn_info and hasattr(conn_info['connection'], 'loaded_models'):
                platform_models[platform] += len(conn_info['connection'].loaded_models)
    
    # Add platform metrics
    metrics['platform_metrics'] = {
        'counts': platform_counts,
        'models': platform_models,
        'models_per_platform': {
            platform: (models / count if count > 0 else 0)
            for platform, count in platform_counts.items()
            for models in [platform_models.get(platform, 0)]
        }
    }
    
    # Collect model type metrics by examining loaded models
    model_type_counts = {}
    
    for conn_id, conn_info in self.browser_connections.items():
        if 'connection' in conn_info and hasattr(conn_info['connection'], 'loaded_models'):
            for model_id in conn_info['connection'].loaded_models:
                if ':' in model_id:
                    model_type = model_id.split(':', 1)[0]
                    model_type_counts[model_type] = model_type_counts.get(model_type, 0) + 1
    
    # Add model type metrics
    metrics['model_type_metrics'] = {
        'counts': model_type_counts
    }
    
    # Calculate connection efficiency
    total_connections = sum(browser_counts.values())
    total_models = sum(browser_models.values())
    
    metrics['connection_efficiency'] = {
        'total_connections': total_connections,
        'total_models': total_models,
        'models_per_connection': total_models / total_connections if total_connections > 0 else 0,
        'connection_utilization': total_connections / self.max_connections if self.max_connections > 0 else 0
    }
    
    # Overall metrics
    metrics['overall'] = {
        'active_browsers': len([b for b, c in browser_counts.items() if c > 0]),
        'active_platforms': len([p for p, c in platform_counts.items() if c > 0]),
        'model_type_diversity': len(model_type_counts),
        'browser_balance': max(browser_counts.values()) / total_connections if total_connections > 0 else 0
    }
    
    return metrics

def extend_resource_pool_bridge():
    """
    Extend ResourcePoolBridgeIntegration with additional methods.
    
    This function adds the defined methods to the ResourcePoolBridgeIntegration class
    to enhance its capabilities without modifying the original class.
    """
    if ResourcePoolBridgeIntegration is None:
        logger.error("ResourcePoolBridgeIntegration not available, cannot extend.")
        return False
    
    # Add get_optimal_browser_connection method
    ResourcePoolBridgeIntegration.get_optimal_browser_connection = get_optimal_browser_connection
    
    # Add detect_model_family method
    ResourcePoolBridgeIntegration.detect_model_family = detect_model_family
    
    # Add balance_model_components method
    ResourcePoolBridgeIntegration.balance_model_components = balance_model_components
    
    # Add collect_enhanced_metrics method
    ResourcePoolBridgeIntegration.collect_enhanced_metrics = collect_enhanced_metrics
    
    logger.info("ResourcePoolBridgeIntegration extended with additional methods.")
    return True

# Auto-extend when imported
if __name__ != "__main__":
    extend_resource_pool_bridge()