"""
Enhanced Inference Tools for IPFS Accelerate MCP Server

This module provides enhanced MCP tools for running inference with:
- HuggingFace model integration
- API multiplexing (OpenAI, Anthropic, local models)
- libp2p distributed inference
- Endpoint handlers for load balancing
- CLI tool integration (Claude Code, OpenAI Codex, Google Gemini)
"""

import os
import time
import logging
import json
try:
    import aiohttp
    import asyncio
except ImportError:
    aiohttp = None
    asyncio = None
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Import CLI endpoint adapters
try:
    from .cli_endpoint_adapters import (
        ClaudeCodeAdapter,
        OpenAICodexAdapter,
        GeminiCLIAdapter,
        VSCodeCLIAdapter,
        register_cli_endpoint,
        execute_cli_inference,
        list_cli_endpoints,
        CLI_ADAPTER_REGISTRY
    )
    HAVE_CLI_ADAPTERS = True
except ImportError:
    HAVE_CLI_ADAPTERS = False
    logger = logging.getLogger("ipfs_accelerate_mcp.tools.enhanced_inference")
    logger.warning("CLI endpoint adapters not available")

logger = logging.getLogger("ipfs_accelerate_mcp.tools.enhanced_inference")

# Configuration for API providers
API_PROVIDERS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "text-embedding-ada-002"],
        "requires_key": True
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-3-sonnet", "claude-3-haiku", "claude-2.1"],
        "requires_key": True
    },
    "huggingface": {
        "base_url": "https://api-inference.huggingface.co",
        "models": ["meta-llama/Llama-2-7b-chat-hf", "microsoft/DialoGPT-medium", "gpt2"],
        "requires_key": False
    }
}

# Configuration for CLI providers
CLI_PROVIDERS = {
    "claude_cli": {
        "adapter_class": "ClaudeCodeAdapter",
        "models": ["claude-3-sonnet", "claude-3-opus", "claude-3-haiku"],
        "description": "Claude Code CLI tool for local Anthropic model access"
    },
    "openai_cli": {
        "adapter_class": "OpenAICodexAdapter", 
        "models": ["gpt-3.5-turbo", "gpt-4", "codex"],
        "description": "OpenAI CLI tool (ChatGPT/Codex)"
    },
    "gemini_cli": {
        "adapter_class": "GeminiCLIAdapter",
        "models": ["gemini-pro", "gemini-ultra"],
        "description": "Google Gemini CLI tool"
    },
    "vscode_cli": {
        "adapter_class": "VSCodeCLIAdapter",
        "models": ["copilot-chat", "copilot-code"],
        "description": "Visual Studio Code CLI with GitHub Copilot integration"
    }
}

# Endpoint registry for multiplexing
ENDPOINT_REGISTRY = {}

# Global queue monitoring data
QUEUE_MONITOR = {
    "global_queue": [],
    "endpoint_queues": {},
    "stats": {
        "total_tasks": 0,
        "completed_tasks": 0,
        "failed_tasks": 0,
        "pending_tasks": 0
    },
    "endpoint_stats": {}
}

def register_tools(mcp):
    """Register enhanced inference tools with the MCP server"""
    
    @mcp.tool()
    def multiplex_inference(
        prompt: str,
        task_type: str = "text_generation",
        model_preferences: List[str] = None,
        fallback_providers: List[str] = None,
        max_retries: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference using multiplexed endpoints with automatic fallback
        
        This tool attempts inference across multiple endpoints:
        1. Local HuggingFace models
        2. libp2p distributed nodes  
        3. External API providers (OpenAI, Anthropic, etc.)
        
        Args:
            prompt: The input prompt for inference
            task_type: Type of task ("text_generation", "embedding", "classification")
            model_preferences: Ordered list of preferred models
            fallback_providers: List of fallback API providers
            max_retries: Maximum number of retry attempts
            **kwargs: Additional parameters for the specific task
            
        Returns:
            Dictionary with inference results and metadata
        """
        start_time = time.time()
        attempts = []
        
        try:
            # Default preferences if not specified
            if model_preferences is None:
                model_preferences = ["local/gpt2", "huggingface/gpt2", "openai/gpt-3.5-turbo"]
            
            if fallback_providers is None:
                fallback_providers = ["huggingface", "openai"]
            
            # Attempt inference with each preference
            for attempt in range(max_retries):
                for preference in model_preferences:
                    provider, model = preference.split("/", 1) if "/" in preference else ("local", preference)
                    
                    attempt_info = {
                        "attempt": attempt + 1,
                        "provider": provider,
                        "model": model,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    try:
                        if provider == "local":
                            result = _run_local_inference(model, prompt, task_type, **kwargs)
                        elif provider == "libp2p":
                            result = _run_distributed_inference(model, prompt, task_type, **kwargs)
                        elif provider in API_PROVIDERS:
                            result = _run_api_inference(provider, model, prompt, task_type, **kwargs)
                        elif HAVE_CLI_ADAPTERS and provider in CLI_PROVIDERS:
                            # Use CLI endpoint if available
                            result = _run_cli_inference(provider, model, prompt, task_type, **kwargs)
                        else:
                            raise ValueError(f"Unknown provider: {provider}")
                        
                        # If successful, return result with metadata
                        elapsed_time = time.time() - start_time
                        attempt_info["status"] = "success"
                        attempt_info["elapsed_time"] = elapsed_time
                        attempts.append(attempt_info)
                        
                        return {
                            "result": result,
                            "provider_used": provider,
                            "model_used": model,
                            "total_elapsed_time": elapsed_time,
                            "attempts": attempts,
                            "status": "success"
                        }
                        
                    except Exception as e:
                        attempt_info["status"] = "failed"
                        attempt_info["error"] = str(e)
                        attempts.append(attempt_info)
                        logger.warning(f"Inference attempt failed: {provider}/{model} - {e}")
                        continue
            
            # If all attempts failed
            elapsed_time = time.time() - start_time
            return {
                "error": "All inference attempts failed",
                "total_elapsed_time": elapsed_time,
                "attempts": attempts,
                "status": "failed"
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            return {
                "error": f"Multiplexed inference error: {str(e)}",
                "total_elapsed_time": elapsed_time,
                "attempts": attempts,
                "status": "error"
            }
    
    @mcp.tool()
    def register_endpoint(
        endpoint_id: str,
        provider: str,
        model: str,
        endpoint_url: str,
        api_key: Optional[str] = None,
        priority: int = 5,
        health_check_url: Optional[str] = None,
        **config
    ) -> Dict[str, Any]:
        """
        Register a new inference endpoint for multiplexing
        
        Args:
            endpoint_id: Unique identifier for the endpoint
            provider: Provider type (local, libp2p, openai, anthropic, etc.)
            model: Model name/identifier
            endpoint_url: URL for the inference endpoint
            api_key: API key if required
            priority: Priority level (1=highest, 10=lowest)
            health_check_url: URL for health checks
            **config: Additional configuration parameters
            
        Returns:
            Dictionary with registration status
        """
        try:
            endpoint_config = {
                "endpoint_id": endpoint_id,
                "provider": provider,
                "model": model,
                "endpoint_url": endpoint_url,
                "api_key": api_key,
                "priority": priority,
                "health_check_url": health_check_url or endpoint_url,
                "config": config,
                "registered_at": datetime.now().isoformat(),
                "status": "active",
                "last_health_check": None,
                "request_count": 0,
                "success_count": 0,
                "avg_response_time": 0.0
            }
            
            ENDPOINT_REGISTRY[endpoint_id] = endpoint_config
            
            logger.info(f"Registered endpoint: {endpoint_id} ({provider}/{model})")
            
            return {
                "status": "success",
                "endpoint_id": endpoint_id,
                "message": f"Endpoint {endpoint_id} registered successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to register endpoint {endpoint_id}: {e}")
            return {
                "status": "error",
                "error": f"Failed to register endpoint: {str(e)}"
            }
    
    @mcp.tool()
    def get_endpoint_status(endpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of registered endpoints
        
        Args:
            endpoint_id: Specific endpoint ID, or None for all endpoints
            
        Returns:
            Dictionary with endpoint status information
        """
        try:
            if endpoint_id:
                if endpoint_id not in ENDPOINT_REGISTRY:
                    return {
                        "error": f"Endpoint '{endpoint_id}' not found"
                    }
                return {
                    "endpoint": ENDPOINT_REGISTRY[endpoint_id],
                    "status": "success"
                }
            else:
                return {
                    "endpoints": list(ENDPOINT_REGISTRY.values()),
                    "count": len(ENDPOINT_REGISTRY),
                    "status": "success"
                }
                
        except Exception as e:
            return {
                "error": f"Failed to get endpoint status: {str(e)}",
                "status": "error"
            }
    
    @mcp.tool()
    def configure_api_provider(
        provider: str,
        api_key: str,
        base_url: Optional[str] = None,
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Configure an external API provider for multiplexing
        
        Args:
            provider: Provider name (openai, anthropic, huggingface)
            api_key: API key for the provider
            base_url: Custom base URL (optional)
            models: List of available models (optional)
            
        Returns:
            Dictionary with configuration status
        """
        try:
            if provider not in API_PROVIDERS:
                return {
                    "error": f"Unknown provider: {provider}. Supported: {list(API_PROVIDERS.keys())}",
                    "status": "error"
                }
            
            # Update provider configuration
            config = API_PROVIDERS[provider].copy()
            config["api_key"] = api_key
            
            if base_url:
                config["base_url"] = base_url
            if models:
                config["models"] = models
            
            API_PROVIDERS[provider] = config
            
            logger.info(f"Configured API provider: {provider}")
            
            return {
                "status": "success",
                "provider": provider,
                "message": f"Provider {provider} configured successfully",
                "available_models": config["models"]
            }
            
        except Exception as e:
            logger.error(f"Failed to configure provider {provider}: {e}")
            return {
                "error": f"Failed to configure provider: {str(e)}",
                "status": "error"
            }
    
    @mcp.tool()
    def search_huggingface_models(
        query: str,
        task: Optional[str] = None,
        sort: str = "downloads",
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search HuggingFace model hub
        
        Args:
            query: Search query
            task: Filter by task type (text-generation, text-classification, etc.)
            sort: Sort by (downloads, likes, trending)
            limit: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        try:
            # This would normally make an API call to HuggingFace
            # For now, return simulated results
            
            # Simulate some popular models based on query
            mock_results = []
            
            if "gpt" in query.lower():
                mock_results.extend([
                    {
                        "id": "gpt2",
                        "author": "openai-community", 
                        "downloads": 150000,
                        "likes": 1200,
                        "task": "text-generation",
                        "description": "GPT-2 is a transformers model pretrained on a very large corpus of English data"
                    },
                    {
                        "id": "microsoft/DialoGPT-medium",
                        "author": "microsoft",
                        "downloads": 75000,
                        "likes": 800,
                        "task": "conversational",
                        "description": "DialoGPT is a large-scale tunable neural conversational response generation model"
                    }
                ])
            
            if "bert" in query.lower():
                mock_results.extend([
                    {
                        "id": "bert-base-uncased",
                        "author": "google-bert",
                        "downloads": 200000,
                        "likes": 1500,
                        "task": "fill-mask",
                        "description": "BERT base model (uncased)"
                    }
                ])
            
            if "llama" in query.lower():
                mock_results.extend([
                    {
                        "id": "meta-llama/Llama-2-7b-chat-hf",
                        "author": "meta-llama",
                        "downloads": 120000,
                        "likes": 2000,
                        "task": "text-generation",
                        "description": "Llama 2 is a collection of pretrained and fine-tuned generative text models"
                    }
                ])
            
            # Filter by task if specified
            if task:
                mock_results = [r for r in mock_results if r.get("task") == task]
            
            # Sort and limit results
            if sort == "downloads":
                mock_results.sort(key=lambda x: x.get("downloads", 0), reverse=True)
            elif sort == "likes":
                mock_results.sort(key=lambda x: x.get("likes", 0), reverse=True)
            
            return {
                "models": mock_results[:limit],
                "query": query,
                "total_found": len(mock_results),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"HuggingFace search failed: {e}")
            return {
                "error": f"Search failed: {str(e)}",
                "status": "error"
            }
    
    @mcp.tool()
    def get_queue_status() -> Dict[str, Any]:
        """
        Get comprehensive queue status for all endpoints and model types
        
        Returns:
            Dictionary with queue status information broken down by model type and endpoint handler
        """
        try:
            # Simulate queue data with realistic examples
            endpoint_queues = {
                    # Local CUDA devices
                    "cuda1": {
                        "endpoint_type": "local_gpu",
                        "device": "CUDA:0",
                        "model_types": ["text-generation", "image-generation"],
                        "queue_size": 2,
                        "processing": 1,
                        "avg_processing_time": 1.2,
                        "status": "active",
                        "current_task": {
                            "task_id": "task_123",
                            "model": "meta-llama/Llama-2-7b-chat-hf",
                            "task_type": "text_generation",
                            "started_at": datetime.now().isoformat(),
                            "estimated_completion": "2 minutes"
                        }
                    },
                    "cuda2": {
                        "endpoint_type": "local_gpu", 
                        "device": "CUDA:1",
                        "model_types": ["computer-vision", "multimodal"],
                        "queue_size": 1,
                        "processing": 0,
                        "avg_processing_time": 0.8,
                        "status": "idle",
                        "current_task": None
                    },
                    # libp2p distributed nodes
                    "peer_abc123": {
                        "endpoint_type": "libp2p_peer",
                        "peer_id": "12D3KooWABC123...",
                        "model_types": ["text-generation", "embedding"],
                        "queue_size": 5,
                        "processing": 2,
                        "avg_processing_time": 2.5,
                        "status": "active",
                        "network_latency": 150,
                        "current_task": {
                            "task_id": "task_456",
                            "model": "sentence-transformers/all-MiniLM-L6-v2",
                            "task_type": "embedding",
                            "started_at": datetime.now().isoformat(),
                            "estimated_completion": "30 seconds"
                        }
                    },
                    "peer_def789": {
                        "endpoint_type": "libp2p_peer",
                        "peer_id": "12D3KooWDEF789...",
                        "model_types": ["image-generation", "audio-processing"],
                        "queue_size": 0,
                        "processing": 0,
                        "avg_processing_time": 3.2,
                        "status": "offline",
                        "network_latency": 999,
                        "current_task": None
                    },
                    # OpenAI API keys
                    "openai_key1": {
                        "endpoint_type": "api_provider",
                        "provider": "openai",
                        "key_name": "primary",
                        "model_types": ["text-generation", "embedding"],
                        "queue_size": 3,
                        "processing": 1,
                        "avg_processing_time": 1.8,
                        "status": "active",
                        "rate_limit_remaining": 850,
                        "rate_limit_reset": "3600 seconds",
                        "current_task": {
                            "task_id": "task_789",
                            "model": "gpt-3.5-turbo",
                            "task_type": "text_generation",
                            "started_at": datetime.now().isoformat(),
                            "estimated_completion": "15 seconds"
                        }
                    },
                    "openai_key2": {
                        "endpoint_type": "api_provider",
                        "provider": "openai",
                        "key_name": "secondary",
                        "model_types": ["text-generation"],
                        "queue_size": 0,
                        "processing": 0,
                        "avg_processing_time": 1.5,
                        "status": "idle",
                        "rate_limit_remaining": 1000,
                        "rate_limit_reset": "3600 seconds",
                        "current_task": None
                    },
                    # Anthropic API
                    "anthropic_key1": {
                        "endpoint_type": "api_provider",
                        "provider": "anthropic",
                        "key_name": "primary",
                        "model_types": ["text-generation"],
                        "queue_size": 1,
                        "processing": 0,
                        "avg_processing_time": 2.1,
                        "status": "idle",
                        "rate_limit_remaining": 500,
                        "rate_limit_reset": "1800 seconds",
                        "current_task": None
                    },
                    # HuggingFace API
                    "huggingface_api": {
                        "endpoint_type": "api_provider",
                        "provider": "huggingface",
                        "key_name": "default",
                        "model_types": ["text-generation", "image-generation", "audio-processing"],
                        "queue_size": 4,
                        "processing": 2,
                        "avg_processing_time": 3.5,
                        "status": "active",
                        "rate_limit_remaining": 100,
                        "rate_limit_reset": "600 seconds",
                        "current_task": {
                            "task_id": "task_101",
                            "model": "stabilityai/stable-diffusion-xl-base-1.0",
                            "task_type": "image_generation",
                            "started_at": datetime.now().isoformat(),
                            "estimated_completion": "45 seconds"
                        }
                    }
                }
            
            # Add CLI endpoints if available
            if HAVE_CLI_ADAPTERS:
                for endpoint_id, adapter in CLI_ADAPTER_REGISTRY.items():
                    stats = adapter.get_stats()
                    endpoint_queues[endpoint_id] = {
                        "endpoint_type": "cli_tool",
                        "provider": stats.get("endpoint_id", "unknown").split("_")[0],
                        "cli_path": stats.get("cli_path", "unknown"),
                        "model_types": ["text-generation", "code-generation"],
                        "queue_size": 0,
                        "processing": 0,
                        "avg_processing_time": stats["stats"].get("avg_time", 0.0),
                        "status": "available" if stats.get("available") else "unavailable",
                        "total_requests": stats["stats"].get("requests", 0),
                        "success_rate": (stats["stats"].get("successes", 0) / stats["stats"].get("requests", 1) * 100) if stats["stats"].get("requests", 0) > 0 else 0,
                        "current_task": None
                    }
            
            # Calculate summary statistics
            cli_endpoint_count = len(CLI_ADAPTER_REGISTRY) if HAVE_CLI_ADAPTERS else 0
            
            queue_status = {
                "global_queue": {
                    "total_tasks": QUEUE_MONITOR["stats"]["total_tasks"] + len(QUEUE_MONITOR["global_queue"]),
                    "pending_tasks": len(QUEUE_MONITOR["global_queue"]),
                    "processing_tasks": 3,
                    "completed_tasks": QUEUE_MONITOR["stats"]["completed_tasks"],
                    "failed_tasks": QUEUE_MONITOR["stats"]["failed_tasks"]
                },
                "endpoint_queues": endpoint_queues,
                "summary": {
                    "total_endpoints": 8 + cli_endpoint_count,
                    "active_endpoints": 4,
                    "idle_endpoints": 3,
                    "offline_endpoints": 1,
                    "cli_endpoints": cli_endpoint_count,
                    "total_queue_size": 16,
                    "total_processing": 7,
                    "average_processing_time": 2.1,
                    "healthy_endpoints": 7 + cli_endpoint_count,
                    "unhealthy_endpoints": 1
                },
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
            return queue_status
            
        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {
                "error": f"Failed to get queue status: {str(e)}",
                "status": "error"
            }
    
    @mcp.tool()
    def get_queue_history() -> Dict[str, Any]:
        """
        Get queue performance history and trends
        
        Returns:
            Dictionary with historical queue metrics
        """
        try:
            # Simulate historical data
            history = {
                "time_series": {
                    "timestamps": [
                        (datetime.now().timestamp() - 300), # 5 min ago
                        (datetime.now().timestamp() - 240), # 4 min ago
                        (datetime.now().timestamp() - 180), # 3 min ago
                        (datetime.now().timestamp() - 120), # 2 min ago
                        (datetime.now().timestamp() - 60),  # 1 min ago
                        datetime.now().timestamp()          # now
                    ],
                    "queue_sizes": [12, 15, 18, 14, 16, 16],
                    "processing_tasks": [5, 6, 8, 7, 7, 7],
                    "completed_tasks": [45, 52, 59, 66, 71, 78],
                    "failed_tasks": [2, 2, 3, 3, 3, 4],
                    "avg_processing_time": [2.3, 2.1, 2.5, 2.0, 2.2, 2.1]
                },
                "endpoint_performance": {
                    "cuda1": {"uptime": 98.5, "success_rate": 99.2, "avg_response_time": 1.2},
                    "cuda2": {"uptime": 95.0, "success_rate": 98.8, "avg_response_time": 0.8},
                    "peer_abc123": {"uptime": 89.2, "success_rate": 95.5, "avg_response_time": 2.5},
                    "openai_key1": {"uptime": 99.8, "success_rate": 99.9, "avg_response_time": 1.8},
                    "openai_key2": {"uptime": 99.5, "success_rate": 99.7, "avg_response_time": 1.5},
                    "anthropic_key1": {"uptime": 98.9, "success_rate": 99.1, "avg_response_time": 2.1},
                    "huggingface_api": {"uptime": 92.3, "success_rate": 94.2, "avg_response_time": 3.5}
                },
                "model_type_stats": {
                    "text-generation": {"total_requests": 1250, "avg_time": 1.8, "success_rate": 98.5},
                    "image-generation": {"total_requests": 340, "avg_time": 12.3, "success_rate": 95.2},
                    "embedding": {"total_requests": 890, "avg_time": 0.6, "success_rate": 99.8},
                    "audio-processing": {"total_requests": 123, "avg_time": 4.5, "success_rate": 96.1},
                    "computer-vision": {"total_requests": 234, "avg_time": 2.1, "success_rate": 97.3}
                },
                "status": "success"
            }
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get queue history: {e}")
            return {
                "error": f"Failed to get queue history: {str(e)}",
                "status": "error"
            }
    
    # CLI Endpoint Tools
    if HAVE_CLI_ADAPTERS:
        @mcp.tool()
        def register_cli_endpoint_tool(
            cli_type: str,
            endpoint_id: Optional[str] = None,
            cli_path: Optional[str] = None,
            model: Optional[str] = None,
            **config
        ) -> Dict[str, Any]:
            """
            Register a CLI-based endpoint for inference multiplexing
            
            This tool registers CLI tools like Claude Code, OpenAI Codex, or Google Gemini
            as endpoints that can be used in the multiplexing queue system.
            
            Args:
                cli_type: Type of CLI tool (claude_cli, openai_cli, gemini_cli)
                endpoint_id: Unique identifier (auto-generated if None)
                cli_path: Path to CLI executable (auto-detected if None)
                model: Default model to use
                **config: Additional configuration (temperature, max_tokens, etc.)
                
            Returns:
                Dictionary with registration status
            """
            try:
                if cli_type not in CLI_PROVIDERS:
                    return {
                        "error": f"Unknown CLI type: {cli_type}. Supported: {list(CLI_PROVIDERS.keys())}",
                        "status": "error"
                    }
                
                # Generate endpoint_id if not provided
                if not endpoint_id:
                    endpoint_id = f"{cli_type}_{len(CLI_ADAPTER_REGISTRY)}"
                
                # Add model to config if provided
                if model:
                    config["model"] = model
                
                # Create appropriate adapter
                adapter_class_name = CLI_PROVIDERS[cli_type]["adapter_class"]
                if adapter_class_name == "ClaudeCodeAdapter":
                    adapter = ClaudeCodeAdapter(endpoint_id, cli_path, config)
                elif adapter_class_name == "OpenAICodexAdapter":
                    adapter = OpenAICodexAdapter(endpoint_id, cli_path, config)
                elif adapter_class_name == "GeminiCLIAdapter":
                    adapter = GeminiCLIAdapter(endpoint_id, cli_path, config)
                elif adapter_class_name == "VSCodeCLIAdapter":
                    adapter = VSCodeCLIAdapter(endpoint_id, cli_path, config)
                else:
                    return {
                        "error": f"Unknown adapter class: {adapter_class_name}",
                        "status": "error"
                    }
                
                # Register the adapter
                result = register_cli_endpoint(adapter)
                
                # Also add to endpoint registry for queue monitoring
                if result["status"] == "success":
                    ENDPOINT_REGISTRY[endpoint_id] = {
                        "endpoint_id": endpoint_id,
                        "provider": cli_type,
                        "endpoint_type": "cli",
                        "available": adapter.is_available(),
                        "cli_path": adapter.cli_path,
                        "config": config,
                        "registered_at": datetime.now().isoformat()
                    }
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to register CLI endpoint: {e}")
                return {
                    "error": f"Failed to register CLI endpoint: {str(e)}",
                    "status": "error"
                }
        
        @mcp.tool()
        def list_cli_endpoints_tool() -> Dict[str, Any]:
            """
            List all registered CLI endpoints
            
            Returns:
                Dictionary with list of CLI endpoints and their status
            """
            try:
                endpoints = list_cli_endpoints()
                return {
                    "endpoints": endpoints,
                    "count": len(endpoints),
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Failed to list CLI endpoints: {e}")
                return {
                    "error": f"Failed to list CLI endpoints: {str(e)}",
                    "status": "error"
                }
        
        @mcp.tool()
        def cli_inference(
            endpoint_id: str,
            prompt: str,
            task_type: str = "text_generation",
            timeout: int = 30,
            **kwargs
        ) -> Dict[str, Any]:
            """
            Run inference using a registered CLI endpoint
            
            Args:
                endpoint_id: ID of the registered CLI endpoint
                prompt: Input prompt for the model
                task_type: Type of task (text_generation, code_generation, etc.)
                timeout: Maximum execution time in seconds
                **kwargs: Additional task parameters (model, temperature, max_tokens, etc.)
                
            Returns:
                Dictionary with inference results
            """
            try:
                result = execute_cli_inference(endpoint_id, prompt, task_type, timeout, **kwargs)
                
                # Update queue monitoring stats
                if result.get("status") == "success":
                    QUEUE_MONITOR["stats"]["completed_tasks"] += 1
                else:
                    QUEUE_MONITOR["stats"]["failed_tasks"] += 1
                
                return result
                
            except Exception as e:
                logger.error(f"CLI inference failed: {e}")
                QUEUE_MONITOR["stats"]["failed_tasks"] += 1
                return {
                    "error": f"CLI inference failed: {str(e)}",
                    "status": "error"
                }
        
        @mcp.tool()
        def get_cli_providers() -> Dict[str, Any]:
            """
            Get information about available CLI providers
            
            Returns:
                Dictionary with CLI provider information
            """
            return {
                "providers": CLI_PROVIDERS,
                "status": "success"
            }
        
        @mcp.tool()
        def get_cli_config(cli_type: str) -> Dict[str, Any]:
            """
            Get configuration instructions for a CLI tool
            
            Args:
                cli_type: Type of CLI tool (claude_cli, openai_cli, gemini_cli, vscode_cli)
                
            Returns:
                Dictionary with configuration steps and requirements
            """
            try:
                if cli_type not in CLI_PROVIDERS:
                    return {
                        "error": f"Unknown CLI type: {cli_type}. Supported: {list(CLI_PROVIDERS.keys())}",
                        "status": "error"
                    }
                
                # Create a temporary adapter to get config
                adapter_class_name = CLI_PROVIDERS[cli_type]["adapter_class"]
                if adapter_class_name == "ClaudeCodeAdapter":
                    adapter = ClaudeCodeAdapter("temp", None, {})
                elif adapter_class_name == "OpenAICodexAdapter":
                    adapter = OpenAICodexAdapter("temp", None, {})
                elif adapter_class_name == "GeminiCLIAdapter":
                    adapter = GeminiCLIAdapter("temp", None, {})
                elif adapter_class_name == "VSCodeCLIAdapter":
                    adapter = VSCodeCLIAdapter("temp", None, {})
                else:
                    return {
                        "error": f"Unknown adapter class: {adapter_class_name}",
                        "status": "error"
                    }
                
                config_info = adapter._config()
                config_info["status"] = "success"
                return config_info
                
            except Exception as e:
                logger.error(f"Failed to get CLI config: {e}")
                return {
                    "error": f"Failed to get CLI config: {str(e)}",
                    "status": "error"
                }
        
        @mcp.tool()
        def get_cli_install(cli_type: str) -> Dict[str, Any]:
            """
            Get installation instructions for a CLI tool
            
            Args:
                cli_type: Type of CLI tool (claude_cli, openai_cli, gemini_cli, vscode_cli)
                
            Returns:
                Dictionary with installation commands and steps
            """
            try:
                if cli_type not in CLI_PROVIDERS:
                    return {
                        "error": f"Unknown CLI type: {cli_type}. Supported: {list(CLI_PROVIDERS.keys())}",
                        "status": "error"
                    }
                
                # Create a temporary adapter to get install instructions
                adapter_class_name = CLI_PROVIDERS[cli_type]["adapter_class"]
                if adapter_class_name == "ClaudeCodeAdapter":
                    adapter = ClaudeCodeAdapter("temp", None, {})
                elif adapter_class_name == "OpenAICodexAdapter":
                    adapter = OpenAICodexAdapter("temp", None, {})
                elif adapter_class_name == "GeminiCLIAdapter":
                    adapter = GeminiCLIAdapter("temp", None, {})
                elif adapter_class_name == "VSCodeCLIAdapter":
                    adapter = VSCodeCLIAdapter("temp", None, {})
                else:
                    return {
                        "error": f"Unknown adapter class: {adapter_class_name}",
                        "status": "error"
                    }
                
                install_info = adapter._install()
                install_info["status"] = "success"
                return install_info
                
            except Exception as e:
                logger.error(f"Failed to get CLI install info: {e}")
                return {
                    "error": f"Failed to get CLI install info: {str(e)}",
                    "status": "error"
                }
        
        @mcp.tool()
        def validate_cli_config(endpoint_id: str) -> Dict[str, Any]:
            """
            Validate configuration of a registered CLI endpoint
            
            Args:
                endpoint_id: ID of the registered CLI endpoint
                
            Returns:
                Dictionary with validation results
            """
            try:
                from .cli_endpoint_adapters import get_cli_endpoint
                
                adapter = get_cli_endpoint(endpoint_id)
                if not adapter:
                    return {
                        "error": f"CLI endpoint '{endpoint_id}' not found",
                        "status": "error"
                    }
                
                validation = adapter.validate_config()
                validation["status"] = "success"
                return validation
                
            except Exception as e:
                logger.error(f"CLI config validation failed: {e}")
                return {
                    "error": f"Validation failed: {str(e)}",
                    "status": "error"
                }
        
        @mcp.tool()
        def check_cli_version(endpoint_id: str) -> Dict[str, Any]:
            """
            Check the version of a CLI tool
            
            Args:
                endpoint_id: ID of the registered CLI endpoint
                
            Returns:
                Dictionary with version information
            """
            try:
                from .cli_endpoint_adapters import get_cli_endpoint
                
                adapter = get_cli_endpoint(endpoint_id)
                if not adapter:
                    return {
                        "error": f"CLI endpoint '{endpoint_id}' not found",
                        "status": "error"
                    }
                
                version_info = adapter.check_version()
                version_info["status"] = "success"
                return version_info
                
            except Exception as e:
                logger.error(f"Version check failed: {e}")
                return {
                    "error": f"Version check failed: {str(e)}",
                    "status": "error"
                }
        
        @mcp.tool()
        def get_cli_capabilities(endpoint_id: str) -> Dict[str, Any]:
            """
            Get capabilities of a registered CLI endpoint
            
            Args:
                endpoint_id: ID of the registered CLI endpoint
                
            Returns:
                Dictionary describing endpoint capabilities
            """
            try:
                from .cli_endpoint_adapters import get_cli_endpoint
                
                adapter = get_cli_endpoint(endpoint_id)
                if not adapter:
                    return {
                        "error": f"CLI endpoint '{endpoint_id}' not found",
                        "status": "error"
                    }
                
                capabilities = adapter.get_capabilities()
                capabilities["status"] = "success"
                return capabilities
                
            except Exception as e:
                logger.error(f"Failed to get capabilities: {e}")
                return {
                    "error": f"Failed to get capabilities: {str(e)}",
                    "status": "error"
                }


def _run_local_inference(model: str, prompt: str, task_type: str, **kwargs) -> Dict[str, Any]:
    """Run inference using local models"""
    # Simulate local inference
    time.sleep(0.5)  # Simulate processing time
    
    if task_type == "text_generation":
        return {
            "generated_text": f"[Local {model}] Generated response to: {prompt[:50]}...",
            "model": model,
            "provider": "local"
        }
    elif task_type == "embedding":
        return {
            "embedding": [0.1] * 384,  # Simulated embedding
            "model": model,
            "provider": "local"
        }
    else:
        return {
            "result": f"[Local {model}] Processed {task_type} task",
            "model": model,
            "provider": "local"
        }


def _run_distributed_inference(model: str, prompt: str, task_type: str, **kwargs) -> Dict[str, Any]:
    """Run inference using libp2p distributed network"""
    # Simulate distributed inference
    time.sleep(1.0)  # Simulate network latency
    
    return {
        "result": f"[Distributed {model}] Processed via libp2p network: {prompt[:50]}...",
        "model": model,
        "provider": "libp2p",
        "nodes_used": 3,
        "sharding_strategy": "tensor_parallel"
    }


def _run_api_inference(provider: str, model: str, prompt: str, task_type: str, **kwargs) -> Dict[str, Any]:
    """Run inference using external API providers"""
    # Simulate API call
    time.sleep(0.8)  # Simulate API latency
    
    provider_config = API_PROVIDERS.get(provider, {})
    
    if provider == "openai":
        return {
            "choices": [{
                "message": {
                    "content": f"[OpenAI {model}] Response to: {prompt[:50]}..."
                }
            }],
            "model": model,
            "provider": "openai"
        }
    elif provider == "anthropic":
        return {
            "content": [{
                "text": f"[Anthropic {model}] Response to: {prompt[:50]}..."
            }],
            "model": model,
            "provider": "anthropic"
        }
    elif provider == "huggingface":
        return {
            "generated_text": f"[HuggingFace {model}] Generated: {prompt[:30]}... [continued]",
            "model": model,
            "provider": "huggingface"
        }
    else:
        return {
            "result": f"[{provider} {model}] Generic API response",
            "model": model,
            "provider": provider
        }


def _run_cli_inference(provider: str, model: str, prompt: str, task_type: str, **kwargs) -> Dict[str, Any]:
    """Run inference using CLI tools"""
    if not HAVE_CLI_ADAPTERS:
        raise ImportError("CLI adapters not available")
    
    # Find or create an endpoint for this provider/model combination
    endpoint_id = f"{provider}_{model.replace('/', '_')}"
    
    # Check if endpoint exists in CLI adapter registry
    adapter = CLI_ADAPTER_REGISTRY.get(endpoint_id)
    
    if not adapter:
        # Auto-register a CLI endpoint for this provider
        adapter_class_name = CLI_PROVIDERS[provider]["adapter_class"]
        config = {"model": model}
        
        if adapter_class_name == "ClaudeCodeAdapter":
            from .cli_endpoint_adapters import ClaudeCodeAdapter
            adapter = ClaudeCodeAdapter(endpoint_id, None, config)
        elif adapter_class_name == "OpenAICodexAdapter":
            from .cli_endpoint_adapters import OpenAICodexAdapter
            adapter = OpenAICodexAdapter(endpoint_id, None, config)
        elif adapter_class_name == "GeminiCLIAdapter":
            from .cli_endpoint_adapters import GeminiCLIAdapter
            adapter = GeminiCLIAdapter(endpoint_id, None, config)
        elif adapter_class_name == "VSCodeCLIAdapter":
            from .cli_endpoint_adapters import VSCodeCLIAdapter
            adapter = VSCodeCLIAdapter(endpoint_id, None, config)
        else:
            raise ValueError(f"Unknown adapter class: {adapter_class_name}")
        
        # Register the adapter
        from .cli_endpoint_adapters import register_cli_endpoint
        register_cli_endpoint(adapter)
    
    # Execute inference
    result = adapter.execute(prompt, task_type, **kwargs)
    
    # Return result (already includes provider info)
    return result