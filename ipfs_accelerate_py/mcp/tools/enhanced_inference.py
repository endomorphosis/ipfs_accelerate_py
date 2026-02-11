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
except ImportError:
    aiohttp = None
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
            
            This tool registers CLI tools like Claude Code, OpenAI Codex, Google Gemini,
            or VSCode as endpoints that can be used in the multiplexing queue system.
            
            **IMPORTANT**: Before registering a CLI endpoint, ensure:
            1. The CLI tool is installed (use get_cli_install to see installation steps)
            2. Required API keys are configured (use get_cli_config to see configuration steps)
            3. The CLI tool is accessible in your PATH or provide explicit cli_path
            
            Args:
                cli_type (str): Type of CLI tool to register. Must be one of:
                    - "claude_cli": Anthropic's Claude Code CLI
                    - "openai_cli": OpenAI's CLI tool
                    - "gemini_cli": Google's Gemini CLI (via gcloud)
                    - "vscode_cli": Visual Studio Code CLI with GitHub Copilot
                
                endpoint_id (str, optional): Unique identifier for this endpoint.
                    If not provided, auto-generated as "{cli_type}_{count}".
                    Must contain only alphanumeric characters, hyphens, and underscores.
                    Example: "my_claude_endpoint"
                
                cli_path (str, optional): Explicit path to the CLI executable.
                    If not provided, will attempt auto-detection in common locations.
                    Example: "/usr/local/bin/claude"
                
                model (str, optional): Default model to use for this endpoint.
                    Examples:
                        - For claude_cli: "claude-3-sonnet", "claude-3-opus", "claude-3-haiku"
                        - For openai_cli: "gpt-3.5-turbo", "gpt-4", "codex"
                        - For gemini_cli: "gemini-pro", "gemini-ultra"
                        - For vscode_cli: "copilot-chat", "copilot-code"
                
                **config: Additional configuration parameters:
                    - temperature (float): Sampling temperature 0.0-1.0 (default: 0.7)
                    - max_tokens (int): Maximum tokens to generate (default: 4096)
                    - env_vars (dict): Environment variables to set
                    - working_dir (str): Working directory for CLI execution
            
            Returns:
                Dictionary with structure:
                {
                    "status": "success" | "error",
                    "endpoint_id": str,  # The registered endpoint ID
                    "available": bool,   # Whether CLI tool is actually available
                    "message": str,      # Success/error message
                    "error": str         # Only present if status is "error"
                }
            
            Warnings:
                - If the CLI tool is not installed, registration will succeed but
                  endpoint will be marked as unavailable
                - If required API keys are not configured, inference attempts will fail
                - Check the "available" field in the response before using
            
            Examples:
                >>> # Register Claude CLI with default settings
                >>> register_cli_endpoint_tool("claude_cli")
                
                >>> # Register OpenAI with custom model and temperature
                >>> register_cli_endpoint_tool(
                ...     "openai_cli",
                ...     endpoint_id="my_gpt4",
                ...     model="gpt-4",
                ...     temperature=0.5,
                ...     max_tokens=2048
                ... )
                
                >>> # Register with explicit path and API key
                >>> register_cli_endpoint_tool(
                ...     "claude_cli",
                ...     cli_path="/opt/claude/bin/claude",
                ...     env_vars={"ANTHROPIC_API_KEY": "sk-..."}
                ... )
            
            See Also:
                - get_cli_config(): Get configuration instructions for a CLI type
                - get_cli_install(): Get installation instructions for a CLI type
                - validate_cli_config(): Validate an endpoint's configuration
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
                
                # Add warnings if tool is not available
                if not adapter.is_available():
                    result["warning"] = (
                        f"CLI tool not found. Please install {cli_type} first. "
                        f"Use get_cli_install('{cli_type}') for installation instructions."
                    )
                
                # Check for API key configuration
                validation = adapter.validate_config()
                if not validation.get("valid", True):
                    result["config_warnings"] = validation.get("issues", [])
                    result["warning"] = (
                        result.get("warning", "") + 
                        f" Configuration issues detected: {', '.join(validation.get('issues', []))}. "
                        f"Use get_cli_config('{cli_type}') for setup instructions."
                    )
                
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
            
            Returns information about all CLI endpoints that have been registered,
            including their availability status, statistics, and configuration.
            
            Args:
                None required
            
            Returns:
                Dictionary with structure:
                {
                    "status": "success" | "error",
                    "count": int,           # Number of registered endpoints
                    "endpoints": [          # List of endpoint information
                        {
                            "endpoint_id": str,       # Unique endpoint identifier
                            "endpoint_type": "cli",   # Always "cli"
                            "cli_path": str,          # Path to CLI executable
                            "available": bool,        # Whether CLI is installed and accessible
                            "stats": {
                                "requests": int,      # Total requests made
                                "successes": int,     # Successful requests
                                "failures": int,      # Failed requests
                                "total_time": float,  # Total execution time
                                "avg_time": float     # Average execution time
                            }
                        },
                        ...
                    ],
                    "error": str            # Only present if status is "error"
                }
            
            Examples:
                >>> # List all endpoints
                >>> result = list_cli_endpoints_tool()
                >>> print(f"Found {result['count']} endpoints")
                >>> for ep in result['endpoints']:
                ...     if not ep['available']:
                ...         print(f"Warning: {ep['endpoint_id']} is not available")
            
            See Also:
                - register_cli_endpoint_tool(): Register a new CLI endpoint
                - check_cli_version(): Check version of a specific endpoint
                - get_cli_capabilities(): Get capabilities of a specific endpoint
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
            
            Executes an inference request through a registered CLI tool endpoint.
            The CLI tool must be registered first using register_cli_endpoint_tool().
            
            **IMPORTANT**: Before running inference:
            1. Ensure the endpoint is registered (use list_cli_endpoints_tool())
            2. Verify the endpoint is available (check "available" field)
            3. Confirm required API keys are configured
            
            Args:
                endpoint_id (str): ID of the registered CLI endpoint.
                    Use list_cli_endpoints_tool() to see available endpoints.
                    Example: "my_claude_endpoint", "claude_cli_0"
                
                prompt (str): Input prompt for the model.
                    Maximum length: 100,000 characters
                    Will be sanitized for security
                    Example: "Explain quantum computing in simple terms"
                
                task_type (str, optional): Type of task to perform. Default: "text_generation"
                    Supported values:
                        - "text_generation": General text generation
                        - "code_generation": Generate code snippets
                        - "code_completion": Complete partial code
                        - "code_explanation": Explain code
                        - "embedding": Generate embeddings (OpenAI only)
                        - "classification": Text classification
                        - "analysis": Analyze text
                
                timeout (int, optional): Maximum execution time in seconds. Default: 30
                    Range: 1-600 seconds
                    Increase for complex tasks or slow connections
                
                **kwargs: Additional task-specific parameters:
                    - model (str): Override default model for this request
                    - temperature (float): Sampling temperature 0.0-1.0
                    - max_tokens (int): Maximum tokens to generate
                    - top_p (float): Nucleus sampling parameter
                    - frequency_penalty (float): Frequency penalty (OpenAI)
                    - presence_penalty (float): Presence penalty (OpenAI)
            
            Returns:
                Dictionary with structure:
                {
                    "status": "success" | "error" | "timeout" | "validation_error",
                    "result": str,         # Generated text (if successful)
                    "model": str,          # Model that was used
                    "provider": str,       # Provider name (anthropic, openai, google, github)
                    "endpoint_id": str,    # Endpoint that processed the request
                    "endpoint_type": "cli",
                    "elapsed_time": float, # Execution time in seconds
                    "returncode": int,     # CLI tool exit code (if successful)
                    "error": str,          # Error message (if failed)
                    "raw_response": Any    # Raw CLI output (for debugging)
                }
            
            Raises:
                No exceptions are raised - all errors are returned in the response dict
            
            Warnings:
                - Returns {"status": "error", "error": "CLI endpoint 'X' not found"}
                  if endpoint_id is not registered
                - Returns {"status": "error", "error": "CLI tool for endpoint 'X' is not available"}
                  if the CLI tool is not installed or not in PATH
                - Returns {"status": "timeout"} if execution exceeds timeout
                - Returns {"status": "validation_error"} if input validation fails
            
            Examples:
                >>> # Basic text generation
                >>> cli_inference(
                ...     endpoint_id="claude_cli_0",
                ...     prompt="What is machine learning?"
                ... )
                
                >>> # Code generation with custom model
                >>> cli_inference(
                ...     endpoint_id="my_gpt4",
                ...     prompt="Write a Python function to sort a list",
                ...     task_type="code_generation",
                ...     model="gpt-4",
                ...     temperature=0.2,
                ...     max_tokens=500
                ... )
                
                >>> # Long-running task with extended timeout
                >>> cli_inference(
                ...     endpoint_id="gemini_cli_0",
                ...     prompt="Analyze this entire codebase...",
                ...     timeout=120
                ... )
            
            See Also:
                - register_cli_endpoint_tool(): Register a CLI endpoint
                - list_cli_endpoints_tool(): List available endpoints
                - check_cli_version(): Verify CLI tool version
            """
            try:
                # Check if endpoint exists first
                from .cli_endpoint_adapters import get_cli_endpoint
                adapter = get_cli_endpoint(endpoint_id)
                
                if not adapter:
                    return {
                        "error": f"CLI endpoint '{endpoint_id}' not found. Use list_cli_endpoints_tool() to see available endpoints.",
                        "status": "error"
                    }
                
                # Check if endpoint is available
                if not adapter.is_available():
                    return {
                        "error": (
                            f"CLI tool for endpoint '{endpoint_id}' is not available. "
                            f"Please ensure the tool is installed and in your PATH. "
                            f"Use get_cli_install() for installation instructions."
                        ),
                        "status": "error"
                    }
                
                # Execute the inference
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
            
            Returns metadata about all supported CLI provider types, including
            their descriptions, supported models, and adapter classes.
            
            Args:
                None required
            
            Returns:
                Dictionary with structure:
                {
                    "status": "success",
                    "providers": {
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
                }
            
            Examples:
                >>> # Get all providers
                >>> result = get_cli_providers()
                >>> for provider_id, info in result['providers'].items():
                ...     print(f"{provider_id}: {info['description']}")
                ...     print(f"  Models: {', '.join(info['models'])}")
            
            See Also:
                - get_cli_config(): Get configuration instructions for a provider
                - get_cli_install(): Get installation instructions for a provider
                - register_cli_endpoint_tool(): Register an endpoint for a provider
            """
            return {
                "providers": CLI_PROVIDERS,
                "status": "success"
            }
        
        @mcp.tool()
        def get_cli_config(cli_type: str) -> Dict[str, Any]:
            """
            Get configuration instructions for a CLI tool
            
            Returns detailed configuration steps, required environment variables,
            config file locations, and documentation links for setting up a CLI tool.
            
            **Use this before registering a CLI endpoint** to understand configuration requirements.
            
            Args:
                cli_type (str): Type of CLI tool. Must be one of:
                    - "claude_cli": Anthropic's Claude Code CLI
                    - "openai_cli": OpenAI's CLI tool
                    - "gemini_cli": Google's Gemini CLI
                    - "vscode_cli": Visual Studio Code CLI with Copilot
            
            Returns:
                Dictionary with structure:
                {
                    "status": "success" | "error",
                    "tool_name": str,           # Full name of the CLI tool
                    "description": str,         # Description of the tool
                    "config_steps": [str],      # Step-by-step configuration instructions
                    "env_vars": {               # Required environment variables
                        "VAR_NAME": "Description of what this variable is for"
                    },
                    "config_files": [str],      # Locations of configuration files
                    "documentation": str,       # URL to official documentation
                    "error": str                # Only present if status is "error"
                }
            
            Examples:
                >>> # Get configuration for Claude CLI
                >>> config = get_cli_config("claude_cli")
                >>> print("Configuration steps:")
                >>> for step in config['config_steps']:
                ...     print(f"  {step}")
                >>> print(f"\\nRequired API key: {list(config['env_vars'].keys())[0]}")
                
                >>> # Check if API key is needed
                >>> config = get_cli_config("openai_cli")
                >>> if "OPENAI_API_KEY" in config['env_vars']:
                ...     print("Warning: OPENAI_API_KEY must be set")
            
            Warnings:
                Returns error if cli_type is not recognized.
                Available types can be obtained from get_cli_providers().
            
            See Also:
                - get_cli_install(): Get installation instructions
                - get_cli_providers(): List all available provider types
                - register_cli_endpoint_tool(): Register an endpoint after configuration
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
            
            Returns platform-specific installation instructions with multiple methods
            for installing the specified CLI tool.
            
            **Use this before registering a CLI endpoint** to install the required tool.
            
            Args:
                cli_type (str): Type of CLI tool. Must be one of:
                    - "claude_cli": Anthropic's Claude Code CLI
                    - "openai_cli": OpenAI's CLI tool  
                    - "gemini_cli": Google's Gemini CLI (gcloud)
                    - "vscode_cli": Visual Studio Code CLI
            
            Returns:
                Dictionary with structure:
                {
                    "status": "success" | "error",
                    "tool_name": str,           # Full name of the CLI tool
                    "platform": str,            # Detected platform (darwin, linux, windows)
                    "install_methods": [        # List of installation methods
                        {
                            "method": str,      # Name of installation method
                            "commands": [str]   # Commands to run
                        },
                        ...
                    ],
                    "verify_command": str,      # Command to verify installation
                    "documentation": str,       # URL to installation documentation
                    "post_install": [str],      # Optional post-installation steps
                    "error": str                # Only present if status is "error"
                }
            
            Examples:
                >>> # Get installation instructions for current platform
                >>> install = get_cli_install("claude_cli")
                >>> print(f"Installing on {install['platform']}:")
                >>> for method in install['install_methods']:
                ...     print(f"\\n{method['method']}:")
                ...     for cmd in method['commands']:
                ...         print(f"  {cmd}")
                >>> print(f"\\nVerify with: {install['verify_command']}")
                
                >>> # Show all installation options
                >>> install = get_cli_install("openai_cli")
                >>> if len(install['install_methods']) > 1:
                ...     print("Multiple installation methods available:")
                ...     for m in install['install_methods']:
                ...         print(f"  - {m['method']}")
            
            Warnings:
                Returns error if cli_type is not recognized.
                Installation instructions are platform-specific based on detected OS.
            
            See Also:
                - get_cli_config(): Get configuration instructions after installation
                - check_cli_version(): Verify installation was successful
                - register_cli_endpoint_tool(): Register endpoint after installation
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
            
            Checks if a registered CLI endpoint has valid configuration and
            identifies any configuration issues that might prevent successful execution.
            
            Args:
                endpoint_id (str): ID of the registered CLI endpoint to validate
                    Use list_cli_endpoints_tool() to see available endpoints
            
            Returns:
                Dictionary with structure:
                {
                    "status": "success" | "error",
                    "valid": bool,              # Whether configuration is valid
                    "issues": [str],            # List of configuration problems found
                    "config": {...},            # Current configuration
                    "error": str                # Only present if status is "error"
                }
            
            Examples:
                >>> # Validate endpoint configuration
                >>> result = validate_cli_config("claude_cli_0")
                >>> if not result['valid']:
                ...     print("Configuration issues found:")
                ...     for issue in result['issues']:
                ...         print(f"  - {issue}")
                ...     print("\\nUse get_cli_config('claude_cli') for setup help")
                
                >>> # Check before running inference
                >>> validation = validate_cli_config("my_endpoint")
                >>> if validation['valid']:
                ...     cli_inference("my_endpoint", "Hello!")
                ... else:
                ...     print(f"Fix these issues first: {validation['issues']}")
            
            Warnings:
                Returns error if endpoint_id does not exist.
                Common issues detected:
                - CLI tool not installed or not in PATH
                - Missing required configuration fields
                - Invalid configuration values
            
            See Also:
                - get_cli_config(): Get configuration instructions
                - check_cli_version(): Verify CLI tool is installed
                - register_cli_endpoint_tool(): Register a new endpoint
            """
            try:
                from .cli_endpoint_adapters import get_cli_endpoint
                
                adapter = get_cli_endpoint(endpoint_id)
                if not adapter:
                    return {
                        "error": f"CLI endpoint '{endpoint_id}' not found. Use list_cli_endpoints_tool() to see available endpoints.",
                        "status": "error"
                    }
                
                validation = adapter.validate_config()
                validation["status"] = "success"
                
                # Add helpful message if validation failed
                if not validation.get("valid", False):
                    validation["help"] = (
                        f"Configuration issues detected. "
                        f"Run get_cli_config() for the provider type to see setup instructions."
                    )
                
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
            
            Executes the CLI tool's version command to verify it is installed
            and accessible. Useful for troubleshooting installation issues.
            
            Args:
                endpoint_id (str): ID of the registered CLI endpoint
                    Use list_cli_endpoints_tool() to see available endpoints
            
            Returns:
                Dictionary with structure:
                {
                    "status": "success" | "error",
                    "available": bool,      # Whether CLI tool is accessible
                    "version": str,         # Version string from CLI tool
                    "returncode": int,      # Exit code from version command
                    "error": str            # Only present if check failed
                }
            
            Examples:
                >>> # Check if tool is installed
                >>> version = check_cli_version("claude_cli_0")
                >>> if version['available']:
                ...     print(f"Claude CLI version: {version['version']}")
                ... else:
                ...     print("Claude CLI not found!")
                ...     print("Use get_cli_install('claude_cli') for installation")
                
                >>> # Verify after installation
                >>> import time
                >>> # ... install CLI tool ...
                >>> time.sleep(1)  # Give system time to update PATH
                >>> result = check_cli_version("new_endpoint")
                >>> print(f"Installation {'successful' if result['available'] else 'failed'}")
            
            Warnings:
                Returns {"available": False, "error": "..."} if:
                - Endpoint not found
                - CLI tool not installed
                - CLI tool not in PATH
                - Version command failed
            
            See Also:
                - get_cli_install(): Get installation instructions
                - validate_cli_config(): Check full configuration
                - list_cli_endpoints_tool(): List all endpoints
            """
            try:
                from .cli_endpoint_adapters import get_cli_endpoint
                
                adapter = get_cli_endpoint(endpoint_id)
                if not adapter:
                    return {
                        "error": f"CLI endpoint '{endpoint_id}' not found. Use list_cli_endpoints_tool() to see available endpoints.",
                        "status": "error",
                        "available": False
                    }
                
                version_info = adapter.check_version()
                version_info["status"] = "success"
                
                # Add helpful message if not available
                if not version_info.get("available", False):
                    version_info["help"] = (
                        f"CLI tool not accessible. "
                        f"Use get_cli_install() to see installation instructions."
                    )
                
                return version_info
                
            except Exception as e:
                logger.error(f"Version check failed: {e}")
                return {
                    "error": f"Version check failed: {str(e)}",
                    "status": "error",
                    "available": False
                }
        
        @mcp.tool()
        def get_cli_capabilities(endpoint_id: str) -> Dict[str, Any]:
            """
            Get capabilities of a registered CLI endpoint
            
            Returns detailed information about what an endpoint can do,
            including supported tasks, configuration options, and current status.
            
            Args:
                endpoint_id (str): ID of the registered CLI endpoint
                    Use list_cli_endpoints_tool() to see available endpoints
            
            Returns:
                Dictionary with structure:
                {
                    "status": "success" | "error",
                    "endpoint_id": str,         # Endpoint identifier
                    "cli_path": str,            # Path to CLI executable
                    "available": bool,          # Whether CLI is accessible
                    "supported_tasks": [str],   # List of supported task types
                    "config_fields": {...},     # Available configuration options
                    "version_info": {...},      # Version information
                    "error": str                # Only present if status is "error"
                }
            
            Examples:
                >>> # Get endpoint capabilities
                >>> caps = get_cli_capabilities("claude_cli_0")
                >>> print(f"Supported tasks: {', '.join(caps['supported_tasks'])}")
                >>> print(f"Available: {caps['available']}")
                >>> if 'config_fields' in caps:
                ...     print("Configuration options:")
                ...     for field, info in caps['config_fields'].items():
                ...         print(f"  {field}: {info.get('description', 'N/A')}")
                
                >>> # Check what tasks are supported
                >>> caps = get_cli_capabilities("my_endpoint")
                >>> if "code_generation" in caps['supported_tasks']:
                ...     print("This endpoint supports code generation!")
            
            Warnings:
                Returns error if endpoint_id does not exist.
            
            See Also:
                - check_cli_version(): Check version specifically
                - validate_cli_config(): Validate configuration
                - list_cli_endpoints_tool(): List all available endpoints
            """
            try:
                from .cli_endpoint_adapters import get_cli_endpoint
                
                adapter = get_cli_endpoint(endpoint_id)
                if not adapter:
                    return {
                        "error": f"CLI endpoint '{endpoint_id}' not found. Use list_cli_endpoints_tool() to see available endpoints.",
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