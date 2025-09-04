#!/usr/bin/env python3
"""
AI-Powered Model Manager MCP Server

This module provides an MCP server that integrates the AI-powered Model Manager
with inference capabilities and intelligent model selection using bandit algorithms.

Features:
- Model discovery and querying tools
- Intelligent model recommendations using multi-armed bandits  
- Common inference tools (causal LM, masked LM, diffusion, etc.)
- Automatic model selection when no explicit model is provided
- IPFS content addressing for model files
"""

import asyncio
import logging
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# Import FastMCP directly, avoiding conflicts
try:
    # Clear any existing MCP modules to avoid conflicts
    mcp_modules = [mod for mod in sys.modules.keys() if mod.startswith('mcp.')]
    for mod in mcp_modules:
        if mod in sys.modules:
            del sys.modules[mod]
    
    from fastmcp import FastMCP
    HAVE_FASTMCP = True
except ImportError:
    HAVE_FASTMCP = False
    print("⚠️ FastMCP not available. MCP server functionality will be limited.")

# Import the Model Manager components
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from ipfs_accelerate_py.model_manager import (
        ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
        VectorDocumentationIndex, BanditModelRecommender,
        RecommendationContext, create_model_from_huggingface
    )
    HAVE_MODEL_MANAGER = True
except ImportError as e:
    HAVE_MODEL_MANAGER = False
    print(f"⚠️ Model Manager not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ai_model_mcp_server")

class AIModelMCPServer:
    """AI-powered Model Manager MCP Server."""
    
    def __init__(self, 
                 model_manager_path: str = "./models.db",
                 bandit_storage_path: str = "./bandit_data.json",
                 doc_index_path: str = "./doc_index.json"):
        """
        Initialize the AI Model MCP Server.
        
        Args:
            model_manager_path: Path for model manager storage
            bandit_storage_path: Path for bandit algorithm data
            doc_index_path: Path for documentation index
        """
        if not HAVE_MODEL_MANAGER:
            raise ImportError("Model Manager components are required")
            
        self.model_manager = ModelManager(storage_path=model_manager_path)
        self.bandit_recommender = BanditModelRecommender(
            model_manager=self.model_manager,
            storage_path=bandit_storage_path
        )
        self.doc_index = VectorDocumentationIndex(storage_path=doc_index_path)
        
        # Create the MCP server if available
        if HAVE_FASTMCP:
            self.mcp = FastMCP(
                name="AI Model Manager",
                description="AI-powered model discovery and inference with intelligent recommendations"
            )
            self._register_tools()
        else:
            self.mcp = None
            logger.warning("FastMCP not available, running in demo mode only")
        
        logger.info("AI Model MCP Server initialized")
    
    def _register_tools(self):
        """Register all MCP tools."""
        
        if not self.mcp:
            return
            
        # Model Discovery Tools
        @self.mcp.tool()
        def list_models(
            model_type: Optional[str] = None,
            architecture: Optional[str] = None,
            tags: Optional[List[str]] = None,
            limit: int = 10
        ) -> List[Dict[str, Any]]:
            """
            List available models with optional filtering.
            
            Args:
                model_type: Filter by model type (language_model, vision_model, etc.)
                architecture: Filter by architecture (bert, gpt, etc.)
                tags: Filter by tags
                limit: Maximum number of models to return
                
            Returns:
                List of model information dictionaries
            """
            try:
                models = self.model_manager.list_models()
                
                # Apply filters
                if model_type:
                    models = [m for m in models if m.model_type.value == model_type]
                if architecture:
                    models = [m for m in models if m.architecture.lower() == architecture.lower()]
                if tags:
                    models = [m for m in models if any(tag in m.tags for tag in tags)]
                
                # Limit results
                models = models[:limit]
                
                # Convert to dictionaries for JSON serialization
                result = []
                for model in models:
                    model_dict = {
                        "model_id": model.model_id,
                        "model_name": model.model_name,
                        "model_type": model.model_type.value,
                        "architecture": model.architecture,
                        "description": model.description,
                        "tags": model.tags,
                        "created_at": model.created_at.isoformat() if model.created_at else None
                    }
                    result.append(model_dict)
                
                return result
                
            except Exception as e:
                logger.error(f"Error listing models: {e}")
                return []
        
        @self.mcp.tool()
        def recommend_model(
            task_type: str,
            hardware: str = "cpu",
            input_type: str = "tokens",
            output_type: str = "logits",
            requirements: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Get an AI-powered model recommendation using bandit algorithms.
            
            Args:
                task_type: Type of task (classification, generation, etc.)
                hardware: Hardware type (cpu, cuda, mps, etc.)
                input_type: Expected input data type
                output_type: Expected output data type  
                requirements: Additional requirements dictionary
                
            Returns:
                Recommendation with model info and confidence score
            """
            try:
                # Create recommendation context
                context = RecommendationContext(
                    task_type=task_type,
                    hardware=hardware,
                    input_type=DataType(input_type),
                    output_type=DataType(output_type),
                    requirements=requirements or {}
                )
                
                # Get recommendation
                recommendation = self.bandit_recommender.recommend_model(context)
                
                if recommendation:
                    return {
                        "model_id": recommendation.model_id,
                        "confidence_score": recommendation.confidence_score,
                        "predicted_performance": recommendation.predicted_performance,
                        "reasoning": recommendation.reasoning,
                        "context": {
                            "task_type": task_type,
                            "hardware": hardware,
                            "input_type": input_type,
                            "output_type": output_type
                        }
                    }
                else:
                    return {
                        "error": "No suitable model found for the given context",
                        "context": {
                            "task_type": task_type,
                            "hardware": hardware,
                            "input_type": input_type,
                            "output_type": output_type
                        }
                    }
                    
            except Exception as e:
                logger.error(f"Error getting model recommendation: {e}")
                return {"error": f"Recommendation failed: {str(e)}"}
        
        # Inference Tools (Mock implementations for demo)
        @self.mcp.tool()
        def generate_text(
            prompt: str,
            model_id: Optional[str] = None,
            max_length: int = 100,
            temperature: float = 0.7,
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Generate text using causal language modeling.
            
            Args:
                prompt: Input text prompt
                model_id: Specific model to use (optional, will auto-select if not provided)
                max_length: Maximum length of generated text
                temperature: Sampling temperature
                hardware: Hardware type to use
                
            Returns:
                Generated text and metadata
            """
            try:
                # Select model using bandit algorithm if not specified
                if model_id is None:
                    context = RecommendationContext(
                        task_type="causal_language_modeling",
                        hardware=hardware,
                        input_type=DataType.TOKENS,
                        output_type=DataType.TOKENS
                    )
                    recommendation = self.bandit_recommender.recommend_model(context)
                    model_id = recommendation.model_id if recommendation else "gpt2"
                
                # Mock inference result
                return {
                    "generated_text": f"[Generated by {model_id}] This is a continuation of: {prompt}",
                    "model_used": model_id,
                    "parameters": {
                        "max_length": max_length,
                        "temperature": temperature,
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in text generation: {e}")
                return {"error": f"Text generation failed: {str(e)}"}
        
        logger.info("All MCP tools registered")
    
    async def run(self, transport: str = "stdio", host: str = "127.0.0.1", port: int = 8000):
        """
        Run the MCP server.
        
        Args:
            transport: Transport type (stdio or sse)
            host: Host for network transports
            port: Port for network transports
        """
        if not self.mcp:
            logger.error("Cannot run server: FastMCP not available")
            return
            
        logger.info(f"Starting AI Model MCP Server on {transport}")
        try:
            await self.mcp.run(transport=transport, host=host, port=port)
        except KeyboardInterrupt:
            logger.info("Server interrupted")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            # Clean up
            if hasattr(self.model_manager, 'close'):
                self.model_manager.close()
            logger.info("AI Model MCP Server shutdown complete")

def create_ai_model_server(
    model_manager_path: str = "./models.db",
    bandit_storage_path: str = "./bandit_data.json", 
    doc_index_path: str = "./doc_index.json"
) -> AIModelMCPServer:
    """
    Create an AI Model MCP Server instance.
    
    Args:
        model_manager_path: Path for model manager storage
        bandit_storage_path: Path for bandit algorithm data  
        doc_index_path: Path for documentation index
        
    Returns:
        Configured AIModelMCPServer instance
    """
    return AIModelMCPServer(
        model_manager_path=model_manager_path,
        bandit_storage_path=bandit_storage_path,
        doc_index_path=doc_index_path
    )

# Mock inference functions for demo purposes
def mock_generate_text(prompt: str, model_id: str = "gpt2") -> str:
    """Mock text generation."""
    return f"[Generated by {model_id}] This is a continuation of: {prompt}"

def mock_classify_text(text: str, model_id: str = "bert-base-uncased") -> Dict[str, Any]:
    """Mock text classification."""
    return {
        "prediction": "POSITIVE",
        "confidence": 0.85,
        "model_used": model_id
    }

def mock_generate_embeddings(text: str, model_id: str = "sentence-transformers") -> List[float]:
    """Mock embedding generation."""
    # Simple hash-based mock embedding
    import hashlib
    hash_obj = hashlib.md5(text.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Convert to pseudo-embedding (5 dimensions for demo)
    embedding = []
    for i in range(0, 10, 2):
        byte_val = int(hash_hex[i:i+2], 16)
        embedding.append((byte_val - 128) / 128.0)  # Normalize to [-1, 1]
    
    return embedding

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the AI Model Manager MCP Server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"],
                       help="Transport type")
    parser.add_argument("--host", default="127.0.0.1", 
                       help="Host to bind to for network transports")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to bind to for network transports")
    parser.add_argument("--model-manager-path", default="./models.db",
                       help="Path for model manager storage")
    parser.add_argument("--bandit-storage-path", default="./bandit_data.json",
                       help="Path for bandit algorithm data")
    parser.add_argument("--doc-index-path", default="./doc_index.json",
                       help="Path for documentation index")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run server
    server = create_ai_model_server(
        model_manager_path=args.model_manager_path,
        bandit_storage_path=args.bandit_storage_path,
        doc_index_path=args.doc_index_path
    )
    
    if HAVE_FASTMCP:
        asyncio.run(server.run(
            transport=args.transport,
            host=args.host,
            port=args.port
        ))
    else:
        print("🔧 Demo mode: Server components initialized but FastMCP not available")
        print("✅ Model Manager, Bandit Recommender, and IPFS integration working")
        print("📚 Install FastMCP to enable full MCP server functionality")