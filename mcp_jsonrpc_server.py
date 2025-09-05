#!/usr/bin/env python3
"""
JSON-RPC MCP Server

This module provides a FastAPI-based JSON-RPC server that exposes
all MCP tools through a JSON-RPC 2.0 interface.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import our comprehensive MCP server
from comprehensive_mcp_server import ComprehensiveMCPServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONRPCError(Exception):
    """JSON-RPC error with code and message."""
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)

class MCPJSONRPCServer:
    """JSON-RPC 2.0 server for MCP tools."""
    
    def __init__(self):
        """Initialize the JSON-RPC MCP server."""
        self.app = FastAPI(
            title="MCP JSON-RPC Server", 
            description="JSON-RPC 2.0 interface for MCP AI inference tools",
            version="1.0.0"
        )
        
        # Enable CORS for web dashboard
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize the comprehensive MCP server
        try:
            self.mcp_server = ComprehensiveMCPServer()
            logger.info("MCP server initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            self.mcp_server = None
        
        # Setup routes
        self._setup_routes()
        
        # Available JSON-RPC methods
        self.methods = {
            # Model Management
            "list_models": self._list_models,
            "get_model": self._get_model,
            "add_model": self._add_model,
            "search_models": self._search_models,
            "get_model_recommendations": self._get_model_recommendations,
            
            # Text Processing
            "generate_text": self._generate_text,
            "classify_text": self._classify_text,
            "generate_embeddings": self._generate_embeddings,
            "fill_mask": self._fill_mask,
            "translate_text": self._translate_text,
            "summarize_text": self._summarize_text,
            "answer_question": self._answer_question,
            
            # Audio Processing
            "transcribe_audio": self._transcribe_audio,
            "classify_audio": self._classify_audio,
            "synthesize_speech": self._synthesize_speech,
            "generate_audio": self._generate_audio,
            
            # Vision Processing
            "classify_image": self._classify_image,
            "detect_objects": self._detect_objects,
            "segment_image": self._segment_image,
            "generate_image": self._generate_image,
            
            # Multimodal Processing
            "generate_image_caption": self._generate_image_caption,
            "answer_visual_question": self._answer_visual_question,
            "process_document": self._process_document,
            
            # Specialized
            "predict_timeseries": self._predict_timeseries,
            "generate_code": self._generate_code,
            "process_tabular_data": self._process_tabular_data,
            
            # System
            "list_methods": self._list_methods,
            "get_server_info": self._get_server_info,
        }
        
        logger.info(f"JSON-RPC server initialized with {len(self.methods)} methods")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.post("/jsonrpc")
        async def jsonrpc_endpoint(request: Request):
            """Handle JSON-RPC 2.0 requests."""
            try:
                body = await request.json()
                response = await self._handle_jsonrpc_request(body)
                return JSONResponse(content=response)
            except Exception as e:
                logger.error(f"JSON-RPC error: {e}")
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request",
                            "data": str(e)
                        },
                        "id": None
                    },
                    status_code=400
                )
        
        @self.app.get("/")
        async def root():
            """Root endpoint with server information."""
            return {
                "name": "MCP JSON-RPC Server",
                "version": "1.0.0",
                "jsonrpc": "2.0",
                "methods": list(self.methods.keys()),
                "endpoint": "/jsonrpc"
            }
        
        # Serve static files
        try:
            self.app.mount("/static", StaticFiles(directory="static"), name="static")
        except Exception as e:
            logger.warning(f"Could not mount static files: {e}")
    
    async def _handle_jsonrpc_request(self, request_data: Union[Dict, List]) -> Union[Dict, List]:
        """Handle JSON-RPC 2.0 request."""
        
        # Handle batch requests
        if isinstance(request_data, list):
            tasks = [self._handle_single_request(req) for req in request_data]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            return [resp for resp in responses if resp is not None]
        
        # Handle single request
        return await self._handle_single_request(request_data)
    
    async def _handle_single_request(self, request: Dict) -> Optional[Dict]:
        """Handle a single JSON-RPC request."""
        request_id = request.get("id")
        
        try:
            # Validate JSON-RPC 2.0 format
            if request.get("jsonrpc") != "2.0":
                raise JSONRPCError(-32600, "Invalid Request", "Missing or invalid jsonrpc version")
            
            method = request.get("method")
            if not method:
                raise JSONRPCError(-32600, "Invalid Request", "Missing method")
            
            # Handle notification (no response expected)
            if request_id is None:
                await self._call_method(method, request.get("params", {}))
                return None
            
            # Handle method call
            if method not in self.methods:
                raise JSONRPCError(-32601, "Method not found", f"Method '{method}' not found")
            
            params = request.get("params", {})
            result = await self._call_method(method, params)
            
            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            }
            
        except JSONRPCError as e:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "data": e.data
                },
                "id": request_id
            }
        except Exception as e:
            logger.error(f"Internal error in method call: {e}")
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                },
                "id": request_id
            }
    
    async def _call_method(self, method: str, params: Union[Dict, List]) -> Any:
        """Call a JSON-RPC method."""
        handler = self.methods[method]
        
        # Convert params to dict if it's a list (positional parameters)
        if isinstance(params, list):
            # For now, we'll convert to keyword arguments based on method signature
            # This is a simplified approach - in production you'd want better parameter handling
            params = {}
        
        return await handler(params)
    
    # Method implementations
    async def _list_methods(self, params: Dict) -> List[str]:
        """List all available JSON-RPC methods."""
        return list(self.methods.keys())
    
    async def _get_server_info(self, params: Dict) -> Dict:
        """Get server information."""
        return {
            "name": "MCP JSON-RPC Server",
            "version": "1.0.0",
            "jsonrpc": "2.0",
            "timestamp": datetime.now().isoformat(),
            "methods_count": len(self.methods),
            "mcp_server_available": self.mcp_server is not None
        }
    
    async def _list_models(self, params: Dict) -> Dict:
        """List available models."""
        if not self.mcp_server:
            return {"models": [], "total": 0}
        
        try:
            models = self.mcp_server.model_manager.list_models()
            return {
                "models": [model.dict() if hasattr(model, 'dict') else str(model) for model in models],
                "total": len(models)
            }
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return {"models": [], "total": 0, "error": str(e)}
    
    async def _get_model(self, params: Dict) -> Dict:
        """Get model information."""
        model_id = params.get("model_id")
        if not model_id:
            raise JSONRPCError(-32602, "Invalid params", "model_id is required")
        
        if not self.mcp_server:
            raise JSONRPCError(-32603, "Internal error", "MCP server not available")
        
        try:
            model = self.mcp_server.model_manager.get_model(model_id)
            if model:
                return model.dict() if hasattr(model, 'dict') else {"model_id": model_id, "data": str(model)}
            else:
                raise JSONRPCError(-32602, "Model not found", f"Model '{model_id}' not found")
        except Exception as e:
            logger.error(f"Error getting model {model_id}: {e}")
            raise JSONRPCError(-32603, "Internal error", str(e))
    
    async def _search_models(self, params: Dict) -> Dict:
        """Search models."""
        query = params.get("query", "")
        limit = params.get("limit", 10)
        
        if not self.mcp_server:
            return {"models": [], "total": 0}
        
        try:
            # Mock search for now - implement actual search logic
            models = self.mcp_server.model_manager.list_models()[:limit]
            return {
                "models": [model.dict() if hasattr(model, 'dict') else str(model) for model in models],
                "query": query,
                "total": len(models)
            }
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return {"models": [], "total": 0, "error": str(e)}
    
    async def _get_model_recommendations(self, params: Dict) -> Dict:
        """Get model recommendations using bandit algorithms."""
        task_type = params.get("task_type", "text_generation")
        input_type = params.get("input_type", "text")
        
        if not self.mcp_server:
            return {"recommendations": [], "algorithm": "none"}
        
        try:
            from ipfs_accelerate_py.model_manager import RecommendationContext
            
            context = RecommendationContext(
                task_type=task_type,
                input_types=[input_type],
                output_types=["text"],
                hardware_constraints={}
            )
            
            recommendation = self.mcp_server.bandit_recommender.recommend_model(context)
            
            return {
                "recommendations": [recommendation] if recommendation else [],
                "algorithm": self.mcp_server.bandit_recommender.algorithm,
                "context": {
                    "task_type": task_type,
                    "input_type": input_type
                }
            }
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return {"recommendations": [], "error": str(e)}
    
    # Text Processing Methods
    async def _generate_text(self, params: Dict) -> Dict:
        """Generate text using causal language modeling."""
        prompt = params.get("prompt", "")
        model_id = params.get("model_id")
        max_length = params.get("max_length", 100)
        temperature = params.get("temperature", 0.7)
        
        # Mock implementation - replace with actual MCP tool call
        return {
            "generated_text": f"[Generated text for prompt: '{prompt}' using model: {model_id or 'auto-selected'}]",
            "model_used": model_id or "gpt2",
            "parameters": {
                "max_length": max_length,
                "temperature": temperature
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def _classify_text(self, params: Dict) -> Dict:
        """Classify text."""
        text = params.get("text", "")
        model_id = params.get("model_id")
        
        # Mock implementation
        return {
            "classification": {
                "label": "POSITIVE",
                "confidence": 0.85,
                "all_scores": [
                    {"label": "POSITIVE", "score": 0.85},
                    {"label": "NEGATIVE", "score": 0.15}
                ]
            },
            "model_used": model_id or "bert-base-uncased",
            "text": text,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _generate_embeddings(self, params: Dict) -> Dict:
        """Generate text embeddings."""
        text = params.get("text", "")
        model_id = params.get("model_id")
        
        # Mock implementation
        import random
        embeddings = [round(random.uniform(-1, 1), 4) for _ in range(768)]
        
        return {
            "embeddings": embeddings,
            "dimension": len(embeddings),
            "model_used": model_id or "sentence-transformers/all-MiniLM-L6-v2",
            "text": text,
            "timestamp": datetime.now().isoformat()
        }
    
    # Add placeholder implementations for other methods
    async def _fill_mask(self, params: Dict) -> Dict:
        return {"result": "mask filling result", "model_used": params.get("model_id", "bert-base-uncased")}
    
    async def _translate_text(self, params: Dict) -> Dict:
        return {"translation": f"[Translated: {params.get('text', '')}]", "model_used": params.get("model_id", "t5-base")}
    
    async def _summarize_text(self, params: Dict) -> Dict:
        return {"summary": f"[Summary of: {params.get('text', '')[:50]}...]", "model_used": params.get("model_id", "bart-large-cnn")}
    
    async def _answer_question(self, params: Dict) -> Dict:
        return {"answer": f"[Answer to: {params.get('question', '')}]", "model_used": params.get("model_id", "bert-base-uncased")}
    
    # Audio Processing
    async def _transcribe_audio(self, params: Dict) -> Dict:
        return {"transcription": "[Audio transcription result]", "model_used": params.get("model_id", "whisper-base")}
    
    async def _classify_audio(self, params: Dict) -> Dict:
        return {"classification": "speech", "confidence": 0.9, "model_used": params.get("model_id", "wav2vec2")}
    
    async def _synthesize_speech(self, params: Dict) -> Dict:
        return {"audio_url": "/synthesized_audio.wav", "model_used": params.get("model_id", "tacotron2")}
    
    async def _generate_audio(self, params: Dict) -> Dict:
        return {"audio_url": "/generated_audio.wav", "model_used": params.get("model_id", "musicgen")}
    
    # Vision Processing
    async def _classify_image(self, params: Dict) -> Dict:
        return {"classification": "cat", "confidence": 0.95, "model_used": params.get("model_id", "vit-base")}
    
    async def _detect_objects(self, params: Dict) -> Dict:
        return {"objects": [{"label": "person", "bbox": [0.1, 0.1, 0.8, 0.9], "confidence": 0.9}], "model_used": params.get("model_id", "detr")}
    
    async def _segment_image(self, params: Dict) -> Dict:
        return {"segmentation_mask": "mask_data", "model_used": params.get("model_id", "segformer")}
    
    async def _generate_image(self, params: Dict) -> Dict:
        return {"image_url": "/generated_image.png", "model_used": params.get("model_id", "stable-diffusion")}
    
    # Multimodal Processing
    async def _generate_image_caption(self, params: Dict) -> Dict:
        return {"caption": "A beautiful landscape with mountains", "model_used": params.get("model_id", "blip")}
    
    async def _answer_visual_question(self, params: Dict) -> Dict:
        return {"answer": f"[Visual answer to: {params.get('question', '')}]", "model_used": params.get("model_id", "blip-vqa")}
    
    async def _process_document(self, params: Dict) -> Dict:
        return {"processed_text": "[Extracted and processed document text]", "model_used": params.get("model_id", "layoutlm")}
    
    # Specialized
    async def _predict_timeseries(self, params: Dict) -> Dict:
        return {"predictions": [1.1, 1.2, 1.3], "model_used": params.get("model_id", "time-series-transformer")}
    
    async def _generate_code(self, params: Dict) -> Dict:
        return {"code": f"# Generated code for: {params.get('description', '')}\nprint('Hello, world!')", "model_used": params.get("model_id", "codegen")}
    
    async def _process_tabular_data(self, params: Dict) -> Dict:
        return {"processed_data": "tabular analysis result", "model_used": params.get("model_id", "tabnet")}
    
    async def _add_model(self, params: Dict) -> Dict:
        """Add a new model to the manager."""
        model_id = params.get("model_id")
        if not model_id:
            raise JSONRPCError(-32602, "Invalid params", "model_id is required")
        
        # Mock implementation - would integrate with actual model manager
        return {
            "success": True,
            "model_id": model_id,
            "message": f"Model {model_id} added successfully"
        }

def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    server = MCPJSONRPCServer()
    return server.app

def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the JSON-RPC server."""
    app = create_app()
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()