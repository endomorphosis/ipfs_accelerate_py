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
import os
import threading
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

# Best-effort ensure minimal server deps when allowed
try:
    from ipfs_accelerate_py.utils.auto_install import ensure_packages
    ensure_packages({
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        # optional but commonly used in this server
        "huggingface_hub": "huggingface_hub",
    })
except Exception:
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Set up logging first
logger = logging.getLogger(__name__)

# Defer importing the comprehensive MCP server to avoid heavy deps at import-time
ComprehensiveMCPServer = None  # will be imported lazily in __init__

# Import HuggingFace model search service
from tools.huggingface_model_search import get_hf_search_service

# Try to import model management dependencies
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError
    HAVE_HF_HUB = True
except ImportError:
    HAVE_HF_HUB = False
    logger.warning("HuggingFace Hub not available for downloads")

# Try to import ipfs_accelerate_py model manager
try:
    # Avoid importing heavy core components when only model manager is needed
    os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
    from ipfs_accelerate_py.model_manager import ModelManager
    HAVE_MODEL_MANAGER = True
except ImportError:
    try:
        from .ipfs_accelerate_py.model_manager import ModelManager
        HAVE_MODEL_MANAGER = True
    except ImportError:
        HAVE_MODEL_MANAGER = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDownloadManager:
    """Manages model downloads and tracks download progress."""
    
    def __init__(self):
        self.downloads = {}  # download_id -> download_info
        self.downloaded_models = {}  # model_id -> model_info
        self._load_downloaded_models()
    
    def _load_downloaded_models(self):
        """Load information about previously downloaded models."""
        try:
            models_file = Path("./downloaded_models/models_registry.json")
            if models_file.exists():
                with open(models_file, 'r') as f:
                    self.downloaded_models = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load downloaded models registry: {e}")
            self.downloaded_models = {}
    
    def _save_downloaded_models(self):
        """Save information about downloaded models."""
        try:
            models_file = Path("./downloaded_models/models_registry.json")
            models_file.parent.mkdir(exist_ok=True)
            with open(models_file, 'w') as f:
                json.dump(self.downloaded_models, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save downloaded models registry: {e}")
    
    def start_download(self, model_id: str, model_name: str, download_type: str = "snapshot") -> str:
        """Start a model download and return download ID."""
        download_id = str(uuid.uuid4())
        
        self.downloads[download_id] = {
            "id": download_id,
            "model_id": model_id,
            "model_name": model_name,
            "download_type": download_type,
            "status": "starting",
            "progress": 0,
            "total_size": None,
            "downloaded_size": 0,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "error": None,
            "files": []
        }
        
        return download_id
    
    def update_download_progress(self, download_id: str, progress: int, 
                                downloaded_size: int = 0, total_size: int = None):
        """Update download progress."""
        if download_id in self.downloads:
            self.downloads[download_id].update({
                "progress": progress,
                "downloaded_size": downloaded_size,
                "total_size": total_size,
                "status": "downloading" if progress < 100 else "completing"
            })
    
    def complete_download(self, download_id: str, model_path: str):
        """Mark download as completed."""
        if download_id in self.downloads:
            download_info = self.downloads[download_id]
            download_info.update({
                "status": "completed",
                "progress": 100,
                "completed_at": datetime.now().isoformat(),
                "model_path": model_path
            })
            
            # Add to downloaded models registry
            model_id = download_info["model_id"]
            self.downloaded_models[model_id] = {
                "model_id": model_id,
                "model_name": download_info["model_name"],
                "download_type": download_info["download_type"],
                "downloaded_at": download_info["completed_at"],
                "model_path": model_path,
                "size_mb": download_info.get("total_size", 0) / (1024 * 1024) if download_info.get("total_size") else 0
            }
            
            self._save_downloaded_models()
    
    def fail_download(self, download_id: str, error: str):
        """Mark download as failed."""
        if download_id in self.downloads:
            self.downloads[download_id].update({
                "status": "failed",
                "completed_at": datetime.now().isoformat(),
                "error": error
            })
    
    def get_download_status(self, download_id: str) -> Optional[Dict]:
        """Get download status."""
        return self.downloads.get(download_id)
    
    def is_model_downloaded(self, model_id: str) -> bool:
        """Check if model is already downloaded."""
        return model_id in self.downloaded_models
    
    def get_downloaded_model_info(self, model_id: str) -> Optional[Dict]:
        """Get information about a downloaded model."""
        return self.downloaded_models.get(model_id)
    
    def list_downloaded_models(self) -> List[Dict]:
        """List all downloaded models."""
        return list(self.downloaded_models.values())
    
    def remove_downloaded_model(self, model_id: str) -> bool:
        """Remove a downloaded model."""
        if model_id in self.downloaded_models:
            model_info = self.downloaded_models[model_id]
            
            # Try to remove model files
            try:
                model_path = Path(model_info["model_path"])
                if model_path.exists():
                    if model_path.is_dir():
                        import shutil
                        shutil.rmtree(model_path)
                    else:
                        model_path.unlink()
            except Exception as e:
                logger.warning(f"Could not remove model files for {model_id}: {e}")
            
            # Remove from registry
            del self.downloaded_models[model_id]
            self._save_downloaded_models()
            return True
        return False

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
        
        # Initialize the comprehensive MCP server (optional)
        try:
            global ComprehensiveMCPServer
            if ComprehensiveMCPServer is None:
                from tools.comprehensive_mcp_server import ComprehensiveMCPServer as _CMS
                ComprehensiveMCPServer = _CMS
            self.mcp_server = ComprehensiveMCPServer()
            logger.info("MCP server initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            self.mcp_server = None
        
        # Initialize model download management
        self.download_manager = ModelDownloadManager()
        
        # Initialize model storage directory
        self.models_dir = Path("./downloaded_models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Setup routes
        self._setup_routes()
        
        # Available JSON-RPC methods
        self.methods = {
            # Core System Methods (required by portable SDK)
            "ping": self._ping,
            "get_server_info": self._get_server_info,
            "get_available_methods": self._get_available_methods,
            "get_models": self._get_models,
            
            # Model Management
            "list_models": self._list_models,
            "get_model": self._get_model,
            "add_model": self._add_model,
            "search_models": self._search_models,
            "get_model_recommendations": self._get_model_recommendations,
            
            # Text Processing
            "generate_text": self._generate_text,
            "classify_text": self._classify_text,
            "get_text_embedding": self._get_text_embedding,
            "generate_embeddings": self._generate_embeddings,
            "fill_mask": self._fill_mask,
            "translate_text": self._translate_text,
            "summarize_text": self._summarize_text,
            "answer_question": self._answer_question,
            "analyze_sentiment": self._analyze_sentiment,
            "extract_entities": self._extract_entities,
            
            # Audio Processing
            "transcribe_audio": self._transcribe_audio,
            "classify_audio": self._classify_audio,
            "synthesize_speech": self._synthesize_speech,
            "generate_audio": self._generate_audio,
            "denoise_audio": self._denoise_audio,
            
            # Vision Processing
            "classify_image": self._classify_image,
            "detect_objects": self._detect_objects,
            "segment_image": self._segment_image,
            "generate_image": self._generate_image,
            "caption_image": self._caption_image,
            "enhance_image": self._enhance_image,
            
            # Code Processing
            "generate_code": self._generate_code,
            "complete_code": self._complete_code,
            "explain_code": self._explain_code,
            "debug_code": self._debug_code,
            "optimize_code": self._optimize_code,
            
            # Multimodal Processing
            "generate_image_caption": self._generate_image_caption,
            "visual_question_answering": self._visual_question_answering,
            "answer_visual_question": self._answer_visual_question,
            "multimodal_chat": self._multimodal_chat,
            "multimodal_analysis": self._multimodal_analysis,
            "process_document": self._process_document,
            
            # Specialized
            "predict_timeseries": self._predict_timeseries,
            "process_tabular_data": self._process_tabular_data,
            
            # System and Hardware Methods (enhanced for ipfs_accelerate_py integration)
            "get_hardware_info": self._get_hardware_info,
            "get_system_metrics": self._get_system_metrics,
            "analyze_emotion": self._analyze_emotion,
            "extract_topics": self._extract_topics,
            "extract_text_from_image": self._extract_text_from_image,
            
            # HuggingFace Model Search Methods
            "search_huggingface_models": self._search_huggingface_models,
            "get_huggingface_model_details": self._get_huggingface_model_details,
            "get_model_search_suggestions": self._get_model_search_suggestions,
            "get_model_search_stats": self._get_model_search_stats,
            "initialize_model_search": self._initialize_model_search,
            
            # Model Download and Management Methods
            "download_huggingface_model": self._download_huggingface_model,
            "get_download_status": self._get_download_status,
            "list_downloaded_models": self._list_downloaded_models,
            "remove_downloaded_model": self._remove_downloaded_model,
            "test_model_inference": self._test_model_inference,
            "get_model_download_info": self._get_model_download_info,
            
            # Legacy aliases
            "list_methods": self._get_available_methods,
        }
        
        logger.info(f"JSON-RPC server initialized with {len(self.methods)} methods")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        # Mount static files
        try:
            import os
            static_path = os.path.join(os.path.dirname(__file__), "static")
            if os.path.exists(static_path):
                self.app.mount("/static", StaticFiles(directory=static_path), name="static")
                logger.info(f"Mounted static files from {static_path}")
        except Exception as e:
            logger.warning(f"Could not mount static files: {e}")
        
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
            """Root endpoint - serve the reorganized dashboard."""
            try:
                import os
                from fastapi.responses import FileResponse
                templates_path = os.path.join(os.path.dirname(__file__), "templates")
                # Priority: reorganized > enhanced > original
                reorganized_path = os.path.join(templates_path, "reorganized_dashboard.html")
                enhanced_path = os.path.join(templates_path, "enhanced_dashboard.html")
                original_path = os.path.join(templates_path, "sdk_dashboard.html")
                
                if os.path.exists(reorganized_path):
                    logger.info("Serving reorganized modular dashboard")
                    return FileResponse(reorganized_path)
                elif os.path.exists(enhanced_path):
                    logger.info("Serving enhanced dashboard")
                    return FileResponse(enhanced_path)
                elif os.path.exists(original_path):
                    logger.info("Serving original dashboard")
                    return FileResponse(original_path)
            except Exception as e:
                logger.warning(f"Could not serve dashboard: {e}")
            
            # Fallback to API information
            return {
                "name": "IPFS Accelerate AI - MCP Server",
                "version": "2.0.0",
                "jsonrpc": "2.0",
                "methods": list(self.methods.keys()),
                "endpoint": "/jsonrpc",
                "dashboard": "Reorganized Modular AI Dashboard with portable SDK and enhanced ipfs_accelerate_py integration"
            }
        
        @self.app.get("/dashboard")
        async def dashboard():
            """Dashboard endpoint - serve reorganized dashboard."""
            try:
                import os
                from fastapi.responses import FileResponse
                templates_path = os.path.join(os.path.dirname(__file__), "templates")
                reorganized_path = os.path.join(templates_path, "reorganized_dashboard.html")
                enhanced_path = os.path.join(templates_path, "enhanced_dashboard.html")
                original_path = os.path.join(templates_path, "sdk_dashboard.html")
                
                if os.path.exists(reorganized_path):
                    return FileResponse(reorganized_path)
                elif os.path.exists(enhanced_path):
                    return FileResponse(enhanced_path)
                elif os.path.exists(original_path):
                    return FileResponse(original_path)
            except Exception as e:
                logger.warning(f"Could not serve dashboard: {e}")
            
            return {"error": "Dashboard not found"}
        
        
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
        """Get comprehensive server information including ipfs_accelerate_py integration."""
        try:
            # Try to get hardware info from ipfs_accelerate_py
            hardware_info = {}
            try:
                from ipfs_accelerate_py import hardware_detection
                hardware_info = hardware_detection.detect_hardware()
            except ImportError:
                logger.warning("ipfs_accelerate_py hardware detection not available")
            except Exception as e:
                logger.warning(f"Hardware detection failed: {e}")
            
            return {
                "name": "IPFS Accelerate AI - MCP Server",
                "version": "1.0.0",
                "jsonrpc": "2.0",
                "timestamp": datetime.now().isoformat(),
                "methods_count": len(self.methods),
                "mcp_server_available": self.mcp_server is not None,
                "ipfs_accelerate_py_integration": True,
                "hardware_info": hardware_info,
                "features": [
                    "Text Generation & Analysis",
                    "Computer Vision",
                    "Audio Processing", 
                    "Multimodal AI",
                    "Code Generation",
                    "Model Management",
                    "Hardware Detection",
                    "Performance Monitoring"
                ]
            }
        except Exception as e:
            logger.error(f"Error getting server info: {e}")
            return {
                "name": "IPFS Accelerate AI - MCP Server",
                "version": "1.0.0",
                "jsonrpc": "2.0",
                "timestamp": datetime.now().isoformat(),
                "methods_count": len(self.methods),
                "mcp_server_available": self.mcp_server is not None,
                "error": str(e)
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

    # ============================================
    # ENHANCED INTEGRATION METHODS
    # ============================================

    async def _get_hardware_info(self, params: Dict) -> Dict:
        """Get detailed hardware information from ipfs_accelerate_py."""
        try:
            from ipfs_accelerate_py import hardware_detection
            hardware_info = hardware_detection.detect_hardware()
            
            # Enhance with additional system info
            import psutil
            import platform
            
            enhanced_info = {
                **hardware_info,
                "system": {
                    "platform": platform.platform(),
                    "architecture": platform.architecture()[0],
                    "processor": platform.processor(),
                    "python_version": platform.python_version(),
                },
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent
                },
                "cpu": {
                    "count": psutil.cpu_count(),
                    "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                    "usage": psutil.cpu_percent(interval=1)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return enhanced_info
            
        except ImportError:
            logger.warning("Hardware detection dependencies not available")
            return {
                "error": "Hardware detection not available",
                "message": "ipfs_accelerate_py or dependencies not installed"
            }
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            return {
                "error": "Hardware detection failed",
                "message": str(e)
            }

    async def _get_system_metrics(self, params: Dict) -> Dict:
        """Get current system performance metrics."""
        try:
            import psutil
            
            # Get current metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Try to get GPU info if available
            gpu_usage = 0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except ImportError:
                pass
            
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "gpu_usage": gpu_usage,
                "memory_total": memory.total,
                "memory_available": memory.available,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"System metrics failed: {e}")
            return {
                "error": "System metrics not available",
                "message": str(e)
            }

    async def _analyze_emotion(self, params: Dict) -> Dict:
        """Analyze emotion in text using advanced models."""
        text = params.get("text", "")
        model_id = params.get("model_id")
        
        if not text:
            raise JSONRPCError(-32602, "Invalid params", "text is required")
        
        # Enhanced emotion analysis - integrate with ipfs_accelerate_py if available
        try:
            if self.mcp_server:
                # Try to use actual emotion analysis model
                result = await self._call_ipfs_accelerate_model("emotion_analysis", {
                    "text": text,
                    "model_id": model_id
                })
                if result:
                    return result
        except Exception as e:
            logger.warning(f"Advanced emotion analysis failed: {e}")
        
        # Fallback to mock implementation
        emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
        import random
        
        emotion_scores = {}
        for emotion in emotions:
            emotion_scores[emotion] = round(random.uniform(0, 1), 3)
        
        # Normalize scores
        total = sum(emotion_scores.values())
        emotion_scores = {k: round(v/total, 3) for k, v in emotion_scores.items()}
        
        # Get dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        return {
            "dominant_emotion": dominant_emotion[0],
            "confidence": dominant_emotion[1],
            "all_emotions": emotion_scores,
            "text": text,
            "model_used": model_id or "emotion-analysis-model",
            "timestamp": datetime.now().isoformat()
        }

    async def _extract_topics(self, params: Dict) -> Dict:
        """Extract topics from text using topic modeling."""
        text = params.get("text", "")
        model_id = params.get("model_id")
        num_topics = params.get("num_topics", 5)
        
        if not text:
            raise JSONRPCError(-32602, "Invalid params", "text is required")
        
        # Enhanced topic extraction - integrate with ipfs_accelerate_py if available
        try:
            if self.mcp_server:
                result = await self._call_ipfs_accelerate_model("topic_modeling", {
                    "text": text,
                    "model_id": model_id,
                    "num_topics": num_topics
                })
                if result:
                    return result
        except Exception as e:
            logger.warning(f"Advanced topic modeling failed: {e}")
        
        # Fallback to mock implementation
        sample_topics = [
            "Technology", "Science", "Business", "Health", "Education", 
            "Entertainment", "Sports", "Politics", "Travel", "Food"
        ]
        
        import random
        topics = []
        for i in range(min(num_topics, len(sample_topics))):
            topic = random.choice(sample_topics)
            if topic not in [t["topic"] for t in topics]:
                topics.append({
                    "topic": topic,
                    "confidence": round(random.uniform(0.3, 0.9), 3),
                    "keywords": random.sample(text.split()[:10], min(3, len(text.split())))
                })
        
        return {
            "topics": topics,
            "num_topics_found": len(topics),
            "text_length": len(text),
            "model_used": model_id or "topic-modeling-model",
            "timestamp": datetime.now().isoformat()
        }

    async def _extract_text_from_image(self, params: Dict) -> Dict:
        """Extract text from images using OCR."""
        image_data = params.get("image", "")
        model_id = params.get("model_id")
        
        if not image_data:
            raise JSONRPCError(-32602, "Invalid params", "image data is required")
        
        # Enhanced OCR - integrate with ipfs_accelerate_py if available
        try:
            if self.mcp_server:
                result = await self._call_ipfs_accelerate_model("ocr", {
                    "image": image_data,
                    "model_id": model_id
                })
                if result:
                    return result
        except Exception as e:
            logger.warning(f"Advanced OCR failed: {e}")
        
        # Fallback to mock implementation
        return {
            "extracted_text": "[OCR would extract text from the provided image]",
            "confidence": 0.85,
            "detected_languages": ["en"],
            "text_regions": [
                {
                    "text": "Sample extracted text",
                    "bbox": [0.1, 0.1, 0.8, 0.2],
                    "confidence": 0.9
                }
            ],
            "model_used": model_id or "ocr-model",
            "timestamp": datetime.now().isoformat()
        }

    async def _ping(self, params: Dict) -> str:
        """Ping the server to check if it's alive."""
        return "pong"

    async def _get_available_methods(self, params: Dict) -> List[str]:
        """Get list of available JSON-RPC methods."""
        return list(self.methods.keys())

    async def _get_models(self, params: Dict) -> List[Dict]:
        """Get list of available models (alias for list_models)."""
        return await self._list_models(params)

    async def _get_text_embedding(self, params: Dict) -> Dict:
        """Get text embeddings (alias for generate_embeddings)."""
        return await self._generate_embeddings(params)

    async def _analyze_sentiment(self, params: Dict) -> Dict:
        """Analyze sentiment of text."""
        text = params.get("text", "")
        if not text:
            raise JSONRPCError(-32602, "Invalid params", "text is required")

        # Mock sentiment analysis - replace with actual model
        import random
        sentiments = ["positive", "negative", "neutral"]
        sentiment = random.choice(sentiments)
        confidence = round(random.uniform(0.7, 0.95), 3)

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "scores": {
                "positive": round(random.uniform(0, 1), 3),
                "negative": round(random.uniform(0, 1), 3),
                "neutral": round(random.uniform(0, 1), 3)
            },
            "text": text,
            "timestamp": datetime.now().isoformat()
        }

    async def _extract_entities(self, params: Dict) -> Dict:
        """Extract named entities from text."""
        text = params.get("text", "")
        entity_types = params.get("entity_types", [])
        
        if not text:
            raise JSONRPCError(-32602, "Invalid params", "text is required")

        # Mock entity extraction - replace with actual model
        entities = [
            {
                "text": "example entity",
                "label": "PERSON",
                "start": 0,
                "end": 14,
                "confidence": 0.95
            }
        ]

        return {
            "entities": entities,
            "text": text,
            "entity_types": entity_types,
            "timestamp": datetime.now().isoformat()
        }

    async def _denoise_audio(self, params: Dict) -> Dict:
        """Denoise audio data."""
        audio_data = params.get("audio_data", "")
        if not audio_data:
            raise JSONRPCError(-32602, "Invalid params", "audio_data is required")

        # Mock denoising - replace with actual model
        return {
            "denoised_audio": "[Denoised audio data would be returned here]",
            "noise_reduction_db": 12.5,
            "sample_rate": 44100,
            "duration": 5.2,
            "timestamp": datetime.now().isoformat()
        }

    async def _caption_image(self, params: Dict) -> Dict:
        """Generate image caption (alias for generate_image_caption)."""
        return await self._generate_image_caption(params)

    async def _enhance_image(self, params: Dict) -> Dict:
        """Enhance image quality."""
        image_data = params.get("image_data", "")
        options = params.get("options", {})
        
        if not image_data:
            raise JSONRPCError(-32602, "Invalid params", "image_data is required")

        # Mock enhancement - replace with actual model
        return {
            "enhanced_image": "[Enhanced image data would be returned here]",
            "enhancement_type": options.get("type", "auto"),
            "improvements": {
                "sharpness": "+15%",
                "brightness": "+5%",
                "contrast": "+10%"
            },
            "timestamp": datetime.now().isoformat()
        }

    async def _complete_code(self, params: Dict) -> Dict:
        """Complete code snippet."""
        code = params.get("code", "")
        language = params.get("language", "python")
        
        if not code:
            raise JSONRPCError(-32602, "Invalid params", "code is required")

        # Mock code completion - replace with actual model
        completion = f"\n    # Completed code for {language}\n    return result"
        
        return {
            "completed_code": code + completion,
            "original_code": code,
            "language": language,
            "completion_suggestions": [completion],
            "timestamp": datetime.now().isoformat()
        }

    async def _explain_code(self, params: Dict) -> Dict:
        """Explain what code does."""
        code = params.get("code", "")
        language = params.get("language", "python")
        
        if not code:
            raise JSONRPCError(-32602, "Invalid params", "code is required")

        # Mock explanation - replace with actual model
        return {
            "explanation": f"This {language} code appears to perform a specific function.",
            "code": code,
            "language": language,
            "complexity": "medium",
            "key_concepts": ["variables", "functions", "control flow"],
            "timestamp": datetime.now().isoformat()
        }

    async def _debug_code(self, params: Dict) -> Dict:
        """Debug code and suggest fixes."""
        code = params.get("code", "")
        error = params.get("error", "")
        language = params.get("language", "python")
        
        if not code:
            raise JSONRPCError(-32602, "Invalid params", "code is required")

        # Mock debugging - replace with actual model
        return {
            "fixed_code": code,
            "error_analysis": error,
            "suggested_fixes": ["Check variable names", "Verify syntax"],
            "language": language,
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat()
        }

    async def _optimize_code(self, params: Dict) -> Dict:
        """Optimize code for performance."""
        code = params.get("code", "")
        language = params.get("language", "python")
        
        if not code:
            raise JSONRPCError(-32602, "Invalid params", "code is required")

        # Mock optimization - replace with actual model
        return {
            "optimized_code": code,
            "original_code": code,
            "optimizations": ["Removed redundant loops", "Improved algorithm"],
            "performance_gain": "15%",
            "language": language,
            "timestamp": datetime.now().isoformat()
        }

    async def _visual_question_answering(self, params: Dict) -> Dict:
        """Answer questions about images (alias for answer_visual_question)."""
        return await self._answer_visual_question(params)

    async def _multimodal_chat(self, params: Dict) -> Dict:
        """Multimodal chat with text and images."""
        messages = params.get("messages", [])
        image_data = params.get("image_data")
        
        if not messages:
            raise JSONRPCError(-32602, "Invalid params", "messages is required")

        # Mock multimodal chat - replace with actual model
        return {
            "response": "I can see the image and understand your question. Here's my response...",
            "messages": messages,
            "has_image": bool(image_data),
            "confidence": 0.88,
            "timestamp": datetime.now().isoformat()
        }

    async def _multimodal_analysis(self, params: Dict) -> Dict:
        """Analyze multimodal data."""
        data = params.get("data", "")
        data_type = params.get("data_type", "auto")
        
        if not data:
            raise JSONRPCError(-32602, "Invalid params", "data is required")

        # Mock analysis - replace with actual model
        return {
            "analysis": "Multimodal analysis results...",
            "data_type": data_type,
            "detected_modalities": ["text", "image"],
            "insights": ["Key insight 1", "Key insight 2"],
            "confidence": 0.92,
            "timestamp": datetime.now().isoformat()
        }

    # ============================================
    # HUGGINGFACE MODEL SEARCH METHODS
    # ============================================

    async def _search_huggingface_models(self, params: Dict) -> Dict:
        """
        Search HuggingFace models with vector and BM25 search capabilities.
        
        Parameters:
        - query (str): Search query
        - search_type (str): "vector", "bm25", or "hybrid" (default: "hybrid")
        - filters (dict): Filters to apply (task, library, language, author, etc.)
        - sort_by (str): Sort field ("relevance", "downloads", "likes", "date", "name")
        - sort_order (str): "asc" or "desc" (default: "desc")
        - offset (int): Results offset for pagination (default: 0)
        - limit (int): Maximum results to return (default: 20)
        """
        try:
            query = params.get("query", "")
            search_type = params.get("search_type", "hybrid")
            filters = params.get("filters", {})
            sort_by = params.get("sort_by", "relevance")
            sort_order = params.get("sort_order", "desc")
            offset = params.get("offset", 0)
            limit = params.get("limit", 20)
            
            # Validate parameters
            if limit > 100:
                limit = 100
            if offset < 0:
                offset = 0
            
            # Get search service
            search_service = await get_hf_search_service()
            
            # Perform search
            results = await search_service.search_models(
                query=query,
                search_type=search_type,
                filters=filters,
                sort_by=sort_by,
                sort_order=sort_order,
                offset=offset,
                limit=limit
            )
            
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                **results
            }
            
        except Exception as e:
            logger.error(f"HuggingFace model search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "total": 0,
                "timestamp": datetime.now().isoformat()
            }

    async def _get_huggingface_model_details(self, params: Dict) -> Dict:
        """
        Get detailed information about a specific HuggingFace model.
        
        Parameters:
        - model_id (str): The HuggingFace model ID (e.g., "bert-base-uncased")
        """
        try:
            model_id = params.get("model_id")
            if not model_id:
                raise JSONRPCError(-32602, "Invalid params", "model_id is required")
            
            # Get search service
            search_service = await get_hf_search_service()
            
            # Get model details
            model_details = await search_service.get_model_details(model_id)
            
            if not model_details:
                return {
                    "success": False,
                    "error": f"Model '{model_id}' not found",
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "success": True,
                "model": model_details,
                "timestamp": datetime.now().isoformat()
            }
            
        except JSONRPCError:
            raise
        except Exception as e:
            logger.error(f"Failed to get model details: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _get_model_search_suggestions(self, params: Dict) -> Dict:
        """
        Get search suggestions based on partial query.
        
        Parameters:
        - query (str): Partial search query
        - limit (int): Maximum suggestions to return (default: 10)
        """
        try:
            query = params.get("query", "")
            limit = params.get("limit", 10)
            
            if limit > 50:
                limit = 50
            
            # Get search service
            search_service = await get_hf_search_service()
            
            # Get suggestions
            suggestions = await search_service.get_search_suggestions(query, limit)
            
            return {
                "success": True,
                "suggestions": suggestions,
                "query": query,
                "count": len(suggestions),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return {
                "success": False,
                "error": str(e),
                "suggestions": [],
                "timestamp": datetime.now().isoformat()
            }

    async def _get_model_search_stats(self, params: Dict) -> Dict:
        """
        Get statistics about the model search index.
        """
        try:
            # Get search service
            search_service = await get_hf_search_service()
            
            # Get statistics
            stats = search_service.get_search_stats()
            
            return {
                "success": True,
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _initialize_model_search(self, params: Dict) -> Dict:
        """
        Initialize or refresh the model search index.
        
        Parameters:
        - force_rebuild (bool): Force rebuild of indices even if cache is valid
        """
        try:
            force_rebuild = params.get("force_rebuild", False)
            
            # Get search service
            search_service = await get_hf_search_service()
            
            if force_rebuild:
                # Force rebuild indices
                await search_service._rebuild_indices()
                message = "Model search indices rebuilt successfully"
            else:
                # Regular initialization
                success = await search_service.initialize()
                message = "Model search initialized successfully" if success else "Model search initialization failed"
            
            stats = search_service.get_search_stats()
            
            return {
                "success": True,
                "message": message,
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize model search: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # ===== MODEL DOWNLOAD AND MANAGEMENT METHODS =====

    async def _download_huggingface_model(self, params: Dict) -> Dict:
        """
        Download a HuggingFace model to local storage.
        
        Parameters:
        - model_id (str): HuggingFace model ID (e.g., "microsoft/DialoGPT-medium")
        - download_type (str): "snapshot" (full model) or "files" (specific files)
        - files (list): List of specific files to download (if download_type="files")
        - cache_dir (str): Optional custom cache directory
        """
        try:
            model_id = params.get("model_id")
            if not model_id:
                return {
                    "success": False,
                    "error": "model_id parameter is required",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Check if model is already downloaded
            if self.download_manager.is_model_downloaded(model_id):
                model_info = self.download_manager.get_downloaded_model_info(model_id)
                return {
                    "success": True,
                    "already_downloaded": True,
                    "model_path": model_info["model_path"],
                    "download_id": None,
                    "message": f"Model {model_id} is already downloaded",
                    "timestamp": datetime.now().isoformat()
                }
            
            download_type = params.get("download_type", "snapshot")
            files = params.get("files", [])
            cache_dir = params.get("cache_dir", str(self.models_dir))
            
            # Start download tracking
            download_id = self.download_manager.start_download(
                model_id=model_id,
                model_name=model_id.split("/")[-1],
                download_type=download_type
            )
            
            # Start download in background thread
            def _download_model():
                try:
                    self.download_manager.update_download_progress(download_id, 10)
                    
                    if not HAVE_HF_HUB:
                        raise Exception("HuggingFace Hub not available. Please install with: pip install huggingface_hub")
                    
                    self.download_manager.update_download_progress(download_id, 25)
                    
                    if download_type == "snapshot":
                        # Download entire model
                        model_path = snapshot_download(
                            repo_id=model_id,
                            cache_dir=cache_dir,
                            local_dir=os.path.join(cache_dir, model_id.replace("/", "_"))
                        )
                    else:
                        # Download specific files
                        model_dir = os.path.join(cache_dir, model_id.replace("/", "_"))
                        os.makedirs(model_dir, exist_ok=True)
                        
                        if not files:
                            files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
                        
                        downloaded_files = []
                        for i, filename in enumerate(files):
                            try:
                                file_path = hf_hub_download(
                                    repo_id=model_id,
                                    filename=filename,
                                    cache_dir=cache_dir,
                                    local_dir=model_dir
                                )
                                downloaded_files.append(file_path)
                                progress = 25 + (70 * (i + 1) / len(files))
                                self.download_manager.update_download_progress(download_id, int(progress))
                            except Exception as e:
                                logger.warning(f"Could not download {filename}: {e}")
                        
                        model_path = model_dir
                    
                    self.download_manager.update_download_progress(download_id, 95)
                    
                    # Complete download
                    self.download_manager.complete_download(download_id, model_path)
                    logger.info(f"Successfully downloaded model {model_id} to {model_path}")
                    
                except Exception as e:
                    logger.error(f"Download failed for {model_id}: {e}")
                    self.download_manager.fail_download(download_id, str(e))
            
            # Start download thread
            thread = threading.Thread(target=_download_model)
            thread.daemon = True
            thread.start()
            
            return {
                "success": True,
                "download_id": download_id,
                "model_id": model_id,
                "download_type": download_type,
                "message": f"Download started for {model_id}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to start download: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _get_download_status(self, params: Dict) -> Dict:
        """
        Get the status of a model download.
        
        Parameters:
        - download_id (str): Download ID returned from download_huggingface_model
        """
        try:
            download_id = params.get("download_id")
            if not download_id:
                return {
                    "success": False,
                    "error": "download_id parameter is required",
                    "timestamp": datetime.now().isoformat()
                }
            
            status = self.download_manager.get_download_status(download_id)
            if not status:
                return {
                    "success": False,
                    "error": f"Download ID {download_id} not found",
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "success": True,
                "download_status": status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get download status: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _list_downloaded_models(self, params: Dict) -> Dict:
        """
        List all downloaded models.
        
        Parameters:
        - include_details (bool): Include detailed information about each model
        """
        try:
            include_details = params.get("include_details", True)
            
            models = self.download_manager.list_downloaded_models()
            
            if include_details:
                # Add additional details for each model
                for model in models:
                    model_path = Path(model["model_path"])
                    if model_path.exists():
                        # Calculate actual disk usage
                        total_size = 0
                        if model_path.is_dir():
                            for file_path in model_path.rglob("*"):
                                if file_path.is_file():
                                    total_size += file_path.stat().st_size
                        else:
                            total_size = model_path.stat().st_size
                        
                        model["actual_size_mb"] = total_size / (1024 * 1024)
                        model["exists"] = True
                    else:
                        model["exists"] = False
                        model["actual_size_mb"] = 0
            
            return {
                "success": True,
                "models": models,
                "total_models": len(models),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to list downloaded models: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _remove_downloaded_model(self, params: Dict) -> Dict:
        """
        Remove a downloaded model from local storage.
        
        Parameters:
        - model_id (str): Model ID to remove
        - confirm (bool): Confirmation flag (required to prevent accidental deletion)
        """
        try:
            model_id = params.get("model_id")
            confirm = params.get("confirm", False)
            
            if not model_id:
                return {
                    "success": False,
                    "error": "model_id parameter is required",
                    "timestamp": datetime.now().isoformat()
                }
            
            if not confirm:
                return {
                    "success": False,
                    "error": "confirm parameter must be set to true to remove model",
                    "timestamp": datetime.now().isoformat()
                }
            
            if not self.download_manager.is_model_downloaded(model_id):
                return {
                    "success": False,
                    "error": f"Model {model_id} is not downloaded",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Remove the model
            removed = self.download_manager.remove_downloaded_model(model_id)
            
            if removed:
                return {
                    "success": True,
                    "message": f"Model {model_id} removed successfully",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to remove model {model_id}",
                    "timestamp": datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Failed to remove model: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _test_model_inference(self, params: Dict) -> Dict:
        """
        Test inference with a downloaded model.
        
        Parameters:
        - model_id (str): Model ID to test
        - input_text (str): Input text for testing
        - task (str): Task type ("text-generation", "text-classification", etc.)
        - max_length (int): Maximum output length (default: 100)
        - temperature (float): Sampling temperature (default: 0.7)
        """
        try:
            model_id = params.get("model_id")
            input_text = params.get("input_text", "Hello, how are you?")
            task = params.get("task", "text-generation")
            max_length = params.get("max_length", 100)
            temperature = params.get("temperature", 0.7)
            
            if not model_id:
                return {
                    "success": False,
                    "error": "model_id parameter is required",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Check if model is downloaded
            if not self.download_manager.is_model_downloaded(model_id):
                return {
                    "success": False,
                    "error": f"Model {model_id} is not downloaded. Download it first.",
                    "timestamp": datetime.now().isoformat()
                }
            
            model_info = self.download_manager.get_downloaded_model_info(model_id)
            model_path = model_info["model_path"]
            
            # Try to use ipfs_accelerate_py for inference
            if HAVE_MODEL_MANAGER:
                try:
                    # Load model using ipfs_accelerate_py
                    model_manager = ModelManager()
                    
                    # For now, return a simulated inference result
                    # In a real implementation, this would load and run the model
                    result = {
                        "model_id": model_id,
                        "input_text": input_text,
                        "task": task,
                        "output": f"[Generated response from {model_id}] This is a simulated inference result for input: '{input_text}'. In a real implementation, this would be the actual model output.",
                        "confidence": 0.85,
                        "processing_time_ms": 250,
                        "model_path": model_path,
                        "inference_engine": "ipfs_accelerate_py"
                    }
                    
                    return {
                        "success": True,
                        "inference_result": result,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    logger.warning(f"ipfs_accelerate_py inference failed, using fallback: {e}")
            
            # Fallback: simulated inference
            result = {
                "model_id": model_id,
                "input_text": input_text,
                "task": task,
                "output": f"[Simulated inference] Response from {model_id}: {input_text} -> Generated output (fallback mode)",
                "confidence": 0.75,
                "processing_time_ms": 150,
                "model_path": model_path,
                "inference_engine": "simulated"
            }
            
            return {
                "success": True,
                "inference_result": result,
                "note": "This is a simulated inference result. Full inference requires additional setup.",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to test model inference: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _get_model_download_info(self, params: Dict) -> Dict:
        """
        Get download information for a specific model.
        
        Parameters:
        - model_id (str): Model ID to get download info for
        """
        try:
            model_id = params.get("model_id")
            if not model_id:
                return {
                    "success": False,
                    "error": "model_id parameter is required",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Check if model is downloaded
            is_downloaded = self.download_manager.is_model_downloaded(model_id)
            
            result = {
                "model_id": model_id,
                "is_downloaded": is_downloaded,
                "can_download": HAVE_HF_HUB
            }
            
            if is_downloaded:
                model_info = self.download_manager.get_downloaded_model_info(model_id)
                result.update(model_info)
            
            # Check for active downloads
            active_downloads = []
            for download_id, download_info in self.download_manager.downloads.items():
                if (download_info["model_id"] == model_id and 
                    download_info["status"] in ["starting", "downloading", "completing"]):
                    active_downloads.append({
                        "download_id": download_id,
                        "status": download_info["status"],
                        "progress": download_info["progress"]
                    })
            
            result["active_downloads"] = active_downloads
            
            return {
                "success": True,
                "download_info": result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get model download info: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _call_ipfs_accelerate_model(self, task: str, params: Dict) -> Optional[Dict]:
        """Call ipfs_accelerate_py model for actual inference."""
        try:
            # This would integrate with the actual ipfs_accelerate_py inference engine
            # For now, we return None to use fallback implementations
            return None
        except Exception as e:
            logger.error(f"ipfs_accelerate_py model call failed: {e}")
            return None

def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    server = MCPJSONRPCServer()
    return server.app

def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the JSON-RPC server."""
    app = create_app()
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP JSON-RPC Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting MCP JSON-RPC server on {args.host}:{args.port}")
    run_server(host=args.host, port=args.port)