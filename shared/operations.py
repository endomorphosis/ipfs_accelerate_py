"""
Shared operations for IPFS Accelerate CLI and MCP server.

This module provides specific operation implementations that can be used
by both the CLI and MCP server interfaces.
"""

import logging
import os
import json
import time
from typing import Dict, Any, List, Optional, Union
from .core import SharedCore

logger = logging.getLogger(__name__)

class InferenceOperations:
    """Inference-related operations"""
    
    def __init__(self, shared_core: SharedCore):
        self.core = shared_core
        
    def run_text_generation(
        self,
        model: str,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Run text generation inference"""
        
        if not self.core.validate_model_id(model):
            return {"error": "Invalid model ID", "model": model}
        
        if not prompt or not isinstance(prompt, str):
            return {"error": "Prompt is required and must be a string"}
        
        # Use the safe_call method to call the inference
        result = self.core.safe_call(
            "run_inference",
            model=model,
            inputs=[prompt],
            max_length=max_length,
            temperature=temperature,
            **kwargs
        )
        
        # Add operation-specific metadata
        result.update({
            "operation": "text_generation",
            "model": model,
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature
            }
        })
        
        return result
    
    def run_text_classification(
        self,
        model: str,
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Run text classification inference"""
        
        if not self.core.validate_model_id(model):
            return {"error": "Invalid model ID", "model": model}
        
        if not text or not isinstance(text, str):
            return {"error": "Text is required and must be a string"}
        
        result = self.core.safe_call(
            "run_inference",
            model=model,
            inputs=[text],
            task="text-classification",
            **kwargs
        )
        
        result.update({
            "operation": "text_classification",
            "model": model,
            "text": text[:100] + "..." if len(text) > 100 else text
        })
        
        return result
    
    def get_supported_tasks(self) -> Dict[str, Any]:
        """Get list of supported inference tasks"""
        return {
            "tasks": [
                "text-generation",
                "text-classification", 
                "text2text-generation",
                "feature-extraction",
                "question-answering",
                "summarization",
                "translation"
            ],
            "operation": "get_supported_tasks",
            "success": True
        }

class FileOperations:
    """File-related operations"""
    
    def __init__(self, shared_core: SharedCore):
        self.core = shared_core
        
    def add_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Add file to IPFS"""
        
        if not self.core.validate_file_path(file_path):
            return {"error": "Invalid file path or file does not exist", "file_path": file_path}
        
        # Get file info
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)
        
        result = self.core.safe_call("add_file", file_path, **kwargs)
        
        result.update({
            "operation": "add_file",
            "file_path": file_path,
            "file_name": file_name,
            "file_size": file_size
        })
        
        return result
    
    def get_file(self, cid: str, output_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Get file from IPFS by CID"""
        
        if not cid or not isinstance(cid, str):
            return {"error": "CID is required and must be a string"}
        
        result = self.core.safe_call("get_file", cid, output_path, **kwargs)
        
        result.update({
            "operation": "get_file",
            "cid": cid,
            "output_path": output_path
        })
        
        return result
    
    def list_files(self, **kwargs) -> Dict[str, Any]:
        """List files in IPFS"""
        
        result = self.core.safe_call("list_files", **kwargs)
        
        result.update({
            "operation": "list_files"
        })
        
        return result
    
    def pin_file(self, cid: str, **kwargs) -> Dict[str, Any]:
        """Pin file in IPFS"""
        
        if not cid or not isinstance(cid, str):
            return {"error": "CID is required and must be a string"}
        
        result = self.core.safe_call("pin_file", cid, **kwargs)
        
        result.update({
            "operation": "pin_file",
            "cid": cid
        })
        
        return result

class ModelOperations:
    """Model-related operations"""
    
    def __init__(self, shared_core: SharedCore):
        self.core = shared_core
        
    def list_models(self, **kwargs) -> Dict[str, Any]:
        """List available models"""
        
        result = self.core.safe_call("list_models", **kwargs)
        
        # If the core method doesn't exist, provide a fallback
        if result.get("error") and "not available" in result["error"]:
            result = {
                "models": [
                    {"id": "gpt2", "type": "text-generation", "size": "small"},
                    {"id": "bert-base-uncased", "type": "text-classification", "size": "base"},
                    {"id": "distilbert-base-uncased", "type": "text-classification", "size": "small"},
                    {"id": "t5-small", "type": "text2text-generation", "size": "small"},
                    {"id": "sentence-transformers/all-MiniLM-L6-v2", "type": "feature-extraction", "size": "small"}
                ],
                "count": 5,
                "fallback": True,
                "success": True
            }
        
        result.update({
            "operation": "list_models"
        })
        
        return result
    
    def get_model_info(self, model_id: str, **kwargs) -> Dict[str, Any]:
        """Get information about a specific model"""
        
        if not self.core.validate_model_id(model_id):
            return {"error": "Invalid model ID", "model_id": model_id}
        
        result = self.core.safe_call("get_model_info", model_id, **kwargs)
        
        result.update({
            "operation": "get_model_info",
            "model_id": model_id
        })
        
        return result
    
    def download_model(self, model_id: str, **kwargs) -> Dict[str, Any]:
        """Download a model"""
        
        if not self.core.validate_model_id(model_id):
            return {"error": "Invalid model ID", "model_id": model_id}
        
        result = self.core.safe_call("download_model", model_id, **kwargs)
        
        result.update({
            "operation": "download_model",
            "model_id": model_id
        })
        
        return result

class NetworkOperations:
    """Network-related operations"""
    
    def __init__(self, shared_core: SharedCore):
        self.core = shared_core
        
    def get_network_status(self, **kwargs) -> Dict[str, Any]:
        """Get network status"""
        
        result = self.core.safe_call("get_network_status", **kwargs)
        
        # If the core method doesn't exist, provide a fallback
        if result.get("error") and "not available" in result["error"]:
            result = {
                "status": "connected",
                "peers": 5,
                "bandwidth": {"in": 1024, "out": 512},
                "fallback": True,
                "success": True
            }
        
        result.update({
            "operation": "get_network_status"
        })
        
        return result
    
    def get_peers(self, **kwargs) -> Dict[str, Any]:
        """Get list of connected peers"""
        
        result = self.core.safe_call("get_peers", **kwargs)
        
        result.update({
            "operation": "get_peers"
        })
        
        return result
    
    def connect_peer(self, peer_id: str, **kwargs) -> Dict[str, Any]:
        """Connect to a peer"""
        
        if not peer_id or not isinstance(peer_id, str):
            return {"error": "Peer ID is required and must be a string"}
        
        result = self.core.safe_call("connect_peer", peer_id, **kwargs)
        
        result.update({
            "operation": "connect_peer",
            "peer_id": peer_id
        })
        
        return result