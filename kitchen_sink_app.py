#!/usr/bin/env python3
"""
Kitchen Sink AI Model Testing Interface

A comprehensive web-based testing interface for AI model inference
with model selection via autocomplete from the model manager.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# Add the parent directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from ipfs_accelerate_py.model_manager import (
        ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
        VectorDocumentationIndex, BanditModelRecommender,
        RecommendationContext, create_model_from_huggingface
    )
    from ipfs_mcp.ai_model_server import create_ai_model_server
    HAVE_MODEL_MANAGER = True
except ImportError as e:
    HAVE_MODEL_MANAGER = False
    print(f"⚠️ Model Manager not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KitchenSinkApp:
    """Kitchen Sink AI Model Testing Application."""
    
    def __init__(self):
        """Initialize the application."""
        self.app = Flask(__name__)
        CORS(self.app)
        self.app.config['SECRET_KEY'] = 'kitchen-sink-testing-2025'
        
        # Initialize AI components if available
        if HAVE_MODEL_MANAGER:
            self.model_manager = ModelManager(storage_path="./kitchen_sink_models.db")
            self.bandit_recommender = BanditModelRecommender(
                model_manager=self.model_manager,
                storage_path="./kitchen_sink_bandit.json"
            )
            self.ai_server = create_ai_model_server(
                model_manager_path="./kitchen_sink_models.db",
                bandit_storage_path="./kitchen_sink_bandit.json"
            )
            logger.info("AI components initialized")
        else:
            self.model_manager = None
            self.bandit_recommender = None
            self.ai_server = None
            logger.warning("AI components not available - running in demo mode")
        
        # Setup routes
        self._setup_routes()
        
        # Initialize with some sample models for testing
        self._initialize_sample_models()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main testing interface."""
            return render_template('index.html')
        
        @self.app.route('/static/<path:filename>')
        def static_files(filename):
            """Serve static files."""
            return send_from_directory('static', filename)
        
        # API Routes
        @self.app.route('/api/models')
        def list_models():
            """List all available models."""
            try:
                if not self.model_manager:
                    return jsonify({"models": self._get_demo_models()})
                
                models = self.model_manager.list_models()
                model_list = []
                
                for model in models:
                    model_dict = {
                        "model_id": model.model_id,
                        "model_name": model.model_name,
                        "model_type": model.model_type.value if hasattr(model.model_type, 'value') else str(model.model_type),
                        "architecture": model.architecture,
                        "description": model.description,
                        "tags": model.tags,
                        "created_at": model.created_at.isoformat() if model.created_at else None
                    }
                    model_list.append(model_dict)
                
                return jsonify({"models": model_list})
                
            except Exception as e:
                logger.error(f"Error listing models: {e}")
                return jsonify({"error": str(e), "models": []}), 500
        
        @self.app.route('/api/models/search')
        def search_models():
            """Search models for autocomplete."""
            query = request.args.get('q', '').lower()
            limit = int(request.args.get('limit', 10))
            
            try:
                if not self.model_manager:
                    models = self._get_demo_models()
                else:
                    models_obj = self.model_manager.list_models()
                    models = []
                    for model in models_obj:
                        models.append({
                            "model_id": model.model_id,
                            "model_name": model.model_name,
                            "architecture": model.architecture,
                            "description": model.description
                        })
                
                # Filter models based on query
                filtered_models = []
                for model in models:
                    if (query in model["model_id"].lower() or 
                        query in model["model_name"].lower() or
                        query in model["architecture"].lower()):
                        filtered_models.append(model)
                
                # Limit results
                filtered_models = filtered_models[:limit]
                
                return jsonify({"models": filtered_models})
                
            except Exception as e:
                logger.error(f"Error searching models: {e}")
                return jsonify({"error": str(e), "models": []}), 500
        
        @self.app.route('/api/models/<model_id>')
        def get_model_info(model_id):
            """Get detailed information about a specific model."""
            try:
                if not self.model_manager:
                    # Return demo data
                    demo_models = self._get_demo_models()
                    for model in demo_models:
                        if model["model_id"] == model_id:
                            return jsonify(model)
                    return jsonify({"error": "Model not found"}), 404
                
                model = self.model_manager.get_model(model_id)
                if not model:
                    return jsonify({"error": "Model not found"}), 404
                
                model_dict = {
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "model_type": model.model_type.value if hasattr(model.model_type, 'value') else str(model.model_type),
                    "architecture": model.architecture,
                    "description": model.description,
                    "tags": model.tags,
                    "input_spec": {
                        "data_type": model.input_spec.data_type.value if hasattr(model.input_spec.data_type, 'value') else str(model.input_spec.data_type),
                        "shape": model.input_spec.shape,
                        "description": model.input_spec.description
                    } if model.input_spec else None,
                    "output_spec": {
                        "data_type": model.output_spec.data_type.value if hasattr(model.output_spec.data_type, 'value') else str(model.output_spec.data_type),
                        "shape": model.output_spec.shape,
                        "description": model.output_spec.description
                    } if model.output_spec else None,
                    "created_at": model.created_at.isoformat() if model.created_at else None
                }
                
                return jsonify(model_dict)
                
            except Exception as e:
                logger.error(f"Error getting model info: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/recommend', methods=['POST'])
        def recommend_model():
            """Get model recommendations."""
            try:
                data = request.get_json()
                task_type = data.get('task_type', 'generation')
                hardware = data.get('hardware', 'cpu')
                input_type = data.get('input_type', 'tokens')
                output_type = data.get('output_type', 'tokens')
                
                if not self.bandit_recommender:
                    # Return demo recommendation
                    demo_models = self._get_demo_models()
                    if demo_models:
                        return jsonify({
                            "model_id": demo_models[0]["model_id"],
                            "confidence_score": 0.85,
                            "predicted_performance": 0.92,
                            "reasoning": "Demo recommendation - install Model Manager for real recommendations"
                        })
                    return jsonify({"error": "No models available"}), 404
                
                # Create recommendation context
                try:
                    context = RecommendationContext(
                        task_type=task_type,
                        hardware=hardware,
                        input_type=DataType(input_type),
                        output_type=DataType(output_type),
                        performance_requirements=data.get('requirements', {})
                    )
                except:
                    # Fallback if DataType enum doesn't work
                    context = RecommendationContext(
                        task_type=task_type,
                        hardware=hardware,
                        input_type=input_type,
                        output_type=output_type,
                        performance_requirements=data.get('requirements', {})
                    )
                
                recommendation = self.bandit_recommender.recommend_model(context)
                
                if recommendation:
                    return jsonify({
                        "model_id": recommendation.model_id,
                        "confidence_score": recommendation.confidence_score,
                        "predicted_performance": recommendation.predicted_performance,
                        "reasoning": recommendation.reasoning
                    })
                else:
                    return jsonify({"error": "No suitable model found"}), 404
                    
            except Exception as e:
                logger.error(f"Error getting recommendation: {e}")
                return jsonify({"error": str(e)}), 500
        
        # Inference endpoints
        @self.app.route('/api/inference/generate', methods=['POST'])
        def generate_text():
            """Generate text using causal language modeling."""
            try:
                data = request.get_json()
                prompt = data.get('prompt', '')
                model_id = data.get('model_id')
                max_length = data.get('max_length', 100)
                temperature = data.get('temperature', 0.7)
                hardware = data.get('hardware', 'cpu')
                
                # Mock implementation for demo
                start_time = datetime.now()
                
                if model_id:
                    generated_text = f"[Generated by {model_id}] This is a continuation of: {prompt}..."
                else:
                    generated_text = f"[Auto-selected model] This is a continuation of: {prompt}..."
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                return jsonify({
                    "generated_text": generated_text,
                    "model_used": model_id or "auto-selected",
                    "processing_time": processing_time,
                    "parameters": {
                        "max_length": max_length,
                        "temperature": temperature,
                        "hardware": hardware
                    },
                    "token_count": len(generated_text.split())
                })
                
            except Exception as e:
                logger.error(f"Error in text generation: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/inference/classify', methods=['POST'])
        def classify_text():
            """Classify text."""
            try:
                data = request.get_json()
                text = data.get('text', '')
                model_id = data.get('model_id')
                hardware = data.get('hardware', 'cpu')
                
                start_time = datetime.now()
                
                # Mock classification result
                classes = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
                prediction = "POSITIVE" if "good" in text.lower() or "great" in text.lower() else "NEUTRAL"
                confidence = 0.85 if prediction == "POSITIVE" else 0.72
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                return jsonify({
                    "prediction": prediction,
                    "confidence": confidence,
                    "all_scores": {
                        "POSITIVE": 0.85 if prediction == "POSITIVE" else 0.15,
                        "NEGATIVE": 0.10,
                        "NEUTRAL": 0.72 if prediction == "NEUTRAL" else 0.05
                    },
                    "model_used": model_id or "auto-selected",
                    "processing_time": processing_time
                })
                
            except Exception as e:
                logger.error(f"Error in text classification: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/inference/embed', methods=['POST'])
        def generate_embeddings():
            """Generate text embeddings."""
            try:
                data = request.get_json()
                text = data.get('text', '')
                model_id = data.get('model_id')
                normalize = data.get('normalize', True)
                
                start_time = datetime.now()
                
                # Mock embedding generation
                import hashlib
                hash_obj = hashlib.md5(text.encode())
                hash_hex = hash_obj.hexdigest()
                
                # Convert to pseudo-embedding (16 dimensions for demo)
                embedding = []
                for i in range(0, 32, 2):
                    byte_val = int(hash_hex[i:i+2], 16)
                    embedding.append((byte_val - 128) / 128.0)  # Normalize to [-1, 1]
                
                end_time = datetime.now()
                processing_time = (end_time - start_time).total_seconds()
                
                return jsonify({
                    "embedding": embedding,
                    "dimensions": len(embedding),
                    "normalized": normalize,
                    "model_used": model_id or "auto-selected",
                    "processing_time": processing_time
                })
                
            except Exception as e:
                logger.error(f"Error in embedding generation: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/feedback', methods=['POST'])
        def submit_feedback():
            """Submit feedback for model performance."""
            try:
                data = request.get_json()
                model_id = data.get('model_id')
                task_type = data.get('task_type')
                score = data.get('score', 0.5)
                
                if self.bandit_recommender:
                    # Submit real feedback
                    context = RecommendationContext(
                        task_type=task_type,
                        hardware=data.get('hardware', 'cpu'),
                        input_type=data.get('input_type', 'tokens'),
                        output_type=data.get('output_type', 'tokens')
                    )
                    self.bandit_recommender.provide_feedback(model_id, score, context)
                
                return jsonify({"success": True, "message": "Feedback recorded"})
                
            except Exception as e:
                logger.error(f"Error submitting feedback: {e}")
                return jsonify({"error": str(e)}), 500
    
    def _get_demo_models(self):
        """Get demo models for when Model Manager is not available."""
        return [
            {
                "model_id": "gpt2",
                "model_name": "GPT-2",
                "model_type": "language_model",
                "architecture": "transformer",
                "description": "Small GPT-2 model for text generation",
                "tags": ["generation", "transformer", "openai"]
            },
            {
                "model_id": "bert-base-uncased",
                "model_name": "BERT Base Uncased",
                "model_type": "language_model", 
                "architecture": "bert",
                "description": "BERT model for masked language modeling and classification",
                "tags": ["classification", "bert", "google"]
            },
            {
                "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "model_name": "All MiniLM L6 v2",
                "model_type": "embedding_model",
                "architecture": "sentence-transformer",
                "description": "Sentence embedding model",
                "tags": ["embeddings", "sentence-transformers"]
            }
        ]
    
    def _initialize_sample_models(self):
        """Initialize some sample models for testing."""
        if not self.model_manager:
            return
            
        try:
            # Check if we already have models
            existing_models = self.model_manager.list_models()
            if existing_models:
                logger.info(f"Found {len(existing_models)} existing models")
                return
            
            # Add some sample models
            sample_models = [
                {
                    "model_id": "gpt2",
                    "model_name": "GPT-2",
                    "model_type": ModelType.LANGUAGE_MODEL,
                    "architecture": "transformer",
                    "description": "Small GPT-2 model for text generation",
                    "tags": ["generation", "transformer", "openai"],
                    "inputs": [IOSpec(name="input_ids", data_type=DataType.TOKENS, shape=(-1,), description="Input tokens")],
                    "outputs": [IOSpec(name="logits", data_type=DataType.LOGITS, shape=(-1, 50257), description="Token logits")]
                },
                {
                    "model_id": "bert-base-uncased", 
                    "model_name": "BERT Base Uncased",
                    "model_type": ModelType.LANGUAGE_MODEL,
                    "architecture": "bert",
                    "description": "BERT model for masked language modeling and classification",
                    "tags": ["classification", "bert", "google"],
                    "inputs": [IOSpec(name="input_ids", data_type=DataType.TOKENS, shape=(-1,), description="Input tokens")],
                    "outputs": [IOSpec(name="hidden_states", data_type=DataType.LOGITS, shape=(-1, 768), description="Hidden states")]
                }
            ]
            
            for model_data in sample_models:
                model = ModelMetadata(**model_data)
                self.model_manager.add_model(model)
                logger.info(f"Added sample model: {model.model_id}")
                
        except Exception as e:
            logger.warning(f"Could not initialize sample models: {e}")
            # Add basic demo models without complex specs
            try:
                basic_models = [
                    {
                        "model_id": "gpt2",
                        "model_name": "GPT-2",
                        "model_type": ModelType.LANGUAGE_MODEL,
                        "architecture": "transformer",
                        "description": "Small GPT-2 model for text generation",
                        "tags": ["generation", "transformer", "openai"],
                        "inputs": [IOSpec(name="input_ids", data_type=DataType.TOKENS, shape=(-1,), description="Input tokens")],
                        "outputs": [IOSpec(name="logits", data_type=DataType.LOGITS, shape=(-1, 50257), description="Token logits")]
                    },
                    {
                        "model_id": "bert-base-uncased", 
                        "model_name": "BERT Base Uncased",
                        "model_type": ModelType.LANGUAGE_MODEL,
                        "architecture": "bert",
                        "description": "BERT model for masked language modeling and classification", 
                        "tags": ["classification", "bert", "google"],
                        "inputs": [IOSpec(name="input_ids", data_type=DataType.TOKENS, shape=(-1,), description="Input tokens")],
                        "outputs": [IOSpec(name="hidden_states", data_type=DataType.LOGITS, shape=(-1, 768), description="Hidden states")]
                    }
                ]
                
                for model_data in basic_models:
                    model = ModelMetadata(**model_data)
                    self.model_manager.add_model(model)
                    logger.info(f"Added basic sample model: {model.model_id}")
                    
            except Exception as e2:
                logger.error(f"Could not add even basic models: {e2}")
                # Fall back to demo mode completely
    
    def run(self, host='127.0.0.1', port=5000, debug=True):
        """Run the Flask application."""
        logger.info(f"Starting Kitchen Sink Testing Interface on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# Flask app instance for external access
app = None

def create_app():
    """Create the Flask application."""
    global app
    kitchen_sink = KitchenSinkApp()
    app = kitchen_sink.app
    return kitchen_sink

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kitchen Sink AI Model Testing Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    kitchen_sink = create_app()
    kitchen_sink.run(host=args.host, port=args.port, debug=args.debug)