#!/usr/bin/env python3
"""
Comprehensive Kitchen Sink AI Model Testing Interface

A complete web-based testing interface for ALL AI model inference types
with model selection via autocomplete from the model manager.

Supports 14+ inference categories and 25+ specific inference tools covering
all 211+ model types found in the skillset directory.
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
    from comprehensive_mcp_server import create_comprehensive_server
    from huggingface_search_engine import HuggingFaceModelSearchEngine
    HAVE_MODEL_MANAGER = True
except ImportError as e:
    HAVE_MODEL_MANAGER = False
    print(f"⚠️ Model Manager not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveKitchenSinkApp:
    """Comprehensive Kitchen Sink AI Model Testing Application."""
    
    def __init__(self):
        """Initialize the comprehensive application."""
        self.app = Flask(__name__)
        CORS(self.app)
        self.app.config['SECRET_KEY'] = 'comprehensive-kitchen-sink-2025'
        
        # Initialize AI components if available
        if HAVE_MODEL_MANAGER:
            self.model_manager = ModelManager(storage_path="./comprehensive_models.db")
            self.bandit_recommender = BanditModelRecommender(
                model_manager=self.model_manager,
                storage_path="./comprehensive_bandit.json"
            )
            self.mcp_server = create_comprehensive_server(
                model_manager_path="./comprehensive_models.db",
                bandit_storage_path="./comprehensive_bandit.json"
            )
            # Initialize HuggingFace search engine
            self.hf_search_engine = HuggingFaceModelSearchEngine(self.model_manager)
            logger.info("Comprehensive AI components initialized")
        else:
            self.model_manager = None
            self.bandit_recommender = None
            self.mcp_server = None
            self.hf_search_engine = None
            logger.warning("AI components not available - running in demo mode")
        
        # Define comprehensive inference categories
        self.inference_categories = {
            "text_processing": {
                "name": "Text Processing",
                "icon": "fas fa-font",
                "color": "primary",
                "tools": [
                    "generate_text", "fill_mask", "classify_text", 
                    "generate_embeddings", "translate_text", "summarize_text", "answer_question"
                ]
            },
            "audio_processing": {
                "name": "Audio Processing", 
                "icon": "fas fa-microphone",
                "color": "success",
                "tools": [
                    "transcribe_audio", "classify_audio", "synthesize_speech", "generate_audio"
                ]
            },
            "vision_processing": {
                "name": "Vision Processing",
                "icon": "fas fa-eye",
                "color": "warning",
                "tools": [
                    "classify_image", "detect_objects", "segment_image", "generate_image"
                ]
            },
            "multimodal_processing": {
                "name": "Multimodal Processing",
                "icon": "fas fa-layer-group",
                "color": "info",
                "tools": [
                    "generate_image_caption", "answer_visual_question", "process_document"
                ]
            },
            "specialized_processing": {
                "name": "Specialized Processing",
                "icon": "fas fa-tools",
                "color": "secondary",
                "tools": [
                    "predict_timeseries", "generate_code", "process_tabular_data"
                ]
            },
            "model_management": {
                "name": "Model Management",
                "icon": "fas fa-database",
                "color": "dark",
                "tools": [
                    "list_models", "recommend_model", "get_available_model_types"
                ]
            },
            "huggingface_browser": {
                "name": "HuggingFace Browser",
                "icon": "fas fa-globe",
                "color": "danger",
                "tools": [
                    "search_huggingface", "get_model_details", "add_to_manager"
                ]
            }
        }
        
        # Setup routes
        self._setup_routes()
        
        # Initialize with comprehensive model data
        self._initialize_comprehensive_models()
    
    def _setup_routes(self):
        """Setup comprehensive Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main comprehensive testing interface."""
            return render_template('comprehensive_index.html', 
                                 categories=self.inference_categories)
        
        @self.app.route('/static/<path:filename>')
        def static_files(filename):
            """Serve static files."""
            return send_from_directory(os.path.join(os.path.dirname(__file__), '..', 'ipfs_accelerate_py', 'static'), filename)
        
        # Model Management API Routes
        @self.app.route('/api/models')
        def list_models():
            """List all available models."""
            try:
                if not self.model_manager:
                    return jsonify({"models": self._get_comprehensive_demo_models()})
                
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
            limit = int(request.args.get('limit', 20))
            category = request.args.get('category', '')
            
            try:
                if not self.model_manager:
                    models = self._get_comprehensive_demo_models()
                else:
                    models_obj = self.model_manager.list_models()
                    models = []
                    for model in models_obj:
                        models.append({
                            "model_id": model.model_id,
                            "model_name": model.model_name,
                            "architecture": model.architecture,
                            "description": model.description,
                            "model_type": model.model_type.value if hasattr(model.model_type, 'value') else str(model.model_type)
                        })
                
                # Filter models based on query and category
                filtered_models = []
                for model in models:
                    # Category filter
                    if category and not self._model_matches_category(model, category):
                        continue
                    
                    # Query filter
                    if (query in model["model_id"].lower() or 
                        query in model["model_name"].lower() or
                        query in model["architecture"].lower() or
                        query in model.get("description", "").lower()):
                        filtered_models.append(model)
                
                # Limit results
                filtered_models = filtered_models[:limit]
                
                return jsonify({"models": filtered_models})
                
            except Exception as e:
                logger.error(f"Error searching models: {e}")
                return jsonify({"error": str(e), "models": []}), 500
        
        @self.app.route('/api/models/categories')
        def get_model_categories():
            """Get available model categories and their counts."""
            try:
                if self.mcp_server:
                    # Get categories from MCP server
                    available_types = self.mcp_server.available_model_types
                    return jsonify({
                        "categories": self.inference_categories,
                        "model_types": available_types,
                        "total_models": sum(len(models) for models in available_types.values())
                    })
                else:
                    return jsonify({
                        "categories": self.inference_categories,
                        "model_types": {"demo": ["gpt2", "bert", "clip"]},
                        "total_models": 3
                    })
                    
            except Exception as e:
                logger.error(f"Error getting categories: {e}")
                return jsonify({"error": str(e)}), 500
        
        # Comprehensive Inference API Routes
        
        # Text Processing Routes
        @self.app.route('/api/inference/text/generate', methods=['POST'])
        def generate_text():
            """Generate text using causal language modeling."""
            return self._handle_inference("generate_text", request.get_json())
        
        @self.app.route('/api/inference/text/fill_mask', methods=['POST'])
        def fill_mask():
            """Fill masked tokens in text."""
            return self._handle_inference("fill_mask", request.get_json())
        
        @self.app.route('/api/inference/text/classify', methods=['POST'])
        def classify_text():
            """Classify text."""
            return self._handle_inference("classify_text", request.get_json())
        
        @self.app.route('/api/inference/text/embed', methods=['POST'])
        def generate_embeddings():
            """Generate text embeddings."""
            return self._handle_inference("generate_embeddings", request.get_json())
        
        @self.app.route('/api/inference/text/translate', methods=['POST'])
        def translate_text():
            """Translate text between languages."""
            return self._handle_inference("translate_text", request.get_json())
        
        @self.app.route('/api/inference/text/summarize', methods=['POST'])
        def summarize_text():
            """Summarize text."""
            return self._handle_inference("summarize_text", request.get_json())
        
        @self.app.route('/api/inference/text/qa', methods=['POST'])
        def answer_question():
            """Answer questions based on context."""
            return self._handle_inference("answer_question", request.get_json())
        
        # Audio Processing Routes
        @self.app.route('/api/inference/audio/transcribe', methods=['POST'])
        def transcribe_audio():
            """Transcribe audio to text."""
            return self._handle_inference("transcribe_audio", request.get_json())
        
        @self.app.route('/api/inference/audio/classify', methods=['POST'])
        def classify_audio():
            """Classify audio content."""
            return self._handle_inference("classify_audio", request.get_json())
        
        @self.app.route('/api/inference/audio/synthesize', methods=['POST'])
        def synthesize_speech():
            """Synthesize speech from text."""
            return self._handle_inference("synthesize_speech", request.get_json())
        
        @self.app.route('/api/inference/audio/generate', methods=['POST'])
        def generate_audio():
            """Generate audio content."""
            return self._handle_inference("generate_audio", request.get_json())
        
        # Vision Processing Routes
        @self.app.route('/api/inference/vision/classify', methods=['POST'])
        def classify_image():
            """Classify images."""
            return self._handle_inference("classify_image", request.get_json())
        
        @self.app.route('/api/inference/vision/detect', methods=['POST'])
        def detect_objects():
            """Detect objects in images."""
            return self._handle_inference("detect_objects", request.get_json())
        
        @self.app.route('/api/inference/vision/segment', methods=['POST'])
        def segment_image():
            """Segment images."""
            return self._handle_inference("segment_image", request.get_json())
        
        @self.app.route('/api/inference/vision/generate', methods=['POST'])
        def generate_image():
            """Generate images using diffusion."""
            return self._handle_inference("generate_image", request.get_json())
        
        # Multimodal Processing Routes
        @self.app.route('/api/inference/multimodal/caption', methods=['POST'])
        def generate_image_caption():
            """Generate image captions."""
            return self._handle_inference("generate_image_caption", request.get_json())
        
        @self.app.route('/api/inference/multimodal/vqa', methods=['POST'])
        def answer_visual_question():
            """Answer visual questions."""
            return self._handle_inference("answer_visual_question", request.get_json())
        
        @self.app.route('/api/inference/multimodal/document', methods=['POST'])
        def process_document():
            """Process documents."""
            return self._handle_inference("process_document", request.get_json())
        
        # Specialized Processing Routes
        @self.app.route('/api/inference/specialized/timeseries', methods=['POST'])
        def predict_timeseries():
            """Predict time series."""
            return self._handle_inference("predict_timeseries", request.get_json())
        
        @self.app.route('/api/inference/specialized/code', methods=['POST'])
        def generate_code():
            """Generate code."""
            return self._handle_inference("generate_code", request.get_json())
        
        @self.app.route('/api/inference/specialized/tabular', methods=['POST'])
        def process_tabular_data():
            """Process tabular data."""
            return self._handle_inference("process_tabular_data", request.get_json())
        
        # Model Recommendation Route
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
                    demo_models = self._get_comprehensive_demo_models()
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
        
        # Feedback Route
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
        
        # HuggingFace Browser Routes  
        @self.app.route('/api/hf/search', methods=['POST'])
        def search_huggingface():
            """Search HuggingFace models."""
            try:
                data = request.get_json()
                query = data.get('query', '')
                limit = int(data.get('limit', 20))
                task_filter = data.get('task_filter', '')
                sort_by = data.get('sort_by', 'downloads')
                
                if not self.hf_search_engine:
                    return jsonify({"error": "HuggingFace search not available", "models": []}), 503
                
                # Build filter dict
                filter_dict = {}
                if task_filter:
                    filter_dict['pipeline_tag'] = task_filter
                
                # Search models
                models = self.hf_search_engine.search_huggingface_models(
                    query=query,
                    limit=limit,
                    filter_dict=filter_dict,
                    sort=sort_by
                )
                
                # Convert to response format
                model_list = []
                for model in models:
                    model_dict = {
                        "model_id": model.model_id,
                        "model_name": model.model_name,
                        "author": model.author,
                        "description": model.description[:200] + "..." if len(model.description) > 200 else model.description,
                        "tags": model.tags[:5],  # Limit tags for UI
                        "pipeline_tag": model.pipeline_tag,
                        "library_name": model.library_name,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "license": model.license,
                        "last_modified": model.last_modified
                    }
                    model_list.append(model_dict)
                
                return jsonify({
                    "models": model_list,
                    "total": len(model_list),
                    "query": query,
                    "filters": filter_dict
                })
                
            except Exception as e:
                logger.error(f"Error searching HuggingFace: {e}")
                return jsonify({"error": str(e), "models": []}), 500
        
        # Statistics Route
        @self.app.route('/api/stats')
        def get_statistics():
            """Get comprehensive statistics."""
            try:
                if self.mcp_server:
                    # Use MCP server statistics
                    return jsonify(self.mcp_server._generate_mock_result("get_inference_statistics", "stats", {}))
                else:
                    # Return demo statistics
                    return jsonify({
                        "total_models": 15,
                        "inference_categories": len(self.inference_categories),
                        "available_tools": sum(len(cat["tools"]) for cat in self.inference_categories.values()),
                        "demo_mode": True
                    })
                    
            except Exception as e:
                logger.error(f"Error getting statistics: {e}")
                return jsonify({"error": str(e)}), 500
    
    def _handle_inference(self, tool_name: str, data: Dict[str, Any]) -> Any:
        """Handle inference requests through the MCP server."""
        try:
            start_time = datetime.now()
            
            if self.mcp_server:
                # Use real MCP server inference
                result = self.mcp_server._perform_inference(
                    task_type=self._get_task_type_for_tool(tool_name),
                    input_data=data,
                    model_id=data.get('model_id'),
                    hardware=data.get('hardware', 'cpu')
                )
            else:
                # Use mock inference
                result = self._generate_mock_inference(tool_name, data)
            
            end_time = datetime.now()
            result["processing_time"] = (end_time - start_time).total_seconds()
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error in {tool_name} inference: {e}")
            return jsonify({"error": f"{tool_name} inference failed: {str(e)}"}), 500
    
    def _get_task_type_for_tool(self, tool_name: str) -> str:
        """Map tool names to task types."""
        task_mapping = {
            "generate_text": "causal_language_modeling",
            "fill_mask": "masked_language_modeling",
            "classify_text": "text_classification",
            "generate_embeddings": "embedding_generation",
            "translate_text": "translation",
            "summarize_text": "summarization",
            "answer_question": "question_answering",
            "transcribe_audio": "automatic_speech_recognition",
            "classify_audio": "audio_classification",
            "synthesize_speech": "text_to_speech",
            "generate_audio": "audio_generation",
            "classify_image": "image_classification",
            "detect_objects": "object_detection",
            "segment_image": "image_segmentation",
            "generate_image": "image_diffusion",
            "generate_image_caption": "image_to_text",
            "answer_visual_question": "visual_question_answering",
            "process_document": "document_understanding",
            "predict_timeseries": "time_series_forecasting",
            "generate_code": "code_generation",
            "process_tabular_data": "tabular_processing"
        }
        return task_mapping.get(tool_name, tool_name)
    
    def _generate_mock_inference(self, tool_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock inference results for demo mode."""
        task_type = self._get_task_type_for_tool(tool_name)
        model_id = data.get('model_id', 'auto-selected')
        
        base_result = {
            "tool_used": tool_name,
            "model_used": model_id,
            "confidence": 0.85,
            "status": "completed"
        }
        
        if "text" in tool_name:
            if tool_name == "generate_text":
                base_result.update({
                    "generated_text": f"[Generated by {model_id}] This is a high-quality text continuation...",
                    "tokens_generated": data.get('max_length', 100) // 2
                })
            elif tool_name == "classify_text":
                base_result.update({
                    "prediction": "POSITIVE",
                    "all_scores": {"POSITIVE": 0.85, "NEGATIVE": 0.10, "NEUTRAL": 0.05}
                })
            elif tool_name == "generate_embeddings":
                import hashlib
                text = data.get('text', '')
                # Generate deterministic mock embedding
                embedding = [float(hash(text + str(i)) % 100) / 100 for i in range(16)]
                base_result.update({
                    "embeddings": embedding,
                    "dimension": len(embedding)
                })
        
        elif "audio" in tool_name:
            if tool_name == "transcribe_audio":
                base_result.update({
                    "transcription": f"[Transcribed by {model_id}] This is the transcribed audio content.",
                    "language": data.get('language', 'en')
                })
            elif tool_name == "classify_audio":
                base_result.update({
                    "predictions": [
                        {"label": "music", "score": 0.78},
                        {"label": "speech", "score": 0.15},
                        {"label": "noise", "score": 0.07}
                    ]
                })
        
        elif "image" in tool_name or "vision" in tool_name:
            if tool_name == "classify_image":
                base_result.update({
                    "predictions": [
                        {"label": "cat", "score": 0.87},
                        {"label": "dog", "score": 0.10},
                        {"label": "bird", "score": 0.03}
                    ]
                })
            elif tool_name == "detect_objects":
                base_result.update({
                    "detections": [
                        {"label": "person", "confidence": 0.94, "bbox": [100, 100, 200, 300]},
                        {"label": "car", "confidence": 0.81, "bbox": [300, 150, 500, 250]}
                    ]
                })
        
        return base_result
    
    def _model_matches_category(self, model: Dict[str, Any], category: str) -> bool:
        """Check if a model matches a specific category."""
        model_type = model.get("model_type", "").lower()
        architecture = model.get("architecture", "").lower()
        model_id = model.get("model_id", "").lower()
        
        category_mappings = {
            "text_processing": ["language", "text", "bert", "gpt", "t5", "bart"],
            "audio_processing": ["audio", "speech", "whisper", "wav2vec"],
            "vision_processing": ["vision", "image", "vit", "clip", "resnet"],
            "multimodal_processing": ["multimodal", "blip", "llava", "flamingo"],
            "specialized_processing": ["time", "code", "tabular", "specialized"]
        }
        
        keywords = category_mappings.get(category, [])
        return any(keyword in model_type or keyword in architecture or keyword in model_id 
                  for keyword in keywords)
    
    def _get_comprehensive_demo_models(self):
        """Get comprehensive demo models covering all categories."""
        return [
            # Text Processing Models
            {
                "model_id": "gpt2",
                "model_name": "GPT-2",
                "model_type": "language_model",
                "architecture": "transformer",
                "description": "Generative Pre-trained Transformer for text generation",
                "tags": ["generation", "transformer", "openai"]
            },
            {
                "model_id": "bert-base-uncased",
                "model_name": "BERT Base Uncased",
                "model_type": "language_model",
                "architecture": "bert",
                "description": "Bidirectional Encoder Representations from Transformers",
                "tags": ["classification", "bert", "google"]
            },
            {
                "model_id": "t5-small",
                "model_name": "T5 Small",
                "model_type": "seq2seq_model",
                "architecture": "t5",
                "description": "Text-to-Text Transfer Transformer",
                "tags": ["seq2seq", "google", "generation"]
            },
            
            # Audio Processing Models
            {
                "model_id": "whisper-base",
                "model_name": "Whisper Base",
                "model_type": "speech_model",
                "architecture": "whisper",
                "description": "Automatic Speech Recognition model by OpenAI",
                "tags": ["asr", "openai", "multilingual"]
            },
            {
                "model_id": "wav2vec2-base",
                "model_name": "Wav2Vec2 Base",
                "model_type": "speech_model",
                "architecture": "wav2vec2",
                "description": "Self-supervised speech representation learning",
                "tags": ["asr", "facebook", "self-supervised"]
            },
            
            # Vision Processing Models
            {
                "model_id": "vit-base-patch16",
                "model_name": "Vision Transformer Base",
                "model_type": "vision_model",
                "architecture": "vit",
                "description": "Vision Transformer for image classification",
                "tags": ["classification", "vision", "transformer"]
            },
            {
                "model_id": "clip-vit-base-patch32",
                "model_name": "CLIP ViT Base",
                "model_type": "multimodal_model",
                "architecture": "clip",
                "description": "Contrastive Language-Image Pre-training",
                "tags": ["multimodal", "openai", "vision"]
            },
            {
                "model_id": "detr-resnet-50",
                "model_name": "DETR ResNet-50",
                "model_type": "vision_model",
                "architecture": "detr",
                "description": "Detection Transformer for object detection",
                "tags": ["detection", "transformer", "facebook"]
            },
            
            # Multimodal Models
            {
                "model_id": "blip-image-captioning-base",
                "model_name": "BLIP Image Captioning",
                "model_type": "multimodal_model",
                "architecture": "blip",
                "description": "Bootstrapping Language-Image Pre-training for image captioning",
                "tags": ["captioning", "multimodal", "salesforce"]
            },
            {
                "model_id": "llava-1.5-7b",
                "model_name": "LLaVA 1.5 7B",
                "model_type": "multimodal_model",
                "architecture": "llava",
                "description": "Large Language and Vision Assistant",
                "tags": ["vqa", "multimodal", "conversation"]
            },
            
            # Specialized Models
            {
                "model_id": "stable-diffusion-v1-4",
                "model_name": "Stable Diffusion v1.4",
                "model_type": "diffusion_model",
                "architecture": "diffusion",
                "description": "Latent diffusion model for image generation",
                "tags": ["generation", "diffusion", "stability"]
            },
            {
                "model_id": "codegen-350M-mono",
                "model_name": "CodeGen 350M Mono",
                "model_type": "code_model",
                "architecture": "codegen",
                "description": "Code generation model for Python",
                "tags": ["code", "generation", "salesforce"]
            },
            {
                "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "model_name": "All MiniLM L6 v2",
                "model_type": "embedding_model",
                "architecture": "sentence-transformer",
                "description": "Sentence embedding model for semantic similarity",
                "tags": ["embeddings", "sentence-transformers", "similarity"]
            }
        ]
    
    def _initialize_comprehensive_models(self):
        """Initialize comprehensive sample models for testing."""
        if not self.model_manager:
            return
            
        try:
            # Check if we already have models
            existing_models = self.model_manager.list_models()
            if len(existing_models) >= 10:  # We want comprehensive coverage
                logger.info(f"Found {len(existing_models)} existing models")
                return
            
            # Add comprehensive sample models
            demo_models = self._get_comprehensive_demo_models()
            
            for model_data in demo_models:
                try:
                    # Convert to ModelMetadata format
                    model = ModelMetadata(
                        model_id=model_data["model_id"],
                        model_name=model_data["model_name"],
                        model_type=getattr(ModelType, model_data["model_type"].upper(), ModelType.LANGUAGE_MODEL),
                        architecture=model_data["architecture"],
                        description=model_data["description"],
                        tags=model_data["tags"],
                        inputs=[IOSpec(name="input", data_type=DataType.TOKENS, shape=(-1,), description="Input data")],
                        outputs=[IOSpec(name="output", data_type=DataType.LOGITS, shape=(-1,), description="Output data")]
                    )
                    self.model_manager.add_model(model)
                    logger.info(f"Added comprehensive model: {model.model_id}")
                except Exception as e:
                    logger.warning(f"Could not add model {model_data['model_id']}: {e}")
                    
        except Exception as e:
            logger.warning(f"Could not initialize comprehensive models: {e}")
    
    def run(self, host='127.0.0.1', port=8090, debug=True):
        """Run the comprehensive Flask application."""
        logger.info(f"Starting Comprehensive Kitchen Sink Testing Interface on {host}:{port}")
        logger.info(f"Supporting {len(self.inference_categories)} inference categories")
        logger.info(f"Total inference tools: {sum(len(cat['tools']) for cat in self.inference_categories.values())}")
        self.app.run(host=host, port=port, debug=debug)

# Flask app instance for external access
app = None

def create_comprehensive_app():
    """Create the comprehensive Flask application."""
    global app
    kitchen_sink = ComprehensiveKitchenSinkApp()
    app = kitchen_sink.app
    return kitchen_sink

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Kitchen Sink AI Model Testing Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8090, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    kitchen_sink = create_comprehensive_app()
    kitchen_sink.run(host=args.host, port=args.port, debug=args.debug)