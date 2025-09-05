#!/usr/bin/env python3
"""
AI-Powered Inference Tools for MCP Server

This module provides inference capabilities with automatic model selection
using bandit algorithms when no explicit model is provided.

Supported inference types:
- Causal Language Modeling (GPT-style)
- Masked Language Modeling (BERT-style)  
- Image Diffusion
- Text Classification
- Embedding Generation
- Question Answering
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union, Tuple
import json

# Import the Model Manager components
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ipfs_accelerate_py.model_manager import (
    ModelManager, BanditModelRecommender, RecommendationContext,
    DataType, ModelType
)

# Configure logging
logger = logging.getLogger("ai_inference_tools")

class InferenceEngine:
    """AI-powered inference engine with automatic model selection."""
    
    def __init__(self, 
                 model_manager: ModelManager,
                 bandit_recommender: BanditModelRecommender):
        """
        Initialize the inference engine.
        
        Args:
            model_manager: Model manager instance
            bandit_recommender: Bandit recommender instance
        """
        self.model_manager = model_manager
        self.bandit_recommender = bandit_recommender
        
        # Mock inference results for demonstration
        # In a real implementation, these would call actual model inference
        self._mock_mode = True
        
        logger.info("Inference engine initialized")
    
    def _select_model_for_task(self,
                              task_type: str,
                              model_id: Optional[str] = None,
                              hardware: str = "cpu",
                              input_type: str = "tokens",
                              output_type: str = "logits",
                              requirements: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        """
        Select the best model for a task using bandit algorithms if no model specified.
        
        Args:
            task_type: Type of inference task
            model_id: Specific model to use (optional)
            hardware: Hardware type
            input_type: Input data type
            output_type: Output data type
            requirements: Additional requirements
            
        Returns:
            Tuple of (selected_model_id, confidence_score)
        """
        if model_id:
            # Use explicit model if provided
            model = self.model_manager.get_model(model_id)
            if model:
                return model_id, 1.0
            else:
                logger.warning(f"Specified model {model_id} not found, using recommendation")
        
        # Use bandit algorithm for model selection
        context = RecommendationContext(
            task_type=task_type,
            hardware=hardware,
            input_type=DataType(input_type),
            output_type=DataType(output_type),
            requirements=requirements or {}
        )
        
        recommendation = self.bandit_recommender.recommend_model(context)
        if recommendation:
            return recommendation.model_id, recommendation.confidence_score
        
        # Fallback: return any available model
        models = self.model_manager.list_models()
        if models:
            return models[0].model_id, 0.1
        
        raise ValueError("No models available for inference")
    
    def _mock_inference(self, 
                       task_type: str, 
                       model_id: str,
                       input_data: Any) -> Dict[str, Any]:
        """
        Mock inference for demonstration purposes.
        
        Args:
            task_type: Type of inference
            model_id: Model being used
            input_data: Input data
            
        Returns:
            Mock inference results
        """
        if not self._mock_mode:
            raise NotImplementedError("Real inference not implemented")
        
        # Generate mock results based on task type
        if task_type == "causal_language_modeling":
            return {
                "generated_text": f"[Mock output from {model_id}] This is a generated continuation of the input text.",
                "confidence": 0.85,
                "tokens_generated": 15
            }
        elif task_type == "masked_language_modeling":
            return {
                "predictions": [
                    {"token": "the", "score": 0.45},
                    {"token": "a", "score": 0.23},
                    {"token": "an", "score": 0.12}
                ],
                "confidence": 0.78
            }
        elif task_type == "text_classification":
            return {
                "labels": [
                    {"label": "POSITIVE", "score": 0.82},
                    {"label": "NEGATIVE", "score": 0.18}
                ],
                "prediction": "POSITIVE",
                "confidence": 0.82
            }
        elif task_type == "embedding_generation":
            return {
                "embeddings": [0.1, -0.2, 0.3, 0.05, -0.15],  # Mock 5-dim embedding
                "dimension": 5,
                "norm": 0.367
            }
        elif task_type == "image_diffusion":
            return {
                "status": "generated",
                "image_url": f"mock://generated_image_{hash(str(input_data))}.png",
                "steps": 50,
                "guidance_scale": 7.5
            }
        elif task_type == "question_answering":
            return {
                "answer": "The answer to your question based on the context.",
                "confidence": 0.91,
                "start": 10,
                "end": 25
            }
        elif task_type == "automatic_speech_recognition":
            return {
                "text": f"Mock transcription from {model_id}: This is the transcribed text from the audio input.",
                "confidence": 0.89,
                "segments": [
                    {"start": 0.0, "end": 2.5, "text": "This is the transcribed"},
                    {"start": 2.5, "end": 5.0, "text": "text from the audio input."}
                ]
            }
        elif task_type == "image_classification":
            return {
                "predictions": [
                    {"label": "cat", "score": 0.85},
                    {"label": "dog", "score": 0.12},
                    {"label": "bird", "score": 0.03}
                ],
                "confidence": 0.85
            }
        elif task_type == "object_detection":
            return {
                "detections": [
                    {
                        "label": "person",
                        "confidence": 0.92,
                        "bbox": [100, 100, 200, 300]
                    },
                    {
                        "label": "car",
                        "confidence": 0.78,
                        "bbox": [300, 150, 500, 250]
                    }
                ],
                "confidence": 0.85
            }
        elif task_type == "image_to_text":
            return {
                "caption": f"A detailed description of the image generated by {model_id} showing various objects and scenes.",
                "confidence": 0.88
            }
        elif task_type == "visual_question_answering":
            return {
                "answer": f"Based on the image analysis by {model_id}, the answer to your question is provided here.",
                "confidence": 0.87
            }
        elif task_type == "text_to_speech":
            return {
                "audio_url": f"mock://synthesized_{hash(str(input_data))}.wav",
                "duration": len(str(input_data.get('text', ''))) * 0.1,
                "sample_rate": 22050,
                "confidence": 0.92
            }
        elif task_type == "translation":
            return {
                "translated_text": f"[Translation by {model_id}] This is the translated version of the input text.",
                "confidence": 0.91
            }
        elif task_type == "summarization":
            return {
                "summary": f"[Summary by {model_id}] This is a concise summary of the input text highlighting the key points.",
                "confidence": 0.86
            }
        elif task_type == "audio_classification":
            return {
                "predictions": [
                    {"label": "music", "score": 0.78},
                    {"label": "speech", "score": 0.15},
                    {"label": "noise", "score": 0.07}
                ],
                "confidence": 0.78
            }
        else:
            return {
                "result": f"Mock result for {task_type}",
                "confidence": 0.75
            }

class InferenceTools:
    """Collection of inference tools for the MCP server."""
    
    def __init__(self, inference_engine: InferenceEngine):
        """
        Initialize inference tools.
        
        Args:
            inference_engine: The inference engine instance
        """
        self.engine = inference_engine
    
    def register_tools(self, mcp):
        """Register all inference tools with the MCP server."""
        
        @mcp.tool()
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
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="causal_language_modeling",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="tokens",
                    output_type="tokens"
                )
                
                # Perform inference
                result = self.engine._mock_inference(
                    task_type="causal_language_modeling",
                    model_id=selected_model,
                    input_data={"prompt": prompt, "max_length": max_length, "temperature": temperature}
                )
                
                return {
                    "generated_text": result["generated_text"],
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "inference_confidence": result["confidence"],
                    "tokens_generated": result["tokens_generated"],
                    "parameters": {
                        "max_length": max_length,
                        "temperature": temperature,
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in text generation: {e}")
                return {"error": f"Text generation failed: {str(e)}"}
        
        @mcp.tool()
        def fill_mask(
            text_with_mask: str,
            model_id: Optional[str] = None,
            top_k: int = 5,
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Fill masked tokens in text using masked language modeling.
            
            Args:
                text_with_mask: Text with [MASK] tokens to fill
                model_id: Specific model to use (optional)
                top_k: Number of top predictions to return
                hardware: Hardware type to use
                
            Returns:
                Predictions for masked tokens
            """
            try:
                # Select model using bandit algorithm if not specified
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="masked_language_modeling",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="tokens",
                    output_type="logits"
                )
                
                # Perform inference
                result = self.engine._mock_inference(
                    task_type="masked_language_modeling",
                    model_id=selected_model,
                    input_data={"text": text_with_mask, "top_k": top_k}
                )
                
                return {
                    "predictions": result["predictions"][:top_k],
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "inference_confidence": result["confidence"],
                    "parameters": {
                        "top_k": top_k,
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in mask filling: {e}")
                return {"error": f"Mask filling failed: {str(e)}"}
        
        @mcp.tool()
        def classify_text(
            text: str,
            model_id: Optional[str] = None,
            return_all_scores: bool = False,
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Classify text using a classification model.
            
            Args:
                text: Text to classify
                model_id: Specific model to use (optional)
                return_all_scores: Whether to return all label scores
                hardware: Hardware type to use
                
            Returns:
                Classification results
            """
            try:
                # Select model using bandit algorithm if not specified
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="text_classification",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="tokens",
                    output_type="logits"
                )
                
                # Perform inference
                result = self.engine._mock_inference(
                    task_type="text_classification",
                    model_id=selected_model,
                    input_data={"text": text}
                )
                
                response = {
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "parameters": {
                        "return_all_scores": return_all_scores,
                        "hardware": hardware
                    }
                }
                
                if return_all_scores:
                    response["all_scores"] = result["labels"]
                
                return response
                
            except Exception as e:
                logger.error(f"Error in text classification: {e}")
                return {"error": f"Text classification failed: {str(e)}"}
        
        @mcp.tool()
        def generate_embeddings(
            text: str,
            model_id: Optional[str] = None,
            normalize: bool = True,
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Generate text embeddings.
            
            Args:
                text: Text to embed
                model_id: Specific model to use (optional)
                normalize: Whether to normalize embeddings
                hardware: Hardware type to use
                
            Returns:
                Text embeddings
            """
            try:
                # Select model using bandit algorithm if not specified
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="embedding_generation",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="tokens", 
                    output_type="embeddings"
                )
                
                # Perform inference
                result = self.engine._mock_inference(
                    task_type="embedding_generation",
                    model_id=selected_model,
                    input_data={"text": text, "normalize": normalize}
                )
                
                return {
                    "embeddings": result["embeddings"],
                    "dimension": result["dimension"],
                    "norm": result["norm"],
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "parameters": {
                        "normalize": normalize,
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in embedding generation: {e}")
                return {"error": f"Embedding generation failed: {str(e)}"}
        
        @mcp.tool()
        def generate_image(
            prompt: str,
            model_id: Optional[str] = None,
            width: int = 512,
            height: int = 512,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Generate image using diffusion models.
            
            Args:
                prompt: Text prompt for image generation
                model_id: Specific model to use (optional)
                width: Image width in pixels
                height: Image height in pixels
                num_inference_steps: Number of denoising steps
                guidance_scale: Guidance scale for generation
                hardware: Hardware type to use
                
            Returns:
                Generated image information
            """
            try:
                # Select model using bandit algorithm if not specified
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="image_diffusion",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="tokens",
                    output_type="image"
                )
                
                # Perform inference
                result = self.engine._mock_inference(
                    task_type="image_diffusion",
                    model_id=selected_model,
                    input_data={
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "steps": num_inference_steps,
                        "guidance_scale": guidance_scale
                    }
                )
                
                return {
                    "status": result["status"],
                    "image_url": result["image_url"],
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "parameters": {
                        "width": width,
                        "height": height,
                        "num_inference_steps": result["steps"],
                        "guidance_scale": result["guidance_scale"],
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in image generation: {e}")
                return {"error": f"Image generation failed: {str(e)}"}
        
        @mcp.tool()
        def answer_question(
            question: str,
            context: str,
            model_id: Optional[str] = None,
            max_answer_length: int = 100,
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Answer questions based on provided context.
            
            Args:
                question: Question to answer
                context: Context text containing the answer
                model_id: Specific model to use (optional)
                max_answer_length: Maximum length of the answer
                hardware: Hardware type to use
                
            Returns:
                Answer and metadata
            """
            try:
                # Select model using bandit algorithm if not specified
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="question_answering",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="tokens",
                    output_type="tokens"
                )
                
                # Perform inference
                result = self.engine._mock_inference(
                    task_type="question_answering",
                    model_id=selected_model,
                    input_data={
                        "question": question,
                        "context": context,
                        "max_answer_length": max_answer_length
                    }
                )
                
                return {
                    "answer": result["answer"],
                    "confidence": result["confidence"],
                    "start_position": result["start"],
                    "end_position": result["end"],
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "parameters": {
                        "max_answer_length": max_answer_length,
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in question answering: {e}")
                return {"error": f"Question answering failed: {str(e)}"}
        
        # Advanced Inference Tools
        @mcp.tool()
        def transcribe_audio(
            audio_data: str,  # Base64 encoded audio or file path
            model_id: Optional[str] = None,
            language: Optional[str] = None,
            task: str = "transcribe",  # "transcribe" or "translate"
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Transcribe or translate audio using speech recognition models.
            
            Args:
                audio_data: Base64 encoded audio or file path
                model_id: Specific model to use (optional, will auto-select Whisper-type model)
                language: Language code for transcription
                task: "transcribe" or "translate"
                hardware: Hardware type to use
                
            Returns:
                Transcription results with confidence scores
            """
            try:
                # Select model using bandit algorithm if not specified
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="automatic_speech_recognition",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="audio",
                    output_type="text"
                )
                
                # Mock inference result
                result = self.engine._mock_inference(
                    task_type="automatic_speech_recognition",
                    model_id=selected_model,
                    input_data={"audio": audio_data, "language": language, "task": task}
                )
                
                return {
                    "transcription": result.get("text", f"[Mock transcription from {selected_model}] This is the transcribed text from the audio."),
                    "language": language or "en",
                    "confidence": result.get("confidence", 0.89),
                    "segments": result.get("segments", []),
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "task": task,
                    "parameters": {
                        "language": language,
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in audio transcription: {e}")
                return {"error": f"Audio transcription failed: {str(e)}"}

        @mcp.tool()
        def classify_image(
            image_data: str,  # Base64 encoded image or file path
            model_id: Optional[str] = None,
            top_k: int = 5,
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Classify images using vision models.
            
            Args:
                image_data: Base64 encoded image or file path
                model_id: Specific model to use (optional, will auto-select vision model)
                top_k: Number of top predictions to return
                hardware: Hardware type to use
                
            Returns:
                Classification results with confidence scores
            """
            try:
                # Select model using bandit algorithm if not specified
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="image_classification",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="image",
                    output_type="logits"
                )
                
                # Mock inference result
                result = self.engine._mock_inference(
                    task_type="image_classification",
                    model_id=selected_model,
                    input_data={"image": image_data, "top_k": top_k}
                )
                
                return {
                    "predictions": result.get("predictions", [
                        {"label": "cat", "score": 0.85},
                        {"label": "dog", "score": 0.12},
                        {"label": "bird", "score": 0.03}
                    ][:top_k]),
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "inference_confidence": result.get("confidence", 0.85),
                    "parameters": {
                        "top_k": top_k,
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in image classification: {e}")
                return {"error": f"Image classification failed: {str(e)}"}

        @mcp.tool()
        def detect_objects(
            image_data: str,  # Base64 encoded image or file path
            model_id: Optional[str] = None,
            confidence_threshold: float = 0.5,
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Detect objects in images using object detection models.
            
            Args:
                image_data: Base64 encoded image or file path
                model_id: Specific model to use (optional, will auto-select detection model)
                confidence_threshold: Minimum confidence for detections
                hardware: Hardware type to use
                
            Returns:
                Object detection results with bounding boxes and confidence scores
            """
            try:
                # Select model using bandit algorithm if not specified
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="object_detection",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="image",
                    output_type="boxes"
                )
                
                # Mock inference result
                result = self.engine._mock_inference(
                    task_type="object_detection",
                    model_id=selected_model,
                    input_data={"image": image_data, "threshold": confidence_threshold}
                )
                
                return {
                    "detections": result.get("detections", [
                        {
                            "label": "person",
                            "confidence": 0.92,
                            "bbox": [100, 100, 200, 300]  # x1, y1, x2, y2
                        },
                        {
                            "label": "car", 
                            "confidence": 0.78,
                            "bbox": [300, 150, 500, 250]
                        }
                    ]),
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "inference_confidence": result.get("confidence", 0.85),
                    "parameters": {
                        "confidence_threshold": confidence_threshold,
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in object detection: {e}")
                return {"error": f"Object detection failed: {str(e)}"}

        @mcp.tool()
        def generate_image_caption(
            image_data: str,  # Base64 encoded image or file path
            model_id: Optional[str] = None,
            max_length: int = 50,
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Generate captions for images using multimodal models.
            
            Args:
                image_data: Base64 encoded image or file path
                model_id: Specific model to use (optional, will auto-select multimodal model)
                max_length: Maximum caption length
                hardware: Hardware type to use
                
            Returns:
                Generated caption with confidence score
            """
            try:
                # Select model using bandit algorithm if not specified
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="image_to_text",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="image",
                    output_type="text"
                )
                
                # Mock inference result
                result = self.engine._mock_inference(
                    task_type="image_to_text",
                    model_id=selected_model,
                    input_data={"image": image_data, "max_length": max_length}
                )
                
                return {
                    "caption": result.get("caption", f"[Generated by {selected_model}] A detailed description of the image showing various objects and scenes."),
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "inference_confidence": result.get("confidence", 0.88),
                    "parameters": {
                        "max_length": max_length,
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in image captioning: {e}")
                return {"error": f"Image captioning failed: {str(e)}"}

        @mcp.tool()
        def answer_visual_question(
            image_data: str,  # Base64 encoded image or file path
            question: str,
            model_id: Optional[str] = None,
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Answer questions about images using visual question answering models.
            
            Args:
                image_data: Base64 encoded image or file path
                question: Question about the image
                model_id: Specific model to use (optional, will auto-select VQA model)
                hardware: Hardware type to use
                
            Returns:
                Answer with confidence score
            """
            try:
                # Select model using bandit algorithm if not specified
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="visual_question_answering",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="multimodal",
                    output_type="text"
                )
                
                # Mock inference result
                result = self.engine._mock_inference(
                    task_type="visual_question_answering",
                    model_id=selected_model,
                    input_data={"image": image_data, "question": question}
                )
                
                return {
                    "answer": result.get("answer", f"[Answer from {selected_model}] Based on the image, the answer to your question is provided here."),
                    "question": question,
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "inference_confidence": result.get("confidence", 0.87),
                    "parameters": {
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in visual question answering: {e}")
                return {"error": f"Visual question answering failed: {str(e)}"}

        @mcp.tool()
        def synthesize_speech(
            text: str,
            model_id: Optional[str] = None,
            speaker: Optional[str] = None,
            language: str = "en",
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Synthesize speech from text using text-to-speech models.
            
            Args:
                text: Text to synthesize
                model_id: Specific model to use (optional, will auto-select TTS model)
                speaker: Speaker voice to use
                language: Language code
                hardware: Hardware type to use
                
            Returns:
                Synthesized speech metadata and information
            """
            try:
                # Select model using bandit algorithm if not specified
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="text_to_speech",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="text",
                    output_type="audio"
                )
                
                # Mock inference result
                result = self.engine._mock_inference(
                    task_type="text_to_speech",
                    model_id=selected_model,
                    input_data={"text": text, "speaker": speaker, "language": language}
                )
                
                return {
                    "status": "synthesized",
                    "audio_url": result.get("audio_url", f"mock://synthesized_{hash(text)}.wav"),
                    "duration": result.get("duration", len(text) * 0.1),  # Mock duration
                    "sample_rate": result.get("sample_rate", 22050),
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "parameters": {
                        "speaker": speaker,
                        "language": language,
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in speech synthesis: {e}")
                return {"error": f"Speech synthesis failed: {str(e)}"}

        @mcp.tool()
        def translate_text(
            text: str,
            source_language: str,
            target_language: str,
            model_id: Optional[str] = None,
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Translate text between languages using translation models.
            
            Args:
                text: Text to translate
                source_language: Source language code
                target_language: Target language code
                model_id: Specific model to use (optional, will auto-select translation model)
                hardware: Hardware type to use
                
            Returns:
                Translated text with confidence score
            """
            try:
                # Select model using bandit algorithm if not specified
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="translation",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="text",
                    output_type="text"
                )
                
                # Mock inference result
                result = self.engine._mock_inference(
                    task_type="translation",
                    model_id=selected_model,
                    input_data={
                        "text": text,
                        "source_language": source_language,
                        "target_language": target_language
                    }
                )
                
                return {
                    "translated_text": result.get("translated_text", f"[Translation by {selected_model}] This is the translated version of the input text."),
                    "source_language": source_language,
                    "target_language": target_language,
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "inference_confidence": result.get("confidence", 0.91),
                    "parameters": {
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in text translation: {e}")
                return {"error": f"Text translation failed: {str(e)}"}

        @mcp.tool()
        def summarize_text(
            text: str,
            model_id: Optional[str] = None,
            max_length: int = 150,
            min_length: int = 30,
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Summarize text using summarization models.
            
            Args:
                text: Text to summarize
                model_id: Specific model to use (optional, will auto-select summarization model)
                max_length: Maximum summary length
                min_length: Minimum summary length
                hardware: Hardware type to use
                
            Returns:
                Summarized text with confidence score
            """
            try:
                # Select model using bandit algorithm if not specified
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="summarization",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="text",
                    output_type="text"
                )
                
                # Mock inference result
                result = self.engine._mock_inference(
                    task_type="summarization",
                    model_id=selected_model,
                    input_data={"text": text, "max_length": max_length, "min_length": min_length}
                )
                
                return {
                    "summary": result.get("summary", f"[Summary by {selected_model}] This is a concise summary of the input text highlighting the key points."),
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "inference_confidence": result.get("confidence", 0.86),
                    "parameters": {
                        "max_length": max_length,
                        "min_length": min_length,
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in text summarization: {e}")
                return {"error": f"Text summarization failed: {str(e)}"}

        @mcp.tool()
        def classify_audio(
            audio_data: str,  # Base64 encoded audio or file path
            model_id: Optional[str] = None,
            top_k: int = 5,
            hardware: str = "cpu"
        ) -> Dict[str, Any]:
            """
            Classify audio using audio classification models.
            
            Args:
                audio_data: Base64 encoded audio or file path
                model_id: Specific model to use (optional, will auto-select audio classifier)
                top_k: Number of top predictions to return
                hardware: Hardware type to use
                
            Returns:
                Audio classification results with confidence scores
            """
            try:
                # Select model using bandit algorithm if not specified
                selected_model, confidence = self.engine._select_model_for_task(
                    task_type="audio_classification",
                    model_id=model_id,
                    hardware=hardware,
                    input_type="audio",
                    output_type="logits"
                )
                
                # Mock inference result
                result = self.engine._mock_inference(
                    task_type="audio_classification",
                    model_id=selected_model,
                    input_data={"audio": audio_data, "top_k": top_k}
                )
                
                return {
                    "predictions": result.get("predictions", [
                        {"label": "music", "score": 0.78},
                        {"label": "speech", "score": 0.15},
                        {"label": "noise", "score": 0.07}
                    ][:top_k]),
                    "model_used": selected_model,
                    "model_confidence": confidence,
                    "inference_confidence": result.get("confidence", 0.78),
                    "parameters": {
                        "top_k": top_k,
                        "hardware": hardware
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in audio classification: {e}")
                return {"error": f"Audio classification failed: {str(e)}"}

        # Feedback tool for improving recommendations
        @mcp.tool()
        def provide_inference_feedback(
            task_type: str,
            model_id: str,
            performance_score: float,
            hardware: str = "cpu",
            input_type: str = "tokens",
            output_type: str = "logits",
            details: Optional[Dict[str, Any]] = None
        ) -> Dict[str, str]:
            """
            Provide feedback on inference performance to improve future model selection.
            
            Args:
                task_type: Type of inference task that was performed
                model_id: Model that was used
                performance_score: Performance score (0.0 to 1.0)
                hardware: Hardware that was used
                input_type: Input data type
                output_type: Output data type
                details: Additional details about the performance
                
            Returns:
                Status message
            """
            try:
                # Create context for feedback
                context = RecommendationContext(
                    task_type=task_type,
                    hardware=hardware,
                    input_type=DataType(input_type),
                    output_type=DataType(output_type),
                    requirements=details or {}
                )
                
                # Provide feedback to improve recommendations
                self.engine.bandit_recommender.provide_feedback(
                    model_id, performance_score, context
                )
                
                return {
                    "status": "success",
                    "message": f"Feedback recorded for {task_type} task using {model_id} (score: {performance_score})"
                }
                
            except Exception as e:
                logger.error(f"Error providing inference feedback: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to record feedback: {str(e)}"
                }
        
        logger.info("All inference tools registered (13 total tools)")

def create_inference_tools(model_manager: ModelManager, 
                         bandit_recommender: BanditModelRecommender) -> InferenceTools:
    """
    Create inference tools instance.
    
    Args:
        model_manager: Model manager instance
        bandit_recommender: Bandit recommender instance
        
    Returns:
        Configured InferenceTools instance
    """
    engine = InferenceEngine(model_manager, bandit_recommender)
    return InferenceTools(engine)