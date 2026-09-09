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

import anyio
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union, Tuple
import json

# Import the Model Manager components
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ipfs_accelerate_py.model_manager import (
    ModelManager,
    BanditModelRecommender,
    RecommendationContext,
    DataType,
    ModelType,
)

# Configure logging
logger = logging.getLogger("ai_inference_tools")


class InferenceEngine:
    """AI-powered inference engine with automatic model selection."""

    def __init__(self, model_manager: ModelManager, bandit_recommender: BanditModelRecommender):
        """
        Initialize the inference engine.

        Args:
            model_manager: Model manager instance
            bandit_recommender: Bandit recommender instance
        """
        self.model_manager = model_manager
        self.bandit_recommender = bandit_recommender

        # PCPR-033: mock inference is not live. Ordinary runtime stays typed
        # unavailable unless the caller selected explicit simulation.
        self._mock_mode = False
        self._explicit_simulation = False

        logger.info("Inference engine initialized")

    def _select_model_for_task(
        self,
        task_type: str,
        model_id: Optional[str] = None,
        hardware: str = "cpu",
        input_type: str = "tokens",
        output_type: str = "logits",
        requirements: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float]:
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
            requirements=requirements or {},
        )

        recommendation = self.bandit_recommender.recommend_model(context)
        if recommendation:
            return recommendation.model_id, recommendation.confidence_score

        # Fallback: return any available model
        models = self.model_manager.list_models()
        if models:
            return models[0].model_id, 0.1

        raise ValueError("No models available for inference")

    def infer(
        self,
        task_type: str,
        model_id: str,
        input_data: Any,
        *,
        explicit_simulation: bool | None = None,
    ) -> Dict[str, Any]:
        """Run inference. Missing live evidence stays typed unavailable."""

        from ipfs_accelerate_py.compatibility.simulation.fabricated_endpoint_success import (
            run_inference,
        )

        simulated = (
            bool(explicit_simulation)
            if explicit_simulation is not None
            else bool(self._explicit_simulation or self._mock_mode)
        )
        return run_inference(
            task_type,
            model_id,
            input_data,
            explicit_simulation=simulated,
        )

    def _mock_inference(self, task_type: str, model_id: str, input_data: Any) -> Dict[str, Any]:
        """Compatibility name. Ordinary calls are typed unavailable, never live success."""

        return self.infer(task_type, model_id, input_data)


class InferenceTools:
    """Collection of inference tools for the MCP server."""

    def __init__(self, inference_engine: InferenceEngine):
        """
        Initialize inference tools.

        Args:
            inference_engine: The inference engine instance
        """
        self.engine = inference_engine

    def _present_inference_tool(
        self,
        *,
        task_type: str,
        model_id: Optional[str],
        hardware: str,
        input_type: str,
        output_type: str,
        input_data: Any,
        parameters: Optional[Dict[str, Any]] = None,
        error_label: str = "Inference failed",
    ) -> Dict[str, Any]:
        """Return live Observed/Verified results only. Otherwise typed unavailable."""

        try:
            selected_model, confidence = self.engine._select_model_for_task(
                task_type=task_type,
                model_id=model_id,
                hardware=hardware,
                input_type=input_type,
                output_type=output_type,
            )
            result = self.engine.infer(task_type, selected_model, input_data)
            envelope = dict(result)
            envelope["model_used"] = selected_model
            envelope["model_confidence"] = confidence
            envelope["parameters"] = parameters or {"hardware": hardware}
            if envelope.get("outcome") not in ("Observed", "Verified"):
                envelope["live"] = False
                envelope.setdefault(
                    "status",
                    str(envelope.get("outcome", "Unavailable")).lower(),
                )
            return envelope
        except Exception as e:
            logger.error(f"{error_label}: {e}")
            return {
                "status": "failed",
                "outcome": "Failed",
                "live": False,
                "simulated": False,
                "error": f"{error_label}: {str(e)}",
            }

    def register_tools(self, mcp):
        """Register all inference tools with the MCP server."""

        @mcp.tool()
        def generate_text(
            prompt: str,
            model_id: Optional[str] = None,
            max_length: int = 100,
            temperature: float = 0.7,
            hardware: str = "cpu",
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
                Generated text and metadata when live evidence exists; otherwise
                a typed unavailable envelope.
            """
            return self._present_inference_tool(
                task_type="causal_language_modeling",
                model_id=model_id,
                hardware=hardware,
                input_type="tokens",
                output_type="tokens",
                input_data={
                    "prompt": prompt,
                    "max_length": max_length,
                    "temperature": temperature,
                },
                parameters={
                    "max_length": max_length,
                    "temperature": temperature,
                    "hardware": hardware,
                },
                error_label="Text generation failed",
            )

        @mcp.tool()
        def fill_mask(
            text_with_mask: str,
            model_id: Optional[str] = None,
            top_k: int = 5,
            hardware: str = "cpu",
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
            return self._present_inference_tool(
                task_type="masked_language_modeling",
                model_id=model_id,
                hardware=hardware,
                input_type="tokens",
                output_type="logits",
                input_data={"text": text_with_mask, "top_k": top_k},
                parameters={"top_k": top_k, "hardware": hardware},
                error_label="Mask filling failed",
            )

        @mcp.tool()
        def classify_text(
            text: str,
            model_id: Optional[str] = None,
            return_all_scores: bool = False,
            hardware: str = "cpu",
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
            return self._present_inference_tool(
                task_type="text_classification",
                model_id=model_id,
                hardware=hardware,
                input_type="tokens",
                output_type="logits",
                input_data={"text": text, "return_all_scores": return_all_scores},
                parameters={"return_all_scores": return_all_scores, "hardware": hardware},
                error_label="Text classification failed",
            )

        @mcp.tool()
        def generate_embeddings(
            text: str, model_id: Optional[str] = None, normalize: bool = True, hardware: str = "cpu"
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
            return self._present_inference_tool(
                task_type="embedding_generation",
                model_id=model_id,
                hardware=hardware,
                input_type="tokens",
                output_type="embeddings",
                input_data={"text": text, "normalize": normalize},
                parameters={"normalize": normalize, "hardware": hardware},
                error_label="Embedding generation failed",
            )

        @mcp.tool()
        def generate_image(
            prompt: str,
            model_id: Optional[str] = None,
            width: int = 512,
            height: int = 512,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            hardware: str = "cpu",
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
            return self._present_inference_tool(
                task_type="image_diffusion",
                model_id=model_id,
                hardware=hardware,
                input_type="tokens",
                output_type="image",
                input_data={
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                },
                parameters={
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "hardware": hardware,
                },
                error_label="Image generation failed",
            )

        @mcp.tool()
        def answer_question(
            question: str,
            context: str,
            model_id: Optional[str] = None,
            max_answer_length: int = 100,
            hardware: str = "cpu",
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
            return self._present_inference_tool(
                task_type="question_answering",
                model_id=model_id,
                hardware=hardware,
                input_type="tokens",
                output_type="tokens",
                input_data={
                    "question": question,
                    "context": context,
                    "max_answer_length": max_answer_length,
                },
                parameters={"max_answer_length": max_answer_length, "hardware": hardware},
                error_label="Question answering failed",
            )

        # Advanced Inference Tools
        @mcp.tool()
        def transcribe_audio(
            audio_data: str,  # Base64 encoded audio or file path
            model_id: Optional[str] = None,
            language: Optional[str] = None,
            task: str = "transcribe",  # "transcribe" or "translate"
            hardware: str = "cpu",
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
            return self._present_inference_tool(
                task_type="automatic_speech_recognition",
                model_id=model_id,
                hardware=hardware,
                input_type="audio",
                output_type="text",
                input_data={"audio": audio_data, "language": language, "task": task},
                parameters={"language": language, "task": task, "hardware": hardware},
                error_label="Audio transcription failed",
            )

        @mcp.tool()
        def classify_image(
            image_data: str,  # Base64 encoded image or file path
            model_id: Optional[str] = None,
            top_k: int = 5,
            hardware: str = "cpu",
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
            return self._present_inference_tool(
                task_type="image_classification",
                model_id=model_id,
                hardware=hardware,
                input_type="image",
                output_type="logits",
                input_data={"image": image_data, "top_k": top_k},
                parameters={"top_k": top_k, "hardware": hardware},
                error_label="Image classification failed",
            )

        @mcp.tool()
        def detect_objects(
            image_data: str,  # Base64 encoded image or file path
            model_id: Optional[str] = None,
            confidence_threshold: float = 0.5,
            hardware: str = "cpu",
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
            return self._present_inference_tool(
                task_type="object_detection",
                model_id=model_id,
                hardware=hardware,
                input_type="image",
                output_type="boxes",
                input_data={"image": image_data, "threshold": confidence_threshold},
                parameters={
                    "confidence_threshold": confidence_threshold,
                    "hardware": hardware,
                },
                error_label="Object detection failed",
            )

        @mcp.tool()
        def generate_image_caption(
            image_data: str,  # Base64 encoded image or file path
            model_id: Optional[str] = None,
            max_length: int = 50,
            hardware: str = "cpu",
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
            return self._present_inference_tool(
                task_type="image_to_text",
                model_id=model_id,
                hardware=hardware,
                input_type="image",
                output_type="text",
                input_data={"image": image_data, "max_length": max_length},
                parameters={"max_length": max_length, "hardware": hardware},
                error_label="Image captioning failed",
            )

        @mcp.tool()
        def answer_visual_question(
            image_data: str,  # Base64 encoded image or file path
            question: str,
            model_id: Optional[str] = None,
            hardware: str = "cpu",
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
            return self._present_inference_tool(
                task_type="visual_question_answering",
                model_id=model_id,
                hardware=hardware,
                input_type="multimodal",
                output_type="text",
                input_data={"image": image_data, "question": question},
                parameters={"question": question, "hardware": hardware},
                error_label="Visual question answering failed",
            )

        @mcp.tool()
        def synthesize_speech(
            text: str,
            model_id: Optional[str] = None,
            speaker: Optional[str] = None,
            language: str = "en",
            hardware: str = "cpu",
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
            return self._present_inference_tool(
                task_type="text_to_speech",
                model_id=model_id,
                hardware=hardware,
                input_type="text",
                output_type="audio",
                input_data={"text": text, "speaker": speaker, "language": language},
                parameters={"speaker": speaker, "language": language, "hardware": hardware},
                error_label="Speech synthesis failed",
            )

        @mcp.tool()
        def translate_text(
            text: str,
            source_language: str,
            target_language: str,
            model_id: Optional[str] = None,
            hardware: str = "cpu",
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
            return self._present_inference_tool(
                task_type="translation",
                model_id=model_id,
                hardware=hardware,
                input_type="text",
                output_type="text",
                input_data={
                    "text": text,
                    "source_language": source_language,
                    "target_language": target_language,
                },
                parameters={
                    "source_language": source_language,
                    "target_language": target_language,
                    "hardware": hardware,
                },
                error_label="Text translation failed",
            )

        @mcp.tool()
        def summarize_text(
            text: str,
            model_id: Optional[str] = None,
            max_length: int = 150,
            min_length: int = 30,
            hardware: str = "cpu",
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
            return self._present_inference_tool(
                task_type="summarization",
                model_id=model_id,
                hardware=hardware,
                input_type="text",
                output_type="text",
                input_data={"text": text, "max_length": max_length, "min_length": min_length},
                parameters={
                    "max_length": max_length,
                    "min_length": min_length,
                    "hardware": hardware,
                },
                error_label="Text summarization failed",
            )

        @mcp.tool()
        def classify_audio(
            audio_data: str,  # Base64 encoded audio or file path
            model_id: Optional[str] = None,
            top_k: int = 5,
            hardware: str = "cpu",
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
            return self._present_inference_tool(
                task_type="audio_classification",
                model_id=model_id,
                hardware=hardware,
                input_type="audio",
                output_type="logits",
                input_data={"audio": audio_data, "top_k": top_k},
                parameters={"top_k": top_k, "hardware": hardware},
                error_label="Audio classification failed",
            )

        # Feedback tool for improving recommendations
        @mcp.tool()
        def provide_inference_feedback(
            task_type: str,
            model_id: str,
            performance_score: float,
            hardware: str = "cpu",
            input_type: str = "tokens",
            output_type: str = "logits",
            details: Optional[Dict[str, Any]] = None,
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
                    requirements=details or {},
                )

                # Provide feedback to improve recommendations
                self.engine.bandit_recommender.provide_feedback(
                    model_id, performance_score, context
                )

                return {
                    "status": "success",
                    "message": f"Feedback recorded for {task_type} task using {model_id} (score: {performance_score})",
                }

            except Exception as e:
                logger.error(f"Error providing inference feedback: {e}")
                return {"status": "error", "message": f"Failed to record feedback: {str(e)}"}

        def _set_execution_context(tool_name: str, execution_context: str) -> None:
            tools = getattr(mcp, "tools", None)
            if not isinstance(tools, dict):
                return
            tool_entry = tools.get(tool_name)
            if not isinstance(tool_entry, dict):
                return
            tool_entry["execution_context"] = execution_context

        for _tool_name in [
            "generate_text",
            "fill_mask",
            "classify_text",
            "generate_embeddings",
            "generate_image",
            "answer_question",
            "transcribe_audio",
            "classify_image",
            "detect_objects",
            "generate_image_caption",
            "answer_visual_question",
            "synthesize_speech",
            "translate_text",
            "summarize_text",
            "classify_audio",
        ]:
            _set_execution_context(_tool_name, "worker")

        _set_execution_context("provide_inference_feedback", "server")

        logger.info("All inference tools registered (16 total tools)")


def create_inference_tools(
    model_manager: ModelManager, bandit_recommender: BanditModelRecommender
) -> InferenceTools:
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
