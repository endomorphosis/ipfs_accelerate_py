#!/usr/bin/env python
"""
Enhanced Implementation of OpenAI API for ipfs_accelerate_py
This includes support for latest OpenAI models and features and integration with
the Distributed Testing Framework.
"""

import os
import sys
import json
import time
import base64
import random
import logging
import threading
import concurrent.futures
from typing import List, Dict, Any, Optional, Union, Callable, BinaryIO, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO

try:
    # Import the OpenAI API
    import openai
    from openai import OpenAI
    from openai.types.chat import ChatCompletion
    from openai.types.audio import Transcription, Translation, SpeechResponse
except ImportError:
    openai = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OpenAIAPI")

class VoiceType(str, Enum):
    """Voice options for text-to-speech"""
    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"

class AudioFormat(str, Enum):
    """Audio format options for text-to-speech"""
    MP3 = "mp3"
    OPUS = "opus"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"

class OpenAIAPI:
    """
    Enhanced OpenAI API implementation for ipfs_accelerate_py.
    Includes support for:
    - Latest models (GPT-4o, GPT-4o-mini)
    - Advanced function calling with parallel execution
    - Integration with distributed testing framework
    - Enhanced audio features (TTS with multiple voices, STT)
    - Performance metrics collection
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 resources: Dict[str, Any] = None, 
                 metadata: Dict[str, Any] = None,
                 organization: Optional[str] = None,
                 metrics_enabled: bool = True,
                 distributed_mode: bool = False):
        """
        Initialize the OpenAI API with enhanced features.
        
        Args:
            api_key: OpenAI API key (optional, will use env var if not provided)
            resources: Resources dictionary for ipfs_accelerate_py
            metadata: Metadata dictionary for ipfs_accelerate_py
            organization: OpenAI organization ID (optional)
            metrics_enabled: Whether to collect performance metrics
            distributed_mode: Enable integration with distributed testing framework
        """
        self.resources = resources if resources else {}
        self.metadata = metadata if metadata else {}
        
        # Get API key from parameters, metadata, or environment variable
        self.api_key = api_key or (metadata.get("openai_api_key") if metadata else None) or os.environ.get("OPENAI_API_KEY")
        self.organization = organization or (metadata.get("openai_organization") if metadata else None) or os.environ.get("OPENAI_ORGANIZATION")
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided. API calls will not work without an API key.")
        
        # Set up the OpenAI API client
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                organization=self.organization
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
        
        # Metrics and distributed testing settings
        self.metrics_enabled = metrics_enabled
        self.distributed_mode = distributed_mode
        self.metrics_collector = self._init_metrics_collector() if metrics_enabled else None
        
        # Request tracking for rate limiting and backoff
        self.current_requests = 0
        self.max_concurrent_requests = 5
        self.request_queue = []
        self.queue_lock = threading.Lock()
        self.max_retries = 5
        self.base_wait_time = 1.0  # in seconds
        
        # Message and completion tracking
        self.messages = []
        self.model = None
        self.method = None
        self.temperature = 0.7
        self.max_tokens = None
        
        # Model mappings for easier reference
        self.model_mappings = {
            "chat": {
                "default": "gpt-4o",
                "fast": "gpt-4o-mini",
                "gpt4": "gpt-4",
                "gpt3": "gpt-3.5-turbo"
            },
            "embedding": {
                "default": "text-embedding-3-small",
                "large": "text-embedding-3-large"
            },
            "moderation": {
                "default": "text-moderation-latest"
            },
            "image": {
                "default": "dall-e-3"
            },
            "audio": {
                "transcription": "whisper-1",
                "translation": "whisper-1",
                "speech": {
                    "default": "tts-1",
                    "hd": "tts-1-hd"
                }
            }
        }
        
        logger.info("Enhanced OpenAI API initialized")
    
    def _init_metrics_collector(self) -> Dict[str, Any]:
        """Initialize metrics collector for performance tracking"""
        return {
            "requests": {
                "total": 0,
                "success": 0,
                "error": 0,
                "retries": 0
            },
            "latency": {
                "total_ms": 0,
                "count": 0,
                "min_ms": float('inf'),
                "max_ms": 0
            },
            "tokens": {
                "prompt": 0,
                "completion": 0,
                "total": 0
            },
            "models": {},
            "start_time": datetime.now(),
            "errors": {}
        }
    
    def _record_metrics(self, 
                       request_type: str, 
                       success: bool, 
                       latency_ms: float, 
                       model: str, 
                       tokens: Optional[Dict[str, int]] = None,
                       error: Optional[str] = None) -> None:
        """
        Record metrics for API requests
        
        Args:
            request_type: Type of request (chat, embedding, etc.)
            success: Whether the request was successful
            latency_ms: Request latency in milliseconds
            model: Model used for the request
            tokens: Token usage information
            error: Error message if the request failed
        """
        if not self.metrics_enabled or not self.metrics_collector:
            return
        
        # Update request counts
        self.metrics_collector["requests"]["total"] += 1
        if success:
            self.metrics_collector["requests"]["success"] += 1
        else:
            self.metrics_collector["requests"]["error"] += 1
            
        # Update latency metrics
        self.metrics_collector["latency"]["total_ms"] += latency_ms
        self.metrics_collector["latency"]["count"] += 1
        self.metrics_collector["latency"]["min_ms"] = min(self.metrics_collector["latency"]["min_ms"], latency_ms)
        self.metrics_collector["latency"]["max_ms"] = max(self.metrics_collector["latency"]["max_ms"], latency_ms)
        
        # Update token usage
        if tokens:
            self.metrics_collector["tokens"]["prompt"] += tokens.get("prompt_tokens", 0)
            self.metrics_collector["tokens"]["completion"] += tokens.get("completion_tokens", 0)
            self.metrics_collector["tokens"]["total"] += tokens.get("total_tokens", 0)
        
        # Update model usage
        if model not in self.metrics_collector["models"]:
            self.metrics_collector["models"][model] = {
                "requests": 0,
                "tokens": 0,
                "latency_ms": 0
            }
        
        self.metrics_collector["models"][model]["requests"] += 1
        self.metrics_collector["models"][model]["latency_ms"] += latency_ms
        if tokens:
            self.metrics_collector["models"][model]["tokens"] += tokens.get("total_tokens", 0)
        
        # Record errors
        if not success and error:
            error_type = error.split(":")[0] if ":" in error else error
            if error_type not in self.metrics_collector["errors"]:
                self.metrics_collector["errors"][error_type] = 0
            self.metrics_collector["errors"][error_type] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected performance metrics"""
        if not self.metrics_enabled or not self.metrics_collector:
            return {"metrics_enabled": False}
        
        # Calculate average latency
        avg_latency = 0
        if self.metrics_collector["latency"]["count"] > 0:
            avg_latency = self.metrics_collector["latency"]["total_ms"] / self.metrics_collector["latency"]["count"]
        
        # Calculate runtime
        runtime = (datetime.now() - self.metrics_collector["start_time"]).total_seconds()
        
        # Return the metrics
        return {
            "metrics_enabled": True,
            "requests": self.metrics_collector["requests"],
            "latency": {
                **self.metrics_collector["latency"],
                "avg_ms": avg_latency
            },
            "tokens": self.metrics_collector["tokens"],
            "models": self.metrics_collector["models"],
            "errors": self.metrics_collector["errors"],
            "runtime_seconds": runtime
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics"""
        if self.metrics_enabled:
            self.metrics_collector = self._init_metrics_collector()
    
    def _determine_model(self, request_type: str, model: Optional[str] = None) -> str:
        """
        Determine the appropriate model to use based on request type and user preference
        
        Args:
            request_type: Type of request (chat, embedding, etc)
            model: User-specified model or model type
            
        Returns:
            The appropriate model identifier
        """
        # If a specific model is provided, use it directly
        if model and not model in ["default", "fast", "large", "hd"]:
            return model
        
        # Handle special cases for model types
        if request_type in self.model_mappings:
            if request_type == "audio" and model in ["transcription", "translation", "speech"]:
                # Handle audio model types
                if model == "speech":
                    speech_type = "default" if not model or model == "speech" else model
                    return self.model_mappings["audio"]["speech"].get(speech_type, self.model_mappings["audio"]["speech"]["default"])
                else:
                    return self.model_mappings["audio"].get(model, self.model_mappings["audio"]["transcription"])
            elif model and request_type == "audio" and model in self.model_mappings["audio"]["speech"]:
                # Handle TTS model variants
                return self.model_mappings["audio"]["speech"][model]
            elif model and isinstance(self.model_mappings[request_type], dict) and model in self.model_mappings[request_type]:
                # Handle model variants like "default", "fast", etc.
                return self.model_mappings[request_type][model]
            elif isinstance(self.model_mappings[request_type], dict):
                # Use default for the request type
                return self.model_mappings[request_type]["default"]
            else:
                # Direct model reference
                return self.model_mappings[request_type]
        
        # Fallback to a suitable default
        if request_type == "chat":
            return "gpt-4o"
        elif request_type == "embedding":
            return "text-embedding-3-small"
        elif request_type == "transcription" or request_type == "translation":
            return "whisper-1"
        elif request_type == "speech":
            return "tts-1"
        else:
            return model or "gpt-4o"  # Default fallback
    
    def create_openai_api_endpoint_handler(self) -> Callable:
        """
        Create an endpoint handler function for the OpenAI API
        
        Returns:
            A function that can be used as an endpoint handler
        """
        def endpoint_handler(request_data: Dict[str, Any]) -> Dict[str, Any]:
            """
            Handle API requests with appropriate method
            
            Args:
                request_data: The request data containing method and parameters
                
            Returns:
                The API response
            """
            method = request_data.get("method", "")
            params = request_data.get("params", {})
            
            if method == "chat":
                return self.chat_completion(
                    model=params.get("model"),
                    messages=params.get("messages", []),
                    temperature=params.get("temperature", 0.7),
                    max_tokens=params.get("max_tokens"),
                    tools=params.get("tools"),
                    tool_choice=params.get("tool_choice")
                )
            elif method == "embedding":
                return self.embedding(
                    model=params.get("model"),
                    text=params.get("text", ""),
                    encoding_format=params.get("encoding_format", "float")
                )
            elif method == "moderation":
                return self.moderation(
                    model=params.get("model"),
                    text=params.get("text", "")
                )
            elif method == "text_to_image":
                return self.text_to_image(
                    model=params.get("model"),
                    size=params.get("size", "1024x1024"),
                    n=params.get("n", 1),
                    prompt=params.get("prompt", ""),
                    quality=params.get("quality", "standard"),
                    style=params.get("style", "vivid")
                )
            elif method == "text_to_speech":
                return self.text_to_speech(
                    model=params.get("model"),
                    text=params.get("text", ""),
                    voice=params.get("voice", "nova"),
                    speed=params.get("speed", 1.0),
                    response_format=params.get("response_format", "mp3")
                )
            elif method == "speech_to_text":
                return self.speech_to_text(
                    model=params.get("model"),
                    audio=params.get("audio"),
                    language=params.get("language"),
                    prompt=params.get("prompt"),
                    temperature=params.get("temperature", 0.0),
                    response_format=params.get("response_format", "json")
                )
            elif method == "audio_translation":
                return self.audio_translation(
                    model=params.get("model"),
                    audio=params.get("audio"),
                    prompt=params.get("prompt"),
                    temperature=params.get("temperature", 0.0),
                    response_format=params.get("response_format", "json")
                )
            elif method == "metrics":
                return self.get_metrics()
            else:
                return {
                    "error": f"Unsupported method: {method}",
                    "success": False
                }
        
        return endpoint_handler
    
    def embedding(self, 
                 model: Optional[str] = None, 
                 text: Union[str, List[str]] = "", 
                 encoding_format: str = "float",
                 user: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate text embeddings
        
        Args:
            model: The embedding model to use
            text: The text to generate embeddings for
            encoding_format: The format to return the embeddings in (float or base64)
            user: A unique identifier for the end-user
            
        Returns:
            Dictionary containing the embedding results
        """
        start_time = time.time()
        success = False
        error_msg = None
        token_usage = None
        
        try:
            if not self.client:
                raise ValueError("OpenAI client not initialized")
            
            # Determine which model to use
            embedding_model = self._determine_model("embedding", model)
            
            # Create embeddings request
            response = self.client.embeddings.create(
                model=embedding_model,
                input=text,
                encoding_format=encoding_format,
                user=user
            )
            
            # Extract embeddings from response
            embeddings = []
            if hasattr(response, "data") and len(response.data) > 0:
                embeddings = [item.embedding for item in response.data]
                
            # Record token usage
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Format results
            result = {
                "text": text,
                "model": embedding_model,
                "embedding": embeddings[0] if len(embeddings) == 1 else embeddings,
                "usage": token_usage
            }
            
            success = True
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Embedding error: {error_msg}")
            return {
                "error": error_msg,
                "success": False
            }
        finally:
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            model_used = embedding_model if 'embedding_model' in locals() else "unknown"
            self._record_metrics(
                request_type="embedding",
                success=success,
                latency_ms=latency_ms,
                model=model_used,
                tokens=token_usage,
                error=error_msg
            )
    
    def moderation(self, 
                  model: Optional[str] = None, 
                  text: str = "") -> Dict[str, Any]:
        """
        Check if text complies with OpenAI's content policy
        
        Args:
            model: The moderation model to use
            text: The text to moderate
            
        Returns:
            Moderation results
        """
        start_time = time.time()
        success = False
        error_msg = None
        
        try:
            if not self.client:
                raise ValueError("OpenAI client not initialized")
            
            # Determine which model to use
            moderation_model = self._determine_model("moderation", model)
            
            # Create moderation request
            response = self.client.moderations.create(
                model=moderation_model,
                input=text
            )
            
            success = True
            return response
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Moderation error: {error_msg}")
            return {
                "error": error_msg,
                "success": False
            }
        finally:
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            model_used = moderation_model if 'moderation_model' in locals() else "unknown"
            self._record_metrics(
                request_type="moderation",
                success=success,
                latency_ms=latency_ms,
                model=model_used,
                error=error_msg
            )
    
    def text_to_image(self, 
                     model: Optional[str] = None, 
                     size: str = "1024x1024", 
                     n: int = 1, 
                     prompt: str = "",
                     quality: str = "standard",
                     style: str = "vivid",
                     user: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate images from text prompts
        
        Args:
            model: The image generation model to use
            size: The size of the image to generate (1024x1024, 1024x1792, 1792x1024)
            n: The number of images to generate
            prompt: The prompt to generate images from
            quality: The quality of the image (standard or hd)
            style: The style of the image (vivid or natural)
            user: A unique identifier for the end-user
            
        Returns:
            Dictionary containing generated images
        """
        start_time = time.time()
        success = False
        error_msg = None
        
        try:
            if not self.client:
                raise ValueError("OpenAI client not initialized")
            
            # Determine which model to use
            image_model = self._determine_model("image", model)
            
            # Create image generation request
            response = self.client.images.generate(
                model=image_model,
                prompt=prompt,
                size=size,
                n=n,
                quality=quality,
                style=style,
                user=user
            )
            
            # Extract URLs and other data
            urls = [item.url for item in response.data]
            revised_prompts = [item.revised_prompt for item in response.data if hasattr(item, 'revised_prompt')]
            
            # Format results
            result = {
                "text": prompt,
                "urls": urls,
                "revised_prompts": revised_prompts if revised_prompts else None
            }
            
            success = True
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Text-to-Image error: {error_msg}")
            return {
                "error": error_msg,
                "success": False
            }
        finally:
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            model_used = image_model if 'image_model' in locals() else "unknown"
            self._record_metrics(
                request_type="text_to_image",
                success=success,
                latency_ms=latency_ms,
                model=model_used,
                error=error_msg
            )
    
    def text_to_speech(self, 
                      model: Optional[str] = None, 
                      text: str = "", 
                      voice: Union[str, VoiceType] = VoiceType.NOVA, 
                      speed: float = 1.0,
                      response_format: Union[str, AudioFormat] = AudioFormat.MP3) -> Dict[str, Any]:
        """
        Generate speech from text
        
        Args:
            model: The TTS model to use (tts-1 or tts-1-hd)
            text: The text to generate speech from
            voice: The voice to use (alloy, echo, fable, onyx, nova, shimmer)
            speed: The speed of the generated speech (0.25 to 4.0)
            response_format: The format of the audio (mp3, opus, aac, flac, wav)
            
        Returns:
            Dictionary containing generated audio
        """
        start_time = time.time()
        success = False
        error_msg = None
        
        try:
            if not self.client:
                raise ValueError("OpenAI client not initialized")
            
            # Validate parameters
            if isinstance(voice, VoiceType):
                voice = voice.value
            
            if isinstance(response_format, AudioFormat):
                response_format = response_format.value
            
            # Determine which model to use
            speech_model = self._determine_model("speech", model)
            
            # Create text-to-speech request
            audio_response = self.client.audio.speech.create(
                model=speech_model,
                voice=voice,
                input=text,
                speed=speed,
                response_format=response_format
            )
            
            # Get the audio data
            audio_data = audio_response.content
            
            # Format results
            result = {
                "text": text,
                "audio": base64.b64encode(audio_data).decode('utf-8'),
                "format": response_format,
                "voice": voice
            }
            
            success = True
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Text-to-Speech error: {error_msg}")
            return {
                "error": error_msg,
                "success": False
            }
        finally:
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            model_used = speech_model if 'speech_model' in locals() else "unknown"
            self._record_metrics(
                request_type="text_to_speech",
                success=success,
                latency_ms=latency_ms,
                model=model_used,
                error=error_msg
            )
    
    def speech_to_text(self, 
                      model: Optional[str] = None, 
                      audio: BinaryIO = None,
                      language: Optional[str] = None,
                      prompt: Optional[str] = None,
                      temperature: float = 0.0,
                      response_format: str = "json") -> Dict[str, Any]:
        """
        Transcribe audio to text
        
        Args:
            model: The transcription model to use
            audio: The audio file to transcribe
            language: The language of the audio
            prompt: Optional text to guide the model's transcription
            temperature: Sampling temperature
            response_format: The format to return the transcription in
            
        Returns:
            Dictionary containing the transcription
        """
        start_time = time.time()
        success = False
        error_msg = None
        
        try:
            if not self.client:
                raise ValueError("OpenAI client not initialized")
            
            if not audio:
                raise ValueError("Audio data is required")
            
            # Determine which model to use
            transcription_model = self._determine_model("transcription", model)
            
            # Handle audio file
            audio_file = audio
            
            # Create transcription request
            transcription = self.client.audio.transcriptions.create(
                model=transcription_model,
                file=audio_file,
                language=language,
                prompt=prompt,
                temperature=temperature,
                response_format=response_format
            )
            
            # Format results
            result = {
                "text": transcription.text,
                "language": language,
                "duration": getattr(transcription, "duration", None)
            }
            
            success = True
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Speech-to-Text error: {error_msg}")
            return {
                "error": error_msg,
                "success": False
            }
        finally:
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            model_used = transcription_model if 'transcription_model' in locals() else "unknown"
            self._record_metrics(
                request_type="speech_to_text",
                success=success,
                latency_ms=latency_ms,
                model=model_used,
                error=error_msg
            )
    
    def audio_translation(self, 
                         model: Optional[str] = None, 
                         audio: BinaryIO = None,
                         prompt: Optional[str] = None,
                         temperature: float = 0.0,
                         response_format: str = "json") -> Dict[str, Any]:
        """
        Translate audio to English text
        
        Args:
            model: The translation model to use
            audio: The audio file to translate
            prompt: Optional text to guide the model's translation
            temperature: Sampling temperature
            response_format: The format to return the translation in
            
        Returns:
            Dictionary containing the translation
        """
        start_time = time.time()
        success = False
        error_msg = None
        
        try:
            if not self.client:
                raise ValueError("OpenAI client not initialized")
            
            if not audio:
                raise ValueError("Audio data is required")
            
            # Determine which model to use
            translation_model = self._determine_model("translation", model)
            
            # Handle audio file
            audio_file = audio
            
            # Create translation request
            translation = self.client.audio.translations.create(
                model=translation_model,
                file=audio_file,
                prompt=prompt,
                temperature=temperature,
                response_format=response_format
            )
            
            # Format results
            result = {
                "text": translation.text
            }
            
            success = True
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Audio Translation error: {error_msg}")
            return {
                "error": error_msg,
                "success": False
            }
        finally:
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            model_used = translation_model if 'translation_model' in locals() else "unknown"
            self._record_metrics(
                request_type="audio_translation",
                success=success,
                latency_ms=latency_ms,
                model=model_used,
                error=error_msg
            )
    
    def process_messages(self, 
                        messages: List[Dict[str, Any]], 
                        system_message: Optional[str] = None,
                        instruction: Optional[str] = None,
                        default_system: str = "You are a helpful assistant") -> List[Dict[str, Any]]:
        """
        Process messages to ensure proper format and add system message if needed
        
        Args:
            messages: List of message dictionaries
            system_message: Optional system message to use
            instruction: Optional instruction to add to system message
            default_system: Default system message if none provided
            
        Returns:
            Processed messages list
        """
        processed_messages = []
        
        # Check if we already have a system message
        has_system = any(msg.get("role") == "system" for msg in messages)
        
        # Add system message if needed
        if not has_system and (system_message or default_system):
            system_content = system_message or default_system
            if instruction:
                system_content = f"{system_content}\n\n{instruction}"
            
            processed_messages.append({
                "role": "system",
                "content": system_content
            })
        
        # Add the rest of the messages
        processed_messages.extend(messages)
        
        return processed_messages
    
    def chat_completion(self, 
                       model: Optional[str] = None, 
                       messages: List[Dict[str, Any]] = None, 
                       temperature: float = 0.7, 
                       max_tokens: Optional[int] = None,
                       tools: Optional[List[Dict[str, Any]]] = None,
                       tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
                       system_message: Optional[str] = None,
                       seed: Optional[int] = None,
                       stream: bool = False,
                       user: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a chat completion
        
        Args:
            model: The model to use
            messages: The messages to generate a completion for
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            tools: List of tools the model may call
            tool_choice: Controls which (if any) tool is called by the model
            system_message: System message to use (if not in messages)
            seed: Optional seed for deterministic generation
            stream: Whether to stream the response
            user: A unique identifier for the end-user
            
        Returns:
            Dictionary containing the completion
        """
        start_time = time.time()
        success = False
        error_msg = None
        token_usage = None
        
        try:
            if not self.client:
                raise ValueError("OpenAI client not initialized")
            
            if not messages:
                raise ValueError("Messages are required")
            
            # Process messages to ensure proper format
            processed_messages = self.process_messages(messages, system_message)
            
            # Determine which model to use
            chat_model = self._determine_model("chat", model)
            
            # Create chat completion request
            response = self.client.chat.completions.create(
                model=chat_model,
                messages=processed_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                seed=seed,
                stream=stream,
                user=user
            )
            
            if stream:
                # Return stream iterator
                return {
                    "text": "[STREAM]",
                    "stream": response,
                    "success": True
                }
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Extract token usage
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Extract tool calls if present
            tool_calls = response.choices[0].message.tool_calls if hasattr(response.choices[0].message, 'tool_calls') else None
            
            # Format results
            result = {
                "text": content,
                "model": chat_model,
                "usage": token_usage,
                "finish_reason": response.choices[0].finish_reason,
                "tool_calls": tool_calls
            }
            
            success = True
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Chat Completion error: {error_msg}")
            return {
                "error": error_msg,
                "success": False
            }
        finally:
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            model_used = chat_model if 'chat_model' in locals() else "unknown"
            self._record_metrics(
                request_type="chat",
                success=success,
                latency_ms=latency_ms,
                model=model_used,
                tokens=token_usage,
                error=error_msg
            )
    
    # === Queue and Retry Management ===
    
    def _exponential_backoff(self, attempt: int, max_delay: int = 60) -> float:
        """
        Calculate exponential backoff delay
        
        Args:
            attempt: The current attempt number
            max_delay: Maximum delay in seconds
            
        Returns:
            Delay time in seconds
        """
        # Add some jitter to avoid thundering herd
        jitter = random.uniform(0.8, 1.2)
        delay = min(max_delay, self.base_wait_time * (2 ** attempt)) * jitter
        return delay
    
    def _handle_rate_limit(self, e: Exception, attempt: int) -> Tuple[bool, float]:
        """
        Handle rate limit errors
        
        Args:
            e: The exception raised
            attempt: The current attempt number
            
        Returns:
            Tuple of (should_retry, delay_seconds)
        """
        # Check if we should retry based on the error
        should_retry = False
        delay = 0
        
        # Extract retry-after header if available
        retry_after = None
        if hasattr(e, 'headers') and 'retry-after' in e.headers:
            retry_after = float(e.headers['retry-after'])
        
        # Check error type
        error_string = str(e).lower()
        if "rate limit" in error_string or "rate_limit" in error_string:
            should_retry = True
            delay = retry_after if retry_after else self._exponential_backoff(attempt)
        elif "overloaded" in error_string or "capacity" in error_string:
            should_retry = True
            delay = self._exponential_backoff(attempt, max_delay=120)  # Longer backoff for server overload
        
        return should_retry, delay
    
    def _add_to_queue(self, future: Dict[str, Any]) -> None:
        """
        Add a request to the queue
        
        Args:
            future: The future request dictionary
        """
        with self.queue_lock:
            self.request_queue.append({
                "future": future,
                "added_time": time.time()
            })
            
            # Start processing queue if not already running
            if not hasattr(self, '_queue_thread') or not getattr(self, '_queue_thread').is_alive():
                self._queue_thread = threading.Thread(target=self._process_queue)
                self._queue_thread.daemon = True
                self._queue_thread.start()
    
    def _process_queue(self) -> None:
        """Process queued requests when capacity becomes available"""
        while True:
            with self.queue_lock:
                # Check if there are items in the queue
                if not self.request_queue:
                    return
                
                # Check if we can process more requests
                if self.current_requests >= self.max_concurrent_requests:
                    time.sleep(0.1)  # Small sleep to avoid busy waiting
                    continue
                
                # Get the next request
                request = self.request_queue.pop(0)
                self.current_requests += 1
            
            # Process the request
            try:
                future = request["future"]
                
                # Set up the request based on the method
                method = future.get("method")
                if method == "chat":
                    result = self.chat_completion(
                        model=future.get("model"),
                        messages=future.get("messages"),
                        temperature=future.get("temperature"),
                        max_tokens=future.get("max_tokens"),
                        tools=future.get("tools"),
                        tool_choice=future.get("tool_choice")
                    )
                elif method == "embedding":
                    result = self.embedding(
                        model=future.get("model"),
                        text=future.get("text"),
                        encoding_format=future.get("encoding_format")
                    )
                else:
                    result = {"error": f"Unsupported method: {method}"}
                
                # Update the future with the result
                future["result"] = result
                future["completed"] = True
                future["error"] = result.get("error")
                
            except Exception as e:
                logger.error(f"Error processing queued request: {e}")
                # Update the future with the error
                future["error"] = str(e)
                future["completed"] = True
            finally:
                # Decrement counter
                with self.queue_lock:
                    self.current_requests -= 1
    
    def request_complete(self, 
                        method: Optional[str] = None, 
                        model: Optional[str] = None, 
                        wait: bool = True,
                        timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Submit a request for completion
        
        Args:
            method: The method to use (chat, embedding, etc.), defaults to self.method
            model: The model to use
            wait: Whether to wait for the request to complete
            timeout: Maximum time to wait for the request
            
        Returns:
            The completion result
        """
        # Determine the method to use
        request_method = method or self.method
        if not request_method:
            return {"error": "No method specified"}
        
        # Determine the model to use
        request_model = model or self.model
        
        # Create a future object for the request
        future = {
            "method": request_method,
            "model": request_model,
            "completed": False,
            "result": None,
            "error": None
        }
        
        # Add method-specific parameters
        if request_method == "chat":
            future["messages"] = self.messages
            future["temperature"] = self.temperature
            future["max_tokens"] = self.max_tokens
        elif request_method == "embedding":
            future["text"] = self.messages[0]["content"] if self.messages else ""
            future["encoding_format"] = "float"
        
        # Check if we need to queue the request
        if self.current_requests >= self.max_concurrent_requests:
            # Queue the request
            self._add_to_queue(future)
            
            if not wait:
                return {"status": "queued", "success": True}
            
            # Wait for the request to complete
            start_time = time.time()
            while not future["completed"]:
                time.sleep(0.1)
                
                # Check for timeout
                if timeout and time.time() - start_time > timeout:
                    return {"error": "Request timed out", "success": False}
            
            return future["result"]
        
        # We have capacity, process the request directly
        try:
            self.current_requests += 1
            
            # Retry logic
            for attempt in range(self.max_retries):
                try:
                    # Process the request based on the method
                    if request_method == "chat":
                        result = self.chat_completion(
                            model=request_model,
                            messages=self.messages,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens
                        )
                    elif request_method == "embedding":
                        text = self.messages[0]["content"] if self.messages else ""
                        result = self.embedding(
                            model=request_model,
                            text=text
                        )
                    else:
                        result = {"error": f"Unsupported method: {request_method}"}
                    
                    return result
                
                except Exception as e:
                    # Check if we should retry
                    should_retry, delay = self._handle_rate_limit(e, attempt)
                    
                    if should_retry and attempt < self.max_retries - 1:
                        # Increment retry counter for metrics
                        if self.metrics_enabled and self.metrics_collector:
                            self.metrics_collector["requests"]["retries"] += 1
                        
                        # Log retry attempt
                        logger.warning(f"Rate limit hit, retrying in {delay:.2f}s (attempt {attempt+1}/{self.max_retries})")
                        
                        # Wait before retrying
                        time.sleep(delay)
                    else:
                        # Last attempt failed or non-retryable error
                        logger.error(f"Request failed after {attempt+1} attempts: {e}")
                        raise
            
        except Exception as e:
            return {"error": str(e), "success": False}
        finally:
            self.current_requests -= 1
    
    # === Voice Agents Support ===
    
    def create_voice_agent(self,
                          messages: List[Dict[str, Any]] = None,
                          system_message: Optional[str] = None,
                          voice: Union[str, VoiceType] = VoiceType.NOVA,
                          model: Optional[str] = None,
                          temperature: float = 0.7,
                          max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a voice agent for text-to-speech and speech-to-text conversations
        
        Args:
            messages: Initial conversation messages
            system_message: System message for the agent
            voice: Voice to use for text-to-speech
            model: Model to use for conversation
            temperature: Sampling temperature
            max_tokens: Maximum tokens for responses
            
        Returns:
            Dictionary with voice agent configuration
        """
        # Initialize conversation if provided
        conversation = []
        if messages:
            conversation = self.process_messages(messages, system_message)
        elif system_message:
            conversation = [{"role": "system", "content": system_message}]
        
        # Set up agent configuration
        agent_config = {
            "id": f"agent_{int(time.time())}",
            "voice": voice.value if isinstance(voice, VoiceType) else voice,
            "model": self._determine_model("chat", model),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "conversation": conversation,
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "agent": agent_config
        }
    
    def voice_agent_process_speech(self,
                                 agent_config: Dict[str, Any],
                                 audio: BinaryIO = None,
                                 language: Optional[str] = None) -> Dict[str, Any]:
        """
        Process speech input for a voice agent and get spoken response
        
        Args:
            agent_config: Voice agent configuration
            audio: Audio input to process
            language: Language of the audio input
            
        Returns:
            Dictionary with transcription, response text, and audio response
        """
        start_time = time.time()
        
        try:
            if not self.client:
                raise ValueError("OpenAI client not initialized")
            
            if not audio:
                raise ValueError("Audio input is required")
            
            # Transcribe the audio
            transcription = self.speech_to_text(
                model="whisper-1",
                audio=audio,
                language=language
            )
            
            if "error" in transcription:
                return transcription
            
            # Add user message to conversation
            agent_config["conversation"].append({
                "role": "user",
                "content": transcription["text"]
            })
            
            # Get AI response
            response = self.chat_completion(
                model=agent_config["model"],
                messages=agent_config["conversation"],
                temperature=agent_config["temperature"],
                max_tokens=agent_config["max_tokens"]
            )
            
            if "error" in response:
                return response
            
            # Add assistant response to conversation
            agent_config["conversation"].append({
                "role": "assistant",
                "content": response["text"]
            })
            
            # Convert response to speech
            speech_response = self.text_to_speech(
                model="tts-1-hd",
                text=response["text"],
                voice=agent_config["voice"]
            )
            
            if "error" in speech_response:
                return speech_response
            
            # Calculate total processing time
            total_time_ms = (time.time() - start_time) * 1000
            
            # Return combined results
            return {
                "success": True,
                "transcription": transcription["text"],
                "response": response["text"],
                "audio": speech_response["audio"],
                "audio_format": speech_response["format"],
                "processing_time_ms": total_time_ms,
                "agent": agent_config
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Voice agent processing error: {error_msg}")
            return {
                "error": error_msg,
                "success": False
            }
    
    # === Integration with Distributed Testing Framework ===
    
    def get_backend_status(self) -> Dict[str, Any]:
        """
        Get status information about the OpenAI API backend
        
        Returns:
            Dictionary with backend status
        """
        try:
            # Check if client is initialized
            client_ok = self.client is not None
            
            # Get metrics for status
            metrics = self.get_metrics()
            
            # Format backend status
            status = {
                "name": "openai",
                "initialized": client_ok,
                "healthy": client_ok,
                "api_key_configured": bool(self.api_key),
                "organization_configured": bool(self.organization),
                "metrics": metrics,
                "current_requests": self.current_requests,
                "max_concurrent_requests": self.max_concurrent_requests,
                "queue_size": len(self.request_queue),
                "timestamp": datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting backend status: {e}")
            return {
                "name": "openai",
                "initialized": False,
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_health_check(self) -> Dict[str, Any]:
        """
        Run a simple health check by making a test API call
        
        Returns:
            Dictionary with health check results
        """
        try:
            # Run a simple embedding request as health check
            test_result = self.embedding(
                model="text-embedding-3-small",
                text="Health check test"
            )
            
            # Check if the test was successful
            is_healthy = "error" not in test_result
            
            # Format health check results
            health_check = {
                "name": "openai",
                "status": "healthy" if is_healthy else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "latency_ms": test_result.get("latency_ms") if is_healthy else None,
                "error": test_result.get("error") if not is_healthy else None
            }
            
            return health_check
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "name": "openai",
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def update_distributed_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update settings for distributed testing integration
        
        Args:
            settings: Dictionary with settings to update
            
        Returns:
            Dictionary with updated settings
        """
        try:
            # Update max concurrent requests
            if "max_concurrent_requests" in settings:
                self.max_concurrent_requests = int(settings["max_concurrent_requests"])
            
            # Update metrics enabled
            if "metrics_enabled" in settings:
                self.metrics_enabled = bool(settings["metrics_enabled"])
                if self.metrics_enabled and not self.metrics_collector:
                    self.metrics_collector = self._init_metrics_collector()
                elif not self.metrics_enabled:
                    self.metrics_collector = None
            
            # Update distributed mode
            if "distributed_mode" in settings:
                self.distributed_mode = bool(settings["distributed_mode"])
            
            # Update max retries
            if "max_retries" in settings:
                self.max_retries = int(settings["max_retries"])
            
            # Update base wait time
            if "base_wait_time" in settings:
                self.base_wait_time = float(settings["base_wait_time"])
            
            # Return updated settings
            return {
                "success": True,
                "settings": {
                    "max_concurrent_requests": self.max_concurrent_requests,
                    "metrics_enabled": self.metrics_enabled,
                    "distributed_mode": self.distributed_mode,
                    "max_retries": self.max_retries,
                    "base_wait_time": self.base_wait_time
                }
            }
            
        except Exception as e:
            logger.error(f"Error updating distributed settings: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# === Example Usage ===

def main():
    """Example usage of the OpenAI API"""
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No OpenAI API key found in environment variables.")
        print("Please set OPENAI_API_KEY in your environment.")
        return
    
    # Initialize the OpenAI API
    openai_api = OpenAIAPI(api_key=api_key, metrics_enabled=True)
    
    # Example 1: Chat completion
    print("\nExample 1: Chat completion with GPT-4o")
    response = openai_api.chat_completion(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the 3 most important trends in AI in 2025?"}
        ],
        temperature=0.7
    )
    
    if "error" not in response:
        print(f"Response: {response['text']}")
        print(f"Token usage: {response['usage']}")
    else:
        print(f"Error: {response['error']}")
    
    # Example 2: Text embedding
    print("\nExample 2: Text embedding")
    response = openai_api.embedding(
        model="text-embedding-3-small",
        text="Embedding example"
    )
    
    if "error" not in response:
        print(f"Embedding dimensions: {len(response['embedding'])}")
        print(f"Token usage: {response['usage']}")
    else:
        print(f"Error: {response['error']}")
    
    # Example 3: View metrics
    print("\nExample 3: View metrics")
    metrics = openai_api.get_metrics()
    print(f"Total requests: {metrics['requests']['total']}")
    print(f"Average latency: {metrics['latency']['avg_ms']:.2f}ms")
    print(f"Total tokens: {metrics['tokens']['total']}")
    
    print("\nDone")

if __name__ == "__main__":
    main()