"""
OpenAI-compatible API schemas

Pydantic models for request/response matching OpenAI API format
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field


# ============================================================================
# Completions API
# ============================================================================

class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request"""
    model: str = Field(..., description="Model to use for completion")
    prompt: Union[str, List[str]] = Field(..., description="Prompt(s) to generate from")
    max_tokens: int = Field(default=16, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, ge=0, le=2, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0, le=1, description="Nucleus sampling parameter")
    n: int = Field(default=1, description="Number of completions to generate")
    stream: bool = Field(default=False, description="Stream partial results")
    logprobs: Optional[int] = Field(default=None, description="Include log probabilities")
    echo: bool = Field(default=False, description="Echo the prompt in response")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Stop sequences")
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    user: Optional[str] = Field(default=None, description="User identifier")


class CompletionChoice(BaseModel):
    """Single completion choice"""
    text: str
    index: int
    logprobs: Optional[Dict] = None
    finish_reason: str


class CompletionUsage(BaseModel):
    """Token usage information"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response"""
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


# ============================================================================
# Chat Completions API
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message"""
    role: Literal["system", "user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(default=None, description="Name of the message sender")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""
    model: str = Field(..., description="Model to use")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    n: int = Field(default=1, description="Number of completions")
    stream: bool = Field(default=False, description="Stream results")
    stop: Optional[Union[str, List[str]]] = Field(default=None)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    user: Optional[str] = Field(default=None)


class ChatCompletionChoice(BaseModel):
    """Chat completion choice"""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage


# ============================================================================
# Embeddings API
# ============================================================================

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    model: str = Field(..., description="Model to use for embeddings")
    input: Union[str, List[str]] = Field(..., description="Text(s) to embed")
    user: Optional[str] = Field(default=None)


class EmbeddingData(BaseModel):
    """Single embedding"""
    object: Literal["embedding"] = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    object: Literal["list"] = "list"
    data: List[EmbeddingData]
    model: str
    usage: CompletionUsage


# ============================================================================
# Models API
# ============================================================================

class ModelInfo(BaseModel):
    """Model information"""
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "hf-model-server"
    permission: List[Dict] = Field(default_factory=list)
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelListResponse(BaseModel):
    """List of available models"""
    object: Literal["list"] = "list"
    data: List[ModelInfo]


# ============================================================================
# Error Responses
# ============================================================================

class ErrorDetail(BaseModel):
    """Error detail"""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """API error response"""
    error: ErrorDetail


# ============================================================================
# Extended Model Management (Non-OpenAI)
# ============================================================================

class LoadModelRequest(BaseModel):
    """Load a model"""
    model_id: str = Field(..., description="Model ID to load")
    hardware: Optional[str] = Field(default=None, description="Specific hardware to use")
    options: Dict[str, Any] = Field(default_factory=dict, description="Model loading options")


class LoadModelResponse(BaseModel):
    """Model load response"""
    model_id: str
    status: str
    hardware: str
    message: str


class UnloadModelRequest(BaseModel):
    """Unload a model"""
    model_id: str = Field(..., description="Model ID to unload")


class UnloadModelResponse(BaseModel):
    """Model unload response"""
    model_id: str
    status: str
    message: str


class ServerStatus(BaseModel):
    """Server status information"""
    status: str
    version: str
    models_loaded: int
    models_available: int
    hardware_available: List[str]
    uptime_seconds: float
