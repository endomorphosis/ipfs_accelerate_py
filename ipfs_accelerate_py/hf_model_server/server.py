"""
Main FastAPI server application for unified HF model server
"""

import time
import uuid
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import ServerConfig
from .registry import SkillRegistry
from .hardware import HardwareDetector, HardwareSelector
from .monitoring.metrics import PrometheusMetrics
from .api.schemas import (
    CompletionRequest, CompletionResponse, CompletionChoice, CompletionUsage,
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice,
    EmbeddingRequest, EmbeddingResponse, EmbeddingData,
    ModelListResponse, ModelInfo,
    LoadModelRequest, LoadModelResponse,
    UnloadModelRequest, UnloadModelResponse,
    ServerStatus, ErrorResponse, ErrorDetail,
    ChatMessage
)

# Import WebSocket handler
try:
    from .websocket_handler import get_connection_manager, WebSocketInferenceHandler
    HAVE_WEBSOCKET = True
except ImportError:
    HAVE_WEBSOCKET = False
    logger.warning("WebSocket handler not available")

logger = logging.getLogger(__name__)


class HFModelServer:
    """
    Unified HuggingFace Model Server
    
    Features:
    - OpenAI-compatible API
    - Automatic skill discovery
    - Intelligent hardware selection
    - Multi-model serving
    - Request batching and caching
    - Health checks and metrics
    """
    
    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.start_time = time.time()
        
        # Core components
        self.skill_registry: Optional[SkillRegistry] = None
        self.hardware_detector: Optional[HardwareDetector] = None
        self.hardware_selector: Optional[HardwareSelector] = None
        
        # Metrics
        self.metrics: Optional[PrometheusMetrics] = None
        if self.config.enable_metrics:
            self.metrics = PrometheusMetrics()
            logger.info("Prometheus metrics enabled")
        
        # WebSocket components
        self.connection_manager = None
        self.websocket_handler = None
        if HAVE_WEBSOCKET:
            self.connection_manager = get_connection_manager()
            self.websocket_handler = WebSocketInferenceHandler(self.connection_manager)
        
        # FastAPI app
        self.app = FastAPI(
            title="Unified HuggingFace Model Server",
            description="OpenAI-compatible API for HuggingFace models with WebSocket support",
            version="0.1.0",
            lifespan=self.lifespan
        )
        
        # Setup
        self._setup_middleware()
        self._setup_routes()
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Lifespan context manager for startup/shutdown"""
        # Startup
        await self.startup()
        yield
        # Shutdown
        await self.shutdown()
    
    async def startup(self):
        """Initialize server on startup"""
        logger.info("Starting HF Model Server...")
        
        # Initialize hardware detection
        if self.config.enable_hardware_detection:
            self.hardware_detector = HardwareDetector()
            self.hardware_selector = HardwareSelector(self.hardware_detector)
            logger.info(f"Hardware available: {self.hardware_detector.get_available_hardware()}")
        
        # Initialize skill registry
        self.skill_registry = SkillRegistry(
            skill_directories=self.config.skill_directories,
            skill_pattern=self.config.skill_pattern
        )
        
        # Discover skills
        if self.config.auto_discover:
            count = await self.skill_registry.discover_skills()
            logger.info(f"Discovered {count} skills")
        
        logger.info("HF Model Server started successfully")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Shutting down HF Model Server...")
        # Cleanup resources here
        logger.info("HF Model Server shutdown complete")
    
    def _setup_middleware(self):
        """Setup middleware"""
        # CORS
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health checks
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy"}
        
        @self.app.get("/ready")
        async def readiness_check():
            """Readiness check endpoint"""
            if self.skill_registry is None:
                raise HTTPException(status_code=503, detail="Server not ready")
            return {"status": "ready"}
        
        # Metrics endpoint
        if self.metrics:
            @self.app.get("/metrics")
            async def get_metrics():
                """Prometheus metrics endpoint"""
                return Response(
                    content=self.metrics.generate_metrics(),
                    media_type=self.metrics.get_content_type()
                )
        
        # Server status
        @self.app.get("/status", response_model=ServerStatus)
        async def get_status():
            """Get server status"""
            return ServerStatus(
                status="running",
                version="0.1.0",
                models_loaded=0,  # TODO: Track loaded models
                models_available=len(self.skill_registry.list_models()) if self.skill_registry else 0,
                hardware_available=self.hardware_detector.get_available_hardware() if self.hardware_detector else [],
                uptime_seconds=time.time() - self.start_time
            )
        
        # OpenAI-compatible endpoints
        @self.app.post("/v1/completions", response_model=CompletionResponse)
        async def create_completion(request: CompletionRequest):
            """OpenAI-compatible completions endpoint"""
            # TODO: Implement actual completion logic
            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    CompletionChoice(
                        text="This is a placeholder response",
                        index=0,
                        finish_reason="length"
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=len(str(request.prompt).split()),
                    completion_tokens=5,
                    total_tokens=len(str(request.prompt).split()) + 5
                )
            )
        
        @self.app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def create_chat_completion(request: ChatCompletionRequest):
            """OpenAI-compatible chat completions endpoint"""
            # TODO: Implement actual chat completion logic
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content="This is a placeholder response"
                        ),
                        finish_reason="stop"
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=sum(len(m.content.split()) for m in request.messages),
                    completion_tokens=5,
                    total_tokens=sum(len(m.content.split()) for m in request.messages) + 5
                )
            )
        
        @self.app.post("/v1/embeddings", response_model=EmbeddingResponse)
        async def create_embedding(request: EmbeddingRequest):
            """OpenAI-compatible embeddings endpoint"""
            # TODO: Implement actual embedding logic
            inputs = [request.input] if isinstance(request.input, str) else request.input
            return EmbeddingResponse(
                data=[
                    EmbeddingData(
                        embedding=[0.0] * 768,  # Placeholder
                        index=i
                    )
                    for i in range(len(inputs))
                ],
                model=request.model,
                usage=CompletionUsage(
                    prompt_tokens=sum(len(text.split()) for text in inputs),
                    completion_tokens=0,
                    total_tokens=sum(len(text.split()) for text in inputs)
                )
            )
        
        @self.app.get("/v1/models", response_model=ModelListResponse)
        async def list_models():
            """List available models"""
            if not self.skill_registry:
                return ModelListResponse(data=[])
            
            models = []
            for model_id in self.skill_registry.list_models():
                models.append(ModelInfo(
                    id=model_id,
                    created=int(time.time()),
                    owned_by="hf-model-server"
                ))
            
            return ModelListResponse(data=models)
        
        # Extended model management endpoints
        @self.app.post("/models/load", response_model=LoadModelResponse)
        async def load_model(request: LoadModelRequest):
            """Load a model"""
            # TODO: Implement actual model loading
            return LoadModelResponse(
                model_id=request.model_id,
                status="loaded",
                hardware=request.hardware or "cpu",
                message=f"Model {request.model_id} loaded successfully"
            )
        
        @self.app.post("/models/unload", response_model=UnloadModelResponse)
        async def unload_model(request: UnloadModelRequest):
            """Unload a model"""
            # TODO: Implement actual model unloading
            return UnloadModelResponse(
                model_id=request.model_id,
                status="unloaded",
                message=f"Model {request.model_id} unloaded successfully"
            )
        
        # WebSocket endpoint
        if HAVE_WEBSOCKET and self.websocket_handler:
            @self.app.websocket("/ws/{client_id}")
            async def websocket_endpoint(websocket: WebSocket, client_id: str):
                """WebSocket endpoint for real-time inference and monitoring"""
                await self.websocket_handler.handle_client(websocket, client_id)
    
    def run(self):
        """Run the server"""
        import uvicorn
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level=self.config.log_level.lower()
        )


def create_server(config: Optional[ServerConfig] = None) -> HFModelServer:
    """Factory function to create server instance"""
    return HFModelServer(config)


# For running with uvicorn directly
app = HFModelServer().app
