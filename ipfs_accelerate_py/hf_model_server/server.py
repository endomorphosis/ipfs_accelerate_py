"""
Main FastAPI server application for unified HF model server
"""

import time
import uuid
import hashlib
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket, Response, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager

from .config import ServerConfig
from .registry import SkillRegistry
from .hardware import HardwareDetector, HardwareSelector
from .monitoring.metrics import PrometheusMetrics
from .auth.api_keys import APIKeyManager, APIKey
from .auth.middleware import AuthMiddleware
from .auth.rate_limiter import RateLimiter
from .middleware.request_queue import QueueManager, RequestPriority
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

# Security
security = HTTPBearer(auto_error=False)

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
        
        # Phase 5: Authentication & Authorization
        self.api_key_manager: Optional[APIKeyManager] = None
        self.auth_middleware: Optional[AuthMiddleware] = None
        if self.config.enable_auth:
            self.api_key_manager = APIKeyManager()
            self.auth_middleware = AuthMiddleware(
                api_key_manager=self.api_key_manager,
                enabled=self.config.require_auth
            )
            logger.info("Authentication enabled")
            
            # Generate admin key if provided
            if self.config.admin_api_key:
                # Pre-load admin key (in production, load from secure storage)
                self.api_key_manager._keys[
                    hashlib.sha256(self.config.admin_api_key.encode()).hexdigest()
                ] = APIKey(
                    key_id=hashlib.sha256(self.config.admin_api_key.encode()).hexdigest()[:16],
                    key_hash=hashlib.sha256(self.config.admin_api_key.encode()).hexdigest(),
                    name="admin",
                    rate_limit=1000,
                    is_active=True
                )
        
        # Phase 5: Rate Limiting
        self.rate_limiter: Optional[RateLimiter] = None
        if self.config.enable_rate_limiting:
            self.rate_limiter = RateLimiter(enabled=True)
            logger.info("Rate limiting enabled")
        
        # Phase 5: Request Queuing
        self.queue_manager: Optional[QueueManager] = None
        if self.config.enable_request_queue:
            self.queue_manager = QueueManager(
                default_max_size=self.config.max_queue_size,
                default_timeout=self.config.queue_timeout_seconds
            )
            logger.info("Request queuing enabled")
        
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
    
    async def _verify_auth(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> Optional[APIKey]:
        """Dependency for verifying authentication."""
        if not self.auth_middleware or not self.config.enable_auth:
            return None
        
        try:
            api_key = await self.auth_middleware.verify_request(request, credentials)
            return api_key
        except HTTPException:
            if self.config.require_auth:
                raise
            return None
    
    async def _check_rate_limit(
        self,
        request: Request,
        api_key: Optional[APIKey] = None
    ) -> None:
        """Check rate limit for request."""
        if not self.rate_limiter or not self.config.enable_rate_limiting:
            return
        
        if not api_key:
            # Use default rate limit for unauthenticated requests
            # Create a temporary API key for rate limiting
            api_key = APIKey(
                key_id="anonymous",
                key_hash="anonymous",
                name="Anonymous",
                rate_limit=self.config.default_rate_limit
            )
        
        allowed, remaining = await self.rate_limiter.check_limit(api_key)
        
        # Add rate limit headers
        rate_limit_headers = self.rate_limiter.get_headers(api_key)
        
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers=rate_limit_headers
            )
        
        # Store headers for response
        request.state.rate_limit_headers = rate_limit_headers
    
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
        
        # Phase 5: API Key Management Endpoints
        if self.api_key_manager:
            @self.app.post("/admin/keys/generate")
            async def generate_api_key(
                name: str,
                rate_limit: int = 100,
                request: Request = None,
                api_key: Optional[APIKey] = Depends(self._verify_auth)
            ):
                """Generate new API key (admin only)."""
                # Verify admin access
                if not api_key or api_key.name != "admin":
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Admin access required"
                    )
                
                key_string, key_obj = self.api_key_manager.generate_key(
                    name=name,
                    rate_limit=rate_limit
                )
                
                return {
                    "api_key": key_string,
                    "key_id": key_obj.key_id,
                    "name": key_obj.name,
                    "rate_limit": key_obj.rate_limit,
                    "created_at": key_obj.created_at.isoformat()
                }
            
            @self.app.get("/admin/keys/list")
            async def list_api_keys(
                request: Request = None,
                api_key: Optional[APIKey] = Depends(self._verify_auth)
            ):
                """List all API keys (admin only)."""
                if not api_key or api_key.name != "admin":
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Admin access required"
                    )
                
                keys = self.api_key_manager.list_keys()
                return {
                    "keys": [
                        {
                            "key_id": k.key_id,
                            "name": k.name,
                            "rate_limit": k.rate_limit,
                            "created_at": k.created_at.isoformat(),
                            "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                            "is_active": k.is_active
                        }
                        for k in keys
                    ]
                }
            
            @self.app.post("/admin/keys/{key_id}/revoke")
            async def revoke_api_key(
                key_id: str,
                request: Request = None,
                api_key: Optional[APIKey] = Depends(self._verify_auth)
            ):
                """Revoke API key (admin only)."""
                if not api_key or api_key.name != "admin":
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Admin access required"
                    )
                
                success = self.api_key_manager.revoke_key(key_id)
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="API key not found"
                    )
                
                return {"success": True, "message": f"API key {key_id} revoked"}
        
        # Phase 5: Queue Statistics Endpoint
        if self.queue_manager:
            @self.app.get("/admin/queue/stats")
            async def get_queue_stats(
                request: Request = None,
                api_key: Optional[APIKey] = Depends(self._verify_auth)
            ):
                """Get request queue statistics."""
                stats = await self.queue_manager.get_stats()
                return stats
        
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
