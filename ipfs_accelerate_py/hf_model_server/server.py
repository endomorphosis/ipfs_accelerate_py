"""
Main FastAPI server application for unified HF model server
"""

import time
import uuid
import hashlib
import json
import logging
from typing import Optional, Any, Dict, List
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

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

try:
    from ..inference_backend_manager import get_backend_manager
    HAVE_BACKEND_MANAGER = True
except Exception:
    HAVE_BACKEND_MANAGER = False
    get_backend_manager = None

try:
    from ..ipfs_kit_integration import get_storage
    HAVE_IPFS_KIT_STORAGE = True
except Exception:
    HAVE_IPFS_KIT_STORAGE = False
    get_storage = None

try:
    from ..datasets_integration import DatasetsManager
    HAVE_DATASETS_MANAGER = True
except Exception:
    HAVE_DATASETS_MANAGER = False
    DatasetsManager = None

try:
    from ..model_manager import ModelManager, ModelMetadata, ModelType, IOSpec, DataType
    HAVE_MODEL_MANAGER = True
except Exception:
    HAVE_MODEL_MANAGER = False
    ModelManager = None
    ModelMetadata = None
    ModelType = None
    IOSpec = None
    DataType = None

# Import WebSocket handler
try:
    from .websocket_handler import get_connection_manager, WebSocketInferenceHandler
    HAVE_WEBSOCKET = True
except ImportError:
    HAVE_WEBSOCKET = False
    logger.warning("WebSocket handler not available")


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
        self.backend_manager = None
        self._storage_client: Optional[Any] = None
        self._datasets_manager: Optional[Any] = None
        self._model_manager: Optional[Any] = None
        self._loaded_models: set[str] = set()
        
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

    def _get_storage_client(self) -> Optional[Any]:
        if self._storage_client is not None:
            return self._storage_client
        if not HAVE_IPFS_KIT_STORAGE or get_storage is None:
            return None
        try:
            self._storage_client = get_storage(enable_ipfs_kit=True)
        except Exception as exc:
            logger.debug(f"HFModelServer storage initialization failed: {exc}")
            self._storage_client = None
        return self._storage_client

    def _get_datasets_manager(self) -> Optional[Any]:
        if self._datasets_manager is not None:
            return self._datasets_manager
        if not HAVE_DATASETS_MANAGER or DatasetsManager is None:
            return None
        try:
            self._datasets_manager = DatasetsManager({
                "enable_audit": True,
                "enable_provenance": True,
            })
        except Exception as exc:
            logger.debug(f"HFModelServer datasets initialization failed: {exc}")
            self._datasets_manager = None
        return self._datasets_manager

    def _record_inference_result(
        self,
        *,
        model: str,
        inputs: List[Any],
        result: Dict[str, Any],
        backend_id: Optional[str] = None,
        backend_type: Optional[str] = None,
        endpoint: Optional[str] = None,
        device: Optional[str] = None,
        protocol: Optional[str] = None,
        protocols: Optional[List[str]] = None,
        hardware_type: Optional[str] = None,
        hardware_types: Optional[List[str]] = None,
        placement_node: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "input_cid": None,
            "output_cid": None,
            "provenance_cid": None,
            "audit_logged": False,
        }

        storage = self._get_storage_client()
        if storage is not None:
            try:
                input_payload = json.dumps({"model": model, "inputs": inputs}, sort_keys=True)
                output_payload = json.dumps(result, sort_keys=True)
                metadata["input_cid"] = storage.store(input_payload, filename=f"{model}_input.json")
                metadata["output_cid"] = storage.store(output_payload, filename=f"{model}_output.json")
            except Exception as exc:
                logger.debug(f"HFModelServer storage persistence failed: {exc}")

        manager = self._get_datasets_manager()
        if manager is not None:
            payload = {
                "model": model,
                "backend_id": backend_id,
                "backend_type": backend_type,
                "endpoint": endpoint,
                "protocol": protocol,
                "protocols": list(protocols or []),
                "hardware_type": hardware_type,
                "hardware_types": list(hardware_types or []),
                "placement_node": placement_node,
                "device": device or result.get("device"),
                "input_cid": metadata["input_cid"],
                "output_cid": metadata["output_cid"],
                "duration_ms": float(result.get("processing_time", 0.0)) * 1000.0,
                "status": "completed",
            }
            try:
                metadata["audit_logged"] = bool(
                    manager.log_event("inference_completed", payload, category="PERFORMANCE")
                )
            except Exception as exc:
                logger.debug(f"HFModelServer audit logging failed: {exc}")
            try:
                metadata["provenance_cid"] = manager.track_provenance("inference", payload)
            except Exception as exc:
                logger.debug(f"HFModelServer provenance tracking failed: {exc}")

        merged = dict(result)
        merged.update(metadata)
        return merged

    def _record_inference_failure(
        self,
        *,
        model: str,
        inputs: List[Any],
        error: str,
        backend_id: Optional[str] = None,
        backend_type: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "input_cid": None,
            "output_cid": None,
            "provenance_cid": None,
            "audit_logged": False,
        }

        storage = self._get_storage_client()
        if storage is not None:
            try:
                input_payload = json.dumps({"model": model, "inputs": inputs}, sort_keys=True)
                failure_payload = json.dumps({"model": model, "error": str(error)}, sort_keys=True)
                metadata["input_cid"] = storage.store(input_payload, filename=f"{model}_input.json")
                metadata["output_cid"] = storage.store(failure_payload, filename=f"{model}_failure.json")
            except Exception as exc:
                logger.debug(f"HFModelServer failure persistence failed: {exc}")

        manager = self._get_datasets_manager()
        if manager is not None:
            payload = {
                "model": model,
                "backend_id": backend_id,
                "backend_type": backend_type,
                "endpoint": endpoint,
                "input_cid": metadata["input_cid"],
                "output_cid": metadata["output_cid"],
                "status": "failed",
                "error": str(error),
            }
            try:
                metadata["audit_logged"] = bool(
                    manager.log_event("inference_failed", payload, level="ERROR", category="PERFORMANCE")
                )
            except Exception as exc:
                logger.debug(f"HFModelServer failure audit logging failed: {exc}")
            try:
                metadata["provenance_cid"] = manager.track_provenance("inference_failed", payload)
            except Exception as exc:
                logger.debug(f"HFModelServer failure provenance tracking failed: {exc}")

        return metadata

    def _record_model_lifecycle_failure(self, *, event_type: str, payload: Dict[str, Any]) -> None:
        manager = self._get_datasets_manager()
        if manager is None:
            return
        try:
            manager.log_event(event_type, payload, level="ERROR", category="GENERAL")
        except Exception:
            pass
        try:
            manager.track_provenance(event_type, payload)
        except Exception:
            pass

    def _get_model_manager(self) -> Optional[Any]:
        if self._model_manager is not None:
            return self._model_manager
        if not HAVE_MODEL_MANAGER or ModelManager is None:
            return None
        try:
            self._model_manager = ModelManager()
        except Exception as exc:
            logger.debug(f"HFModelServer model manager initialization failed: {exc}")
            self._model_manager = None
        return self._model_manager

    def _build_model_metadata(self, request: LoadModelRequest) -> Any:
        options = dict(request.options or {})

        model_type_token = str(options.get("model_type") or "language_model").strip().lower()
        model_type = ModelType.LANGUAGE_MODEL
        if model_type_token in {"embedding", "embedding_model"}:
            model_type = ModelType.EMBEDDING_MODEL
        elif model_type_token in {"vision", "vision_model"}:
            model_type = ModelType.VISION_MODEL
        elif model_type_token in {"audio", "audio_model"}:
            model_type = ModelType.AUDIO_MODEL
        elif model_type_token in {"multimodal"}:
            model_type = ModelType.MULTIMODAL

        architecture = str(options.get("architecture") or request.model_id)
        input_spec = IOSpec(name="input", data_type=DataType.TEXT, optional=False)
        output_data_type = DataType.TEXT if model_type != ModelType.EMBEDDING_MODEL else DataType.EMBEDDINGS
        output_spec = IOSpec(name="output", data_type=output_data_type, optional=False)

        metadata = ModelMetadata(
            model_id=request.model_id,
            model_name=str(options.get("model_name") or request.model_id),
            model_type=model_type,
            architecture=architecture,
            inputs=[input_spec],
            outputs=[output_spec],
            description=str(options.get("description") or ""),
            tags=[str(tag) for tag in list(options.get("tags") or [])],
            supported_backends=[str(request.hardware or "cpu")],
            source_url=str(options.get("source_url") or "") or None,
            license=str(options.get("license") or "") or None,
            model_revision=str(options.get("model_revision") or "") or None,
        )
        return metadata

    def _extract_generated_text(self, result: Dict[str, Any]) -> str:
        outputs = result.get("outputs")
        if isinstance(outputs, list) and outputs:
            return str(outputs[0])
        if isinstance(outputs, str):
            return outputs
        if isinstance(result.get("result"), dict):
            nested = result.get("result")
            nested_outputs = nested.get("outputs")
            if isinstance(nested_outputs, list) and nested_outputs:
                return str(nested_outputs[0])
            if isinstance(nested_outputs, str):
                return nested_outputs
            if nested.get("text") is not None:
                return str(nested.get("text"))
        if result.get("text") is not None:
            return str(result.get("text"))
        return ""

    def _track_model_usage(self, *, model: str, result: Dict[str, Any], run_id: str) -> None:
        model_manager = self._get_model_manager()
        if model_manager is None or not hasattr(model_manager, "mark_model_used"):
            return
        try:
            model_manager.mark_model_used(
                model,
                inference_cid=str(result.get("output_cid") or "") or None,
                run_id=run_id,
            )
        except Exception as exc:
            logger.debug(f"HFModelServer model usage tracking failed: {exc}")
    
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

        if HAVE_BACKEND_MANAGER and get_backend_manager is not None:
            try:
                self.backend_manager = get_backend_manager()
                self.backend_manager._result_recorder = self._record_inference_result
            except Exception as exc:
                logger.warning(f"Backend manager initialization failed: {exc}")
                self.backend_manager = None
        
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
                models_loaded=len(self._loaded_models),
                models_available=len(self.skill_registry.list_models()) if self.skill_registry else 0,
                hardware_available=self.hardware_detector.get_available_hardware() if self.hardware_detector else [],
                uptime_seconds=time.time() - self.start_time
            )
        
        # OpenAI-compatible endpoints
        @self.app.post("/v1/completions", response_model=CompletionResponse)
        async def create_completion(request: CompletionRequest):
            """OpenAI-compatible completions endpoint"""
            if self.backend_manager is None:
                raise HTTPException(status_code=503, detail="Backend manager unavailable")

            prompt_value = request.prompt[0] if isinstance(request.prompt, list) else request.prompt
            try:
                backend_result = await self.backend_manager.execute_task(
                    task="text-generation",
                    model=request.model,
                    inputs=[str(prompt_value)],
                    parameters={
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "stop": request.stop,
                        "n": request.n,
                        "stream": request.stream,
                    },
                )
            except Exception as exc:
                self._record_inference_failure(
                    model=request.model,
                    inputs=[str(prompt_value)],
                    error=str(exc),
                )
                raise HTTPException(status_code=503, detail=f"Inference failed: {exc}") from exc

            text = self._extract_generated_text(backend_result)
            prompt_tokens = len(str(prompt_value).split())
            completion_tokens = len(text.split())
            completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"

            self._track_model_usage(model=request.model, result=backend_result, run_id=completion_id)

            return CompletionResponse(
                id=completion_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    CompletionChoice(
                        text=text,
                        index=0,
                        finish_reason="stop"
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
        
        @self.app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def create_chat_completion(request: ChatCompletionRequest):
            """OpenAI-compatible chat completions endpoint"""
            if self.backend_manager is None:
                raise HTTPException(status_code=503, detail="Backend manager unavailable")

            prompt = "\n".join([f"{message.role}: {message.content}" for message in request.messages])
            try:
                backend_result = await self.backend_manager.execute_task(
                    task="text-generation",
                    model=request.model,
                    inputs=[prompt],
                    parameters={
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "stop": request.stop,
                        "n": request.n,
                        "stream": request.stream,
                    },
                )
            except Exception as exc:
                self._record_inference_failure(
                    model=request.model,
                    inputs=[prompt],
                    error=str(exc),
                )
                raise HTTPException(status_code=503, detail=f"Inference failed: {exc}") from exc

            response_text = self._extract_generated_text(backend_result)
            prompt_tokens = sum(len(m.content.split()) for m in request.messages)
            completion_tokens = len(response_text.split())
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

            self._track_model_usage(model=request.model, result=backend_result, run_id=completion_id)

            return ChatCompletionResponse(
                id=completion_id,
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=response_text
                        ),
                        finish_reason="stop"
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
        
        @self.app.post("/v1/embeddings", response_model=EmbeddingResponse)
        async def create_embedding(request: EmbeddingRequest):
            """OpenAI-compatible embeddings endpoint"""
            if self.backend_manager is None:
                raise HTTPException(status_code=503, detail="Backend manager unavailable")

            inputs = [request.input] if isinstance(request.input, str) else request.input

            try:
                backend_result = await self.backend_manager.execute_task(
                    task="text-embedding",
                    model=request.model,
                    inputs=[str(x) for x in inputs],
                    parameters={},
                )
            except Exception as exc:
                self._record_inference_failure(
                    model=request.model,
                    inputs=[str(x) for x in inputs],
                    error=str(exc),
                )
                raise HTTPException(status_code=503, detail=f"Embedding inference failed: {exc}") from exc

            embeddings_data = backend_result.get("embeddings")
            if not isinstance(embeddings_data, list):
                embeddings_data = []

            self._track_model_usage(model=request.model, result=backend_result, run_id=f"embed-{uuid.uuid4().hex[:8]}")

            return EmbeddingResponse(
                data=[
                    EmbeddingData(
                        embedding=[float(x) for x in (embeddings_data[i] if i < len(embeddings_data) else [])],
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
            model_manager = self._get_model_manager()
            if model_manager is None:
                raise HTTPException(status_code=503, detail="Model manager unavailable")

            options = dict(request.options or {})
            metadata = self._build_model_metadata(request)
            model_path = options.get("model_path")
            config_path = options.get("config_path")
            tokenizer_path = options.get("tokenizer_path")
            store_to_ipfs = bool(options.get("store_to_ipfs", True))

            success, artifact_cid = model_manager.add_model_with_ipfs_storage(
                metadata,
                model_path=model_path,
                config_path=config_path,
                tokenizer_path=tokenizer_path,
                store_to_ipfs=store_to_ipfs,
            )
            if not success:
                self._record_model_lifecycle_failure(
                    event_type="model_load_failed",
                    payload={
                        "model_id": request.model_id,
                        "hardware": request.hardware or "cpu",
                        "options": dict(request.options or {}),
                        "error": f"Failed to register model {request.model_id}",
                    },
                )
                raise HTTPException(status_code=500, detail=f"Failed to register model {request.model_id}")

            self._loaded_models.add(request.model_id)
            manager = self._get_datasets_manager()
            provenance_cid = None
            if manager is not None:
                try:
                    payload = {
                        "model_id": request.model_id,
                        "hardware": request.hardware or "cpu",
                        "options": dict(request.options or {}),
                        "model_cid": metadata.model_cid,
                        "config_cid": metadata.config_cid,
                        "tokenizer_cid": metadata.tokenizer_cid,
                        "artifact_cid": artifact_cid,
                        "status": "loaded",
                    }
                    manager.log_event("model_loaded", payload, category="GENERAL")
                    provenance_cid = manager.track_provenance("model_load", payload)
                except Exception:
                    pass
            return LoadModelResponse(
                model_id=request.model_id,
                status="loaded",
                hardware=request.hardware or "cpu",
                message=f"Model {request.model_id} loaded successfully (artifact_cid={artifact_cid})",
                artifact_cid=artifact_cid,
                model_cid=metadata.model_cid,
                config_cid=metadata.config_cid,
                tokenizer_cid=metadata.tokenizer_cid,
                provenance_cid=provenance_cid,
            )
        
        @self.app.post("/models/unload", response_model=UnloadModelResponse)
        async def unload_model(request: UnloadModelRequest):
            """Unload a model"""
            model_manager = self._get_model_manager()
            if model_manager is None:
                raise HTTPException(status_code=503, detail="Model manager unavailable")

            existing = None
            if hasattr(model_manager, "get_model"):
                try:
                    existing = model_manager.get_model(request.model_id)
                except Exception:
                    existing = None

            removed = model_manager.remove_model(request.model_id)
            if not removed:
                self._record_model_lifecycle_failure(
                    event_type="model_unload_failed",
                    payload={
                        "model_id": request.model_id,
                        "error": f"Model not found: {request.model_id}",
                    },
                )
                raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")

            self._loaded_models.discard(request.model_id)
            manager = self._get_datasets_manager()
            provenance_cid = None
            if manager is not None:
                try:
                    payload = {
                        "model_id": request.model_id,
                        "model_cid": getattr(existing, "model_cid", None),
                        "config_cid": getattr(existing, "config_cid", None),
                        "tokenizer_cid": getattr(existing, "tokenizer_cid", None),
                        "artifact_cid": getattr(existing, "artifact_cid", None),
                        "status": "unloaded",
                    }
                    manager.log_event("model_unloaded", payload, category="GENERAL")
                    provenance_cid = manager.track_provenance("model_unload", payload)
                except Exception:
                    pass
            return UnloadModelResponse(
                model_id=request.model_id,
                status="unloaded",
                message=f"Model {request.model_id} unloaded successfully",
                artifact_cid=getattr(existing, "artifact_cid", None),
                model_cid=getattr(existing, "model_cid", None),
                config_cid=getattr(existing, "config_cid", None),
                tokenizer_cid=getattr(existing, "tokenizer_cid", None),
                provenance_cid=provenance_cid,
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
