"""
IPFS Accelerate Python package.

This package provides a framework for hardware-accelerated machine learning inference
with IPFS network-based distribution and acceleration. Key features include:

- Hardware acceleration (CPU, GPU, OpenVINO, WebNN, WebGPU)
- IPFS-based content distribution and caching
- Browser integration for client-side inference
- Model type detection and optimization
- Cross-platform support
"""

import importlib
import os
import sys
import threading
from pathlib import Path
from types import ModuleType

SKIP_CORE = os.environ.get("IPFS_ACCEL_SKIP_CORE", "0") == "1"

# HF Space transport symbols historically imported requests at package root.
# Keep them lazy so Planner/Doctor cold discovery never loads network clients.
_HF_SPACE_EXPORT_NAMES = frozenset(
    {
        "BatchProcessor",
        "BatchState",
        "EndpointContract",
        "HFBucketBackend",
        "HFBucketBackendError",
        "HFSpaceClient",
        "LocalFileSystemBackend",
        "OutputBackend",
        "RefreshableGradioFile",
        "SpaceRuntimeInfo",
        "is_hf_space_transport_error",
        "is_retryable_hf_space_error",
        "is_stale_gradio_file_error",
        "normalize_api_name",
    }
)
_hf_space_exports = None
_hf_space_lock = threading.Lock()


def _load_hf_space_exports():
    """Resolve HF Space symbols on first explicit access only."""

    global _hf_space_exports
    with _hf_space_lock:
        if _hf_space_exports is not None:
            return _hf_space_exports
        from .hf_space_inference import (
            BatchProcessor,
            BatchState,
            EndpointContract,
            HFBucketBackend,
            HFBucketBackendError,
            HFSpaceClient,
            LocalFileSystemBackend,
            OutputBackend,
            RefreshableGradioFile,
            SpaceRuntimeInfo,
            is_hf_space_transport_error,
            is_retryable_hf_space_error,
            is_stale_gradio_file_error,
            normalize_api_name,
        )

        resolved = {
            "BatchProcessor": BatchProcessor,
            "BatchState": BatchState,
            "EndpointContract": EndpointContract,
            "HFBucketBackend": HFBucketBackend,
            "HFBucketBackendError": HFBucketBackendError,
            "HFSpaceClient": HFSpaceClient,
            "LocalFileSystemBackend": LocalFileSystemBackend,
            "OutputBackend": OutputBackend,
            "RefreshableGradioFile": RefreshableGradioFile,
            "SpaceRuntimeInfo": SpaceRuntimeInfo,
            "is_hf_space_transport_error": is_hf_space_transport_error,
            "is_retryable_hf_space_error": is_retryable_hf_space_error,
            "is_stale_gradio_file_error": is_stale_gradio_file_error,
            "normalize_api_name": normalize_api_name,
        }
        for name, value in resolved.items():
            globals()[name] = value
        existing_export = globals().get("export")
        if isinstance(existing_export, dict):
            for name, value in resolved.items():
                dict.__setitem__(existing_export, name, value)
        _hf_space_exports = resolved
        return resolved


# Import original components (skip heavy backends under cold/skip-core profiles).
if not SKIP_CORE:
    try:
        from .container_backends import backends
    except Exception:
        backends = None
else:
    backends = None

if not SKIP_CORE:
    try:
        from .install_depends import install_depends
    except Exception:
        install_depends = None
else:
    install_depends = None

def _add_external_package(package_name: str) -> None:
    """Ensure external bundled packages are importable without pip install."""
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "external" / package_name
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Optionally skip importing the heavy core (avoids ipfs_kit_py import at import-time)
if not SKIP_CORE:
    try:
        _add_external_package("ipfs_kit_py")
        _add_external_package("ipfs_model_manager_py")
        _add_external_package("ipfs_transformers_py")
        from .ipfs_accelerate import ipfs_accelerate_py as original_ipfs_accelerate_py
    except Exception:
        original_ipfs_accelerate_py = None
else:
    original_ipfs_accelerate_py = None

if not SKIP_CORE:
    try:
        from .ipfs_multiformats import ipfs_multiformats_py
    except Exception:
        ipfs_multiformats_py = None

    # ``worker`` is resolved by the module-level ``__getattr__`` below.  Keep
    # package discovery light: importing the legacy worker package eagerly
    # loads every model skillset (including torch and transformers).

    try:
        from .config import config
    except Exception:
        config = None
else:
    ipfs_multiformats_py = None
    worker = None
    config = None

# Import WebNN/WebGPU integration (skip when core is disabled)
if not SKIP_CORE:
    try:
        from .webnn_webgpu_integration import (
            accelerate_with_browser,
            WebNNWebGPUAccelerator,
            get_accelerator
        )
        webnn_webgpu_available = True
    except Exception:
        webnn_webgpu_available = False
        
        # Create stubs if not available
        def accelerate_with_browser(*args, **kwargs):
            raise NotImplementedError("WebNN/WebGPU integration is not available")
        
        def get_accelerator(*args, **kwargs):
            raise NotImplementedError("WebNN/WebGPU integration is not available")
        
        class WebNNWebGPUAccelerator:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("WebNN/WebGPU integration is not available")
else:
    webnn_webgpu_available = False
    def accelerate_with_browser(*args, **kwargs):
        raise NotImplementedError("WebNN/WebGPU integration is disabled (IPFS_ACCEL_SKIP_CORE=1)")
    def get_accelerator(*args, **kwargs):
        raise NotImplementedError("WebNN/WebGPU integration is disabled (IPFS_ACCEL_SKIP_CORE=1)")
    class WebNNWebGPUAccelerator:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("WebNN/WebGPU integration is disabled (IPFS_ACCEL_SKIP_CORE=1)")

# Import Model Manager (skip by default to avoid heavy optional deps at import time)
if os.environ.get("IPFS_ACCEL_IMPORT_EAGER", "0") == "1":
    try:
        from .model_manager import (
            ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
            ServingConfig, create_model_from_huggingface, get_default_model_manager
        )
        model_manager_available = True
    except Exception:
        model_manager_available = False
        def get_default_model_manager(*args, **kwargs):
            raise NotImplementedError("Model Manager is not available")
        class ModelManager:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("Model Manager is not available")
else:
    model_manager_available = True

    def _lazy_import_model_manager():
        try:
            from .model_manager import (
                ModelManager as _RealModelManager,
                ModelMetadata as _RealModelMetadata,
                IOSpec as _RealIOSpec,
                ModelType as _RealModelType,
                DataType as _RealDataType,
                ServingConfig as _RealServingConfig,
                create_model_from_huggingface as _create_model_from_huggingface,
                get_default_model_manager as _real_get_default_model_manager,
            )
            return {
                "ModelManager": _RealModelManager,
                "ModelMetadata": _RealModelMetadata,
                "IOSpec": _RealIOSpec,
                "ModelType": _RealModelType,
                "DataType": _RealDataType,
                "ServingConfig": _RealServingConfig,
                "create_model_from_huggingface": _create_model_from_huggingface,
                "get_default_model_manager": _real_get_default_model_manager,
            }
        except Exception as e:
            raise NotImplementedError(f"Model Manager is not available: {e}") from e

    def get_default_model_manager(*args, **kwargs):
        exports = _lazy_import_model_manager()
        return exports["get_default_model_manager"](*args, **kwargs)

    class ModelManager:
        def __new__(cls, *args, **kwargs):
            exports = _lazy_import_model_manager()
            return exports["ModelManager"](*args, **kwargs)

_global_instance = None

# Public constructor/entrypoint (may be unavailable when core is disabled or missing deps)
if original_ipfs_accelerate_py is not None:
    ipfs_accelerate_py = original_ipfs_accelerate_py

    def get_instance(**kwargs):
        """Get or create a process-wide singleton instance of ipfs_accelerate_py.

        Accepts optional dependency injections (e.g., ``deps``, ``ipfs_kit``) and
        forwards them to the constructor on first creation.
        """
        global _global_instance
        if _global_instance is None:
            _global_instance = ipfs_accelerate_py(**kwargs)
        elif kwargs:
            # Best-effort: attach injected deps to existing singleton.
            for k, v in kwargs.items():
                try:
                    setattr(_global_instance, k, v)
                except Exception:
                    pass
        return _global_instance
else:
    def ipfs_accelerate_py(*args, **kwargs):
        raise NotImplementedError(
            "IPFS Accelerate core is not available (missing deps) or disabled. "
            "Set IPFS_ACCEL_SKIP_CORE=0 and install core dependencies to enable."
        )

    def get_instance():
        raise NotImplementedError(
            "IPFS Accelerate core is not available (missing deps) or disabled. "
            "Set IPFS_ACCEL_SKIP_CORE=0 and install core dependencies to enable."
        )

_WORKER_UNRESOLVED = object()
_worker_export_value = None if SKIP_CORE else _WORKER_UNRESOLVED
_worker_loading = False
_worker_loader_thread_id = None
_worker_condition = threading.Condition()


class _LazyWorkerSnapshot(ModuleType):
    """Module-like value used only by raw, non-virtual dictionary copies."""

    def __getattr__(self, name):
        resolved = _load_legacy_worker()
        if resolved is None:
            raise AttributeError(
                "legacy worker is unavailable because optional dependencies "
                "could not be imported"
            )
        return getattr(resolved, name)

    def __dir__(self):
        resolved = _load_legacy_worker()
        return [] if resolved is None else dir(resolved)


_worker_snapshot = _LazyWorkerSnapshot(
    f"{__name__}.worker",
    "Lazy compatibility view of the legacy worker module.",
)


def _load_legacy_worker():
    """Resolve the historical module-valued worker export on explicit access."""

    global _worker_export_value, _worker_loading, _worker_loader_thread_id
    current_thread_id = threading.get_ident()
    with _worker_condition:
        if _worker_export_value is not _WORKER_UNRESOLVED:
            return _worker_export_value
        if _worker_loading and _worker_loader_thread_id == current_thread_id:
            # Same-thread recursion can occur while importlib installs the
            # worker package on this parent module.
            return globals().get("worker")
        while _worker_loading:
            _worker_condition.wait()
            if _worker_export_value is not _WORKER_UNRESOLVED:
                return _worker_export_value
        _worker_loading = True
        _worker_loader_thread_id = current_thread_id
    try:
        try:
            resolved = importlib.import_module(f"{__name__}.worker.worker")
        except Exception:
            resolved = None
        with _worker_condition:
            _worker_export_value = resolved
            globals()["worker"] = resolved
            existing_export = globals().get("export")
            if isinstance(existing_export, dict):
                dict.__setitem__(existing_export, "worker", resolved)
            return resolved
    finally:
        with _worker_condition:
            _worker_loading = False
            _worker_loader_thread_id = None
            _worker_condition.notify_all()


class _LazyRootExport(dict):
    """Dictionary-compatible exports with optional lazy legacy values.

    Virtual mapping access resolves ``worker`` and HF Space transport symbols.
    Raw base-dict inspection keeps a module-like lazy worker snapshot so
    provider-free discovery remains possible without exposing a misleading
    permanent ``None`` value, and without loading network clients.
    """

    def __contains__(self, key):
        if key in _HF_SPACE_EXPORT_NAMES:
            return True
        return dict.__contains__(self, key)

    def __getitem__(self, key):
        if key == "worker":
            with _worker_condition:
                if not dict.__contains__(self, key):
                    raise KeyError(key)
                stored = dict.__getitem__(self, key)
                if stored is not _worker_snapshot:
                    return stored
            return _load_legacy_worker()
        if key in _HF_SPACE_EXPORT_NAMES:
            if dict.__contains__(self, key):
                return dict.__getitem__(self, key)
            return _load_hf_space_exports()[key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key != "worker":
            return super().__setitem__(key, value)
        # Route public-dictionary monkeypatches through the same synchronized
        # assignment contract as ``package.worker = value``.
        setattr(sys.modules[__name__], "worker", value)

    def __delitem__(self, key):
        global _worker_export_value
        if key != "worker":
            return super().__delitem__(key)
        with _worker_condition:
            while (
                _worker_loading
                and _worker_loader_thread_id != threading.get_ident()
            ):
                _worker_condition.wait()
            if not dict.__contains__(self, key):
                raise KeyError(key)
            _worker_export_value = (
                None if SKIP_CORE else _WORKER_UNRESOLVED
            )
            globals().pop("worker", None)
            dict.__delitem__(self, key)
            _worker_condition.notify_all()

    def get(self, key, default=None):
        if key == "worker" and key in self:
            return self[key]
        if key in _HF_SPACE_EXPORT_NAMES:
            return self[key]
        return super().get(key, default)

    def setdefault(self, key, default=None):
        if key != "worker":
            return super().setdefault(key, default)
        with _worker_condition:
            if dict.__contains__(self, key):
                # Preserve the raw lazy snapshot behavior of the historical
                # dict API without causing provider imports.
                return dict.__getitem__(self, key)
            self[key] = default
            return default

    def update(self, *args, **kwargs):
        staged = {}
        dict.update(staged, *args, **kwargs)
        for key, value in staged.items():
            self[key] = value

    def __ior__(self, other):
        self.update(other)
        return self

    def pop(self, key, *default):
        if len(default) > 1:
            raise TypeError(f"pop expected at most 2 arguments, got {1 + len(default)}")
        if key != "worker":
            return super().pop(key, *default)
        with _worker_condition:
            if not dict.__contains__(self, key):
                if default:
                    return default[0]
                raise KeyError(key)
            value = dict.__getitem__(self, key)
            self.__delitem__(key)
            return value

    def popitem(self):
        if not self:
            raise KeyError("popitem(): dictionary is empty")
        key = next(reversed(self))
        return key, self.pop(key)

    def clear(self):
        global _worker_export_value
        with _worker_condition:
            while (
                _worker_loading
                and _worker_loader_thread_id != threading.get_ident()
            ):
                _worker_condition.wait()
            had_worker = dict.__contains__(self, "worker")
            dict.clear(self)
            if had_worker:
                _worker_export_value = (
                    None if SKIP_CORE else _WORKER_UNRESOLVED
                )
                globals().pop("worker", None)
            _worker_condition.notify_all()

    def items(self):
        if "worker" in self:
            self["worker"]
        return super().items()

    def values(self):
        if "worker" in self:
            self["worker"]
        return super().values()

    def copy(self):
        if "worker" in self:
            self["worker"]
        return super().copy()


# Export all components. HF Space transport symbols resolve on first access so
# cold Planner/Doctor imports never load requests/httpx through package root.
export = _LazyRootExport({
    "backends": backends,
    "config": config,
    "install_depends": install_depends,
    "ipfs_accelerate_py": ipfs_accelerate_py,
    # Synchronized with the historical root export on first explicit access.
    "worker": None if SKIP_CORE else _worker_snapshot,
    "ipfs_multiformats_py": ipfs_multiformats_py,
    "get_instance": get_instance,
    "accelerate_with_browser": accelerate_with_browser,
    "WebNNWebGPUAccelerator": WebNNWebGPUAccelerator,
    "get_accelerator": get_accelerator,
    "webnn_webgpu_available": webnn_webgpu_available,
    "ModelManager": ModelManager,
    "get_default_model_manager": get_default_model_manager,
    "model_manager_available": model_manager_available,
})


def __getattr__(name):
    """Lazily expose optional legacy package-root components."""

    if name == "worker":
        return _load_legacy_worker()
    if name in _HF_SPACE_EXPORT_NAMES:
        return _load_hf_space_exports()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

if not SKIP_CORE:
    # Add CLI entry point for package access
    def cli_main(*args, **kwargs):
        from .cli_entry import main as _cli_main

        return _cli_main(*args, **kwargs)

    export["cli_main"] = cli_main

    # Add system logs access
    try:
        from .logs import get_system_logs, SystemLogs

        export["get_system_logs"] = get_system_logs
        export["SystemLogs"] = SystemLogs
    except ImportError:
        get_system_logs = None
        SystemLogs = None

    # Add P2P workflow scheduler access
    try:
        from .p2p_workflow_scheduler import (
            P2PWorkflowScheduler,
            P2PTask,
            WorkflowTag,
            MerkleClock,
            FibonacciHeap,
            calculate_hamming_distance,
        )

        export["P2PWorkflowScheduler"] = P2PWorkflowScheduler
        export["P2PTask"] = P2PTask
        export["WorkflowTag"] = WorkflowTag
        export["MerkleClock"] = MerkleClock
        export["FibonacciHeap"] = FibonacciHeap
        export["calculate_hamming_distance"] = calculate_hamming_distance
    except ImportError:
        P2PWorkflowScheduler = None
        P2PTask = None
        WorkflowTag = None
        MerkleClock = None
        FibonacciHeap = None
        calculate_hamming_distance = None
else:
    cli_main = None
    get_system_logs = None
    SystemLogs = None
    P2PWorkflowScheduler = None
    P2PTask = None
    WorkflowTag = None
    MerkleClock = None
    FibonacciHeap = None
    calculate_hamming_distance = None

# Add IPFS Kit integration
if not SKIP_CORE:
    try:
        from .ipfs_kit_integration import (
            IPFSKitStorage,
            get_storage,
            reset_storage,
            StorageBackendConfig,
        )

        export["IPFSKitStorage"] = IPFSKitStorage
        export["get_storage"] = get_storage
        export["reset_storage"] = reset_storage
        export["StorageBackendConfig"] = StorageBackendConfig
    except ImportError:
        IPFSKitStorage = None
        get_storage = None
        reset_storage = None
        StorageBackendConfig = None
else:
    IPFSKitStorage = None
    get_storage = None
    reset_storage = None
    StorageBackendConfig = None

# Add inference backend manager
if not SKIP_CORE:
    try:
        from .inference_backend_manager import (
            InferenceBackendManager,
            get_backend_manager,
            register_backend_from_config,
        )

        export["InferenceBackendManager"] = InferenceBackendManager
        export["get_backend_manager"] = get_backend_manager
        export["register_backend_from_config"] = register_backend_from_config
        inference_backend_manager_available = True
    except ImportError:
        InferenceBackendManager = None
        get_backend_manager = None
        register_backend_from_config = None
        inference_backend_manager_available = False
else:
    InferenceBackendManager = None
    get_backend_manager = None
    register_backend_from_config = None
    inference_backend_manager_available = False

# Add auto-patching for transformers (applies automatically on import if enabled)
if not SKIP_CORE:
    try:
        from . import auto_patch_transformers

        export["auto_patch_transformers"] = auto_patch_transformers
    except ImportError:
        auto_patch_transformers = None
else:
    auto_patch_transformers = None

# Add LLM router functionality
if not SKIP_CORE:
    try:
        from .llm_router import (
            generate_text,
            get_llm_provider,
            register_llm_provider,
            clear_llm_router_caches,
            LLMProvider,
            MistralVibeInstallResult,
            ensure_mistral_vibe,
        )
        from .router_deps import (
            RouterDeps,
            get_default_router_deps,
            set_default_router_deps,
        )

        export["generate_text"] = generate_text
        export["get_llm_provider"] = get_llm_provider
        export["register_llm_provider"] = register_llm_provider
        export["clear_llm_router_caches"] = clear_llm_router_caches
        export["LLMProvider"] = LLMProvider
        export["MistralVibeInstallResult"] = MistralVibeInstallResult
        export["ensure_mistral_vibe"] = ensure_mistral_vibe
        export["RouterDeps"] = RouterDeps
        export["get_default_router_deps"] = get_default_router_deps
        export["set_default_router_deps"] = set_default_router_deps
        llm_router_available = True
    except ImportError:
        generate_text = None
        get_llm_provider = None
        register_llm_provider = None
        clear_llm_router_caches = None
        LLMProvider = None
        MistralVibeInstallResult = None
        ensure_mistral_vibe = None
        RouterDeps = None
        get_default_router_deps = None
        set_default_router_deps = None
        llm_router_available = False
else:
    generate_text = None
    get_llm_provider = None
    register_llm_provider = None
    clear_llm_router_caches = None
    LLMProvider = None
    MistralVibeInstallResult = None
    ensure_mistral_vibe = None
    RouterDeps = None
    get_default_router_deps = None
    set_default_router_deps = None
    llm_router_available = False

# Add Embeddings router functionality
if not SKIP_CORE:
    try:
        from .embeddings_router import (
            embed_texts,
            embed_texts_batched,
            embed_text,
            get_embeddings_provider,
            get_embedding_progress,
            get_last_embedding_trace,
            register_embeddings_provider,
            clear_embeddings_router_caches,
            EmbeddingsRouterError,
            EmbeddingsProvider,
        )

        export["embed_texts"] = embed_texts
        export["embed_texts_batched"] = embed_texts_batched
        export["embed_text"] = embed_text
        export["get_embeddings_provider"] = get_embeddings_provider
        export["get_embedding_progress"] = get_embedding_progress
        export["get_last_embedding_trace"] = get_last_embedding_trace
        export["register_embeddings_provider"] = register_embeddings_provider
        export["clear_embeddings_router_caches"] = clear_embeddings_router_caches
        export["EmbeddingsRouterError"] = EmbeddingsRouterError
        export["EmbeddingsProvider"] = EmbeddingsProvider
        embeddings_router_available = True
    except ImportError:
        embed_texts = None
        embed_texts_batched = None
        embed_text = None
        get_embeddings_provider = None
        get_embedding_progress = None
        get_last_embedding_trace = None
        register_embeddings_provider = None
        clear_embeddings_router_caches = None
        EmbeddingsRouterError = None
        EmbeddingsProvider = None
        embeddings_router_available = False
else:
    embed_texts = None
    embed_texts_batched = None
    embed_text = None
    get_embeddings_provider = None
    get_embedding_progress = None
    get_last_embedding_trace = None
    register_embeddings_provider = None
    clear_embeddings_router_caches = None
    EmbeddingsRouterError = None
    EmbeddingsProvider = None
    embeddings_router_available = False

# Add Multimodal router functionality
if not SKIP_CORE:
    try:
        from .multimodal_router import (
            generate_multimodal,
            get_multimodal_provider,
            register_multimodal_provider,
            clear_multimodal_router_caches,
            MultimodalProvider,
        )

        export["generate_multimodal"] = generate_multimodal
        export["get_multimodal_provider"] = get_multimodal_provider
        export["register_multimodal_provider"] = register_multimodal_provider
        export["clear_multimodal_router_caches"] = clear_multimodal_router_caches
        export["MultimodalProvider"] = MultimodalProvider
        multimodal_router_available = True
    except ImportError:
        generate_multimodal = None
        get_multimodal_provider = None
        register_multimodal_provider = None
        clear_multimodal_router_caches = None
        MultimodalProvider = None
        multimodal_router_available = False
else:
    generate_multimodal = None
    get_multimodal_provider = None
    register_multimodal_provider = None
    clear_multimodal_router_caches = None
    MultimodalProvider = None
    multimodal_router_available = False

# Add Voice router functionality (TTS + STT); also provides backward-compat
# aliases for the former tts_router (get_tts_provider, register_tts_provider,
# clear_tts_router_caches, TTSProvider).
if not SKIP_CORE:
    try:
        from .voice_router import (
            text_to_speech,
            speech_to_text,
            get_voice_provider,
            register_voice_provider,
            get_voice_provider_capabilities,
            clear_voice_router_caches,
            VoiceProvider,
            VoiceProviderCapabilities,
            ProviderInfo,
            VOICE_TURN_CONTRACT_VERSION,
            VOICE_STAGE_STATUSES,
            VOICE_TURN_STATUSES,
            DEFAULT_GROUNDED_FALLBACK,
            GroundingEvidence,
            VoiceGroundingSource,
            GroundedSlot,
            VoiceResponsePlan,
            VoiceTemplateProvider,
            GraphRAGVoiceTemplateProvider,
            buildVoiceGraphRagPromptParts,
            VoiceStageTrace,
            VoiceTurnRequest,
            VoiceTurnProvenance,
            VoiceTurnResult,
            voice_turn_cache_key,
            process_voice_turn,
            # backward-compat TTS aliases
            get_tts_provider,
            register_tts_provider,
            clear_tts_router_caches,
            TTSProvider,
        )

        export["text_to_speech"] = text_to_speech
        export["speech_to_text"] = speech_to_text
        export["get_voice_provider"] = get_voice_provider
        export["register_voice_provider"] = register_voice_provider
        export["get_voice_provider_capabilities"] = get_voice_provider_capabilities
        export["clear_voice_router_caches"] = clear_voice_router_caches
        export["VoiceProvider"] = VoiceProvider
        export["VoiceProviderCapabilities"] = VoiceProviderCapabilities
        export["ProviderInfo"] = ProviderInfo
        export["VOICE_TURN_CONTRACT_VERSION"] = VOICE_TURN_CONTRACT_VERSION
        export["VOICE_STAGE_STATUSES"] = VOICE_STAGE_STATUSES
        export["VOICE_TURN_STATUSES"] = VOICE_TURN_STATUSES
        export["DEFAULT_GROUNDED_FALLBACK"] = DEFAULT_GROUNDED_FALLBACK
        export["GroundingEvidence"] = GroundingEvidence
        export["VoiceGroundingSource"] = VoiceGroundingSource
        export["GroundedSlot"] = GroundedSlot
        export["VoiceResponsePlan"] = VoiceResponsePlan
        export["VoiceTemplateProvider"] = VoiceTemplateProvider
        export["GraphRAGVoiceTemplateProvider"] = GraphRAGVoiceTemplateProvider
        export["buildVoiceGraphRagPromptParts"] = buildVoiceGraphRagPromptParts
        export["VoiceStageTrace"] = VoiceStageTrace
        export["VoiceTurnRequest"] = VoiceTurnRequest
        export["VoiceTurnProvenance"] = VoiceTurnProvenance
        export["VoiceTurnResult"] = VoiceTurnResult
        export["voice_turn_cache_key"] = voice_turn_cache_key
        export["process_voice_turn"] = process_voice_turn
        export["get_tts_provider"] = get_tts_provider
        export["register_tts_provider"] = register_tts_provider
        export["clear_tts_router_caches"] = clear_tts_router_caches
        export["TTSProvider"] = TTSProvider
        voice_router_available = True
        tts_router_available = True
    except ImportError:
        text_to_speech = None
        speech_to_text = None
        get_voice_provider = None
        register_voice_provider = None
        get_voice_provider_capabilities = None
        clear_voice_router_caches = None
        VoiceProvider = None
        VoiceProviderCapabilities = None
        ProviderInfo = None
        VOICE_TURN_CONTRACT_VERSION = None
        VOICE_STAGE_STATUSES = None
        VOICE_TURN_STATUSES = None
        DEFAULT_GROUNDED_FALLBACK = None
        GroundingEvidence = None
        VoiceGroundingSource = None
        GroundedSlot = None
        VoiceResponsePlan = None
        VoiceTemplateProvider = None
        GraphRAGVoiceTemplateProvider = None
        buildVoiceGraphRagPromptParts = None
        VoiceStageTrace = None
        VoiceTurnRequest = None
        VoiceTurnProvenance = None
        VoiceTurnResult = None
        voice_turn_cache_key = None
        process_voice_turn = None
        get_tts_provider = None
        register_tts_provider = None
        clear_tts_router_caches = None
        TTSProvider = None
        voice_router_available = False
        tts_router_available = False
else:
    text_to_speech = None
    speech_to_text = None
    get_voice_provider = None
    register_voice_provider = None
    get_voice_provider_capabilities = None
    clear_voice_router_caches = None
    VoiceProvider = None
    VoiceProviderCapabilities = None
    ProviderInfo = None
    VOICE_TURN_CONTRACT_VERSION = None
    VOICE_STAGE_STATUSES = None
    VOICE_TURN_STATUSES = None
    DEFAULT_GROUNDED_FALLBACK = None
    GroundingEvidence = None
    VoiceGroundingSource = None
    GroundedSlot = None
    VoiceResponsePlan = None
    VoiceTemplateProvider = None
    GraphRAGVoiceTemplateProvider = None
    buildVoiceGraphRagPromptParts = None
    VoiceStageTrace = None
    VoiceTurnRequest = None
    VoiceTurnProvenance = None
    VoiceTurnResult = None
    voice_turn_cache_key = None
    process_voice_turn = None
    get_tts_provider = None
    register_tts_provider = None
    clear_tts_router_caches = None
    TTSProvider = None
    voice_router_available = False
    tts_router_available = False

__all__ = [
    'ipfs_accelerate_py', 'get_instance', 'backends', 'config',
    'install_depends', 'worker', 'ipfs_multiformats_py',
    'accelerate_with_browser', 'WebNNWebGPUAccelerator', 'get_accelerator',
    'webnn_webgpu_available', 'ModelManager', 'get_default_model_manager',
    'model_manager_available', 'SpaceRuntimeInfo', 'EndpointContract',
    'OutputBackend', 'LocalFileSystemBackend', 'HFBucketBackend',
    'HFBucketBackendError',
    'HFSpaceClient', 'RefreshableGradioFile', 'BatchState', 'BatchProcessor',
    'is_hf_space_transport_error', 'is_retryable_hf_space_error',
    'is_stale_gradio_file_error', 'normalize_api_name',
    'cli_main', 'get_system_logs', 'SystemLogs',
    'P2PWorkflowScheduler', 'P2PTask', 'WorkflowTag', 'MerkleClock',
    'FibonacciHeap', 'calculate_hamming_distance',
    'IPFSKitStorage', 'get_storage', 'reset_storage', 'StorageBackendConfig',
    'InferenceBackendManager', 'get_backend_manager', 'register_backend_from_config',
    'inference_backend_manager_available',
    'auto_patch_transformers',
    'generate_text', 'get_llm_provider', 'register_llm_provider',
    'clear_llm_router_caches', 'LLMProvider', 'RouterDeps',
    'MistralVibeInstallResult', 'ensure_mistral_vibe',
    'get_default_router_deps', 'set_default_router_deps', 'llm_router_available',
    'embed_texts', 'embed_texts_batched', 'embed_text', 'get_embeddings_provider',
    'get_embedding_progress', 'get_last_embedding_trace', 'register_embeddings_provider',
    'clear_embeddings_router_caches', 'EmbeddingsRouterError', 'EmbeddingsProvider',
    'embeddings_router_available',
    'generate_multimodal', 'get_multimodal_provider', 'register_multimodal_provider',
    'clear_multimodal_router_caches', 'MultimodalProvider', 'multimodal_router_available',
    'text_to_speech', 'get_tts_provider', 'register_tts_provider',
    'clear_tts_router_caches', 'TTSProvider', 'tts_router_available',
    'speech_to_text', 'get_voice_provider', 'register_voice_provider',
    'get_voice_provider_capabilities',
    'clear_voice_router_caches', 'VoiceProvider', 'voice_router_available',
    'VoiceProviderCapabilities', 'ProviderInfo', 'VOICE_TURN_CONTRACT_VERSION',
    'VOICE_STAGE_STATUSES', 'VOICE_TURN_STATUSES', 'DEFAULT_GROUNDED_FALLBACK',
    'GroundingEvidence', 'VoiceGroundingSource', 'GroundedSlot',
    'VoiceResponsePlan', 'VoiceTemplateProvider', 'GraphRAGVoiceTemplateProvider',
    'buildVoiceGraphRagPromptParts',
    'VoiceStageTrace', 'VoiceTurnRequest', 'VoiceTurnProvenance',
    'VoiceTurnResult', 'voice_turn_cache_key', 'process_voice_turn',
]


class _IPFSAccelerateModule(ModuleType):
    """Keep the lazy root worker canonical across subpackage import order."""

    def __getattribute__(self, name):
        if name == "worker":
            namespace = ModuleType.__getattribute__(self, "__dict__")
            return namespace["_load_legacy_worker"]()
        return ModuleType.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name != "worker":
            ModuleType.__setattr__(self, name, value)
            return
        namespace = ModuleType.__getattribute__(self, "__dict__")
        # Importlib installs the worker package on its parent before the
        # historical concrete ``worker.worker`` module is selected. Preserve
        # that raw bookkeeping without turning it into the canonical export.
        if (
            isinstance(value, ModuleType)
            and value.__name__ == f"{namespace['__name__']}.worker"
        ):
            ModuleType.__setattr__(self, name, value)
            return
        condition = namespace["_worker_condition"]
        with condition:
            while (
                namespace["_worker_loading"]
                and namespace["_worker_loader_thread_id"] != threading.get_ident()
            ):
                condition.wait()
            namespace["_worker_export_value"] = value
            ModuleType.__setattr__(self, name, value)
            export_map = namespace.get("export")
            if isinstance(export_map, dict):
                dict.__setitem__(export_map, "worker", value)
            condition.notify_all()

    def __delattr__(self, name):
        if name != "worker":
            ModuleType.__delattr__(self, name)
            return
        namespace = ModuleType.__getattribute__(self, "__dict__")
        condition = namespace["_worker_condition"]
        with condition:
            while namespace["_worker_loading"]:
                condition.wait()
            namespace["_worker_export_value"] = (
                None if namespace["SKIP_CORE"] else namespace["_WORKER_UNRESOLVED"]
            )
            namespace.pop("worker", None)
            export_map = namespace.get("export")
            if isinstance(export_map, dict):
                dict.__setitem__(
                    export_map,
                    "worker",
                    None
                    if namespace["SKIP_CORE"]
                    else namespace["_worker_snapshot"],
                )
            condition.notify_all()

    def __dir__(self):
        namespace = ModuleType.__getattribute__(self, "__dict__")
        return sorted(set(namespace).union(namespace.get("__all__", ())))


sys.modules[__name__].__class__ = _IPFSAccelerateModule

# Package version -- keep aligned with pyproject.toml / setup.py on every release.
# A separate runtime string previously drifted to 0.4.0; packaging metadata is
# the product pin (CHANGELOG, PyPI workflow).
_PACKAGING_VERSION = "0.0.45"
__version__ = _PACKAGING_VERSION
