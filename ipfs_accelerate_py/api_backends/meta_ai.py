"""Meta Model API (Muse Spark) backend for ipfs_accelerate_py.

Provides access to Muse Spark through Meta's OpenAI-compatible hosted API.
Legacy Llama model identifiers remain in the catalogue for callers using a
custom compatible endpoint, but the hosted default is Muse Spark 1.1.

References:
  https://developer.meta.com/ai/resources/blog/build-with-muse-spark/

Environment variables:
  MODEL_API_KEY, META_AI_API_KEY, or encrypted meta_ai_api_key - API key
  ipfs_accelerate_py_META_AI_MODEL                        - Default model (muse-spark-1.1)
  ipfs_accelerate_py_META_AI_BASE_URL                     - Override base URL
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from ..common.meta_model_api import (
    META_MODEL_API_BASE_URL,
    META_MODEL_API_DEFAULT_MODEL,
    normalize_meta_model_name,
    resolve_meta_model_api_key,
)

try:
    from ..common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        HAVE_STORAGE_WRAPPER = False
        def get_storage_wrapper(*args, **kwargs):
            return None

try:
    from ..datasets_integration import (
        is_datasets_available,
        ProvenanceLogger,
        DatasetsManager,
    )
    HAVE_DATASETS_INTEGRATION = True
except ImportError:
    try:
        from datasets_integration import (
            is_datasets_available,
            ProvenanceLogger,
            DatasetsManager,
        )
        HAVE_DATASETS_INTEGRATION = True
    except ImportError:
        HAVE_DATASETS_INTEGRATION = False
        is_datasets_available = lambda: False
        ProvenanceLogger = None
        DatasetsManager = None

try:
    import requests as _requests_lib
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    _requests_lib = None

logger = logging.getLogger("meta_ai_api")

try:
    from .base import BaseAPIBackend
except ImportError:
    try:
        from base import BaseAPIBackend
    except ImportError:
        BaseAPIBackend = object

_DEFAULT_BASE_URL = META_MODEL_API_BASE_URL
_DEFAULT_MODEL = META_MODEL_API_DEFAULT_MODEL

# Meta AI models available through the Llama API and Spark/Muse platform.
# https://llama.meta.com/docs/model-cards-and-prompt-formats/
CHAT_MODELS = {
    "muse-spark-1.1": {
        "context_window": 1_048_576,
        "max_output_tokens": 131_072,
        "description": "Meta Muse Spark 1.1 multimodal reasoning and agentic model",
        "modalities": ["text", "image", "video", "audio", "pdf"],
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "context_window": 128000,
        "description": "Meta Llama 3.3 70B instruction-tuned – flagship chat model",
    },
    "meta-llama/Llama-3.1-405B-Instruct": {
        "context_window": 128000,
        "description": "Meta Llama 3.1 405B instruction-tuned – most capable open model",
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "context_window": 128000,
        "description": "Meta Llama 3.1 70B instruction-tuned",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "context_window": 128000,
        "description": "Meta Llama 3.1 8B instruction-tuned – fast and efficient",
    },
    "meta-llama/Llama-3.2-90B-Vision-Instruct": {
        "context_window": 128000,
        "description": "Meta Llama 3.2 90B vision model",
    },
    "meta-llama/Llama-3.2-11B-Vision-Instruct": {
        "context_window": 128000,
        "description": "Meta Llama 3.2 11B vision model",
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "context_window": 128000,
        "description": "Meta Llama 3.2 3B – compact instruction-tuned model",
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "context_window": 128000,
        "description": "Meta Llama 3.2 1B – ultra-compact instruction-tuned model",
    },
    # Backward-compatible spelling used by the pre-release integration.
    "meta-spark/Spark-1.1": {
        "context_window": 1_048_576,
        "max_output_tokens": 131_072,
        "description": "Deprecated alias for muse-spark-1.1",
        "deprecated": True,
        "replacement": "muse-spark-1.1",
    },
}

ALL_MODELS = dict(CHAT_MODELS)


class meta_ai(BaseAPIBackend):
    """Meta Model API client.

    Supports Muse Spark chat completions through Meta's OpenAI-compatible
    endpoint. Custom base URLs may continue serving legacy catalogue entries.
    """

    def __init__(self, resources=None, metadata=None):
        self.resources = resources or {}
        self.metadata = metadata or {}

        self.api_key = self._get_api_key()
        self.base_url = (
            self.metadata.get("base_url")
            or self.metadata.get("api_base")
            or os.environ.get("ipfs_accelerate_py_META_AI_BASE_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self.default_model = normalize_meta_model_name(
            self.metadata.get("model")
            or os.environ.get("ipfs_accelerate_py_META_AI_MODEL")
            or _DEFAULT_MODEL
        )

        self.max_retries = int(self.metadata.get("max_retries", 3))
        self.timeout = float(self.metadata.get("timeout", 60.0))

        self._init_circuit_breaker()

        if HAVE_STORAGE_WRAPPER:
            try:
                self._storage = get_storage_wrapper(auto_detect_ci=True)
            except Exception:
                self._storage = None
        else:
            self._storage = None

    # ------------------------------------------------------------------
    # API key resolution
    # ------------------------------------------------------------------

    def _get_api_key(self) -> Optional[str]:
        explicit = (
            self.metadata.get("api_key")
            or self.metadata.get("meta_ai_api_key")
            or self.metadata.get("MODEL_API_KEY")
            or self.metadata.get("META_AI_API_KEY")
        )
        use_secrets_manager = str(
            self.metadata.get("use_secrets_manager", "true")
        ).strip().lower() not in {"0", "false", "no", "off"}
        return resolve_meta_model_api_key(
            str(explicit) if explicit is not None else None,
            use_secrets_manager=use_secrets_manager,
        )

    # ------------------------------------------------------------------
    # Low-level HTTP request
    # ------------------------------------------------------------------

    def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        *,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError(
                "Meta AI API key not configured. "
                "Store meta_ai_api_key in the encrypted credentials manager or "
                "set MODEL_API_KEY, META_AI_API_KEY, or "
                "ipfs_accelerate_py_META_AI_API_KEY."
            )
        if not self.check_circuit_breaker():
            raise RuntimeError("Meta AI circuit breaker is OPEN; too many recent failures")

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        _timeout = timeout if timeout is not None else self.timeout

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                if REQUESTS_AVAILABLE:
                    resp = _requests_lib.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=_timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                else:
                    import urllib.request
                    import urllib.error

                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        method="POST",
                        headers=headers,
                    )
                    with urllib.request.urlopen(req, timeout=_timeout) as r:
                        data = json.loads(r.read().decode("utf-8", errors="replace"))

                self.track_request_result(True)
                return data
            except Exception as exc:
                last_exc = exc
                retry_after = 2 ** attempt
                time.sleep(min(retry_after, 16))

        self.track_request_result(False)
        raise RuntimeError(
            f"Meta AI API request failed after {self.max_retries} retries: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send a chat completion request to Meta AI."""
        _model = normalize_meta_model_name(model or self.default_model)
        max_completion_tokens = kwargs.get(
            "max_completion_tokens",
            kwargs.get("max_tokens", kwargs.get("max_new_tokens", max_tokens)),
        )
        payload: Dict[str, Any] = {
            "model": _model,
            "messages": messages,
            "max_completion_tokens": int(max_completion_tokens),
            "temperature": temperature,
        }
        ignored = {
            "max_completion_tokens",
            "max_new_tokens",
            "max_tokens",
            "timeout",
        }
        payload.update(
            {
                k: v
                for k, v in kwargs.items()
                if v is not None and k not in ignored
            }
        )
        return self._make_request("chat/completions", payload, timeout=kwargs.get("timeout"))

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text response for the given prompt."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.chat(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        choices = response.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
        return ""

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: List[str],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        """Return embeddings for a list of texts."""
        _model = model or self.default_model
        payload = {"model": _model, "input": texts}
        data = self._make_request("embeddings", payload)
        result = data.get("data") or []
        return [item.get("embedding", []) for item in result]

    # ------------------------------------------------------------------
    # List models
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """Return the list of known Meta AI model identifiers."""
        return list(ALL_MODELS.keys())

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a specific model, or None if unknown."""
        return ALL_MODELS.get(model_name)
