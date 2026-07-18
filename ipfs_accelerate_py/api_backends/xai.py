"""xAI Grok API backend for ipfs_accelerate_py.

Provides access to xAI's Grok models via the OpenAI-compatible REST API at
https://api.x.ai/v1.

Environment variables:
  XAI_API_KEY or ipfs_accelerate_py_XAI_API_KEY  - Required API key
  ipfs_accelerate_py_XAI_MODEL                    - Default model (grok-3)
  ipfs_accelerate_py_XAI_BASE_URL                 - Override base URL
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

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

logger = logging.getLogger("xai_api")

try:
    from .base import BaseAPIBackend
except ImportError:
    try:
        from base import BaseAPIBackend
    except ImportError:
        BaseAPIBackend = object

_DEFAULT_BASE_URL = "https://api.x.ai/v1"
_DEFAULT_MODEL = "grok-3"

# Models available via xAI API (https://docs.x.ai/docs/models)
CHAT_MODELS = {
    "grok-3": {
        "context_window": 131072,
        "description": "xAI's most capable Grok 3 model",
    },
    "grok-3-fast": {
        "context_window": 131072,
        "description": "Faster variant of Grok 3",
    },
    "grok-3-mini": {
        "context_window": 131072,
        "description": "Lightweight Grok 3 model for cost-efficiency",
    },
    "grok-3-mini-fast": {
        "context_window": 131072,
        "description": "Fast, lightweight Grok 3 mini model",
    },
    "grok-2-1212": {
        "context_window": 131072,
        "description": "Grok 2 December 2024 release",
    },
    "grok-2-vision-1212": {
        "context_window": 32768,
        "description": "Grok 2 with vision capabilities",
    },
    "grok-beta": {
        "context_window": 131072,
        "description": "Grok beta model",
    },
}

ALL_MODELS = dict(CHAT_MODELS)


class xai(BaseAPIBackend):
    """xAI Grok API client.

    Supports chat completions via the OpenAI-compatible xAI endpoint.
    """

    def __init__(self, resources=None, metadata=None):
        self.resources = resources or {}
        self.metadata = metadata or {}

        self.api_key = self._get_api_key()
        self.base_url = (
            self.metadata.get("base_url")
            or self.metadata.get("api_base")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self.default_model = self.metadata.get("model") or _DEFAULT_MODEL

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
        key = (
            self.metadata.get("api_key")
            or self.metadata.get("xai_api_key")
            or self.metadata.get("XAI_API_KEY")
        )
        if key:
            return str(key).strip()
        import os
        return (
            os.environ.get("XAI_API_KEY")
            or os.environ.get("ipfs_accelerate_py_XAI_API_KEY")
            or ""
        ).strip() or None

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
                "xAI API key not configured. Set XAI_API_KEY or ipfs_accelerate_py_XAI_API_KEY."
            )
        if not self.check_circuit_breaker():
            raise RuntimeError("xAI circuit breaker is OPEN; too many recent failures")

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
        raise RuntimeError(f"xAI API request failed after {self.max_retries} retries: {last_exc}")

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
        """Send a chat completion request to xAI."""
        _model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": _model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        payload.update({k: v for k, v in kwargs.items() if v is not None})
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
        """Return embeddings for a list of texts (if supported by the model)."""
        _model = model or "grok-3"
        payload = {"model": _model, "input": texts}
        data = self._make_request("embeddings", payload)
        result = data.get("data") or []
        return [item.get("embedding", []) for item in result]

    # ------------------------------------------------------------------
    # List models
    # ------------------------------------------------------------------

    def list_models(self) -> List[str]:
        """Return the list of known Grok model identifiers."""
        return list(ALL_MODELS.keys())

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a specific model, or None if unknown."""
        return ALL_MODELS.get(model_name)
