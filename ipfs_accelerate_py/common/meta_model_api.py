"""Shared Meta Model API configuration and credential resolution."""

from __future__ import annotations

import hashlib
import os
from typing import Any, Optional

META_MODEL_API_BASE_URL = "https://api.meta.ai/v1"
META_MODEL_API_DEFAULT_MODEL = "muse-spark-1.1"
META_MODEL_API_SECRET_NAME = "meta_ai_api_key"
META_MODEL_API_ENV_VARS = (
    "MODEL_API_KEY",
    "META_AI_API_KEY",
    "ipfs_accelerate_py_META_AI_API_KEY",
)

_MODEL_ALIASES = {
    "meta-spark/Spark-1.1": META_MODEL_API_DEFAULT_MODEL,
    "meta/muse-spark-1.1": META_MODEL_API_DEFAULT_MODEL,
    "meta-ai/muse-spark-1.1": META_MODEL_API_DEFAULT_MODEL,
    "muse-spark-1.1": META_MODEL_API_DEFAULT_MODEL,
}


def normalize_meta_model_name(model_name: Optional[str]) -> str:
    value = str(model_name or "").strip()
    if not value:
        return META_MODEL_API_DEFAULT_MODEL
    return _MODEL_ALIASES.get(value, value)


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_meta_model_api_key(
    explicit: Optional[str] = None,
    *,
    secrets_manager: Any = None,
    use_secrets_manager: bool = True,
) -> Optional[str]:
    value = str(explicit or "").strip()
    if value:
        return value
    for variable in META_MODEL_API_ENV_VARS:
        value = str(os.environ.get(variable) or "").strip()
        if value:
            return value
    if not use_secrets_manager or _truthy(
        os.environ.get("IPFS_ACCELERATE_PY_DISABLE_SECRET_MANAGER")
    ):
        return None
    manager = secrets_manager
    if manager is None:
        try:
            from .secrets_manager import get_global_secrets_manager

            manager = get_global_secrets_manager()
        except Exception:
            return None
    for credential_name in (META_MODEL_API_SECRET_NAME, "model_api_key"):
        try:
            value = str(manager.get_credential(credential_name) or "").strip()
        except Exception:
            continue
        if value:
            return value
    return None


def meta_model_api_key_fingerprint() -> str:
    value = resolve_meta_model_api_key()
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "META_MODEL_API_BASE_URL",
    "META_MODEL_API_DEFAULT_MODEL",
    "META_MODEL_API_ENV_VARS",
    "META_MODEL_API_SECRET_NAME",
    "meta_model_api_key_fingerprint",
    "normalize_meta_model_name",
    "resolve_meta_model_api_key",
]
