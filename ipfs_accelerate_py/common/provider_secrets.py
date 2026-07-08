"""Provider secret discovery for encrypted distributed caching.

This module centralizes best-effort discovery of provider API keys / auth tokens
from environment variables.

These secrets are used only as shared inputs to derive encryption keys for
remote cache payloads; the secrets themselves are never written to the remote
cache.
"""

from __future__ import annotations

import os
from typing import Optional


_PROVIDER_ENV_VARS: dict[str, list[str]] = {
    # OpenAI / Codex
    "openai": ["OPENAI_API_KEY"],
    "codex": ["OPENAI_API_KEY"],
    # Anthropic / Claude
    "anthropic": ["ANTHROPIC_API_KEY"],
    "claude": ["ANTHROPIC_API_KEY"],
    # Google / Gemini
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    # Groq
    "groq": ["GROQ_API_KEY"],
    # GitHub / Copilot
    "github": ["GITHUB_TOKEN"],
    "copilot": ["GITHUB_TOKEN"],
    # VastAI
    "vastai": ["VASTAI_API_KEY", "VAST_API_KEY"],

    # HuggingFace Hub
    "huggingface": ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"],
    "hf": ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"],
}


def get_provider_cache_secret(provider: str, explicit_secret: Optional[str] = None) -> Optional[str]:
    """Return a shared secret string for a provider.

    Preference order:
    1) explicit_secret
    2) provider-specific env vars

    Returns None when not available.
    """

    if explicit_secret is not None:
        secret = str(explicit_secret).strip()
        return secret or None

    key = (provider or "").strip().lower()
    envs = _PROVIDER_ENV_VARS.get(key, [])

    for env in envs:
        val = (os.environ.get(env) or "").strip()
        if val:
            return val

    return None
