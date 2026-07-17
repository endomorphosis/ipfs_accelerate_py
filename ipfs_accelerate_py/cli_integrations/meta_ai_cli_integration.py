"""
Meta AI CLI Integration with Common Cache

Wraps the Meta AI API (Llama and Spark/Muse family) and the OpenAI-compatible
REST endpoint at ``https://api.llamameta.net/v1`` to use the common cache
infrastructure.

Unique Meta AI capabilities surfaced by this integration
---------------------------------------------------------
* **Creative Mode** – Routes requests through Meta Spark 1.1, Meta's
  generative creative AI model from the Muse & Spark platform, optimised
  for story-telling, creative writing, and artistic content generation.
* **Vision Chat** – Passes image URLs alongside text prompts to the
  Llama 3.2 Vision models (11B and 90B variants).
* **Model Selector** – Automatically picks the right Llama size (1B–405B)
  for a task based on a ``task_hint`` parameter, balancing cost and quality.
* **Headless / CI Mode** – When ``headless=True``, creative-mode approval
  prompts are skipped and the integration runs fully non-interactively.

Multi-user / parallel execution
--------------------------------
Pass ``api_keys`` to distribute load across multiple Meta AI keys::

    meta = MetaAICLIIntegration(api_keys=["m-key-1", "m-key-2"])
    result = await meta.achat("Hello", user_id="alice")

Async support (Trio / Hypercorn)
---------------------------------
Every blocking public method has an ``a``-prefixed async twin that offloads
the network call to a worker thread via ``anyio.to_thread.run_sync``, keeping
the Trio / asyncio event loop free for other MCP requests::

    # Creative mode – non-blocking
    story = await meta.acreative_mode("Write a short sci-fi story about IPFS")

    # Vision chat – non-blocking
    desc = await meta.avision_chat("Describe this diagram", image_url="https://...")

    # Standard chat – non-blocking
    reply = await meta.achat("What is IPFS?")

Environment variables
---------------------
``META_AI_API_KEY``                       – Required Meta AI API key.
``ipfs_accelerate_py_META_AI_API_KEY``    – Alternative key env var.
``ipfs_accelerate_py_META_AI_BASE_URL``   – Override API base URL.
``META_AI_HEADLESS``                      – Set to ``1`` to enable headless mode.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Dict, List, Optional

from .dual_mode_wrapper import DualModeWrapper, detect_cli_tool
from ..common.llm_cache import LLMAPICache, get_global_llm_cache, get_llm_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Meta AI model catalogue
# https://llama.meta.com/docs/model-cards-and-prompt-formats/
# https://developer.meta.com/ai/resources/blog/build-with-muse-spark/
# ---------------------------------------------------------------------------
META_AI_MODELS: Dict[str, Dict[str, Any]] = {
    # Llama 3.3 – flagship instruction-tuned
    "meta-llama/Llama-3.3-70B-Instruct": {
        "context_window": 128_000,
        "description": "Meta Llama 3.3 70B instruction-tuned – flagship chat model",
        "recommended_for": ["chat", "code_generation", "reasoning"],
        "modalities": ["text"],
    },
    # Llama 3.1 – most capable open model
    "meta-llama/Llama-3.1-405B-Instruct": {
        "context_window": 128_000,
        "description": "Meta Llama 3.1 405B instruction-tuned – most capable open model",
        "recommended_for": ["complex_reasoning", "code_generation"],
        "modalities": ["text"],
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "context_window": 128_000,
        "description": "Meta Llama 3.1 70B instruction-tuned",
        "recommended_for": ["chat", "code_generation"],
        "modalities": ["text"],
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "context_window": 128_000,
        "description": "Meta Llama 3.1 8B instruction-tuned – fast and efficient",
        "recommended_for": ["chat", "subagents"],
        "modalities": ["text"],
    },
    # Llama 3.2 – vision models
    "meta-llama/Llama-3.2-90B-Vision-Instruct": {
        "context_window": 128_000,
        "description": "Meta Llama 3.2 90B vision model – high-quality multimodal",
        "recommended_for": ["vision_chat", "image_analysis"],
        "modalities": ["text", "image"],
    },
    "meta-llama/Llama-3.2-11B-Vision-Instruct": {
        "context_window": 128_000,
        "description": "Meta Llama 3.2 11B vision model – fast multimodal",
        "recommended_for": ["vision_chat"],
        "modalities": ["text", "image"],
    },
    # Llama 3.2 – compact text models
    "meta-llama/Llama-3.2-3B-Instruct": {
        "context_window": 128_000,
        "description": "Meta Llama 3.2 3B – compact instruction-tuned model",
        "recommended_for": ["subagents", "simple_chat"],
        "modalities": ["text"],
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "context_window": 128_000,
        "description": "Meta Llama 3.2 1B – ultra-compact instruction-tuned model",
        "recommended_for": ["subagents"],
        "modalities": ["text"],
    },
    # Meta Spark 1.1 – creative AI (Muse & Spark platform)
    "meta-spark/Spark-1.1": {
        "context_window": 32_768,
        "description": "Meta Spark 1.1 – creative AI model from the Muse & Spark platform",
        "recommended_for": ["creative_writing", "storytelling", "creative_mode"],
        "modalities": ["text"],
    },
}

_DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
_DEFAULT_VISION_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
_DEFAULT_CREATIVE_MODEL = "meta-spark/Spark-1.1"
_DEFAULT_BASE_URL = "https://api.llamameta.net/v1"

# Task hint → suggested model mapping for automatic model selection.
_TASK_MODEL_MAP: Dict[str, str] = {
    "subagent": "meta-llama/Llama-3.2-1B-Instruct",
    "simple": "meta-llama/Llama-3.2-3B-Instruct",
    "fast": "meta-llama/Llama-3.1-8B-Instruct",
    "chat": "meta-llama/Llama-3.3-70B-Instruct",
    "code": "meta-llama/Llama-3.3-70B-Instruct",
    "reasoning": "meta-llama/Llama-3.1-405B-Instruct",
    "creative": "meta-spark/Spark-1.1",
    "vision": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}


class MetaAICLIIntegration(DualModeWrapper):
    """
    Meta AI integration with common cache infrastructure.

    Supports dual-mode operation:
    - SDK mode:  Uses Meta's OpenAI-compatible API directly (primary mode;
      no official Meta AI CLI exists).
    - CLI mode:  Falls back to CLI if a future official CLI becomes available.

    Unique features:
    - Creative Mode via Meta Spark 1.1 for generative/artistic tasks.
    - Vision Chat support for Llama 3.2 multimodal models.
    - Automatic model selection by task hint.
    - Headless / CI mode (``headless=True`` or ``META_AI_HEADLESS=1``).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        enable_cache: bool = True,
        cache: Optional[LLMAPICache] = None,
        prefer_cli: bool = False,
        headless: bool = False,
        base_url: str = _DEFAULT_BASE_URL,
        **kwargs: Any,
    ) -> None:
        """
        Initialise Meta AI integration.

        Parameters
        ----------
        api_key:
            Meta AI API key.  Resolved from ``META_AI_API_KEY`` /
            ``ipfs_accelerate_py_META_AI_API_KEY`` / secrets manager when
            *None*.
        api_keys:
            List of Meta AI API keys for multi-user round-robin pool.  When
            provided, each call that supplies a ``user_id`` is pinned to a
            consistent key from the pool.
        enable_cache:
            Whether to use the LLM cache.
        cache:
            Custom ``LLMAPICache`` instance.  A provider-keyed cache is
            created automatically when *None*.
        prefer_cli:
            Try a Meta AI CLI binary first (default: ``False`` – no official
            Meta AI CLI exists; this is reserved for future use).
        headless:
            Skip interactive creative-mode approval – useful for CI/CD
            pipelines.  Overridden to ``True`` when the environment variable
            ``META_AI_HEADLESS=1`` is set.
        base_url:
            Override the Meta AI REST API base URL.
        """
        cache_was_none = cache is None
        if cache_was_none:
            cache = get_global_llm_cache()

        super().__init__(
            cli_path=None,  # auto-detected; no official Meta AI CLI yet
            api_key=api_key,
            api_keys=api_keys,
            cache=cache,
            enable_cache=enable_cache,
            prefer_cli=prefer_cli,
            **kwargs,
        )

        if cache_was_none:
            self.cache = get_llm_cache("meta_ai", api_key=self.api_key)

        self.base_url = base_url.rstrip("/")
        self.headless = headless or (os.environ.get("META_AI_HEADLESS", "0") == "1")

        # Lazy-initialised OpenAI-compatible client pointed at Meta's endpoint
        self._openai_client: Any = None

    # ------------------------------------------------------------------
    # DualModeWrapper abstract method implementations
    # ------------------------------------------------------------------

    def get_tool_name(self) -> str:
        return "Meta AI"

    def _detect_cli_path(self) -> Optional[str]:
        """Meta AI has no official CLI binary; return ``None``."""
        return detect_cli_tool(["meta-ai", "llama-cli"])

    def _get_api_key_from_secrets(self) -> Optional[str]:
        """Retrieve Meta AI API key from the secrets manager or env."""
        key = self.secrets_manager.get_credential("meta_ai_api_key")
        if not key:
            key = (
                os.environ.get("META_AI_API_KEY")
                or os.environ.get("ipfs_accelerate_py_META_AI_API_KEY")
            )
        return key or None

    def _create_sdk_client(self) -> Any:
        """Create an ``openai.OpenAI`` client pointed at the Meta AI endpoint."""
        try:
            import openai  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "openai SDK not installed. Install with: pip install openai"
            )
        return openai.OpenAI(
            api_key=self.api_key or "",
            base_url=self.base_url,
        )

    def _get_openai_client(self, api_key: Optional[str] = None) -> Any:
        """Get or create an OpenAI-compatible client for the given key.

        When *api_key* differs from ``self.api_key`` a fresh client is created
        (and cached) so that concurrent requests using different keys do not
        share state.
        """
        effective_key = api_key or self.api_key or ""
        if effective_key == (self.api_key or ""):
            if self._openai_client is None:
                self._openai_client = self._create_sdk_client()
            return self._openai_client
        if not hasattr(self, "_client_cache"):
            self._client_cache: Dict[str, Any] = {}
        if effective_key not in self._client_cache:
            try:
                import openai  # type: ignore[import]
            except ImportError:
                raise ImportError(
                    "openai SDK not installed. Install with: pip install openai"
                )
            self._client_cache[effective_key] = openai.OpenAI(
                api_key=effective_key,
                base_url=self.base_url,
            )
        return self._client_cache[effective_key]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat_sdk(
        self,
        message: str,
        model: str,
        temperature: float,
        system_prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute a single chat completion via the Meta AI SDK.

        Parameters
        ----------
        image_url:
            Optional image URL.  When provided the message is wrapped in a
            multimodal content list (text + image_url) for Llama Vision models.
        api_key:
            Optional per-request key override.
        """
        if image_url:
            user_content: Any = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:
            user_content = message

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        # Cache lookup (skip for vision – image content changes semantics)
        if self.enable_cache and not image_url:
            cache_msgs = [
                {"role": m["role"], "content": m["content"]}
                for m in messages
                if isinstance(m["content"], str)
            ]
            cached = self.cache.get_chat_completion(
                messages=cache_msgs,
                model=model,
                temperature=temperature,
            )
            if cached:
                logger.info("Cache hit for Meta AI chat")
                return {"response": cached, "cached": True}

        client = self._get_openai_client(api_key=api_key)

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        response = client.chat.completions.create(**create_kwargs)
        result = response.choices[0].message.content or ""

        if self.enable_cache and not image_url:
            cache_msgs_for_store = [
                {"role": m["role"], "content": m["content"]}
                for m in messages
                if isinstance(m["content"], str)
            ]
            self.cache.cache_chat_completion(
                messages=cache_msgs_for_store,
                response=result,
                model=model,
                temperature=temperature,
            )

        return {"response": result, "cached": False}

    @staticmethod
    def suggest_model(task_hint: str) -> str:
        """Return the suggested Meta AI model for the given *task_hint*.

        Parameters
        ----------
        task_hint:
            Short description of the task type.  Recognised values:
            ``"subagent"``, ``"simple"``, ``"fast"``, ``"chat"``, ``"code"``,
            ``"reasoning"``, ``"creative"``, ``"vision"``.  Falls back to the
            default chat model when *task_hint* is not recognised.
        """
        return _TASK_MODEL_MAP.get(task_hint.lower(), _DEFAULT_MODEL)

    # ------------------------------------------------------------------
    # Creative Mode  (unique to Meta Spark 1.1)
    # ------------------------------------------------------------------

    def creative_mode(
        self,
        prompt: str,
        model: str = _DEFAULT_CREATIVE_MODEL,
        temperature: float = 0.9,
        auto_approve: Optional[bool] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Meta AI Creative Mode – generate artistic or narrative content using
        Meta Spark 1.1.

        Meta Spark 1.1 is Meta's creative AI model from the Muse & Spark
        platform, optimised for story-telling, creative writing, and
        generative artistic text.  Unless the integration is running in
        headless mode (or ``auto_approve=True``), the generated preview is
        presented for approval before the full content is returned.

        Parameters
        ----------
        prompt:
            Creative prompt / description of the desired content.
        model:
            Meta AI model to use (default: ``meta-spark/Spark-1.1``).
        temperature:
            Sampling temperature (high values recommended for creative tasks).
        auto_approve:
            ``True``  – skip approval prompt regardless of headless setting.
            ``False`` – always prompt even in headless mode.
            ``None``  – honour ``self.headless``.
        user_id:
            Optional user identifier for per-user key pinning.

        Returns
        -------
        dict with keys:
            - ``response``  – The generated creative content.
            - ``approved``  – Whether the output was approved.
            - ``cached``    – Whether the result came from cache.
            - ``mode``      – Execution mode (``"SDK"`` or ``"CLI"``).
            - ``model``     – Model used.
        """
        raw = self._execute_with_fallback(
            sdk_func=self._chat_sdk,
            operation="creative_mode",
            message=prompt,
            model=model,
            temperature=temperature,
            api_key=self.get_api_key(user_id=user_id),
            **kwargs,
        )

        should_skip_approval = (
            (auto_approve is True)
            or (auto_approve is None and self.headless)
        )

        approved = should_skip_approval
        if not should_skip_approval:
            approved = self._prompt_creative_approval(raw.get("response", ""))

        return {
            "response": raw.get("response", "") if approved else "",
            "approved": approved,
            "cached": raw.get("cached", False),
            "mode": raw.get("mode", "SDK"),
            "model": model,
        }

    async def acreative_mode(
        self,
        prompt: str,
        model: str = _DEFAULT_CREATIVE_MODEL,
        temperature: float = 0.9,
        auto_approve: Optional[bool] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Async version of :meth:`creative_mode` – safe for Trio / Hypercorn.

        The blocking Meta AI API call is offloaded to a worker thread via
        ``anyio.to_thread.run_sync``.  Approval prompts still run in the
        same thread when ``headless=False``.
        """
        raw = await self._aexecute_with_fallback(
            sdk_func=self._chat_sdk,
            operation="creative_mode",
            message=prompt,
            model=model,
            temperature=temperature,
            api_key=self.get_api_key(user_id=user_id),
            **kwargs,
        )

        should_skip_approval = (
            (auto_approve is True)
            or (auto_approve is None and self.headless)
        )

        approved = should_skip_approval
        if not should_skip_approval:
            approved = self._prompt_creative_approval(raw.get("response", ""))

        return {
            "response": raw.get("response", "") if approved else "",
            "approved": approved,
            "cached": raw.get("cached", False),
            "mode": raw.get("mode", "SDK"),
            "model": model,
        }

    @staticmethod
    def _prompt_creative_approval(preview: str) -> bool:
        """
        Interactively prompt the user to approve the generated creative content.

        Returns ``True`` if approved, ``False`` otherwise.
        """
        print("\n" + "=" * 60)
        print("Meta AI – Creative Mode Preview")
        print("=" * 60)
        snippet = preview[:500] + ("…" if len(preview) > 500 else "")
        print(snippet)
        print("=" * 60)
        try:
            answer = input("Accept this content and continue? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        return answer in ("y", "yes")

    # ------------------------------------------------------------------
    # Vision Chat  (Llama 3.2 Vision models)
    # ------------------------------------------------------------------

    def vision_chat(
        self,
        message: str,
        image_url: str,
        model: str = _DEFAULT_VISION_MODEL,
        temperature: float = 0.7,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Send a multimodal chat message with an image to a Llama Vision model.

        The image is referenced by URL and passed via the standard OpenAI
        multimodal content format (``image_url`` type) to the Llama 3.2
        Vision models.

        Parameters
        ----------
        message:
            Text portion of the prompt (question / instruction about the image).
        image_url:
            Publicly accessible URL of the image to analyse.
        model:
            Vision-capable model (default: ``Llama-3.2-11B-Vision-Instruct``).
        temperature:
            Sampling temperature.
        user_id:
            Optional user identifier for per-user key pinning.

        Returns
        -------
        Dict with ``response``, ``cached`` (always ``False``), and ``mode``.
        """
        return self._execute_with_fallback(
            sdk_func=self._chat_sdk,
            operation="vision_chat",
            message=message,
            model=model,
            temperature=temperature,
            image_url=image_url,
            api_key=self.get_api_key(user_id=user_id),
            **kwargs,
        )

    async def avision_chat(
        self,
        message: str,
        image_url: str,
        model: str = _DEFAULT_VISION_MODEL,
        temperature: float = 0.7,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async version of :meth:`vision_chat` – safe for Trio / Hypercorn."""
        return await self._aexecute_with_fallback(
            sdk_func=self._chat_sdk,
            operation="vision_chat",
            message=message,
            model=model,
            temperature=temperature,
            image_url=image_url,
            api_key=self.get_api_key(user_id=user_id),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Standard chat / code-generation  (shared with other integrations)
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.7,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Send a chat message to Meta AI (synchronous).

        Parameters
        ----------
        message:
            User message / prompt.
        model:
            Meta AI model (default: ``Llama-3.3-70B-Instruct``).
        temperature:
            Sampling temperature.
        user_id:
            Optional user identifier for per-user key pinning.

        Returns
        -------
        Dict with ``response``, ``cached``, and ``mode``.
        """
        return self._execute_with_fallback(
            sdk_func=self._chat_sdk,
            operation="chat",
            message=message,
            model=model,
            temperature=temperature,
            api_key=self.get_api_key(user_id=user_id),
            **kwargs,
        )

    async def achat(
        self,
        message: str,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.7,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Async version of :meth:`chat` – safe for Trio / Hypercorn.

        The blocking Meta AI API call is offloaded to a worker thread via
        ``anyio.to_thread.run_sync``.
        """
        return await self._aexecute_with_fallback(
            sdk_func=self._chat_sdk,
            operation="chat",
            message=message,
            model=model,
            temperature=temperature,
            api_key=self.get_api_key(user_id=user_id),
            **kwargs,
        )

    def generate_code(
        self,
        prompt: str,
        model: str = _DEFAULT_MODEL,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate code from a natural-language prompt (synchronous).

        Parameters
        ----------
        prompt:
            Code generation prompt.
        model:
            Meta AI model.  Defaults to Llama-3.3-70B-Instruct which is
            strong at code generation.  Pass
            ``model="meta-llama/Llama-3.1-405B-Instruct"`` for maximum
            code quality.
        user_id:
            Optional user identifier for per-user key pinning.

        Returns
        -------
        Dict with ``response``, ``cached``, and ``mode``.
        """
        return self.chat(prompt, model=model, temperature=0.0, user_id=user_id, **kwargs)

    async def agenerate_code(
        self,
        prompt: str,
        model: str = _DEFAULT_MODEL,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Async version of :meth:`generate_code` – safe for Trio / Hypercorn."""
        return await self.achat(
            prompt, model=model, temperature=0.0, user_id=user_id, **kwargs
        )

    def list_models(self) -> List[str]:
        """Return the list of known Meta AI model identifiers."""
        return list(META_AI_MODELS.keys())

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a specific Meta AI model, or ``None`` if unknown."""
        return META_AI_MODELS.get(model_name)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_meta_ai_cli: Optional[MetaAICLIIntegration] = None


def get_meta_ai_cli_integration() -> MetaAICLIIntegration:
    """Get or create the global Meta AI CLI integration instance."""
    global _global_meta_ai_cli

    if _global_meta_ai_cli is None:
        _global_meta_ai_cli = MetaAICLIIntegration()

    return _global_meta_ai_cli
