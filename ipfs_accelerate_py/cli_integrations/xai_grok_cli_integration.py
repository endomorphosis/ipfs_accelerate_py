"""
xAI Grok Build CLI Integration with Common Cache

Wraps the xAI Grok Build CLI (``grok``, installable via ``npm i -g @xai-official/grok``
or ``curl -fsSL https://x.ai/cli/install.sh | bash``) and the xAI OpenAI-compatible
REST API (https://api.x.ai/v1) to use the common cache infrastructure.

Unique Grok Build capabilities surfaced by this integration
-----------------------------------------------------------
* **Plan Mode** – Before applying multi-step code changes, Grok generates and
  presents a structured, step-by-step plan.  Execution is blocked until the
  caller explicitly approves (or the integration is running in headless/CI
  mode with ``headless=True``).
* **Subagents** – Complex tasks are decomposed and dispatched to parallel
  ``ThreadPoolExecutor`` workers, each running an independent Grok API call.
  Results are gathered and merged automatically.
* **Live Web / X Search** – Uses xAI's ``search_parameters`` API extension to
  pull real-time documentation, package info, and X (Twitter) posts directly
  into the model context.
* **Headless / CI Mode** – When ``headless=True``, plan approval is skipped
  and the integration runs fully non-interactively.

Environment variables
---------------------
``XAI_API_KEY``                        – Required xAI API key.
``ipfs_accelerate_py_XAI_API_KEY``     – Alternative key env var.
``XAI_HEADLESS``                       – Set to ``1`` to enable headless mode.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .dual_mode_wrapper import DualModeWrapper, detect_cli_tool
from ..common.llm_cache import LLMAPICache, get_global_llm_cache, get_llm_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grok model catalogue (https://docs.x.ai/docs/models)
# ---------------------------------------------------------------------------
GROK_MODELS: Dict[str, Dict[str, Any]] = {
    "grok-4": {
        "context_window": 256_000,
        "description": "xAI's most capable Grok 4 model – optimised for coding agents",
        "recommended_for": ["plan_mode", "subagents", "code_generation"],
    },
    "grok-4-heavy": {
        "context_window": 256_000,
        "description": "Grok 4 with extended reasoning budget",
        "recommended_for": ["plan_mode", "complex_reasoning"],
    },
    "grok-3": {
        "context_window": 131_072,
        "description": "Grok 3 flagship model",
        "recommended_for": ["chat", "code_generation"],
    },
    "grok-3-fast": {
        "context_window": 131_072,
        "description": "Fast variant of Grok 3",
        "recommended_for": ["chat"],
    },
    "grok-3-mini": {
        "context_window": 131_072,
        "description": "Lightweight Grok 3 for cost-efficient tasks",
        "recommended_for": ["subagents"],
    },
    "grok-3-mini-fast": {
        "context_window": 131_072,
        "description": "Fast, lightweight Grok 3 mini",
        "recommended_for": ["subagents"],
    },
}

_DEFAULT_MODEL = "grok-3"
_DEFAULT_BASE_URL = "https://api.x.ai/v1"

# System prompt used when Plan Mode is engaged
_PLAN_MODE_SYSTEM_PROMPT = (
    "You are Grok Build, an expert software engineering agent. "
    "When asked to implement a task you MUST first produce a numbered, "
    "step-by-step plan in Markdown before writing any code. "
    "Prefix the plan section with the exact header '## Plan' and the "
    "implementation section with '## Implementation'. "
    "Wait for explicit approval before proceeding with the implementation."
)


class XAIGrokCLIIntegration(DualModeWrapper):
    """
    xAI Grok Build CLI integration with common cache infrastructure.

    Supports dual-mode operation:
    - CLI mode:  Delegates to the ``grok`` binary (installed via npm/curl).
    - SDK mode:  Uses the xAI OpenAI-compatible API directly (primary).

    Unique features:
    - Plan Mode with optional interactive approval.
    - Parallel Subagents via ``ThreadPoolExecutor``.
    - Live Web / X Search through xAI ``search_parameters``.
    - Headless / CI mode (``headless=True`` or ``XAI_HEADLESS=1``).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_cache: bool = True,
        cache: Optional[LLMAPICache] = None,
        prefer_cli: bool = True,
        headless: bool = False,
        max_subagents: int = 4,
        base_url: str = _DEFAULT_BASE_URL,
        **kwargs: Any,
    ) -> None:
        """
        Initialise xAI Grok Build integration.

        Parameters
        ----------
        api_key:
            xAI API key.  Resolved from ``XAI_API_KEY`` / secrets manager
            when *None*.
        enable_cache:
            Whether to use the LLM cache.
        cache:
            Custom ``LLMAPICache`` instance.  A provider-keyed cache is
            created automatically when *None*.
        prefer_cli:
            Try the ``grok`` CLI binary first (default: ``True``).
        headless:
            Skip interactive plan approval – useful for CI/CD pipelines.
            Overridden to ``True`` when the environment variable
            ``XAI_HEADLESS=1`` is set.
        max_subagents:
            Maximum number of parallel subagent workers.
        base_url:
            Override the xAI REST API base URL.
        """
        cache_was_none = cache is None
        if cache_was_none:
            cache = get_global_llm_cache()

        super().__init__(
            cli_path=None,  # auto-detected below
            api_key=api_key,
            cache=cache,
            enable_cache=enable_cache,
            prefer_cli=prefer_cli,
            **kwargs,
        )

        if cache_was_none:
            self.cache = get_llm_cache("xai", api_key=self.api_key)

        self.base_url = base_url.rstrip("/")
        self.headless = headless or (os.environ.get("XAI_HEADLESS", "0") == "1")
        self.max_subagents = max(1, max_subagents)

        # Lazy-initialised openai client (pointed at xAI endpoint)
        self._openai_client: Any = None

    # ------------------------------------------------------------------
    # DualModeWrapper abstract method implementations
    # ------------------------------------------------------------------

    def get_tool_name(self) -> str:
        return "xAI Grok Build"

    def _detect_cli_path(self) -> Optional[str]:
        """Detect the official ``grok`` CLI binary (``@xai-official/grok``)."""
        return detect_cli_tool(["grok", "grok-build"])

    def _get_api_key_from_secrets(self) -> Optional[str]:
        """Retrieve xAI API key from the secrets manager."""
        key = self.secrets_manager.get_credential("xai_api_key")
        if not key:
            key = (
                os.environ.get("XAI_API_KEY")
                or os.environ.get("ipfs_accelerate_py_XAI_API_KEY")
            )
        return key or None

    def _create_sdk_client(self) -> Any:
        """Create an ``openai.OpenAI`` client pointed at the xAI API."""
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

    def _get_openai_client(self) -> Any:
        if self._openai_client is None:
            self._openai_client = self._create_sdk_client()
        return self._openai_client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat_sdk(
        self,
        message: str,
        model: str,
        temperature: float,
        system_prompt: Optional[str] = None,
        search_enabled: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a single chat completion via the xAI SDK."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        # Cache lookup
        if self.enable_cache:
            cached = self.cache.get_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
            )
            if cached:
                logger.info("Cache hit for xAI Grok chat")
                return {"response": cached, "cached": True}

        client = self._get_openai_client()

        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        # Grok-unique: live web/X search via search_parameters
        if search_enabled:
            create_kwargs["search_parameters"] = {
                "mode": "auto",
                "sources": [{"type": "web"}, {"type": "x"}],
            }

        response = client.chat.completions.create(**create_kwargs)
        result = response.choices[0].message.content or ""

        if self.enable_cache and not search_enabled:
            # Don't cache live-search results – they are time-sensitive
            self.cache.cache_chat_completion(
                messages=messages,
                response=result,
                model=model,
                temperature=temperature,
            )

        return {"response": result, "cached": False}

    # ------------------------------------------------------------------
    # Plan Mode  (unique to Grok Build)
    # ------------------------------------------------------------------

    def plan_mode(
        self,
        task: str,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.2,
        auto_approve: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Grok Build Plan Mode – generate a structured plan before executing.

        Grok first produces a numbered step-by-step plan.  Unless the
        integration is running in headless mode (or ``auto_approve=True``),
        the caller is prompted to approve the plan before implementation
        proceeds.

        Parameters
        ----------
        task:
            Natural-language description of the coding task.
        model:
            Grok model to use (default ``grok-3``).
        temperature:
            Sampling temperature for plan generation (low values recommended).
        auto_approve:
            ``True``  – skip approval prompt regardless of headless setting.
            ``False`` – always prompt even in headless mode.
            ``None``  – honour ``self.headless``.

        Returns
        -------
        dict with keys:
            - ``plan``          – The Markdown plan section.
            - ``implementation``– The implementation section (if approved).
            - ``approved``      – Whether the plan was approved.
            - ``cached``        – Whether the result came from cache.
            - ``mode``          – Execution mode (``"SDK"`` or ``"CLI"``).
        """
        plan_prompt = (
            f"Task: {task}\n\n"
            "Please first produce a detailed ## Plan (numbered steps), "
            "then add a ## Implementation section with the actual code."
        )

        raw = self._execute_with_fallback(
            sdk_func=self._chat_sdk,
            operation="plan_mode",
            message=plan_prompt,
            model=model,
            temperature=temperature,
            system_prompt=_PLAN_MODE_SYSTEM_PROMPT,
            **kwargs,
        )

        full_text: str = raw.get("response", "")

        # Split response into plan / implementation sections
        plan_section, impl_section = self._split_plan_impl(full_text)

        # Determine whether to seek approval
        should_skip_approval = (
            (auto_approve is True)
            or (auto_approve is None and self.headless)
        )

        approved = should_skip_approval
        if not should_skip_approval:
            approved = self._prompt_plan_approval(plan_section)

        return {
            "plan": plan_section,
            "implementation": impl_section if approved else "",
            "approved": approved,
            "cached": raw.get("cached", False),
            "mode": raw.get("mode", "SDK"),
        }

    @staticmethod
    def _split_plan_impl(text: str):
        """Split a Grok response into plan and implementation sections."""
        plan_marker = "## Plan"
        impl_marker = "## Implementation"

        plan_start = text.find(plan_marker)
        impl_start = text.find(impl_marker)

        if plan_start == -1 and impl_start == -1:
            return text, ""

        if plan_start != -1 and impl_start != -1:
            plan = text[plan_start:impl_start].strip()
            impl = text[impl_start:].strip()
        elif plan_start != -1:
            plan = text[plan_start:].strip()
            impl = ""
        else:
            plan = ""
            impl = text[impl_start:].strip()

        return plan, impl

    @staticmethod
    def _prompt_plan_approval(plan: str) -> bool:
        """
        Interactively prompt the user to approve the generated plan.

        Returns ``True`` if approved, ``False`` otherwise.
        """
        print("\n" + "=" * 60)
        print("xAI Grok Build – Plan Mode")
        print("=" * 60)
        print(plan)
        print("=" * 60)
        try:
            answer = input("Approve this plan and proceed? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        return answer in ("y", "yes")

    # ------------------------------------------------------------------
    # Subagents  (unique to Grok Build)
    # ------------------------------------------------------------------

    def spawn_subagents(
        self,
        subtasks: List[str],
        model: str = "grok-3-mini",
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Dispatch multiple independent subtasks to parallel Grok subagents.

        Each subtask is executed concurrently in its own thread (up to
        ``self.max_subagents`` workers).  Results preserve the original
        subtask order.

        Parameters
        ----------
        subtasks:
            List of independent task descriptions / prompts.
        model:
            Grok model for each subagent (``grok-3-mini`` by default for
            cost-efficiency at scale).
        temperature:
            Sampling temperature.

        Returns
        -------
        List of result dicts (same length and order as *subtasks*), each
        containing ``response``, ``cached``, ``mode``, and ``subtask``.
        """
        results: List[Optional[Dict[str, Any]]] = [None] * len(subtasks)

        def _run_subtask(idx: int, subtask: str) -> tuple[int, Dict[str, Any]]:
            logger.debug("Subagent %d/%d: %s", idx + 1, len(subtasks), subtask[:80])
            result = self._execute_with_fallback(
                sdk_func=self._chat_sdk,
                operation="subagent",
                message=subtask,
                model=model,
                temperature=temperature,
                **kwargs,
            )
            result["subtask"] = subtask
            return idx, result

        workers = min(self.max_subagents, len(subtasks))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_subtask, i, task): i
                for i, task in enumerate(subtasks)
            }
            for future in as_completed(futures):
                try:
                    idx, res = future.result()
                    results[idx] = res
                except Exception as exc:
                    idx = futures[future]
                    logger.error("Subagent %d failed: %s", idx, exc)
                    results[idx] = {
                        "response": "",
                        "error": str(exc),
                        "subtask": subtasks[idx],
                        "cached": False,
                    }

        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Live Web / X Search  (unique to Grok Build)
    # ------------------------------------------------------------------

    def web_search(
        self,
        query: str,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Query Grok with real-time web and X (Twitter) search enabled.

        xAI's ``search_parameters`` API extension augments the model
        context with live web pages and X posts before generating the
        response, keeping results up-to-date without manual copy-pasting
        documentation.

        Parameters
        ----------
        query:
            Search query or question.
        model:
            Grok model to use.
        temperature:
            Sampling temperature (0.0 recommended for factual lookups).

        Returns
        -------
        Dict with ``response`` (model text enriched with live search),
        ``cached`` (always ``False`` for search results),
        and ``mode``.
        """
        return self._execute_with_fallback(
            sdk_func=self._chat_sdk,
            operation="web_search",
            message=query,
            model=model,
            temperature=temperature,
            search_enabled=True,
            **kwargs,
        )

    def x_search(
        self,
        query: str,
        model: str = _DEFAULT_MODEL,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Search X (Twitter) posts via Grok's live X search capability.

        Convenience wrapper around :meth:`web_search` for X-focused queries
        such as trending discussions, real-time announcements, or community
        Q&A around a library/API.
        """
        x_query = f"Search X (Twitter) for the latest posts about: {query}"
        return self.web_search(x_query, model=model, **kwargs)

    # ------------------------------------------------------------------
    # Standard chat / code-generation  (shared with other integrations)
    # ------------------------------------------------------------------

    def chat(
        self,
        message: str,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Send a chat message to Grok.

        Parameters
        ----------
        message:
            User message / prompt.
        model:
            Grok model (default ``grok-3``).
        temperature:
            Sampling temperature.

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
            **kwargs,
        )

    def generate_code(
        self,
        prompt: str,
        model: str = _DEFAULT_MODEL,
        use_plan_mode: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate code from a natural-language prompt.

        When ``use_plan_mode=True`` the Grok Build plan-first workflow is
        engaged: a structured plan is produced and (unless headless) presented
        for approval before the implementation is returned.

        Parameters
        ----------
        prompt:
            Code generation prompt.
        model:
            Grok model.
        use_plan_mode:
            Enable Grok Build Plan Mode for this request.

        Returns
        -------
        Dict with ``response`` (or ``plan`` + ``implementation`` in plan mode),
        ``cached``, and ``mode``.
        """
        if use_plan_mode:
            return self.plan_mode(task=prompt, model=model, **kwargs)

        return self.chat(prompt, model=model, temperature=0.0, **kwargs)

    def list_models(self) -> List[str]:
        """Return the list of known Grok model identifiers."""
        return list(GROK_MODELS.keys())

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a specific Grok model, or ``None`` if unknown."""
        return GROK_MODELS.get(model_name)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_xai_grok_cli: Optional[XAIGrokCLIIntegration] = None


def get_xai_grok_cli_integration() -> XAIGrokCLIIntegration:
    """Get or create the global xAI Grok CLI integration instance."""
    global _global_xai_grok_cli

    if _global_xai_grok_cli is None:
        _global_xai_grok_cli = XAIGrokCLIIntegration()

    return _global_xai_grok_cli
