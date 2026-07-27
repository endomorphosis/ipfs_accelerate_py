"""Goose CLI integration — peer of Codex/Copilot, Meta Spark backend by default."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .dual_mode_wrapper import DualModeWrapper, detect_cli_tool
from ..common.llm_cache import get_llm_cache


class GooseCLIIntegration(DualModeWrapper):
    def __init__(
        self,
        goose_path: Optional[str] = None,
        enable_cache: bool = True,
        cache: Any = None,
        **kwargs: Any,
    ) -> None:
        if cache is None and enable_cache:
            cache = get_llm_cache("goose_cli")
        super().__init__(
            cli_path=goose_path or detect_cli_tool(["goose"]) or "goose",
            cache=cache,
            enable_cache=enable_cache,
            **kwargs,
        )

    def get_tool_name(self) -> str:
        return "Goose CLI"

    def _detect_cli_path(self) -> Optional[str]:
        return detect_cli_tool(["goose"])

    def _get_api_key_from_secrets(self) -> Optional[str]:
        try:
            from ..common.meta_model_api import resolve_meta_model_api_key

            return resolve_meta_model_api_key()
        except Exception:
            return None

    def _create_sdk_client(self) -> Any:
        from ..llm_router import get_llm_provider

        return get_llm_provider("goose_cli")

    def chat(self, prompt: str, *, model: Optional[str] = None, timeout: float = 180.0, **kwargs: Any) -> Dict[str, Any]:
        from ..llm_router import get_llm_provider

        text = get_llm_provider("goose_cli").generate(
            prompt, model_name=model, timeout=timeout, agent=False, **kwargs
        )
        return {"mode": "cli", "provider": "goose_cli", "text": text, "success": True}

    def agent(
        self,
        prompt: str,
        *,
        workspace: str,
        model: Optional[str] = None,
        timeout: float = 600.0,
        max_turns: int = 40,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from ..llm_router import get_llm_provider

        text = get_llm_provider("goose_cli").generate(
            prompt,
            model_name=model,
            timeout=timeout,
            agent=True,
            workspace=workspace,
            max_turns=max_turns,
            with_developer=True,
            **kwargs,
        )
        return {
            "mode": "agent",
            "provider": "goose_cli",
            "workspace": workspace,
            "text": text,
            "success": True,
        }


_global_goose_cli: Optional[GooseCLIIntegration] = None


def get_goose_cli_integration() -> GooseCLIIntegration:
    global _global_goose_cli
    if _global_goose_cli is None:
        _global_goose_cli = GooseCLIIntegration()
    return _global_goose_cli
