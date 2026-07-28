"""
GitHub Copilot CLI Integration (compatibility facade)

Delegates to the production Copilot command contract used by llm_router
(``npx --yes @github/copilot`` / ``copilot`` with ``--silent --stream off``
and ``-i`` / ``--prompt``), not the obsolete ``github-copilot-cli`` suggest
syntax. Operator command overrides are argv-only (shell-free).
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Union

from .base_cli_wrapper import (
    BaseCLIWrapper,
    parse_argv_override,
    resolve_command_override_from_env,
)
from ..common.llm_cache import get_llm_cache
from ..common.base_cache import BaseAPICache

logger = logging.getLogger(__name__)

_COPILOT_CMD_ENV = (
    "ipfs_accelerate_py_COPILOT_CLI_CMD",
    "IPFS_ACCELERATE_PY_COPILOT_CLI_CMD",
    "IPFS_DATASETS_PY_COPILOT_CLI_CMD",
)

# Production default matching llm_router._get_copilot_cli_provider.
DEFAULT_COPILOT_CMD: List[str] = ["npx", "--yes", "@github/copilot"]


def _default_copilot_base_argv() -> List[str]:
    standalone = shutil.which("copilot")
    if standalone:
        return [standalone]
    return list(DEFAULT_COPILOT_CMD)


def build_copilot_suggest_argv(
    *,
    base_argv: Sequence[str],
    prompt: str,
    model: str = "gpt-5-mini",
    use_prompt_flag: bool = False,
) -> List[str]:
    """Build production Copilot CLI argv for a one-shot prompt.

    Matches llm_router non-template mode: ``--silent --stream off --model … -i``.
    When ``use_prompt_flag`` is True (standalone file-mode style), uses
    ``--prompt`` instead of ``-i``.
    """
    cmd = list(base_argv)
    cmd.extend(
        [
            "--silent",
            "--stream",
            "off",
            "--model",
            str(model),
        ]
    )
    if use_prompt_flag:
        cmd.extend(["--prompt", str(prompt)])
    else:
        cmd.extend(["-i", str(prompt)])
    return cmd


def build_copilot_explain_argv(
    *,
    base_argv: Sequence[str],
    command: str,
    model: str = "gpt-5-mini",
) -> List[str]:
    """Explain a shell command via the production Copilot prompt contract."""
    explain_prompt = f"Explain this shell command in detail:\n\n{command}"
    return build_copilot_suggest_argv(
        base_argv=base_argv,
        prompt=explain_prompt,
        model=model,
        use_prompt_flag=False,
    )


class CopilotCLIIntegration(BaseCLIWrapper):
    """
    GitHub Copilot CLI integration with common cache infrastructure.

    Uses the production ``@github/copilot`` / ``copilot`` command contract.
    """

    def __init__(
        self,
        copilot_path: str = "npx",
        enable_cache: bool = True,
        cache: Optional[BaseAPICache] = None,
        *,
        command_override: Optional[Union[str, Sequence[str]]] = None,
        verify_on_init: bool = False,
        default_model: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize Copilot CLI integration.

        Args:
            copilot_path: Legacy path hint; ignored when a production default
                or env override is used. Kept for call-site compatibility.
            enable_cache: Whether to enable caching for non-side-effecting ops
            cache: Custom cache instance (uses LLM cache if None)
            command_override: Optional full argv prefix (env wins if unset)
            verify_on_init: When True, probe ``--version`` (default False)
            default_model: Default ``--model`` value
            **kwargs: Additional arguments for BaseCLIWrapper
        """
        if cache is None:
            cache = get_llm_cache("copilot")

        override = parse_argv_override(command_override)
        if override is None:
            override = resolve_command_override_from_env(*_COPILOT_CMD_ENV)

        if override is not None:
            base_path = override[0]
            resolved_override = override
        else:
            # Prefer production defaults over the obsolete github-copilot-cli path.
            resolved_override = _default_copilot_base_argv()
            base_path = resolved_override[0]
            _ = copilot_path  # retained for signature compatibility only

        self.default_model = (
            (default_model or "").strip()
            or os.getenv("ipfs_accelerate_py_COPILOT_CLI_MODEL", "").strip()
            or os.getenv("IPFS_ACCELERATE_PY_COPILOT_CLI_MODEL", "").strip()
            or "gpt-5-mini"
        )

        # Suggest/explain may still mutate local state via tools; default to
        # side-effecting so generic cache/retry stay off unless callers opt in.
        super().__init__(
            cli_path=base_path,
            cache=cache,
            enable_cache=enable_cache,
            verify_on_init=verify_on_init,
            command_override=resolved_override,
            side_effecting_default=True,
            **kwargs,
        )

    def get_tool_name(self) -> str:
        return "GitHub Copilot CLI"

    def suggest_command(
        self,
        prompt: str,
        shell: str = "bash",
        *,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        side_effecting: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Get command suggestion from Copilot using the production CLI contract.

        Args:
            prompt: Natural language description of desired command
            shell: Target shell (included in the prompt text, not a CLI flag)
            model: Optional model override
            timeout: Command timeout override
            side_effecting: Disable generic cache/retry when True (default)
            **kwargs: Additional arguments (ignored for argv construction)

        Returns:
            Command result dict with suggestion
        """
        _ = kwargs
        model_name = (model or self.default_model).strip() or "gpt-5-mini"
        composed = (
            f"Suggest a {shell} shell command for the following request. "
            f"Reply with the command and a brief explanation.\n\n{prompt}"
        )
        full_argv = build_copilot_suggest_argv(
            base_argv=self._base_argv(),
            prompt=composed,
            model=model_name,
        )
        response = self._run_command_with_retry(
            args=[],
            operation="suggest_command",
            stdin=None,
            timeout=timeout,
            side_effecting=side_effecting,
            full_argv=full_argv,
            prompt=prompt,
            shell=shell,
            model=model_name,
        )
        response["command_contract"] = "copilot_production"
        response["text"] = (response.get("stdout") or "").strip()
        response["shell"] = shell
        return response

    def explain_command(
        self,
        command: str,
        *,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        side_effecting: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Get explanation of a command via the production Copilot contract.

        Args:
            command: Command to explain
            model: Optional model override
            timeout: Command timeout override
            side_effecting: Disable generic cache/retry when True (default)
            **kwargs: Additional arguments (ignored for argv construction)

        Returns:
            Command result dict with explanation
        """
        _ = kwargs
        model_name = (model or self.default_model).strip() or "gpt-5-mini"
        full_argv = build_copilot_explain_argv(
            base_argv=self._base_argv(),
            command=command,
            model=model_name,
        )
        response = self._run_command_with_retry(
            args=[],
            operation="explain_command",
            stdin=None,
            timeout=timeout,
            side_effecting=side_effecting,
            full_argv=full_argv,
            command=command,
            model=model_name,
        )
        response["command_contract"] = "copilot_production"
        response["text"] = (response.get("stdout") or "").strip()
        return response


# Global instance
_global_copilot_cli: Optional[CopilotCLIIntegration] = None


def get_copilot_cli_integration() -> CopilotCLIIntegration:
    """Get or create the global Copilot CLI integration instance."""
    global _global_copilot_cli

    if _global_copilot_cli is None:
        _global_copilot_cli = CopilotCLIIntegration()

    return _global_copilot_cli


def reset_copilot_cli_integration() -> None:
    """Test helper: drop the module-level singleton."""
    global _global_copilot_cli
    _global_copilot_cli = None
