"""
OpenAI Codex CLI Integration (compatibility facade)

Uses current ``codex exec`` semantics rather than the obsolete
``openai api completions.create`` command. Operator command overrides are
argv-only (shell-free). Side-effecting exec runs disable generic cache/retry.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Union

from .base_cli_wrapper import (
    BaseCLIWrapper,
    parse_argv_override,
    resolve_command_override_from_env,
)
from ..common.llm_cache import LLMAPICache, get_llm_cache

logger = logging.getLogger(__name__)

# Env vars (first wins) for full argv override of the codex binary/prefix.
_CODEX_CMD_ENV = (
    "ipfs_accelerate_py_CODEX_CLI_CMD",
    "IPFS_ACCELERATE_PY_CODEX_CLI_CMD",
    "IPFS_DATASETS_PY_CODEX_CLI_CMD",
)


def _default_codex_path() -> str:
    return shutil.which("codex") or "codex"


def build_codex_exec_argv(
    *,
    base_argv: Sequence[str],
    model: str,
    prompt: str,
    last_message_path: str,
    sandbox: Optional[str] = None,
    skip_git_repo_check: bool = True,
    json_mode: bool = False,
) -> List[str]:
    """Build ``codex exec`` argv matching llm_router production semantics.

    The prompt is *not* placed on the argv; callers must pass it via stdin
    (``cmd.append("-")``). Dynamic values remain single argv entries.
    """
    _ = prompt  # documented for callers; supplied via stdin
    cmd: List[str] = list(base_argv)
    # If operator override is a bare binary, append the exec subcommand.
    # If override already includes "exec", do not duplicate it.
    lowered = [part.lower() for part in cmd]
    if "exec" not in lowered:
        cmd.append("exec")
    if skip_git_repo_check:
        cmd.append("--skip-git-repo-check")
    # Treat sandbox "auto"/empty as omit-flag (CLI chooses default).
    if sandbox and str(sandbox).strip().lower() not in {"", "auto"}:
        cmd.extend(["--sandbox", str(sandbox).strip()])
    if model:
        cmd.extend(["-m", str(model)])
    cmd.extend(["--output-last-message", last_message_path])
    if json_mode:
        cmd.append("--json")
    cmd.append("-")
    return cmd


class OpenAICodexCLIIntegration(BaseCLIWrapper):
    """
    OpenAI Codex CLI integration with common cache infrastructure.

    Command contract: ``codex exec`` (stdin prompt, ``--output-last-message``).
    """

    def __init__(
        self,
        codex_path: str = "codex",
        enable_cache: bool = True,
        cache: Optional[LLMAPICache] = None,
        *,
        command_override: Optional[Union[str, Sequence[str]]] = None,
        verify_on_init: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize OpenAI Codex CLI integration.

        Args:
            codex_path: Path to ``codex`` executable (not the obsolete ``openai`` CLI)
            enable_cache: Whether to enable caching for non-side-effecting reads
            cache: Custom cache instance (uses LLM cache if None)
            command_override: Optional argv prefix (env override wins if unset)
            verify_on_init: When True, probe ``--version`` (default False)
            **kwargs: Additional arguments for BaseCLIWrapper
        """
        if cache is None:
            cache = get_llm_cache("openai_codex")

        override = parse_argv_override(command_override)
        if override is None:
            override = resolve_command_override_from_env(*_CODEX_CMD_ENV)

        # codex exec can write files / run tools depending on sandbox — treat
        # generation as side-effecting so generic cache/retry stay off.
        super().__init__(
            cli_path=codex_path if codex_path else _default_codex_path(),
            cache=cache,
            enable_cache=enable_cache,
            verify_on_init=verify_on_init,
            command_override=override,
            side_effecting_default=True,
            **kwargs,
        )

    def get_tool_name(self) -> str:
        return "OpenAI Codex CLI"

    def generate_code(
        self,
        prompt: str,
        model: str = "chatgpt-5.6-terra",
        temperature: float = 0.0,
        *,
        sandbox: Optional[str] = None,
        skip_git_repo_check: bool = True,
        json_mode: bool = False,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate code from prompt via ``codex exec``.

        Args:
            prompt: Code generation prompt (passed on stdin, not shell-interpolated)
            model: Model to use (``-m``)
            temperature: Accepted for API compatibility; Codex CLI ignores it
            sandbox: Optional sandbox mode (omit / ``auto`` skips the flag)
            skip_git_repo_check: Pass ``--skip-git-repo-check`` (default True)
            json_mode: Pass ``--json`` when tracing is needed
            timeout: Command timeout override
            **kwargs: Additional arguments (ignored for command construction)

        Returns:
            Command result dict with generated code in ``stdout`` / ``text``
        """
        _ = temperature
        _ = kwargs
        env_sandbox = sandbox
        if env_sandbox is None:
            env_sandbox = os.getenv("ipfs_accelerate_py_CODEX_SANDBOX", "auto")

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False
        ) as handle:
            last_msg_path = handle.name

        try:
            full_argv = build_codex_exec_argv(
                base_argv=self._base_argv(),
                model=model,
                prompt=prompt,
                last_message_path=last_msg_path,
                sandbox=env_sandbox,
                skip_git_repo_check=skip_git_repo_check,
                json_mode=json_mode,
            )
            # Side-effecting: no cache, no retry.
            response = self._run_command_with_retry(
                args=[],
                operation="codex_exec",
                stdin=str(prompt),
                timeout=timeout,
                side_effecting=True,
                full_argv=full_argv,
                prompt=prompt,
                model=model,
            )

            text_out = ""
            try:
                with open(last_msg_path, "r", encoding="utf-8", errors="replace") as fh:
                    text_out = fh.read().strip()
            except OSError:
                text_out = ""

            if text_out:
                response["stdout"] = text_out
                response["text"] = text_out
            else:
                response["text"] = (response.get("stdout") or "").strip()

            response["command_contract"] = "codex exec"
            response["model"] = model
            return response
        finally:
            try:
                os.unlink(last_msg_path)
            except OSError:
                pass


# Global instance
_global_openai_codex_cli: Optional[OpenAICodexCLIIntegration] = None


def get_openai_codex_cli_integration() -> OpenAICodexCLIIntegration:
    """Get or create the global OpenAI Codex CLI integration instance."""
    global _global_openai_codex_cli

    if _global_openai_codex_cli is None:
        _global_openai_codex_cli = OpenAICodexCLIIntegration()

    return _global_openai_codex_cli


def reset_openai_codex_cli_integration() -> None:
    """Test helper: drop the module-level singleton."""
    global _global_openai_codex_cli
    _global_openai_codex_cli = None
