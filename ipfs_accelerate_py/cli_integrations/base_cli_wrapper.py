"""
Base CLI Wrapper with Common Cache Integration

Provides a base class for all CLI tool wrappers to use the common cache
infrastructure, with side-effect-aware cache/retry policy and argv-only
operator command overrides (never shell=True).
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

from ..common.base_cache import BaseAPICache

try:
    from ...common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from ..common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        try:
            from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
        except ImportError:
            HAVE_STORAGE_WRAPPER = False

if HAVE_STORAGE_WRAPPER:
    try:
        _storage = get_storage_wrapper(auto_detect_ci=True)
    except Exception:
        _storage = None
else:
    _storage = None

logger = logging.getLogger(__name__)


def parse_argv_override(
    value: Optional[Union[str, Sequence[str]]],
) -> Optional[List[str]]:
    """Parse an operator command override as argv-only (shell-free).

    Strings are split with :func:`shlex.split` so spaces and metacharacters in
    a single token stay one argv entry. Sequences are coerced to ``str`` items.
    Empty or whitespace-only values yield ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parts = shlex.split(text)
        return parts or None
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value if str(item).strip()]
        return parts or None
    raise TypeError(
        f"command override must be str or sequence of str, got {type(value).__name__}"
    )


def resolve_command_override_from_env(
    *env_names: str,
    environ: Optional[Any] = None,
) -> Optional[List[str]]:
    """Return the first non-empty argv override from environment variables."""
    env = os.environ if environ is None else environ
    for name in env_names:
        if not name:
            continue
        raw = env.get(name)
        if raw is None or not str(raw).strip():
            continue
        parsed = parse_argv_override(str(raw))
        if parsed:
            return parsed
    return None


class BaseCLIWrapper(ABC):
    """
    Base class for CLI wrappers with common cache integration.

    All CLI wrappers should inherit from this class to get:
    - CID-based caching (disabled for side-effecting operations)
    - Retry logic with exponential backoff (disabled for side-effecting ops)
    - Command execution helpers (``shell=False`` only)
    - Optional lazy installation verification (no process probe on init by default)
    - Operator command overrides as argv-only configuration
    """

    def __init__(
        self,
        cli_path: str,
        cache: BaseAPICache,
        enable_cache: bool = True,
        default_timeout: int = 60,
        max_retries: int = 3,
        *,
        verify_on_init: bool = False,
        command_override: Optional[Union[str, Sequence[str]]] = None,
        side_effecting_default: bool = False,
    ):
        """
        Initialize base CLI wrapper.

        Args:
            cli_path: Path to CLI executable (or first argv element)
            cache: Cache instance to use
            enable_cache: Whether to enable caching for non-side-effecting ops
            default_timeout: Default command timeout in seconds
            max_retries: Maximum number of retry attempts for non-side-effecting ops
            verify_on_init: When True, probe ``--version`` during construction.
                Default False so listing/import paths do not start processes.
            command_override: Optional full argv prefix replacing ``cli_path``
                (string is shlex-split; never executed via a shell).
            side_effecting_default: When True, operations default to no cache/retry.
        """
        self.cli_path = cli_path
        self.cache = cache
        self.enable_cache = enable_cache
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.side_effecting_default = bool(side_effecting_default)
        self._verified: Optional[bool] = None
        self._command_override = parse_argv_override(command_override)

        if verify_on_init:
            self._verify_installation()

    @abstractmethod
    def get_tool_name(self) -> str:
        """Get the name of this CLI tool."""
        pass

    def _base_argv(self) -> List[str]:
        """Return the argv prefix for this tool (override or cli_path)."""
        if self._command_override:
            return list(self._command_override)
        return [self.cli_path]

    def set_command_override(
        self, value: Optional[Union[str, Sequence[str]]]
    ) -> None:
        """Replace the operator command override (argv-only, shell-free)."""
        self._command_override = parse_argv_override(value)

    def _verify_installation(self) -> None:
        """Verify that the CLI tool is installed (explicit / opt-in probe)."""
        argv = self._base_argv() + ["--version"]
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=5,
                shell=False,
            )
            if result.returncode == 0:
                logger.info(
                    f"{self.get_tool_name()} version: {result.stdout.strip()}"
                )
                self._verified = True
            else:
                logger.warning(
                    f"{self.get_tool_name()} verification returned non-zero: "
                    f"{result.returncode}"
                )
                self._verified = False
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.warning(
                f"Could not verify {self.get_tool_name()} installation: {e}"
            )
            self._verified = False

    def is_available(self, *, probe: bool = False) -> bool:
        """Return whether the CLI looks available without (by default) running it.

        When ``probe`` is False, only PATH / executable existence is checked.
        When True, runs a ``--version`` probe once and caches the result.
        """
        if probe:
            if self._verified is None:
                self._verify_installation()
            return bool(self._verified)

        if self._command_override:
            exe = self._command_override[0]
        else:
            exe = self.cli_path
        if not exe:
            return False
        # Absolute or relative path
        if os.path.sep in exe or (os.path.altsep and os.path.altsep in exe):
            return os.path.isfile(exe) and os.access(exe, os.X_OK)
        # Special-case npx: treated as available (package runner may download
        # on demand), matching llm_router._cli_available.
        if exe in {"npx", "npx.cmd"}:
            return True
        return shutil.which(exe) is not None

    def _run_command_with_retry(
        self,
        args: List[str],
        operation: str,
        stdin: Optional[str] = None,
        timeout: Optional[int] = None,
        *,
        side_effecting: Optional[bool] = None,
        full_argv: Optional[Sequence[str]] = None,
        **cache_params: Any,
    ) -> Dict[str, Any]:
        """
        Run a command with optional retry logic and caching.

        Side-effecting operations never use the generic response cache and are
        not blindly retried (exactly one attempt).

        Args:
            args: Command arguments appended after the base argv
            operation: Operation name for caching
            stdin: Optional stdin input
            timeout: Command timeout (uses default if None)
            side_effecting: Override default side-effect policy for this call
            full_argv: When set, run this argv as-is (must be shell-free list)
            **cache_params: Additional parameters for cache key

        Returns:
            Dict with stdout, stderr, returncode, and metadata
        """
        timeout = timeout or self.default_timeout
        is_side_effecting = (
            self.side_effecting_default
            if side_effecting is None
            else bool(side_effecting)
        )
        allow_cache = bool(self.enable_cache) and not is_side_effecting
        attempts_allowed = 1 if is_side_effecting else max(1, int(self.max_retries))

        if full_argv is not None:
            cmd = [str(part) for part in full_argv]
        else:
            cmd = self._base_argv() + list(args)

        # Check cache first if enabled and not side-effecting
        if allow_cache:
            cache_key_params = {
                "args": " ".join(cmd),
                **cache_params,
            }
            cached = self.cache.get(operation, **cache_key_params)
            if cached is not None:
                logger.debug(f"Cache HIT for {operation}")
                return cached

        last_error: Optional[BaseException] = None
        for attempt in range(attempts_allowed):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    input=stdin,
                    timeout=timeout,
                    shell=False,
                )

                response: Dict[str, Any] = {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "success": result.returncode == 0,
                    "attempts": attempt + 1,
                    "tool": self.get_tool_name(),
                    "side_effecting": is_side_effecting,
                    "argv": list(cmd),
                    "cached": False,
                }

                # Cache successful non-side-effecting responses only
                if allow_cache and result.returncode == 0:
                    cache_key_params = {
                        "args": " ".join(cmd),
                        **cache_params,
                    }
                    self.cache.put(operation, response, **cache_key_params)

                return response

            except subprocess.TimeoutExpired as e:
                last_error = e
                if attempt < attempts_allowed - 1:
                    delay = min(2 ** attempt, 30)
                    logger.warning(
                        f"{self.get_tool_name()} timeout on attempt "
                        f"{attempt + 1}, retrying in {delay}s"
                    )
                    time.sleep(delay)
            except Exception as e:
                last_error = e
                logger.error(f"{self.get_tool_name()} command failed: {e}")
                break

        return {
            "stdout": "",
            "stderr": str(last_error),
            "returncode": -1,
            "success": False,
            "attempts": attempts_allowed if is_side_effecting else self.max_retries,
            "error": str(last_error),
            "tool": self.get_tool_name(),
            "side_effecting": is_side_effecting,
            "argv": list(cmd),
            "cached": False,
        }

    def _parse_json_output(self, output: str) -> Optional[Any]:
        """
        Parse JSON output from CLI command.

        Args:
            output: Command output to parse

        Returns:
            Parsed JSON or None if parsing fails
        """
        try:
            return json.loads(output)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON output: {e}")
            return None
