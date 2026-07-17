"""
Dual Mode CLI/API Wrapper

Provides a wrapper that tries CLI execution first, then falls back to Python SDK.
This enables unified interfaces that can work with or without CLI tools installed.

Multi-user / parallel execution
--------------------------------
Pass ``api_keys`` (a list) instead of ``api_key`` (a single string) to enable
round-robin key selection across concurrent requests::

    wrapper = MyIntegration(api_keys=["key-1", "key-2", "key-3"])
    key = wrapper.get_api_key(user_id="alice")  # pinned for "alice"

Async support (Trio / Hypercorn)
---------------------------------
All blocking SDK calls can be offloaded to a thread via
``_aexecute_with_fallback()``, which uses ``anyio.to_thread.run_sync`` and
therefore works on both Trio and asyncio event loops::

    result = await wrapper._aexecute_with_fallback(
        sdk_func=wrapper._chat_sdk,
        operation="chat",
        message="Hello",
        model="grok-3",
        temperature=0.0,
    )
"""

import functools
import logging
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from .base_cli_wrapper import BaseCLIWrapper
from ..common.base_cache import BaseAPICache
from ..common.secrets_manager import get_global_secrets_manager

logger = logging.getLogger(__name__)


class DualModeWrapper(ABC):
    """
    Base class for dual-mode CLI/SDK wrappers.
    
    Features:
    - Tries CLI execution first if available
    - Falls back to Python SDK if CLI not found or fails
    - Unified caching for both modes
    - Credential management via secrets manager
    - Multi-key pool support for parallel multi-user execution
    - Async execution via anyio (Trio / asyncio compatible)
    """
    
    def __init__(
        self,
        cli_path: Optional[str] = None,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        cache: Optional[BaseAPICache] = None,
        enable_cache: bool = True,
        prefer_cli: bool = True,
        default_timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize dual-mode wrapper.
        
        Args:
            cli_path: Path to CLI executable (auto-detected if None)
            api_key: Single API key for SDK mode (from secrets manager if None)
            api_keys: List of API keys for multi-user round-robin pool.
                      When provided, ``api_key`` is used only as a fallback if
                      the pool is exhausted.  The pool is stored as
                      ``self.key_pool``.
            cache: Cache instance to use
            enable_cache: Whether to enable caching
            prefer_cli: Whether to prefer CLI over SDK when both available
            default_timeout: Default command timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.enable_cache = enable_cache
        self.prefer_cli = prefer_cli
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.cache = cache
        
        # Get secrets manager for credential retrieval
        self.secrets_manager = get_global_secrets_manager()
        
        # Detect CLI availability
        if cli_path is None:
            cli_path = self._detect_cli_path()
        
        self.cli_path = cli_path
        self.cli_available = self._check_cli_available()
        
        # Get API key from secrets manager if not provided
        if api_key is None:
            api_key = self._get_api_key_from_secrets()
        
        self.api_key = api_key

        # Build key pool when multiple keys are supplied
        if api_keys:
            from .api_key_pool import ApiKeyPool
            # Include single api_key in pool if not already present
            all_keys = list(api_keys)
            if api_key and api_key not in all_keys:
                all_keys.insert(0, api_key)
            self.key_pool: Optional[Any] = ApiKeyPool(all_keys)
        else:
            self.key_pool = None
        
        # Initialize SDK client (lazy loaded)
        self._sdk_client = None
        
        # Log mode
        if self.cli_available and self.prefer_cli:
            logger.info(f"{self.get_tool_name()}: CLI mode enabled (with SDK fallback)")
        elif self.cli_available:
            logger.info(f"{self.get_tool_name()}: SDK mode enabled (with CLI fallback)")
        else:
            logger.info(f"{self.get_tool_name()}: SDK-only mode (CLI not available)")
    
    @abstractmethod
    def get_tool_name(self) -> str:
        """Get the name of this tool."""
        pass
    
    @abstractmethod
    def _detect_cli_path(self) -> Optional[str]:
        """
        Auto-detect CLI executable path.
        
        Returns:
            Path to CLI or None if not found
        """
        pass
    
    @abstractmethod
    def _get_api_key_from_secrets(self) -> Optional[str]:
        """
        Get API key from secrets manager.
        
        Returns:
            API key or None
        """
        pass
    
    @abstractmethod
    def _create_sdk_client(self):
        """
        Create and return SDK client.
        
        Returns:
            SDK client instance
        """
        pass

    # ------------------------------------------------------------------
    # Multi-key support
    # ------------------------------------------------------------------

    def get_api_key(self, user_id: Optional[str] = None) -> Optional[str]:
        """
        Return the API key to use for a request.

        When a ``key_pool`` is configured (``api_keys`` was passed at
        construction) the pool's round-robin / pinning logic is used.
        Otherwise the single ``self.api_key`` is returned.

        Parameters
        ----------
        user_id:
            Opaque identifier for the end-user.  When provided and a pool is
            active, the user is pinned to a consistent key for the lifetime
            of the pool.

        Returns
        -------
        str | None
            API key, or ``None`` if none is configured.
        """
        if self.key_pool is not None:
            try:
                return self.key_pool.get_key(user_id=user_id)
            except RuntimeError:
                pass  # pool empty – fall through to self.api_key
        return self.api_key

    # ------------------------------------------------------------------
    # CLI availability
    # ------------------------------------------------------------------

    def _check_cli_available(self) -> bool:
        """Check if CLI tool is available."""
        if not self.cli_path:
            return False
        
        try:
            result = subprocess.run(
                [self.cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"{self.get_tool_name()} CLI found: {self.cli_path}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            pass
        
        return False
    
    def _get_sdk_client(self):
        """Get or create SDK client."""
        if self._sdk_client is None:
            self._sdk_client = self._create_sdk_client()
        return self._sdk_client

    # ------------------------------------------------------------------
    # Synchronous execution with CLI/SDK fallback
    # ------------------------------------------------------------------
    
    def _execute_with_fallback(
        self,
        cli_func: Optional[Callable] = None,
        sdk_func: Optional[Callable] = None,
        operation: str = "execute",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute operation with CLI/SDK fallback.
        
        Args:
            cli_func: Function to execute in CLI mode
            sdk_func: Function to execute in SDK mode
            operation: Operation name for logging
            **kwargs: Arguments to pass to both functions
            
        Returns:
            Result dict with data and metadata
        """
        errors = []
        
        # Determine execution order
        if self.cli_available and self.prefer_cli:
            first_mode = ("CLI", cli_func)
            second_mode = ("SDK", sdk_func)
        else:
            first_mode = ("SDK", sdk_func)
            second_mode = ("CLI", cli_func) if self.cli_available else (None, None)
        
        # Try first mode
        if first_mode[1] is not None:
            try:
                logger.debug(f"Trying {first_mode[0]} mode for {operation}")
                result = first_mode[1](**kwargs)
                result["mode"] = first_mode[0]
                return result
            except Exception as e:
                logger.warning(f"{first_mode[0]} mode failed for {operation}: {e}")
                errors.append(f"{first_mode[0]}: {e}")
        
        # Try second mode as fallback
        if second_mode[0] is not None and second_mode[1] is not None:
            try:
                logger.debug(f"Falling back to {second_mode[0]} mode for {operation}")
                result = second_mode[1](**kwargs)
                result["mode"] = second_mode[0]
                result["fallback"] = True
                return result
            except Exception as e:
                logger.error(f"{second_mode[0]} mode also failed for {operation}: {e}")
                errors.append(f"{second_mode[0]}: {e}")
        
        # Both modes failed
        error_msg = f"Both CLI and SDK modes failed for {operation}: {'; '.join(errors)}"
        raise RuntimeError(error_msg)

    # ------------------------------------------------------------------
    # Async execution (Trio / asyncio via anyio)
    # ------------------------------------------------------------------

    async def _aexecute_with_fallback(
        self,
        cli_func: Optional[Callable] = None,
        sdk_func: Optional[Callable] = None,
        operation: str = "execute",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Async version of :meth:`_execute_with_fallback`.

        Blocking SDK / CLI calls are offloaded to a worker thread via
        ``anyio.to_thread.run_sync`` so they do not stall the Trio or
        asyncio event loop.  This makes it safe to ``await`` this method
        directly from MCP tool handlers running under Hypercorn/Trio.

        Falls back to the synchronous path when ``anyio`` is not installed.

        Parameters
        ----------
        cli_func:
            Callable executed in CLI mode.
        sdk_func:
            Callable executed in SDK mode.
        operation:
            Name used in log messages.
        **kwargs:
            Forwarded verbatim to whichever function is invoked.

        Returns
        -------
        dict
            Same structure as :meth:`_execute_with_fallback`.
        """
        try:
            import anyio
        except ImportError:
            # anyio not installed – run synchronously (blocks event loop)
            logger.debug("anyio not available; running %s synchronously", operation)
            return self._execute_with_fallback(
                cli_func=cli_func, sdk_func=sdk_func, operation=operation, **kwargs
            )

        errors: list[str] = []

        if self.cli_available and self.prefer_cli:
            first_mode = ("CLI", cli_func)
            second_mode = ("SDK", sdk_func)
        else:
            first_mode = ("SDK", sdk_func)
            second_mode = ("CLI", cli_func) if self.cli_available else (None, None)

        if first_mode[1] is not None:
            try:
                logger.debug("Trying async %s mode for %s", first_mode[0], operation)
                fn = functools.partial(first_mode[1], **kwargs)
                result = await anyio.to_thread.run_sync(fn)
                result["mode"] = first_mode[0]
                return result
            except Exception as exc:
                logger.warning("Async %s mode failed for %s: %s", first_mode[0], operation, exc)
                errors.append(f"{first_mode[0]}: {exc}")

        if second_mode[0] is not None and second_mode[1] is not None:
            try:
                logger.debug("Falling back to async %s mode for %s", second_mode[0], operation)
                fn = functools.partial(second_mode[1], **kwargs)
                result = await anyio.to_thread.run_sync(fn)
                result["mode"] = second_mode[0]
                result["fallback"] = True
                return result
            except Exception as exc:
                logger.error("Async %s mode also failed for %s: %s", second_mode[0], operation, exc)
                errors.append(f"{second_mode[0]}: {exc}")

        raise RuntimeError(
            f"Both CLI and SDK async modes failed for {operation}: {'; '.join(errors)}"
        )


def detect_cli_tool(tool_names: list, version_args: list = ["--version"]) -> Optional[str]:
    """
    Detect CLI tool in system PATH.
    
    Args:
        tool_names: List of possible tool names to try
        version_args: Arguments to verify tool (default: --version)
        
    Returns:
        Path to tool or None if not found
    """
    for tool_name in tool_names:
        tool_path = shutil.which(tool_name)
        if tool_path:
            try:
                # Verify it works
                result = subprocess.run(
                    [tool_path] + version_args,
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return tool_path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
    
    return None
