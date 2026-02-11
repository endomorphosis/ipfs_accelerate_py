"""
Dual Mode CLI/API Wrapper

Provides a wrapper that tries CLI execution first, then falls back to Python SDK.
This enables unified interfaces that can work with or without CLI tools installed.
"""

import logging
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

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
    """
    
    def __init__(
        self,
        cli_path: Optional[str] = None,
        api_key: Optional[str] = None,
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
            api_key: API key for SDK mode (from secrets manager if None)
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
