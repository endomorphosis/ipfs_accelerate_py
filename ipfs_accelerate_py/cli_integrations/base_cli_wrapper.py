"""
Base CLI Wrapper with Common Cache Integration

Provides a base class for all CLI tool wrappers to use the common cache infrastructure.
"""

import json
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path

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


class BaseCLIWrapper(ABC):
    """
    Base class for CLI wrappers with common cache integration.
    
    All CLI wrappers should inherit from this class to get:
    - CID-based caching
    - Retry logic with exponential backoff
    - Command execution helpers
    - Statistics tracking
    """
    
    def __init__(
        self,
        cli_path: str,
        cache: BaseAPICache,
        enable_cache: bool = True,
        default_timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize base CLI wrapper.
        
        Args:
            cli_path: Path to CLI executable
            cache: Cache instance to use
            enable_cache: Whether to enable caching
            default_timeout: Default command timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.cli_path = cli_path
        self.cache = cache
        self.enable_cache = enable_cache
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        
        self._verify_installation()
    
    @abstractmethod
    def get_tool_name(self) -> str:
        """Get the name of this CLI tool."""
        pass
    
    def _verify_installation(self) -> None:
        """Verify that the CLI tool is installed."""
        try:
            result = subprocess.run(
                [self.cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"{self.get_tool_name()} version: {result.stdout.strip()}")
            else:
                logger.warning(f"{self.get_tool_name()} verification returned non-zero: {result.returncode}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Could not verify {self.get_tool_name()} installation: {e}")
    
    def _run_command_with_retry(
        self,
        args: List[str],
        operation: str,
        stdin: Optional[str] = None,
        timeout: Optional[int] = None,
        **cache_params
    ) -> Dict[str, Any]:
        """
        Run a command with retry logic and caching.
        
        Args:
            args: Command arguments
            operation: Operation name for caching
            stdin: Optional stdin input
            timeout: Command timeout (uses default if None)
            **cache_params: Additional parameters for cache key
            
        Returns:
            Dict with stdout, stderr, returncode, and metadata
        """
        timeout = timeout or self.default_timeout
        
        # Check cache first if enabled
        if self.enable_cache:
            # Build cache key parameters
            cache_key_params = {
                "args": " ".join(args),
                **cache_params
            }
            
            cached = self.cache.get(operation, **cache_key_params)
            if cached is not None:
                logger.debug(f"Cache HIT for {operation}")
                return cached
        
        # Execute command with retry
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = subprocess.run(
                    [self.cli_path] + args,
                    capture_output=True,
                    text=True,
                    input=stdin,
                    timeout=timeout
                )
                
                response = {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "success": result.returncode == 0,
                    "attempts": attempt + 1,
                    "tool": self.get_tool_name()
                }
                
                # Cache successful responses
                if self.enable_cache and result.returncode == 0:
                    cache_key_params = {
                        "args": " ".join(args),
                        **cache_params
                    }
                    self.cache.put(operation, response, **cache_key_params)
                
                return response
                
            except subprocess.TimeoutExpired as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = min(2 ** attempt, 30)  # Exponential backoff, max 30s
                    logger.warning(f"{self.get_tool_name()} timeout on attempt {attempt + 1}, retrying in {delay}s")
                    time.sleep(delay)
            except Exception as e:
                last_error = e
                logger.error(f"{self.get_tool_name()} command failed: {e}")
                break
        
        # All retries failed
        return {
            "stdout": "",
            "stderr": str(last_error),
            "returncode": -1,
            "success": False,
            "attempts": self.max_retries,
            "error": str(last_error),
            "tool": self.get_tool_name()
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
