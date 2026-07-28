"""
CLI Endpoint Adapters for IPFS Accelerate MCP Server

This module provides adapters for CLI-based AI tools to integrate them
into the IPFS Accelerate multiplexing and queue system.

Supported CLI Tools:
- Claude Code (Anthropic)
- OpenAI Codex CLI
- Google Gemini CLI
- VSCode CLI (GitHub Copilot)


.. deprecated::
    Registration, listing, and execution are owned by the canonical factory at
    ``ipfs_accelerate_py.cli_runtime.endpoints``. Concrete adapter classes remain
    here for import compatibility; ``register_cli_endpoint``,
    ``list_cli_endpoints``, ``get_cli_endpoint``, ``execute_cli_inference``, and
    ``CLI_ADAPTER_REGISTRY`` are compatibility shims over that factory.
    Prefer importing from ``ipfs_accelerate_py.cli_runtime.endpoints`` for new code.
"""

import os
import subprocess
import json
import logging
import time
import shutil
import re
import platform
import threading
from typing import Dict, List, Any, Optional, Union, Mapping
from datetime import datetime
from abc import ABC, abstractmethod

try:
    from ipfs_accelerate_py.cli_runtime.contracts import (
        MAX_PROMPT_CHARS as _MAX_PROMPT_CHARS,
        MAX_TEXT_CHARS as _MAX_TEXT_CHARS,
    )
except Exception:  # pragma: no cover - defensive fallback
    _MAX_PROMPT_CHARS = 100000
    _MAX_TEXT_CHARS = 1048576

# Try to import storage wrapper with comprehensive fallback
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
            def get_storage_wrapper(*args, **kwargs):
                return None

logger = logging.getLogger("ipfs_accelerate_mcp.tools.cli_endpoint_adapters")

# Initialize storage wrapper at module level
_storage = get_storage_wrapper() if HAVE_STORAGE_WRAPPER else None


def sanitize_input(value: str, max_length: int = 10000, allowed_pattern: Optional[str] = None) -> str:
    """
    Sanitize input string to prevent command injection and other security issues
    
    Args:
        value: Input string to sanitize
        max_length: Maximum allowed length
        allowed_pattern: Optional regex pattern for allowed characters
        
    Returns:
        Sanitized string
        
    Raises:
        ValueError: If input fails validation
    """
    if not isinstance(value, str):
        raise ValueError(f"Input must be string, got {type(value)}")
    
    if len(value) > max_length:
        raise ValueError(f"Input too long: {len(value)} > {max_length}")
    
    # Check for null bytes
    if '\x00' in value:
        raise ValueError("Null bytes not allowed in input")
    
    # Apply pattern if provided
    if allowed_pattern and not re.match(allowed_pattern, value):
        raise ValueError(f"Input does not match allowed pattern")
    
    return value


def _clip_text(value: Any, maximum: int) -> str:
    text = str("" if value is None else value)
    if len(text) <= maximum:
        return text
    return text[: max(0, maximum - 3)] + "..."


def validate_cli_args(args: List[str]) -> List[str]:
    """
    Validate CLI arguments to prevent injection attacks
    
    Args:
        args: List of command arguments
        
    Returns:
        Validated arguments list
        
    Raises:
        ValueError: If arguments contain suspicious patterns
    """
    dangerous_patterns = [
        r';\s*',  # Command chaining
        r'\|\s*',  # Pipes
        r'&&',  # Command chaining
        r'\$\(',  # Command substitution
        r'`',  # Command substitution
        r'>\s*',  # Redirects
        r'<\s*',  # Redirects
    ]
    
    for arg in args:
        for pattern in dangerous_patterns:
            if re.search(pattern, arg):
                logger.warning(f"Potentially dangerous pattern detected in arg: {arg}")
                # Don't reject, just log - some legitimate uses might match
    
    return args


class CLIEndpointAdapter(ABC):
    """Base class for CLI endpoint adapters.

    Abstract: never instantiate directly. Use the concrete factory in
    ``ipfs_accelerate_py.cli_runtime.endpoints`` (or a concrete subclass).
    """

    def __init__(
        self,
        endpoint_id: str,
        cli_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize CLI endpoint adapter
        
        Args:
            endpoint_id: Unique identifier for this endpoint
            cli_path: Path to the CLI executable (auto-detected if None)
            config: Additional configuration parameters
        """
        self.endpoint_id = sanitize_input(endpoint_id, max_length=100, 
                                          allowed_pattern=r'^[a-zA-Z0-9_\-]+$')
        self.cli_path = cli_path or self._detect_cli_path()
        self.config = config or {}
        self._stats_lock = threading.Lock()
        self.stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0.0,
            "avg_time": 0.0
        }
        
        # Validate CLI is available
        if not self.is_available():
            logger.warning(f"CLI tool for {self.endpoint_id} not found at {self.cli_path}")

    def _record_success(self, elapsed_time: float) -> None:
        with self._stats_lock:
            self.stats["requests"] += 1
            self.stats["successes"] += 1
            self.stats["total_time"] += elapsed_time
            requests = self.stats["requests"]
            self.stats["avg_time"] = (
                self.stats["total_time"] / requests if requests else 0.0
            )

    def _record_failure(self, elapsed_time: float = 0.0) -> None:
        with self._stats_lock:
            self.stats["requests"] += 1
            self.stats["failures"] += 1
            self.stats["total_time"] += elapsed_time
            requests = self.stats["requests"]
            self.stats["avg_time"] = (
                self.stats["total_time"] / requests if requests else 0.0
            )

    def _stats_snapshot(self) -> Dict[str, Any]:
        with self._stats_lock:
            return dict(self.stats)
    
    @abstractmethod
    def _detect_cli_path(self) -> Optional[str]:
        """Detect the CLI tool path automatically"""
        pass
    
    @abstractmethod
    def _format_prompt(self, prompt: str, task_type: str, **kwargs) -> List[str]:
        """Format the prompt and kwargs into CLI arguments"""
        pass
    
    @abstractmethod
    def _parse_response(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse CLI output into standardized response format"""
        pass
    
    @abstractmethod
    def _config(self) -> Dict[str, Any]:
        """
        Get configuration instructions for the CLI tool
        
        Returns:
            Dictionary with configuration steps and requirements
        """
        pass
    
    @abstractmethod
    def _install(self) -> Dict[str, Any]:
        """
        Get installation instructions for the CLI tool
        
        Returns:
            Dictionary with installation commands and steps for current platform
        """
        pass
    
    def is_available(self) -> bool:
        """Check if the CLI tool is available"""
        if not self.cli_path:
            return False
        
        # Check if file exists and is executable
        if os.path.isfile(self.cli_path) and os.access(self.cli_path, os.X_OK):
            return True
        
        # Check if it's in PATH
        return shutil.which(self.cli_path) is not None
    
    def check_version(self) -> Dict[str, Any]:
        """
        Check the version of the CLI tool
        
        Returns:
            Dictionary with version information
        """
        if not self.is_available():
            return {
                "available": False,
                "error": "CLI tool not available"
            }
        
        try:
            result = subprocess.run(
                [self.cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return {
                "available": True,
                "version": result.stdout.strip() or result.stderr.strip(),
                "returncode": result.returncode
            }
        except Exception as e:
            return {
                "available": True,
                "error": f"Version check failed: {type(e).__name__}"
            }
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate the current configuration
        
        Returns:
            Dictionary with validation results
        """
        issues = []
        
        # Check if CLI is available
        if not self.is_available():
            issues.append(f"CLI tool not found at {self.cli_path}")
        
        # Check required config fields
        required_fields = getattr(self, 'required_config_fields', [])
        for field in required_fields:
            if field not in self.config:
                issues.append(f"Missing required config field: {field}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config": self.config
        }
    
    def execute(
        self,
        prompt: str,
        task_type: str = "text_generation",
        timeout: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute inference using the CLI tool
        
        Args:
            prompt: Input prompt
            task_type: Type of task to perform
            timeout: Maximum execution time in seconds
            **kwargs: Additional task-specific parameters
            
        Returns:
            Dictionary with inference results and metadata.
            Nonzero subprocess exit status is always a failure.
            Error payloads never echo the prompt.
        """
        start_time = time.time()
        
        try:
            # Sanitize / bound prompt input
            prompt = sanitize_input(prompt, max_length=_MAX_PROMPT_CHARS)
            
            # Format command
            cmd_args = self._format_prompt(prompt, task_type, **kwargs)
            
            # Validate command arguments
            cmd_args = validate_cli_args(cmd_args)
            
            logger.info(
                "Executing CLI command for %s: %s...",
                self.endpoint_id,
                " ".join(str(a) for a in cmd_args[:3]),
            )
            
            # Execute CLI command with security constraints
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, **self.config.get("env_vars", {})},
                cwd=self.config.get("working_dir"),  # Optional working directory
                shell=False  # Never use shell=True for security
            )
            
            elapsed_time = time.time() - start_time
            returncode = int(result.returncode)

            # Nonzero exit is always failure (do not treat as success).
            if returncode != 0:
                self._record_failure(elapsed_time)
                stderr_diag = _clip_text(
                    (result.stderr or "").strip(), 1024
                )
                payload: Dict[str, Any] = {
                    "error": f"CLI exited with status {returncode}",
                    "endpoint_id": self.endpoint_id,
                    "endpoint_type": "cli",
                    "elapsed_time": elapsed_time,
                    "status": "error",
                    "success": False,
                    "returncode": returncode,
                    "error_code": "nonzero_exit",
                }
                if stderr_diag:
                    payload["stderr"] = stderr_diag
                return payload

            # Parse response and bound result text
            response = self._parse_response(result.stdout, result.stderr)
            if isinstance(response.get("result"), str):
                response["result"] = _clip_text(
                    response["result"], _MAX_TEXT_CHARS
                )
            if isinstance(response.get("raw_response"), str):
                response["raw_response"] = _clip_text(
                    response["raw_response"], _MAX_TEXT_CHARS
                )

            self._record_success(elapsed_time)
            
            # Add metadata (never include prompt)
            response.update({
                "endpoint_id": self.endpoint_id,
                "endpoint_type": "cli",
                "elapsed_time": elapsed_time,
                "status": "success",
                "success": True,
                "returncode": returncode,
            })
            response.pop("prompt", None)
            
            return response
            
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            self._record_failure(elapsed_time)
            logger.error(f"CLI execution timeout for {self.endpoint_id}")
            return {
                "error": "CLI execution timeout",
                "endpoint_id": self.endpoint_id,
                "elapsed_time": elapsed_time,
                "status": "timeout",
                "success": False,
            }
        
        except ValueError as e:
            # Input validation error — message only, never the prompt body
            elapsed_time = time.time() - start_time
            self._record_failure(elapsed_time)
            logger.error(
                "Input validation error for %s: %s",
                self.endpoint_id,
                type(e).__name__,
            )
            return {
                "error": f"Input validation error: {type(e).__name__}",
                "endpoint_id": self.endpoint_id,
                "elapsed_time": elapsed_time,
                "status": "validation_error",
                "success": False,
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self._record_failure(elapsed_time)
            logger.error(
                "CLI execution error for %s: %s",
                self.endpoint_id,
                type(e).__name__,
            )
            return {
                "error": f"CLI execution error: {type(e).__name__}",
                "endpoint_id": self.endpoint_id,
                "elapsed_time": elapsed_time,
                "status": "error",
                "success": False,
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get endpoint statistics (concurrency-safe snapshot)."""
        return {
            "endpoint_id": self.endpoint_id,
            "endpoint_type": "cli",
            "cli_path": self.cli_path,
            "available": self.is_available(),
            "stats": self._stats_snapshot(),
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get capabilities and features of this CLI adapter
        
        Returns:
            Dictionary describing adapter capabilities
        """
        return {
            "endpoint_id": self.endpoint_id,
            "cli_path": self.cli_path,
            "available": self.is_available(),
            "supported_tasks": getattr(self, 'supported_tasks', ["text_generation"]),
            "config_fields": getattr(self, 'config_fields', {}),
            "version_info": self.check_version()
        }

    async def async_execute(
        self,
        prompt: str,
        task_type: str = "text_generation",
        timeout: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Async version of :meth:`execute` safe for Trio / Hypercorn.

        The blocking ``subprocess.run()`` call is offloaded to a worker
        thread via ``anyio.to_thread.run_sync`` so the event loop is never
        stalled by a slow CLI subprocess.  Falls back to the synchronous
        path when ``anyio`` is not installed.

        Parameters
        ----------
        prompt:
            Input prompt to pass to the CLI tool.
        task_type:
            Task type string forwarded to :meth:`execute`.
        timeout:
            Maximum subprocess execution time in seconds.
        **kwargs:
            Additional keyword arguments forwarded to :meth:`execute`.

        Returns
        -------
        dict
            Same structure as :meth:`execute`.
        """
        try:
            import anyio
        except ImportError:
            # Fall back to sync (blocks event loop – should be avoided in Trio)
            logger.warning(
                "anyio not installed; async_execute for %s will block the event loop. "
                "Install anyio to enable true async execution.",
                self.endpoint_id,
            )
            return self.execute(prompt, task_type=task_type, timeout=timeout, **kwargs)

        import functools
        fn = functools.partial(
            self.execute,
            prompt,
            task_type=task_type,
            timeout=timeout,
            **kwargs,
        )
        return await anyio.to_thread.run_sync(fn)


class ClaudeCodeAdapter(CLIEndpointAdapter):
    """Adapter for Claude Code CLI tool"""
    
    # Configuration fields
    config_fields = {
        "model": {
            "type": "string",
            "description": "Claude model to use",
            "default": "claude-3-sonnet",
            "options": ["claude-3-sonnet", "claude-3-opus", "claude-3-haiku"]
        },
        "max_tokens": {
            "type": "integer",
            "description": "Maximum tokens to generate",
            "default": 4096
        },
        "temperature": {
            "type": "float",
            "description": "Sampling temperature (0.0-1.0)",
            "default": 0.7
        }
    }
    
    supported_tasks = ["text_generation", "code_generation", "analysis"]
    
    def _detect_cli_path(self) -> Optional[str]:
        """Detect claude CLI path"""
        # Common locations for claude CLI
        possible_paths = [
            "claude",  # In PATH
            "/usr/local/bin/claude",
            "/usr/bin/claude",
            os.path.expanduser("~/.local/bin/claude"),
            os.path.expanduser("~/bin/claude")
        ]
        
        for path in possible_paths:
            if shutil.which(path) or (os.path.isfile(path) and os.access(path, os.X_OK)):
                return path
        
        return "claude"  # Fallback to PATH lookup
    
    def _format_prompt(self, prompt: str, task_type: str, **kwargs) -> List[str]:
        """Format prompt for claude CLI"""
        cmd = [self.cli_path]
        
        # Add model parameter if specified
        model = kwargs.get("model", self.config.get("model", "claude-3-sonnet"))
        cmd.extend(["--model", sanitize_input(model, max_length=50)])
        
        # Add max tokens if specified
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 4096))
        if isinstance(max_tokens, (int, str)):
            cmd.extend(["--max-tokens", str(int(max_tokens))])
        
        # Add temperature if specified
        temperature = kwargs.get("temperature", self.config.get("temperature"))
        if temperature is not None:
            temp_val = float(temperature)
            if 0.0 <= temp_val <= 1.0:
                cmd.extend(["--temperature", str(temp_val)])
        
        # Add the prompt
        cmd.append(prompt)
        
        return cmd
    
    def _parse_response(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse claude CLI output"""
        try:
            # Try to parse as JSON first
            data = json.loads(stdout)
            return {
                "result": data.get("content", [{"text": stdout}])[0].get("text", stdout),
                "model": data.get("model", "claude"),
                "provider": "anthropic",
                "raw_response": data
            }
        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            return {
                "result": stdout.strip(),
                "model": "claude",
                "provider": "anthropic",
                "raw_response": stdout
            }
    
    def _config(self) -> Dict[str, Any]:
        """Get configuration instructions for Claude CLI"""
        return {
            "tool_name": "Claude Code CLI",
            "description": "Anthropic's Claude AI assistant CLI tool",
            "config_steps": [
                "1. Obtain an Anthropic API key from https://console.anthropic.com/",
                "2. Set environment variable: export ANTHROPIC_API_KEY='your-key-here'",
                "3. Or configure via: claude configure",
                "4. Test with: claude --version"
            ],
            "env_vars": {
                "ANTHROPIC_API_KEY": "Your Anthropic API key"
            },
            "config_files": [
                "~/.config/claude/config.json",
                "~/.claude/config.json"
            ],
            "documentation": "https://docs.anthropic.com/claude/reference/claude-cli"
        }
    
    def _install(self) -> Dict[str, Any]:
        """Get installation instructions for Claude CLI"""
        system = platform.system().lower()
        
        instructions = {
            "tool_name": "Claude Code CLI",
            "platform": system,
            "install_methods": []
        }
        
        if system == "darwin":  # macOS
            instructions["install_methods"] = [
                {
                    "method": "Homebrew",
                    "commands": [
                        "brew tap anthropic/claude",
                        "brew install claude"
                    ]
                },
                {
                    "method": "Direct Download",
                    "commands": [
                        "curl -fsSL https://claude.ai/cli/install.sh | sh"
                    ]
                }
            ]
        elif system == "linux":
            instructions["install_methods"] = [
                {
                    "method": "Package Manager (apt/yum)",
                    "commands": [
                        "# For Debian/Ubuntu:",
                        "wget https://claude.ai/cli/claude_latest_amd64.deb",
                        "sudo dpkg -i claude_latest_amd64.deb"
                    ]
                },
                {
                    "method": "Direct Download",
                    "commands": [
                        "curl -fsSL https://claude.ai/cli/install.sh | sh"
                    ]
                }
            ]
        elif system == "windows":
            instructions["install_methods"] = [
                {
                    "method": "Installer",
                    "commands": [
                        "# Download from https://claude.ai/cli/windows",
                        "# Run claude-setup.exe"
                    ]
                },
                {
                    "method": "Chocolatey",
                    "commands": [
                        "choco install claude-cli"
                    ]
                }
            ]
        
        instructions["verify_command"] = "claude --version"
        instructions["documentation"] = "https://docs.anthropic.com/claude/reference/claude-cli"
        
        return instructions


class OpenAICodexAdapter(CLIEndpointAdapter):
    """Adapter for OpenAI Codex/ChatGPT CLI tool"""
    
    def _detect_cli_path(self) -> Optional[str]:
        """Detect openai CLI path"""
        # Common locations for openai CLI
        possible_paths = [
            "openai",  # In PATH
            "chatgpt",  # Alternative name
            "/usr/local/bin/openai",
            "/usr/bin/openai",
            os.path.expanduser("~/.local/bin/openai"),
            os.path.expanduser("~/bin/openai")
        ]
        
        for path in possible_paths:
            if shutil.which(path) or (os.path.isfile(path) and os.access(path, os.X_OK)):
                return path
        
        return "openai"  # Fallback to PATH lookup
    
    def _format_prompt(self, prompt: str, task_type: str, **kwargs) -> List[str]:
        """Format prompt for openai CLI"""
        cmd = [self.cli_path]
        
        # OpenAI CLI typically has subcommands
        if task_type == "text_generation" or task_type == "code_generation":
            cmd.append("api")
            cmd.append("chat.completions.create")
        elif task_type == "embedding":
            cmd.append("api")
            cmd.append("embeddings.create")
        else:
            cmd.append("api")
            cmd.append("completions.create")
        
        # Add model parameter
        model = kwargs.get("model", self.config.get("model", "gpt-3.5-turbo"))
        cmd.extend(["-m", model])
        
        # Add max tokens if specified
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens"))
        if max_tokens:
            cmd.extend(["--max-tokens", str(max_tokens)])
        
        # Add temperature if specified
        temperature = kwargs.get("temperature", self.config.get("temperature"))
        if temperature is not None:
            cmd.extend(["--temperature", str(temperature)])
        
        # Add the prompt
        cmd.extend(["-g", prompt])
        
        return cmd
    
    def _parse_response(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse openai CLI output"""
        try:
            # Try to parse as JSON
            data = json.loads(stdout)
            
            # Handle different response formats
            if "choices" in data:
                result = data["choices"][0].get("message", {}).get("content", 
                         data["choices"][0].get("text", stdout))
            elif "data" in data:
                result = data["data"]
            else:
                result = stdout.strip()
            
            return {
                "result": result,
                "model": data.get("model", "gpt-3.5-turbo"),
                "provider": "openai",
                "raw_response": data
            }
        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            return {
                "result": stdout.strip(),
                "model": "gpt-3.5-turbo",
                "provider": "openai",
                "raw_response": stdout
            }
    
    def _config(self) -> Dict[str, Any]:
        """Get configuration instructions for OpenAI CLI"""
        return {
            "tool_name": "OpenAI CLI",
            "description": "OpenAI's official CLI tool for ChatGPT and Codex",
            "config_steps": [
                "1. Obtain an OpenAI API key from https://platform.openai.com/api-keys",
                "2. Set environment variable: export OPENAI_API_KEY='your-key-here'",
                "3. Or configure via: openai api_key.set YOUR_KEY",
                "4. Test with: openai api models.list"
            ],
            "env_vars": {
                "OPENAI_API_KEY": "Your OpenAI API key"
            },
            "config_files": [
                "~/.openai/auth.json",
                "~/.config/openai/config.json"
            ],
            "documentation": "https://platform.openai.com/docs/api-reference/introduction"
        }
    
    def _install(self) -> Dict[str, Any]:
        """Get installation instructions for OpenAI CLI"""
        system = platform.system().lower()
        
        instructions = {
            "tool_name": "OpenAI CLI",
            "platform": system,
            "install_methods": [
                {
                    "method": "pip (Recommended)",
                    "commands": [
                        "pip install openai",
                        "# Or for latest version:",
                        "pip install --upgrade openai"
                    ]
                }
            ]
        }
        
        if system == "darwin":  # macOS
            instructions["install_methods"].append({
                "method": "Homebrew",
                "commands": [
                    "brew install openai"
                ]
            })
        
        instructions["verify_command"] = "openai --version"
        instructions["documentation"] = "https://github.com/openai/openai-python"
        
        return instructions


class GeminiCLIAdapter(CLIEndpointAdapter):
    """Adapter for Google Gemini CLI tool"""
    
    def _detect_cli_path(self) -> Optional[str]:
        """Detect gemini CLI path"""
        # Common locations for gemini CLI
        possible_paths = [
            "gemini",  # In PATH
            "google-gemini",
            "gcloud",  # Google Cloud SDK
            "/usr/local/bin/gemini",
            "/usr/bin/gemini",
            os.path.expanduser("~/.local/bin/gemini"),
            os.path.expanduser("~/bin/gemini")
        ]
        
        for path in possible_paths:
            if shutil.which(path) or (os.path.isfile(path) and os.access(path, os.X_OK)):
                return path
        
        return "gemini"  # Fallback to PATH lookup
    
    def _format_prompt(self, prompt: str, task_type: str, **kwargs) -> List[str]:
        """Format prompt for gemini CLI"""
        cmd = [self.cli_path]
        
        # Gemini CLI structure (may vary based on actual implementation)
        if self.cli_path.endswith("gcloud"):
            cmd.extend(["ai", "models", "generate-content"])
        
        # Add model parameter
        model = kwargs.get("model", self.config.get("model", "gemini-pro"))
        cmd.extend(["--model", model])
        
        # Add temperature if specified
        temperature = kwargs.get("temperature", self.config.get("temperature"))
        if temperature is not None:
            cmd.extend(["--temperature", str(temperature)])
        
        # Add max tokens if specified
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens"))
        if max_tokens:
            cmd.extend(["--max-output-tokens", str(max_tokens)])
        
        # Add the prompt
        cmd.extend(["--prompt", prompt])
        
        return cmd
    
    def _parse_response(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse gemini CLI output"""
        try:
            # Try to parse as JSON
            data = json.loads(stdout)
            
            # Handle Gemini response format
            if "candidates" in data:
                result = data["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", stdout)
            elif "text" in data:
                result = data["text"]
            else:
                result = stdout.strip()
            
            return {
                "result": result,
                "model": data.get("model", "gemini-pro"),
                "provider": "google",
                "raw_response": data
            }
        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            return {
                "result": stdout.strip(),
                "model": "gemini-pro",
                "provider": "google",
                "raw_response": stdout
            }
    
    def _config(self) -> Dict[str, Any]:
        """Get configuration instructions for Gemini CLI"""
        return {
            "tool_name": "Google Gemini CLI",
            "description": "Google's Gemini AI via gcloud CLI",
            "config_steps": [
                "1. Install Google Cloud SDK from https://cloud.google.com/sdk/docs/install",
                "2. Authenticate: gcloud auth login",
                "3. Set project: gcloud config set project YOUR_PROJECT_ID",
                "4. Enable AI Platform API: gcloud services enable aiplatform.googleapis.com",
                "5. Test with: gcloud ai models list"
            ],
            "env_vars": {
                "GOOGLE_APPLICATION_CREDENTIALS": "Path to service account key JSON (optional)",
                "GCLOUD_PROJECT": "Your Google Cloud project ID"
            },
            "config_files": [
                "~/.config/gcloud/configurations/config_default"
            ],
            "documentation": "https://cloud.google.com/sdk/gcloud/reference/ai"
        }
    
    def _install(self) -> Dict[str, Any]:
        """Get installation instructions for Gemini CLI (gcloud)"""
        system = platform.system().lower()
        
        instructions = {
            "tool_name": "Google Cloud SDK (gcloud)",
            "platform": system,
            "install_methods": []
        }
        
        if system == "darwin":  # macOS
            instructions["install_methods"] = [
                {
                    "method": "Homebrew",
                    "commands": [
                        "brew install --cask google-cloud-sdk"
                    ]
                },
                {
                    "method": "Direct Download",
                    "commands": [
                        "curl https://sdk.cloud.google.com | bash",
                        "exec -l $SHELL",
                        "gcloud init"
                    ]
                }
            ]
        elif system == "linux":
            instructions["install_methods"] = [
                {
                    "method": "Package Manager",
                    "commands": [
                        "# Add the Cloud SDK distribution URI as a package source:",
                        "echo \"deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main\" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list",
                        "sudo apt-get update && sudo apt-get install google-cloud-cli"
                    ]
                },
                {
                    "method": "Direct Download",
                    "commands": [
                        "curl https://sdk.cloud.google.com | bash",
                        "exec -l $SHELL",
                        "gcloud init"
                    ]
                }
            ]
        elif system == "windows":
            instructions["install_methods"] = [
                {
                    "method": "Installer",
                    "commands": [
                        "# Download from https://cloud.google.com/sdk/docs/install-sdk#windows",
                        "# Run GoogleCloudSDKInstaller.exe"
                    ]
                }
            ]
        
        instructions["verify_command"] = "gcloud --version"
        instructions["documentation"] = "https://cloud.google.com/sdk/docs/install"
        
        return instructions


class VSCodeCLIAdapter(CLIEndpointAdapter):
    """Adapter for Visual Studio Code CLI (GitHub Copilot)"""
    
    config_fields = {
        "model": {
            "type": "string",
            "description": "Model to use (copilot-chat, copilot-code)",
            "default": "copilot-chat"
        },
        "temperature": {
            "type": "float",
            "description": "Sampling temperature",
            "default": 0.7
        }
    }
    
    supported_tasks = ["code_generation", "code_completion", "code_explanation", "text_generation"]
    
    def _detect_cli_path(self) -> Optional[str]:
        """Detect VSCode CLI path"""
        possible_paths = [
            "code",  # In PATH
            "/usr/local/bin/code",
            "/usr/bin/code",
            "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code",  # macOS
            os.path.expanduser("~/.local/bin/code"),
            os.path.expanduser("~/bin/code"),
            "code-insiders",  # Insiders version
        ]
        
        for path in possible_paths:
            if shutil.which(path) or (os.path.isfile(path) and os.access(path, os.X_OK)):
                return path
        
        return "code"  # Fallback to PATH lookup
    
    def _format_prompt(self, prompt: str, task_type: str, **kwargs) -> List[str]:
        """Format prompt for VSCode CLI"""
        cmd = [self.cli_path]
        
        # VSCode CLI uses extension commands for Copilot
        # This is a simplified interface - actual usage may vary
        if task_type in ["code_generation", "code_completion"]:
            # Use stdin mode for code generation
            cmd.extend(["--stdin"])
        
        # Add custom arguments if provided
        custom_args = kwargs.get("cli_args", self.config.get("cli_args", []))
        if custom_args:
            cmd.extend(custom_args)
        
        return cmd
    
    def _parse_response(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Parse VSCode CLI output"""
        try:
            # Try to parse as JSON first
            data = json.loads(stdout)
            return {
                "result": data.get("text", data.get("code", stdout)),
                "model": "vscode-copilot",
                "provider": "github",
                "raw_response": data
            }
        except json.JSONDecodeError:
            # If not JSON, treat as plain text/code
            return {
                "result": stdout.strip(),
                "model": "vscode-copilot",
                "provider": "github",
                "raw_response": stdout
            }
    
    def _config(self) -> Dict[str, Any]:
        """Get configuration instructions for VSCode CLI"""
        return {
            "tool_name": "Visual Studio Code CLI (GitHub Copilot)",
            "description": "VSCode CLI with GitHub Copilot integration",
            "config_steps": [
                "1. Install Visual Studio Code from https://code.visualstudio.com/",
                "2. Install GitHub Copilot extension in VSCode",
                "3. Sign in to GitHub in VSCode",
                "4. Verify CLI: code --version",
                "5. Enable Copilot Chat for CLI usage"
            ],
            "env_vars": {
                "GITHUB_TOKEN": "Your GitHub Personal Access Token (optional for CLI)"
            },
            "config_files": [
                "~/.vscode/extensions/github.copilot-*/",
                "~/.config/Code/User/settings.json"
            ],
            "documentation": "https://docs.github.com/en/copilot/using-github-copilot/using-github-copilot-in-the-command-line"
        }
    
    def _install(self) -> Dict[str, Any]:
        """Get installation instructions for VSCode CLI"""
        system = platform.system().lower()
        
        instructions = {
            "tool_name": "Visual Studio Code CLI",
            "platform": system,
            "install_methods": []
        }
        
        if system == "darwin":  # macOS
            instructions["install_methods"] = [
                {
                    "method": "Homebrew",
                    "commands": [
                        "brew install --cask visual-studio-code"
                    ]
                },
                {
                    "method": "Direct Download",
                    "commands": [
                        "# Download from https://code.visualstudio.com/download",
                        "# Install VSCode.app to Applications",
                        "# Add to PATH:",
                        "sudo ln -s '/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code' /usr/local/bin/code"
                    ]
                }
            ]
        elif system == "linux":
            instructions["install_methods"] = [
                {
                    "method": "Snap",
                    "commands": [
                        "sudo snap install --classic code"
                    ]
                },
                {
                    "method": "apt (Debian/Ubuntu)",
                    "commands": [
                        "wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg",
                        "sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg",
                        "sudo sh -c 'echo \"deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main\" > /etc/apt/sources.list.d/vscode.list'",
                        "sudo apt update",
                        "sudo apt install code"
                    ]
                }
            ]
        elif system == "windows":
            instructions["install_methods"] = [
                {
                    "method": "Installer",
                    "commands": [
                        "# Download from https://code.visualstudio.com/download",
                        "# Run VSCodeUserSetup-{version}.exe",
                        "# CLI should be automatically added to PATH"
                    ]
                },
                {
                    "method": "winget",
                    "commands": [
                        "winget install Microsoft.VisualStudioCode"
                    ]
                }
            ]
        
        instructions["post_install"] = [
            "Install GitHub Copilot extension:",
            "code --install-extension GitHub.copilot",
            "code --install-extension GitHub.copilot-chat"
        ]
        instructions["verify_command"] = "code --version"
        instructions["documentation"] = "https://code.visualstudio.com/docs/setup/setup-overview"
        
        return instructions


# ---------------------------------------------------------------------------
# Goose CLI adapter (delegates to canonical GooseCLIProvider)
# ---------------------------------------------------------------------------


# Known kwargs accepted by Goose endpoint execute (bounded authority surface).
_GOOSE_SAFE_EXECUTE_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "model_name",
        "goose_provider",
        "provider",
        "temperature",
        "max_tokens",
        "output_format",
        "stream",
        "streaming",
        "timeout",
        "task_type",
    }
)
_GOOSE_AUTHORITY_EXECUTE_KEYS: frozenset[str] = frozenset(
    {
        "execution_mode",
        "mode",
        "agent",
        "allow_side_effects",
        "enable_agent",
        "package_enable_agent",
        "package_policy",
        "cwd",
        "workspace",
        "path_root",
        "GOOSE_PATH_ROOT",
        "goose_path_root",
        "approval_mode",
        "builtins",
        "extensions",
        "with_builtin",
        "with_extension",
        "allowed_cwd_roots",
        "max_turns",
        "max_tool_repetitions",
        "timeout_seconds",
        "max_output_bytes",
        "session_id",
        "resume_session",
        "agent_policy",
        "side_effecting",
        "with_tools",
    }
)
_GOOSE_KNOWN_EXECUTE_KEYS: frozenset[str] = (
    _GOOSE_SAFE_EXECUTE_KEYS | _GOOSE_AUTHORITY_EXECUTE_KEYS
)


def _goose_package_agent_enabled(
    config: Mapping[str, Any],
    kwargs: Mapping[str, Any],
) -> bool:
    """Return True when the package enable policy explicitly allows agent mode."""
    for source in (kwargs, config):
        if not isinstance(source, Mapping):
            continue
        if source.get("enable_agent") is True:
            return True
        if source.get("package_enable_agent") is True:
            return True
        policy = source.get("package_policy")
        if isinstance(policy, Mapping) and policy.get("enable_agent") is True:
            return True
    return False


def _goose_reject_unknown_authority(kwargs: Mapping[str, Any]) -> Optional[str]:
    """Return an error message if unknown authority-bearing keys are present."""
    # Authority-bearing markers that must never be accepted under unknown names.
    authority_markers = (
        "allow_side",
        "side_effect",
        "path_root",
        "cwd",
        "workspace",
        "approval",
        "builtin",
        "extension",
        "enable_agent",
        "package_policy",
        "agent_policy",
        "execution_mode",
        "max_turn",
        "max_tool",
        "max_output",
        "allowed_cwd",
        "session",
        "resume",
        "GOOSE_PATH",
        "goose_path",
    )
    for key in kwargs:
        k = str(key)
        if k in _GOOSE_KNOWN_EXECUTE_KEYS:
            continue
        lowered = k.lower()
        if any(marker.lower() in lowered for marker in authority_markers):
            return f"unknown authority-bearing option rejected: {k}"
    return None


class GooseCLIAdapter(CLIEndpointAdapter):
    """Concrete Goose CLI endpoint adapter.

    Delegates command construction, parsing, and policy to the canonical
    :class:`~ipfs_accelerate_py.cli_runtime.providers.goose.GooseCLIProvider`.

    Safety:

    - Default execute is chat-only (same safe profile as llm_router / goose run).
    - Agent mode requires ``execution_mode=agent`` (or agent=True),
      ``allow_side_effects=True``, package enable policy, absolute cwd/root,
      explicit approval mode, extension/builtin allowlists, and finite
      turns/time/output.
    - List/liveness never invoke this adapter's execute path; health probes
      never send model prompts.
    - Response envelopes never include the prompt or credential material.
    """

    config_fields = {
        "model": {
            "type": "string",
            "description": "Goose model name (maps to --model / GOOSE_MODEL)",
            "default": None,
        },
        "goose_provider": {
            "type": "string",
            "description": "Underlying provider (maps to --provider / GOOSE_PROVIDER)",
            "default": None,
        },
        "enable_agent": {
            "type": "boolean",
            "description": "Package enable policy: allow agent-mode requests",
            "default": False,
        },
        "allow_install": {
            "type": "boolean",
            "description": "Permit explicit lazy install on ensure_ready only",
            "default": False,
        },
    }

    supported_tasks = ["text_generation", "code_generation", "analysis"]
    tool_name = "goose"

    def __init__(
        self,
        endpoint_id: str,
        cli_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._provider = None  # type: ignore[assignment]
        self._cancel_token = None
        super().__init__(endpoint_id, cli_path, config)
        # Prefer explicit path; keep tool stamp for registry describe.
        self.tool_name = "goose"
        if isinstance(self.config, dict):
            self.config.setdefault("tool", "goose")

    # -- abstract API (unused by execute; kept for ABC completeness) -------

    def _detect_cli_path(self) -> Optional[str]:
        """Detect goose binary without starting a model request."""
        possible_paths = [
            "goose",
            "goose.exe",
            "/usr/local/bin/goose",
            "/usr/bin/goose",
            os.path.expanduser("~/.local/bin/goose"),
            os.path.expanduser("~/bin/goose"),
        ]
        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
            found = shutil.which(path)
            if found:
                return found
        # Env overrides (detect-only; no install).
        for env_name in (
            "IPFS_ACCELERATE_GOOSE_PATH",
            "IPFS_ACCELERATE_PY_GOOSE_PATH",
            "GOOSE_CLI_PATH",
        ):
            raw = os.environ.get(env_name)
            if raw and str(raw).strip():
                candidate = os.path.expanduser(str(raw).strip())
                if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    return candidate
        return "goose"

    def _format_prompt(self, prompt: str, task_type: str, **kwargs) -> List[str]:
        # Execute path never uses this; Goose builds argv via the provider.
        return [self.cli_path or "goose", "run", "--instructions", "-"]

    def _parse_response(self, stdout: str, stderr: str) -> Dict[str, Any]:
        return {"result": (stdout or "").strip(), "raw_response": stdout or ""}

    def _config(self) -> Dict[str, Any]:
        return {
            "tool_name": "Goose CLI",
            "description": (
                "Block/AAIF Goose CLI — chat-safe defaults; agent requires "
                "explicit package enable policy and GooseAgentPolicy fields"
            ),
            "config_fields": self.config_fields,
            "setup_steps": [
                "1. Install Goose CLI (pinned release via ensure_goose or operator install)",
                "2. Configure a provider (GOOSE_PROVIDER / OPENAI_API_KEY / etc.)",
                "3. Register endpoint with tool='goose' or tool='goose_cli'",
                "4. Default execute is chat-only; agent needs enable_agent + policy",
            ],
            "agent_requirements": [
                "execution_mode=agent",
                "allow_side_effects=True",
                "package enable policy (enable_agent / package_enable_agent)",
                "absolute cwd and path_root (GOOSE_PATH_ROOT)",
                "explicit approval_mode (not chat)",
                "extension/builtin allowlists (may be empty)",
                "finite max_turns, timeout_seconds, max_output_bytes",
            ],
        }

    def _install(self) -> Dict[str, Any]:
        system = platform.system().lower()
        return {
            "tool_name": "Goose CLI",
            "platform": system,
            "install_methods": [
                {
                    "method": "ipfs_accelerate lazy installer (explicit only)",
                    "commands": [
                        "from ipfs_accelerate_py.cli_runtime.installers.goose "
                        "import ensure_goose",
                        "ensure_goose(auto_install=True)",
                    ],
                },
                {
                    "method": "Upstream binary",
                    "commands": [
                        "# Download a pinned release from aaif-goose/goose",
                        "# Place goose on PATH or set IPFS_ACCELERATE_GOOSE_PATH",
                    ],
                },
            ],
            "verify_command": "goose --version",
            "documentation": "https://block.github.io/goose/",
        }

    # -- provider / availability / health ----------------------------------

    def _get_provider(self):
        """Lazy construct the canonical GooseCLIProvider (no install, no run)."""
        if self._provider is not None:
            return self._provider
        try:
            from ipfs_accelerate_py.cli_runtime.providers.goose import (
                GooseCLIProvider,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("GooseCLIProvider unavailable") from exc

        cfg = self.config if isinstance(self.config, dict) else {}
        executable = self.cli_path
        # Treat bare "goose" as unresolved so provider discovery can run.
        if executable and not (
            os.path.isfile(executable) or os.sep in str(executable)
        ):
            # Keep name for which() inside provider discover.
            pass
        self._provider = GooseCLIProvider(
            executable=executable if executable else None,
            default_model=cfg.get("model") or cfg.get("model_name"),
            default_goose_provider=cfg.get("goose_provider") or cfg.get("provider"),
            allow_install=bool(cfg.get("allow_install", False)),
        )
        return self._provider

    def is_available(self) -> bool:
        """Binary presence only — never sends a model request."""
        if self.cli_path and os.path.isfile(self.cli_path) and os.access(
            self.cli_path, os.X_OK
        ):
            return True
        if self.cli_path and shutil.which(self.cli_path):
            return True
        return shutil.which("goose") is not None

    def assess_health(self) -> Dict[str, Any]:
        """Typed health without a model request.

        Distinguishes installed, configured, ready, degraded, and
        unsupported_version. May run ``goose --version`` (not a prompt).
        """
        from ipfs_accelerate_py.cli_runtime.endpoints import EndpointHealth
        from ipfs_accelerate_py.cli_runtime.providers.goose import (
            capabilities_for_version,
            goose_auth_available,
        )

        base: Dict[str, Any] = {
            "endpoint_id": self.endpoint_id,
            "provider": "goose_cli",
            "installed": False,
            "configured": False,
            "ready": False,
            "available": False,
            "unsupported_version": False,
            "goose_version": "",
            "version": "",
            "health": EndpointHealth.MISSING.value,
            "reason": "not_installed",
        }

        # Explicit absolute/path-like cli_path that is missing → MISSING
        # without falling through to ambient PATH discovery.
        explicit = self.cli_path
        if explicit and (
            os.sep in str(explicit) or str(explicit).startswith("~")
        ):
            expanded = os.path.expanduser(str(explicit))
            if not (
                os.path.isfile(expanded) and os.access(expanded, os.X_OK)
            ):
                base["reason"] = "not_installed"
                base["cli_path"] = explicit
                return base

        try:
            provider = self._get_provider()
            # Detect-only discovery; never installs.
            install = provider.discover(
                explicit_path=self.cli_path if self.cli_path else None,
                probe_version=True,
            )
        except Exception as exc:  # noqa: BLE001
            # Ambient discovery errors are degraded only when we did not have
            # a clear missing-path signal above.
            if not self.is_available():
                base["health"] = EndpointHealth.MISSING.value
                base["reason"] = f"not_installed:{type(exc).__name__}"
                return base
            base["health"] = EndpointHealth.DEGRADED.value
            base["reason"] = f"discovery_failed:{type(exc).__name__}"
            base["error"] = f"health probe failed: {type(exc).__name__}"
            return base

        if not install.available or not install.executable:
            base["health"] = EndpointHealth.MISSING.value
            base["reason"] = install.reason or "not_installed"
            return base

        version = install.version or getattr(provider, "version", "") or ""
        base["installed"] = True
        base["available"] = True
        base["goose_version"] = version
        base["version"] = version
        base["cli_path"] = install.executable

        # Version capability gate (no model request).
        caps = capabilities_for_version(version or "0.0.0")
        missing_flags = caps.missing_required_chat_flags()
        if missing_flags:
            base["health"] = EndpointHealth.UNSUPPORTED_VERSION.value
            base["unsupported_version"] = True
            base["ready"] = False
            base["reason"] = "unsupported_version"
            base["error"] = (
                "Goose version does not support required chat safety flags"
            )
            base["error_code"] = "unsupported_capability"
            base["missing_flags"] = list(missing_flags)
            return base

        authenticated = False
        try:
            authenticated = bool(goose_auth_available())
        except Exception:  # noqa: BLE001
            authenticated = False

        # Configured: auth markers present and/or explicit model/provider config.
        cfg = self.config if isinstance(self.config, dict) else {}
        has_config = bool(
            authenticated
            or cfg.get("goose_provider")
            or cfg.get("provider")
            or cfg.get("model")
            or cfg.get("model_name")
        )
        base["configured"] = has_config

        if not has_config:
            base["health"] = EndpointHealth.INSTALLED.value
            base["ready"] = False
            base["reason"] = "missing_auth"
            return base

        # Configured but treat ready only when installed + auth/config + safe version.
        if authenticated or has_config:
            base["health"] = EndpointHealth.READY.value
            base["ready"] = True
            base["reason"] = "ready"
            # If only config fields without env auth, still configured/ready for
            # operator-supplied provider routing via kwargs at execute time.
            if not authenticated and has_config:
                base["health"] = EndpointHealth.CONFIGURED.value
                base["ready"] = True
                base["reason"] = "configured"
            return base

        base["health"] = EndpointHealth.INSTALLED.value
        base["reason"] = "installed"
        return base

    def check_version(self) -> Dict[str, Any]:
        """Version probe only (not a model request)."""
        health = self.assess_health()
        return {
            "available": bool(health.get("installed")),
            "version": health.get("goose_version") or health.get("version") or "",
            "health": health.get("health"),
            "unsupported_version": bool(health.get("unsupported_version")),
        }

    # -- execute -----------------------------------------------------------

    def execute(
        self,
        prompt: str,
        task_type: str = "text_generation",
        timeout: int = 30,
        **kwargs,
    ) -> Dict[str, Any]:
        """One-shot Goose execute with chat defaults or explicit agent policy.

        Returns a bounded envelope with provider, execution_mode, text,
        goose_version, underlying provider/model, session, tool_call_count,
        side_effects_started, elapsed_time, and typed error fields. Never
        echoes the prompt or credentials.
        """
        start_time = time.time()
        unknown = _goose_reject_unknown_authority(kwargs)
        if unknown:
            self._record_failure(0.0)
            return {
                "status": "error",
                "success": False,
                "error": unknown,
                "error_code": "policy_denied",
                "provider": "goose_cli",
                "execution_mode": "chat",
                "endpoint_id": self.endpoint_id,
                "text": "",
                "result": "",
                "goose_version": "",
                "underlying_provider": None,
                "model": None,
                "session": None,
                "tool_call_count": 0,
                "side_effects_started": False,
                "elapsed_time": 0.0,
            }

        try:
            prompt = sanitize_input(prompt, max_length=_MAX_PROMPT_CHARS)
        except ValueError:
            self._record_failure(0.0)
            return {
                "status": "error",
                "success": False,
                "error": "Input validation error: ValueError",
                "error_code": "invalid_contract",
                "provider": "goose_cli",
                "execution_mode": "chat",
                "endpoint_id": self.endpoint_id,
                "text": "",
                "result": "",
                "tool_call_count": 0,
                "side_effects_started": False,
                "elapsed_time": 0.0,
            }

        try:
            from ipfs_accelerate_py.cli_runtime.contracts import (
                CLICapabilities,
                CLIRequest,
                ExecutionMode,
            )
            from ipfs_accelerate_py.cli_runtime.errors import (
                CLIRuntimeError,
                PolicyDeniedError,
                ContractValidationError as _CVE,
            )
            from ipfs_accelerate_py.cli_runtime.providers.goose import (
                DEFAULT_AGENT_MAX_TOOL_REPETITIONS,
                DEFAULT_AGENT_MAX_TURNS,
                DEFAULT_AGENT_TIMEOUT_SECONDS,
                DEFAULT_CHAT_TIMEOUT_SECONDS,
                GooseAgentPolicy,
                GooseErrorKind,
                GooseProviderError,
            )
        except ImportError as exc:
            self._record_failure(0.0)
            return {
                "status": "error",
                "success": False,
                "error": f"Goose provider unavailable: {type(exc).__name__}",
                "error_code": "provider_load_failed",
                "provider": "goose_cli",
                "execution_mode": "chat",
                "endpoint_id": self.endpoint_id,
                "text": "",
                "result": "",
                "tool_call_count": 0,
                "side_effects_started": False,
                "elapsed_time": 0.0,
            }

        cfg = self.config if isinstance(self.config, dict) else {}
        mode_raw = (
            kwargs.get("execution_mode")
            or kwargs.get("mode")
            or cfg.get("execution_mode")
            or "chat"
        )
        wants_agent = bool(
            kwargs.get("agent")
            or kwargs.get("side_effecting")
            or kwargs.get("with_tools")
            or str(mode_raw).strip().lower() == "agent"
        )
        execution_mode = "agent" if wants_agent else "chat"

        model_name = (
            kwargs.get("model_name")
            or kwargs.get("model")
            or cfg.get("model_name")
            or cfg.get("model")
        )
        goose_provider = (
            kwargs.get("goose_provider")
            or kwargs.get("provider")
            or cfg.get("goose_provider")
            or cfg.get("provider")
        )
        session_id = kwargs.get("session_id") or cfg.get("session_id")

        policy: Any = None
        if wants_agent:
            # Package enable policy gate (fail-closed).
            if not _goose_package_agent_enabled(cfg, kwargs):
                self._record_failure(0.0)
                return self._goose_error_envelope(
                    message=(
                        "agent mode requires package enable policy "
                        "(enable_agent / package_enable_agent / "
                        "package_policy.enable_agent)"
                    ),
                    error_code="policy_denied",
                    execution_mode="agent",
                    elapsed=0.0,
                    model=model_name,
                    goose_provider=goose_provider,
                    session=session_id,
                )
            allow_side = kwargs.get("allow_side_effects", cfg.get("allow_side_effects"))
            if allow_side is not True:
                self._record_failure(0.0)
                return self._goose_error_envelope(
                    message="agent mode requires allow_side_effects=True",
                    error_code="policy_denied",
                    execution_mode="agent",
                    elapsed=0.0,
                    model=model_name,
                    goose_provider=goose_provider,
                    session=session_id,
                )

            policy_raw = kwargs.get("agent_policy") or cfg.get("agent_policy")
            try:
                if isinstance(policy_raw, GooseAgentPolicy):
                    policy = policy_raw
                elif isinstance(policy_raw, Mapping):
                    # Ensure allow_side_effects is present on the mapping.
                    payload = dict(policy_raw)
                    payload.setdefault("allow_side_effects", True)
                    policy = GooseAgentPolicy.from_mapping(payload)
                else:
                    cwd = (
                        kwargs.get("cwd")
                        or kwargs.get("workspace")
                        or cfg.get("cwd")
                        or cfg.get("workspace")
                    )
                    path_root = (
                        kwargs.get("path_root")
                        or kwargs.get("GOOSE_PATH_ROOT")
                        or kwargs.get("goose_path_root")
                        or cfg.get("path_root")
                        or cfg.get("GOOSE_PATH_ROOT")
                        or cfg.get("goose_path_root")
                    )
                    if not cwd or not path_root:
                        self._record_failure(0.0)
                        return self._goose_error_envelope(
                            message=(
                                "agent mode requires absolute cwd and "
                                "path_root (GOOSE_PATH_ROOT)"
                            ),
                            error_code="policy_denied",
                            execution_mode="agent",
                            elapsed=0.0,
                            model=model_name,
                            goose_provider=goose_provider,
                            session=session_id,
                        )
                    builtins = kwargs.get("builtins") or kwargs.get("with_builtin")
                    extensions = (
                        kwargs.get("extensions") or kwargs.get("with_extension")
                    )
                    roots = kwargs.get("allowed_cwd_roots") or cfg.get(
                        "allowed_cwd_roots"
                    )
                    if isinstance(builtins, str):
                        builtins = tuple(
                            p.strip() for p in builtins.split(",") if p.strip()
                        )
                    if isinstance(extensions, str):
                        extensions = tuple(
                            p.strip() for p in extensions.split(",") if p.strip()
                        )
                    if isinstance(roots, list):
                        roots = tuple(roots)
                    max_turns = int(
                        kwargs.get("max_turns")
                        or cfg.get("max_turns")
                        or DEFAULT_AGENT_MAX_TURNS
                    )
                    max_reps = int(
                        kwargs.get("max_tool_repetitions")
                        or cfg.get("max_tool_repetitions")
                        or DEFAULT_AGENT_MAX_TOOL_REPETITIONS
                    )
                    timeout_s = float(
                        kwargs.get("timeout_seconds")
                        or kwargs.get("timeout")
                        or timeout
                        or DEFAULT_AGENT_TIMEOUT_SECONDS
                    )
                    max_out = kwargs.get("max_output_bytes")
                    if max_out is None:
                        max_out = cfg.get("max_output_bytes")
                    if max_out is None:
                        # Finite output bound required for agent mode.
                        max_out = _MAX_TEXT_CHARS
                    approval = str(
                        kwargs.get("approval_mode")
                        or cfg.get("approval_mode")
                        or "approve"
                    )
                    policy = GooseAgentPolicy(
                        allow_side_effects=True,
                        cwd=str(cwd),
                        path_root=str(path_root),
                        approval_mode=approval,
                        session_id=session_id,
                        resume_session=bool(
                            kwargs.get("resume_session", cfg.get("resume_session", False))
                        ),
                        builtins=tuple(builtins or ()),
                        extensions=tuple(extensions or ()),
                        max_turns=max_turns,
                        max_tool_repetitions=max_reps,
                        timeout_seconds=timeout_s,
                        max_output_bytes=int(max_out) if max_out is not None else None,
                        allowed_cwd_roots=tuple(roots or ()),
                    )
            except (PolicyDeniedError, _CVE, GooseProviderError, CLIRuntimeError) as exc:
                self._record_failure(0.0)
                code = getattr(getattr(exc, "code", None), "value", None) or "policy_denied"
                return self._goose_error_envelope(
                    message=str(exc)[:512] or type(exc).__name__,
                    error_code=str(code),
                    execution_mode="agent",
                    elapsed=0.0,
                    model=model_name,
                    goose_provider=goose_provider,
                    session=session_id,
                )
            except (TypeError, ValueError) as exc:
                self._record_failure(0.0)
                return self._goose_error_envelope(
                    message=f"invalid agent policy: {type(exc).__name__}",
                    error_code="invalid_contract",
                    execution_mode="agent",
                    elapsed=0.0,
                    model=model_name,
                    goose_provider=goose_provider,
                    session=session_id,
                )

            # Finite turns/time/output already validated by GooseAgentPolicy;
            # re-assert fail-closed if somehow missing.
            if (
                policy.max_turns < 1
                or policy.timeout_seconds <= 0
                or (
                    policy.max_output_bytes is not None
                    and policy.max_output_bytes < 1
                )
            ):
                self._record_failure(0.0)
                return self._goose_error_envelope(
                    message="agent mode requires finite turns/time/output bounds",
                    error_code="policy_denied",
                    execution_mode="agent",
                    elapsed=0.0,
                    model=model_name,
                    goose_provider=goose_provider,
                    session=session_id,
                )

        # Chat timeout: prefer explicit timeout_seconds, else adapter timeout.
        if wants_agent and policy is not None:
            run_timeout = float(policy.timeout_seconds)
        else:
            run_timeout = float(
                kwargs.get("timeout_seconds")
                or timeout
                or DEFAULT_CHAT_TIMEOUT_SECONDS
            )

        metadata: Dict[str, str] = {}
        if goose_provider:
            metadata["goose_provider"] = str(goose_provider)
        if not wants_agent:
            if kwargs.get("max_turns") is not None:
                metadata["max_turns"] = str(kwargs["max_turns"])
            if kwargs.get("max_tool_repetitions") is not None:
                metadata["max_tool_repetitions"] = str(
                    kwargs["max_tool_repetitions"]
                )
        output_format = kwargs.get("output_format")
        if output_format:
            metadata["output_format"] = str(output_format)
        streaming = bool(kwargs.get("stream") or kwargs.get("streaming"))

        request = CLIRequest(
            prompt=str(prompt),
            mode=ExecutionMode.AGENT if wants_agent else ExecutionMode.CHAT,
            model_name=str(model_name) if model_name else None,
            provider_name="goose_cli",
            provider_override=str(goose_provider) if goose_provider else None,
            side_effecting=bool(wants_agent),
            cacheable=not wants_agent,
            retryable=not wants_agent,
            streaming=streaming,
            session_id=str(session_id) if (wants_agent and session_id) else None,
            tools=tuple(policy.builtins) if (wants_agent and policy) else (),
            timeout_seconds=run_timeout,
            workspace=(
                str(policy.cwd) if (wants_agent and policy) else None
            ),
            metadata=metadata,
            capabilities=(
                CLICapabilities.agent_defaults()
                if wants_agent
                else CLICapabilities.chat_defaults()
            ),
        )

        try:
            provider = self._get_provider()
            # Prefer configured executable path when present.
            if self.cli_path and (
                os.path.isfile(self.cli_path)
                or (os.sep in str(self.cli_path) and os.path.exists(self.cli_path))
            ):
                provider.executable = self.cli_path
            result = provider.generate_result(
                request,
                agent_policy=policy,
                goose_provider=str(goose_provider) if goose_provider else None,
                output_format=str(output_format) if output_format else None,
            )
        except (GooseProviderError, PolicyDeniedError, _CVE, CLIRuntimeError) as exc:
            elapsed = time.time() - start_time
            self._record_failure(elapsed)
            kind = getattr(exc, "kind", None)
            code = getattr(getattr(exc, "code", None), "value", None)
            if code is None and kind is not None:
                try:
                    from ipfs_accelerate_py.cli_runtime.providers.goose import (
                        goose_error_code,
                    )
                    code = goose_error_code(kind).value
                except Exception:  # noqa: BLE001
                    code = "internal"
            return self._goose_error_envelope(
                message=str(exc)[:512] or type(exc).__name__,
                error_code=str(code or "internal"),
                execution_mode=execution_mode,
                elapsed=elapsed,
                model=model_name,
                goose_provider=goose_provider,
                session=session_id if wants_agent else None,
                goose_version=getattr(self._provider, "version", "") or "",
                side_effects_started=bool(
                    getattr(exc, "side_effects_started", False)
                ),
                goose_error_kind=getattr(kind, "value", None),
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - start_time
            self._record_failure(elapsed)
            return self._goose_error_envelope(
                message=f"CLI execution error: {type(exc).__name__}",
                error_code="internal",
                execution_mode=execution_mode,
                elapsed=elapsed,
                model=model_name,
                goose_provider=goose_provider,
                session=session_id if wants_agent else None,
            )

        elapsed = time.time() - start_time
        meta = dict(result.metadata or {})
        text = result.text or ""
        if isinstance(text, str) and len(text) > _MAX_TEXT_CHARS:
            text = text[: max(0, _MAX_TEXT_CHARS - 3)] + "..."

        tool_call_count = 0
        if "tool_call_count" in meta:
            try:
                tool_call_count = int(meta["tool_call_count"])
            except (TypeError, ValueError):
                tool_call_count = 0
        side_effects_started = bool(
            result.had_side_effect_event or result.side_effecting
        )
        goose_version = (
            meta.get("goose_version")
            or getattr(self._provider, "version", "")
            or ""
        )
        underlying = (
            meta.get("goose_provider")
            or (str(goose_provider) if goose_provider else None)
        )
        model_out = result.model_name or (
            str(model_name) if model_name else None
        )
        session_out = (
            request.session_id
            or (policy.session_id if policy is not None else None)
        )

        if not result.ok:
            self._record_failure(elapsed)
            err = result.error
            error_code = err.code.value if err is not None else "nonzero_exit"
            error_msg = (
                err.message if err is not None else "Goose run failed"
            )
            return self._goose_error_envelope(
                message=error_msg,
                error_code=error_code,
                execution_mode=execution_mode,
                elapsed=elapsed,
                model=model_out,
                goose_provider=underlying,
                session=session_out,
                goose_version=goose_version,
                tool_call_count=tool_call_count,
                side_effects_started=side_effects_started,
                text=text,
                returncode=result.exit_code,
                goose_error_kind=meta.get("goose_error_kind"),
            )

        self._record_success(elapsed)
        envelope: Dict[str, Any] = {
            "status": "success",
            "success": True,
            "provider": "goose_cli",
            "execution_mode": execution_mode,
            "text": text,
            "result": text,
            "goose_version": goose_version,
            "underlying_provider": underlying,
            "model": model_out,
            "session": session_out,
            "tool_call_count": tool_call_count,
            "side_effects_started": side_effects_started,
            "elapsed_time": elapsed,
            "endpoint_id": self.endpoint_id,
            "endpoint_type": "cli",
            "returncode": (
                int(result.exit_code) if result.exit_code is not None else 0
            ),
            "error": None,
            "error_code": None,
            "task_type": task_type,
        }
        # Hard guarantee: never echo prompt or credentials.
        envelope.pop("prompt", None)
        return envelope

    def _goose_error_envelope(
        self,
        *,
        message: str,
        error_code: str,
        execution_mode: str,
        elapsed: float,
        model: Any = None,
        goose_provider: Any = None,
        session: Any = None,
        goose_version: str = "",
        tool_call_count: int = 0,
        side_effects_started: bool = False,
        text: str = "",
        returncode: Any = None,
        goose_error_kind: Any = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": "error",
            "success": False,
            "error": _clip_text(message, 512),
            "error_code": str(error_code),
            "provider": "goose_cli",
            "execution_mode": execution_mode,
            "text": _clip_text(text, _MAX_TEXT_CHARS) if text else "",
            "result": "",
            "goose_version": goose_version or "",
            "underlying_provider": goose_provider,
            "model": model,
            "session": session,
            "tool_call_count": int(tool_call_count or 0),
            "side_effects_started": bool(side_effects_started),
            "elapsed_time": float(elapsed),
            "endpoint_id": self.endpoint_id,
            "endpoint_type": "cli",
        }
        if returncode is not None:
            try:
                payload["returncode"] = int(returncode)
            except (TypeError, ValueError):
                pass
        if goose_error_kind:
            payload["goose_error_kind"] = str(goose_error_kind)
        payload.pop("prompt", None)
        return payload

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats["tool"] = "goose"
        stats["provider"] = "goose_cli"
        return stats

    def get_capabilities(self) -> Dict[str, Any]:
        caps = super().get_capabilities()
        caps["provider"] = "goose_cli"
        caps["default_execution_mode"] = "chat"
        caps["agent_requires_policy"] = True
        caps["authority_keys"] = sorted(_GOOSE_AUTHORITY_EXECUTE_KEYS)
        return caps


# ---------------------------------------------------------------------------
# Compatibility shims over the canonical factory
# (ipfs_accelerate_py.cli_runtime.endpoints)
# ---------------------------------------------------------------------------

from ipfs_accelerate_py.cli_runtime.endpoints import (  # noqa: E402
    CLI_ADAPTER_REGISTRY,
    create_cli_endpoint,
    execute_cli_inference as _canonical_execute_cli_inference,
    get_cli_endpoint as _canonical_get_cli_endpoint,
    list_cli_endpoints as _canonical_list_cli_endpoints,
    register_cli_endpoint as _canonical_register_cli_endpoint,
    reset_default_endpoint_registry,
)


def register_cli_endpoint(
    adapter: Optional["CLIEndpointAdapter"] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Register a CLI endpoint adapter via the canonical factory.

    Args:
        adapter: Concrete CLIEndpointAdapter subclass instance, or omit and
            pass ``tool=...`` to construct via the concrete factory.
        **kwargs: Forwarded to the canonical registrar
            (``tool``, ``endpoint_id``, ``config``, ``replace``, ``probe``, ...)

    Returns:
        Dictionary with registration status
    """
    return _canonical_register_cli_endpoint(adapter, **kwargs)


def get_cli_endpoint(endpoint_id: str) -> Optional["CLIEndpointAdapter"]:
    """Get a registered CLI endpoint adapter"""
    return _canonical_get_cli_endpoint(endpoint_id)  # type: ignore[return-value]


def list_cli_endpoints(*, probe: bool = False) -> List[Dict[str, Any]]:
    """List all registered CLI endpoints (no provider probe by default)."""
    endpoints = _canonical_list_cli_endpoints(probe=probe)
    # Preserve legacy shape: prefer adapter.get_stats when available.
    enriched: List[Dict[str, Any]] = []
    for item in endpoints:
        endpoint_id = item.get("endpoint_id")
        adapter = get_cli_endpoint(endpoint_id) if endpoint_id else None
        if adapter is not None and hasattr(adapter, "get_stats"):
            stats = dict(adapter.get_stats())
            stats.setdefault("tool", item.get("tool"))
            stats.setdefault("health", item.get("health"))
            enriched.append(stats)
        else:
            enriched.append(item)
    return enriched


def execute_cli_inference(
    endpoint_id: str,
    prompt: str,
    task_type: str = "text_generation",
    timeout: int = 30,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute inference using a registered CLI endpoint

    Args:
        endpoint_id: ID of the registered CLI endpoint
        prompt: Input prompt
        task_type: Type of task to perform
        timeout: Maximum execution time in seconds
        **kwargs: Additional task-specific parameters

    Returns:
        Dictionary with inference results. Nonzero exit is failure; errors
        never echo the prompt.
    """
    return _canonical_execute_cli_inference(
        endpoint_id,
        prompt,
        task_type=task_type,
        timeout=timeout,
        **kwargs,
    )
