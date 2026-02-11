"""
CLI Endpoint Adapters for IPFS Accelerate MCP Server

This module provides adapters for CLI-based AI tools to integrate them
into the IPFS Accelerate multiplexing and queue system.

Supported CLI Tools:
- Claude Code (Anthropic)
- OpenAI Codex CLI
- Google Gemini CLI
- VSCode CLI (GitHub Copilot)
"""

import os
import subprocess
import json
import logging
import time
import shutil
import re
import platform
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod

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
    """Base class for CLI endpoint adapters"""
    
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
                "error": f"Version check failed: {str(e)}"
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
            Dictionary with inference results and metadata
        """
        start_time = time.time()
        self.stats["requests"] += 1
        
        try:
            # Sanitize prompt input
            prompt = sanitize_input(prompt, max_length=100000)
            
            # Format command
            cmd_args = self._format_prompt(prompt, task_type, **kwargs)
            
            # Validate command arguments
            cmd_args = validate_cli_args(cmd_args)
            
            logger.info(f"Executing CLI command for {self.endpoint_id}: {' '.join(cmd_args[:3])}...")
            
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
            
            # Parse response
            response = self._parse_response(result.stdout, result.stderr)
            
            # Update stats
            elapsed_time = time.time() - start_time
            self.stats["successes"] += 1
            self.stats["total_time"] += elapsed_time
            self.stats["avg_time"] = self.stats["total_time"] / self.stats["requests"]
            
            # Add metadata
            response.update({
                "endpoint_id": self.endpoint_id,
                "endpoint_type": "cli",
                "elapsed_time": elapsed_time,
                "status": "success",
                "returncode": result.returncode
            })
            
            return response
            
        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            self.stats["failures"] += 1
            logger.error(f"CLI execution timeout for {self.endpoint_id}")
            return {
                "error": "CLI execution timeout",
                "endpoint_id": self.endpoint_id,
                "elapsed_time": elapsed_time,
                "status": "timeout"
            }
        
        except ValueError as e:
            # Input validation error
            elapsed_time = time.time() - start_time
            self.stats["failures"] += 1
            logger.error(f"Input validation error for {self.endpoint_id}: {e}")
            return {
                "error": f"Input validation error: {str(e)}",
                "endpoint_id": self.endpoint_id,
                "elapsed_time": elapsed_time,
                "status": "validation_error"
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.stats["failures"] += 1
            logger.error(f"CLI execution error for {self.endpoint_id}: {e}")
            return {
                "error": str(e),
                "endpoint_id": self.endpoint_id,
                "elapsed_time": elapsed_time,
                "status": "error"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get endpoint statistics"""
        return {
            "endpoint_id": self.endpoint_id,
            "endpoint_type": "cli",
            "cli_path": self.cli_path,
            "available": self.is_available(),
            "stats": self.stats
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


# Registry to keep track of registered CLI adapters
CLI_ADAPTER_REGISTRY: Dict[str, CLIEndpointAdapter] = {}


def register_cli_endpoint(adapter: CLIEndpointAdapter) -> Dict[str, Any]:
    """
    Register a CLI endpoint adapter
    
    Args:
        adapter: CLIEndpointAdapter instance
        
    Returns:
        Dictionary with registration status
    """
    try:
        endpoint_id = adapter.endpoint_id
        CLI_ADAPTER_REGISTRY[endpoint_id] = adapter
        
        logger.info(f"Registered CLI endpoint: {endpoint_id} (available: {adapter.is_available()})")
        
        return {
            "status": "success",
            "endpoint_id": endpoint_id,
            "available": adapter.is_available(),
            "message": f"CLI endpoint {endpoint_id} registered successfully"
        }
    except Exception as e:
        logger.error(f"Failed to register CLI endpoint: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def get_cli_endpoint(endpoint_id: str) -> Optional[CLIEndpointAdapter]:
    """Get a registered CLI endpoint adapter"""
    return CLI_ADAPTER_REGISTRY.get(endpoint_id)


def list_cli_endpoints() -> List[Dict[str, Any]]:
    """List all registered CLI endpoints"""
    return [adapter.get_stats() for adapter in CLI_ADAPTER_REGISTRY.values()]


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
        Dictionary with inference results
    """
    adapter = get_cli_endpoint(endpoint_id)
    
    if not adapter:
        return {
            "error": f"CLI endpoint '{endpoint_id}' not found",
            "status": "error"
        }
    
    if not adapter.is_available():
        return {
            "error": f"CLI tool for endpoint '{endpoint_id}' is not available",
            "status": "error"
        }
    
    return adapter.execute(prompt, task_type, timeout, **kwargs)
