"""
CLI Endpoint Adapters for IPFS Accelerate MCP Server

This module provides adapters for CLI-based AI tools to integrate them
into the IPFS Accelerate multiplexing and queue system.

Supported CLI Tools:
- Claude Code (Anthropic)
- OpenAI Codex CLI
- Google Gemini CLI
"""

import os
import subprocess
import json
import logging
import time
import shutil
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger("ipfs_accelerate_mcp.tools.cli_endpoint_adapters")


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
        self.endpoint_id = endpoint_id
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
    
    def is_available(self) -> bool:
        """Check if the CLI tool is available"""
        if not self.cli_path:
            return False
        
        # Check if file exists and is executable
        if os.path.isfile(self.cli_path) and os.access(self.cli_path, os.X_OK):
            return True
        
        # Check if it's in PATH
        return shutil.which(self.cli_path) is not None
    
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
            # Format command
            cmd_args = self._format_prompt(prompt, task_type, **kwargs)
            
            logger.info(f"Executing CLI command for {self.endpoint_id}: {' '.join(cmd_args[:3])}...")
            
            # Execute CLI command
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, **self.config.get("env_vars", {})}
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


class ClaudeCodeAdapter(CLIEndpointAdapter):
    """Adapter for Claude Code CLI tool"""
    
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
        cmd.extend(["--model", model])
        
        # Add max tokens if specified
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 4096))
        cmd.extend(["--max-tokens", str(max_tokens)])
        
        # Add temperature if specified
        temperature = kwargs.get("temperature", self.config.get("temperature"))
        if temperature is not None:
            cmd.extend(["--temperature", str(temperature)])
        
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
