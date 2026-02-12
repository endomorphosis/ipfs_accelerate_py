"""
GitHub Copilot CLI Python Wrapper

This module provides Python wrappers for GitHub Copilot CLI commands,
enabling AI-assisted development features with caching and retry logic.
"""

import logging
import subprocess
import time
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import cache from github_cli
try:
    from ipfs_accelerate_py.github_cli.cache import GitHubAPICache, get_global_cache
except ImportError:
    # Fallback if running standalone
    GitHubAPICache = None
    get_global_cache = None


class CopilotCLI:
    """Python wrapper for GitHub Copilot CLI commands with caching and retry."""
    
    def __init__(
        self,
        copilot_path: str = "github-copilot-cli",
        enable_cache: bool = True,
        cache: Optional['GitHubAPICache'] = None,
        cache_ttl: int = 300
    ):
        """
        Initialize Copilot CLI wrapper.
        
        Args:
            copilot_path: Path to github-copilot-cli executable
            enable_cache: Whether to enable response caching
            cache: Custom cache instance (uses global cache if None)
            cache_ttl: Default cache TTL in seconds (default: 5 minutes)
        """
        self.copilot_path = copilot_path
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        
        # Set up cache (share with GitHub CLI cache)
        if enable_cache and GitHubAPICache:
            self.cache = cache if cache is not None else get_global_cache()
        else:
            self.cache = None
        
        self._verify_installation()
    
    def _verify_installation(self) -> None:
        """Verify that Copilot CLI is installed."""
        try:
            # Try to check if the command exists
            result = subprocess.run(
                ["which", self.copilot_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                logger.warning(f"Copilot CLI not found at {self.copilot_path}")
                # Don't fail - it might be installed differently
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Could not verify Copilot CLI installation: {e}")
    
    def _run_command(
        self,
        args: List[str],
        stdin: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ) -> Dict[str, Any]:
        """
        Run a Copilot CLI command with exponential backoff retry.
        
        Args:
            args: Command arguments
            stdin: Optional stdin input
            timeout: Command timeout in seconds
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
            
        Returns:
            Dict with stdout, stderr, and returncode
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                cmd = [self.copilot_path] + args
                if attempt > 0:
                    logger.debug(f"Retry attempt {attempt}/{max_retries} for command: {' '.join(cmd)}")
                else:
                    logger.debug(f"Running command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    input=stdin,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                # Check for rate limiting or API errors
                stderr_lower = result.stderr.lower()
                if result.returncode != 0 and any(keyword in stderr_lower for keyword in 
                    ['rate limit', 'api rate limit', 'too many requests', 'service unavailable', '503']):
                    if attempt < max_retries:
                        # Exponential backoff with jitter
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                        logger.warning(f"API error detected, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                
                # Success or non-retryable error
                return {
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                    "returncode": result.returncode,
                    "success": result.returncode == 0,
                    "attempts": attempt + 1
                }
                
            except subprocess.TimeoutExpired:
                last_error = f"Command timed out after {timeout}s"
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    logger.warning(f"Timeout, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                    
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    logger.warning(f"Error: {e}, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
        
        # All retries exhausted
        logger.error(f"Command failed after {max_retries + 1} attempts: {last_error}")
        return {
            "stdout": "",
            "stderr": last_error or "Unknown error",
            "returncode": -1,
            "success": False,
            "attempts": max_retries + 1
        }
    
    def suggest_command(
        self,
        prompt: str,
        shell: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get command suggestions from Copilot.
        
        Args:
            prompt: Natural language description of desired command
            shell: Shell type (bash, zsh, powershell, etc.)
            use_cache: Whether to use cached results
            
        Returns:
            Dict with suggested command and metadata
        """
        # Check cache first
        if use_cache and self.cache:
            cached_result = self.cache.get("copilot_suggest", prompt=prompt, shell=shell)
            if cached_result is not None:
                logger.debug(f"Using cached copilot suggestion for prompt: {prompt[:50]}...")
                return cached_result
        
        args = ["suggest"]
        if shell:
            args.extend(["--shell", shell])
        
        result = self._run_command(args, stdin=prompt)
        
        response = {
            "prompt": prompt,
            "suggestion": result["stdout"],
            "error": result["stderr"] if not result["success"] else None,
            "success": result["success"],
            "attempts": result.get("attempts", 1)
        }
        
        # Cache successful results
        if use_cache and self.cache and result["success"]:
            self.cache.put("copilot_suggest", response, ttl=self.cache_ttl, prompt=prompt, shell=shell)
        
        return response
    
    def explain_command(self, command: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get an explanation for a command.
        
        Args:
            command: Command to explain
            use_cache: Whether to use cached results
            
        Returns:
            Dict with explanation and metadata
        """
        # Check cache first
        if use_cache and self.cache:
            cached_result = self.cache.get("copilot_explain", command=command)
            if cached_result is not None:
                logger.debug(f"Using cached copilot explanation for: {command[:50]}...")
                return cached_result
        
        result = self._run_command(["explain"], stdin=command)
        
        response = {
            "command": command,
            "explanation": result["stdout"],
            "error": result["stderr"] if not result["success"] else None,
            "success": result["success"],
            "attempts": result.get("attempts", 1)
        }
        
        # Cache successful results
        if use_cache and self.cache and result["success"]:
            self.cache.put("copilot_explain", response, ttl=self.cache_ttl, command=command)
        
        return response
    
    def suggest_git_command(self, prompt: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get Git command suggestions from Copilot.
        
        Args:
            prompt: Natural language description of desired Git operation
            use_cache: Whether to use cached results
            
        Returns:
            Dict with suggested Git command and metadata
        """
        # Check cache first
        if use_cache and self.cache:
            cached_result = self.cache.get("copilot_git", prompt=prompt)
            if cached_result is not None:
                logger.debug(f"Using cached copilot git suggestion for: {prompt[:50]}...")
                return cached_result
        
        result = self._run_command(["git-assist"], stdin=prompt)
        
        response = {
            "prompt": prompt,
            "suggestion": result["stdout"],
            "error": result["stderr"] if not result["success"] else None,
            "success": result["success"],
            "attempts": result.get("attempts", 1)
        }
        
        # Cache successful results
        if use_cache and self.cache and result["success"]:
            self.cache.put("copilot_git", response, ttl=self.cache_ttl, prompt=prompt)
        
        return response
