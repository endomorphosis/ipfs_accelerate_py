"""
GitHub Copilot CLI Python Wrapper

This module provides Python wrappers for GitHub Copilot CLI commands,
enabling AI-assisted development features.
"""

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CopilotCLI:
    """Python wrapper for GitHub Copilot CLI commands."""
    
    def __init__(self, copilot_path: str = "github-copilot-cli"):
        """
        Initialize Copilot CLI wrapper.
        
        Args:
            copilot_path: Path to github-copilot-cli executable
        """
        self.copilot_path = copilot_path
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
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Run a Copilot CLI command and return the result.
        
        Args:
            args: Command arguments
            stdin: Optional stdin input
            timeout: Command timeout in seconds
            
        Returns:
            Dict with stdout, stderr, and returncode
        """
        try:
            cmd = [self.copilot_path] + args
            logger.debug(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s")
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "returncode": -1,
                "success": False
            }
        except Exception as e:
            logger.error(f"Error running command: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "success": False
            }
    
    def suggest_command(
        self,
        prompt: str,
        shell: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get command suggestions from Copilot.
        
        Args:
            prompt: Natural language description of desired command
            shell: Shell type (bash, zsh, powershell, etc.)
            
        Returns:
            Dict with suggested command and metadata
        """
        args = ["suggest"]
        if shell:
            args.extend(["--shell", shell])
        
        result = self._run_command(args, stdin=prompt)
        
        return {
            "prompt": prompt,
            "suggestion": result["stdout"],
            "error": result["stderr"] if not result["success"] else None,
            "success": result["success"]
        }
    
    def explain_command(self, command: str) -> Dict[str, Any]:
        """
        Get an explanation for a command.
        
        Args:
            command: Command to explain
            
        Returns:
            Dict with explanation and metadata
        """
        result = self._run_command(["explain"], stdin=command)
        
        return {
            "command": command,
            "explanation": result["stdout"],
            "error": result["stderr"] if not result["success"] else None,
            "success": result["success"]
        }
    
    def suggest_git_command(self, prompt: str) -> Dict[str, Any]:
        """
        Get Git command suggestions from Copilot.
        
        Args:
            prompt: Natural language description of desired Git operation
            
        Returns:
            Dict with suggested Git command and metadata
        """
        result = self._run_command(["git-assist"], stdin=prompt)
        
        return {
            "prompt": prompt,
            "suggestion": result["stdout"],
            "error": result["stderr"] if not result["success"] else None,
            "success": result["success"]
        }
