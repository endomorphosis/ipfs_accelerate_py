"""
GitHub Copilot CLI Integration with Common Cache

Wraps GitHub Copilot CLI to use the common cache infrastructure.
"""

import logging
from typing import Any, Dict, Optional

from .base_cli_wrapper import BaseCLIWrapper
from ..common.llm_cache import get_llm_cache
from ..common.base_cache import BaseAPICache

logger = logging.getLogger(__name__)


class CopilotCLIIntegration(BaseCLIWrapper):
    """
    GitHub Copilot CLI integration with common cache infrastructure.
    
    Uses LLM cache for AI-generated suggestions.
    """
    
    def __init__(
        self,
        copilot_path: str = "github-copilot-cli",
        enable_cache: bool = True,
        cache: Optional[BaseAPICache] = None,
        **kwargs
    ):
        """
        Initialize Copilot CLI integration.
        
        Args:
            copilot_path: Path to github-copilot-cli executable
            enable_cache: Whether to enable caching
            cache: Custom cache instance (uses LLM cache if None)
            **kwargs: Additional arguments for BaseCLIWrapper
        """
        if cache is None:
            cache = get_llm_cache("copilot")
        
        super().__init__(
            cli_path=copilot_path,
            cache=cache,
            enable_cache=enable_cache,
            **kwargs
        )
    
    def get_tool_name(self) -> str:
        return "GitHub Copilot CLI"
    
    def suggest_command(
        self,
        prompt: str,
        shell: str = "bash",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get command suggestion from Copilot.
        
        Args:
            prompt: Natural language description of desired command
            shell: Target shell (bash, zsh, powershell, etc.)
            **kwargs: Additional arguments
            
        Returns:
            Command result dict with suggestion
        """
        args = ["--", prompt]
        
        return self._run_command_with_retry(
            args,
            "suggest_command",
            prompt=prompt,
            shell=shell
        )
    
    def explain_command(
        self,
        command: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get explanation of a command.
        
        Args:
            command: Command to explain
            **kwargs: Additional arguments
            
        Returns:
            Command result dict with explanation
        """
        args = ["--explain", command]
        
        return self._run_command_with_retry(
            args,
            "explain_command",
            command=command
        )


# Global instance
_global_copilot_cli: Optional[CopilotCLIIntegration] = None


def get_copilot_cli_integration() -> CopilotCLIIntegration:
    """Get or create the global Copilot CLI integration instance."""
    global _global_copilot_cli
    
    if _global_copilot_cli is None:
        _global_copilot_cli = CopilotCLIIntegration()
    
    return _global_copilot_cli
