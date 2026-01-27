"""
GitHub CLI Integration with Common Cache

Wraps the existing GitHub CLI to use the common cache infrastructure.
"""

import logging
from typing import Any, Dict, List, Optional

from .base_cli_wrapper import BaseCLIWrapper
from ..common.base_cache import BaseAPICache, register_cache

logger = logging.getLogger(__name__)


class GitHubCLICache(BaseAPICache):
    """Cache adapter for GitHub CLI operations."""
    
    DEFAULT_TTLS = {
        "repo_list": 300,  # 5 minutes
        "repo_view": 300,  # 5 minutes
        "pr_list": 60,  # 1 minute
        "pr_view": 60,  # 1 minute
        "issue_list": 60,  # 1 minute
        "issue_view": 60,  # 1 minute
        "run_list": 60,  # 1 minute
        "run_view": 30,  # 30 seconds
        "workflow_list": 300,  # 5 minutes
    }
    
    def get_cache_namespace(self) -> str:
        return "github_cli"
    
    def extract_validation_fields(self, operation: str, data: Any) -> Optional[Dict[str, Any]]:
        """Extract validation fields from GitHub CLI responses."""
        if not isinstance(data, dict):
            return None
        
        validation = {}
        
        # For responses with stdout
        stdout = data.get("stdout", "")
        if stdout and isinstance(stdout, str):
            # Try to get length as a simple validation
            validation["output_length"] = len(stdout)
        
        return validation if validation else None
    
    def get_default_ttl_for_operation(self, operation: str) -> int:
        return self.DEFAULT_TTLS.get(operation, self.default_ttl)


class GitHubCLIIntegration(BaseCLIWrapper):
    """
    GitHub CLI integration with common cache infrastructure.
    
    Provides caching for gh CLI commands using CID-based lookups.
    """
    
    def __init__(
        self,
        gh_path: str = "gh",
        enable_cache: bool = True,
        cache: Optional[BaseAPICache] = None,
        **kwargs
    ):
        """
        Initialize GitHub CLI integration.
        
        Args:
            gh_path: Path to gh executable
            enable_cache: Whether to enable caching
            cache: Custom cache instance (creates new if None)
            **kwargs: Additional arguments for BaseCLIWrapper
        """
        if cache is None:
            cache = GitHubCLICache()
            register_cache("github_cli", cache)
        
        super().__init__(
            cli_path=gh_path,
            cache=cache,
            enable_cache=enable_cache,
            **kwargs
        )
    
    def get_tool_name(self) -> str:
        return "GitHub CLI"
    
    def list_repos(
        self,
        owner: Optional[str] = None,
        limit: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        List repositories.
        
        Args:
            owner: Repository owner (uses authenticated user if None)
            limit: Maximum number of repos to return
            **kwargs: Additional gh arguments
            
        Returns:
            Command result dict
        """
        args = ["repo", "list"]
        if owner:
            args.append(owner)
        args.extend(["--limit", str(limit), "--json", "name,description,url,updatedAt"])
        
        return self._run_command_with_retry(
            args,
            "repo_list",
            owner=owner or "self",
            limit=limit
        )
    
    def view_repo(self, repo: str, **kwargs) -> Dict[str, Any]:
        """
        View repository details.
        
        Args:
            repo: Repository in format owner/repo
            **kwargs: Additional arguments
            
        Returns:
            Command result dict
        """
        args = ["repo", "view", repo, "--json", "name,description,url,updatedAt,pushedAt"]
        
        return self._run_command_with_retry(
            args,
            "repo_view",
            repo=repo
        )
    
    def list_prs(
        self,
        repo: str,
        state: str = "open",
        limit: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        List pull requests.
        
        Args:
            repo: Repository in format owner/repo
            state: PR state (open, closed, merged, all)
            limit: Maximum number of PRs to return
            **kwargs: Additional arguments
            
        Returns:
            Command result dict
        """
        args = ["pr", "list", "--repo", repo, "--state", state, "--limit", str(limit),
                "--json", "number,title,state,updatedAt"]
        
        return self._run_command_with_retry(
            args,
            "pr_list",
            repo=repo,
            state=state,
            limit=limit
        )
    
    def view_pr(self, repo: str, pr_number: int, **kwargs) -> Dict[str, Any]:
        """
        View pull request details.
        
        Args:
            repo: Repository in format owner/repo
            pr_number: PR number
            **kwargs: Additional arguments
            
        Returns:
            Command result dict
        """
        args = ["pr", "view", str(pr_number), "--repo", repo,
                "--json", "number,title,body,state,updatedAt"]
        
        return self._run_command_with_retry(
            args,
            "pr_view",
            repo=repo,
            pr_number=pr_number
        )
    
    def list_issues(
        self,
        repo: str,
        state: str = "open",
        limit: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """
        List issues.
        
        Args:
            repo: Repository in format owner/repo
            state: Issue state (open, closed, all)
            limit: Maximum number of issues to return
            **kwargs: Additional arguments
            
        Returns:
            Command result dict
        """
        args = ["issue", "list", "--repo", repo, "--state", state, "--limit", str(limit),
                "--json", "number,title,state,updatedAt"]
        
        return self._run_command_with_retry(
            args,
            "issue_list",
            repo=repo,
            state=state,
            limit=limit
        )
    
    def list_workflow_runs(
        self,
        repo: str,
        limit: int = 20,
        **kwargs
    ) -> Dict[str, Any]:
        """
        List workflow runs.
        
        Args:
            repo: Repository in format owner/repo
            limit: Maximum number of runs to return
            **kwargs: Additional arguments
            
        Returns:
            Command result dict
        """
        args = ["run", "list", "--repo", repo, "--limit", str(limit),
                "--json", "databaseId,name,status,conclusion,createdAt"]
        
        return self._run_command_with_retry(
            args,
            "run_list",
            repo=repo,
            limit=limit
        )


# Global instance
_global_github_cli: Optional[GitHubCLIIntegration] = None


def get_github_cli_integration() -> GitHubCLIIntegration:
    """Get or create the global GitHub CLI integration instance."""
    global _global_github_cli
    
    if _global_github_cli is None:
        _global_github_cli = GitHubCLIIntegration()
    
    return _global_github_cli
