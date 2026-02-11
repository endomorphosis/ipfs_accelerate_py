"""
GitHub Kit - Core GitHub Operations

This module provides core GitHub operations without CLI dependencies.
It can be used by both the unified CLI and MCP server.
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class GitHubResult:
    """Result from a GitHub operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    raw_output: Optional[str] = None


class GitHubKit:
    """
    Core GitHub operations module.
    
    Provides GitHub functionality that can be used by CLI, MCP tools,
    or directly in Python code.
    """
    
    def __init__(
        self,
        gh_path: str = "gh",
        timeout: int = 30
    ):
        """
        Initialize GitHub Kit.
        
        Args:
            gh_path: Path to gh CLI executable
            timeout: Default timeout for operations
        """
        self.gh_path = gh_path
        self.timeout = timeout
        self._verify_installation()
    
    def _verify_installation(self) -> bool:
        """
        Verify that GitHub CLI is installed.
        
        Returns:
            True if installed, False otherwise
        """
        try:
            result = subprocess.run(
                [self.gh_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"GitHub CLI version: {result.stdout.strip()}")
                return True
            else:
                logger.warning(f"GitHub CLI verification failed: {result.returncode}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"GitHub CLI not found: {e}")
            return False
    
    def _run_command(
        self,
        args: List[str],
        timeout: Optional[int] = None
    ) -> GitHubResult:
        """
        Run a GitHub CLI command.
        
        Args:
            args: Command arguments
            timeout: Timeout in seconds
            
        Returns:
            GitHubResult with command output
        """
        timeout = timeout or self.timeout
        full_command = [self.gh_path] + args
        
        try:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                # Try to parse JSON if output looks like JSON
                data = result.stdout
                if data.strip().startswith('{') or data.strip().startswith('['):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        pass
                
                return GitHubResult(
                    success=True,
                    data=data,
                    raw_output=result.stdout
                )
            else:
                return GitHubResult(
                    success=False,
                    error=result.stderr or result.stdout,
                    raw_output=result.stdout
                )
        
        except subprocess.TimeoutExpired:
            return GitHubResult(
                success=False,
                error=f"Command timed out after {timeout} seconds"
            )
        except Exception as e:
            return GitHubResult(
                success=False,
                error=str(e)
            )
    
    # Repository Operations
    
    def list_repos(
        self,
        owner: Optional[str] = None,
        limit: int = 30
    ) -> GitHubResult:
        """
        List repositories.
        
        Args:
            owner: Repository owner (uses authenticated user if None)
            limit: Maximum number of repos to return
            
        Returns:
            GitHubResult with repository list
        """
        args = ["repo", "list"]
        if owner:
            args.append(owner)
        args.extend(["--limit", str(limit), "--json", "name,description,url,updatedAt"])
        
        return self._run_command(args)
    
    def get_repo(self, repo: str) -> GitHubResult:
        """
        Get repository details.
        
        Args:
            repo: Repository (owner/name)
            
        Returns:
            GitHubResult with repository details
        """
        args = ["repo", "view", repo, "--json", "name,description,url,updatedAt,defaultBranch,isPrivate"]
        return self._run_command(args)
    
    def clone_repo(self, repo: str, path: Optional[str] = None) -> GitHubResult:
        """
        Clone a repository.
        
        Args:
            repo: Repository (owner/name or URL)
            path: Target path (optional)
            
        Returns:
            GitHubResult with clone status
        """
        args = ["repo", "clone", repo]
        if path:
            args.append(path)
        
        return self._run_command(args, timeout=300)  # 5 min timeout for clones
    
    # Pull Request Operations
    
    def list_prs(
        self,
        repo: str,
        state: str = "open",
        limit: int = 30
    ) -> GitHubResult:
        """
        List pull requests.
        
        Args:
            repo: Repository (owner/name)
            state: PR state (open, closed, merged, all)
            limit: Maximum number of PRs to return
            
        Returns:
            GitHubResult with PR list
        """
        args = ["pr", "list", "--repo", repo, "--state", state, "--limit", str(limit),
                "--json", "number,title,state,url,createdAt,updatedAt"]
        
        return self._run_command(args)
    
    def get_pr(self, repo: str, number: int) -> GitHubResult:
        """
        Get pull request details.
        
        Args:
            repo: Repository (owner/name)
            number: PR number
            
        Returns:
            GitHubResult with PR details
        """
        args = ["pr", "view", str(number), "--repo", repo,
                "--json", "number,title,state,body,url,createdAt,updatedAt,author"]
        
        return self._run_command(args)
    
    def create_pr(
        self,
        repo: str,
        title: str,
        body: str,
        base: str = "main",
        head: Optional[str] = None
    ) -> GitHubResult:
        """
        Create a pull request.
        
        Args:
            repo: Repository (owner/name)
            title: PR title
            body: PR body/description
            base: Base branch
            head: Head branch (current branch if None)
            
        Returns:
            GitHubResult with created PR
        """
        args = ["pr", "create", "--repo", repo, "--title", title, "--body", body, "--base", base]
        if head:
            args.extend(["--head", head])
        
        return self._run_command(args)
    
    # Issue Operations
    
    def list_issues(
        self,
        repo: str,
        state: str = "open",
        limit: int = 30
    ) -> GitHubResult:
        """
        List issues.
        
        Args:
            repo: Repository (owner/name)
            state: Issue state (open, closed, all)
            limit: Maximum number of issues to return
            
        Returns:
            GitHubResult with issue list
        """
        args = ["issue", "list", "--repo", repo, "--state", state, "--limit", str(limit),
                "--json", "number,title,state,url,createdAt,updatedAt"]
        
        return self._run_command(args)
    
    def get_issue(self, repo: str, number: int) -> GitHubResult:
        """
        Get issue details.
        
        Args:
            repo: Repository (owner/name)
            number: Issue number
            
        Returns:
            GitHubResult with issue details
        """
        args = ["issue", "view", str(number), "--repo", repo,
                "--json", "number,title,state,body,url,createdAt,updatedAt,author"]
        
        return self._run_command(args)
    
    def create_issue(
        self,
        repo: str,
        title: str,
        body: str
    ) -> GitHubResult:
        """
        Create an issue.
        
        Args:
            repo: Repository (owner/name)
            title: Issue title
            body: Issue body/description
            
        Returns:
            GitHubResult with created issue
        """
        args = ["issue", "create", "--repo", repo, "--title", title, "--body", body]
        return self._run_command(args)
    
    # Workflow Operations
    
    def list_workflows(self, repo: str) -> GitHubResult:
        """
        List workflows.
        
        Args:
            repo: Repository (owner/name)
            
        Returns:
            GitHubResult with workflow list
        """
        args = ["workflow", "list", "--repo", repo, "--json", "id,name,state,path"]
        return self._run_command(args)
    
    def list_workflow_runs(
        self,
        repo: str,
        workflow: Optional[str] = None,
        limit: int = 20
    ) -> GitHubResult:
        """
        List workflow runs.
        
        Args:
            repo: Repository (owner/name)
            workflow: Workflow name or ID (all if None)
            limit: Maximum number of runs to return
            
        Returns:
            GitHubResult with workflow run list
        """
        args = ["run", "list", "--repo", repo, "--limit", str(limit),
                "--json", "databaseId,name,status,conclusion,createdAt,updatedAt"]
        
        if workflow:
            args.extend(["--workflow", workflow])
        
        return self._run_command(args)
    
    def get_workflow_run(self, repo: str, run_id: int) -> GitHubResult:
        """
        Get workflow run details.
        
        Args:
            repo: Repository (owner/name)
            run_id: Run ID
            
        Returns:
            GitHubResult with run details
        """
        args = ["run", "view", str(run_id), "--repo", repo,
                "--json", "databaseId,name,status,conclusion,createdAt,updatedAt,jobs"]
        
        return self._run_command(args)


# Convenience functions for common operations

def get_github_kit(gh_path: str = "gh", timeout: int = 30) -> GitHubKit:
    """
    Get a GitHubKit instance.
    
    Args:
        gh_path: Path to gh CLI executable
        timeout: Default timeout for operations
        
    Returns:
        GitHubKit instance
    """
    return GitHubKit(gh_path=gh_path, timeout=timeout)


__all__ = [
    'GitHubKit',
    'GitHubResult',
    'get_github_kit',
    'list_repos',
    'get_repo',
    'list_prs',
    'get_pr',
    'list_issues',
    'get_issue',
]


_DEFAULT_KIT: Optional[GitHubKit] = None


def _get_default_kit() -> GitHubKit:
    global _DEFAULT_KIT
    if _DEFAULT_KIT is None:
        _DEFAULT_KIT = GitHubKit()
    return _DEFAULT_KIT


def list_repos(owner: Optional[str] = None, limit: int = 30) -> GitHubResult:
    """Module-level wrapper for unified registry tooling."""
    return _get_default_kit().list_repos(owner=owner, limit=limit)


def get_repo(owner: str, repo: str) -> GitHubResult:
    """Module-level wrapper for unified registry tooling."""
    return _get_default_kit().get_repo(f"{owner}/{repo}")


def list_prs(owner: str, repo: str, state: str = "open", limit: int = 30) -> GitHubResult:
    """Module-level wrapper for unified registry tooling."""
    return _get_default_kit().list_prs(f"{owner}/{repo}", state=state, limit=limit)


def get_pr(owner: str, repo: str, pr_number: int) -> GitHubResult:
    """Module-level wrapper for unified registry tooling."""
    return _get_default_kit().get_pr(f"{owner}/{repo}", number=pr_number)


def list_issues(owner: str, repo: str, state: str = "open", limit: int = 30) -> GitHubResult:
    """Module-level wrapper for unified registry tooling."""
    return _get_default_kit().list_issues(f"{owner}/{repo}", state=state, limit=limit)


def get_issue(owner: str, repo: str, issue_number: int) -> GitHubResult:
    """Module-level wrapper for unified registry tooling."""
    return _get_default_kit().get_issue(f"{owner}/{repo}", number=issue_number)
