"""
GitHub CLI Tools for IPFS Accelerate MCP Server

This module provides MCP tools for interacting with GitHub via the GitHub CLI,
including workflow management, runner management, and cache operations.
These tools use the GitHub CLI wrapper with P2P caching and minimize API calls.
"""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Protocol
else:
    Protocol = object

logger = logging.getLogger("ipfs_accelerate_mcp.tools.github_tools")


def register_tools(mcp: Any) -> None:
    """
    Register GitHub CLI tools with the MCP server.
    
    Args:
        mcp: MCP server instance
    """
    logger.debug("Registering GitHub CLI tools")
    
    @mcp.tool()
    def gh_list_runners(
        repo: Optional[str] = None,
        org: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List GitHub Actions self-hosted runners.
        
        Lists self-hosted runners for a repository or organization.
        Results are cached via P2P network to minimize GitHub API calls.
        
        Args:
            repo: Repository in format "owner/repo" (for repo-level runners)
            org: Organization name (for org-level runners)
            
        Returns:
            Dictionary containing:
            - runners: List of runner dictionaries with status, labels, etc.
            - cached: Whether result came from cache
            - error: Error message if failed
            
        Example:
            >>> gh_list_runners(repo="owner/repo")
            >>> gh_list_runners(org="my-org")
        """
        try:
            from ipfs_accelerate_py.github_cli import GitHubCLI, RunnerManager
            
            # Initialize with P2P caching enabled
            gh = GitHubCLI(enable_cache=True)
            runner_mgr = RunnerManager(gh_cli=gh)
            
            # List runners (uses cache automatically)
            runners = runner_mgr.list_runners(repo=repo, org=org, use_cache=True)
            
            return {
                'success': True,
                'runners': runners,
                'count': len(runners),
                'cached': gh.cache is not None,
                'repo': repo,
                'org': org
            }
        except Exception as e:
            logger.error(f"Error listing runners: {e}")
            return {
                'success': False,
                'error': str(e),
                'runners': []
            }
    
    @mcp.tool()
    def gh_create_workflow_queues(
        owner: Optional[str] = None,
        since_days: int = 1,
        filter_by_arch: bool = True
    ) -> Dict[str, Any]:
        """
        Create workflow queues for repositories with recent activity.
        
        Finds repositories with recent updates and creates queues of running
        or failed workflows. Filters by system architecture compatibility.
        Uses P2P cache to minimize GitHub API calls.
        
        Args:
            owner: Repository owner (user or org). If None, uses authenticated user.
            since_days: Only include repos/workflows from the last N days (default: 1)
            filter_by_arch: Filter workflows by architecture compatibility (default: True)
            
        Returns:
            Dictionary containing:
            - queues: Dict mapping repo names to lists of workflow runs
            - total_workflows: Total number of workflows across all repos
            - cached: Whether result came from cache
            - system_arch: Detected system architecture
            - error: Error message if failed
            
        Example:
            >>> gh_create_workflow_queues(since_days=1)
            >>> gh_create_workflow_queues(owner="my-org", since_days=7)
        """
        try:
            from ipfs_accelerate_py.github_cli import GitHubCLI, WorkflowManager, RunnerManager
            
            # Initialize with P2P caching enabled
            gh = GitHubCLI(enable_cache=True)
            workflow_mgr = WorkflowManager(gh_cli=gh)
            runner_mgr = RunnerManager(gh_cli=gh)
            
            # Detect system architecture for filtering
            system_arch = runner_mgr.get_system_architecture() if filter_by_arch else None
            
            # Create workflow queues (uses cache automatically)
            queues = workflow_mgr.create_workflow_queues(
                owner=owner,
                since_days=since_days,
                system_arch=system_arch,
                filter_by_arch=filter_by_arch
            )
            
            # Count total workflows
            total_workflows = sum(len(workflows) for workflows in queues.values())
            
            return {
                'success': True,
                'queues': queues,
                'total_repos': len(queues),
                'total_workflows': total_workflows,
                'cached': gh.cache is not None,
                'system_arch': system_arch,
                'owner': owner,
                'since_days': since_days
            }
        except Exception as e:
            logger.error(f"Error creating workflow queues: {e}")
            return {
                'success': False,
                'error': str(e),
                'queues': {}
            }
    
    @mcp.tool()
    def gh_get_cache_stats() -> Dict[str, Any]:
        """
        Get GitHub API cache statistics.
        
        Returns cache performance metrics including hit rate, total entries,
        and P2P sharing status. This helps monitor API call reduction.
        
        Returns:
            Dictionary containing:
            - total_entries: Total number of cached entries
            - cache_size: Size of cache in MB
            - hit_rate: Cache hit rate percentage
            - p2p_enabled: Whether P2P cache sharing is enabled
            - p2p_peers: Number of P2P peers (if enabled)
            - error: Error message if failed
            
        Example:
            >>> gh_get_cache_stats()
        """
        try:
            from ipfs_accelerate_py.github_cli.cache import get_global_cache
            
            cache = get_global_cache()
            stats = cache.get_stats()
            
            return {
                'success': True,
                **stats
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    @mcp.tool()
    def gh_get_auth_status() -> Dict[str, Any]:
        """
        Get GitHub authentication status.
        
        Checks if GitHub CLI is authenticated and returns user information.
        This is the same data shown in the dashboard's "User Information" section.
        
        Returns:
            Dictionary containing:
            - authenticated: Whether user is authenticated
            - username: GitHub username (if authenticated)
            - token_type: Type of authentication token
            - error: Error message if authentication failed
            
        Example:
            >>> gh_get_auth_status()
        """
        try:
            from ipfs_accelerate_py.mcp.tools.dashboard_data import get_user_info
            return get_user_info()
        except Exception as e:
            logger.error(f"Error getting auth status: {e}")
            return {
                'authenticated': False,
                'error': str(e)
            }
    
    @mcp.tool()
    def gh_list_workflow_runs(
        repo: str,
        status: Optional[str] = None,
        limit: int = 20,
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List workflow runs for a repository.
        
        Lists workflow runs with optional filtering by status and branch.
        Results are cached via P2P network to minimize GitHub API calls.
        
        Args:
            repo: Repository in format "owner/repo"
            status: Filter by status (queued, in_progress, completed)
            limit: Maximum number of runs to return (default: 20)
            branch: Filter by branch name
            
        Returns:
            Dictionary containing:
            - runs: List of workflow run dictionaries
            - cached: Whether result came from cache
            - error: Error message if failed
            
        Example:
            >>> gh_list_workflow_runs(repo="owner/repo", status="in_progress")
            >>> gh_list_workflow_runs(repo="owner/repo", branch="main", limit=10)
        """
        try:
            from ipfs_accelerate_py.github_cli import GitHubCLI, WorkflowManager
            
            # Initialize with P2P caching enabled
            gh = GitHubCLI(enable_cache=True)
            workflow_mgr = WorkflowManager(gh_cli=gh)
            
            # List workflow runs (uses cache automatically)
            runs = workflow_mgr.list_workflow_runs(
                repo=repo,
                status=status,
                limit=limit,
                branch=branch,
                use_cache=True
            )
            
            return {
                'success': True,
                'runs': runs,
                'count': len(runs),
                'cached': gh.cache is not None,
                'repo': repo,
                'status': status,
                'branch': branch
            }
        except Exception as e:
            logger.error(f"Error listing workflow runs: {e}")
            return {
                'success': False,
                'error': str(e),
                'runs': []
            }
    
    @mcp.tool()
    def gh_get_runner_labels() -> Dict[str, Any]:
        """
        Get runner labels for the current system.
        
        Returns the labels that would be assigned to a self-hosted runner
        on this system, including architecture, OS, and hardware capabilities.
        
        Returns:
            Dictionary containing:
            - labels: List of runner labels
            - system_arch: Detected system architecture
            - error: Error message if failed
            
        Example:
            >>> gh_get_runner_labels()
            # Returns: {'labels': ['self-hosted', 'linux', 'x64', 'docker'], ...}
        """
        try:
            from ipfs_accelerate_py.github_cli import RunnerManager
            
            runner_mgr = RunnerManager()
            labels = runner_mgr.get_runner_labels()
            system_arch = runner_mgr.get_system_architecture()
            
            return {
                'success': True,
                'labels': labels.split(','),
                'labels_string': labels,
                'system_arch': system_arch
            }
        except Exception as e:
            logger.error(f"Error getting runner labels: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    logger.debug("Registered 7 GitHub CLI tools")
