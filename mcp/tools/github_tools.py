"""
GitHub CLI Tools for MCP Server

This module provides MCP tools for GitHub CLI integration,
including workflow queue management and runner provisioning.
"""

import logging
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger("ipfs_accelerate_mcp.tools.github")

# Try imports with fallbacks
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from fastmcp import FastMCP
    except ImportError:
        from mcp.mock_mcp import FastMCP

# Import GitHub operations
try:
    from ...shared import SharedCore, GitHubOperations
    shared_core = SharedCore()
    github_ops = GitHubOperations(shared_core)
    HAVE_GITHUB = True
except ImportError as e:
    logger.warning(f"GitHub operations not available: {e}")
    HAVE_GITHUB = False
    github_ops = None


def register_github_tools(mcp: FastMCP) -> None:
    """Register GitHub CLI tools with the MCP server."""
    logger.info("Registering GitHub CLI tools")
    
    if not HAVE_GITHUB:
        logger.warning("GitHub operations not available, skipping registration")
        return
    
    @mcp.tool()
    def gh_auth_status() -> Dict[str, Any]:
        """
        Check GitHub CLI authentication status
        
        Returns:
            Authentication status and details
        """
        try:
            result = github_ops.get_auth_status()
            result["tool"] = "gh_auth_status"
            return result
        except Exception as e:
            logger.error(f"Error in gh_auth_status: {e}")
            return {
                "error": str(e),
                "tool": "gh_auth_status",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_list_repos(
        owner: Optional[str] = None,
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        List GitHub repositories
        
        Args:
            owner: Repository owner (user or organization)
            limit: Maximum number of repositories to return
            
        Returns:
            List of repositories
        """
        try:
            result = github_ops.list_repos(owner=owner, limit=limit)
            result["tool"] = "gh_list_repos"
            return result
        except Exception as e:
            logger.error(f"Error in gh_list_repos: {e}")
            return {
                "error": str(e),
                "tool": "gh_list_repos",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_list_workflow_runs(
        repo: str,
        status: Optional[str] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        List workflow runs for a repository
        
        Args:
            repo: Repository in format "owner/repo"
            status: Filter by status (queued, in_progress, completed)
            limit: Maximum number of runs to return
            
        Returns:
            List of workflow runs
        """
        try:
            result = github_ops.list_workflow_runs(repo, status=status, limit=limit)
            result["tool"] = "gh_list_workflow_runs"
            return result
        except Exception as e:
            logger.error(f"Error in gh_list_workflow_runs: {e}")
            return {
                "error": str(e),
                "tool": "gh_list_workflow_runs",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_get_workflow_run(
        repo: str,
        run_id: str
    ) -> Dict[str, Any]:
        """
        Get details of a specific workflow run
        
        Args:
            repo: Repository in format "owner/repo"
            run_id: Workflow run ID
            
        Returns:
            Workflow run details
        """
        try:
            result = github_ops.get_workflow_run(repo, run_id)
            result["tool"] = "gh_get_workflow_run"
            return result
        except Exception as e:
            logger.error(f"Error in gh_get_workflow_run: {e}")
            return {
                "error": str(e),
                "tool": "gh_get_workflow_run",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_create_workflow_queues(
        owner: Optional[str] = None,
        since_days: int = 1
    ) -> Dict[str, Any]:
        """
        Create workflow queues for repositories with recent activity
        
        This tool finds all repositories with recent updates and creates
        queues of running or failed workflows for each repository.
        
        Args:
            owner: Repository owner (user or organization)
            since_days: Only include repos/workflows from the last N days
            
        Returns:
            Workflow queues organized by repository
        """
        try:
            result = github_ops.create_workflow_queues(owner=owner, since_days=since_days)
            result["tool"] = "gh_create_workflow_queues"
            return result
        except Exception as e:
            logger.error(f"Error in gh_create_workflow_queues: {e}")
            return {
                "error": str(e),
                "tool": "gh_create_workflow_queues",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_list_runners(
        repo: Optional[str] = None,
        org: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List self-hosted runners
        
        Args:
            repo: Repository in format "owner/repo" (for repo-level runners)
            org: Organization name (for org-level runners)
            
        Returns:
            List of self-hosted runners
        """
        try:
            result = github_ops.list_runners(repo=repo, org=org)
            result["tool"] = "gh_list_runners"
            return result
        except Exception as e:
            logger.error(f"Error in gh_list_runners: {e}")
            return {
                "error": str(e),
                "tool": "gh_list_runners",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_provision_runners(
        owner: Optional[str] = None,
        since_days: int = 1,
        max_runners: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Provision self-hosted runners based on workflow queues
        
        This tool analyzes workflow queues and provisions runners based on
        system capacity and workflow load. It automatically derives tokens
        from the gh CLI and attaches them to self-hosted runners.
        
        Args:
            owner: Repository owner (user or organization)
            since_days: Only include repos/workflows from the last N days
            max_runners: Maximum runners to provision (defaults to system cores)
            
        Returns:
            Provisioning status for each repository
        """
        try:
            result = github_ops.provision_runners(
                owner=owner,
                since_days=since_days,
                max_runners=max_runners
            )
            result["tool"] = "gh_provision_runners"
            return result
        except Exception as e:
            logger.error(f"Error in gh_provision_runners: {e}")
            return {
                "error": str(e),
                "tool": "gh_provision_runners",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_get_cache_stats() -> Dict[str, Any]:
        """
        Get GitHub API cache statistics
        
        Returns cache performance metrics including:
        - Hit/miss rates
        - Total cached entries
        - P2P peer status
        - IPLD/multiformats data tracking
        - API calls saved
        
        Returns:
            Cache statistics and performance metrics
        """
        try:
            from ...ipfs_accelerate_py.github_cli import get_global_cache
            cache = get_global_cache()
            
            stats = cache.get_stats()
            
            # Add P2P peer information if available
            if hasattr(cache, '_p2p_connected_peers'):
                stats['p2p_peers'] = {
                    'connected': len(cache._p2p_connected_peers),
                    'peers': list(cache._p2p_connected_peers.keys())
                }
            
            # Add IPLD/multiformats information
            stats['content_addressing'] = {
                'enabled': cache.__class__.__name__ == 'GitHubAPICache',
                'multiformats_available': hasattr(cache, '_compute_validation_hash')
            }
            
            stats['tool'] = 'gh_get_cache_stats'
            stats['timestamp'] = time.time()
            return stats
        except Exception as e:
            logger.error(f"Error in gh_get_cache_stats: {e}")
            return {
                "error": str(e),
                "tool": "gh_get_cache_stats",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_get_workflow_details(
        repo: str,
        run_id: str,
        include_jobs: bool = True
    ) -> Dict[str, Any]:
        """
        Get detailed information about a workflow run including jobs
        
        Args:
            repo: Repository in format "owner/repo"
            run_id: Workflow run ID
            include_jobs: Whether to include job details
            
        Returns:
            Detailed workflow run information with job status
        """
        try:
            # Get workflow run details
            run_result = github_ops.get_workflow_run(repo, run_id)
            
            if not run_result.get('success'):
                return run_result
            
            result = {
                'run': run_result.get('run'),
                'repo': repo,
                'run_id': run_id,
                'tool': 'gh_get_workflow_details',
                'timestamp': time.time()
            }
            
            # Get job details if requested
            if include_jobs and github_ops.gh_cli:
                try:
                    jobs_cmd = github_ops.gh_cli._run_command([
                        'api',
                        f'repos/{repo}/actions/runs/{run_id}/jobs',
                        '--jq', '.jobs[]'
                    ])
                    
                    if jobs_cmd.get('success'):
                        import json
                        jobs_data = jobs_cmd.get('stdout', '')
                        if jobs_data:
                            # Parse multiple JSON objects
                            jobs = []
                            for line in jobs_data.strip().split('\n'):
                                if line.strip():
                                    try:
                                        jobs.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        pass
                            result['jobs'] = jobs
                            result['jobs_summary'] = {
                                'total': len(jobs),
                                'in_progress': sum(1 for j in jobs if j.get('status') == 'in_progress'),
                                'completed': sum(1 for j in jobs if j.get('status') == 'completed'),
                                'failed': sum(1 for j in jobs if j.get('conclusion') == 'failure'),
                                'success': sum(1 for j in jobs if j.get('conclusion') == 'success')
                            }
                except Exception as e:
                    logger.warning(f"Failed to fetch job details: {e}")
                    result['jobs_error'] = str(e)
            
            result['success'] = True
            return result
        except Exception as e:
            logger.error(f"Error in gh_get_workflow_details: {e}")
            return {
                "error": str(e),
                "tool": "gh_get_workflow_details",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_invalidate_cache(
        pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Invalidate GitHub API cache entries
        
        Args:
            pattern: Optional pattern to match cache keys (e.g., 'list_repos', 'workflow')
                    If None, invalidates all cache entries
            
        Returns:
            Number of cache entries invalidated
        """
        try:
            from ...ipfs_accelerate_py.github_cli import get_global_cache
            cache = get_global_cache()
            
            if pattern:
                invalidated = cache.invalidate_pattern(pattern)
            else:
                invalidated = cache.clear()
            
            return {
                'invalidated': invalidated,
                'pattern': pattern or 'all',
                'tool': 'gh_invalidate_cache',
                'timestamp': time.time(),
                'success': True
            }
        except Exception as e:
            logger.error(f"Error in gh_invalidate_cache: {e}")
            return {
                "error": str(e),
                "tool": "gh_invalidate_cache",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_get_rate_limit() -> Dict[str, Any]:
        """
        Get current GitHub API rate limit status
        
        Returns rate limit information including:
        - Remaining requests
        - Total limit
        - Reset time
        - Resources breakdown
        
        Returns:
            Rate limit status
        """
        try:
            if not github_ops.gh_cli:
                return {"error": "GitHub CLI not available", "success": False}
            
            result = github_ops.gh_cli._run_command([
                'api',
                'rate_limit'
            ])
            
            if result.get('success'):
                import json
                rate_limit = json.loads(result.get('stdout', '{}'))
                
                return {
                    'rate_limit': rate_limit,
                    'tool': 'gh_get_rate_limit',
                    'timestamp': time.time(),
                    'success': True
                }
            else:
                return {
                    'error': result.get('stderr', 'Unknown error'),
                    'tool': 'gh_get_rate_limit',
                    'timestamp': time.time(),
                    'success': False
                }
        except Exception as e:
            logger.error(f"Error in gh_get_rate_limit: {e}")
            return {
                "error": str(e),
                "tool": "gh_get_rate_limit",
                "timestamp": time.time()
            }
    
    logger.info("GitHub CLI tools registered successfully")
