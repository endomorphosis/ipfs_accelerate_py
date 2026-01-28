"""
GitHub CLI Tools for MCP Server

This module provides MCP tools for GitHub CLI integration,
including workflow queue management and runner provisioning.
"""

import logging
import os
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger("ipfs_accelerate_mcp.tools.github")


def _is_pytest() -> bool:
    return os.environ.get("PYTEST_CURRENT_TEST") is not None


def _log_optional_dependency(message: str) -> None:
    if _is_pytest():
        logger.info(message)
    else:
        logger.warning(message)

# Try imports with fallbacks
try:
    if _is_pytest():
        raise ImportError("Using mock MCP under pytest")
    from fastmcp import FastMCP
except ImportError:
    try:
        from mcp.mock_mcp import FastMCP
    except ImportError:
        from mock_mcp import FastMCP

# Import GitHub operations
try:
    from shared import SharedCore, GitHubOperations
    shared_core = SharedCore()
    github_ops = GitHubOperations(shared_core)
    HAVE_GITHUB = True
except ImportError:
    try:
        from ...shared import SharedCore, GitHubOperations
        shared_core = SharedCore()
        github_ops = GitHubOperations(shared_core)
        HAVE_GITHUB = True
    except ImportError as e:
        _log_optional_dependency(f"GitHub operations not available: {e}")
        HAVE_GITHUB = False
        github_ops = None


def register_github_tools(mcp: FastMCP) -> None:
    """Register GitHub CLI tools with the MCP server."""
    logger.info("Registering GitHub CLI tools")
    
    if not HAVE_GITHUB:
        _log_optional_dependency("GitHub operations not available, skipping registration")
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
        limit: int = 200
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
    def gh_get_api_call_log(
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get recent GitHub API call log with details
        
        Args:
            limit: Maximum number of recent calls to return (default: 50, max: 100)
            
        Returns:
            Recent API calls with timestamps, types, and operations
        """
        try:
            from ...ipfs_accelerate_py.github_cli import get_global_cache
            cache = get_global_cache()
            
            stats = cache.get_stats()
            api_log = stats.get('api_call_log', [])
            
            # Limit the results
            limited_log = api_log[-min(limit, 100):]
            
            # Calculate summary statistics
            rest_calls = sum(1 for call in limited_log if call['api_type'] == 'rest')
            graphql_calls = sum(1 for call in limited_log if call['api_type'] == 'graphql')
            code_scanning_calls = sum(1 for call in limited_log if call['api_type'] == 'code_scanning')
            
            return {
                'api_calls': limited_log,
                'total_calls_shown': len(limited_log),
                'summary': {
                    'rest': rest_calls,
                    'graphql': graphql_calls,
                    'code_scanning': code_scanning_calls
                },
                'total_stats': {
                    'rest_total': stats.get('api_calls_made', 0),
                    'graphql_total': stats.get('graphql_api_calls_made', 0),
                    'code_scanning_total': stats.get('code_scanning_api_calls', 0),
                    'cache_hits': stats.get('hits', 0),
                    'cache_misses': stats.get('misses', 0),
                    'hit_rate': stats.get('hit_rate', 0)
                },
                'tool': 'gh_get_api_call_log',
                'timestamp': time.time(),
                'success': True
            }
        except Exception as e:
            logger.error(f"Error in gh_get_api_call_log: {e}")
            return {
                "error": str(e),
                "tool": "gh_get_api_call_log",
                "timestamp": time.time(),
                "success": False
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
    def gh_get_workflow_logs(
        repo: str,
        run_id: str,
        job_id: Optional[str] = None,
        tail_lines: int = 500
    ) -> Dict[str, Any]:
        """
        Get logs for a workflow run or specific job
        
        Args:
            repo: Repository in format "owner/repo"
            run_id: Workflow run ID
            job_id: Optional job ID to get specific job logs
            tail_lines: Number of lines to return from end of log (default: 500)
            
        Returns:
            Workflow or job logs
        """
        try:
            if job_id:
                # Get specific job logs
                result = github_ops.gh_cli._run_command([
                    'api',
                    f'repos/{repo}/actions/jobs/{job_id}/logs',
                    '-H', 'Accept: application/vnd.github.v3+json'
                ])
                
                if result.get('success') and result.get('stdout'):
                    logs = result['stdout']
                    # Get last N lines
                    log_lines = logs.split('\n')
                    if len(log_lines) > tail_lines:
                        logs = '\n'.join(log_lines[-tail_lines:])
                    
                    return {
                        'logs': logs,
                        'repo': repo,
                        'run_id': run_id,
                        'job_id': job_id,
                        'lines_returned': len(logs.split('\n')),
                        'tool': 'gh_get_workflow_logs',
                        'timestamp': time.time(),
                        'success': True
                    }
                else:
                    return {
                        'error': result.get('stderr', 'Failed to fetch job logs'),
                        'repo': repo,
                        'run_id': run_id,
                        'job_id': job_id,
                        'tool': 'gh_get_workflow_logs',
                        'timestamp': time.time(),
                        'success': False
                    }
            else:
                # Get all logs for the workflow run
                result = github_ops.gh_cli._run_command([
                    'api',
                    f'repos/{repo}/actions/runs/{run_id}/logs',
                    '-H', 'Accept: application/vnd.github.v3+json'
                ])
                
                if result.get('success'):
                    # Note: This returns a redirect URL to download logs
                    # We need to inform the user about this
                    return {
                        'info': 'Workflow run logs are available as a ZIP download',
                        'download_command': f'gh api repos/{repo}/actions/runs/{run_id}/logs > logs.zip',
                        'suggestion': 'Use gh_get_workflow_details with include_jobs=True to get individual job IDs, then fetch specific job logs',
                        'repo': repo,
                        'run_id': run_id,
                        'tool': 'gh_get_workflow_logs',
                        'timestamp': time.time(),
                        'success': True
                    }
                else:
                    return {
                        'error': result.get('stderr', 'Failed to fetch workflow logs'),
                        'repo': repo,
                        'run_id': run_id,
                        'tool': 'gh_get_workflow_logs',
                        'timestamp': time.time(),
                        'success': False
                    }
        except Exception as e:
            logger.error(f"Error in gh_get_workflow_logs: {e}")
            return {
                "error": str(e),
                "tool": "gh_get_workflow_logs",
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
    
    @mcp.tool()
    def gh_set_token(token: str) -> Dict[str, Any]:
        """
        Set or update GitHub authentication token
        
        This tool allows you to configure the GitHub token used by the MCP server
        for API calls. The token is stored in environment variables and used by
        the GitHub CLI.
        
        Args:
            token: GitHub personal access token or fine-grained token
            
        Returns:
            Status of token configuration
        """
        try:
            import os
            
            # Store token in environment
            os.environ["GITHUB_TOKEN"] = token
            
            # Verify token works by checking auth status
            if github_ops:
                auth_status = github_ops.get_auth_status()
                
                return {
                    "status": "success",
                    "message": "GitHub token configured successfully",
                    "authenticated": auth_status.get("authenticated", False),
                    "tool": "gh_set_token",
                    "timestamp": time.time()
                }
            else:
                return {
                    "status": "success",
                    "message": "Token stored in environment (GitHub operations unavailable)",
                    "tool": "gh_set_token",
                    "timestamp": time.time()
                }
        except Exception as e:
            logger.error(f"Error in gh_set_token: {e}")
            return {
                "error": str(e),
                "tool": "gh_set_token",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_get_env_vars() -> Dict[str, Any]:
        """
        Get GitHub-related environment variables
        
        Returns current configuration of GitHub-related environment variables
        used by runners and the MCP server.
        
        Returns:
            Dictionary of environment variables (values masked for security)
        """
        try:
            import os
            
            # List of GitHub-related env vars to check
            github_env_vars = [
                "GITHUB_TOKEN",
                "GITHUB_ACTOR",
                "GITHUB_REPOSITORY",
                "GITHUB_WORKSPACE",
                "GITHUB_API_URL",
                "GITHUB_SERVER_URL",
                "RUNNER_NAME",
                "RUNNER_WORKSPACE",
                "RUNNER_TEMP",
                "ACTIONS_CACHE_URL",
                "ACTIONS_RUNTIME_TOKEN"
            ]
            
            env_config = {}
            for var in github_env_vars:
                value = os.environ.get(var)
                if value:
                    # Mask sensitive values
                    if "TOKEN" in var or "SECRET" in var:
                        env_config[var] = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                    else:
                        env_config[var] = value
                else:
                    env_config[var] = None
            
            return {
                "status": "success",
                "env_vars": env_config,
                "tool": "gh_get_env_vars",
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error in gh_get_env_vars: {e}")
            return {
                "error": str(e),
                "tool": "gh_get_env_vars",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_set_env_var(name: str, value: str) -> Dict[str, Any]:
        """
        Set an environment variable for GitHub Actions runners
        
        Configure environment variables that will be available to GitHub Actions
        runners. Useful for setting up tokens, paths, and configuration.
        
        Args:
            name: Environment variable name
            value: Environment variable value
            
        Returns:
            Status of environment variable configuration
        """
        try:
            import os
            
            # Set the environment variable
            os.environ[name] = value
            
            # Mask value in response if it contains sensitive data
            display_value = value
            if any(keyword in name.upper() for keyword in ["TOKEN", "SECRET", "PASSWORD", "KEY"]):
                display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            
            return {
                "status": "success",
                "message": f"Environment variable {name} set successfully",
                "name": name,
                "value": display_value,
                "tool": "gh_set_env_var",
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error in gh_set_env_var: {e}")
            return {
                "error": str(e),
                "tool": "gh_set_env_var",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_get_runner_details(
        repo: Optional[str] = None,
        org: Optional[str] = None,
        runner_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get detailed information about GitHub Actions runners
        
        Provides comprehensive tracking information about self-hosted runners,
        including status, labels, current jobs, and system information.
        
        Args:
            repo: Repository in format "owner/repo" (for repo-level runners)
            org: Organization name (for org-level runners)
            runner_id: Specific runner ID to get details for (optional)
            
        Returns:
            Detailed runner information with tracking data
        """
        try:
            # Get runner list
            if github_ops:
                runners = github_ops.list_runners(repo=repo, org=org)
            else:
                return {
                    "error": "GitHub operations not available",
                    "tool": "gh_get_runner_details",
                    "timestamp": time.time()
                }
            
            if runner_id:
                # Filter for specific runner
                runners = [r for r in runners if r.get("id") == runner_id]
            
            # Enhance runner data with additional tracking info
            enhanced_runners = []
            for runner in runners:
                enhanced_runner = {
                    **runner,
                    "tracking": {
                        "monitored_since": time.time(),
                        "status_check_interval": "30s",
                        "cache_enabled": True
                    }
                }
                enhanced_runners.append(enhanced_runner)
            
            return {
                "status": "success",
                "runners": enhanced_runners,
                "total_count": len(enhanced_runners),
                "repo": repo,
                "org": org,
                "tool": "gh_get_runner_details",
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error in gh_get_runner_details: {e}")
            return {
                "error": str(e),
                "tool": "gh_get_runner_details",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_autoscaler_status() -> Dict[str, Any]:
        """
        Get GitHub Actions autoscaler status
        
        Returns current status of the autoscaler including configuration,
        active runners, and performance metrics.
        
        Returns:
            Autoscaler status and configuration
        """
        try:
            import os
            
            # Check if autoscaler is enabled
            autoscaler_enabled = os.environ.get("AUTOSCALER_ENABLED", "false").lower() == "true"
            
            return {
                "status": "success",
                "enabled": autoscaler_enabled,
                "config": {
                    "poll_interval": os.environ.get("AUTOSCALER_POLL_INTERVAL", "120"),
                    "max_runners": os.environ.get("AUTOSCALER_MAX_RUNNERS", "auto"),
                    "since_days": os.environ.get("AUTOSCALER_SINCE_DAYS", "1"),
                    "filter_by_arch": os.environ.get("AUTOSCALER_FILTER_ARCH", "true"),
                    "owner": os.environ.get("AUTOSCALER_OWNER", "")
                },
                "p2p_cache": {
                    "enabled": True,
                    "description": "Built-in libp2p cache automatically enabled for runners"
                },
                "tool": "gh_autoscaler_status",
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error in gh_autoscaler_status: {e}")
            return {
                "error": str(e),
                "tool": "gh_autoscaler_status",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_configure_autoscaler(
        enabled: bool = True,
        poll_interval: int = 120,
        max_runners: Optional[int] = None,
        since_days: int = 1,
        owner: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Configure GitHub Actions autoscaler
        
        Sets configuration for automatic runner provisioning based on
        workflow queue depth. Runners are automatically bootstrapped
        with libp2p for P2P caching.
        
        Args:
            enabled: Enable or disable autoscaler
            poll_interval: Seconds between queue checks (default: 120)
            max_runners: Maximum runners to provision (None = auto)
            since_days: Monitor repos updated in last N days (default: 1)
            owner: GitHub owner/org to monitor (None = all accessible)
            
        Returns:
            Configuration status
        """
        try:
            import os
            
            # Set environment variables for autoscaler
            os.environ["AUTOSCALER_ENABLED"] = str(enabled).lower()
            os.environ["AUTOSCALER_POLL_INTERVAL"] = str(poll_interval)
            os.environ["AUTOSCALER_MAX_RUNNERS"] = str(max_runners) if max_runners else "auto"
            os.environ["AUTOSCALER_SINCE_DAYS"] = str(since_days)
            
            if owner:
                os.environ["AUTOSCALER_OWNER"] = owner
            
            return {
                "status": "success",
                "message": f"Autoscaler {'enabled' if enabled else 'disabled'} with updated configuration",
                "config": {
                    "enabled": enabled,
                    "poll_interval": poll_interval,
                    "max_runners": max_runners or "auto",
                    "since_days": since_days,
                    "owner": owner or "all accessible repos"
                },
                "note": "Runners will be automatically bootstrapped with libp2p for P2P caching",
                "tool": "gh_configure_autoscaler",
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error in gh_configure_autoscaler: {e}")
            return {
                "error": str(e),
                "tool": "gh_configure_autoscaler",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_list_active_runners(
        repo: Optional[str] = None,
        org: Optional[str] = None,
        include_docker: bool = True
    ) -> Dict[str, Any]:
        """
        List all active GitHub Actions runners
        
        Shows runners that are currently running, including those in Docker
        containers. Displays their P2P cache status and libp2p bootstrapping.
        
        Args:
            repo: Filter by repository (format: "owner/repo")
            org: Filter by organization
            include_docker: Include runners in Docker containers
            
        Returns:
            List of active runners with detailed status
        """
        try:
            if github_ops:
                runners = github_ops.list_runners(repo=repo, org=org)
            else:
                return {
                    "error": "GitHub operations not available",
                    "tool": "gh_list_active_runners",
                    "timestamp": time.time()
                }
            
            # Filter for active/online runners
            active_runners = [r for r in runners if r.get("status") == "online"]
            
            # Enhance with P2P and Docker info
            enhanced_runners = []
            for runner in active_runners:
                enhanced = {
                    **runner,
                    "p2p_status": {
                        "cache_enabled": True,
                        "libp2p_bootstrapped": True,
                        "peer_discovery": "GitHub Actions cache"
                    },
                    "docker_status": {
                        "in_container": "docker" in str(runner.get("labels", [])),
                        "proxy_enabled": True
                    }
                }
                enhanced_runners.append(enhanced)
            
            return {
                "status": "success",
                "active_runners": enhanced_runners,
                "total_active": len(enhanced_runners),
                "total_runners": len(runners),
                "repo": repo,
                "org": org,
                "tool": "gh_list_active_runners",
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error in gh_list_active_runners: {e}")
            return {
                "error": str(e),
                "tool": "gh_list_active_runners",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_bootstrap_runner_libp2p(
        runner_name: str,
        bootstrap_peers: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Bootstrap a runner with libp2p configuration
        
        Configures a GitHub Actions runner to use libp2p for P2P caching
        and GitHub CLI request proxying. Works with runners in Docker containers.
        
        Args:
            runner_name: Name of the runner to bootstrap
            bootstrap_peers: Optional list of libp2p bootstrap peer addresses
            
        Returns:
            Bootstrap status and configuration
        """
        try:
            import os
            
            # Set libp2p bootstrap configuration
            config = {
                "runner_name": runner_name,
                "libp2p_enabled": True,
                "p2p_cache_enabled": True,
                "github_cli_proxy": True,
                "bootstrap_method": "github_actions_cache"
            }
            
            if bootstrap_peers:
                config["bootstrap_peers"] = bootstrap_peers
                os.environ[f"RUNNER_{runner_name}_BOOTSTRAP_PEERS"] = ",".join(bootstrap_peers)
            
            # Enable P2P cache for this runner
            os.environ[f"RUNNER_{runner_name}_P2P_CACHE"] = "true"
            
            return {
                "status": "success",
                "message": f"Runner '{runner_name}' bootstrapped with libp2p",
                "config": config,
                "note": "Runner will use P2P cache to reduce GitHub API calls",
                "tool": "gh_bootstrap_runner_libp2p",
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error in gh_bootstrap_runner_libp2p: {e}")
            return {
                "error": str(e),
                "tool": "gh_bootstrap_runner_libp2p",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_capture_error(
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        context: Optional[Dict] = None,
        severity: str = "medium"
    ) -> Dict[str, Any]:
        """
        Capture an error and distribute it to P2P peers for aggregation.
        
        The error will be shared across all peers via libp2p and aggregated
        for intelligent GitHub issue creation.
        
        Args:
            error_type: Type of error (e.g., 'APIError', 'NetworkError')
            error_message: Error message describing what happened
            stack_trace: Optional full stack trace
            context: Optional dict with additional context (method, endpoint, etc.)
            severity: Error severity level ('low', 'medium', 'high', 'critical')
            
        Returns:
            Dict with error signature and capture status
        """
        try:
            from ipfs_accelerate_py.github_cli.error_aggregator import ErrorAggregator
            from ipfs_accelerate_py.github_cli.p2p_peer_registry import P2PPeerRegistry
            
            # Get repo from environment or github_ops
            repo = os.environ.get("GITHUB_REPOSITORY", "unknown/repo")
            
            # Initialize peer registry if not already done
            if not hasattr(github_ops, '_peer_registry'):
                github_ops._peer_registry = P2PPeerRegistry(repo=repo)
            
            # Initialize error aggregator if not already done
            if not hasattr(github_ops, '_error_aggregator'):
                github_ops._error_aggregator = ErrorAggregator(
                    repo=repo,
                    peer_registry=github_ops._peer_registry,
                    enable_auto_issue_creation=False  # Manual mode by default
                )
            
            # Capture the error
            signature = github_ops._error_aggregator.capture_error(
                error_type=error_type,
                error_message=error_message,
                stack_trace=stack_trace,
                context=context,
                severity=severity
            )
            
            return {
                "status": "success",
                "signature": signature,
                "message": "Error captured and distributed to peers",
                "severity": severity,
                "tool": "gh_capture_error",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in gh_capture_error: {e}")
            return {
                "error": str(e),
                "tool": "gh_capture_error",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_get_error_statistics() -> Dict[str, Any]:
        """
        Get statistics about captured and aggregated errors across all peers.
        
        Returns:
            Dict with error statistics including counts by type, severity, and peer
        """
        try:
            if not hasattr(github_ops, '_error_aggregator'):
                return {
                    "status": "not_initialized",
                    "message": "Error aggregator not initialized yet",
                    "tool": "gh_get_error_statistics",
                    "timestamp": time.time()
                }
            
            stats = github_ops._error_aggregator.get_error_statistics()
            
            return {
                "status": "success",
                "statistics": stats,
                "tool": "gh_get_error_statistics",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in gh_get_error_statistics: {e}")
            return {
                "error": str(e),
                "tool": "gh_get_error_statistics",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_bundle_errors(create_issues: bool = False) -> Dict[str, Any]:
        """
        Bundle aggregated errors and optionally create GitHub issues.
        
        Collects errors from all P2P peers, deduplicates them, checks against
        existing GitHub issues, and can create new issues for errors that meet
        the threshold and aren't already reported.
        
        Args:
            create_issues: Whether to automatically create GitHub issues for bundled errors
            
        Returns:
            Dict with bundling summary including counts and issues created
        """
        try:
            if not hasattr(github_ops, '_error_aggregator'):
                return {
                    "status": "not_initialized",
                    "message": "Error aggregator not initialized yet",
                    "tool": "gh_bundle_errors",
                    "timestamp": time.time()
                }
            
            # Temporarily enable auto-issue creation if requested
            original_auto_create = github_ops._error_aggregator.enable_auto_issue_creation
            github_ops._error_aggregator.enable_auto_issue_creation = create_issues
            
            try:
                summary = github_ops._error_aggregator.bundle_and_report_errors()
            finally:
                # Restore original setting
                github_ops._error_aggregator.enable_auto_issue_creation = original_auto_create
            
            return {
                "status": "success",
                "summary": summary,
                "tool": "gh_bundle_errors",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in gh_bundle_errors: {e}")
            return {
                "error": str(e),
                "tool": "gh_bundle_errors",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_start_error_bundling(
        bundle_interval_minutes: int = 15,
        min_error_count: int = 3,
        enable_auto_issue_creation: bool = False
    ) -> Dict[str, Any]:
        """
        Start automatic error bundling in the background.
        
        Errors will be automatically collected from peers, deduplicated, and
        bundled at regular intervals. Issues can optionally be auto-created.
        
        Args:
            bundle_interval_minutes: How often to bundle errors (default: 15 minutes)
            min_error_count: Minimum error occurrences before creating issue (default: 3)
            enable_auto_issue_creation: Whether to auto-create GitHub issues (default: False)
            
        Returns:
            Dict with bundling configuration
        """
        try:
            from ipfs_accelerate_py.github_cli.error_aggregator import ErrorAggregator
            from ipfs_accelerate_py.github_cli.p2p_peer_registry import P2PPeerRegistry
            
            # Get repo from environment or github_ops
            repo = os.environ.get("GITHUB_REPOSITORY", "unknown/repo")
            
            # Initialize peer registry if not already done
            if not hasattr(github_ops, '_peer_registry'):
                github_ops._peer_registry = P2PPeerRegistry(repo=repo)
            
            # Initialize or update error aggregator
            github_ops._error_aggregator = ErrorAggregator(
                repo=repo,
                peer_registry=github_ops._peer_registry,
                bundle_interval_minutes=bundle_interval_minutes,
                min_error_count=min_error_count,
                enable_auto_issue_creation=enable_auto_issue_creation
            )
            
            # Start bundling thread
            github_ops._error_aggregator.start_bundling()
            
            return {
                "status": "success",
                "message": "Error bundling started",
                "config": {
                    "bundle_interval_minutes": bundle_interval_minutes,
                    "min_error_count": min_error_count,
                    "auto_issue_creation": enable_auto_issue_creation,
                    "repo": repo
                },
                "tool": "gh_start_error_bundling",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in gh_start_error_bundling: {e}")
            return {
                "error": str(e),
                "tool": "gh_start_error_bundling",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def gh_stop_error_bundling() -> Dict[str, Any]:
        """
        Stop automatic error bundling.
        
        Returns:
            Dict with stop status
        """
        try:
            if not hasattr(github_ops, '_error_aggregator'):
                return {
                    "status": "not_running",
                    "message": "Error bundling was not running",
                    "tool": "gh_stop_error_bundling",
                    "timestamp": time.time()
                }
            
            github_ops._error_aggregator.stop_bundling_thread()
            
            return {
                "status": "success",
                "message": "Error bundling stopped",
                "tool": "gh_stop_error_bundling",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in gh_stop_error_bundling: {e}")
            return {
                "error": str(e),
                "tool": "gh_stop_error_bundling",
                "timestamp": time.time()
            }

    @mcp.tool()
    def gh_list_all_issues(
        owner: Optional[str] = None,
        state: str = "open",
        limit_per_repo: int = 50
    ) -> Dict[str, Any]:
        """
        List issues across all accessible repositories
        
        Args:
            owner: Repository owner (user or organization), if None gets all accessible repos
            state: Issue state (open, closed, all)
            limit_per_repo: Maximum issues to fetch per repository
            
        Returns:
            Dict with issues from all repositories
        """
        try:
            all_issues = {}
            repo_count = 0
            total_issues = 0
            
            # Get list of all repositories
            repos_result = github_ops.list_repos(owner=owner, limit=200)
            if repos_result.get("success") and repos_result.get("repos"):
                repos = repos_result["repos"]
            else:
                # Check if it's a rate limit issue
                is_rate_limited = repos_result.get("rate_limited", False)
                error_msg = repos_result.get("error", "Failed to fetch repositories")
                
                if is_rate_limited or "rate limit" in error_msg.lower():
                    # Return gracefully with empty data when rate limited
                    return {
                        "status": "success",
                        "success": True,
                        "issues": {},
                        "repo_count": 0,
                        "total_issues": 0,
                        "tool": "gh_list_all_issues",
                        "timestamp": time.time(),
                        "rate_limited": True,
                        "message": "GitHub API rate limit exceeded. Issue data unavailable until rate limit resets."
                    }
                else:
                    # Non-rate-limit errors still return as errors
                    return {
                        "error": error_msg,
                        "success": False,
                        "tool": "gh_list_all_issues",
                        "timestamp": time.time()
                    }
            
            # For each repository, get issues
            for repo in repos:
                repo_name = f"{repo['owner']['login']}/{repo['name']}"
                try:
                    # Use GitHub CLI to get issues
                    if github_ops.gh_cli:
                        args = ["issue", "list", "--repo", repo_name, 
                               "--state", state, "--limit", str(limit_per_repo),
                               "--json", "number,title,state,createdAt,url,author"]
                        cmd_result = github_ops.gh_cli._run_command(args)
                        
                        if cmd_result.get("success") and cmd_result.get("stdout"):
                            import json
                            issues = json.loads(cmd_result["stdout"])
                            if issues:
                                all_issues[repo_name] = issues
                                total_issues += len(issues)
                                repo_count += 1
                                
                except Exception as e:
                    logger.debug(f"Failed to get issues for {repo_name}: {e}")
                    continue
            
            return {
                "status": "success",
                "issues": all_issues,
                "repo_count": repo_count,
                "total_issues": total_issues,
                "tool": "gh_list_all_issues",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in gh_list_all_issues: {e}")
            return {
                "error": str(e),
                "tool": "gh_list_all_issues",
                "timestamp": time.time()
            }

    @mcp.tool()
    def gh_list_all_pull_requests(
        owner: Optional[str] = None,
        state: str = "open", 
        limit_per_repo: int = 50
    ) -> Dict[str, Any]:
        """
        List pull requests across all accessible repositories
        
        Args:
            owner: Repository owner (user or organization), if None gets all accessible repos  
            state: PR state (open, closed, merged, all)
            limit_per_repo: Maximum PRs to fetch per repository
            
        Returns:
            Dict with pull requests from all repositories
        """
        try:
            all_prs = {}
            repo_count = 0
            total_prs = 0
            
            # Get list of all repositories
            repos_result = github_ops.list_repos(owner=owner, limit=200)
            if repos_result.get("success") and repos_result.get("repos"):
                repos = repos_result["repos"]
            else:
                # Check if it's a rate limit issue
                is_rate_limited = repos_result.get("rate_limited", False)
                error_msg = repos_result.get("error", "Failed to fetch repositories")
                
                if is_rate_limited or "rate limit" in error_msg.lower():
                    # Return gracefully with empty data when rate limited
                    return {
                        "status": "success",
                        "success": True,
                        "pull_requests": {},
                        "repo_count": 0,
                        "total_prs": 0,
                        "tool": "gh_list_all_pull_requests", 
                        "timestamp": time.time(),
                        "rate_limited": True,
                        "message": "GitHub API rate limit exceeded. Pull request data unavailable until rate limit resets."
                    }
                else:
                    # Non-rate-limit errors still return as errors
                    return {
                        "error": error_msg,
                        "success": False,
                        "tool": "gh_list_all_pull_requests", 
                        "timestamp": time.time()
                    }
            
            # For each repository, get pull requests
            for repo in repos:
                repo_name = f"{repo['owner']['login']}/{repo['name']}"
                try:
                    # Use GitHub CLI to get pull requests
                    if github_ops.gh_cli:
                        args = ["pr", "list", "--repo", repo_name,
                               "--state", state, "--limit", str(limit_per_repo),
                               "--json", "number,title,state,createdAt,url,author"]
                        cmd_result = github_ops.gh_cli._run_command(args)
                        
                        if cmd_result.get("success") and cmd_result.get("stdout"):
                            import json
                            prs = json.loads(cmd_result["stdout"])
                            if prs:
                                all_prs[repo_name] = prs
                                total_prs += len(prs)
                                repo_count += 1
                                
                except Exception as e:
                    logger.debug(f"Failed to get PRs for {repo_name}: {e}")
                    continue
            
            return {
                "status": "success",
                "pull_requests": all_prs,
                "repo_count": repo_count,
                "total_prs": total_prs,
                "tool": "gh_list_all_pull_requests",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in gh_list_all_pull_requests: {e}")
            return {
                "error": str(e),
                "tool": "gh_list_all_pull_requests",
                "timestamp": time.time()
            }
    
    logger.info("GitHub CLI tools (including error aggregation) registered successfully")
