"""
GitHub GraphQL API Python Wrapper

This module provides Python wrappers for GitHub GraphQL API,
which has separate rate limits from the REST API and can be more efficient.
Use this when REST API rate limits are exhausted.
"""

import json
import logging
import subprocess
from typing import Any, Dict, List, Optional

from .cache import GitHubAPICache, get_global_cache

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

if HAVE_STORAGE_WRAPPER:
    try:
        _storage = get_storage_wrapper(auto_detect_ci=True)
    except Exception:
        _storage = None
else:
    _storage = None

logger = logging.getLogger(__name__)


class GitHubGraphQL:
    """Python wrapper for GitHub GraphQL API with optional caching."""
    
    def __init__(
        self,
        gh_path: str = "gh",
        enable_cache: bool = True,
        cache: Optional[GitHubAPICache] = None,
        cache_ttl: int = 300
    ):
        """
        Initialize GitHub GraphQL wrapper.
        
        Args:
            gh_path: Path to gh executable (default: "gh" from PATH)
            enable_cache: Whether to enable response caching
            cache: Custom cache instance (uses global cache if None)
            cache_ttl: Default cache TTL in seconds (default: 5 minutes)
        """
        self.gh_path = gh_path
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        
        # Set up cache
        if enable_cache:
            self.cache = cache if cache is not None else get_global_cache()
        else:
            self.cache = None
    
    def _run_graphql_query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute a GraphQL query.
        
        Args:
            query: GraphQL query string
            variables: Optional query variables
            
        Returns:
            Dict with 'success' bool and 'data' or 'error'
        """
        try:
            # Track GraphQL API call
            if self.cache:
                self.cache.increment_api_call_count(api_type="graphql")
            
            cmd = [self.gh_path, "api", "graphql", "-f", f"query={query}"]
            
            # Add variables if provided
            if variables:
                for key, value in variables.items():
                    if isinstance(value, (int, bool)):
                        cmd.extend(["-F", f"{key}={value}"])
                    else:
                        cmd.extend(["-f", f"{key}={value}"])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if "errors" in data:
                    return {
                        "success": False,
                        "error": f"GraphQL errors: {data['errors']}"
                    }
                return {
                    "success": True,
                    "data": data.get("data", {})
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr.strip()
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Query timeout"}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON response: {e}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_workflow_runs(
        self,
        owner: str,
        repo: str,
        status: Optional[str] = None,
        limit: int = 20,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        List workflow runs using GraphQL.
        
        Args:
            owner: Repository owner
            repo: Repository name
            status: Filter by status (QUEUED, IN_PROGRESS, COMPLETED)
            limit: Maximum number of runs to return
            use_cache: Whether to use cached results
            
        Returns:
            Dict with success bool and list of workflow runs
        """
        # Check cache first
        cache_key = f"graphql_workflow_runs_{owner}_{repo}_{status}_{limit}"
        if use_cache and self.cache:
            cached_result = self.cache.get("graphql_workflow_runs", 
                                          owner=owner, repo=repo, 
                                          status=status, limit=limit)
            if cached_result is not None:
                logger.debug(f"Using cached GraphQL workflow runs for {owner}/{repo}")
                self.cache.increment_graphql_cache_hit()
                return {"success": True, "data": cached_result}
        
        # Build GraphQL query
        # Note: GitHub GraphQL uses UPPER_CASE for workflow run status
        status_filter = ""
        if status:
            gql_status = status.upper()
            if gql_status == "QUEUED":
                status_filter = 'status: QUEUED'
            elif gql_status == "IN_PROGRESS":
                status_filter = 'status: IN_PROGRESS'
            elif gql_status == "COMPLETED":
                status_filter = 'status: COMPLETED'
        
        query = f"""
        query {{
          repository(owner: "{owner}", name: "{repo}") {{
            workflowRuns(first: {limit}, {status_filter}) {{
              nodes {{
                id
                databaseId
                name
                status
                conclusion
                createdAt
                updatedAt
                event
                headBranch
                workflow {{
                  name
                }}
              }}
              totalCount
            }}
          }}
        }}
        """
        
        result = self._run_graphql_query(query)
        
        if result["success"]:
            workflow_data = result["data"].get("repository", {}).get("workflowRuns", {})
            runs = workflow_data.get("nodes", [])
            
            # Cache the result
            if use_cache and self.cache:
                self.cache.put(
                    "graphql_workflow_runs",
                    runs,
                    ttl=60,  # Short TTL for workflow runs
                    owner=owner,
                    repo=repo,
                    status=status,
                    limit=limit
                )
            
            return {
                "success": True,
                "data": {
                    "workflow_runs": runs,
                    "total_count": workflow_data.get("totalCount", len(runs))
                }
            }
        else:
            return result
    
    def list_runners(
        self,
        owner: str,
        repo: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        List self-hosted runners using GraphQL.
        
        Args:
            owner: Organization or user name
            repo: Repository name (optional, for repo-level runners)
            use_cache: Whether to use cached results
            
        Returns:
            Dict with success bool and list of runners
        """
        # Check cache first
        if use_cache and self.cache:
            cached_result = self.cache.get("graphql_runners", 
                                          owner=owner, repo=repo)
            if cached_result is not None:
                logger.debug(f"Using cached GraphQL runners for {owner}/{repo or 'org'}")
                self.cache.increment_graphql_cache_hit()
                return {"success": True, "data": cached_result}
        
        # Build GraphQL query for org-level runners
        if repo:
            # Repository-level runners
            query = f"""
            query {{
              repository(owner: "{owner}", name: "{repo}") {{
                runners(first: 100) {{
                  nodes {{
                    id
                    name
                    status
                    busy
                    labels {{
                      name
                    }}
                    os
                  }}
                  totalCount
                }}
              }}
            }}
            """
        else:
            # Organization-level runners
            query = f"""
            query {{
              organization(login: "{owner}") {{
                runners(first: 100) {{
                  nodes {{
                    id
                    name
                    status
                    busy
                    labels {{
                      name
                    }}
                    os
                  }}
                  totalCount
                }}
              }}
            }}
            """
        
        result = self._run_graphql_query(query)
        
        if result["success"]:
            entity = result["data"].get("repository" if repo else "organization", {})
            runner_data = entity.get("runners", {})
            runners = runner_data.get("nodes", [])
            
            # Cache the result
            if use_cache and self.cache:
                self.cache.put(
                    "graphql_runners",
                    runners,
                    ttl=30,  # Short TTL for runner status
                    owner=owner,
                    repo=repo
                )
            
            return {
                "success": True,
                "data": {
                    "runners": runners,
                    "total_count": runner_data.get("totalCount", len(runners))
                }
            }
        else:
            return result
    
    def get_rate_limit(self) -> Dict[str, Any]:
        """
        Get GraphQL API rate limit status.
        
        Returns:
            Dict with rate limit information
        """
        query = """
        query {
          rateLimit {
            limit
            cost
            remaining
            resetAt
          }
        }
        """
        
        result = self._run_graphql_query(query)
        
        if result["success"]:
            return {
                "success": True,
                "data": result["data"].get("rateLimit", {})
            }
        else:
            return result
