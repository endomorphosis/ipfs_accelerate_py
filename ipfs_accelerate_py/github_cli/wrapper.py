"""
GitHub CLI Python Wrapper

This module provides Python wrappers for GitHub CLI (gh) commands,
enabling programmatic access to GitHub features with optional caching.
"""

import json
import logging
import os
import subprocess
import sys
import time
import random
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Union

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

# Try to import datasets integration for GitHub operation tracking
try:
    from ...datasets_integration import (
        is_datasets_available,
        ProvenanceLogger,
        DatasetsManager
    )
    HAVE_DATASETS_INTEGRATION = True
except ImportError:
    try:
        from ..datasets_integration import (
            is_datasets_available,
            ProvenanceLogger,
            DatasetsManager
        )
        HAVE_DATASETS_INTEGRATION = True
    except ImportError:
        try:
            from datasets_integration import (
                is_datasets_available,
                ProvenanceLogger,
                DatasetsManager
            )
            HAVE_DATASETS_INTEGRATION = True
        except ImportError:
            HAVE_DATASETS_INTEGRATION = False
            is_datasets_available = lambda: False
            ProvenanceLogger = None
            DatasetsManager = None

if HAVE_STORAGE_WRAPPER:
    try:
        _storage = get_storage_wrapper(auto_detect_ci=True)
    except Exception:
        _storage = None
else:
    _storage = None

logger = logging.getLogger(__name__)


class GitHubCLI:
    """Python wrapper for GitHub CLI (gh) commands with optional caching."""
    
    def __init__(
        self,
        gh_path: str = "gh",
        enable_cache: bool = True,
        cache: Optional[GitHubAPICache] = None,
        cache_ttl: int = 300,
        auto_refresh_token: bool = True,
        token_refresh_threshold: int = 3600  # Refresh if token expires in less than 1 hour
    ):
        """
        Initialize GitHub CLI wrapper.
        
        Args:
            gh_path: Path to gh executable (default: "gh" from PATH)
            enable_cache: Whether to enable response caching
            cache: Custom cache instance (uses global cache if None)
            cache_ttl: Default cache TTL in seconds (default: 5 minutes)
            auto_refresh_token: Whether to automatically refresh GitHub token
            token_refresh_threshold: Refresh token if expires within this many seconds
        """
        # Initialize datasets integration for GitHub operation tracking
        self._provenance_logger = None
        self._datasets_manager = None
        if HAVE_DATASETS_INTEGRATION and is_datasets_available():
            try:
                self._provenance_logger = ProvenanceLogger()
                self._datasets_manager = DatasetsManager({
                    'enable_audit': True,
                    'enable_provenance': True
                })
                logger.info("GitHub CLI wrapper using datasets integration for operation tracking")
            except Exception as e:
                logger.debug(f"Datasets integration initialization skipped: {e}")
        
        self.gh_path = gh_path
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl

        # Heuristic: if we're running without a TTY (common for systemd services),
        # avoid any interactive authentication flows.
        self._non_interactive = (not sys.stdin.isatty()) or bool(os.environ.get("IPFS_ACCELERATE_NONINTERACTIVE"))
        
        # Disable auto_refresh_token if GITHUB_TOKEN env var is set (it can't be refreshed)
        if os.environ.get("GITHUB_TOKEN"):
            self.auto_refresh_token = False
            logger.debug("Disabled auto_refresh_token because GITHUB_TOKEN env var is set")
        elif self._non_interactive and auto_refresh_token:
            # In non-interactive contexts, token refresh attempts can trigger blocking auth flows.
            self.auto_refresh_token = False
            logger.debug("Disabled auto_refresh_token because running non-interactively")
        else:
            self.auto_refresh_token = auto_refresh_token
            
        self.token_refresh_threshold = token_refresh_threshold
        
        # Token tracking
        self._token_last_checked = 0
        self._token_expires_at = None
        
        # Set up cache
        if enable_cache:
            self.cache = cache if cache is not None else get_global_cache()
        else:
            self.cache = None
        
        self._verify_installation()

        # Global REST API rate-limit backoff.
        # When `gh api ...` hits a rate limit, continuing to call it every loop
        # just burns CPU/logs. We apply a cooldown so callers can fall back to
        # stale cache while GitHub resets the limit.
        self._rest_rate_limit_until: float = 0.0
        self._rest_rate_limit_backoff: float = 60.0
        self._rest_rate_limit_lock = Lock()
        self._last_rest_rate_limit_log: float = 0.0
        
        # Check token status on initialization
        if self.auto_refresh_token:
            self._check_and_refresh_token()

    @staticmethod
    def _stderr_indicates_rate_limit(stderr: str) -> bool:
        stderr_lower = (stderr or "").lower()
        return any(keyword in stderr_lower for keyword in ["rate limit", "api rate limit", "too many requests"])

    def _rest_cooldown_active(self) -> bool:
        return time.time() < self._rest_rate_limit_until

    def _note_rest_rate_limit(self, stderr: str) -> None:
        now = time.time()
        with self._rest_rate_limit_lock:
            # Escalate the backoff if we're still in a cooldown window.
            if now < self._rest_rate_limit_until:
                self._rest_rate_limit_backoff = min(self._rest_rate_limit_backoff * 2.0, 3600.0)
            else:
                self._rest_rate_limit_backoff = max(self._rest_rate_limit_backoff, 60.0)
            self._rest_rate_limit_until = now + self._rest_rate_limit_backoff

            # Avoid log spam if multiple calls hit the same limit.
            if now - self._last_rest_rate_limit_log >= 60:
                logger.warning(
                    f"REST API rate limit hit; backing off for {int(self._rest_rate_limit_backoff)}s (stale cache will be used when available)"
                )
                self._last_rest_rate_limit_log = now
            else:
                logger.debug(f"REST API rate limit hit (suppressed): {stderr}")
    
    def _verify_installation(self) -> None:
        """Verify that gh CLI is installed and authenticated."""
        try:
            logger.debug(f"Attempting to verify gh CLI at: {self.gh_path}")
            result = subprocess.run(
                [self.gh_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            logger.debug(f"gh CLI returncode: {result.returncode}, stdout: {result.stdout}, stderr: {result.stderr}")
            if result.returncode != 0:
                raise RuntimeError(f"gh CLI returned error (code {result.returncode}): stderr={result.stderr}, stdout={result.stdout}")
            logger.info(f"GitHub CLI version: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to verify gh CLI installation: {e}")
    
    def _check_and_refresh_token(self) -> bool:
        """
        Check if GitHub token needs refreshing and refresh if necessary.
        
        Returns:
            True if token is valid (or was refreshed), False otherwise
        """
        current_time = time.time()
        
        # Only check every 5 minutes to avoid excessive checks
        if current_time - self._token_last_checked < 300:
            if self._token_expires_at and self._token_expires_at > current_time:
                return True
        
        self._token_last_checked = current_time
        
        try:
            # Check token validity via gh CLI
            result = subprocess.run(
                [self.gh_path, "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                # Token is invalid or expired, try to refresh
                logger.warning("GitHub token appears invalid, attempting refresh...")
                return self._refresh_token()
            
            # Parse token expiration from status output
            # gh auth status output includes token expiration info
            if "Token expires" in result.stderr or "Token expires" in result.stdout:
                # Try to extract expiration time
                # Format: "✓ Token expires in X days/hours"
                import re
                output = result.stderr + result.stdout
                
                # Look for expiration patterns
                days_match = re.search(r'expires in (\d+) days?', output)
                hours_match = re.search(r'expires in (\d+) hours?', output)
                
                if days_match:
                    days = int(days_match.group(1))
                    expires_in = days * 86400
                elif hours_match:
                    hours = int(hours_match.group(1))
                    expires_in = hours * 3600
                else:
                    # Assume token is valid for at least threshold time
                    expires_in = self.token_refresh_threshold + 1
                
                self._token_expires_at = current_time + expires_in
                
                # Refresh if expiring soon
                if expires_in < self.token_refresh_threshold:
                    logger.info(f"Token expires in {expires_in}s, refreshing...")
                    return self._refresh_token()
                else:
                    logger.debug(f"Token valid for {expires_in}s")
                    return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking token status: {e}")
            return False
    
    def _refresh_token(self) -> bool:
        """
        Refresh GitHub authentication token.
        
        Returns:
            True if refresh succeeded, False otherwise
        """
        try:
            logger.info("Refreshing GitHub authentication token...")
            
            # Try to refresh using gh auth refresh
            result = subprocess.run(
                [self.gh_path, "auth", "refresh"],
                capture_output=True,
                text=True,
                timeout=30,
                input="y\n"  # Auto-confirm if prompted
            )
            
            if result.returncode == 0:
                logger.info("✓ GitHub token refreshed successfully")
                self._token_expires_at = time.time() + (7 * 86400)  # Assume 7 days validity
                return True
            else:
                logger.error(f"Failed to refresh token: {result.stderr}")

                # If refresh fails, only attempt interactive re-auth when running with a TTY.
                if self._non_interactive:
                    logger.warning("Skipping interactive re-authentication (non-interactive environment)")
                    return False

                logger.warning("Attempting to re-authenticate...")
                result = subprocess.run(
                    [self.gh_path, "auth", "login", "--web"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode == 0:
                    logger.info("✓ Re-authentication successful")
                    self._token_expires_at = time.time() + (7 * 86400)
                    return True
                else:
                    logger.error(f"Re-authentication failed: {result.stderr}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error refreshing token: {e}")
            return False
    
    def _run_command(
        self,
        args: List[str],
        stdin: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ) -> Dict[str, Any]:
        """
        Run a gh CLI command with exponential backoff retry.
        
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
        # Check and refresh token before running command
        if self.auto_refresh_token:
            self._check_and_refresh_token()
        
        last_error = None

        # If we're currently in a rate-limit cooldown window, avoid invoking gh commands
        # that hit the GitHub API. This is especially important for long-running
        # services that would otherwise hammer the API repeatedly and spam the journal.
        # Allow `gh auth ...` to run even during cooldown.
        if args and args[0] != "auth" and self._rest_cooldown_active():
            return {
                "stdout": "",
                "stderr": "GitHub API rate limit cooldown active (rate limit)",
                "returncode": 1,
                "success": False,
                "attempts": 0,
            }
        
        for attempt in range(max_retries + 1):
            try:
                cmd = [self.gh_path] + args
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
                
                # Check for rate limiting in stderr
                if result.returncode != 0 and self._stderr_indicates_rate_limit(result.stderr):
                    # Note rate limit and return immediately; retrying within seconds
                    # rarely helps for GitHub rate limits.
                    self._note_rest_rate_limit(result.stderr)
                    return {
                        "stdout": result.stdout.strip(),
                        "stderr": result.stderr.strip(),
                        "returncode": result.returncode,
                        "success": False,
                        "attempts": attempt + 1,
                    }
                
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
    
    def get_auth_status(self) -> Dict[str, Any]:
        """Get GitHub authentication status."""
        # Check for GITHUB_TOKEN environment variable first
        import os
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            # When using GITHUB_TOKEN, return immediately without API calls
            result = {
                "authenticated": True,
                "output": "Authenticated via GITHUB_TOKEN environment variable",
                "error": "",
                "success": True,
                "username": "endomorphosis",  # From token
                "token_type": "environment"
            }
            
            # Try to get rate limit info quickly (non-blocking, no retries)
            try:
                import subprocess
                rate_result = subprocess.run(
                    [self.gh_path, 'api', 'rate_limit'],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    env={**os.environ, 'GH_TOKEN': token}
                )
                if rate_result.returncode == 0:
                    import json
                    data = json.loads(rate_result.stdout)
                    if 'rate' in data:
                        result["rate_limit"] = data['rate']
            except Exception:
                pass  # Rate limit check is optional
            
            return result
        
        # Use short timeout and no retries for auth status check
        result = self._run_command(["auth", "status"], timeout=5, max_retries=0)
        return {
            "authenticated": result["success"],
            "output": result["stdout"],
            "error": result["stderr"],
            "success": result["success"]
        }
    
    def get_auth_token(self) -> Optional[str]:
        """Get GitHub authentication token."""
        result = self._run_command(["auth", "token"])
        if result["success"]:
            return result["stdout"]
        return None
    
    def list_repos(
        self,
        owner: Optional[str] = None,
        limit: int = 200,
        visibility: str = "all",
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List GitHub repositories.
        
        Args:
            owner: Repository owner (user or org)
            limit: Maximum number of repos to return
            visibility: Repository visibility (all, public, private)
            use_cache: Whether to use cached results
            
        Returns:
            List of repository dictionaries
        """
        # Check cache first
        if use_cache and self.cache:
            cached_result = self.cache.get("list_repos", owner=owner, limit=limit, visibility=visibility)
            if cached_result is not None:
                logger.debug(f"Using cached repo list for owner={owner}")
                return cached_result

        # If we're rate-limited, do not attempt API calls; prefer stale cache.
        if self._rest_cooldown_active():
            if self.cache:
                stale_data = self.cache.get_stale("list_repos", owner=owner, limit=limit, visibility=visibility)
                if stale_data is not None:
                    logger.info("Returning stale cache data for repos (rate limit cooldown)")
                    return stale_data
            return []
        
        # Use REST API instead of GraphQL to avoid separate rate limit
        # gh repo list uses GraphQL which has its own (often exhausted) rate limit
        # REST API: GET /user/repos or GET /users/{username}/repos
        
        # Use REST API with bounded pagination (avoid `--paginate`, which can exhaust rate limits).
        # We fetch pages until we reach `limit`.
        endpoint = f"users/{owner}/repos" if owner else "user/repos"
        per_page = 100
        page = 1
        repos: List[Dict[str, Any]] = []

        # Safety: limit excessive looping even if an endpoint behaves unexpectedly.
        max_pages = max(1, (max(1, int(limit)) + per_page - 1) // per_page)

        while len(repos) < limit and page <= max_pages:
            args = ["api", endpoint, "-F", f"per_page={per_page}", "-F", f"page={page}"]
            if not owner and visibility and visibility != "all":
                args += ["-F", f"visibility={visibility}"]

            if self.cache:
                self.cache.increment_api_call_count(
                    api_type="rest", operation=f"list_repos(owner={owner}, limit={limit}, page={page})"
                )

            result = self._run_command(args, max_retries=0)
            if not (result["success"] and result["stdout"]):
                if result.get("stderr") and self._stderr_indicates_rate_limit(result["stderr"]):
                    # Prefer stale cache and avoid logging the full gh stderr repeatedly.
                    logger.warning("REST API rate limit hit while listing repos")
                    if self.cache:
                        stale_data = self.cache.get_stale("list_repos", owner=owner, limit=limit, visibility=visibility)
                        if stale_data is not None:
                            logger.info("Returning stale cache data for repos (rate limit fallback)")
                            return stale_data
                elif result.get("stderr"):
                    logger.warning(f"Failed to list repos: {result['stderr']}")
                return []

            try:
                page_data = json.loads(result["stdout"])
            except json.JSONDecodeError:
                logger.error("Failed to parse repo list JSON from gh api")
                return []

            if not isinstance(page_data, list) or not page_data:
                break

            for item in page_data:
                if not isinstance(item, dict):
                    continue
                if item.get("archived") is True:
                    continue
                owner_login = None
                owner_obj = item.get("owner")
                if isinstance(owner_obj, dict):
                    owner_login = owner_obj.get("login")
                repo_obj: Dict[str, Any] = {
                    "name": item.get("name"),
                    "owner": {"login": owner_login} if owner_login else owner_obj,
                    "url": item.get("html_url"),
                    "updatedAt": item.get("updated_at"),
                }
                repos.append(repo_obj)
                if len(repos) >= limit:
                    break

            # If the API returned less than a full page, we're done.
            if len(page_data) < per_page:
                break

            page += 1

        repos = repos[:limit]

        if use_cache and self.cache:
            self.cache.put("list_repos", repos, ttl=self.cache_ttl, owner=owner, limit=limit, visibility=visibility)

        logger.info(f"Successfully fetched {len(repos)} repositories via REST API")
        return repos
    
    def get_repo_info(self, repo: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific repository.
        
        Args:
            repo: Repository in format "owner/repo"
            use_cache: Whether to use cached results
            
        Returns:
            Repository information dictionary
        """
        # Check cache first
        if use_cache and self.cache:
            cached_result = self.cache.get("get_repo_info", repo=repo)
            if cached_result is not None:
                logger.debug(f"Using cached repo info for {repo}")
                return cached_result
        
        args = ["repo", "view", repo, "--json", 
                "name,owner,url,description,createdAt,updatedAt,pushedAt"]
        
        # Track API call with operation details
        if self.cache:
            self.cache.increment_api_call_count(api_type="rest", operation=f"get_repo_info(repo={repo})")
        
        result = self._run_command(args)
        if result["success"] and result["stdout"]:
            try:
                repo_info = json.loads(result["stdout"])
                
                # Cache the result
                if use_cache and self.cache:
                    self.cache.put("get_repo_info", repo_info, ttl=self.cache_ttl, repo=repo)
                
                return repo_info
            except json.JSONDecodeError:
                logger.error(f"Failed to parse repo info: {result['stdout']}")
                return None
        else:
            # Check if it's a rate limit error and try stale cache
            if result.get("stderr") and "rate limit" in result.get("stderr", "").lower():
                logger.warning("API rate limit hit for get_repo_info")
                logger.debug(f"get_repo_info rate limit details: {result['stderr']}")
                if self.cache:
                    stale_data = self.cache.get_stale("get_repo_info", repo=repo)
                    if stale_data is not None:
                        logger.info(f"Returning stale cache data for repo info (rate limit fallback)")
                        return stale_data
        return None


class WorkflowQueue:
    """Manage GitHub Actions workflow queues."""
    
    def __init__(self, gh_cli: Optional[GitHubCLI] = None):
        """
        Initialize workflow queue manager.
        
        Args:
            gh_cli: GitHubCLI instance (creates new one if None)
        """
        self.gh = gh_cli or GitHubCLI()
    
    def list_workflow_runs(
        self,
        repo: str,
        status: Optional[str] = None,
        limit: int = 20,
        branch: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List workflow runs for a repository.
        
        Args:
            repo: Repository in format "owner/repo"
            status: Filter by status (queued, in_progress, completed)
            limit: Maximum number of runs to return
            branch: Filter by branch
            use_cache: Whether to use cached results
            
        Returns:
            List of workflow run dictionaries
        """
        # Check cache first (shorter TTL for workflow runs - 60s)
        if use_cache and self.gh.cache:
            cached_result = self.gh.cache.get("list_workflow_runs", repo=repo, status=status, limit=limit, branch=branch)
            if cached_result is not None:
                logger.debug(f"Using cached workflow runs for {repo}")
                return cached_result

        # If the underlying GitHub CLI is in a rate-limit cooldown, prefer stale cache
        # and avoid spamming warnings.
        if self.gh._rest_cooldown_active():
            if self.gh.cache:
                stale_data = self.gh.cache.get_stale("list_workflow_runs", repo=repo, status=status, limit=limit, branch=branch)
                if stale_data is not None:
                    logger.info("Returning stale cache data for workflow runs (rate limit cooldown)")
                    return stale_data
            return []
        
        args = ["run", "list", "--repo", repo, "--json",
                "databaseId,name,status,conclusion,createdAt,updatedAt,event,headBranch,workflowName",
                "--limit", str(limit)]
        
        if status:
            args.extend(["--status", status])
        if branch:
            args.extend(["--branch", branch])
        
        result = self.gh._run_command(args)
        if result["success"] and result["stdout"]:
            try:
                runs = json.loads(result["stdout"])
                
                # Cache with shorter TTL (60s) since workflow status changes frequently
                if use_cache and self.gh.cache:
                    self.gh.cache.put("list_workflow_runs", runs, ttl=60, repo=repo, status=status, limit=limit, branch=branch)
                
                return runs
            except json.JSONDecodeError:
                logger.error(f"Failed to parse workflow runs: {result['stdout']}")
                return []
        else:
            # Check if it's a rate limit error and try stale cache
            if result.get("stderr") and "rate limit" in result.get("stderr", "").lower():
                # If we're in cooldown, this is expected; keep logs quiet.
                if "cooldown active" in result.get("stderr", "").lower():
                    logger.debug("list_workflow_runs skipped due to cooldown")
                else:
                    logger.warning("API rate limit hit for list_workflow_runs")
                    logger.debug(f"list_workflow_runs rate limit details: {result['stderr']}")
                if self.gh.cache:
                    stale_data = self.gh.cache.get_stale("list_workflow_runs", repo=repo, status=status, limit=limit, branch=branch)
                    if stale_data is not None:
                        logger.info(f"Returning stale cache data for workflow runs (rate limit fallback)")
                        return stale_data
        return []
    
    def get_workflow_run(self, repo: str, run_id: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get details of a specific workflow run.
        
        Args:
            repo: Repository in format "owner/repo"
            run_id: Workflow run ID
            use_cache: Whether to use cached results
            
        Returns:
            Workflow run details
        """
        # Check cache first (shorter TTL - 60s)
        if use_cache and self.gh.cache:
            cached_result = self.gh.cache.get("get_workflow_run", repo=repo, run_id=run_id)
            if cached_result is not None:
                logger.debug(f"Using cached workflow run {run_id} for {repo}")
                return cached_result

        if self.gh._rest_cooldown_active():
            if self.gh.cache:
                stale_data = self.gh.cache.get_stale("get_workflow_run", repo=repo, run_id=run_id)
                if stale_data is not None:
                    logger.info("Returning stale cache data for workflow run (rate limit cooldown)")
                    return stale_data
            return None
        
        args = ["run", "view", run_id, "--repo", repo, "--json",
                "databaseId,name,status,conclusion,createdAt,updatedAt,event,headBranch,workflowName,jobs"]
        
        result = self.gh._run_command(args)
        if result["success"] and result["stdout"]:
            try:
                run_details = json.loads(result["stdout"])
                
                # Cache with shorter TTL (60s)
                if use_cache and self.gh.cache:
                    self.gh.cache.put("get_workflow_run", run_details, ttl=60, repo=repo, run_id=run_id)
                
                return run_details
            except json.JSONDecodeError:
                logger.error(f"Failed to parse workflow run: {result['stdout']}")
                return None
        else:
            # Check if it's a rate limit error and try stale cache
            if result.get("stderr") and "rate limit" in result.get("stderr", "").lower():
                if "cooldown active" in result.get("stderr", "").lower():
                    logger.debug("get_workflow_run skipped due to cooldown")
                else:
                    logger.warning("API rate limit hit for get_workflow_run")
                    logger.debug(f"get_workflow_run rate limit details: {result['stderr']}")
                if self.gh.cache:
                    stale_data = self.gh.cache.get_stale("get_workflow_run", repo=repo, run_id=run_id)
                    if stale_data is not None:
                        logger.info(f"Returning stale cache data for workflow run (rate limit fallback)")
                        return stale_data
        return None
    
    def list_failed_runs(
        self,
        repo: str,
        since_days: int = 1,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List failed workflow runs for a repository.
        
        Args:
            repo: Repository in format "owner/repo"
            since_days: Only include runs from the last N days
            limit: Maximum number of runs to return
            
        Returns:
            List of failed workflow run dictionaries
        """
        # Get all completed runs
        all_runs = self.list_workflow_runs(repo, status="completed", limit=limit)
        
        # Filter for failures within time window
        cutoff_date = datetime.now().replace(tzinfo=timezone.utc) - timedelta(days=since_days)
        failed_runs = []
        
        for run in all_runs:
            if run.get("conclusion") in ["failure", "timed_out", "cancelled"]:
                created_at = datetime.fromisoformat(run["createdAt"].replace("Z", "+00:00"))
                if created_at >= cutoff_date:
                    failed_runs.append(run)
        
        return failed_runs
    
    def get_repos_with_recent_activity(
        self,
        owner: Optional[str] = None,
        since_days: int = 7,
        limit: int = 200
    ) -> List[str]:
        """
        Get list of repositories with recent activity.
        
        Args:
            owner: Repository owner (user or org)
            since_days: Only include repos updated in the last N days
            limit: Maximum number of repos to check
            
        Returns:
            List of repository names in format "owner/repo"
        """
        repos = self.gh.list_repos(owner=owner, limit=limit)
        cutoff_date = datetime.now().replace(tzinfo=timezone.utc) - timedelta(days=since_days)
        
        recent_repos = []
        for repo in repos:
            updated_at = datetime.fromisoformat(repo["updatedAt"].replace("Z", "+00:00"))
            if updated_at >= cutoff_date:
                owner_name = repo["owner"]["login"]
                repo_name = repo["name"]
                recent_repos.append(f"{owner_name}/{repo_name}")
        
        return recent_repos
    
    def _check_workflow_runner_compatibility(
        self,
        workflow: Dict[str, Any],
        repo: str,
        system_arch: str
    ) -> bool:
        """
        Check if a workflow is compatible with the current system architecture.
        
        Args:
            workflow: Workflow run dictionary
            repo: Repository name
            system_arch: System architecture (e.g., 'x64', 'arm64')
            
        Returns:
            True if the workflow is compatible with this runner
        """
        workflow_name = workflow.get("workflowName", "").lower()
        
        # Architecture-specific workflow patterns
        if "arm64" in workflow_name or "aarch64" in workflow_name:
            # This workflow specifically requires ARM64
            return system_arch == "arm64"
        
        if "amd64" in workflow_name or "x86" in workflow_name or "x64" in workflow_name:
            # This workflow specifically requires x86_64
            return system_arch == "x64"
        
        # Try to get detailed job information to check runner labels
        try:
            run_id = workflow.get("databaseId")
            if run_id:
                detailed_run = self.get_workflow_run(repo, str(run_id))
                if detailed_run and "jobs" in detailed_run:
                    for job in detailed_run["jobs"]:
                        runner_labels = job.get("labels", [])
                        # Check if the job has specific architecture requirements
                        if "arm64" in runner_labels or "aarch64" in runner_labels:
                            return system_arch == "arm64"
                        if "x64" in runner_labels or "amd64" in runner_labels:
                            return system_arch == "x64"
        except Exception as e:
            logger.debug(f"Could not get detailed job info: {e}")
        
        # If no specific architecture is mentioned, assume it's compatible
        # (most workflows use ubuntu-latest which is x64)
        return True
    
    def create_workflow_queues(
        self,
        owner: Optional[str] = None,
        since_days: int = 1,
        system_arch: Optional[str] = None,
        filter_by_arch: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create workflow queues for repositories with recent activity.
        
        This method finds all repositories with recent updates and creates
        queues of running or failed workflows for each repository.
        
        Args:
            owner: Repository owner (user or org)
            since_days: Only include repos/workflows from the last N days
            system_arch: System architecture (e.g., 'x64', 'arm64') for filtering
            filter_by_arch: Whether to filter workflows by architecture compatibility
            
        Returns:
            Dict mapping repo names to lists of workflow runs
        """
        queues = {}
        
        # Get repositories with recent activity
        recent_repos = self.get_repos_with_recent_activity(owner=owner, since_days=since_days)
        logger.info(f"Found {len(recent_repos)} repositories with recent activity")
        
        # For each repo, get workflow runs
        for repo in recent_repos:
            logger.info(f"Processing repository: {repo}")
            
            # Get running workflows
            running = self.list_workflow_runs(repo, status="in_progress", limit=20)
            
            # Get failed workflows
            failed = self.list_failed_runs(repo, since_days=since_days, limit=20)
            
            # Combine workflows
            all_workflows = running + failed
            
            # Filter by architecture compatibility if requested
            if filter_by_arch and system_arch and all_workflows:
                compatible_workflows = [
                    w for w in all_workflows
                    if self._check_workflow_runner_compatibility(w, repo, system_arch)
                ]
                
                if len(compatible_workflows) < len(all_workflows):
                    logger.info(f"  Filtered {len(all_workflows) - len(compatible_workflows)} incompatible workflows for {system_arch}")
                
                all_workflows = compatible_workflows
            
            if all_workflows:
                queues[repo] = all_workflows
                logger.info(f"  Found {len(running)} running and {len(failed)} failed workflows (after filtering)")
        
        return queues


class RunnerManager:
    """Manage GitHub self-hosted runners."""
    
    def __init__(self, gh_cli: Optional[GitHubCLI] = None):
        """
        Initialize runner manager.
        
        Args:
            gh_cli: GitHubCLI instance (creates new one if None)
        """
        self.gh = gh_cli or GitHubCLI()
        self._system_arch = self._detect_system_architecture()
        self._runner_labels = self._generate_runner_labels()
    
    def _detect_system_architecture(self) -> str:
        """
        Detect the system architecture.
        
        Returns:
            Architecture string ('x64', 'arm64', etc.)
        """
        import platform
        arch = platform.machine().lower()
        
        # Map common architecture names to GitHub runner labels
        arch_map = {
            'x86_64': 'x64',
            'amd64': 'x64',
            'aarch64': 'arm64',
            'arm64': 'arm64',
        }
        
        return arch_map.get(arch, arch)
    
    def _generate_runner_labels(self) -> str:
        """
        Generate appropriate labels for this runner based on system capabilities.
        
        Returns:
            Comma-separated string of labels
        """
        import shutil
        
        labels = ['self-hosted', 'linux', self._system_arch, 'docker']
        
        # Add GPU labels if available
        try:
            # Check for NVIDIA GPU
            if shutil.which('nvidia-smi'):
                labels.extend(['cuda', 'gpu'])
        except Exception:
            pass
        
        try:
            # Check for AMD GPU
            if shutil.which('rocm-smi'):
                labels.extend(['rocm', 'gpu'])
        except Exception:
            pass
        
        if 'gpu' not in labels:
            labels.append('cpu-only')
        
        return ','.join(labels)
    
    def get_system_architecture(self) -> str:
        """Get the detected system architecture."""
        return self._system_arch
    
    def get_runner_labels(self) -> str:
        """Get the generated runner labels."""
        return self._runner_labels
    
    def get_system_cores(self) -> int:
        """Get the number of CPU cores on the system."""
        import multiprocessing
        return multiprocessing.cpu_count()
    
    def list_runners(
        self,
        repo: Optional[str] = None,
        org: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List self-hosted runners.
        
        Args:
            repo: Repository in format "owner/repo" (for repo-level runners)
            org: Organization name (for org-level runners)
            use_cache: Whether to use cached results
            
        Returns:
            List of runner dictionaries
        """
        # Check cache first (shorter TTL - 30s for runner status)
        if use_cache and self.gh.cache:
            cached_result = self.gh.cache.get("list_runners", repo=repo, org=org)
            if cached_result is not None:
                logger.debug(f"Using cached runner list for repo={repo}, org={org}")
                return cached_result
        
        if repo:
            # Repo-level runners (requires appropriate permissions)
            result = self.gh._run_command(
                ["api", f"repos/{repo}/actions/runners", "--jq", ".runners"]
            )
        elif org:
            # Org-level runners
            result = self.gh._run_command(
                ["api", f"orgs/{org}/actions/runners", "--jq", ".runners"]
            )
        else:
            logger.error("Must specify either repo or org")
            return []
        
        if result["success"] and result["stdout"]:
            try:
                runners = json.loads(result["stdout"])
                
                # Cache with very short TTL (30s) since runner status changes frequently
                if use_cache and self.gh.cache:
                    self.gh.cache.put("list_runners", runners, ttl=30, repo=repo, org=org)
                
                return runners
            except json.JSONDecodeError:
                logger.error(f"Failed to parse runners: {result['stdout']}")
                return []
        return []
    
    def get_runner_registration_token(
        self,
        repo: Optional[str] = None,
        org: Optional[str] = None
    ) -> Optional[str]:
        """
        Get a registration token for adding a new self-hosted runner.
        
        Args:
            repo: Repository in format "owner/repo" (for repo-level runners)
            org: Organization name (for org-level runners)
            
        Returns:
            Registration token or None if failed
        """
        if repo:
            endpoint = f"repos/{repo}/actions/runners/registration-token"
        elif org:
            endpoint = f"orgs/{org}/actions/runners/registration-token"
        else:
            logger.error("Must specify either repo or org")
            return None
        
        result = self.gh._run_command(
            ["api", "--method", "POST", endpoint, "--jq", ".token"]
        )
        
        if result["success"] and result["stdout"]:
            return result["stdout"]
        return None
    
    def provision_runners_for_queue(
        self,
        queues: Dict[str, List[Dict[str, Any]]],
        max_runners: Optional[int] = None,
        min_runners_per_repo: int = 1
    ) -> Dict[str, Dict[str, Any]]:
        """
        Provision self-hosted runners based on workflow queues.
        
        This method analyzes workflow queues and provisions runners
        based on system capacity and workflow load. Guarantees at least
        one runner per repository with active workflows.
        
        Args:
            queues: Dict mapping repo names to workflow lists
            max_runners: Maximum runners to provision (defaults to system cores)
            min_runners_per_repo: Minimum runners per repository (default: 1)
            
        Returns:
            Dict with provisioning status for each repo
        """
        if max_runners is None:
            max_runners = self.get_system_cores()
        
        logger.info(f"Provisioning runners (max: {max_runners}, min per repo: {min_runners_per_repo}, system cores: {self.get_system_cores()})")
        
        provisioning_status = {}
        runners_provisioned = 0
        
        # Sort repos by number of workflows (prioritize busy repos)
        sorted_repos = sorted(
            queues.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        for repo, workflows in sorted_repos:
            if runners_provisioned >= max_runners:
                logger.info(f"Reached max runners limit: {max_runners}")
                break
            
            # Determine how many runners this repo needs
            running_count = sum(1 for w in workflows if w.get("status") == "in_progress")
            failed_count = sum(1 for w in workflows if w.get("conclusion") in ["failure", "timed_out"])
            queued_count = sum(1 for w in workflows if w.get("status") in ["queued", "waiting"])
            
            # Calculate needed runners:
            # - At least min_runners_per_repo for any repo with workflows
            # - Additional runners for queued workflows (1 per queued workflow)
            # - Don't provision extra runners for failed workflows (they already ran)
            base_runners = min_runners_per_repo
            additional_runners = queued_count  # Only provision for queued workflows
            runners_needed = base_runners + additional_runners
            
            # Don't exceed available capacity
            runners_to_provision = min(runners_needed, max_runners - runners_provisioned)
            # Ensure we provision at least min_runners_per_repo if capacity allows
            runners_to_provision = max(min(min_runners_per_repo, max_runners - runners_provisioned), runners_to_provision)
            
            if runners_to_provision <= 0:
                logger.info(f"No capacity for {repo} (would need {runners_needed}, have {max_runners - runners_provisioned} slots)")
                continue  # Check next repo instead of breaking
            
            # Generate tokens for this repo (one token can be reused by multiple runners)
            token = self.get_runner_registration_token(repo=repo)
            
            if token:
                provisioning_status[repo] = {
                    "token": token,
                    "running_workflows": running_count,
                    "failed_workflows": failed_count,
                    "queued_workflows": queued_count,
                    "total_workflows": len(workflows),
                    "runners_needed": runners_needed,
                    "runners_to_provision": runners_to_provision,
                    "status": "token_generated"
                }
                runners_provisioned += runners_to_provision
                logger.info(f"Generated token for {repo}: provisioning {runners_to_provision} runner(s) (min {min_runners_per_repo} + {queued_count} queued) for {len(workflows)} workflow(s) ({running_count} running, {queued_count} queued, {failed_count} failed)")
            else:
                provisioning_status[repo] = {
                    "error": "Failed to generate registration token",
                    "running_workflows": running_count,
                    "failed_workflows": failed_count,
                    "queued_workflows": queued_count,
                    "total_workflows": len(workflows),
                    "status": "failed"
                }
                logger.error(f"Failed to generate token for {repo}")
        
        return provisioning_status
