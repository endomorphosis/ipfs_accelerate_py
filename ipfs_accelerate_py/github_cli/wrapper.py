"""
GitHub CLI Python Wrapper

This module provides Python wrappers for GitHub CLI (gh) commands,
enabling programmatic access to GitHub features.
"""

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class GitHubCLI:
    """Python wrapper for GitHub CLI (gh) commands."""
    
    def __init__(self, gh_path: str = "gh"):
        """
        Initialize GitHub CLI wrapper.
        
        Args:
            gh_path: Path to gh executable (default: "gh" from PATH)
        """
        self.gh_path = gh_path
        self._verify_installation()
    
    def _verify_installation(self) -> None:
        """Verify that gh CLI is installed and authenticated."""
        try:
            result = subprocess.run(
                [self.gh_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"gh CLI not found at {self.gh_path}")
            logger.info(f"GitHub CLI version: {result.stdout.strip()}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to verify gh CLI installation: {e}")
    
    def _run_command(
        self,
        args: List[str],
        stdin: Optional[str] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Run a gh CLI command and return the result.
        
        Args:
            args: Command arguments
            stdin: Optional stdin input
            timeout: Command timeout in seconds
            
        Returns:
            Dict with stdout, stderr, and returncode
        """
        try:
            cmd = [self.gh_path] + args
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
    
    def get_auth_status(self) -> Dict[str, Any]:
        """Get GitHub authentication status."""
        result = self._run_command(["auth", "status"])
        return {
            "authenticated": result["success"],
            "output": result["stdout"],
            "error": result["stderr"]
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
        limit: int = 30,
        visibility: str = "all"
    ) -> List[Dict[str, Any]]:
        """
        List GitHub repositories.
        
        Args:
            owner: Repository owner (user or org)
            limit: Maximum number of repos to return
            visibility: Repository visibility (all, public, private)
            
        Returns:
            List of repository dictionaries
        """
        args = ["repo", "list", "--json", "name,owner,url,updatedAt", "--limit", str(limit)]
        if owner:
            args.append(owner)
        
        result = self._run_command(args)
        if result["success"] and result["stdout"]:
            try:
                return json.loads(result["stdout"])
            except json.JSONDecodeError:
                logger.error(f"Failed to parse repo list: {result['stdout']}")
                return []
        return []
    
    def get_repo_info(self, repo: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific repository.
        
        Args:
            repo: Repository in format "owner/repo"
            
        Returns:
            Repository information dictionary
        """
        args = ["repo", "view", repo, "--json", 
                "name,owner,url,description,createdAt,updatedAt,pushedAt"]
        
        result = self._run_command(args)
        if result["success"] and result["stdout"]:
            try:
                return json.loads(result["stdout"])
            except json.JSONDecodeError:
                logger.error(f"Failed to parse repo info: {result['stdout']}")
                return None
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
        branch: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List workflow runs for a repository.
        
        Args:
            repo: Repository in format "owner/repo"
            status: Filter by status (queued, in_progress, completed)
            limit: Maximum number of runs to return
            branch: Filter by branch
            
        Returns:
            List of workflow run dictionaries
        """
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
                return json.loads(result["stdout"])
            except json.JSONDecodeError:
                logger.error(f"Failed to parse workflow runs: {result['stdout']}")
                return []
        return []
    
    def get_workflow_run(self, repo: str, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details of a specific workflow run.
        
        Args:
            repo: Repository in format "owner/repo"
            run_id: Workflow run ID
            
        Returns:
            Workflow run details
        """
        args = ["run", "view", run_id, "--repo", repo, "--json",
                "databaseId,name,status,conclusion,createdAt,updatedAt,event,headBranch,workflowName,jobs"]
        
        result = self.gh._run_command(args)
        if result["success"] and result["stdout"]:
            try:
                return json.loads(result["stdout"])
            except json.JSONDecodeError:
                logger.error(f"Failed to parse workflow run: {result['stdout']}")
                return None
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
        cutoff_date = datetime.now() - timedelta(days=since_days)
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
        since_days: int = 1,
        limit: int = 100
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
        cutoff_date = datetime.now() - timedelta(days=since_days)
        
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
        labels = ['self-hosted', 'linux', self._system_arch, 'docker']
        
        # Add GPU labels if available
        try:
            # Check for NVIDIA GPU
            result = subprocess.run(['which', 'nvidia-smi'], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                labels.extend(['cuda', 'gpu'])
        except:
            pass
        
        try:
            # Check for AMD GPU
            result = subprocess.run(['which', 'rocm-smi'], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                labels.extend(['rocm', 'gpu'])
        except:
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
        org: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List self-hosted runners.
        
        Args:
            repo: Repository in format "owner/repo" (for repo-level runners)
            org: Organization name (for org-level runners)
            
        Returns:
            List of runner dictionaries
        """
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
                return json.loads(result["stdout"])
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
        max_runners: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Provision self-hosted runners based on workflow queues.
        
        This method analyzes workflow queues and provisions runners
        based on system capacity and workflow load.
        
        Args:
            queues: Dict mapping repo names to workflow lists
            max_runners: Maximum runners to provision (defaults to system cores)
            
        Returns:
            Dict with provisioning status for each repo
        """
        if max_runners is None:
            max_runners = self.get_system_cores()
        
        logger.info(f"Provisioning runners (max: {max_runners}, system cores: {self.get_system_cores()})")
        
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
            
            # Get registration token
            token = self.get_runner_registration_token(repo=repo)
            
            if token:
                provisioning_status[repo] = {
                    "token": token,
                    "running_workflows": running_count,
                    "failed_workflows": failed_count,
                    "total_workflows": len(workflows),
                    "status": "token_generated"
                }
                runners_provisioned += 1
                logger.info(f"Generated token for {repo} ({running_count} running, {failed_count} failed)")
            else:
                provisioning_status[repo] = {
                    "error": "Failed to generate registration token",
                    "running_workflows": running_count,
                    "failed_workflows": failed_count,
                    "total_workflows": len(workflows),
                    "status": "failed"
                }
                logger.error(f"Failed to generate token for {repo}")
        
        return provisioning_status
