"""
GitHub Actions Runner Kit - Core Runner Autoscaling

This module provides GitHub Actions runner autoscaling without CLI dependencies.
It uses docker_kit.py for container provisioning and github_kit.py for workflow monitoring.
"""

import json
import logging
import os
import socket
import time
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for GitHub Actions runner."""
    owner: Optional[str] = None
    poll_interval: int = 120
    since_days: int = 1
    max_runners: int = 10
    filter_by_arch: bool = True
    enable_p2p: bool = False
    runner_image: str = "myoung34/github-runner:latest"
    runner_work_dir: str = "/tmp/_work"
    token_file: str = "/var/lib/github-runner-autoscaler/runner_tokens.json"
    network_mode: str = "host"
    memory_limit: str = "4g"
    cpu_limit: float = 4.0


@dataclass
class WorkflowQueue:
    """Workflow queue information."""
    repo: str
    workflows: List[Dict] = field(default_factory=list)
    running: int = 0
    failed: int = 0
    pending: int = 0
    
    @property
    def total(self) -> int:
        return len(self.workflows)


@dataclass
class RunnerStatus:
    """Status of a runner container."""
    container_id: str
    repo: str
    status: str
    created_at: datetime
    labels: List[str] = field(default_factory=list)


@dataclass
class AutoscalerStatus:
    """Status of the autoscaler."""
    running: bool
    start_time: Optional[datetime] = None
    iterations: int = 0
    active_runners: int = 0
    queued_workflows: int = 0
    last_check: Optional[datetime] = None
    repositories_monitored: int = 0


class RunnerKit:
    """
    GitHub Actions runner autoscaling module.
    
    Uses docker_kit for container operations and github_kit for
    workflow monitoring to provide autoscaling capabilities.
    """
    
    def __init__(
        self,
        config: Optional[RunnerConfig] = None,
        docker_kit=None,
        github_kit=None
    ):
        """
        Initialize Runner Kit.
        
        Args:
            config: Runner configuration
            docker_kit: DockerKit instance (will be created if None)
            github_kit: GitHubKit instance (will be created if None)
        """
        self.config = config or RunnerConfig()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.status = AutoscalerStatus(running=False)
        self.active_containers: Set[str] = set()
        self.last_cleanup = datetime.now()
        
        # Initialize docker_kit
        if docker_kit is None:
            try:
                from ipfs_accelerate_py.kit.docker_kit import DockerKit
                self.docker = DockerKit()
            except Exception as e:
                logger.error(f"Failed to initialize DockerKit: {e}")
                raise
        else:
            self.docker = docker_kit
        
        # Initialize github_kit
        if github_kit is None:
            try:
                from ipfs_accelerate_py.kit.github_kit import GitHubKit
                self.github = GitHubKit()
            except Exception as e:
                logger.error(f"Failed to initialize GitHubKit: {e}")
                raise
        else:
            self.github = github_kit
        
        logger.info("RunnerKit initialized")
        logger.info(f"  Owner: {self.config.owner or 'All accessible repos'}")
        logger.info(f"  Max runners: {self.config.max_runners}")
        logger.info(f"  Poll interval: {self.config.poll_interval}s")
        logger.info(f"  Runner image: {self.config.runner_image}")
    
    def get_workflow_queues(self) -> List[WorkflowQueue]:
        """
        Get workflow queues for monitored repositories.
        
        Returns:
            List of WorkflowQueue objects
        """
        queues = []
        
        try:
            # Get repositories
            repos_result = self.github.list_repos(
                owner=self.config.owner,
                limit=30
            )
            
            if not repos_result.success or not repos_result.data:
                logger.warning("No repositories found")
                return queues
            
            repos = repos_result.data
            
            # For each repo, get workflow runs
            for repo in repos:
                repo_full_name = repo.get('full_name') or f"{repo['owner']['login']}/{repo['name']}"
                
                # Get workflow runs
                workflows_result = self.github.list_workflow_runs(
                    repo=repo_full_name,
                    status='in_progress',
                    limit=50
                )
                
                workflows = workflows_result.data if workflows_result.success else []
                
                # Calculate stats
                running = sum(1 for w in workflows if w.get('status') == 'in_progress')
                failed = sum(1 for w in workflows if w.get('conclusion') in ['failure', 'timed_out', 'cancelled'])
                pending = len(workflows) - running - failed
                
                queue = WorkflowQueue(
                    repo=repo_full_name,
                    workflows=workflows,
                    running=running,
                    failed=failed,
                    pending=pending
                )
                queues.append(queue)
        
        except Exception as e:
            logger.error(f"Failed to get workflow queues: {e}")
        
        return queues
    
    def generate_runner_token(self, repo: str) -> Optional[str]:
        """
        Generate a runner registration token for a repository.
        
        Args:
            repo: Repository in format 'owner/repo'
            
        Returns:
            Runner token if successful, None otherwise
        """
        # Use GitHub CLI via github_kit
        result = self.github._run_command([
            'api',
            f'repos/{repo}/actions/runners/registration-token',
            '--method', 'POST'
        ])
        
        if result.success and result.stdout:
            try:
                data = json.loads(result.stdout)
                return data.get('token')
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse token response: {e}")
        
        return None
    
    def launch_runner_container(
        self,
        repo: str,
        token: str,
        labels: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Launch a GitHub Actions runner container.
        
        Args:
            repo: Repository in format 'owner/repo'
            token: Runner registration token
            labels: Runner labels
            
        Returns:
            Container ID if successful, None otherwise
        """
        if not token:
            logger.error("Cannot launch runner without token")
            return None
        
        # Generate unique container name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        container_name = f"runner_{repo.replace('/', '_')}_{timestamp}_{unique_id}"
        
        # Prepare labels
        if labels is None:
            labels = ['self-hosted', 'linux', 'docker']
        labels_str = ','.join(labels)
        
        # Prepare environment variables
        environment = {
            'REPO_URL': f'https://github.com/{repo}',
            'RUNNER_TOKEN': token,
            'RUNNER_NAME': container_name,
            'LABELS': labels_str,
            'RUNNER_WORKDIR': self.config.runner_work_dir,
            'EPHEMERAL': 'true',  # Self-destruct after one job
        }
        
        # Prepare volumes
        volumes = {
            self.config.runner_work_dir: self.config.runner_work_dir,
            '/var/run/docker.sock': '/var/run/docker.sock',  # For Docker-in-Docker
        }
        
        # Launch container using docker_kit
        logger.info(f"Launching runner container for {repo}")
        result = self.docker.run_container(
            image=self.config.runner_image,
            name=container_name,
            detach=True,
            remove=True,
            environment=environment,
            volumes=volumes,
            network=self.config.network_mode,
            memory=self.config.memory_limit,
            cpus=self.config.cpu_limit,
            timeout=30
        )
        
        if result.success:
            container_id = result.stdout.strip() if result.stdout else None
            if container_id:
                self.active_containers.add(container_id)
                logger.info(f"✓ Launched container {container_id[:12]} for {repo}")
                return container_id
            else:
                logger.error("Container launched but no ID returned")
        else:
            logger.error(f"Failed to launch container: {result.error or result.stderr}")
        
        return None
    
    def list_runner_containers(self) -> List[RunnerStatus]:
        """
        List active GitHub Actions runner containers.
        
        Returns:
            List of RunnerStatus objects
        """
        runners = []
        
        # List containers with github-runner label
        result = self.docker.list_containers(
            all_containers=True,
            filters={'label': 'github-runner=true'}
        )
        
        if result.success and result.data:
            for container in result.data:
                # Extract repo from labels
                repo = container.get('Labels', {}).get('repo', 'unknown')
                
                runner = RunnerStatus(
                    container_id=container.get('ID', '')[:12],
                    repo=repo,
                    status=container.get('State', 'unknown'),
                    created_at=datetime.now(),  # Would need to parse from container
                    labels=[]
                )
                runners.append(runner)
        
        return runners
    
    def stop_runner_container(self, container: str) -> bool:
        """
        Stop a runner container.
        
        Args:
            container: Container ID or name
            
        Returns:
            True if successful, False otherwise
        """
        result = self.docker.stop_container(container, timeout=10)
        
        if result.success:
            self.active_containers.discard(container)
            logger.info(f"Stopped runner container {container[:12]}")
            return True
        else:
            logger.error(f"Failed to stop container: {result.error or result.stderr}")
            return False
    
    def cleanup_exited_containers(self) -> int:
        """
        Clean up exited runner containers.
        
        Returns:
            Number of containers cleaned up
        """
        removed = 0
        
        # Get all containers including exited
        result = self.docker.list_containers(
            all_containers=True,
            filters={'label': 'github-runner=true'}
        )
        
        if result.success and result.data:
            for container in result.data:
                container_id = container.get('ID', '')
                status = container.get('State', '')
                
                # Remove from active set if exited
                if container_id in self.active_containers and status.lower() in ['exited', 'dead']:
                    self.active_containers.discard(container_id)
                    removed += 1
                    logger.debug(f"Removed container {container_id[:12]} from active set")
        
        return removed
    
    def provision_runners_for_queues(self, queues: List[WorkflowQueue]) -> Dict[str, Any]:
        """
        Provision runners for workflow queues.
        
        Args:
            queues: List of workflow queues
            
        Returns:
            Dictionary with provisioning results
        """
        results = {}
        
        # Calculate how many runners to provision
        total_workflows = sum(q.total for q in queues)
        active_runners = len(list(self.active_containers))
        available_slots = self.config.max_runners - active_runners
        
        if available_slots <= 0:
            logger.info(f"At capacity: {active_runners}/{self.config.max_runners} runners active")
            return results
        
        logger.info(f"Provisioning runners: {total_workflows} workflows, {available_slots} slots available")
        
        # Provision at least 1 runner per repo with workflows
        provisioned = 0
        for queue in queues:
            if provisioned >= available_slots:
                break
            
            if queue.total > 0 or queue.repo:
                # Generate token
                token = self.generate_runner_token(queue.repo)
                
                if token:
                    # Launch runner
                    container_id = self.launch_runner_container(
                        repo=queue.repo,
                        token=token
                    )
                    
                    results[queue.repo] = {
                        'status': 'provisioned' if container_id else 'failed',
                        'container_id': container_id,
                        'workflows': queue.total,
                        'token_generated': bool(token)
                    }
                    
                    if container_id:
                        provisioned += 1
                else:
                    results[queue.repo] = {
                        'status': 'token_failed',
                        'workflows': queue.total
                    }
        
        logger.info(f"✓ Provisioned {provisioned} runner(s)")
        return results
    
    def check_and_scale(self) -> Dict[str, Any]:
        """
        Check workflow queues and scale runners.
        
        Returns:
            Dictionary with scaling results
        """
        try:
            logger.info("Checking workflow queues...")
            
            # Get workflow queues
            queues = self.get_workflow_queues()
            
            if not queues:
                logger.info("No repositories with active workflows")
                return {'queues': 0, 'workflows': 0, 'provisioned': 0}
            
            # Count workflows
            total_workflows = sum(q.total for q in queues)
            total_running = sum(q.running for q in queues)
            total_failed = sum(q.failed for q in queues)
            
            logger.info(f"Found {len(queues)} repos with {total_workflows} workflows")
            logger.info(f"  Running: {total_running}, Failed: {total_failed}")
            
            # Update status
            self.status.repositories_monitored = len(queues)
            self.status.queued_workflows = total_workflows
            self.status.last_check = datetime.now()
            
            # Provision runners
            results = self.provision_runners_for_queues(queues)
            
            # Update active runner count
            self.status.active_runners = len(list(self.active_containers))
            
            return {
                'queues': len(queues),
                'workflows': total_workflows,
                'running': total_running,
                'failed': total_failed,
                'provisioned': len(results),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error during check and scale: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _autoscaler_loop(self) -> None:
        """Internal autoscaler loop (runs in thread)."""
        logger.info("Autoscaler loop started")
        
        while self.running:
            try:
                self.status.iterations += 1
                logger.info(f"--- Check #{self.status.iterations} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
                
                # Check and scale
                self.check_and_scale()
                
                # Periodic cleanup
                if (datetime.now() - self.last_cleanup).total_seconds() >= 300:
                    removed = self.cleanup_exited_containers()
                    if removed > 0:
                        logger.info(f"Cleaned up {removed} exited container(s)")
                    self.last_cleanup = datetime.now()
                
                # Sleep
                if self.running:
                    logger.info(f"Sleeping for {self.config.poll_interval}s...")
                    time.sleep(self.config.poll_interval)
                    
            except Exception as e:
                logger.error(f"Error in autoscaler loop: {e}", exc_info=True)
                time.sleep(10)
        
        logger.info("Autoscaler loop stopped")
    
    def start_autoscaler(self, background: bool = True) -> bool:
        """
        Start the autoscaler.
        
        Args:
            background: Run in background thread
            
        Returns:
            True if started successfully
        """
        if self.running:
            logger.warning("Autoscaler is already running")
            return False
        
        logger.info("=" * 80)
        logger.info("GitHub Actions Runner Autoscaler Starting")
        logger.info("=" * 80)
        
        self.running = True
        self.status.running = True
        self.status.start_time = datetime.now()
        self.status.iterations = 0
        
        if background:
            # Start in background thread
            self.thread = threading.Thread(target=self._autoscaler_loop, daemon=True)
            self.thread.start()
            logger.info("Autoscaler started in background")
        else:
            # Run in foreground
            self._autoscaler_loop()
        
        return True
    
    def stop_autoscaler(self) -> bool:
        """
        Stop the autoscaler.
        
        Returns:
            True if stopped successfully
        """
        if not self.running:
            logger.warning("Autoscaler is not running")
            return False
        
        logger.info("Stopping autoscaler...")
        self.running = False
        self.status.running = False
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        logger.info("Autoscaler stopped")
        return True
    
    def get_status(self) -> AutoscalerStatus:
        """
        Get autoscaler status.
        
        Returns:
            AutoscalerStatus object
        """
        # Update active runner count
        self.status.active_runners = len(list(self.active_containers))
        return self.status


# Convenience functions

def get_runner_kit(config: Optional[RunnerConfig] = None) -> RunnerKit:
    """
    Get a RunnerKit instance.
    
    Args:
        config: Runner configuration
        
    Returns:
        RunnerKit instance
    """
    return RunnerKit(config=config)


__all__ = [
    'RunnerKit',
    'RunnerConfig',
    'WorkflowQueue',
    'RunnerStatus',
    'AutoscalerStatus',
    'get_runner_kit',
]
