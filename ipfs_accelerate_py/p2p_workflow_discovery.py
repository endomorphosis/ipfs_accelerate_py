#!/usr/bin/env python3
"""
P2P Workflow Discovery Service

This service discovers GitHub Actions workflows across repositories that are
tagged for P2P execution and submits them to the P2P workflow scheduler.

It monitors multiple repositories and detects workflows with tags like:
- p2p-only: Must be executed via P2P network
- p2p-eligible: Can be executed via P2P or GitHub
- code-generation, web-scraping, data-processing: Task type tags
"""

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any
from pathlib import Path


def _truthy_env(value: Optional[str], *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

# Try to import storage wrapper
try:
    from .common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        HAVE_STORAGE_WRAPPER = False
        def get_storage_wrapper(*args, **kwargs):
            return None

logger = logging.getLogger(__name__)

# Setup path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import P2P scheduler
try:
    from ipfs_accelerate_py.p2p_workflow_scheduler import (
        P2PWorkflowScheduler,
        P2PTask,
        WorkflowTag
    )
    HAVE_P2P_SCHEDULER = True
except ImportError:
    try:
        from p2p_workflow_scheduler import (
            P2PWorkflowScheduler,
            P2PTask,
            WorkflowTag
        )
        HAVE_P2P_SCHEDULER = True
    except ImportError as e:
        logger.warning(f"P2P workflow scheduler not available: {e}")
        HAVE_P2P_SCHEDULER = False

# Import GitHub CLI components
try:
    from ipfs_accelerate_py.github_cli import GitHubCLI, GitHubGraphQL
except ImportError:
    try:
        from github_cli import GitHubCLI, GitHubGraphQL
    except ImportError as e:
        logger.warning(f"GitHub CLI not available: {e}")
        GitHubCLI = None
        GitHubGraphQL = None


# Optional: integrate with the libp2p TaskQueue RPC layer.
try:
    from ipfs_accelerate_py.p2p_tasks.client import (
        RemoteQueue as TaskQueueRemote,
        submit_task_with_info as taskqueue_submit_task_with_info,
        claim_next as taskqueue_claim_next,
        complete_task as taskqueue_complete_task,
    )

    HAVE_TASKQUEUE_P2P = True
except Exception as e:
    HAVE_TASKQUEUE_P2P = False
    TaskQueueRemote = None  # type: ignore[assignment]
    taskqueue_submit_task_with_info = None  # type: ignore[assignment]
    taskqueue_claim_next = None  # type: ignore[assignment]
    taskqueue_complete_task = None  # type: ignore[assignment]
    logger.debug(f"TaskQueue p2p integration unavailable: {e}")


@dataclass
class WorkflowDiscovery:
    """Information about a discovered workflow"""
    owner: str
    repo: str
    workflow_id: str
    workflow_name: str
    workflow_path: str
    tags: List[str]
    run_id: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[float] = None


class P2PWorkflowDiscoveryService:
    """
    Service that discovers workflows across repositories and submits them
    to the P2P scheduler for execution.
    """
    
    def __init__(
        self,
        owner: Optional[str] = None,
        poll_interval: int = 300,
        scheduler: Optional[P2PWorkflowScheduler] = None
    ):
        """
        Initialize P2P workflow discovery service.
        
        Args:
            owner: GitHub owner (user/org) to monitor
            poll_interval: Seconds between discovery checks
            scheduler: Optional P2P scheduler instance (creates one if not provided)
        """
        self.owner = owner
        self.poll_interval = poll_interval
        self.running = False
        
        # Initialize storage wrapper for distributed storage
        self._storage = get_storage_wrapper() if HAVE_STORAGE_WRAPPER else None
        
        # Initialize GitHub CLI
        if GitHubCLI is None:
            raise RuntimeError("GitHub CLI not available")
        
        try:
            gh_path = "/usr/bin/gh" if os.path.exists("/usr/bin/gh") else "gh"
            self.gh = GitHubCLI(gh_path=gh_path)
            self.graphql = GitHubGraphQL(gh_path=gh_path) if GitHubGraphQL else None
            logger.info("✓ GitHub CLI initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize GitHub CLI: {e}")
            raise
        
        # Verify authentication
        auth_status = self.gh.get_auth_status()
        if not auth_status["authenticated"]:
            raise RuntimeError("GitHub CLI not authenticated. Run: gh auth login")
        
        logger.info("✓ Authenticated with GitHub")
        
        # Initialize or use provided scheduler
        if not HAVE_P2P_SCHEDULER:
            raise RuntimeError("P2P scheduler not available")
        
        if scheduler:
            self.scheduler = scheduler
        else:
            # Generate peer ID
            import socket
            import uuid
            peer_id = f"peer-discovery-{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
            self.scheduler = P2PWorkflowScheduler(peer_id=peer_id)
        
        logger.info(f"✓ P2P scheduler initialized (peer: {self.scheduler.peer_id})")
        
        # Track discovered workflows to avoid duplicates
        self.discovered_workflows: Set[str] = set()
        
        logger.info(f"P2P Workflow Discovery Service initialized")
        logger.info(f"  Owner: {self.owner or 'All accessible repos'}")
        logger.info(f"  Poll interval: {self.poll_interval}s")

        # Best-effort: a configured remote TaskQueue to publish to / consume from.
        self._taskqueue_remote = self._build_taskqueue_remote()

    def _build_taskqueue_remote(self):
        if not HAVE_TASKQUEUE_P2P or TaskQueueRemote is None:
            return None

        remote_peer_id = (
            os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_PEER_ID")
            or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_REMOTE_PEER_ID")
            or ""
        ).strip()
        remote_multiaddr = (
            os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_MULTIADDR")
            or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_REMOTE_MULTIADDR")
            or ""
        ).strip()

        # If nothing is explicitly configured, use an empty RemoteQueue which
        # triggers the client's dial chain (announce-file → bootstrap → rendezvous → dht → mdns).
        return TaskQueueRemote(peer_id=remote_peer_id, multiaddr=remote_multiaddr)

    def _taskqueue_enabled(self) -> bool:
        return _truthy_env(
            os.environ.get("IPFS_ACCELERATE_PY_P2P_WORKFLOW_TASKQUEUE", os.environ.get("IPFS_DATASETS_PY_P2P_WORKFLOW_TASKQUEUE")),
            default=True,
        )

    def _taskqueue_task_types(self) -> List[str]:
        return ["p2p-workflow-discovery", "p2p-workflow-scheduler-snapshot"]

    def _workflow_payload_from_discovery(self, discovery: WorkflowDiscovery, *, priority: int) -> Dict[str, Any]:
        return {
            "kind": "workflow_discovery",
            "source_peer": str(getattr(self.scheduler, "peer_id", "")),
            "owner": discovery.owner,
            "repo": discovery.repo,
            "workflow_id": discovery.workflow_id,
            "workflow_name": discovery.workflow_name,
            "workflow_path": discovery.workflow_path,
            "tags": list(discovery.tags or []),
            "priority": int(priority),
            "created_at": float(discovery.created_at or time.time()),
        }

    def _scheduler_snapshot_payload(self) -> Dict[str, Any]:
        # Keep payload compact but sufficient to converge on tasks + clocks.
        def _task_to_dict(t: P2PTask) -> Dict[str, Any]:
            return {
                "task_id": t.task_id,
                "workflow_id": t.workflow_id,
                "name": t.name,
                "tags": [getattr(tag, "value", str(tag)) for tag in (t.tags or [])],
                "priority": int(t.priority),
                "created_at": float(t.created_at),
                "task_hash": t.task_hash,
                "assigned_peer": t.assigned_peer,
            }

        return {
            "kind": "scheduler_snapshot",
            "peer_id": str(getattr(self.scheduler, "peer_id", "")),
            "timestamp": float(time.time()),
            "merkle_clock": self.scheduler.merkle_clock.to_dict() if hasattr(self.scheduler, "merkle_clock") else {},
            "pending_tasks": [_task_to_dict(t) for t in list(getattr(self.scheduler, "pending_tasks", {}).values())],
            "assigned_tasks": [_task_to_dict(t) for t in list(getattr(self.scheduler, "assigned_tasks", {}).values())],
            "completed_tasks": [_task_to_dict(t) for t in list(getattr(self.scheduler, "completed_tasks", {}).values())],
        }

    def _merge_workflow_payload_into_scheduler(self, payload: Dict[str, Any]) -> bool:
        try:
            owner = str(payload.get("owner") or "").strip()
            repo = str(payload.get("repo") or "").strip()
            workflow_name = str(payload.get("workflow_name") or "").strip()
            workflow_id = str(payload.get("workflow_id") or f"{owner}/{repo}").strip()
            tags_raw = payload.get("tags") or []
            priority = int(payload.get("priority") or 5)
            created_at = float(payload.get("created_at") or time.time())

            if not owner or not repo or not workflow_name:
                return False

            workflow_tags: List[WorkflowTag] = []
            if isinstance(tags_raw, (list, tuple, set)):
                for tag_str in tags_raw:
                    try:
                        enum_name = str(tag_str).upper().replace('-', '_')
                        workflow_tags.append(WorkflowTag[enum_name])
                    except Exception:
                        continue

            task_id = f"{owner}/{repo}/{workflow_name}"
            task = P2PTask(
                task_id=task_id,
                workflow_id=workflow_id,
                name=f"{workflow_name} ({owner}/{repo})",
                tags=workflow_tags,
                priority=max(1, min(10, int(priority))),
                created_at=created_at,
            )
            return bool(self.scheduler.submit_task(task))
        except Exception:
            return False

    def _merge_snapshot_payload_into_scheduler(self, payload: Dict[str, Any]) -> int:
        merged = 0
        try:
            from ipfs_accelerate_py.p2p_workflow_scheduler import MerkleClock

            peer_id = str(payload.get("peer_id") or "").strip()
            if not peer_id or peer_id == getattr(self.scheduler, "peer_id", ""):
                return 0

            clock_dict = payload.get("merkle_clock")
            if isinstance(clock_dict, dict) and clock_dict.get("node_id"):
                try:
                    clock = MerkleClock.from_dict(clock_dict)
                    self.scheduler.update_peer_state(peer_id, clock)
                except Exception:
                    pass

            tasks = payload.get("pending_tasks") or []
            if isinstance(tasks, (list, tuple)):
                for t in tasks:
                    if not isinstance(t, dict):
                        continue
                    # Reuse merge logic by mapping into a workflow payload-like dict.
                    wf_id = str(t.get("workflow_id") or "").strip()
                    name = str(t.get("name") or "").strip()
                    # name is "workflow (owner/repo)"; we also may not have owner/repo.
                    # Fall back to decoding from task_id.
                    task_id = str(t.get("task_id") or "").strip()
                    owner = ""
                    repo = ""
                    wf_name = ""
                    if task_id.count("/") >= 2:
                        owner, repo, wf_name = task_id.split("/", 2)
                    if not wf_name and name:
                        wf_name = name.split("(", 1)[0].strip()

                    if owner and repo and wf_name:
                        ok = self._merge_workflow_payload_into_scheduler(
                            {
                                "owner": owner,
                                "repo": repo,
                                "workflow_id": wf_id or f"{owner}/{repo}",
                                "workflow_name": wf_name,
                                "tags": t.get("tags") or [],
                                "priority": t.get("priority") or 5,
                                "created_at": t.get("created_at") or time.time(),
                            }
                        )
                        if ok:
                            merged += 1
        except Exception:
            return merged
        return merged

    async def _taskqueue_submit(self, *, task_type: str, payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
        if not (HAVE_TASKQUEUE_P2P and self._taskqueue_enabled() and self._taskqueue_remote is not None):
            return None
        if taskqueue_submit_task_with_info is None:
            return None
        try:
            return await taskqueue_submit_task_with_info(remote=self._taskqueue_remote, task_type=task_type, model_name="", payload=payload)
        except Exception as e:
            logger.debug(f"TaskQueue submit failed ({task_type}): {e}")
            return None

    async def _taskqueue_drain(self, *, max_tasks: int = 50) -> int:
        if not (HAVE_TASKQUEUE_P2P and self._taskqueue_enabled() and self._taskqueue_remote is not None):
            return 0
        if taskqueue_claim_next is None or taskqueue_complete_task is None:
            return 0

        drained = 0
        worker_id = str(getattr(self.scheduler, "peer_id", "workflow-discovery"))
        for _ in range(max(1, int(max_tasks))):
            try:
                task = await taskqueue_claim_next(
                    remote=self._taskqueue_remote,
                    worker_id=worker_id,
                    supported_task_types=self._taskqueue_task_types(),
                )
            except Exception as e:
                logger.debug(f"TaskQueue claim failed: {e}")
                return drained

            if task is None:
                return drained

            drained += 1
            task_id = str(task.get("task_id") or "")
            task_type = str(task.get("task_type") or "")
            payload = task.get("payload") if isinstance(task.get("payload"), dict) else {}

            try:
                if task_type == "p2p-workflow-discovery":
                    merged = bool(self._merge_workflow_payload_into_scheduler(payload))
                    await taskqueue_complete_task(
                        remote=self._taskqueue_remote,
                        task_id=task_id,
                        status="completed" if merged else "failed",
                        result={"merged": bool(merged)},
                        error=None if merged else "invalid_or_duplicate",
                    )
                elif task_type == "p2p-workflow-scheduler-snapshot":
                    merged_count = int(self._merge_snapshot_payload_into_scheduler(payload))
                    await taskqueue_complete_task(
                        remote=self._taskqueue_remote,
                        task_id=task_id,
                        status="completed",
                        result={"merged_tasks": merged_count},
                        error=None,
                    )
                else:
                    await taskqueue_complete_task(
                        remote=self._taskqueue_remote,
                        task_id=task_id,
                        status="completed",
                        result={"ignored": True},
                        error=None,
                    )
            except Exception as e:
                try:
                    await taskqueue_complete_task(
                        remote=self._taskqueue_remote,
                        task_id=task_id,
                        status="failed",
                        result=None,
                        error=str(e),
                    )
                except Exception:
                    pass

        return drained
    
    def _parse_workflow_tags(self, workflow_content: str) -> List[str]:
        """
        Parse workflow file content to extract P2P tags.
        
        Looks for tags in:
        - env.WORKFLOW_TAGS
        - env.P2P_TAGS
        - comments like: # P2P: p2p-only, code-generation
        
        Args:
            workflow_content: YAML workflow file content
        
        Returns:
            List of tag strings
        """
        tags = []
        
        # Look for WORKFLOW_TAGS or P2P_TAGS in env section
        tag_patterns = [
            r'WORKFLOW_TAGS:\s*["\']?([^"\'\n]+)["\']?',
            r'P2P_TAGS:\s*["\']?([^"\'\n]+)["\']?',
            r'#\s*P2P:\s*([^\n]+)',
            r'#\s*Tags:\s*([^\n]+)'
        ]
        
        for pattern in tag_patterns:
            matches = re.findall(pattern, workflow_content, re.IGNORECASE)
            for match in matches:
                # Split by comma and clean up
                tag_list = [tag.strip() for tag in match.split(',')]
                tags.extend(tag_list)
        
        # Deduplicate and filter valid tags
        valid_tags = set()
        for tag in tags:
            tag = tag.lower().strip()
            if tag and tag in [
                'p2p-only', 'p2p-eligible', 'github-api',
                'code-generation', 'web-scraping', 'data-processing',
                'unit-test'
            ]:
                valid_tags.add(tag)
        
        return list(valid_tags)
    
    def _should_process_workflow(self, tags: List[str]) -> bool:
        """
        Determine if workflow should be processed by P2P scheduler.
        
        Args:
            tags: List of workflow tags
        
        Returns:
            True if workflow should be processed via P2P
        """
        if not tags:
            return False
        
        # Convert string tags to WorkflowTag enums
        workflow_tags = []
        for tag_str in tags:
            try:
                enum_name = tag_str.upper().replace('-', '_')
                workflow_tags.append(WorkflowTag[enum_name])
            except (KeyError, AttributeError):
                pass
        
        return self.scheduler.should_bypass_github(workflow_tags)
    
    def discover_workflows_in_repo(self, owner: str, repo: str) -> List[WorkflowDiscovery]:
        """
        Discover P2P workflows in a specific repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
        
        Returns:
            List of discovered workflows
        """
        discoveries = []
        
        try:
            # Get workflow files from .github/workflows/
            workflow_dir = f"{owner}/{repo}/.github/workflows"
            logger.debug(f"Checking workflows in {workflow_dir}")
            
            # List workflow files using GitHub API
            try:
                # Try to get workflow files
                result = self.gh.api(
                    f"repos/{owner}/{repo}/contents/.github/workflows",
                    method="GET"
                )
                
                if not result or not isinstance(result, list):
                    logger.debug(f"No workflows found in {owner}/{repo}")
                    return discoveries
                
                # Process each workflow file
                for file_info in result:
                    if not file_info.get('name', '').endswith(('.yml', '.yaml')):
                        continue
                    
                    workflow_path = file_info.get('path', '')
                    workflow_name = file_info.get('name', '')
                    
                    # Get workflow file content
                    try:
                        content_result = self.gh.api(
                            f"repos/{owner}/{repo}/contents/{workflow_path}",
                            method="GET"
                        )
                        
                        if not content_result:
                            continue
                        
                        # Decode content (base64)
                        import base64
                        content_b64 = content_result.get('content', '')
                        if content_b64:
                            content = base64.b64decode(content_b64).decode('utf-8')
                            
                            # Parse tags from workflow content
                            tags = self._parse_workflow_tags(content)
                            
                            if self._should_process_workflow(tags):
                                discovery = WorkflowDiscovery(
                                    owner=owner,
                                    repo=repo,
                                    workflow_id=workflow_path,
                                    workflow_name=workflow_name,
                                    workflow_path=workflow_path,
                                    tags=tags,
                                    created_at=time.time()
                                )
                                discoveries.append(discovery)
                                logger.info(f"✓ Discovered P2P workflow: {owner}/{repo}/{workflow_name} (tags: {', '.join(tags)})")
                    
                    except Exception as e:
                        logger.debug(f"Error reading workflow {workflow_path}: {e}")
                        continue
            
            except Exception as e:
                logger.debug(f"Error listing workflows in {owner}/{repo}: {e}")
        
        except Exception as e:
            logger.error(f"Error discovering workflows in {owner}/{repo}: {e}")
        
        return discoveries
    
    def discover_workflows(self) -> List[WorkflowDiscovery]:
        """
        Discover P2P workflows across all accessible repositories.
        
        Returns:
            List of discovered workflows
        """
        discoveries = []
        
        try:
            # Get list of repositories
            repos = self.gh.list_repos(owner=self.owner, limit=100)
            
            if not repos:
                logger.warning("No repositories found")
                return discoveries
            
            logger.info(f"Scanning {len(repos)} repositories for P2P workflows...")
            
            # Check each repository for P2P workflows
            for repo in repos:
                owner = repo['owner']['login']
                repo_name = repo['name']
                
                repo_discoveries = self.discover_workflows_in_repo(owner, repo_name)
                discoveries.extend(repo_discoveries)
            
            logger.info(f"✓ Discovered {len(discoveries)} P2P workflows")
        
        except Exception as e:
            logger.error(f"Error discovering workflows: {e}", exc_info=True)
        
        return discoveries
    
    def submit_workflow_to_scheduler(self, discovery: WorkflowDiscovery) -> bool:
        """
        Submit a discovered workflow to the P2P scheduler.
        
        Args:
            discovery: Workflow discovery information
        
        Returns:
            True if successfully submitted
        """
        try:
            # Create unique task ID
            task_id = f"{discovery.owner}/{discovery.repo}/{discovery.workflow_name}"
            
            # Check if already discovered
            if task_id in self.discovered_workflows:
                logger.debug(f"Workflow already submitted: {task_id}")
                return False
            
            # Convert string tags to WorkflowTag enums
            workflow_tags = []
            for tag_str in discovery.tags:
                try:
                    enum_name = tag_str.upper().replace('-', '_')
                    workflow_tags.append(WorkflowTag[enum_name])
                except (KeyError, AttributeError):
                    logger.warning(f"Unknown tag: {tag_str}")
            
            # Determine priority based on tags
            priority = 5  # Default
            if 'p2p-only' in discovery.tags:
                priority = 8  # High priority for P2P-only workflows
            elif 'p2p-eligible' in discovery.tags:
                priority = 6  # Medium-high priority
            
            # Create task
            task = P2PTask(
                task_id=task_id,
                workflow_id=f"{discovery.owner}/{discovery.repo}",
                name=f"{discovery.workflow_name} ({discovery.owner}/{discovery.repo})",
                tags=workflow_tags,
                priority=priority,
                created_at=discovery.created_at or time.time()
            )
            
            # Submit to scheduler
            success = self.scheduler.submit_task(task)
            
            if success:
                self.discovered_workflows.add(task_id)
                logger.info(f"✓ Submitted workflow to P2P scheduler: {task_id} (priority: {priority})")

                # Task A: also submit to TaskQueue over libp2p as a portable payload.
                if HAVE_TASKQUEUE_P2P and self._taskqueue_enabled() and self._taskqueue_remote is not None:
                    try:
                        import anyio

                        async def _do_submit() -> None:
                            payload = self._workflow_payload_from_discovery(discovery, priority=priority)
                            await self._taskqueue_submit(task_type="p2p-workflow-discovery", payload=payload)

                        anyio.run(_do_submit, backend="trio")
                    except Exception as e:
                        logger.debug(f"Failed to submit workflow to TaskQueue: {e}")
                return True
            else:
                logger.warning(f"Failed to submit workflow: {task_id}")
                return False
        
        except Exception as e:
            logger.error(f"Error submitting workflow to scheduler: {e}", exc_info=True)
            return False
    
    def run_discovery_cycle(self) -> Dict[str, Any]:
        """
        Run one discovery cycle.
        
        Returns:
            Statistics about the discovery cycle
        """
        logger.info("Starting discovery cycle...")

        # Task B (pull): consume any inbound workflow/snapshot messages first.
        drained = 0
        if HAVE_TASKQUEUE_P2P and self._taskqueue_enabled() and self._taskqueue_remote is not None:
            try:
                import anyio

                async def _do_drain() -> int:
                    return await self._taskqueue_drain(max_tasks=50)

                drained = int(anyio.run(_do_drain, backend="trio"))
                if drained:
                    logger.info(f"✓ Merged {drained} inbound TaskQueue message(s)")
            except Exception as e:
                logger.debug(f"TaskQueue drain failed: {e}")
        
        # Discover workflows
        discoveries = self.discover_workflows()
        
        # Submit to scheduler
        submitted = 0
        for discovery in discoveries:
            if self.submit_workflow_to_scheduler(discovery):
                submitted += 1
        
        # Get scheduler status
        scheduler_status = self.scheduler.get_status()
        
        stats = {
            'discovered': len(discoveries),
            'submitted': submitted,
            'taskqueue_drained': drained,
            'scheduler': scheduler_status,
            'timestamp': time.time()
        }

        # Task B (push): publish a scheduler snapshot as a TaskQueue task.
        if HAVE_TASKQUEUE_P2P and self._taskqueue_enabled() and self._taskqueue_remote is not None:
            try:
                import anyio

                async def _do_snap() -> None:
                    await self._taskqueue_submit(task_type="p2p-workflow-scheduler-snapshot", payload=self._scheduler_snapshot_payload())

                anyio.run(_do_snap, backend="trio")
            except Exception as e:
                logger.debug(f"TaskQueue snapshot publish failed: {e}")
        
        # Try to store discovery stats in distributed storage
        if self._storage and hasattr(self._storage, 'is_distributed') and self._storage.is_distributed:
            try:
                cache_key = f"p2p_discovery_stats_{int(time.time())}"
                self._storage.write_file(json.dumps(stats, indent=2), cache_key, pin=False)
                logger.debug(f"Stored discovery stats in distributed storage: {cache_key}")
            except Exception as e:
                logger.debug(f"Failed to store discovery stats: {e}")
        
        logger.info(f"Discovery cycle complete: {len(discoveries)} discovered, {submitted} submitted")
        
        return stats
    
    def start(self) -> None:
        """Start the discovery service (runs continuously)."""
        self.running = True
        logger.info("=" * 80)
        logger.info("P2P Workflow Discovery Service Started")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Monitoring repositories for P2P workflows...")
        logger.info("Press Ctrl+C to stop")
        logger.info("")
        
        iteration = 0
        while self.running:
            iteration += 1
            logger.info(f"--- Discovery Cycle #{iteration} ---")
            
            try:
                stats = self.run_discovery_cycle()
                logger.info(f"Status: {stats['discovered']} discovered, {stats['submitted']} submitted")
                logger.info(f"Scheduler: {stats['scheduler']['pending_tasks']} pending, "
                          f"{stats['scheduler']['assigned_tasks']} assigned, "
                          f"{stats['scheduler']['completed_tasks']} completed")
            except Exception as e:
                logger.error(f"Error in discovery cycle: {e}", exc_info=True)
            
            if self.running:
                logger.info(f"Sleeping for {self.poll_interval}s...")
                logger.info("")
                time.sleep(self.poll_interval)
        
        logger.info("Discovery service stopped")
    
    def stop(self) -> None:
        """Stop the discovery service."""
        logger.info("Stopping discovery service...")
        self.running = False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="P2P Workflow Discovery Service - Discover and schedule P2P workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--owner',
        type=str,
        help='GitHub owner (user or org) to monitor'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Poll interval in seconds (default: 300)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        service = P2PWorkflowDiscoveryService(
            owner=args.owner,
            poll_interval=args.interval
        )
        
        service.start()
        return 0
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Failed to start discovery service: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
