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
            'scheduler': scheduler_status,
            'timestamp': time.time()
        }
        
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
