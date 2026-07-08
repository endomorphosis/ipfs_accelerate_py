#!/usr/bin/env python3
"""
GitHub Actions Auto-Scaling Runner Service

This service automatically monitors GitHub workflows and provisions
self-hosted runners as needed based on workflow queue depth.

Usage:
    python github_autoscaler.py                    # Start in foreground
    python github_autoscaler.py --daemon           # Run as daemon
    python github_autoscaler.py --owner myorg      # Monitor specific org
"""

import sys
import os
import time
import logging
import signal
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Iterable, List, Tuple

# Setup path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import from the correct location
try:
    from ipfs_accelerate_py.github_cli import GitHubCLI, WorkflowQueue, RunnerManager, GitHubGraphQL
except ImportError:
    # Try alternative import path
    from github_cli import GitHubCLI, WorkflowQueue, RunnerManager, GitHubGraphQL

# Import P2P components
try:
    from ipfs_accelerate_py.p2p_workflow_scheduler import P2PWorkflowScheduler
    from ipfs_accelerate_py.p2p_workflow_discovery import P2PWorkflowDiscoveryService
    HAVE_P2P_SCHEDULER = True
except ImportError:
    try:
        from p2p_workflow_scheduler import P2PWorkflowScheduler
        from p2p_workflow_discovery import P2PWorkflowDiscoveryService
        HAVE_P2P_SCHEDULER = True
    except ImportError:
        HAVE_P2P_SCHEDULER = False
        logger.warning("P2P workflow scheduler not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("github_autoscaler")


def _repo_root() -> Path:
    # scripts/utils/<this file> -> parents[2] is repo root
    return Path(__file__).resolve().parents[2]


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen: set[str] = set()
    result: List[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _candidate_token_files(explicit_token_file: Optional[str]) -> List[Path]:
    env_token_file = (
        os.environ.get("GITHUB_RUNNER_AUTOSCALER_TOKEN_FILE")
        or os.environ.get("GITHUB_AUTOSCALER_TOKEN_FILE")
    )
    if env_token_file:
        return [Path(env_token_file)]

    candidates: List[Path] = []
    if explicit_token_file:
        candidates.append(Path(explicit_token_file))

    candidates.append(Path("/var/lib/github-runner-autoscaler/runner_tokens.json"))

    state_dir = (
        os.environ.get("GITHUB_RUNNER_AUTOSCALER_STATE_DIR")
        or os.environ.get("GITHUB_AUTOSCALER_STATE_DIR")
    )
    if state_dir:
        candidates.append(Path(state_dir) / "runner_tokens.json")

    ipfs_cache_dir = os.environ.get("IPFS_ACCELERATE_CACHE_DIR")
    if ipfs_cache_dir:
        candidates.append(Path(ipfs_cache_dir) / "runner_tokens.json")

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        candidates.append(Path(xdg_cache) / "ipfs_accelerate" / "runner_tokens.json")

    candidates.append(_repo_root() / ".cache" / "ipfs_accelerate" / "runner_tokens.json")
    candidates.append(Path("/tmp") / "ipfs_accelerate" / "runner_tokens.json")
    return _dedupe_paths(candidates)


def _write_json_atomic(path: Path, data: Dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            # Best-effort on filesystems that don't support chmod.
            pass
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


class GitHubRunnerAutoscaler:
    """
    Automatic GitHub Actions runner provisioning service.
    
    Monitors workflow queues and automatically provisions self-hosted
    runners based on demand when user is authenticated with GitHub CLI.
    """
    
    def __init__(
        self,
        owner: Optional[str] = None,
        poll_interval: int = 120,
        since_days: int = 1,
        max_runners: Optional[int] = None,
        filter_by_arch: bool = True,
        enable_p2p: bool = True
    ):
        """
        Initialize the autoscaler.
        
        Args:
            owner: GitHub owner (user or org) to monitor
            poll_interval: Seconds between checks (default: 120)
            since_days: Look at repos updated in last N days (default: 1)
            max_runners: Maximum runners to provision (default: system cores)
            filter_by_arch: Whether to filter workflows by architecture (default: True)
            enable_p2p: Enable P2P workflow monitoring and autoscaling (default: True)
        """
        self.owner = owner
        self.poll_interval = poll_interval
        self.since_days = since_days
        self.max_runners = max_runners
        self.filter_by_arch = filter_by_arch
        self.enable_p2p = enable_p2p and HAVE_P2P_SCHEDULER
        self.running = False
        
        # Initialize GitHub CLI components
        try:
            # Use /usr/bin/gh (apt-installed) to avoid snap's privileged capabilities error under systemd
            import os
            gh_path = "/usr/bin/gh" if os.path.exists("/usr/bin/gh") else "gh"
            self.gh = GitHubCLI(gh_path=gh_path)
            self.queue_mgr = WorkflowQueue(self.gh)
            self.runner_mgr = RunnerManager(self.gh)
            self.graphql = GitHubGraphQL(gh_path=gh_path)  # Fallback for rate limits
            logger.info("✓ GitHub CLI components initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize GitHub CLI: {e}")
            raise
        
        # Note: Distributed P2P cache is automatically enabled in GitHubCLI
        # No separate initialization needed - it's built into the cache layer
        
        # Initialize P2P workflow discovery if enabled
        self.p2p_discovery = None
        if self.enable_p2p:
            try:
                # Create P2P scheduler
                import socket
                import uuid
                peer_id = f"peer-autoscaler-{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
                p2p_scheduler = P2PWorkflowScheduler(peer_id=peer_id)
                
                # Create discovery service
                self.p2p_discovery = P2PWorkflowDiscoveryService(
                    owner=self.owner,
                    poll_interval=self.poll_interval,
                    scheduler=p2p_scheduler
                )
                logger.info("✓ P2P workflow discovery enabled")
            except Exception as e:
                logger.warning(f"✗ Failed to initialize P2P discovery: {e}")
                self.enable_p2p = False
        
        # Verify authentication
        auth_status = self.gh.get_auth_status()
        if not auth_status["authenticated"]:
            logger.error("✗ Not authenticated with GitHub CLI")
            logger.error("  Run: gh auth login")
            raise RuntimeError("GitHub CLI not authenticated")
        
        logger.info("✓ Authenticated with GitHub")
        
        # Get system capacity and architecture
        if self.max_runners is None:
            self.max_runners = self.runner_mgr.get_system_cores()
        
        system_arch = self.runner_mgr.get_system_architecture()
        runner_labels = self.runner_mgr.get_runner_labels()
        
        logger.info(f"Auto-scaler configured:")
        logger.info(f"  Owner: {self.owner or 'All accessible repos'}")
        logger.info(f"  Poll interval: {self.poll_interval}s")
        logger.info(f"  Max runners: {self.max_runners}")
        logger.info(f"  Monitor window: {self.since_days} day(s)")
        logger.info(f"  System architecture: {system_arch}")
        logger.info(f"  Runner labels: {runner_labels}")
        logger.info(f"  Architecture filtering: {'enabled' if filter_by_arch else 'disabled'}")
        logger.info(f"  P2P workflow monitoring: {'enabled' if self.enable_p2p else 'disabled'}")
        logger.info(f"  Docker isolation: enabled (see CONTAINERIZED_CI_SECURITY.md)")
    
    def _write_tokens_to_file(self, provisioning: Dict, system_arch: str, token_file: str = "/var/lib/github-runner-autoscaler/runner_tokens.json") -> None:
        """
        Write runner tokens to file for containerized launcher to consume.
        
        Args:
            provisioning: Dictionary of provisioning results from runner manager
            system_arch: System architecture (x64, arm64, etc.)
            token_file: Path to token file
        """
        from datetime import datetime
        
        tokens = []
        for repo, status in provisioning.items():
            if status.get("status") == "token_generated":
                # Get labels from runner manager
                labels = self.runner_mgr.get_runner_labels()
                
                # Get the number of runners to provision for this repo
                runners_to_provision = status.get("runners_to_provision", 1)
                
                # Create multiple token entries (same token, reused by multiple runners)
                for i in range(runners_to_provision):
                    tokens.append({
                        "repo": repo,
                        "token": status["token"],
                        "labels": labels,
                        "workflow_count": status.get("total_workflows", 0),
                        "running": status.get("running", 0),
                        "failed": status.get("failed", 0),
                        "architecture": system_arch,
                        "runner_number": i + 1,
                        "total_runners_for_repo": runners_to_provision,
                        "created_at": datetime.utcnow().isoformat() + "Z"
                    })

        
        if tokens:
            data = {
                "tokens": tokens,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "architecture": system_arch,
            }

            candidates = _candidate_token_files(token_file)
            errors: List[Tuple[Path, Exception]] = []
            for candidate in candidates:
                try:
                    _write_json_atomic(candidate, data)
                    if str(candidate) != str(token_file):
                        logger.warning(
                            "Token file '%s' not writable; wrote tokens to '%s'. "
                            "Set GITHUB_RUNNER_AUTOSCALER_TOKEN_FILE to control this.",
                            token_file,
                            str(candidate),
                        )
                    else:
                        logger.info(f"✓ Wrote {len(tokens)} token(s) to {token_file}")
                    return
                except OSError as e:
                    errors.append((candidate, e))
                    continue

            # If we get here, every candidate failed.
            last_path, last_error = errors[-1] if errors else (Path(token_file), RuntimeError("unknown error"))
            logger.error(
                "Failed to write runner tokens to any token file location. Last attempt '%s': %s",
                str(last_path),
                last_error,
            )
    
    def _get_queues_via_graphql(self, system_arch: Optional[str] = None) -> Dict:
        """
        Get workflow queues using GraphQL API (fallback when REST is rate-limited).
        
        Args:
            system_arch: System architecture to filter by
            
        Returns:
            Dict of repo -> workflows
        """
        logger.info("Fetching workflow queues via GraphQL...")
        queues = {}
        
        try:
            # For now, we need a specific repo since we can't list all repos via GraphQL easily
            # This is a limitation - GraphQL fallback works best when you know which repo to query
            if self.owner:
                # Try common repo names or get from cache
                test_repos = ["ipfs_accelerate_py", "main", "runner"]
                for repo_name in test_repos:
                    result = self.graphql.list_workflow_runs(
                        owner=self.owner,
                        repo=repo_name,
                        status="in_progress",
                        limit=50
                    )
                    if result.get("success"):
                        workflows = result.get("data", {}).get("workflow_runs", [])
                        if workflows:
                            queues[f"{self.owner}/{repo_name}"] = workflows
        except Exception as e:
            logger.error(f"GraphQL fallback failed: {e}")
        
        return queues
    
    def check_and_scale(self) -> None:
        """
        Check workflow queues and provision runners if needed.
        Uses GraphQL API as fallback if REST API is rate-limited.
        Also monitors P2P workflow scheduler for pending tasks.
        """
        try:
            logger.info("Checking workflow queues...")
            
            # Check P2P scheduler if enabled
            p2p_pending = 0
            p2p_assigned = 0
            if self.enable_p2p and self.p2p_discovery:
                try:
                    # Run discovery cycle to find new P2P workflows
                    p2p_stats = self.p2p_discovery.run_discovery_cycle()
                    p2p_pending = p2p_stats['scheduler']['pending_tasks']
                    p2p_assigned = p2p_stats['scheduler']['assigned_tasks']
                    
                    if p2p_pending > 0 or p2p_assigned > 0:
                        logger.info(f"P2P workflows: {p2p_pending} pending, {p2p_assigned} assigned")
                except Exception as e:
                    logger.warning(f"Error checking P2P scheduler: {e}")
            
            # Get system architecture for filtering
            system_arch = self.runner_mgr.get_system_architecture()
            
            # Get workflow queues for repos with recent activity
            # Filter by architecture if enabled
            try:
                queues = self.queue_mgr.create_workflow_queues(
                    owner=self.owner,
                    since_days=self.since_days,
                    system_arch=system_arch if self.filter_by_arch else None,
                    filter_by_arch=self.filter_by_arch
                )
            except Exception as e:
                # Check if it's a rate limit error
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "403" in error_msg:
                    logger.warning("REST API rate limited, using GraphQL API as fallback")
                    queues = self._get_queues_via_graphql(system_arch)
                else:
                    raise
            
            if not queues:
                logger.info("No repositories with active workflows")
                # Still get list of repositories to maintain minimum runners
                repos = self.gh.list_repos(owner=self.owner, limit=30)
                if repos:
                    # Create empty queues for repos to ensure minimum 1 runner each
                    # Note: list_repos returns {name, owner:{login}, url, updatedAt}
                    queues = {f"{repo['owner']['login']}/{repo['name']}": [] for repo in repos}
                    logger.info(f"Maintaining minimum runners for {len(queues)} repository(ies)")
                else:
                    return
            
            # Count workflows needing attention
            total_workflows = sum(len(workflows) for workflows in queues.values())
            total_running = sum(
                sum(1 for w in workflows if w.get("status") == "in_progress")
                for workflows in queues.values()
            )
            total_failed = sum(
                sum(1 for w in workflows if w.get("conclusion") in ["failure", "timed_out", "cancelled"])
                for workflows in queues.values()
            )
            
            logger.info(f"Found {len(queues)} repos with {total_workflows} workflows")
            logger.info(f"  Running: {total_running}, Failed: {total_failed}")
            if self.filter_by_arch:
                logger.info(f"  (Filtered for {system_arch} architecture)")
            
            # Add P2P tasks to the total workload for provisioning calculation
            total_workload = total_workflows + p2p_pending + p2p_assigned
            
            # Check if we need to provision runners
            # Always provision at least 1 runner per repo for availability
            if total_running == 0 and total_failed == 0 and total_workflows > 0:
                logger.info("No workflows need runner provisioning, but maintaining base runners")
            elif total_workflows == 0 and p2p_pending == 0:
                logger.info("Provisioning minimum 1 runner per repository for availability")
            
            # Calculate runners needed for P2P tasks
            # Each P2P task may need a runner
            p2p_runners_needed = min(p2p_pending, self.max_runners // 2) if p2p_pending > 0 else 0
            
            # Adjust max_runners to account for P2P workload
            effective_max_runners = max(1, self.max_runners - p2p_runners_needed)
            
            if p2p_runners_needed > 0:
                logger.info(f"Allocating {p2p_runners_needed} runners for P2P tasks, "
                          f"{effective_max_runners} for GitHub workflows")
            
            # Provision runners for queues (minimum 1 per repo, or 1+queued workflows when active)
            logger.info("Provisioning runners...")
            provisioning = self.runner_mgr.provision_runners_for_queue(
                queues,
                max_runners=effective_max_runners,
                min_runners_per_repo=1  # Guarantee at least 1 runner per repo
            )
            
            # Report results
            success_count = sum(
                1 for status in provisioning.values()
                if status.get("status") == "token_generated"
            )
            
            if success_count > 0:
                logger.info(f"✓ Generated {success_count} runner token(s)")
                
                # Write tokens to file for containerized launcher
                self._write_tokens_to_file(provisioning, system_arch)
                
                for repo, status in provisioning.items():
                    if status.get("status") == "token_generated":
                        logger.info(f"  {repo}: {status['total_workflows']} workflows")
                        logger.info(f"    Token (first 20 chars): {status['token'][:20]}...")
                        logger.info(f"    Note: Runners will use Docker containers for isolation")
            else:
                logger.warning("No runners were provisioned")
            
            # Log P2P scheduler status summary
            if self.enable_p2p and (p2p_pending > 0 or p2p_assigned > 0):
                logger.info(f"P2P Summary: {p2p_pending} pending, {p2p_assigned} assigned, "
                          f"{p2p_runners_needed} runners allocated for P2P")
            
        except Exception as e:
            logger.error(f"Error during check and scale: {e}", exc_info=True)
    
    def start(self, setup_signals=True) -> None:
        """
        Start the autoscaler service.
        
        Args:
            setup_signals: Whether to set up signal handlers (default: True).
                          Set to False when running in a background thread.
        """
        self.running = True
        logger.info("=" * 80)
        logger.info("GitHub Actions Runner Autoscaler Started")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Monitoring for workflow queues and auto-provisioning runners...")
        if setup_signals:
            logger.info("Press Ctrl+C to stop")
        logger.info("")
        
        # Set up signal handlers only if running in main thread
        if setup_signals:
            try:
                signal.signal(signal.SIGINT, self._signal_handler)
                signal.signal(signal.SIGTERM, self._signal_handler)
            except ValueError:
                # Not in main thread, skip signal handlers
                logger.debug("Not in main thread, skipping signal handlers")
        
        # Main loop
        iteration = 0
        while self.running:
            iteration += 1
            logger.info(f"--- Check #{iteration} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
            
            try:
                self.check_and_scale()
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
            
            if self.running:
                logger.info(f"Sleeping for {self.poll_interval}s...")
                logger.info("")
                time.sleep(self.poll_interval)
        
        logger.info("Autoscaler stopped")
    
    def stop(self) -> None:
        """
        Stop the autoscaler service.
        """
        logger.info("Stopping autoscaler...")
        self.running = False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}")
        self.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GitHub Actions Runner Autoscaler - Automatically provision runners based on workflow demand",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start monitoring all accessible repos
  python github_autoscaler.py
  
  # Monitor specific organization
  python github_autoscaler.py --owner myorg
  
  # Custom poll interval and max runners
  python github_autoscaler.py --interval 30 --max-runners 8
  
  # Monitor repos updated in last 2 days
  python github_autoscaler.py --since-days 2

Requirements:
  - GitHub CLI (gh) must be installed
  - Must be authenticated: gh auth login
  - Will auto-provision runners as workflows are detected
        """
    )
    
    parser.add_argument(
        '--owner',
        type=str,
        help='GitHub owner (user or org) to monitor (default: all accessible repos)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Poll interval in seconds (default: 60)'
    )
    parser.add_argument(
        '--since-days',
        type=int,
        default=1,
        help='Monitor repos updated in last N days (default: 1)'
    )
    parser.add_argument(
        '--max-runners',
        type=int,
        help='Maximum runners to provision (default: system CPU cores)'
    )
    parser.add_argument(
        '--no-arch-filter',
        action='store_true',
        help='Disable architecture-based workflow filtering (provision for all workflows)'
    )
    parser.add_argument(
        '--no-p2p',
        action='store_true',
        help='Disable P2P workflow monitoring and autoscaling'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as background daemon (not yet implemented)'
    )
    
    args = parser.parse_args()
    
    if args.daemon:
        print("Daemon mode not yet implemented. Run in foreground for now.")
        return 1
    
    try:
        # Create and start autoscaler
        autoscaler = GitHubRunnerAutoscaler(
            owner=args.owner,
            poll_interval=args.interval,
            since_days=args.since_days,
            max_runners=args.max_runners,
            filter_by_arch=not args.no_arch_filter,
            enable_p2p=not args.no_p2p
        )
        
        autoscaler.start()
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Failed to start autoscaler: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
