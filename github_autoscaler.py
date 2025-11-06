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
from typing import Optional, Dict

# Setup path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import from the correct location
try:
    from ipfs_accelerate_py.github_cli import GitHubCLI, WorkflowQueue, RunnerManager
except ImportError:
    # Try alternative import path
    from github_cli import GitHubCLI, WorkflowQueue, RunnerManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("github_autoscaler")


class GitHubRunnerAutoscaler:
    """
    Automatic GitHub Actions runner provisioning service.
    
    Monitors workflow queues and automatically provisions self-hosted
    runners based on demand when user is authenticated with GitHub CLI.
    """
    
    def __init__(
        self,
        owner: Optional[str] = None,
        poll_interval: int = 60,
        since_days: int = 1,
        max_runners: Optional[int] = None,
        filter_by_arch: bool = True
    ):
        """
        Initialize the autoscaler.
        
        Args:
            owner: GitHub owner (user or org) to monitor
            poll_interval: Seconds between checks (default: 60)
            since_days: Look at repos updated in last N days (default: 1)
            max_runners: Maximum runners to provision (default: system cores)
            filter_by_arch: Whether to filter workflows by architecture (default: True)
        """
        self.owner = owner
        self.poll_interval = poll_interval
        self.since_days = since_days
        self.max_runners = max_runners
        self.filter_by_arch = filter_by_arch
        self.running = False
        
        # Initialize GitHub CLI components
        try:
            # Use /usr/bin/gh (apt-installed) to avoid snap's privileged capabilities error under systemd
            import os
            gh_path = "/usr/bin/gh" if os.path.exists("/usr/bin/gh") else "gh"
            self.gh = GitHubCLI(gh_path=gh_path)
            self.queue_mgr = WorkflowQueue(self.gh)
            self.runner_mgr = RunnerManager(self.gh)
            logger.info("✓ GitHub CLI components initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize GitHub CLI: {e}")
            raise
        
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
        logger.info(f"  Docker isolation: enabled (see CONTAINERIZED_CI_SECURITY.md)")
    
    def _write_tokens_to_file(self, provisioning: Dict, system_arch: str, token_file: str = "/var/lib/github-runner-autoscaler/runner_tokens.json") -> None:
        """
        Write runner tokens to file for containerized launcher to consume.
        
        Args:
            provisioning: Dictionary of provisioning results from runner manager
            system_arch: System architecture (x64, arm64, etc.)
            token_file: Path to token file
        """
        import json
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
            try:
                data = {
                    "tokens": tokens,
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "architecture": system_arch
                }
                
                with open(token_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"✓ Wrote {len(tokens)} token(s) to {token_file}")
            except IOError as e:
                logger.error(f"Failed to write tokens to file: {e}")
    
    def check_and_scale(self) -> None:
        """
        Check workflow queues and provision runners if needed.
        """
        try:
            logger.info("Checking workflow queues...")
            
            # Get system architecture for filtering
            system_arch = self.runner_mgr.get_system_architecture()
            
            # Get workflow queues for repos with recent activity
            # Filter by architecture if enabled
            queues = self.queue_mgr.create_workflow_queues(
                owner=self.owner,
                since_days=self.since_days,
                system_arch=system_arch if self.filter_by_arch else None,
                filter_by_arch=self.filter_by_arch
            )
            
            if not queues:
                logger.info("No repositories with active workflows")
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
            
            # Check if we need to provision runners
            if total_running == 0 and total_failed == 0:
                logger.info("No workflows need runner provisioning")
                return
            
            # Provision runners for queues
            logger.info("Provisioning runners...")
            provisioning = self.runner_mgr.provision_runners_for_queue(
                queues,
                max_runners=self.max_runners
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
            filter_by_arch=not args.no_arch_filter
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
