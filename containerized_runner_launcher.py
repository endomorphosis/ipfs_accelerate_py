#!/usr/bin/env python3
"""
Containerized GitHub Actions Runner Launcher

This service consumes registration tokens from the autoscaler and launches
ephemeral Docker containers running GitHub Actions runners with proper isolation.
Each container is read-only, uses security constraints, and is automatically
cleaned up after job completion.
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DockerRunnerLauncher:
    """Launches and manages ephemeral Docker-based GitHub Actions runners."""
    
    def __init__(
        self,
        token_file: str = "/var/lib/github-runner-autoscaler/runner_tokens.json",
        runner_image: str = "myoung34/github-runner:latest",
        max_containers: int = 10,
        cleanup_interval: int = 300,
        runner_work_dir: str = "/tmp/_work"
    ):
        """
        Initialize the Docker runner launcher.
        
        Args:
            token_file: Path to file containing runner tokens from autoscaler
            runner_image: Docker image for GitHub Actions runner
            max_containers: Maximum number of concurrent runner containers
            cleanup_interval: Seconds between cleanup checks
            runner_work_dir: Work directory for runners (will be volume-mounted)
        """
        self.token_file = token_file
        self.runner_image = runner_image
        self.max_containers = max_containers
        self.cleanup_interval = cleanup_interval
        self.runner_work_dir = runner_work_dir
        self.running = False
        self.active_containers: Set[str] = set()
        self.last_cleanup = datetime.now()
        
        # Verify Docker is available
        self._verify_docker()
        
        # Ensure runner work directory exists
        Path(runner_work_dir).mkdir(parents=True, exist_ok=True)
        
    def _verify_docker(self) -> None:
        """Verify that Docker is installed and accessible."""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Docker is not accessible")
            logger.info("✓ Docker is available")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to verify Docker installation: {e}")
    
    def _pull_runner_image(self) -> None:
        """Pull the GitHub Actions runner Docker image."""
        logger.info(f"Pulling runner image: {self.runner_image}")
        try:
            result = subprocess.run(
                ["docker", "pull", self.runner_image],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                logger.warning(f"Failed to pull image: {result.stderr}")
                logger.info("Continuing with existing local image if available")
            else:
                logger.info("✓ Runner image pulled successfully")
        except subprocess.TimeoutExpired:
            logger.warning("Image pull timed out, continuing with local image")
    
    def _read_tokens(self) -> List[Dict]:
        """Read runner tokens from file."""
        if not os.path.exists(self.token_file):
            return []
        
        try:
            with open(self.token_file, 'r') as f:
                data = json.load(f)
                return data.get('tokens', [])
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to read tokens: {e}")
            return []
    
    def _get_active_containers(self) -> List[str]:
        """Get list of active runner container IDs."""
        try:
            result = subprocess.run(
                [
                    "docker", "ps",
                    "--filter", "label=github-runner=true",
                    "--format", "{{.ID}}"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return [cid.strip() for cid in result.stdout.strip().split('\n') if cid]
            return []
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.error(f"Failed to list containers: {e}")
            return []
    
    def _launch_container(self, token_info: Dict) -> Optional[str]:
        """
        Launch a Docker container running a GitHub Actions runner.
        
        Args:
            token_info: Dictionary containing token, repo, labels, etc.
            
        Returns:
            Container ID if successful, None otherwise
        """
        repo = token_info.get('repo')
        token = token_info.get('token')
        labels = token_info.get('labels', 'self-hosted,linux,docker')
        
        if not repo or not token:
            logger.error("Invalid token info: missing repo or token")
            return None
        
        # Generate container name with unique ID to avoid conflicts
        import uuid
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        container_name = f"runner_{repo.replace('/', '_')}_{timestamp}_{unique_id}"
        
        # Prepare Docker run command
        docker_cmd = [
            "docker", "run",
            "-d",  # Detached mode
            "--rm",  # Auto-remove container when it exits
            f"--name={container_name}",
            "--label=github-runner=true",
            f"--label=repo={repo}",
            
            # User mapping to match host user (fixes Git permission issues)
            "--user=1004:1004",
            
            # Security options
            "--security-opt=no-new-privileges",
            "--cap-drop=ALL",
            "--cap-add=NET_ADMIN",  # Needed for runner networking
            "--cap-add=NET_RAW",
            
            # Resource limits
            "--memory=4g",
            "--cpus=4",
            
            # Volumes
            f"-v={self.runner_work_dir}:/tmp/_work",
            "-v=/var/run/docker.sock:/var/run/docker.sock",  # For Docker-in-Docker if needed
            
            # Environment variables
            f"-e=REPO_URL=https://github.com/{repo}",
            f"-e=RUNNER_TOKEN={token}",
            f"-e=RUNNER_NAME={container_name}",
            f"-e=LABELS={labels}",
            "-e=RUNNER_WORKDIR=/tmp/_work",
            "-e=EPHEMERAL=true",  # Runner will self-destruct after one job
            
            # P2P Cache configuration (so runners can query workflow/state via cache)
            f"-e=GITHUB_TOKEN={os.environ.get('GITHUB_TOKEN', '')}",
            f"-e=CACHE_ENABLE_P2P={os.environ.get('CACHE_ENABLE_P2P', 'true')}",
            f"-e=CACHE_LISTEN_PORT={os.environ.get('CACHE_LISTEN_PORT', '9100')}",
            f"-e=BOOTSTRAP_PEERS={os.environ.get('BOOTSTRAP_PEERS', '/ip4/127.0.0.1/tcp/9100')}",
            
            # Runner image
            self.runner_image
        ]
        
        logger.info(f"Launching runner container for {repo}")
        logger.debug(f"Labels: {labels}")
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                container_id = result.stdout.strip()
                self.active_containers.add(container_id)
                logger.info(f"✓ Launched container {container_id[:12]} for {repo}")
                return container_id
            else:
                logger.error(f"Failed to launch container: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout launching container for {repo}")
            return None
        except Exception as e:
            logger.error(f"Error launching container: {e}")
            return None
    
    def _cleanup_exited_containers(self) -> None:
        """Remove completed/exited runner containers and update active set."""
        try:
            # Get all containers (including exited) with our label
            result = subprocess.run(
                [
                    "docker", "ps", "-a",
                    "--filter", "label=github-runner=true",
                    "--format", "{{.ID}}\t{{.Status}}"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return
            
            removed_count = 0
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                    
                container_id, status = parts[0], parts[1]
                
                # Remove from active set if exited
                if container_id in self.active_containers and 'Exited' in status:
                    self.active_containers.discard(container_id)
                    removed_count += 1
                    logger.info(f"Runner container {container_id[:12]} completed")
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} completed container(s)")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _process_tokens(self) -> None:
        """Read tokens and launch containers for pending workflows."""
        tokens = self._read_tokens()
        
        if not tokens:
            return
        
        # Check current capacity
        active_count = len(self._get_active_containers())
        available_slots = self.max_containers - active_count
        
        if available_slots <= 0:
            logger.debug(f"At capacity: {active_count}/{self.max_containers} runners active")
            return
        
        logger.info(f"Processing {len(tokens)} token(s), {available_slots} slot(s) available")
        
        launched = 0
        for token_info in tokens[:available_slots]:
            container_id = self._launch_container(token_info)
            if container_id:
                launched += 1
        
        if launched > 0:
            logger.info(f"✓ Launched {launched} runner container(s)")
    
    def start(self) -> None:
        """Start the runner launcher service."""
        self.running = True
        
        logger.info("================================================================================")
        logger.info("Containerized GitHub Actions Runner Launcher Started")
        logger.info("================================================================================")
        logger.info(f"Runner image: {self.runner_image}")
        logger.info(f"Max containers: {self.max_containers}")
        logger.info(f"Token file: {self.token_file}")
        logger.info(f"Work directory: {self.runner_work_dir}")
        logger.info(f"Cleanup interval: {self.cleanup_interval}s")
        logger.info("================================================================================")
        logger.info("")
        
        # Pull runner image on startup
        self._pull_runner_image()
        
        logger.info("Monitoring for runner tokens and launching containers...")
        logger.info("Press Ctrl+C to stop")
        logger.info("")
        
        check_count = 0
        while self.running:
            try:
                check_count += 1
                logger.info(f"--- Check #{check_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
                
                # Process tokens and launch containers
                self._process_tokens()
                
                # Periodic cleanup
                if (datetime.now() - self.last_cleanup).total_seconds() >= self.cleanup_interval:
                    logger.info("Running periodic cleanup...")
                    self._cleanup_exited_containers()
                    self.last_cleanup = datetime.now()
                
                # Show current status
                active = len(self._get_active_containers())
                logger.info(f"Active runners: {active}/{self.max_containers}")
                logger.info("")
                
                # Sleep before next check
                time.sleep(60)  # Check every 60 seconds
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(10)
    
    def stop(self) -> None:
        """Stop the runner launcher service."""
        logger.info("Stopping runner launcher...")
        self.running = False
        
        # Optional: Stop all active containers
        # Uncomment if you want to stop runners on service shutdown
        # for container_id in list(self.active_containers):
        #     try:
        #         subprocess.run(["docker", "stop", container_id], timeout=30)
        #     except:
        #         pass


class TokenBridge:
    """
    Bridge between autoscaler and runner launcher.
    Extracts tokens from autoscaler logs/state and writes to token file.
    """
    
    def __init__(self, output_file: str = "/var/lib/github-runner-autoscaler/runner_tokens.json"):
        """
        Initialize the token bridge.
        
        Args:
            output_file: Path to write runner tokens for launcher
        """
        self.output_file = output_file
        
    def import_from_autoscaler(self, autoscaler_module: str = "github_autoscaler") -> None:
        """
        Import the autoscaler and hook into its token generation.
        
        This is a simple implementation - in production you'd want to:
        1. Have autoscaler write tokens to a shared location
        2. Use a message queue or shared database
        3. Implement proper synchronization
        """
        logger.info("Token bridge: This is a placeholder for autoscaler integration")
        logger.info("In production, the autoscaler should write tokens to a shared location")
        logger.info(f"Expected token file format: {self.output_file}")
        logger.info("")
        logger.info("Example token file structure:")
        example = {
            "tokens": [
                {
                    "repo": "owner/repo",
                    "token": "AAZ7LEUWZ4...",
                    "labels": "self-hosted,linux,x64,docker,gpu",
                    "workflow_count": 5,
                    "created_at": "2025-11-02T18:54:52Z"
                }
            ]
        }
        logger.info(json.dumps(example, indent=2))


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}")
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch ephemeral Docker containers for GitHub Actions runners"
    )
    parser.add_argument(
        "--token-file",
        default="/var/lib/github-runner-autoscaler/runner_tokens.json",
        help="Path to token file from autoscaler"
    )
    parser.add_argument(
        "--runner-image",
        default="myoung34/github-runner:latest",
        help="Docker image for GitHub Actions runner"
    )
    parser.add_argument(
        "--max-containers",
        type=int,
        default=10,
        help="Maximum concurrent runner containers"
    )
    parser.add_argument(
        "--cleanup-interval",
        type=int,
        default=300,
        help="Seconds between cleanup checks"
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/_work",
        help="Work directory for runners"
    )
    parser.add_argument(
        "--bridge-mode",
        action="store_true",
        help="Show token bridge integration info"
    )
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if args.bridge_mode:
        bridge = TokenBridge(args.token_file)
        bridge.import_from_autoscaler()
        return 0
    
    try:
        launcher = DockerRunnerLauncher(
            token_file=args.token_file,
            runner_image=args.runner_image,
            max_containers=args.max_containers,
            cleanup_interval=args.cleanup_interval,
            runner_work_dir=args.work_dir
        )
        
        launcher.start()
        
    except KeyboardInterrupt:
        logger.info("\nReceived keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        if 'launcher' in locals():
            launcher.stop()
    
    logger.info("Runner launcher stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
