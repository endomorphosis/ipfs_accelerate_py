"""P2P Peer Registry for MCP++ (refactored from original github_cli module).

P2P Peer Registry for cache-sharing coordination.

This module provides peer discovery for P2P cache sharing across GitHub Actions
runners, refactored to work with the MCP++ Trio-native architecture.

Module: ipfs_accelerate_py.mcplusplus_module.p2p.peer_registry
Refactored from: ipfs_accelerate_py/github_cli/p2p_peer_registry.py

Implementation uses a GitHub Issue as a lightweight registry backend:
- Each peer periodically upserts a single runner-specific issue comment
- Peers discover each other by listing and parsing those comments

This avoids needing a separate rendezvous server and works anywhere `gh` is
authenticated with repo access.
"""

import json
import logging
import os
import socket
import subprocess
import time
import tempfile
from typing import Dict, List, Optional
from datetime import datetime, timedelta

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

storage_wrapper = get_storage_wrapper if HAVE_STORAGE_WRAPPER else None

logger = logging.getLogger(__name__)


class P2PPeerRegistry:
    """
    Manages peer discovery for P2P cache sharing across GitHub Actions runners.
    
    Uses GitHub CLI + GitHub Issues to store/retrieve peer information as issue
    comments, allowing runners to discover each other without a central server.
    """
    
    def __init__(
        self,
        repo: str,
        runner_name: Optional[str] = None,
        cache_prefix: str = "p2p-peer-registry",
        peer_ttl_minutes: int = 30,
        issue_title: Optional[str] = None
    ):
        """
        Initialize peer registry.
        
        Args:
            repo: GitHub repository (e.g., 'owner/repo')
            runner_name: Name of this runner (auto-detected if None)
            cache_prefix: Prefix for cache keys
            peer_ttl_minutes: How long peer entries are valid
        """
        # Initialize storage wrapper
        self.storage = None
        if storage_wrapper is not None:
            try:
                self.storage = storage_wrapper(auto_detect_ci=True)
            except Exception:
                self.storage = None
        
        self.repo = repo
        self.repo_owner, self.repo_name = (repo.split("/", 1) + [""])[0:2]
        self.runner_name = runner_name or self._detect_runner_name()
        self.cache_prefix = cache_prefix
        self.peer_ttl = timedelta(minutes=peer_ttl_minutes)

        # Issue-backed registry configuration
        self.issue_title = issue_title or self.cache_prefix
        self._issue_number: Optional[int] = None
        
        # Detect public IP for this runner
        self.public_ip = self._detect_public_ip()
        
        logger.info(f"P2P Peer Registry initialized: runner={self.runner_name}, ip={self.public_ip}")

    def _run_gh(self, args: List[str], timeout: int = 30) -> subprocess.CompletedProcess:
        """Run a `gh` command and return the CompletedProcess."""
        env = dict(os.environ)
        # gh prefers GH_TOKEN; allow operators to provide only GITHUB_TOKEN.
        if not (env.get("GH_TOKEN") or "").strip() and (env.get("GITHUB_TOKEN") or "").strip():
            env["GH_TOKEN"] = env["GITHUB_TOKEN"].strip()
        # Avoid paging in non-interactive contexts.
        env.setdefault("GH_PAGER", "cat")
        return subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

    def _gh_api(self, method: str, endpoint: str, payload: Optional[Dict] = None, timeout: int = 30) -> subprocess.CompletedProcess:
        """Call GitHub API via `gh api`.

        Uses `--input` to avoid shell-quoting issues with newlines.
        """
        args = ["api", "-X", method.upper(), endpoint]
        tmp_path = None
        if payload is not None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix=f"{self.cache_prefix}-api-",
                suffix=".json",
                delete=False,
            ) as f:
                tmp_path = f.name
                json_data = json.dumps(payload)
                f.write(json_data)
                
            # Store in distributed storage for debugging/caching
            if self.storage and tmp_path:
                try:
                    self.storage.write_file(json_data, filename=os.path.basename(tmp_path), pin=False)
                except Exception:
                    pass  # Silently fail
                
            args.extend(["--input", tmp_path])
        try:
            return self._run_gh(args, timeout=timeout)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _get_or_create_registry_issue_number(self) -> Optional[int]:
        """Find (or create) the issue used as the registry backend."""
        if self._issue_number is not None:
            return self._issue_number

        # Find issue by title
        list_result = self._run_gh(
            [
                "issue",
                "list",
                "--repo",
                self.repo,
                "--search",
                f'in:title "{self.issue_title}"',
                "--state",
                "open",
                "--json",
                "number,title",
                "--limit",
                "50",
            ],
            timeout=30,
        )
        if list_result.returncode != 0:
            logger.warning(f"Failed to list issues for registry: {list_result.stderr.strip()}")
            return None

        try:
            issues = json.loads(list_result.stdout)
        except Exception as e:
            logger.warning(f"Failed to parse issue list JSON: {e}")
            return None

        matching_numbers: List[int] = []
        for issue in issues:
            if issue.get("title") == self.issue_title and isinstance(issue.get("number"), int):
                matching_numbers.append(issue["number"])

        if matching_numbers:
            # Prefer a deterministic selection so multiple runners converge even
            # if duplicates exist due to concurrent creation.
            selected = min(matching_numbers)
            if len(matching_numbers) > 1:
                logger.warning(
                    "Multiple registry issues share the same title; using the lowest number. "
                    f"title={self.issue_title!r} issues={sorted(matching_numbers)}"
                )
            self._issue_number = selected
            return self._issue_number

        # Create issue
        marker_body = (
            "This issue is used as an automated peer registry for IPFS Accelerate P2P cache sharing.\n\n"
            "Each peer writes/updates a single comment containing its current peer info.\n"
        )
        create_result = self._run_gh(
            [
                "issue",
                "create",
                "--repo",
                self.repo,
                "--title",
                self.issue_title,
                "--body",
                marker_body,
            ],
            timeout=60,
        )
        if create_result.returncode != 0:
            logger.warning(f"Failed to create registry issue: {create_result.stderr.strip()}")
            return None

        # Extract issue number from created URL (last path segment)
        url = (create_result.stdout or "").strip()
        try:
            self._issue_number = int(url.rstrip("/").split("/")[-1])
        except Exception:
            # Fallback: re-list and pick again
            self._issue_number = None
            return self._get_or_create_registry_issue_number()

        return self._issue_number

    def _comment_marker(self, runner_name: str) -> str:
        return f"<!-- {self.cache_prefix}:{runner_name} -->"

    def _render_peer_comment(self, peer_info: Dict) -> str:
        marker = self._comment_marker(peer_info.get("runner_name") or self.runner_name)
        payload = json.dumps(peer_info, indent=2, sort_keys=True)
        return f"{marker}\n```json\n{payload}\n```\n"

    def _parse_peer_comment(self, body: str) -> Optional[Dict]:
        # Must start with our marker prefix
        if not body.strip().startswith(f"<!-- {self.cache_prefix}:"):
            return None
        # Extract JSON fenced block
        idx = body.find("```json")
        if idx == -1:
            return None
        start = body.find("\n", idx)
        end = body.rfind("```")
        if start == -1 or end == -1 or end <= start:
            return None
        raw = body[start + 1 : end].strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _list_registry_comments(self, issue_number: int) -> List[Dict]:
        endpoint = f"/repos/{self.repo_owner}/{self.repo_name}/issues/{issue_number}/comments?per_page=100"
        result = self._gh_api("GET", endpoint, payload=None, timeout=30)
        if result.returncode != 0:
            logger.warning(f"Failed to list registry comments: {result.stderr.strip()}")
            return []
        try:
            data = json.loads(result.stdout)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _delete_registry_comment(self, comment_id: int) -> bool:
        endpoint = f"/repos/{self.repo_owner}/{self.repo_name}/issues/comments/{comment_id}"
        result = self._gh_api("DELETE", endpoint, payload=None, timeout=30)
        if result.returncode != 0:
            logger.warning(f"Failed to delete registry comment {comment_id}: {result.stderr.strip()}")
            return False
        return True
    
    def _detect_runner_name(self) -> str:
        """Detect the GitHub Actions runner name."""
        # Try environment variables
        runner_name = os.environ.get("RUNNER_NAME")
        if runner_name:
            return runner_name
        
        # Try hostname
        try:
            return socket.gethostname()
        except Exception:
            return "unknown-runner"
    
    def _detect_public_ip(self) -> Optional[str]:
        """
        Detect the public IP address of this runner.
        
        This is needed for NAT traversal and peer connectivity.
        """
        try:
            # Try multiple services for redundancy
            services = [
                "https://api.ipify.org",
                "https://ifconfig.me/ip",
                "https://icanhazip.com"
            ]
            
            import urllib.request
            for service in services:
                try:
                    with urllib.request.urlopen(service, timeout=5) as response:
                        return response.read().decode('utf-8').strip()
                except Exception:
                    continue
            
            return None
        except Exception as e:
            logger.warning(f"Failed to detect public IP: {e}")
            return None
    
    def register_peer(
        self,
        peer_id: str,
        listen_port: int,
        multiaddr: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Register this runner as an active peer.
        
        Args:
            peer_id: libp2p peer ID
            listen_port: Port the peer is listening on
            multiaddr: Full libp2p multiaddr
            metadata: Optional additional metadata
            
        Returns:
            True if registration succeeded
        """
        try:
            peer_info = {
                "peer_id": peer_id,
                "runner_name": self.runner_name,
                "public_ip": self.public_ip,
                "listen_port": listen_port,
                "multiaddr": multiaddr,
                "last_seen": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }

            issue_number = self._get_or_create_registry_issue_number()
            if issue_number is None:
                return False

            comment_body = self._render_peer_comment(peer_info)
            comments = self._list_registry_comments(issue_number)

            # Upsert our runner-specific comment
            marker = self._comment_marker(self.runner_name)
            existing_id: Optional[int] = None
            for c in comments:
                if isinstance(c, dict) and isinstance(c.get("id"), int) and isinstance(c.get("body"), str):
                    if c["body"].lstrip().startswith(marker):
                        existing_id = c["id"]
                        break

            if existing_id is None:
                endpoint = f"/repos/{self.repo_owner}/{self.repo_name}/issues/{issue_number}/comments"
                result = self._gh_api("POST", endpoint, payload={"body": comment_body}, timeout=60)
            else:
                endpoint = f"/repos/{self.repo_owner}/{self.repo_name}/issues/comments/{existing_id}"
                result = self._gh_api("PATCH", endpoint, payload={"body": comment_body}, timeout=60)

            if result.returncode != 0:
                logger.warning(f"Failed to register peer via issue comment: {result.stderr.strip()}")
                return False

            logger.info(f"✓ Registered peer: {peer_id[:16]}... on {self.public_ip}:{listen_port}")
            return True
                
        except Exception as e:
            logger.error(f"Error registering peer: {e}")
            return False
    
    def discover_peers(self, max_peers: int = 10) -> List[Dict]:
        """
        Discover active peers from the registry.
        
        Args:
            max_peers: Maximum number of peers to return
            
        Returns:
            List of peer info dictionaries
        """
        try:
            issue_number = self._get_or_create_registry_issue_number()
            if issue_number is None:
                return []

            comments = self._list_registry_comments(issue_number)
            peers: List[Dict] = []
            for c in comments:
                body = c.get("body") if isinstance(c, dict) else None
                if not isinstance(body, str):
                    continue
                peer_info = self._parse_peer_comment(body)
                if not peer_info:
                    continue
                if peer_info.get("runner_name") == self.runner_name:
                    continue
                try:
                    last_seen = datetime.fromisoformat(peer_info.get("last_seen", ""))
                except Exception:
                    continue
                if datetime.utcnow() - last_seen < self.peer_ttl:
                    peers.append(peer_info)

            peers = peers[:max_peers]
            logger.info(f"✓ Discovered {len(peers)} active peers")
            return peers
        except Exception as e:
            logger.error(f"Error discovering peers: {e}")
            return []

    def list_peers(self, max_peers: int = 50) -> List[Dict]:
        """Compatibility wrapper used by dashboard/tools."""
        return self.discover_peers(max_peers=max_peers)
    
    def get_bootstrap_addrs(self, max_peers: int = 5) -> List[str]:
        """
        Get bootstrap multiaddrs for discovered peers.
        
        Args:
            max_peers: Maximum number of bootstrap peers
            
        Returns:
            List of libp2p multiaddrs
        """
        peers = self.discover_peers(max_peers)
        return [peer["multiaddr"] for peer in peers if peer.get("multiaddr")]
    
    def cleanup_stale_peers(self) -> int:
        """
        Remove stale peer entries from the registry.
        
        Returns:
            Number of peers cleaned up
        """
        try:
            issue_number = self._get_or_create_registry_issue_number()
            if issue_number is None:
                return 0

            comments = self._list_registry_comments(issue_number)
            cleaned = 0

            for c in comments:
                if not isinstance(c, dict):
                    continue
                comment_id = c.get("id")
                body = c.get("body")
                if not isinstance(comment_id, int) or not isinstance(body, str):
                    continue

                peer_info = self._parse_peer_comment(body)
                if not peer_info:
                    continue

                try:
                    last_seen = datetime.fromisoformat(peer_info.get("last_seen", ""))
                except Exception:
                    continue

                if datetime.utcnow() - last_seen > self.peer_ttl:
                    if self._delete_registry_comment(comment_id):
                        cleaned += 1

            if cleaned > 0:
                logger.info(f"✓ Cleaned up {cleaned} stale peer(s)")
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning up peers: {e}")
            return 0
    
    def heartbeat(self, peer_id: str, listen_port: int, multiaddr: str) -> None:
        """
        Send periodic heartbeat to keep peer entry fresh.
        
        Should be called every ~5-10 minutes.
        """
        self.register_peer(peer_id, listen_port, multiaddr)
