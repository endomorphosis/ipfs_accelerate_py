"""
P2P Error Aggregation and GitHub Issue Creation

Distributes errors among P2P peers via libp2p, aggregates them,
deduplicates against existing GitHub issues, and creates new issues
when necessary to minimize GitHub API usage.
"""

import json
import logging
import hashlib
import os
import subprocess
import time
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import threading

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

# Try to import datasets integration for error logging
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


class ErrorAggregator:
    """
    Aggregates errors across P2P peers and creates GitHub issues intelligently.
    
    Features:
    - Distributes errors via libp2p to all connected peers
    - Aggregates errors across all peers periodically
    - Deduplicates errors by signature
    - Checks existing GitHub issues to avoid duplicates
    - Bundles similar errors together
    - Creates GitHub issues in batches to minimize API calls
    """
    
    def __init__(
        self,
        repo: str,
        peer_registry,
        bundle_interval_minutes: int = 15,
        min_error_count: int = 3,
        enable_auto_issue_creation: bool = False,
        enable_auto_pr_creation: bool = False,
        enable_copilot_autofix: bool = False
    ):
        """
        Initialize error aggregator.
        
        Args:
            repo: GitHub repository (e.g., 'owner/repo')
            peer_registry: P2PPeerRegistry instance for peer discovery
            bundle_interval_minutes: How often to bundle and report errors
            min_error_count: Minimum occurrences before creating an issue
            enable_auto_issue_creation: Whether to automatically create issues
            enable_auto_pr_creation: Whether to automatically create draft PRs from issues
            enable_copilot_autofix: Whether to invoke Copilot for auto-fixing
        """
        # Initialize datasets integration for error logging
        self._provenance_logger = None
        self._datasets_manager = None
        if HAVE_DATASETS_INTEGRATION and is_datasets_available():
            try:
                self._provenance_logger = ProvenanceLogger()
                self._datasets_manager = DatasetsManager({
                    'enable_audit': True,
                    'enable_provenance': True
                })
                logger.info("Error aggregator using datasets integration for error tracking")
            except Exception as e:
                logger.debug(f"Datasets integration initialization skipped: {e}")
        
        # Initialize storage wrapper
        if _storage:
            try:
                self.storage = _storage
            except:
                self.storage = None
        else:
            self.storage = None
        
        self.repo = repo
        self.peer_registry = peer_registry
        self.bundle_interval = timedelta(minutes=bundle_interval_minutes)
        self.min_error_count = min_error_count
        self.enable_auto_issue_creation = enable_auto_issue_creation
        self.enable_auto_pr_creation = enable_auto_pr_creation
        self.enable_copilot_autofix = enable_copilot_autofix
        
        # Local error storage
        self.local_errors: List[Dict] = []
        self.error_signatures: Set[str] = set()
        
        # Aggregated errors from all peers
        self.aggregated_errors: Dict[str, List[Dict]] = defaultdict(list)
        
        # Cache of existing GitHub issues
        self.existing_issues_cache: Dict[str, Dict] = {}
        self.issues_cache_timestamp: Optional[datetime] = None
        self.issues_cache_ttl = timedelta(hours=1)
        
        # Last bundle time
        self.last_bundle_time: Optional[datetime] = None
        
        # Background thread for periodic bundling
        self.bundling_thread: Optional[threading.Thread] = None
        self.stop_bundling = threading.Event()
        
        logger.info(
            f"Error Aggregator initialized: repo={repo}, "
            f"bundle_interval={bundle_interval_minutes}m, "
            f"auto_create={enable_auto_issue_creation}, "
            f"auto_pr={enable_auto_pr_creation}, "
            f"auto_heal={enable_copilot_autofix}"
        )
    
    def start_bundling(self):
        """Start background thread for periodic error bundling."""
        if self.bundling_thread is None or not self.bundling_thread.is_alive():
            self.stop_bundling.clear()
            self.bundling_thread = threading.Thread(
                target=self._bundling_loop,
                daemon=True,
                name="ErrorBundlingThread"
            )
            self.bundling_thread.start()
            logger.info("✓ Error bundling thread started")
    
    def stop_bundling_thread(self):
        """Stop the background bundling thread."""
        self.stop_bundling.set()
        if self.bundling_thread and self.bundling_thread.is_alive():
            self.bundling_thread.join(timeout=5)
            logger.info("✓ Error bundling thread stopped")
    
    def _bundling_loop(self):
        """Background loop for periodic error bundling."""
        while not self.stop_bundling.is_set():
            try:
                # Sleep for bundle interval
                self.stop_bundling.wait(timeout=self.bundle_interval.total_seconds())
                
                if not self.stop_bundling.is_set():
                    # Bundle and potentially create issues
                    self.bundle_and_report_errors()
                    
            except Exception as e:
                logger.error(f"Error in bundling loop: {e}")
    
    def capture_error(
        self,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        context: Optional[Dict] = None,
        severity: str = "medium"
    ) -> str:
        """
        Capture an error locally and distribute to peers.
        
        Args:
            error_type: Type of error (e.g., 'APIError', 'NetworkError')
            error_message: Error message
            stack_trace: Optional stack trace
            context: Optional context dict with additional info
            severity: Error severity ('low', 'medium', 'high', 'critical')
            
        Returns:
            Error signature (hash)
        """
        error_data = {
            "type": error_type,
            "message": error_message,
            "stack_trace": stack_trace,
            "context": context or {},
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
            "peer_id": self.peer_registry.runner_name,
            "repo": self.repo
        }
        
        # Generate error signature
        signature = self._generate_error_signature(error_data)
        error_data["signature"] = signature
        
        # Store locally if not duplicate
        if signature not in self.error_signatures:
            self.local_errors.append(error_data)
            self.error_signatures.add(signature)
            logger.debug(f"Captured error: {error_type} - {signature[:16]}")
        
        # Distribute to peers via libp2p
        self._distribute_error_to_peers(error_data)
        
        return signature
    
    def _generate_error_signature(self, error_data: Dict) -> str:
        """
        Generate a unique signature for error deduplication.
        
        Uses error type, normalized message, and key context fields.
        """
        # Normalize error message (remove timestamps, paths, IDs)
        message = error_data["message"]
        
        # Remove common variable parts
        import re
        message = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', 'TIMESTAMP', message)
        message = re.sub(r'/[^\s]+/', '/PATH/', message)
        message = re.sub(r'\b[0-9a-f]{8,}\b', 'ID', message, flags=re.IGNORECASE)
        
        # Create signature from normalized content
        sig_content = f"{error_data['type']}:{message}"
        
        # Add relevant context if available
        if error_data.get("context"):
            context = error_data["context"]
            if "method" in context:
                sig_content += f":{context['method']}"
            if "endpoint" in context:
                sig_content += f":{context['endpoint']}"
        
        return hashlib.sha256(sig_content.encode()).hexdigest()
    
    def _distribute_error_to_peers(self, error_data: Dict):
        """
        Distribute error to other peers via P2P network.
        
        Uses GitHub Actions cache as the transport mechanism.
        """
        try:
            # Store in cache with error signature as key
            cache_key = f"error-{error_data['signature'][:16]}-{int(time.time())}"
            temp_file = f"/tmp/{cache_key}.json"
            
            error_json = json.dumps(error_data)
            
            # Try distributed storage first
            if self.storage:
                try:
                    self.storage.store_file(temp_file, error_json, pin=False)
                except:
                    pass  # Continue with temp file approach
            
            with open(temp_file, "w") as f:
                f.write(error_json)
            
            # Upload to GitHub Actions cache
            result = subprocess.run(
                [
                    "gh", "cache", "upload",
                    temp_file,
                    "--key", cache_key,
                    "--repo", self.repo
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            os.unlink(temp_file)
            
            if result.returncode == 0:
                logger.debug(f"✓ Distributed error to peers: {cache_key}")
            else:
                logger.debug(f"Failed to distribute error: {result.stderr}")
                
        except Exception as e:
            logger.debug(f"Error distributing to peers: {e}")
    
    def collect_peer_errors(self) -> int:
        """
        Collect errors from other peers via P2P network.
        
        Returns:
            Number of new errors collected
        """
        try:
            # List all error cache entries
            result = subprocess.run(
                [
                    "gh", "cache", "list",
                    "--repo", self.repo,
                    "--json", "key,createdAt"
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return 0
            
            cache_entries = json.loads(result.stdout)
            
            # Filter for error entries
            error_keys = [
                entry["key"]
                for entry in cache_entries
                if entry["key"].startswith("error-")
            ]
            
            new_errors = 0
            for cache_key in error_keys:
                try:
                    # Download error data
                    temp_file = f"/tmp/{cache_key}.json"
                    download_result = subprocess.run(
                        [
                            "gh", "cache", "download",
                            cache_key,
                            "--dir", "/tmp",
                            "--repo", self.repo
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if download_result.returncode == 0 and os.path.exists(temp_file):
                        # Try distributed storage first
                        if self.storage:
                            try:
                                cached_data = self.storage.get_file(temp_file)
                                if cached_data:
                                    error_data = json.loads(cached_data)
                                else:
                                    with open(temp_file, "r") as f:
                                        error_data = json.load(f)
                                    # Cache for future use
                                    self.storage.store_file(temp_file, json.dumps(error_data), pin=False)
                            except:
                                with open(temp_file, "r") as f:
                                    error_data = json.load(f)
                        else:
                            with open(temp_file, "r") as f:
                                error_data = json.load(f)
                        
                        # Add to aggregated errors by signature
                        signature = error_data.get("signature")
                        if signature and signature not in self.error_signatures:
                            self.aggregated_errors[signature].append(error_data)
                            self.error_signatures.add(signature)
                            new_errors += 1
                        
                        os.unlink(temp_file)
                        
                except Exception as e:
                    logger.debug(f"Failed to collect error {cache_key}: {e}")
                    continue
            
            if new_errors > 0:
                logger.info(f"✓ Collected {new_errors} new error(s) from peers")
            
            return new_errors
            
        except Exception as e:
            logger.error(f"Error collecting peer errors: {e}")
            return 0
    
    def _refresh_existing_issues_cache(self):
        """Refresh cache of existing GitHub issues."""
        try:
            # Check if cache is still valid
            if (self.issues_cache_timestamp and 
                datetime.utcnow() - self.issues_cache_timestamp < self.issues_cache_ttl):
                return
            
            # Fetch open issues from GitHub
            result = subprocess.run(
                [
                    "gh", "issue", "list",
                    "--repo", self.repo,
                    "--state", "open",
                    "--json", "number,title,body,labels",
                    "--limit", "100"
                ],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                issues = json.loads(result.stdout)
                
                # Build cache with error signatures from issue bodies
                self.existing_issues_cache = {}
                for issue in issues:
                    # Extract error signature from issue body if present
                    body = issue.get("body", "")
                    if "Error Signature:" in body:
                        for line in body.split("\n"):
                            if "Error Signature:" in line:
                                sig = line.split("Error Signature:")[-1].strip().strip("`")
                                self.existing_issues_cache[sig] = issue
                                break
                
                self.issues_cache_timestamp = datetime.utcnow()
                logger.info(f"✓ Refreshed issues cache: {len(self.existing_issues_cache)} issues")
                
        except Exception as e:
            logger.error(f"Error refreshing issues cache: {e}")
    
    def bundle_and_report_errors(self) -> Dict:
        """
        Bundle errors and create GitHub issues if necessary.
        
        Returns:
            Summary dict with bundling statistics
        """
        try:
            # Collect errors from peers
            new_peer_errors = self.collect_peer_errors()
            
            # Combine local and aggregated errors
            all_errors = defaultdict(list)
            for error in self.local_errors:
                all_errors[error["signature"]].append(error)
            for sig, errors in self.aggregated_errors.items():
                all_errors[sig].extend(errors)
            
            # Refresh issues cache
            self._refresh_existing_issues_cache()
            
            # Filter errors that meet threshold and aren't duplicates
            errors_to_report = {}
            for sig, errors in all_errors.items():
                if len(errors) >= self.min_error_count:
                    # Check if issue already exists
                    if sig not in self.existing_issues_cache:
                        errors_to_report[sig] = errors
            
            # Create issues if auto-creation is enabled
            issues_created = 0
            if self.enable_auto_issue_creation and errors_to_report:
                issues_created = self._create_github_issues(errors_to_report)
            
            # Clear processed errors
            self.local_errors = []
            self.aggregated_errors.clear()
            self.last_bundle_time = datetime.utcnow()
            
            summary = {
                "timestamp": self.last_bundle_time.isoformat(),
                "new_peer_errors": new_peer_errors,
                "total_unique_errors": len(all_errors),
                "errors_meeting_threshold": len(errors_to_report),
                "issues_created": issues_created,
                "existing_issues_checked": len(self.existing_issues_cache)
            }
            
            logger.info(
                f"✓ Error bundling complete: {summary['total_unique_errors']} unique, "
                f"{summary['errors_meeting_threshold']} to report, "
                f"{summary['issues_created']} issues created"
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in bundle_and_report_errors: {e}")
            return {"error": str(e)}
    
    def _create_github_issues(self, errors_to_report: Dict[str, List[Dict]]) -> int:
        """
        Create GitHub issues for bundled errors.
        
        Args:
            errors_to_report: Dict mapping error signatures to error lists
            
        Returns:
            Number of issues created
        """
        issues_created = 0
        
        for signature, errors in errors_to_report.items():
            try:
                # Use first error as template
                template_error = errors[0]
                
                # Build issue title
                title = f"[Auto-Generated] {template_error['type']}: {template_error['message'][:80]}"
                if len(template_error['message']) > 80:
                    title += "..."
                
                # Build issue body
                body_parts = [
                    f"**Error Type:** {template_error['type']}",
                    f"**Severity:** {template_error['severity']}",
                    f"**Occurrences:** {len(errors)} time(s) across {len(set(e['peer_id'] for e in errors))} peer(s)",
                    "",
                    "## Error Message",
                    f"```\n{template_error['message']}\n```",
                    ""
                ]
                
                # Add stack trace if available
                if template_error.get('stack_trace'):
                    body_parts.extend([
                        "## Stack Trace",
                        f"```\n{template_error['stack_trace']}\n```",
                        ""
                    ])
                
                # Add context
                if template_error.get('context'):
                    body_parts.extend([
                        "## Context",
                        f"```json\n{json.dumps(template_error['context'], indent=2)}\n```",
                        ""
                    ])
                
                # Add occurrence details
                body_parts.extend([
                    "## Occurrences",
                    ""
                ])
                for error in errors[:5]:  # Show first 5
                    body_parts.append(
                        f"- {error['timestamp']} on peer `{error['peer_id']}`"
                    )
                if len(errors) > 5:
                    body_parts.append(f"- ... and {len(errors) - 5} more")
                
                # Add error signature for deduplication
                body_parts.extend([
                    "",
                    "---",
                    f"**Error Signature:** `{signature}`",
                    "*This issue was automatically created by the P2P Error Aggregator*"
                ])
                
                body = "\n".join(body_parts)
                
                # Determine labels
                labels = ["auto-generated", "bug"]
                if template_error['severity'] in ['high', 'critical']:
                    labels.append("priority")
                
                # Create the issue
                result = subprocess.run(
                    [
                        "gh", "issue", "create",
                        "--repo", self.repo,
                        "--title", title,
                        "--body", body,
                        "--label", ",".join(labels)
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    issue_url = result.stdout.strip()
                    logger.info(f"✓ Created issue: {issue_url}")
                    issues_created += 1
                    
                    # Add to cache to prevent duplicates
                    self.existing_issues_cache[signature] = {
                        "url": issue_url,
                        "title": title
                    }
                    
                    # Create draft PR if enabled
                    if self.enable_auto_pr_creation:
                        self._create_draft_pr_from_issue(issue_url, template_error, signature)
                    
                else:
                    logger.warning(f"Failed to create issue: {result.stderr}")
                
                # Rate limit protection: small delay between issue creations
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error creating issue for signature {signature[:16]}: {e}")
                continue
        
        return issues_created
    
    def _create_draft_pr_from_issue(
        self,
        issue_url: str,
        error_data: Dict,
        signature: str
    ) -> Optional[str]:
        """
        Create a draft PR to fix the issue.
        
        Args:
            issue_url: URL of the GitHub issue
            error_data: Error data dictionary
            signature: Error signature
            
        Returns:
            PR URL if created, None otherwise
        """
        try:
            # Extract issue number from URL
            issue_number = issue_url.split("/")[-1]
            
            # Create a branch name
            error_type = error_data['type'].lower().replace(' ', '-')
            branch_name = f"auto-fix/issue-{issue_number}-{error_type}"
            
            # Create PR title and body
            pr_title = f"[Auto-Fix] Fix for issue #{issue_number}: {error_data['type']}"
            pr_body = f"""This is an automatically generated draft PR to address issue #{issue_number}.

**Issue:** {issue_url}
**Error Type:** {error_data['type']}
**Severity:** {error_data['severity']}
**Error Signature:** `{signature}`

## Problem Description
{error_data['message']}

## Stack Trace
```python
{error_data.get('stack_trace', 'N/A')}
```

## Action Required
This PR is a draft and needs to be completed with the actual fix.

**Next Steps:**
1. Review the error details in issue #{issue_number}
2. Analyze the stack trace and context
3. Implement the fix
4. Add tests to prevent regression
5. Mark as ready for review

{'*GitHub Copilot has been invoked to suggest fixes.*' if self.enable_copilot_autofix else ''}

Closes #{issue_number}

---
*This PR was automatically created by the P2P Error Aggregator*
"""
            
            logger.info(f"Creating draft PR for issue #{issue_number}")
            
            # Note: Creating a PR requires:
            # 1. A branch to be created
            # 2. Commits on that branch
            # Since we don't have changes yet, we log this for now
            # In a full implementation, we would:
            # - Create branch
            # - Make automated changes (with Copilot)
            # - Push changes
            # - Create PR
            
            logger.info(f"  Branch name: {branch_name}")
            logger.info(f"  Title: {pr_title}")
            
            # Invoke Copilot for auto-fix if enabled
            if self.enable_copilot_autofix:
                self._invoke_copilot_autofix(issue_number, error_data, signature)
            
            # Placeholder - would create actual PR here
            return None
            
        except Exception as e:
            logger.error(f"Error creating draft PR: {e}")
            return None
    
    def _invoke_copilot_autofix(
        self,
        issue_number: str,
        error_data: Dict,
        signature: str
    ) -> bool:
        """
        Invoke GitHub Copilot to suggest fixes for the error.
        
        Args:
            issue_number: GitHub issue number
            error_data: Error data dictionary
            signature: Error signature
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import Copilot SDK
            try:
                from ipfs_accelerate_py.copilot_sdk.wrapper import CopilotSDK, HAVE_COPILOT_SDK
            except ImportError:
                logger.warning("Copilot SDK not available for auto-healing")
                return False
            
            if not HAVE_COPILOT_SDK:
                logger.warning("Copilot SDK not installed")
                return False
            
            # Build analysis prompt for Copilot
            prompt = f"""Analyze the following error from the IPFS Accelerate CLI and suggest fixes:

**Error Type:** {error_data['type']}
**Severity:** {error_data['severity']}
**Error Message:**
{error_data['message']}

**Stack Trace:**
{error_data.get('stack_trace', 'N/A')}

**Context:**
{json.dumps(error_data.get('context', {}), indent=2)}

Please provide:
1. Root cause analysis
2. Suggested code fixes
3. Files that need to be modified
4. Test cases to prevent regression
5. Any related issues or patterns

Issue #{issue_number} tracks this error.
"""
            
            logger.info(f"Invoking Copilot for issue #{issue_number}")
            logger.debug(f"Copilot prompt length: {len(prompt)} characters")
            
            # In a full implementation:
            # 1. Initialize CopilotSDK
            # 2. Create a session
            # 3. Send the prompt
            # 4. Parse the response
            # 5. Apply suggested fixes (with review)
            # 6. Create commits
            # 7. Push to branch
            
            # For now, log that we would invoke it
            logger.info("✓ Copilot analysis would be performed here")
            logger.info("  This would generate fix suggestions for the error")
            
            return True
            
        except Exception as e:
            logger.error(f"Error invoking Copilot auto-fix: {e}")
            return False
    
    def get_error_statistics(self) -> Dict:
        """
        Get statistics about captured and aggregated errors.
        
        Returns:
            Dict with error statistics
        """
        # Combine all errors
        all_errors = list(self.local_errors)
        for errors in self.aggregated_errors.values():
            all_errors.extend(errors)
        
        # Group by type and severity
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        by_peer = defaultdict(int)
        
        for error in all_errors:
            by_type[error['type']] += 1
            by_severity[error['severity']] += 1
            by_peer[error['peer_id']] += 1
        
        return {
            "total_errors": len(all_errors),
            "unique_signatures": len(self.error_signatures),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "by_peer": dict(by_peer),
            "last_bundle_time": self.last_bundle_time.isoformat() if self.last_bundle_time else None,
            "existing_issues_count": len(self.existing_issues_cache)
        }
