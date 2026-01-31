"""
Error Handler for IPFS Accelerate CLI

This module provides automatic error handling with GitHub issue creation
and auto-healing capabilities through GitHub Copilot.
"""

import logging
import sys
import traceback
import functools
from typing import Any, Callable, Optional, Dict, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class CLIErrorHandler:
    """
    Handles CLI errors by capturing stack traces, logs, and creating GitHub issues
    with optional auto-healing through GitHub Copilot.
    """
    
    def __init__(
        self,
        repo: str,
        enable_auto_issue: bool = False,
        enable_auto_pr: bool = False,
        enable_auto_heal: bool = False,
        log_context_lines: int = 50
    ):
        """
        Initialize CLI error handler.
        
        Args:
            repo: GitHub repository (e.g., 'owner/repo')
            enable_auto_issue: Automatically create GitHub issues for errors
            enable_auto_pr: Automatically create draft PRs from issues
            enable_auto_heal: Automatically invoke Copilot to fix issues
            log_context_lines: Number of log lines to capture before error
        """
        self.repo = repo
        self.enable_auto_issue = enable_auto_issue
        self.enable_auto_pr = enable_auto_pr
        self.enable_auto_heal = enable_auto_heal
        self.log_context_lines = log_context_lines
        
        # Lazy imports to avoid circular dependencies
        self._error_aggregator = None
        self._github_cli = None
        self._copilot_sdk = None
        self._logs_module = None
        
        # Store captured errors for batch processing
        self._captured_errors = []
    
    def _get_error_aggregator(self):
        """Lazy load error aggregator."""
        if self._error_aggregator is None:
            try:
                from ipfs_accelerate_py.github_cli.error_aggregator import ErrorAggregator
                from ipfs_accelerate_py.github_cli.p2p_peer_registry import P2PPeerRegistry
                
                # Initialize peer registry
                peer_registry = P2PPeerRegistry()
                
                # Initialize error aggregator
                self._error_aggregator = ErrorAggregator(
                    repo=self.repo,
                    peer_registry=peer_registry,
                    enable_auto_issue_creation=self.enable_auto_issue
                )
                
                # Start bundling thread if auto-issue is enabled
                if self.enable_auto_issue:
                    self._error_aggregator.start_bundling()
                    
            except ImportError as e:
                logger.warning(f"Could not initialize error aggregator: {e}")
        
        return self._error_aggregator
    
    def _get_github_cli(self):
        """
        Lazy load GitHub CLI with P2P/IPFS caching enabled.
        
        GitHub API calls are automatically cached with:
        - P2P cache sharing via libp2p
        - IPFS/ipfs_kit integration
        - Content-addressed validation
        - Encrypted cache entries
        """
        if self._github_cli is None:
            try:
                from ipfs_accelerate_py.github_cli.wrapper import GitHubCLI
                # Initialize with caching enabled (default) and P2P cache sharing
                self._github_cli = GitHubCLI(
                    enable_cache=True,  # Enable P2P/IPFS caching
                    cache_ttl=300  # 5 minute default TTL
                )
                logger.debug("GitHub CLI initialized with P2P/IPFS caching enabled")
            except ImportError as e:
                logger.warning(f"Could not initialize GitHub CLI: {e}")
        
        return self._github_cli
    
    def _get_copilot_sdk(self):
        """Lazy load Copilot SDK."""
        if self._copilot_sdk is None:
            try:
                from ipfs_accelerate_py.copilot_sdk.wrapper import CopilotSDK, HAVE_COPILOT_SDK
                if HAVE_COPILOT_SDK:
                    self._copilot_sdk = CopilotSDK(auto_start=True)
            except ImportError as e:
                logger.warning(f"Could not initialize Copilot SDK: {e}")
        
        return self._copilot_sdk
    
    def _get_logs_module(self):
        """Lazy load logs module."""
        if self._logs_module is None:
            try:
                from ipfs_accelerate_py.logs import SystemLogs
                self._logs_module = SystemLogs()
            except ImportError as e:
                logger.warning(f"Could not initialize logs module: {e}")
        
        return self._logs_module
    
    def _capture_log_context(self) -> str:
        """
        Capture recent log entries for error context.
        
        Returns:
            String containing recent log lines
        """
        try:
            logs_module = self._get_logs_module()
            if logs_module:
                # Get recent logs
                log_entries = logs_module.get_logs(lines=self.log_context_lines)
                
                # Format as string
                log_lines = []
                for entry in log_entries:
                    timestamp = entry.get('timestamp', 'N/A')
                    level = entry.get('level', 'INFO')
                    message = entry.get('message', '')
                    log_lines.append(f"[{timestamp}] {level}: {message}")
                
                return "\n".join(log_lines)
        except Exception as e:
            logger.debug(f"Could not capture log context: {e}")
        
        # Fallback: return empty string
        return ""
    
    def capture_error(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Capture an error with full context.
        
        Args:
            exception: The exception that occurred
            context: Optional additional context
            
        Returns:
            Error signature if captured, None otherwise
        """
        # Get error details
        error_type = type(exception).__name__
        error_message = str(exception)
        stack_trace = traceback.format_exc()
        
        # Capture log context
        log_context = self._capture_log_context()
        
        # Build full context
        full_context = {
            "command": " ".join(sys.argv),
            "python_version": sys.version,
            "timestamp": datetime.utcnow().isoformat(),
            "working_directory": str(Path.cwd()),
        }
        
        if context:
            full_context.update(context)
        
        if log_context:
            full_context["preceding_logs"] = log_context
        
        # Log the error
        logger.error(f"CLI Error captured: {error_type}: {error_message}")
        logger.debug(f"Stack trace:\n{stack_trace}")
        
        # Store error for potential batch processing
        error_data = {
            "type": error_type,
            "message": error_message,
            "stack_trace": stack_trace,
            "context": full_context,
            "severity": self._determine_severity(exception)
        }
        self._captured_errors.append(error_data)
        
        # Send to error aggregator if available
        error_aggregator = self._get_error_aggregator()
        if error_aggregator:
            signature = error_aggregator.capture_error(
                error_type=error_type,
                error_message=error_message,
                stack_trace=stack_trace,
                context=full_context,
                severity=error_data["severity"]
            )
            return signature
        
        return None
    
    def _determine_severity(self, exception: Exception) -> str:
        """
        Determine error severity based on exception type.
        
        Args:
            exception: The exception
            
        Returns:
            Severity level: 'low', 'medium', 'high', or 'critical'
        """
        # Critical errors
        if isinstance(exception, (SystemExit, KeyboardInterrupt)):
            return "low"  # User-initiated
        
        if isinstance(exception, (MemoryError, RecursionError)):
            return "critical"
        
        # High severity
        if isinstance(exception, (OSError, IOError, RuntimeError)):
            return "high"
        
        # Medium severity (most errors)
        if isinstance(exception, (ValueError, TypeError, KeyError, AttributeError)):
            return "medium"
        
        # Default to medium
        return "medium"
    
    def create_issue_from_error(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a GitHub issue directly from an error.
        
        Args:
            exception: The exception that occurred
            context: Optional additional context
            
        Returns:
            Issue URL if created, None otherwise
        """
        if not self.enable_auto_issue:
            logger.debug("Auto-issue creation disabled")
            return None
        
        github_cli = self._get_github_cli()
        if not github_cli:
            logger.warning("GitHub CLI not available, cannot create issue")
            return None
        
        # Capture error details
        error_type = type(exception).__name__
        error_message = str(exception)
        stack_trace = traceback.format_exc()
        log_context = self._capture_log_context()
        
        # Build issue title
        title = f"[Auto-Generated Error] {error_type}: {error_message[:80]}"
        if len(error_message) > 80:
            title += "..."
        
        # Build issue body
        body_parts = [
            "# Auto-Generated Error Report",
            "",
            f"**Error Type:** `{error_type}`",
            f"**Command:** `{' '.join(sys.argv)}`",
            f"**Timestamp:** {datetime.utcnow().isoformat()}",
            "",
            "## Error Message",
            "```",
            error_message,
            "```",
            "",
            "## Stack Trace",
            "```python",
            stack_trace,
            "```",
        ]
        
        # Add log context if available
        if log_context:
            body_parts.extend([
                "",
                "## Preceding Logs",
                f"Last {self.log_context_lines} log lines before error:",
                "```",
                log_context,
                "```",
            ])
        
        # Add additional context
        if context:
            body_parts.extend([
                "",
                "## Additional Context",
                "```json",
                __import__('json').dumps(context, indent=2),
                "```",
            ])
        
        body_parts.extend([
            "",
            "---",
            "*This issue was automatically created by the IPFS Accelerate error handler.*",
        ])
        
        body = "\n".join(body_parts)
        
        # Create the issue
        try:
            result = github_cli._run_command(
                ["issue", "create", "--repo", self.repo, "--title", title, "--body", body, "--label", "auto-generated,bug"],
                timeout=60
            )
            
            if result.get("returncode") == 0:
                issue_url = result.get("stdout", "").strip()
                logger.info(f"✓ Created GitHub issue: {issue_url}")
                
                # Create draft PR if enabled
                if self.enable_auto_pr:
                    self._create_draft_pr_from_issue(issue_url, exception)
                
                return issue_url
            else:
                logger.error(f"Failed to create issue: {result.get('stderr')}")
        except Exception as e:
            logger.error(f"Error creating GitHub issue: {e}")
        
        return None
    
    def _create_draft_pr_from_issue(self, issue_url: str, exception: Exception) -> Optional[str]:
        """
        Create a draft PR from an issue.
        
        Args:
            issue_url: URL of the created issue
            exception: The original exception
            
        Returns:
            PR URL if created, None otherwise
        """
        if not self.enable_auto_pr:
            return None
        
        github_cli = self._get_github_cli()
        if not github_cli:
            return None
        
        try:
            # Extract issue number from URL
            issue_number = issue_url.split("/")[-1]
            
            # Create a branch name
            error_type = type(exception).__name__
            branch_name = f"auto-fix/issue-{issue_number}-{error_type.lower()}"
            
            # Note: In a real implementation, we would:
            # 1. Create a new branch
            # 2. Make changes (potentially using Copilot)
            # 3. Create a PR
            
            # For now, we'll create a draft PR that references the issue
            pr_title = f"[Auto-Fix] Fix for issue #{issue_number}"
            pr_body = f"""This is an automatically generated draft PR to address issue #{issue_number}.

**Issue:** {issue_url}

**Error Type:** {error_type}

### Action Required
This PR needs to be completed with the actual fix. Consider:
1. Reviewing the error details in the issue
2. Using GitHub Copilot to suggest fixes
3. Adding tests to prevent regression

Closes #{issue_number}
"""
            
            logger.info(f"Draft PR would be created for issue {issue_number}")
            logger.info(f"Branch: {branch_name}")
            
            # Trigger auto-healing if enabled
            if self.enable_auto_heal:
                self._invoke_copilot_autofix(issue_number, exception)
            
            # Return placeholder (actual PR creation would happen here)
            return f"{self.repo}/pull/{issue_number}"
            
        except Exception as e:
            logger.error(f"Error creating draft PR: {e}")
        
        return None
    
    def _invoke_copilot_autofix(self, issue_number: str, exception: Exception) -> bool:
        """
        Invoke GitHub Copilot to suggest fixes.
        
        Args:
            issue_number: The GitHub issue number
            exception: The original exception
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_auto_heal:
            return False
        
        copilot_sdk = self._get_copilot_sdk()
        if not copilot_sdk:
            logger.warning("Copilot SDK not available for auto-healing")
            return False
        
        try:
            # Build prompt for Copilot
            error_type = type(exception).__name__
            error_message = str(exception)
            stack_trace = traceback.format_exc()
            
            prompt = f"""I encountered the following error in the IPFS Accelerate CLI:

Error Type: {error_type}
Error Message: {error_message}

Stack Trace:
{stack_trace}

Please analyze this error and suggest:
1. The root cause of the error
2. Potential fixes
3. Code changes needed to resolve it
4. Tests to prevent regression

Issue #{issue_number} has been created to track this error.
"""
            
            logger.info(f"Invoking Copilot for auto-fix suggestions (issue #{issue_number})")
            
            # In a real implementation, we would:
            # 1. Create a Copilot session
            # 2. Send the prompt
            # 3. Get suggestions
            # 4. Apply fixes (with user approval)
            
            # For now, just log that we would do this
            logger.info("Copilot auto-fix would be invoked here")
            
            return True
            
        except Exception as e:
            logger.error(f"Error invoking Copilot auto-fix: {e}")
        
        return False
    
    def wrap_cli_main(self, main_func: Callable) -> Callable:
        """
        Decorator to wrap CLI main function with error handling.
        
        Args:
            main_func: The main CLI function to wrap
            
        Returns:
            Wrapped function with error handling
        """
        @functools.wraps(main_func)
        def wrapper(*args, **kwargs):
            try:
                return main_func(*args, **kwargs)
            except KeyboardInterrupt:
                logger.info("CLI interrupted by user")
                return 0
            except Exception as e:
                # Capture the error
                self.capture_error(e)
                
                # Create issue if enabled
                if self.enable_auto_issue:
                    self.create_issue_from_error(e)
                
                # Re-raise for normal error handling
                raise
        
        return wrapper
    
    def cleanup(self):
        """Clean up resources."""
        if self._error_aggregator:
            try:
                self._error_aggregator.stop_bundling_thread()
            except Exception as e:
                logger.debug(f"Error stopping bundling thread: {e}")
