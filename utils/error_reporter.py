#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Error Reporting System

This module provides functionality to automatically convert runtime errors
into GitHub issues for tracking and resolution.

Author: IPFS Accelerate Python Framework Team
"""

import os
import sys
import json
import logging
import traceback
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    import requests
    requests_available = True
except ImportError:
    requests_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorReporter:
    """
    Automated error reporting system that creates GitHub issues from runtime errors.
    """
    
    def __init__(self, 
                 github_token: Optional[str] = None,
                 github_repo: Optional[str] = None,
                 enabled: bool = True,
                 include_system_info: bool = True,
                 auto_label: bool = True,
                 cache_dir: Optional[str] = None):
        """
        Initialize the error reporter.
        
        Args:
            github_token: GitHub personal access token (or from env GITHUB_TOKEN)
            github_repo: Repository in format 'owner/repo' (or from env GITHUB_REPO)
            enabled: Whether error reporting is enabled
            include_system_info: Whether to include system information in reports
            auto_label: Whether to automatically add labels to issues
            cache_dir: Directory for error cache (default: ~/.ipfs_accelerate)
        """
        self.github_token = github_token or os.environ.get('GITHUB_TOKEN')
        self.github_repo = github_repo or os.environ.get('GITHUB_REPO')
        self.enabled = enabled and self.github_token and self.github_repo
        self.include_system_info = include_system_info
        self.auto_label = auto_label
        
        # Cache environment checks
        self._is_docker = os.path.exists('/.dockerenv')
        self._is_venv = sys.prefix != sys.base_prefix
        
        # Track reported errors to avoid duplicates
        self.reported_errors = set()
        cache_base = cache_dir or os.environ.get('ERROR_REPORTER_CACHE_DIR') or str(Path.home() / '.ipfs_accelerate')
        self.error_cache_file = Path(cache_base) / 'reported_errors.json'
        self._load_reported_errors()
        
        if not requests_available:
            logger.warning("requests library not available, error reporting disabled")
            self.enabled = False
        
        if self.enabled:
            logger.info(f"Error reporter initialized for {self.github_repo}")
        else:
            logger.info("Error reporter disabled (missing configuration or dependencies)")
    
    def _load_reported_errors(self):
        """Load the cache of previously reported errors"""
        try:
            if self.error_cache_file.exists():
                with open(self.error_cache_file, 'r') as f:
                    data = json.load(f)
                    self.reported_errors = set(data.get('error_hashes', []))
        except Exception as e:
            logger.warning(f"Failed to load reported errors cache: {e}")
    
    def _save_reported_errors(self):
        """Save the cache of reported errors"""
        try:
            self.error_cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.error_cache_file, 'w') as f:
                json.dump({
                    'error_hashes': list(self.reported_errors),
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save reported errors cache: {e}")
    
    def _compute_error_hash(self, error_info: Dict[str, Any]) -> str:
        """
        Compute a hash for the error to detect duplicates.
        
        Args:
            error_info: Error information dictionary
            
        Returns:
            Hash string identifying this error type
        """
        # Create a signature from error type, message, and first few stack frames
        signature_parts = [
            error_info.get('error_type', ''),
            error_info.get('error_message', ''),
            error_info.get('source_component', '')
        ]
        
        # Add first 3 stack frames for uniqueness
        traceback_lines = error_info.get('traceback', '').split('\n')
        for line in traceback_lines[:6]:  # First 3 frames (2 lines each)
            if line.strip():
                signature_parts.append(line.strip())
        
        signature = '|'.join(signature_parts)
        return hashlib.sha256(signature.encode()).hexdigest()[:16]
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """
        Gather system information for the error report.
        
        Returns:
            Dictionary with system information
        """
        import platform
        
        info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'architecture': platform.machine(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add environment info if available
        try:
            info['environment'] = {
                'docker': self._is_docker,
                'virtual_env': self._is_venv
            }
        except Exception:
            pass
        
        return info
    
    def _create_issue_body(self, error_info: Dict[str, Any]) -> str:
        """
        Create the issue body content.
        
        Args:
            error_info: Error information dictionary
            
        Returns:
            Formatted issue body as markdown
        """
        body_parts = [
            "## Automated Error Report",
            "",
            f"**Error Type:** `{error_info.get('error_type', 'Unknown')}`",
            f"**Component:** `{error_info.get('source_component', 'Unknown')}`",
            f"**Timestamp:** {error_info.get('timestamp', 'N/A')}",
            "",
            "### Error Message",
            "```",
            error_info.get('error_message', 'No message available'),
            "```",
            "",
            "### Traceback",
            "```python",
            error_info.get('traceback', 'No traceback available'),
            "```"
        ]
        
        # Add context if available
        if error_info.get('context'):
            body_parts.extend([
                "",
                "### Additional Context",
                "```json",
                json.dumps(error_info['context'], indent=2),
                "```"
            ])
        
        # Add system info if enabled
        if self.include_system_info and error_info.get('system_info'):
            body_parts.extend([
                "",
                "### System Information",
                "```json",
                json.dumps(error_info['system_info'], indent=2),
                "```"
            ])
        
        body_parts.extend([
            "",
            "---",
            "_This issue was automatically generated by the IPFS Accelerate error reporting system._"
        ])
        
        return "\n".join(body_parts)
    
    def _determine_labels(self, error_info: Dict[str, Any]) -> List[str]:
        """
        Determine appropriate labels for the issue.
        
        Args:
            error_info: Error information dictionary
            
        Returns:
            List of label names
        """
        labels = ['bug', 'automated-report']
        
        # Add component-specific labels
        component = error_info.get('source_component', '').lower()
        if 'mcp' in component or 'server' in component:
            labels.append('mcp-server')
        elif 'dashboard' in component or 'javascript' in component:
            labels.append('dashboard')
        elif 'docker' in component or 'container' in component:
            labels.append('docker')
        
        # Add priority labels based on error type
        error_type = error_info.get('error_type', '').lower()
        if any(critical in error_type for critical in ['crash', 'fatal', 'critical']):
            labels.append('priority:high')
        
        return labels
    
    def report_error(self,
                    exception: Optional[Exception] = None,
                    error_type: Optional[str] = None,
                    error_message: Optional[str] = None,
                    traceback_str: Optional[str] = None,
                    source_component: str = 'unknown',
                    context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Report an error by creating a GitHub issue.
        
        Args:
            exception: The exception object (if available)
            error_type: Type of error (or will be extracted from exception)
            error_message: Error message (or will be extracted from exception)
            traceback_str: Traceback string (or will be generated)
            source_component: Component where error occurred (e.g., 'mcp-server', 'dashboard', 'docker')
            context: Additional context information
            
        Returns:
            URL of created issue, or None if not created
        """
        if not self.enabled:
            logger.debug("Error reporting is disabled")
            return None
        
        try:
            # Build error information
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'source_component': source_component,
                'context': context or {}
            }
            
            # Extract error details
            if exception:
                error_info['error_type'] = type(exception).__name__
                error_info['error_message'] = str(exception)
                error_info['traceback'] = ''.join(traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                ))
            else:
                error_info['error_type'] = error_type or 'UnknownError'
                error_info['error_message'] = error_message or 'No message provided'
                error_info['traceback'] = traceback_str or 'No traceback available'
            
            # Add system info
            if self.include_system_info:
                error_info['system_info'] = self._gather_system_info()
            
            # Check if we've already reported this error
            error_hash = self._compute_error_hash(error_info)
            if error_hash in self.reported_errors:
                logger.info(f"Error {error_hash} already reported, skipping")
                return None
            
            # Create GitHub issue
            issue_url = self._create_github_issue(error_info)
            
            if issue_url:
                # Mark error as reported
                self.reported_errors.add(error_hash)
                self._save_reported_errors()
                logger.info(f"Error reported successfully: {issue_url}")
            
            return issue_url
            
        except Exception as e:
            logger.error(f"Failed to report error: {e}")
            return None
    
    def _create_github_issue(self, error_info: Dict[str, Any]) -> Optional[str]:
        """
        Create a GitHub issue for the error.
        
        Args:
            error_info: Error information dictionary
            
        Returns:
            URL of created issue, or None if failed
        """
        if not requests_available:
            return None
        
        try:
            # Prepare issue data
            title = f"[Auto] {error_info['error_type']}: {error_info['source_component']}"
            body = self._create_issue_body(error_info)
            labels = self._determine_labels(error_info) if self.auto_label else ['bug', 'automated-report']
            
            # Create issue via GitHub API
            api_url = f"https://api.github.com/repos/{self.github_repo}/issues"
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            data = {
                'title': title,
                'body': body,
                'labels': labels
            }
            
            response = requests.post(api_url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            
            issue_data = response.json()
            return issue_data.get('html_url')
            
        except Exception as e:
            logger.error(f"Failed to create GitHub issue: {e}")
            return None


# Global error reporter instance
_global_reporter: Optional[ErrorReporter] = None


def get_error_reporter() -> ErrorReporter:
    """
    Get or create the global error reporter instance.
    
    Returns:
        Global ErrorReporter instance
    """
    global _global_reporter
    if _global_reporter is None:
        _global_reporter = ErrorReporter()
    return _global_reporter


def report_error(**kwargs) -> Optional[str]:
    """
    Convenience function to report an error using the global reporter.
    
    Args:
        **kwargs: Arguments to pass to ErrorReporter.report_error()
        
    Returns:
        URL of created issue, or None if not created
    """
    return get_error_reporter().report_error(**kwargs)


def install_global_exception_handler(source_component: str = 'python'):
    """
    Install a global exception handler that reports uncaught exceptions.
    
    Args:
        source_component: Component name for error reports
    """
    original_excepthook = sys.excepthook
    
    def exception_handler(exc_type, exc_value, exc_traceback):
        # Call original handler
        original_excepthook(exc_type, exc_value, exc_traceback)
        
        # Report error
        if exc_type is not KeyboardInterrupt:
            reporter = get_error_reporter()
            reporter.report_error(
                exception=exc_value,
                source_component=source_component,
                context={
                    'uncaught_exception': True
                }
            )
    
    sys.excepthook = exception_handler
    logger.info(f"Installed global exception handler for {source_component}")
