#!/usr/bin/env python3
"""
CI Notification System for Hardware Monitoring Tests

This script sends notifications when hardware monitoring tests fail in the CI pipeline.
It supports multiple notification channels (email, Slack, etc.) and integrates with
the CI artifact system to include links to test reports.

Usage:
    python ci_notification.py [options]

Options:
    --test-status STATUS   Test status (success, failure)
    --test-report PATH     Path to test report
    --notification-config  Path to notification config file
    --channels CHANNELS    Comma-separated list of channels to notify
    --dry-run              Don't actually send notifications
    --verbose              Display verbose output
"""

import os
import sys
import json
import argparse
import logging
import smtplib
import requests
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("ci_notification")

# Default notification configuration
DEFAULT_CONFIG = {
    "email": {
        "enabled": False,
        "smtp_server": "smtp.example.com",
        "smtp_port": 587,
        "username": "",
        "password": "",
        "from_address": "ci@example.com",
        "to_addresses": [],
        "subject_template": "[{status}] Hardware Monitoring Tests - {date}",
        "body_template": """
Hardware Monitoring Test Results

Status: {status}
Date: {date}
Workflow: {workflow}
Run ID: {run_id}
Commit: {commit}

{summary}

View the full report: {report_url}
"""
    },
    "slack": {
        "enabled": False,
        "webhook_url": "",
        "channel": "#ci-notifications",
        "username": "CI Bot",
        "icon_emoji": ":robot_face:",
        "template": """
*Hardware Monitoring Test Results*

*Status*: {status}
*Date*: {date}
*Workflow*: {workflow}
*Run ID*: {run_id}
*Commit*: {commit}

{summary}

<{report_url}|View the full report>
"""
    },
    "github": {
        "enabled": False,
        "token": "",
        "repository": "owner/repo",
        "commit_status": True,
        "pr_comment": True,
        "template": """
## Hardware Monitoring Test Results

**Status**: {status}
**Date**: {date}
**Workflow**: {workflow}
**Run ID**: {run_id}

{summary}

[View the full report]({report_url})
"""
    }
}


class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, config):
        self.config = config
        self.enabled = config.get("enabled", False)
    
    def send(self, context):
        """Send notification using this channel."""
        if not self.enabled:
            logger.info(f"{self.__class__.__name__} is disabled, skipping")
            return False
        
        try:
            return self._send_impl(context)
        except Exception as e:
            logger.error(f"Error sending notification via {self.__class__.__name__}: {str(e)}")
            return False
    
    def _send_impl(self, context):
        """Implementation of sending notification. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _send_impl")


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def _send_impl(self, context):
        """Send notification via email."""
        smtp_server = self.config.get("smtp_server")
        smtp_port = self.config.get("smtp_port", 587)
        username = self.config.get("username")
        password = self.config.get("password")
        from_address = self.config.get("from_address")
        to_addresses = self.config.get("to_addresses", [])
        
        if not to_addresses:
            logger.warning("No recipient email addresses specified, skipping email notification")
            return False
        
        # Create message
        subject_template = self.config.get("subject_template")
        body_template = self.config.get("body_template")
        
        subject = subject_template.format(**context)
        body = body_template.format(**context)
        
        message = MIMEMultipart()
        message["From"] = from_address
        message["To"] = ", ".join(to_addresses)
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))
        
        # Send email
        if context.get("dry_run", False):
            logger.info(f"[DRY RUN] Would send email to {to_addresses}: {subject}")
            return True
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            if username and password:
                server.login(username, password)
            server.send_message(message)
        
        logger.info(f"Sent email notification to {to_addresses}")
        return True


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""
    
    def _send_impl(self, context):
        """Send notification via Slack webhook."""
        webhook_url = self.config.get("webhook_url")
        if not webhook_url:
            logger.warning("No Slack webhook URL specified, skipping Slack notification")
            return False
        
        channel = self.config.get("channel", "#ci-notifications")
        username = self.config.get("username", "CI Bot")
        icon_emoji = self.config.get("icon_emoji", ":robot_face:")
        template = self.config.get("template")
        
        # Create message
        text = template.format(**context)
        
        payload = {
            "channel": channel,
            "username": username,
            "text": text,
            "icon_emoji": icon_emoji
        }
        
        # Send to Slack
        if context.get("dry_run", False):
            logger.info(f"[DRY RUN] Would send Slack notification to {channel}: {text[:50]}...")
            return True
        
        response = requests.post(webhook_url, json=payload)
        success = response.status_code == 200
        
        if success:
            logger.info(f"Sent Slack notification to {channel}")
        else:
            logger.error(f"Failed to send Slack notification: {response.status_code} - {response.text}")
        
        return success


class GitHubNotificationChannel(NotificationChannel):
    """GitHub notification channel for status updates and PR comments."""
    
    def _send_impl(self, context):
        """Send notification via GitHub API."""
        token = self.config.get("token")
        repository = self.config.get("repository")
        
        if not token or not repository:
            logger.warning("Missing GitHub token or repository, skipping GitHub notification")
            return False
        
        success = True
        
        # Update commit status if enabled
        if self.config.get("commit_status", False):
            commit_sha = context.get("commit")
            if commit_sha:
                success = success and self._update_commit_status(token, repository, commit_sha, context)
            else:
                logger.warning("No commit SHA provided, skipping GitHub status update")
        
        # Add PR comment if enabled and PR number is provided
        if self.config.get("pr_comment", False):
            pr_number = context.get("pr_number")
            if pr_number:
                success = success and self._add_pr_comment(token, repository, pr_number, context)
            else:
                logger.warning("No PR number provided, skipping GitHub PR comment")
        
        return success
    
    def _update_commit_status(self, token, repository, commit_sha, context):
        """Update commit status on GitHub."""
        url = f"https://api.github.com/repos/{repository}/statuses/{commit_sha}"
        
        status_state = "success" if context.get("status") == "success" else "failure"
        
        payload = {
            "state": status_state,
            "target_url": context.get("report_url", ""),
            "description": f"Hardware monitoring tests {status_state}",
            "context": "hardware-monitoring/tests"
        }
        
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        if context.get("dry_run", False):
            logger.info(f"[DRY RUN] Would update GitHub status for {commit_sha} to {status_state}")
            return True
        
        response = requests.post(url, json=payload, headers=headers)
        success = response.status_code == 201
        
        if success:
            logger.info(f"Updated GitHub status for {commit_sha} to {status_state}")
        else:
            logger.error(f"Failed to update GitHub status: {response.status_code} - {response.text}")
        
        return success
    
    def _add_pr_comment(self, token, repository, pr_number, context):
        """Add comment to a PR on GitHub."""
        url = f"https://api.github.com/repos/{repository}/issues/{pr_number}/comments"
        
        template = self.config.get("template")
        body = template.format(**context)
        
        payload = {"body": body}
        
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        if context.get("dry_run", False):
            logger.info(f"[DRY RUN] Would add GitHub PR comment to PR #{pr_number}")
            return True
        
        response = requests.post(url, json=payload, headers=headers)
        success = response.status_code == 201
        
        if success:
            logger.info(f"Added GitHub PR comment to PR #{pr_number}")
        else:
            logger.error(f"Failed to add GitHub PR comment: {response.status_code} - {response.text}")
        
        return success


def get_channel_instance(channel_type, config):
    """Get notification channel instance based on type."""
    if channel_type == "email":
        return EmailNotificationChannel(config.get("email", {}))
    elif channel_type == "slack":
        return SlackNotificationChannel(config.get("slack", {}))
    elif channel_type == "github":
        return GitHubNotificationChannel(config.get("github", {}))
    else:
        logger.warning(f"Unknown notification channel type: {channel_type}")
        return None


def build_notification_context(args, config):
    """Build notification context with information to use in templates."""
    # Get CI environment variables
    github_workflow = os.environ.get("GITHUB_WORKFLOW", "Unknown")
    github_run_id = os.environ.get("GITHUB_RUN_ID", "Unknown")
    github_sha = os.environ.get("GITHUB_SHA", "Unknown")
    github_ref = os.environ.get("GITHUB_REF", "Unknown")
    github_repository = os.environ.get("GITHUB_REPOSITORY", "Unknown")
    github_pr_number = None
    
    # Extract PR number from ref if it's a PR
    if github_ref.startswith("refs/pull/") and github_ref.endswith("/merge"):
        try:
            github_pr_number = github_ref.split("/")[2]
        except IndexError:
            pass
    
    # Try to get test report summary if provided
    summary = ""
    if args.test_report and os.path.exists(args.test_report):
        try:
            # Extract basic info from HTML report
            with open(args.test_report, "r") as f:
                report_content = f.read()
                
                # Very simple extraction - could be improved with proper HTML parsing
                if "Tests Run:" in report_content:
                    tests_run = report_content.split("Tests Run:")[1].split("</p>")[0].strip()
                    failures = report_content.split("Failures:")[1].split("</p>")[0].strip()
                    errors = report_content.split("Errors:")[1].split("</p>")[0].strip()
                    
                    summary = f"Tests Run: {tests_run}\nFailures: {failures}\nErrors: {errors}"
        except Exception as e:
            logger.warning(f"Failed to extract summary from test report: {str(e)}")
            summary = "Failed to extract test summary from report."
    
    # Build report URL
    report_url = ""
    if args.test_report:
        if github_run_id != "Unknown":
            # For GitHub Actions
            report_url = f"https://github.com/{github_repository}/actions/runs/{github_run_id}/artifacts"
        else:
            # Fallback to local path
            report_url = os.path.abspath(args.test_report)
    
    # Build context
    context = {
        "status": args.test_status,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "workflow": github_workflow,
        "run_id": github_run_id,
        "commit": github_sha,
        "repository": github_repository,
        "pr_number": github_pr_number,
        "ref": github_ref,
        "summary": summary,
        "report_url": report_url,
        "dry_run": args.dry_run
    }
    
    return context


def send_notifications(args, config):
    """Send notifications to all enabled channels."""
    # Build notification context
    context = build_notification_context(args, config)
    
    # Determine which channels to notify
    channels_to_notify = []
    if args.channels:
        channels_to_notify = [c.strip() for c in args.channels.split(",")]
    else:
        # Use all enabled channels from config
        for channel_type, channel_config in config.items():
            if channel_config.get("enabled", False):
                channels_to_notify.append(channel_type)
    
    if not channels_to_notify:
        logger.warning("No notification channels enabled, nothing to do")
        return False
    
    # Send notifications
    success = True
    for channel_type in channels_to_notify:
        channel = get_channel_instance(channel_type, config)
        if channel:
            channel_success = channel.send(context)
            success = success and channel_success
    
    return success


def load_config(config_path=None):
    """Load notification configuration from file or use defaults."""
    config = DEFAULT_CONFIG
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                
                # Merge with defaults
                for channel, channel_config in loaded_config.items():
                    if channel in config:
                        config[channel].update(channel_config)
                    else:
                        config[channel] = channel_config
            
            logger.info(f"Loaded notification config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {str(e)}")
    
    return config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CI Notification System for Hardware Monitoring Tests")
    parser.add_argument("--test-status", choices=["success", "failure"], default="success",
                        help="Test status (success, failure)")
    parser.add_argument("--test-report", help="Path to test report")
    parser.add_argument("--notification-config", help="Path to notification config file")
    parser.add_argument("--channels", help="Comma-separated list of channels to notify")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually send notifications")
    parser.add_argument("--verbose", action="store_true", help="Display verbose output")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Configure verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.notification_config)
    
    # Send notifications
    success = send_notifications(args, config)
    
    # Return exit code based on success
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())