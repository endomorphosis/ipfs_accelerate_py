#!/usr/bin/env python3
"""
CI/CD Integration Plugin for Distributed Testing Framework

This plugin provides seamless integration with CI/CD systems including GitHub Actions,
Jenkins, and GitLab CI. It enables the distributed testing framework to report results
back to CI systems and coordinate test execution within CI/CD pipelines.
"""

import asyncio
import json
import logging
import os
import platform
import requests
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path

# Import plugin base class
from distributed_testing.plugin_architecture import Plugin, PluginType, HookType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CICDIntegrationPlugin(Plugin):
    """
    CI/CD Integration Plugin for the Distributed Testing Framework.
    
    This plugin provides integration with popular CI/CD systems:
    - GitHub Actions
    - Jenkins
    - GitLab CI
    - Azure DevOps
    
    It enables the following capabilities:
    - Test execution results reported to CI/CD systems
    - Build status updates
    - Test summary generation
    - Artifact management
    - Environment variable and secret management
    - Integration with pull request workflows
    """
    
    def __init__(self):
        """Initialize the plugin."""
        super().__init__(
            name="CICDIntegration",
            version="1.0.0",
            plugin_type=PluginType.INTEGRATION
        )
        
        # CI system client (properly initialized in initialize())
        self.ci_client = None
        self.ci_environment = None
        
        # Test run tracking
        self.test_run = {
            "id": None,
            "start_time": None,
            "end_time": None,
            "status": "not_started",
            "tasks": {},
            "summary": {},
            "artifacts": []
        }
        
        # Default configuration
        self.config = {
            "ci_system": "auto",  # auto, github, jenkins, gitlab, azure
            "api_url": None,
            "api_token": None,
            "project": None,
            "repository": None,
            "build_id": None,
            "commit_sha": None,
            "branch": None,
            "pr_number": None,
            "update_interval": 60,
            "update_on_completion_only": False,
            "artifact_dir": "distributed_test_results",
            "result_format": "junit",  # junit, json, html
            "enable_status_updates": True,
            "enable_pr_comments": True,
            "enable_artifacts": True,
            "detailed_logging": False
        }
        
        # Register hooks
        self.register_hook(HookType.COORDINATOR_STARTUP, self.on_coordinator_startup)
        self.register_hook(HookType.COORDINATOR_SHUTDOWN, self.on_coordinator_shutdown)
        self.register_hook(HookType.TASK_CREATED, self.on_task_created)
        self.register_hook(HookType.TASK_COMPLETED, self.on_task_completed)
        self.register_hook(HookType.TASK_FAILED, self.on_task_failed)
        
        logger.info("CICDIntegrationPlugin initialized")
    
    async def initialize(self, coordinator) -> bool:
        """
        Initialize the plugin with reference to the coordinator.
        
        Args:
            coordinator: Reference to the coordinator instance
            
        Returns:
            True if initialization succeeded
        """
        # Store coordinator reference
        self.coordinator = coordinator
        
        # Detect CI environment
        self.ci_environment = await self._detect_ci_environment()
        
        if self.ci_environment:
            logger.info(f"Detected CI environment: {self.ci_environment['type']}")
            
            # Update config with detected environment
            self.config.update({
                "ci_system": self.ci_environment["type"],
                "api_url": self.ci_environment.get("api_url"),
                "project": self.ci_environment.get("project"),
                "repository": self.ci_environment.get("repository"),
                "build_id": self.ci_environment.get("build_id"),
                "commit_sha": self.ci_environment.get("commit_sha"),
                "branch": self.ci_environment.get("branch"),
                "pr_number": self.ci_environment.get("pr_number")
            })
            
            # Initialize CI client
            await self._initialize_ci_client()
            
            # Create artifact directory if needed
            if self.config["enable_artifacts"]:
                os.makedirs(self.config["artifact_dir"], exist_ok=True)
            
            # Start periodic update task if enabled
            if self.config["enable_status_updates"] and not self.config["update_on_completion_only"]:
                self.update_task = asyncio.create_task(self._periodic_updates())
        else:
            logger.warning("No CI environment detected, plugin will operate in limited mode")
        
        logger.info("CICDIntegrationPlugin initialized with coordinator")
        return True
    
    async def shutdown(self) -> bool:
        """
        Shutdown the plugin.
        
        Returns:
            True if shutdown succeeded
        """
        # Cancel update task if running
        if hasattr(self, "update_task") and self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        # Send final report to CI system
        if self.ci_client and self.test_run["id"]:
            await self._send_final_report()
            
            # Upload artifacts if enabled
            if self.config["enable_artifacts"]:
                await self._upload_artifacts()
        
        logger.info("CICDIntegrationPlugin shutdown complete")
        return True
    
    async def _detect_ci_environment(self) -> Optional[Dict[str, Any]]:
        """
        Detect the CI environment from environment variables.
        
        Returns:
            Dictionary with CI environment information
        """
        # GitHub Actions
        if os.environ.get("GITHUB_ACTIONS") == "true":
            return {
                "type": "github",
                "api_url": "https://api.github.com",
                "repository": os.environ.get("GITHUB_REPOSITORY"),
                "build_id": os.environ.get("GITHUB_RUN_ID"),
                "commit_sha": os.environ.get("GITHUB_SHA"),
                "branch": os.environ.get("GITHUB_REF"),
                "workflow": os.environ.get("GITHUB_WORKFLOW"),
                "actor": os.environ.get("GITHUB_ACTOR"),
                "event_name": os.environ.get("GITHUB_EVENT_NAME"),
                "pr_number": self._extract_pr_number_from_github()
            }
            
        # Jenkins
        elif os.environ.get("JENKINS_URL"):
            return {
                "type": "jenkins",
                "api_url": os.environ.get("JENKINS_URL"),
                "build_id": os.environ.get("BUILD_ID"),
                "job_name": os.environ.get("JOB_NAME"),
                "build_url": os.environ.get("BUILD_URL"),
                "branch": os.environ.get("BRANCH_NAME"),
                "commit_sha": os.environ.get("GIT_COMMIT"),
                "workspace": os.environ.get("WORKSPACE")
            }
            
        # GitLab CI
        elif os.environ.get("GITLAB_CI") == "true":
            return {
                "type": "gitlab",
                "api_url": "https://gitlab.com/api/v4",
                "project": os.environ.get("CI_PROJECT_PATH"),
                "repository": os.environ.get("CI_PROJECT_PATH"),
                "build_id": os.environ.get("CI_JOB_ID"),
                "commit_sha": os.environ.get("CI_COMMIT_SHA"),
                "branch": os.environ.get("CI_COMMIT_REF_NAME"),
                "pipeline_id": os.environ.get("CI_PIPELINE_ID"),
                "pr_number": os.environ.get("CI_MERGE_REQUEST_IID")
            }
            
        # Azure DevOps Pipelines
        elif os.environ.get("TF_BUILD") == "True":
            return {
                "type": "azure",
                "api_url": os.environ.get("SYSTEM_COLLECTIONURI"),
                "project": os.environ.get("SYSTEM_TEAMPROJECT"),
                "build_id": os.environ.get("BUILD_BUILDID"),
                "repository": os.environ.get("BUILD_REPOSITORY_NAME"),
                "commit_sha": os.environ.get("BUILD_SOURCEVERSION"),
                "branch": os.environ.get("BUILD_SOURCEBRANCHNAME"),
                "pr_number": os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTID")
            }
            
        # No CI environment detected
        return None
    
    def _extract_pr_number_from_github(self) -> Optional[str]:
        """
        Extract PR number from GitHub environment.
        
        Returns:
            PR number as string or None
        """
        # For pull request events, the PR number is in GITHUB_REF
        # Format: refs/pull/{PR_NUMBER}/merge
        github_ref = os.environ.get("GITHUB_REF", "")
        if github_ref.startswith("refs/pull/"):
            parts = github_ref.split("/")
            if len(parts) >= 3:
                return parts[2]
        
        # For pull request events, can also check the event payload
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if event_path and os.path.exists(event_path):
            try:
                with open(event_path, "r") as f:
                    event_data = json.load(f)
                    if "pull_request" in event_data:
                        return str(event_data["pull_request"]["number"])
            except Exception as e:
                logger.error(f"Error parsing GitHub event data: {e}")
        
        return None
    
    async def _initialize_ci_client(self):
        """Initialize the CI system client."""
        ci_system = self.config["ci_system"]
        
        logger.info(f"Initializing {ci_system} CI client")
        
        if ci_system == "github":
            from distributed_testing.ci.github_client import GitHubClient
            
            # Get token from environment or config
            token = os.environ.get("GITHUB_TOKEN") or self.config["api_token"]
            
            if token:
                self.ci_client = GitHubClient(
                    token=token,
                    repository=self.config["repository"],
                    api_url=self.config["api_url"]
                )
            else:
                logger.warning("No GitHub token available, status updates will be limited")
                self.ci_client = {"type": "github", "limited": True}
            
        elif ci_system == "jenkins":
            from distributed_testing.ci.jenkins_client import JenkinsClient
            
            # Get credentials from environment or config
            user = os.environ.get("JENKINS_USER") or self.config.get("jenkins_user")
            token = os.environ.get("JENKINS_TOKEN") or self.config["api_token"]
            
            if user and token:
                self.ci_client = JenkinsClient(
                    url=self.config["api_url"],
                    user=user,
                    token=token
                )
            else:
                logger.warning("No Jenkins credentials available, status updates will be limited")
                self.ci_client = {"type": "jenkins", "limited": True}
            
        elif ci_system == "gitlab":
            from distributed_testing.ci.gitlab_client import GitLabClient
            
            # Get token from environment or config
            token = os.environ.get("GITLAB_TOKEN") or self.config["api_token"]
            
            if token:
                self.ci_client = GitLabClient(
                    token=token,
                    project=self.config["project"],
                    api_url=self.config["api_url"]
                )
            else:
                logger.warning("No GitLab token available, status updates will be limited")
                self.ci_client = {"type": "gitlab", "limited": True}
            
        elif ci_system == "azure":
            from distributed_testing.ci.azure_client import AzureClient
            
            # Get token from environment or config
            token = os.environ.get("AZURE_DEVOPS_TOKEN") or self.config["api_token"]
            
            if token:
                self.ci_client = AzureClient(
                    token=token,
                    organization=self.config["api_url"],
                    project=self.config["project"]
                )
            else:
                logger.warning("No Azure DevOps token available, status updates will be limited")
                self.ci_client = {"type": "azure", "limited": True}
            
        else:
            logger.warning(f"Unsupported CI system: {ci_system}")
            self.ci_client = {"type": "unknown", "limited": True}
        
        # Create test run in CI system
        await self._create_test_run()
    
    async def _create_test_run(self):
        """Create a new test run in the CI system."""
        if not self.ci_client:
            return
            
        logger.info(f"Creating test run in {self.config['ci_system']} CI system")
        
        try:
            # If we have a real CI client
            if not isinstance(self.ci_client, dict):
                test_run = await self.ci_client.create_test_run({
                    "name": f"Distributed Test Run - {datetime.now().isoformat()}",
                    "build_id": self.config["build_id"],
                    "commit_sha": self.config["commit_sha"],
                    "branch": self.config["branch"],
                    "pr_number": self.config["pr_number"],
                    "system_info": {
                        "python_version": platform.python_version(),
                        "platform": platform.platform(),
                        "hostname": platform.node()
                    }
                })
                
                if test_run:
                    self.test_run.update(test_run)
            else:
                # Simulate creating test run
                self.test_run["id"] = f"run-{int(time.time())}"
                
            # Set start time
            self.test_run["start_time"] = datetime.now().isoformat()
            self.test_run["status"] = "running"
            
            logger.info(f"Created test run {self.test_run['id']}")
            
        except Exception as e:
            logger.error(f"Error creating test run: {e}")
    
    async def _send_final_report(self):
        """Send final test report to CI system."""
        if not self.ci_client or not self.test_run["id"]:
            return
            
        logger.info(f"Sending final report to {self.config['ci_system']} CI system")
        
        try:
            # Update test run
            self.test_run["end_time"] = datetime.now().isoformat()
            
            # Calculate summary
            task_statuses = {}
            
            for task_id, task in self.test_run["tasks"].items():
                status = task["status"]
                
                if status not in task_statuses:
                    task_statuses[status] = 0
                    
                task_statuses[status] += 1
            
            # Determine overall status
            if task_statuses.get("failed", 0) > 0:
                self.test_run["status"] = "failed"
            else:
                self.test_run["status"] = "completed"
            
            # Update summary
            self.test_run["summary"] = {
                "total_tasks": len(self.test_run["tasks"]),
                "task_statuses": task_statuses,
                "duration": (
                    datetime.fromisoformat(self.test_run["end_time"]) - 
                    datetime.fromisoformat(self.test_run["start_time"])
                ).total_seconds()
            }
            
            # Generate report files
            report_files = await self._generate_reports()
            
            # Add report files to artifacts
            for report_file in report_files:
                self.test_run["artifacts"].append({
                    "path": report_file,
                    "name": os.path.basename(report_file),
                    "type": report_file.split(".")[-1]
                })
            
            # If we have a real CI client
            if not isinstance(self.ci_client, dict):
                await self.ci_client.update_test_run(
                    self.test_run["id"],
                    {
                        "status": self.test_run["status"],
                        "summary": self.test_run["summary"],
                        "end_time": self.test_run["end_time"]
                    }
                )
                
                # Add PR comment if enabled and applicable
                if (self.config["enable_pr_comments"] and 
                    self.config["pr_number"] and 
                    hasattr(self.ci_client, "add_pr_comment")):
                    
                    comment = self._generate_pr_comment()
                    await self.ci_client.add_pr_comment(
                        self.config["pr_number"],
                        comment
                    )
            
            logger.info(f"Sent final report for test run {self.test_run['id']}, status: {self.test_run['status']}")
            
        except Exception as e:
            logger.error(f"Error sending final report: {e}")
    
    async def _generate_reports(self) -> List[str]:
        """
        Generate test reports in different formats.
        
        Returns:
            List of report file paths
        """
        report_files = []
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ensure artifact directory exists
            os.makedirs(self.config["artifact_dir"], exist_ok=True)
            
            # Generate JSON report
            json_path = os.path.join(self.config["artifact_dir"], f"test_report_{timestamp}.json")
            with open(json_path, "w") as f:
                json.dump({
                    "test_run": self.test_run,
                    "config": {k: v for k, v in self.config.items() if k != "api_token"}
                }, f, indent=2)
            report_files.append(json_path)
            
            # Generate JUnit XML report if requested
            if self.config["result_format"] == "junit" or self.config["result_format"] == "all":
                junit_path = os.path.join(self.config["artifact_dir"], f"test_report_{timestamp}.xml")
                await self._generate_junit_report(junit_path)
                report_files.append(junit_path)
            
            # Generate HTML report if requested
            if self.config["result_format"] == "html" or self.config["result_format"] == "all":
                html_path = os.path.join(self.config["artifact_dir"], f"test_report_{timestamp}.html")
                await self._generate_html_report(html_path)
                report_files.append(html_path)
            
            logger.info(f"Generated {len(report_files)} report files")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
        
        return report_files
    
    async def _generate_junit_report(self, output_path: str):
        """
        Generate JUnit XML report.
        
        Args:
            output_path: Path to output file
        """
        try:
            # Simple JUnit XML structure
            xml_lines = [
                '<?xml version="1.0" encoding="UTF-8"?>',
                f'<testsuites name="Distributed Test Run {self.test_run["id"]}" '
                f'tests="{self.test_run["summary"]["total_tasks"]}" '
                f'failures="{self.test_run["summary"]["task_statuses"].get("failed", 0)}" '
                f'errors="0" '
                f'time="{self.test_run["summary"]["duration"]}">'
            ]
            
            # Group tasks by type
            tasks_by_type = {}
            for task_id, task in self.test_run["tasks"].items():
                task_type = task["data"].get("type", "unknown")
                if task_type not in tasks_by_type:
                    tasks_by_type[task_type] = []
                tasks_by_type[task_type].append((task_id, task))
            
            # Create testsuite for each task type
            for task_type, tasks in tasks_by_type.items():
                failures = sum(1 for _, task in tasks if task["status"] == "failed")
                total_time = sum(
                    (datetime.fromisoformat(task["completed_at" if "completed_at" in task else "failed_at"]) - 
                     datetime.fromisoformat(task["created_at"])).total_seconds()
                    for _, task in tasks
                    if "created_at" in task and ("completed_at" in task or "failed_at" in task)
                )
                
                xml_lines.append(
                    f'  <testsuite name="{task_type}" '
                    f'tests="{len(tasks)}" '
                    f'failures="{failures}" '
                    f'errors="0" '
                    f'time="{total_time}">'
                )
                
                # Add testcase for each task
                for task_id, task in tasks:
                    task_name = task["data"].get("name", task_id)
                    
                    # Calculate task duration if possible
                    task_time = 0
                    if "created_at" in task and ("completed_at" in task or "failed_at" in task):
                        end_time = task.get("completed_at", task.get("failed_at"))
                        task_time = (
                            datetime.fromisoformat(end_time) - 
                            datetime.fromisoformat(task["created_at"])
                        ).total_seconds()
                    
                    xml_lines.append(f'    <testcase name="{task_name}" classname="{task_type}" time="{task_time}">')
                    
                    # Add failure information if task failed
                    if task["status"] == "failed":
                        error_message = task.get("error", "Unknown error")
                        xml_lines.append(f'      <failure message="{error_message}"></failure>')
                    
                    xml_lines.append('    </testcase>')
                
                xml_lines.append('  </testsuite>')
            
            xml_lines.append('</testsuites>')
            
            # Write to file
            with open(output_path, "w") as f:
                f.write("\n".join(xml_lines))
            
            logger.info(f"Generated JUnit XML report: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating JUnit XML report: {e}")
    
    async def _generate_html_report(self, output_path: str):
        """
        Generate HTML report.
        
        Args:
            output_path: Path to output file
        """
        try:
            # Simple HTML report
            html_lines = [
                '<!DOCTYPE html>',
                '<html>',
                '<head>',
                f'  <title>Distributed Test Run {self.test_run["id"]}</title>',
                '  <style>',
                '    body { font-family: Arial, sans-serif; margin: 20px; }',
                '    .header { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }',
                '    .summary { margin: 20px 0; }',
                '    .tasks { margin: 20px 0; }',
                '    table { border-collapse: collapse; width: 100%; }',
                '    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
                '    tr:nth-child(even) { background-color: #f2f2f2; }',
                '    th { background-color: #4CAF50; color: white; }',
                '    .completed { color: green; }',
                '    .failed { color: red; }',
                '    .created { color: blue; }',
                '  </style>',
                '</head>',
                '<body>',
                '  <div class="header">',
                f'    <h1>Distributed Test Run {self.test_run["id"]}</h1>',
                f'    <p>Status: <strong>{self.test_run["status"]}</strong></p>',
                f'    <p>Start time: {self.test_run["start_time"]}</p>',
                f'    <p>End time: {self.test_run["end_time"]}</p>',
                f'    <p>Duration: {self.test_run["summary"]["duration"]:.2f} seconds</p>',
                '  </div>',
                '  <div class="summary">',
                '    <h2>Summary</h2>',
                '    <table>',
                '      <tr><th>Metric</th><th>Value</th></tr>'
            ]
            
            # Add summary information
            html_lines.append(f'      <tr><td>Total tasks</td><td>{self.test_run["summary"]["total_tasks"]}</td></tr>')
            
            for status, count in self.test_run["summary"]["task_statuses"].items():
                html_lines.append(f'      <tr><td>{status.capitalize()} tasks</td><td>{count}</td></tr>')
            
            html_lines.extend([
                '    </table>',
                '  </div>',
                '  <div class="tasks">',
                '    <h2>Tasks</h2>',
                '    <table>',
                '      <tr>',
                '        <th>ID</th>',
                '        <th>Name</th>',
                '        <th>Type</th>',
                '        <th>Status</th>',
                '        <th>Created</th>',
                '        <th>Completed</th>',
                '        <th>Duration</th>',
                '      </tr>'
            ])
            
            # Add task information
            for task_id, task in self.test_run["tasks"].items():
                task_name = task["data"].get("name", task_id)
                task_type = task["data"].get("type", "unknown")
                status = task["status"]
                created = task.get("created_at", "")
                
                # Get completion time and calculate duration
                completed = ""
                duration = ""
                if "completed_at" in task:
                    completed = task["completed_at"]
                    created_dt = datetime.fromisoformat(task["created_at"])
                    completed_dt = datetime.fromisoformat(completed)
                    duration = f"{(completed_dt - created_dt).total_seconds():.2f}s"
                elif "failed_at" in task:
                    completed = task["failed_at"]
                    created_dt = datetime.fromisoformat(task["created_at"])
                    completed_dt = datetime.fromisoformat(completed)
                    duration = f"{(completed_dt - created_dt).total_seconds():.2f}s"
                
                html_lines.append(
                    f'      <tr>'
                    f'        <td>{task_id}</td>'
                    f'        <td>{task_name}</td>'
                    f'        <td>{task_type}</td>'
                    f'        <td class="{status}">{status}</td>'
                    f'        <td>{created}</td>'
                    f'        <td>{completed}</td>'
                    f'        <td>{duration}</td>'
                    f'      </tr>'
                )
                
                # Add error message if task failed
                if status == "failed" and "error" in task:
                    html_lines.append(
                        f'      <tr>'
                        f'        <td colspan="7" class="failed">'
                        f'          <strong>Error:</strong> {task["error"]}'
                        f'        </td>'
                        f'      </tr>'
                    )
            
            html_lines.extend([
                '    </table>',
                '  </div>',
                '  <div class="footer">',
                f'    <p>Generated by Distributed Testing Framework on {datetime.now().isoformat()}</p>',
                '  </div>',
                '</body>',
                '</html>'
            ])
            
            # Write to file
            with open(output_path, "w") as f:
                f.write("\n".join(html_lines))
            
            logger.info(f"Generated HTML report: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
    
    def _generate_pr_comment(self) -> str:
        """
        Generate comment for pull request.
        
        Returns:
            Markdown comment content
        """
        # Generate PR comment in markdown format
        summary = self.test_run["summary"]
        total_tasks = summary["total_tasks"]
        completed_tasks = summary["task_statuses"].get("completed", 0)
        failed_tasks = summary["task_statuses"].get("failed", 0)
        duration = summary["duration"]
        
        # Create status emoji
        status_emoji = "✅" if self.test_run["status"] == "completed" else "❌"
        
        comment = [
            f"## Distributed Test Results {status_emoji}",
            "",
            f"**Test Run:** {self.test_run['id']}",
            f"**Status:** {self.test_run['status'].upper()}",
            f"**Duration:** {duration:.2f} seconds",
            "",
            "### Summary",
            "",
            f"- **Total tasks:** {total_tasks}",
            f"- **Completed tasks:** {completed_tasks}",
            f"- **Failed tasks:** {failed_tasks}",
            ""
        ]
        
        # Add failed tasks section if any
        if failed_tasks > 0:
            comment.extend([
                "### Failed Tasks",
                "",
                "| Task | Error |",
                "| ---- | ----- |"
            ])
            
            # Add failed task details
            for task_id, task in self.test_run["tasks"].items():
                if task["status"] == "failed":
                    task_name = task["data"].get("name", task_id)
                    error = task.get("error", "Unknown error")
                    comment.append(f"| {task_name} | {error} |")
            
            comment.append("")
        
        # Add build information
        comment.extend([
            "### Build Information",
            "",
            f"- **Branch:** {self.config['branch']}",
            f"- **Commit:** {self.config['commit_sha'][:8] if self.config['commit_sha'] else 'Unknown'}",
            f"- **Build ID:** {self.config['build_id']}",
            ""
        ])
        
        # Add footer
        comment.append(
            "_This comment was generated automatically by the Distributed Testing Framework_"
        )
        
        return "\n".join(comment)
    
    async def _upload_artifacts(self):
        """Upload artifacts to CI system."""
        if not self.ci_client or not self.test_run["id"] or not self.test_run["artifacts"]:
            return
            
        logger.info(f"Uploading {len(self.test_run['artifacts'])} artifacts to {self.config['ci_system']} CI system")
        
        try:
            # If we have a real CI client with upload_artifact method
            if not isinstance(self.ci_client, dict) and hasattr(self.ci_client, "upload_artifact"):
                for artifact in self.test_run["artifacts"]:
                    success = await self.ci_client.upload_artifact(
                        self.test_run["id"],
                        artifact["path"],
                        artifact["name"]
                    )
                    
                    if success:
                        logger.info(f"Uploaded artifact: {artifact['name']}")
                    else:
                        logger.warning(f"Failed to upload artifact: {artifact['name']}")
            else:
                logger.info("CI client does not support artifact upload or is in limited mode")
                
        except Exception as e:
            logger.error(f"Error uploading artifacts: {e}")
    
    async def _periodic_updates(self):
        """Send periodic updates to CI system."""
        while True:
            try:
                # Sleep for update interval
                await asyncio.sleep(self.config["update_interval"])
                
                # Skip if no CI client or test run
                if not self.ci_client or not self.test_run["id"]:
                    continue
                    
                # Skip if we're in limited mode
                if isinstance(self.ci_client, dict) and self.ci_client.get("limited", False):
                    continue
                
                logger.debug(f"Sending periodic update to {self.config['ci_system']} CI system")
                
                # Update task summary
                task_statuses = {}
                
                for task_id, task in self.test_run["tasks"].items():
                    status = task["status"]
                    
                    if status not in task_statuses:
                        task_statuses[status] = 0
                        
                    task_statuses[status] += 1
                
                # Update summary
                self.test_run["summary"] = {
                    "total_tasks": len(self.test_run["tasks"]),
                    "task_statuses": task_statuses
                }
                
                # Send update if we have a real CI client
                if not isinstance(self.ci_client, dict):
                    await self.ci_client.update_test_run(
                        self.test_run["id"],
                        {
                            "status": "running",
                            "summary": self.test_run["summary"]
                        }
                    )
                
                logger.debug(f"Sent update for test run {self.test_run['id']}")
                
            except asyncio.CancelledError:
                logger.info("Periodic update task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in periodic update: {e}")
    
    # Hook handlers
    
    async def on_coordinator_startup(self, coordinator):
        """
        Handle coordinator startup event.
        
        Args:
            coordinator: Coordinator instance
        """
        logger.info("Coordinator startup detected")
        
        # Create test run if not already created
        if not self.test_run["id"] and self.ci_client:
            await self._create_test_run()
    
    async def on_coordinator_shutdown(self, coordinator):
        """
        Handle coordinator shutdown event.
        
        Args:
            coordinator: Coordinator instance
        """
        logger.info("Coordinator shutdown detected")
        
        # Send final report
        if self.ci_client and self.test_run["id"]:
            await self._send_final_report()
    
    async def on_task_created(self, task_id: str, task_data: Dict[str, Any]):
        """
        Handle task created event.
        
        Args:
            task_id: Task ID
            task_data: Task data
        """
        # Add task to test run
        self.test_run["tasks"][task_id] = {
            "id": task_id,
            "created_at": datetime.now().isoformat(),
            "status": "created",
            "data": task_data
        }
        
        if self.config["detailed_logging"]:
            logger.info(f"Added task {task_id} to test run {self.test_run['id']}")
    
    async def on_task_completed(self, task_id: str, result: Any):
        """
        Handle task completed event.
        
        Args:
            task_id: Task ID
            result: Task result
        """
        # Update task in test run
        if task_id in self.test_run["tasks"]:
            self.test_run["tasks"][task_id]["status"] = "completed"
            self.test_run["tasks"][task_id]["completed_at"] = datetime.now().isoformat()
            self.test_run["tasks"][task_id]["result"] = result
            
            if self.config["detailed_logging"]:
                logger.info(f"Updated task {task_id} to completed in test run {self.test_run['id']}")
    
    async def on_task_failed(self, task_id: str, error: str):
        """
        Handle task failed event.
        
        Args:
            task_id: Task ID
            error: Error message
        """
        # Update task in test run
        if task_id in self.test_run["tasks"]:
            self.test_run["tasks"][task_id]["status"] = "failed"
            self.test_run["tasks"][task_id]["failed_at"] = datetime.now().isoformat()
            self.test_run["tasks"][task_id]["error"] = error
            
            if self.config["detailed_logging"]:
                logger.info(f"Updated task {task_id} to failed in test run {self.test_run['id']}")
    
    def get_ci_status(self) -> Dict[str, Any]:
        """
        Get the current CI status.
        
        Returns:
            Dictionary with CI status
        """
        return {
            "ci_system": self.config["ci_system"],
            "repository": self.config["repository"],
            "branch": self.config["branch"],
            "commit_sha": self.config["commit_sha"],
            "build_id": self.config["build_id"],
            "pr_number": self.config["pr_number"],
            "test_run_id": self.test_run["id"],
            "test_run_status": self.test_run["status"],
            "summary": self.test_run["summary"],
            "artifacts": [a["name"] for a in self.test_run["artifacts"]]
        }