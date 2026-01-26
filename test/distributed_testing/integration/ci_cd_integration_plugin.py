#!/usr/bin/env python3
"""
CI/CD Integration Plugin for Distributed Testing Framework

This plugin provides seamless integration with CI/CD systems including GitHub Actions,
Jenkins, GitLab CI, and Azure DevOps. It enables the distributed testing framework to 
report results back to CI systems, coordinate test execution within CI/CD pipelines,
and manage test artifacts and reporting.

Enhanced Features:
- Comprehensive CI/CD system detection and integration
- Standardized API for interacting with different CI/CD systems
- Advanced reporting with customizable formats
- Artifact management and organization
- Pull request integration with automated comments
- Test result visualization
- Failure analysis and reporting
- Historic test results tracking and trending
"""

import anyio
import json
import logging
import os
import platform
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from pathlib import Path
import hashlib
import uuid
import re

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
    
    This plugin provides comprehensive integration with popular CI/CD systems:
    - GitHub Actions
    - Jenkins
    - GitLab CI
    - Azure DevOps
    - CircleCI
    - Travis CI
    - Bitbucket Pipelines
    - TeamCity
    
    It enables the following capabilities:
    - Test execution results reported to CI/CD systems with standardized API
    - Build status updates with detailed progress tracking
    - Test summary generation in multiple formats (JSON, XML, HTML, Markdown)
    - Artifact management with organization and categorization
    - Environment variable and secret management across platforms
    - Integration with pull request workflows with automatic comments
    - Advanced failure analysis and reporting
    - Test history tracking for trend analysis
    - Customizable notification templates
    - Dashboard integration for visualizing test results
    - Cross-platform standardized API access
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
            # CI System Configuration
            "ci_system": "auto",  # auto, github, jenkins, gitlab, azure, circle, travis, bitbucket, teamcity
            "api_url": None,
            "api_token": None,
            "project": None,
            "repository": None,
            "build_id": None,
            "commit_sha": None,
            "branch": None,
            "pr_number": None,
            
            # Update Configuration
            "update_interval": 60,
            "update_on_completion_only": False,
            "enable_status_updates": True,
            "status_update_format": "detailed",  # minimal, basic, detailed
            
            # PR Integration
            "enable_pr_comments": True,
            "pr_comment_on_failure_only": False,
            "pr_comment_template": "default",  # default, minimal, detailed
            "pr_update_existing_comments": True,
            
            # Artifact Management
            "enable_artifacts": True,
            "artifact_dir": "distributed_test_results",
            "artifact_retention_days": 30,
            "artifact_categories": ["reports", "logs", "data", "metrics"],
            "artifact_compression": True,
            
            # Result Reporting
            "result_format": "all",  # junit, json, html, markdown, all
            "include_system_info": True,
            "include_failure_analysis": True,
            "include_performance_metrics": True,
            
            # History Tracking
            "enable_history_tracking": True,
            "history_retention_days": 90,
            "track_performance_trends": True,
            
            # Notifications
            "enable_notifications": False,
            "notification_channels": [],  # email, slack, teams, discord
            "notification_on_failure_only": True,
            
            # Advanced Options
            "enable_failure_analysis": True,
            "failure_analysis_depth": "detailed",  # basic, detailed, comprehensive
            "dashboard_integration": False,
            "dashboard_url": None,
            "detailed_logging": False,
            "retry_failed_api_calls": True,
            "max_retries": 3,
            "retry_delay_seconds": 5
        }
        
        # CI client map to track supported operations by CI system
        self.ci_capabilities = {}
        
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
                self.update_task = # TODO: Replace with task group - asyncio.create_task(self._periodic_updates())
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
            except anyio.get_cancelled_exc_class():
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
            env = {
                "type": "github",
                "api_url": "https://api.github.com",
                "repository": os.environ.get("GITHUB_REPOSITORY"),
                "build_id": os.environ.get("GITHUB_RUN_ID"),
                "commit_sha": os.environ.get("GITHUB_SHA"),
                "branch": os.environ.get("GITHUB_REF"),
                "workflow": os.environ.get("GITHUB_WORKFLOW"),
                "actor": os.environ.get("GITHUB_ACTOR"),
                "event_name": os.environ.get("GITHUB_EVENT_NAME"),
                "pr_number": self._extract_pr_number_from_github(),
                "run_number": os.environ.get("GITHUB_RUN_NUMBER"),
                "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                "server_url": os.environ.get("GITHUB_SERVER_URL"),
                "api_version": "v3",
                "workspace": os.environ.get("GITHUB_WORKSPACE")
            }
            
            # Set capabilities
            self.ci_capabilities = {
                "create_test_run": True,
                "update_test_run": True,
                "upload_artifact": True,
                "download_artifact": True,
                "add_pr_comment": True,
                "update_pr_comment": True,
                "set_status": True,
                "get_test_history": True,
                "create_check_run": True,
                "update_check_run": True
            }
            
            return env
            
        # Jenkins
        elif os.environ.get("JENKINS_URL"):
            env = {
                "type": "jenkins",
                "api_url": os.environ.get("JENKINS_URL"),
                "build_id": os.environ.get("BUILD_ID"),
                "job_name": os.environ.get("JOB_NAME"),
                "build_url": os.environ.get("BUILD_URL"),
                "branch": os.environ.get("BRANCH_NAME") or os.environ.get("GIT_BRANCH"),
                "commit_sha": os.environ.get("GIT_COMMIT"),
                "workspace": os.environ.get("WORKSPACE"),
                "node_name": os.environ.get("NODE_NAME"),
                "job_base_name": os.environ.get("JOB_BASE_NAME"),
                "build_tag": os.environ.get("BUILD_TAG"),
                "executor_number": os.environ.get("EXECUTOR_NUMBER"),
                "build_number": os.environ.get("BUILD_NUMBER"),
                "change_id": os.environ.get("CHANGE_ID"),  # PR ID for multibranch pipeline
                "change_url": os.environ.get("CHANGE_URL"),
                "change_target": os.environ.get("CHANGE_TARGET")
            }
            
            # Set capabilities
            self.ci_capabilities = {
                "create_test_run": True,
                "update_test_run": True,
                "upload_artifact": True,
                "download_artifact": True,
                "add_pr_comment": os.environ.get("CHANGE_ID") is not None,
                "update_pr_comment": os.environ.get("CHANGE_ID") is not None,
                "set_status": True,
                "get_test_history": True,
                "create_check_run": False,
                "update_check_run": False
            }
            
            return env
            
        # GitLab CI
        elif os.environ.get("GITLAB_CI") == "true":
            env = {
                "type": "gitlab",
                "api_url": os.environ.get("CI_API_V4_URL") or "https://gitlab.com/api/v4",
                "project": os.environ.get("CI_PROJECT_PATH"),
                "repository": os.environ.get("CI_PROJECT_PATH"),
                "build_id": os.environ.get("CI_JOB_ID"),
                "commit_sha": os.environ.get("CI_COMMIT_SHA"),
                "branch": os.environ.get("CI_COMMIT_REF_NAME"),
                "pipeline_id": os.environ.get("CI_PIPELINE_ID"),
                "pr_number": os.environ.get("CI_MERGE_REQUEST_IID"),
                "project_id": os.environ.get("CI_PROJECT_ID"),
                "job_name": os.environ.get("CI_JOB_NAME"),
                "job_stage": os.environ.get("CI_JOB_STAGE"),
                "project_url": os.environ.get("CI_PROJECT_URL"),
                "commit_ref_slug": os.environ.get("CI_COMMIT_REF_SLUG"),
                "merge_request_source_branch_name": os.environ.get("CI_MERGE_REQUEST_SOURCE_BRANCH_NAME"),
                "merge_request_target_branch_name": os.environ.get("CI_MERGE_REQUEST_TARGET_BRANCH_NAME"),
                "runner_id": os.environ.get("CI_RUNNER_ID"),
                "workspace": os.environ.get("CI_PROJECT_DIR")
            }
            
            # Set capabilities
            self.ci_capabilities = {
                "create_test_run": True,
                "update_test_run": True,
                "upload_artifact": True,
                "download_artifact": True,
                "add_pr_comment": os.environ.get("CI_MERGE_REQUEST_IID") is not None,
                "update_pr_comment": os.environ.get("CI_MERGE_REQUEST_IID") is not None,
                "set_status": True,
                "get_test_history": True,
                "create_check_run": False,
                "update_check_run": False
            }
            
            return env
            
        # Azure DevOps Pipelines
        elif os.environ.get("TF_BUILD") == "True":
            env = {
                "type": "azure",
                "api_url": os.environ.get("SYSTEM_COLLECTIONURI"),
                "project": os.environ.get("SYSTEM_TEAMPROJECT"),
                "build_id": os.environ.get("BUILD_BUILDID"),
                "repository": os.environ.get("BUILD_REPOSITORY_NAME"),
                "commit_sha": os.environ.get("BUILD_SOURCEVERSION"),
                "branch": os.environ.get("BUILD_SOURCEBRANCHNAME"),
                "pr_number": os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTID"),
                "build_number": os.environ.get("BUILD_BUILDNUMBER"),
                "build_uri": os.environ.get("BUILD_BUILDURI"),
                "pipeline_id": os.environ.get("SYSTEM_DEFINITIONID"),
                "repository_uri": os.environ.get("BUILD_REPOSITORY_URI"),
                "agent_name": os.environ.get("AGENT_NAME"),
                "agent_id": os.environ.get("AGENT_ID"),
                "workspace": os.environ.get("AGENT_BUILDDIRECTORY"),
                "reason": os.environ.get("BUILD_REASON"),
                "request_id": os.environ.get("BUILD_REQUESTEDFOR"),
                "source_branch": os.environ.get("BUILD_SOURCEBRANCH")
            }
            
            # Set capabilities
            self.ci_capabilities = {
                "create_test_run": True,
                "update_test_run": True,
                "upload_artifact": True,
                "download_artifact": True,
                "add_pr_comment": os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTID") is not None,
                "update_pr_comment": os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTID") is not None,
                "set_status": True,
                "get_test_history": True,
                "create_check_run": False,
                "update_check_run": False
            }
            
            return env
            
        # CircleCI
        elif os.environ.get("CIRCLECI") == "true":
            env = {
                "type": "circle",
                "api_url": "https://circleci.com/api/v2",
                "project": f"{os.environ.get('CIRCLE_PROJECT_USERNAME')}/{os.environ.get('CIRCLE_PROJECT_REPONAME')}",
                "repository": os.environ.get("CIRCLE_PROJECT_REPONAME"),
                "build_id": os.environ.get("CIRCLE_BUILD_NUM"),
                "commit_sha": os.environ.get("CIRCLE_SHA1"),
                "branch": os.environ.get("CIRCLE_BRANCH"),
                "pr_number": os.environ.get("CIRCLE_PR_NUMBER"),
                "workflow_id": os.environ.get("CIRCLE_WORKFLOW_ID"),
                "workflow_job_id": os.environ.get("CIRCLE_WORKFLOW_JOB_ID"),
                "username": os.environ.get("CIRCLE_PROJECT_USERNAME"),
                "reponame": os.environ.get("CIRCLE_PROJECT_REPONAME"),
                "compare_url": os.environ.get("CIRCLE_COMPARE_URL"),
                "build_url": os.environ.get("CIRCLE_BUILD_URL"),
                "workspace": os.environ.get("CIRCLE_WORKING_DIRECTORY")
            }
            
            # Set capabilities
            self.ci_capabilities = {
                "create_test_run": True,
                "update_test_run": True,
                "upload_artifact": True,
                "download_artifact": True,
                "add_pr_comment": os.environ.get("CIRCLE_PR_NUMBER") is not None,
                "update_pr_comment": os.environ.get("CIRCLE_PR_NUMBER") is not None,
                "set_status": True,
                "get_test_history": True,
                "create_check_run": False,
                "update_check_run": False
            }
            
            return env
            
        # Travis CI
        elif os.environ.get("TRAVIS") == "true":
            env = {
                "type": "travis",
                "api_url": "https://api.travis-ci.org",
                "repository": os.environ.get("TRAVIS_REPO_SLUG"),
                "build_id": os.environ.get("TRAVIS_BUILD_ID"),
                "commit_sha": os.environ.get("TRAVIS_COMMIT"),
                "branch": os.environ.get("TRAVIS_BRANCH"),
                "pr_number": os.environ.get("TRAVIS_PULL_REQUEST"),
                "job_id": os.environ.get("TRAVIS_JOB_ID"),
                "job_number": os.environ.get("TRAVIS_JOB_NUMBER"),
                "build_number": os.environ.get("TRAVIS_BUILD_NUMBER"),
                "build_web_url": f"https://travis-ci.org/{os.environ.get('TRAVIS_REPO_SLUG')}/builds/{os.environ.get('TRAVIS_BUILD_ID')}",
                "commit_message": os.environ.get("TRAVIS_COMMIT_MESSAGE"),
                "tag": os.environ.get("TRAVIS_TAG"),
                "workspace": os.environ.get("TRAVIS_BUILD_DIR")
            }
            
            # Set capabilities
            self.ci_capabilities = {
                "create_test_run": True,
                "update_test_run": True,
                "upload_artifact": True,
                "download_artifact": True,
                "add_pr_comment": os.environ.get("TRAVIS_PULL_REQUEST") not in (None, "false"),
                "update_pr_comment": os.environ.get("TRAVIS_PULL_REQUEST") not in (None, "false"),
                "set_status": True,
                "get_test_history": True,
                "create_check_run": False,
                "update_check_run": False
            }
            
            return env
            
        # Bitbucket Pipelines
        elif os.environ.get("BITBUCKET_BUILD_NUMBER"):
            env = {
                "type": "bitbucket",
                "api_url": "https://api.bitbucket.org/2.0",
                "repository": os.environ.get("BITBUCKET_REPO_FULL_NAME"),
                "build_id": os.environ.get("BITBUCKET_BUILD_NUMBER"),
                "commit_sha": os.environ.get("BITBUCKET_COMMIT"),
                "branch": os.environ.get("BITBUCKET_BRANCH"),
                "pr_number": os.environ.get("BITBUCKET_PR_ID"),
                "workspace": os.environ.get("BITBUCKET_CLONE_DIR"),
                "repo_slug": os.environ.get("BITBUCKET_REPO_SLUG"),
                "repo_owner": os.environ.get("BITBUCKET_REPO_OWNER"),
                "pipeline_uuid": os.environ.get("BITBUCKET_PIPELINE_UUID"),
                "step_uuid": os.environ.get("BITBUCKET_STEP_UUID"),
                "tag": os.environ.get("BITBUCKET_TAG")
            }
            
            # Set capabilities
            self.ci_capabilities = {
                "create_test_run": True,
                "update_test_run": True,
                "upload_artifact": True,
                "download_artifact": True,
                "add_pr_comment": os.environ.get("BITBUCKET_PR_ID") is not None,
                "update_pr_comment": os.environ.get("BITBUCKET_PR_ID") is not None,
                "set_status": True,
                "get_test_history": True,
                "create_check_run": False,
                "update_check_run": False
            }
            
            return env
            
        # TeamCity
        elif os.environ.get("TEAMCITY_VERSION"):
            env = {
                "type": "teamcity",
                "api_url": os.environ.get("TEAMCITY_SERVER_URL"),
                "build_id": os.environ.get("BUILD_ID"),
                "build_number": os.environ.get("BUILD_NUMBER"),
                "project_name": os.environ.get("TEAMCITY_PROJECT_NAME"),
                "build_conf_name": os.environ.get("TEAMCITY_BUILDCONF_NAME"),
                "agent_name": os.environ.get("AGENT_NAME"),
                "workspace": os.environ.get("BUILD_WORKING_DIR")
            }
            
            # Try to extract git info if available
            if os.environ.get("TEAMCITY_GIT_PATH"):
                env.update({
                    "repository": os.environ.get("TEAMCITY_GIT_REPOSITORY"),
                    "branch": os.environ.get("TEAMCITY_GIT_BRANCH"),
                    "commit_sha": os.environ.get("BUILD_VCS_NUMBER")
                })
            
            # Set capabilities
            self.ci_capabilities = {
                "create_test_run": True,
                "update_test_run": True,
                "upload_artifact": True,
                "download_artifact": True,
                "add_pr_comment": False,  # TeamCity doesn't have direct PR integration
                "update_pr_comment": False,
                "set_status": True,
                "get_test_history": True,
                "create_check_run": False,
                "update_check_run": False
            }
            
            return env
        
        # No CI environment detected
        logger.info("No CI environment detected, using local environment")
        
        # Create a local environment with limited capabilities
        local_env = {
            "type": "local",
            "api_url": None,
            "repository": os.path.basename(os.getcwd()),
            "build_id": f"local-{int(time.time())}",
            "commit_sha": self._get_local_git_commit(),
            "branch": self._get_local_git_branch(),
            "workspace": os.getcwd()
        }
        
        # Set capabilities for local environment
        self.ci_capabilities = {
            "create_test_run": True,
            "update_test_run": True,
            "upload_artifact": True,
            "download_artifact": True,
            "add_pr_comment": False,
            "update_pr_comment": False,
            "set_status": False,
            "get_test_history": True,
            "create_check_run": False,
            "update_check_run": False
        }
        
        return local_env
        
    def _get_local_git_commit(self) -> Optional[str]:
        """Get the current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception:
            return None
            
    def _get_local_git_branch(self) -> Optional[str]:
        """Get the current git branch."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception:
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
        """Initialize the CI system client using the CI client factory."""
        ci_system = self.config["ci_system"]
        
        logger.info(f"Initializing {ci_system} CI client")
        
        # Get API token from environment or config based on CI system
        token = self._get_ci_token(ci_system)
        
        # Use the CI client factory to create the appropriate client
        self.ci_client = await self._create_ci_client(ci_system, token)
        
        # Create test run in CI system
        await self._create_test_run()
    
    def _get_ci_token(self, ci_system: str) -> Optional[str]:
        """
        Get API token for CI system from environment variables or config.
        
        Args:
            ci_system: CI system type
            
        Returns:
            API token if available, None otherwise
        """
        # Map of CI system to environment variable names for tokens
        token_env_vars = {
            "github": ["GITHUB_TOKEN", "GH_TOKEN"],
            "jenkins": ["JENKINS_TOKEN", "JENKINS_API_TOKEN"],
            "gitlab": ["GITLAB_TOKEN", "CI_JOB_TOKEN"],
            "azure": ["AZURE_DEVOPS_TOKEN", "SYSTEM_ACCESSTOKEN"],
            "circle": ["CIRCLE_TOKEN"],
            "travis": ["TRAVIS_API_TOKEN"],
            "bitbucket": ["BITBUCKET_TOKEN", "BITBUCKET_API_TOKEN"],
            "teamcity": ["TEAMCITY_TOKEN"]
        }
        
        # Try to get token from environment variables
        if ci_system in token_env_vars:
            for env_var in token_env_vars[ci_system]:
                token = os.environ.get(env_var)
                if token:
                    return token
        
        # Fall back to config token
        return self.config["api_token"]
    
    async def _create_ci_client(self, ci_system: str, token: Optional[str]) -> Union[Dict[str, Any], 'CIClient']:
        """
        Create CI client for the specified CI system.
        
        Args:
            ci_system: CI system type
            token: API token
            
        Returns:
            CI client instance or limited mode dict
        """
        # Create standardized CI client based on CI system
        if ci_system == "github":
            if token:
                from distributed_testing.ci import GitHubClient
                return await StandardizedCIClient.create(
                    client_impl=GitHubClient(
                        token=token,
                        repository=self.config["repository"],
                        api_url=self.config["api_url"]
                    ),
                    ci_system=ci_system,
                    capabilities=self.ci_capabilities,
                    config=self.config
                )
            else:
                logger.warning("No GitHub token available, status updates will be limited")
                return {"type": "github", "limited": True}
        
        elif ci_system == "jenkins":
            if token:
                from distributed_testing.ci import JenkinsClient
                user = os.environ.get("JENKINS_USER") or self.config.get("jenkins_user", "")
                return await StandardizedCIClient.create(
                    client_impl=JenkinsClient(
                        url=self.config["api_url"],
                        user=user,
                        token=token
                    ),
                    ci_system=ci_system,
                    capabilities=self.ci_capabilities,
                    config=self.config
                )
            else:
                logger.warning("No Jenkins credentials available, status updates will be limited")
                return {"type": "jenkins", "limited": True}
        
        elif ci_system == "gitlab":
            if token:
                from distributed_testing.ci import GitLabClient
                return await StandardizedCIClient.create(
                    client_impl=GitLabClient(
                        token=token,
                        project=self.config["project"],
                        api_url=self.config["api_url"]
                    ),
                    ci_system=ci_system,
                    capabilities=self.ci_capabilities,
                    config=self.config
                )
            else:
                logger.warning("No GitLab token available, status updates will be limited")
                return {"type": "gitlab", "limited": True}
        
        elif ci_system == "azure":
            if token:
                from distributed_testing.ci import AzureClient
                return await StandardizedCIClient.create(
                    client_impl=AzureClient(
                        token=token,
                        organization=self.config["api_url"],
                        project=self.config["project"]
                    ),
                    ci_system=ci_system,
                    capabilities=self.ci_capabilities,
                    config=self.config
                )
            else:
                logger.warning("No Azure DevOps token available, status updates will be limited")
                return {"type": "azure", "limited": True}
        
        elif ci_system == "circle":
            if token:
                from distributed_testing.ci import CircleCIClient
                return await StandardizedCIClient.create(
                    client_impl=CircleCIClient(
                        token=token,
                        project=self.config["project"],
                        api_url=self.config["api_url"]
                    ),
                    ci_system=ci_system,
                    capabilities=self.ci_capabilities,
                    config=self.config
                )
            else:
                logger.warning("No CircleCI token available, status updates will be limited")
                return {"type": "circle", "limited": True}
        
        elif ci_system == "travis":
            if token:
                from distributed_testing.ci import TravisCIClient
                return await StandardizedCIClient.create(
                    client_impl=TravisCIClient(
                        token=token,
                        repository=self.config["repository"],
                        api_url=self.config["api_url"]
                    ),
                    ci_system=ci_system,
                    capabilities=self.ci_capabilities,
                    config=self.config
                )
            else:
                logger.warning("No Travis CI token available, status updates will be limited")
                return {"type": "travis", "limited": True}
        
        elif ci_system == "bitbucket":
            if token:
                from distributed_testing.ci import BitbucketClient
                return await StandardizedCIClient.create(
                    client_impl=BitbucketClient(
                        token=token,
                        repository=self.config["repository"],
                        api_url=self.config["api_url"]
                    ),
                    ci_system=ci_system,
                    capabilities=self.ci_capabilities,
                    config=self.config
                )
            else:
                logger.warning("No Bitbucket token available, status updates will be limited")
                return {"type": "bitbucket", "limited": True}
        
        elif ci_system == "teamcity":
            if token:
                from distributed_testing.ci import TeamCityClient
                return await StandardizedCIClient.create(
                    client_impl=TeamCityClient(
                        token=token,
                        server_url=self.config["api_url"],
                        project=self.config["project_name"]
                    ),
                    ci_system=ci_system,
                    capabilities=self.ci_capabilities,
                    config=self.config
                )
            else:
                logger.warning("No TeamCity token available, status updates will be limited")
                return {"type": "teamcity", "limited": True}
        
        elif ci_system == "local":
            # Create a local CI client with file-based storage
            from distributed_testing.ci import LocalCIClient
            return await StandardizedCIClient.create(
                client_impl=LocalCIClient(
                    storage_dir=self.config["artifact_dir"],
                    repository=self.config.get("repository", os.path.basename(os.getcwd()))
                ),
                ci_system=ci_system,
                capabilities=self.ci_capabilities,
                config=self.config
            )
        
        else:
            logger.warning(f"Unsupported CI system: {ci_system}")
            return {"type": "unknown", "limited": True}


class StandardizedCIClient:
    """
    Standardized API for interacting with CI/CD systems.
    
    This class provides a unified interface for interacting with different CI/CD systems,
    abstracting away the differences between them and providing consistent error handling,
    retry logic, and capability detection.
    """
    
    @classmethod
    async def create(cls, client_impl, ci_system: str, capabilities: Dict[str, bool], config: Dict[str, Any]) -> 'StandardizedCIClient':
        """
        Create a new StandardizedCIClient instance.
        
        Args:
            client_impl: Implementation of the CI client
            ci_system: CI system type
            capabilities: Capabilities of the CI system
            config: Configuration options
            
        Returns:
            StandardizedCIClient instance
        """
        client = cls(client_impl, ci_system, capabilities, config)
        await client._initialize()
        return client
    
    def __init__(self, client_impl, ci_system: str, capabilities: Dict[str, bool], config: Dict[str, Any]):
        """
        Initialize the StandardizedCIClient.
        
        Args:
            client_impl: Implementation of the CI client
            ci_system: CI system type
            capabilities: Capabilities of the CI system
            config: Configuration options
        """
        self.client_impl = client_impl
        self.ci_system = ci_system
        self.capabilities = capabilities
        self.config = config
        self.retry_failed_api_calls = config.get("retry_failed_api_calls", True)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay_seconds = config.get("retry_delay_seconds", 5)
        self.history_db = None
        
    async def _initialize(self):
        """Initialize the client."""
        # Initialize history database if enabled
        if self.config.get("enable_history_tracking", False):
            await self._initialize_history_db()
    
    async def _initialize_history_db(self):
        """Initialize history database for tracking test results over time."""
        try:
            import sqlite3
            import aiosqlite
            from pathlib import Path
            
            # Create history directory if it doesn't exist
            history_dir = Path(self.config["artifact_dir"]) / "history"
            history_dir.mkdir(parents=True, exist_ok=True)
            
            # Connect to history database
            db_path = history_dir / f"{self.ci_system}_history.db"
            self.history_db = await aiosqlite.connect(db_path)
            
            # Create tables if they don't exist
            await self.history_db.execute("""
                CREATE TABLE IF NOT EXISTS test_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT,
                    completed_at TEXT,
                    status TEXT,
                    build_id TEXT,
                    commit_sha TEXT,
                    branch TEXT,
                    total_tasks INTEGER,
                    completed_tasks INTEGER,
                    failed_tasks INTEGER,
                    duration REAL,
                    metadata TEXT
                )
            """)
            
            await self.history_db.execute("""
                CREATE TABLE IF NOT EXISTS test_tasks (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    name TEXT,
                    type TEXT,
                    status TEXT,
                    created_at TEXT,
                    completed_at TEXT,
                    duration REAL,
                    error TEXT,
                    FOREIGN KEY (run_id) REFERENCES test_runs (id)
                )
            """)
            
            await self.history_db.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id TEXT PRIMARY KEY,
                    run_id TEXT,
                    task_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    unit TEXT,
                    recorded_at TEXT,
                    FOREIGN KEY (run_id) REFERENCES test_runs (id),
                    FOREIGN KEY (task_id) REFERENCES test_tasks (id)
                )
            """)
            
            # Create indexes
            await self.history_db.execute("CREATE INDEX IF NOT EXISTS idx_test_runs_branch ON test_runs (branch)")
            await self.history_db.execute("CREATE INDEX IF NOT EXISTS idx_test_runs_commit ON test_runs (commit_sha)")
            await self.history_db.execute("CREATE INDEX IF NOT EXISTS idx_test_tasks_run ON test_tasks (run_id)")
            await self.history_db.execute("CREATE INDEX IF NOT EXISTS idx_test_tasks_type ON test_tasks (type)")
            await self.history_db.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_run ON performance_metrics (run_id)")
            await self.history_db.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_task ON performance_metrics (task_id)")
            
            await self.history_db.commit()
            
            # Set up cleanup for old history data
            # TODO: Replace with task group - asyncio.create_task(self._cleanup_old_history())
            
            logger.info(f"Initialized history database at {db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing history database: {e}")
            self.history_db = None
    
    async def _cleanup_old_history(self):
        """Clean up old history data based on retention settings."""
        if not self.history_db:
            return
            
        try:
            retention_days = self.config.get("history_retention_days", 90)
            cutoff_date = (datetime.now() - timedelta(days=retention_days)).isoformat()
            
            async with self.history_db.execute("DELETE FROM performance_metrics WHERE run_id IN (SELECT id FROM test_runs WHERE created_at < ?)", (cutoff_date,)) as cursor:
                deleted_metrics = cursor.rowcount
                
            async with self.history_db.execute("DELETE FROM test_tasks WHERE run_id IN (SELECT id FROM test_runs WHERE created_at < ?)", (cutoff_date,)) as cursor:
                deleted_tasks = cursor.rowcount
                
            async with self.history_db.execute("DELETE FROM test_runs WHERE created_at < ?", (cutoff_date,)) as cursor:
                deleted_runs = cursor.rowcount
                
            await self.history_db.commit()
            
            if deleted_runs > 0:
                logger.info(f"Cleaned up {deleted_runs} test runs, {deleted_tasks} tasks, and {deleted_metrics} metrics older than {retention_days} days")
                
        except Exception as e:
            logger.error(f"Error cleaning up old history data: {e}")
    
    async def _execute_with_retry(self, func, *args, **kwargs):
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function
        """
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                result = func(*args, **kwargs)
                
                # Handle coroutines
                if inspect.iscoroutine(result):
                    result = await result
                    
                return result
                
            except Exception as e:
                last_error = e
                retries += 1
                
                if retries <= self.max_retries and self.retry_failed_api_calls:
                    delay = self.retry_delay_seconds * (2 ** (retries - 1))  # Exponential backoff
                    logger.warning(f"API call failed, retrying in {delay}s ({retries}/{self.max_retries}): {e}")
                    await anyio.sleep(delay)
                else:
                    break
        
        logger.error(f"API call failed after {retries} retries: {last_error}")
        raise last_error
    
    # Standardized CI Client API methods
    
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new test run in the CI system.
        
        Args:
            test_run_data: Test run data
            
        Returns:
            Created test run data
        """
        if not self.capabilities.get("create_test_run", False):
            logger.warning(f"CI system {self.ci_system} does not support creating test runs")
            return {"id": f"local-{int(time.time())}", "status": "created"}
        
        try:
            # Call implementation with retry
            result = await self._execute_with_retry(
                self.client_impl.create_test_run,
                test_run_data
            )
            
            # Store test run in history database if enabled
            if self.history_db and self.config.get("enable_history_tracking", False):
                try:
                    test_run_id = result.get("id")
                    created_at = result.get("start_time") or datetime.now().isoformat()
                    build_id = test_run_data.get("build_id")
                    commit_sha = test_run_data.get("commit_sha")
                    branch = test_run_data.get("branch")
                    
                    # Convert test_run_data to JSON for storage
                    metadata_json = json.dumps(test_run_data)
                    
                    # Insert test run into history database
                    await self.history_db.execute("""
                        INSERT INTO test_runs (
                            id, created_at, status, build_id, commit_sha, branch, 
                            total_tasks, completed_tasks, failed_tasks, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0, ?)
                    """, (test_run_id, created_at, "created", build_id, commit_sha, branch, metadata_json))
                    
                    await self.history_db.commit()
                    
                except Exception as e:
                    logger.error(f"Error storing test run in history database: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating test run: {e}")
            return {"id": f"local-{int(time.time())}", "status": "created", "error": str(e)}
    
    async def update_test_run(self, test_run_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing test run in the CI system.
        
        Args:
            test_run_id: Test run ID
            update_data: Update data
            
        Returns:
            True if the update was successful
        """
        if not self.capabilities.get("update_test_run", False):
            logger.warning(f"CI system {self.ci_system} does not support updating test runs")
            return False
        
        try:
            # Call implementation with retry
            result = await self._execute_with_retry(
                self.client_impl.update_test_run,
                test_run_id,
                update_data
            )
            
            # Update test run in history database if enabled
            if self.history_db and self.config.get("enable_history_tracking", False):
                try:
                    # Extract data for history update
                    status = update_data.get("status")
                    completed_at = update_data.get("end_time")
                    summary = update_data.get("summary", {})
                    total_tasks = summary.get("total_tasks", 0)
                    completed_tasks = summary.get("task_statuses", {}).get("completed", 0)
                    failed_tasks = summary.get("task_statuses", {}).get("failed", 0)
                    duration = summary.get("duration", 0)
                    
                    # Build update query dynamically based on provided fields
                    update_fields = []
                    params = []
                    
                    if status:
                        update_fields.append("status = ?")
                        params.append(status)
                        
                    if completed_at:
                        update_fields.append("completed_at = ?")
                        params.append(completed_at)
                        
                    if total_tasks:
                        update_fields.append("total_tasks = ?")
                        params.append(total_tasks)
                        
                    if completed_tasks:
                        update_fields.append("completed_tasks = ?")
                        params.append(completed_tasks)
                        
                    if failed_tasks:
                        update_fields.append("failed_tasks = ?")
                        params.append(failed_tasks)
                        
                    if duration:
                        update_fields.append("duration = ?")
                        params.append(duration)
                    
                    # Only update if there are fields to update
                    if update_fields:
                        query = f"UPDATE test_runs SET {', '.join(update_fields)} WHERE id = ?"
                        params.append(test_run_id)
                        
                        await self.history_db.execute(query, tuple(params))
                        await self.history_db.commit()
                    
                except Exception as e:
                    logger.error(f"Error updating test run in history database: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error updating test run: {e}")
            return False
    
    async def upload_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str = None) -> bool:
        """
        Upload an artifact to the CI system.
        
        Args:
            test_run_id: Test run ID
            artifact_path: Path to the artifact file
            artifact_name: Name of the artifact (defaults to the filename)
            
        Returns:
            True if the upload was successful
        """
        if not self.capabilities.get("upload_artifact", False):
            logger.warning(f"CI system {self.ci_system} does not support uploading artifacts")
            return False
        
        try:
            # Default artifact name to filename if not provided
            if not artifact_name:
                artifact_name = os.path.basename(artifact_path)
            
            # Call implementation with retry
            return await self._execute_with_retry(
                self.client_impl.upload_artifact,
                test_run_id,
                artifact_path,
                artifact_name
            )
            
        except Exception as e:
            logger.error(f"Error uploading artifact: {e}")
            return False
    
    async def download_artifact(self, test_run_id: str, artifact_name: str, output_path: str) -> bool:
        """
        Download an artifact from the CI system.
        
        Args:
            test_run_id: Test run ID
            artifact_name: Name of the artifact
            output_path: Path to write the artifact to
            
        Returns:
            True if the download was successful
        """
        if not self.capabilities.get("download_artifact", False):
            logger.warning(f"CI system {self.ci_system} does not support downloading artifacts")
            return False
        
        try:
            # Call implementation with retry
            return await self._execute_with_retry(
                self.client_impl.download_artifact,
                test_run_id,
                artifact_name,
                output_path
            )
            
        except Exception as e:
            logger.error(f"Error downloading artifact: {e}")
            return False
    
    async def add_pr_comment(self, pr_number: str, comment: str, comment_id: str = None) -> bool:
        """
        Add a comment to a pull request.
        
        Args:
            pr_number: Pull request number
            comment: Comment content
            comment_id: Comment ID for updating existing comments (optional)
            
        Returns:
            True if the comment was added successfully
        """
        if not self.capabilities.get("add_pr_comment", False):
            logger.warning(f"CI system {self.ci_system} does not support adding PR comments")
            return False
        
        try:
            # If updating existing comments is enabled and comment ID is provided
            if self.config.get("pr_update_existing_comments", True) and comment_id:
                # Try to update existing comment if supported
                if self.capabilities.get("update_pr_comment", False):
                    return await self._execute_with_retry(
                        self.client_impl.update_pr_comment,
                        pr_number,
                        comment_id,
                        comment
                    )
            
            # Otherwise add a new comment
            return await self._execute_with_retry(
                self.client_impl.add_pr_comment,
                pr_number,
                comment
            )
            
        except Exception as e:
            logger.error(f"Error adding PR comment: {e}")
            return False
    
    async def set_status(self, commit_sha: str, status: str, context: str, description: str = None, url: str = None) -> bool:
        """
        Set a status on a commit.
        
        Args:
            commit_sha: Commit SHA
            status: Status (success, failure, pending, error)
            context: Status context
            description: Description of the status
            url: URL to link to the status
            
        Returns:
            True if the status was set successfully
        """
        if not self.capabilities.get("set_status", False):
            logger.warning(f"CI system {self.ci_system} does not support setting statuses")
            return False
        
        try:
            # Call implementation with retry
            return await self._execute_with_retry(
                self.client_impl.set_status,
                commit_sha,
                status,
                context,
                description,
                url
            )
            
        except Exception as e:
            logger.error(f"Error setting status: {e}")
            return False
    
    async def get_test_history(self, limit: int = 10, branch: str = None, commit_sha: str = None) -> List[Dict[str, Any]]:
        """
        Get test run history.
        
        Args:
            limit: Maximum number of test runs to return
            branch: Filter by branch
            commit_sha: Filter by commit SHA
            
        Returns:
            List of test runs
        """
        # Use history database if available, otherwise try CI client
        if self.history_db and self.config.get("enable_history_tracking", False):
            try:
                # Build query based on filters
                query = "SELECT * FROM test_runs"
                params = []
                
                if branch or commit_sha:
                    query += " WHERE "
                    filters = []
                    
                    if branch:
                        filters.append("branch = ?")
                        params.append(branch)
                    
                    if commit_sha:
                        filters.append("commit_sha = ?")
                        params.append(commit_sha)
                    
                    query += " AND ".join(filters)
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                # Execute query
                async with self.history_db.execute(query, tuple(params)) as cursor:
                    rows = await cursor.fetchall()
                    
                    # Convert rows to dictionaries
                    result = []
                    for row in rows:
                        row_dict = {
                            "id": row[0],
                            "created_at": row[1],
                            "completed_at": row[2],
                            "status": row[3],
                            "build_id": row[4],
                            "commit_sha": row[5],
                            "branch": row[6],
                            "total_tasks": row[7],
                            "completed_tasks": row[8],
                            "failed_tasks": row[9],
                            "duration": row[10],
                            "metadata": json.loads(row[11]) if row[11] else {}
                        }
                        result.append(row_dict)
                        
                    return result
                    
            except Exception as e:
                logger.error(f"Error getting test history from database: {e}")
                
                # Fall back to CI client if database query fails
                if self.capabilities.get("get_test_history", False):
                    return await self._execute_with_retry(
                        self.client_impl.get_test_history,
                        limit,
                        branch,
                        commit_sha
                    )
                    
                return []
                
        elif self.capabilities.get("get_test_history", False):
            try:
                # Call implementation with retry
                return await self._execute_with_retry(
                    self.client_impl.get_test_history,
                    limit,
                    branch,
                    commit_sha
                )
                
            except Exception as e:
                logger.error(f"Error getting test history: {e}")
                return []
                
        else:
            logger.warning(f"CI system {self.ci_system} does not support getting test history")
            return []
    
    async def record_performance_metric(self, test_run_id: str, task_id: str, metric_name: str, 
                                       metric_value: float, unit: str = None) -> bool:
        """
        Record a performance metric for a test run or task.
        
        Args:
            test_run_id: Test run ID
            task_id: Task ID (optional, can be None for run-level metrics)
            metric_name: Name of the metric
            metric_value: Value of the metric
            unit: Unit of the metric (optional)
            
        Returns:
            True if the metric was recorded successfully
        """
        # Only record in history database if available
        if not self.history_db or not self.config.get("enable_history_tracking", False):
            logger.warning("Performance metrics tracking requires history database")
            return False
            
        try:
            # Generate a unique ID for the metric
            metric_id = str(uuid.uuid4())
            recorded_at = datetime.now().isoformat()
            
            # Insert metric into history database
            await self.history_db.execute("""
                INSERT INTO performance_metrics (
                    id, run_id, task_id, metric_name, metric_value, unit, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (metric_id, test_run_id, task_id, metric_name, metric_value, unit, recorded_at))
            
            await self.history_db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error recording performance metric: {e}")
            return False
            
    async def get_performance_metrics(self, test_run_id: str = None, task_id: str = None, 
                                     metric_name: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get performance metrics.
        
        Args:
            test_run_id: Filter by test run ID (optional)
            task_id: Filter by task ID (optional)
            metric_name: Filter by metric name (optional)
            limit: Maximum number of metrics to return
            
        Returns:
            List of performance metrics
        """
        # Only available through history database
        if not self.history_db or not self.config.get("enable_history_tracking", False):
            logger.warning("Performance metrics tracking requires history database")
            return []
            
        try:
            # Build query based on filters
            query = "SELECT * FROM performance_metrics"
            params = []
            
            if test_run_id or task_id or metric_name:
                query += " WHERE "
                filters = []
                
                if test_run_id:
                    filters.append("run_id = ?")
                    params.append(test_run_id)
                
                if task_id:
                    filters.append("task_id = ?")
                    params.append(task_id)
                
                if metric_name:
                    filters.append("metric_name = ?")
                    params.append(metric_name)
                
                query += " AND ".join(filters)
            
            query += " ORDER BY recorded_at DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            async with self.history_db.execute(query, tuple(params)) as cursor:
                rows = await cursor.fetchall()
                
                # Convert rows to dictionaries
                result = []
                for row in rows:
                    row_dict = {
                        "id": row[0],
                        "run_id": row[1],
                        "task_id": row[2],
                        "metric_name": row[3],
                        "metric_value": row[4],
                        "unit": row[5],
                        "recorded_at": row[6]
                    }
                    result.append(row_dict)
                    
                return result
                
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return []
    
    async def analyze_performance_trends(self, metric_name: str, grouping: str = "branch", 
                                        timeframe: str = "1w", limit: int = 5) -> Dict[str, Any]:
        """
        Analyze performance trends for a specific metric.
        
        Args:
            metric_name: Name of the metric to analyze
            grouping: How to group the data (branch, commit, task_type)
            timeframe: Timeframe to analyze (1d, 1w, 1m, all)
            limit: Maximum number of groups to return
            
        Returns:
            Dictionary with trend analysis
        """
        # Only available through history database
        if not self.history_db or not self.config.get("enable_history_tracking", False):
            logger.warning("Performance trend analysis requires history database")
            return {"error": "History database not available"}
            
        try:
            # Convert timeframe to datetime
            now = datetime.now()
            if timeframe == "1d":
                start_date = now - timedelta(days=1)
            elif timeframe == "1w":
                start_date = now - timedelta(weeks=1)
            elif timeframe == "1m":
                start_date = now - timedelta(days=30)
            else:  # All time
                start_date = now - timedelta(days=3650)
            
            start_date_str = start_date.isoformat()
            
            # Build appropriate query based on grouping
            if grouping == "branch":
                query = """
                    SELECT 
                        tr.branch AS group_name, 
                        COUNT(pm.id) AS metric_count,
                        AVG(pm.metric_value) AS avg_value,
                        MIN(pm.metric_value) AS min_value,
                        MAX(pm.metric_value) AS max_value,
                        STDEV(pm.metric_value) AS std_dev,
                        pm.unit
                    FROM performance_metrics pm
                    JOIN test_runs tr ON pm.run_id = tr.id
                    WHERE pm.metric_name = ? AND pm.recorded_at >= ?
                    GROUP BY tr.branch, pm.unit
                    ORDER BY AVG(pm.metric_value) ASC
                    LIMIT ?
                """
            elif grouping == "commit":
                query = """
                    SELECT 
                        SUBSTR(tr.commit_sha, 1, 8) AS group_name, 
                        COUNT(pm.id) AS metric_count,
                        AVG(pm.metric_value) AS avg_value,
                        MIN(pm.metric_value) AS min_value,
                        MAX(pm.metric_value) AS max_value,
                        STDEV(pm.metric_value) AS std_dev,
                        pm.unit
                    FROM performance_metrics pm
                    JOIN test_runs tr ON pm.run_id = tr.id
                    WHERE pm.metric_name = ? AND pm.recorded_at >= ?
                    GROUP BY tr.commit_sha, pm.unit
                    ORDER BY AVG(pm.metric_value) ASC
                    LIMIT ?
                """
            else:  # task_type
                query = """
                    SELECT 
                        tt.type AS group_name, 
                        COUNT(pm.id) AS metric_count,
                        AVG(pm.metric_value) AS avg_value,
                        MIN(pm.metric_value) AS min_value,
                        MAX(pm.metric_value) AS max_value,
                        STDEV(pm.metric_value) AS std_dev,
                        pm.unit
                    FROM performance_metrics pm
                    JOIN test_tasks tt ON pm.task_id = tt.id
                    WHERE pm.metric_name = ? AND pm.recorded_at >= ?
                    GROUP BY tt.type, pm.unit
                    ORDER BY AVG(pm.metric_value) ASC
                    LIMIT ?
                """
            
            # Execute query
            async with self.history_db.execute(query, (metric_name, start_date_str, limit)) as cursor:
                rows = await cursor.fetchall()
                
                # Convert rows to result
                groups = []
                unit = None
                
                for row in rows:
                    unit = row[6]  # Get unit from last column
                    groups.append({
                        "name": row[0],
                        "count": row[1],
                        "avg": row[2],
                        "min": row[3],
                        "max": row[4],
                        "std_dev": row[5] if row[5] else 0
                    })
                
                # Calculate overall stats
                if groups:
                    all_query = """
                        SELECT 
                            COUNT(pm.id) AS metric_count,
                            AVG(pm.metric_value) AS avg_value,
                            MIN(pm.metric_value) AS min_value,
                            MAX(pm.metric_value) AS max_value,
                            STDEV(pm.metric_value) AS std_dev
                        FROM performance_metrics pm
                        WHERE pm.metric_name = ? AND pm.recorded_at >= ?
                    """
                    
                    async with self.history_db.execute(all_query, (metric_name, start_date_str)) as cursor:
                        overall_row = await cursor.fetchone()
                        
                        overall = {
                            "count": overall_row[0],
                            "avg": overall_row[1],
                            "min": overall_row[2],
                            "max": overall_row[3],
                            "std_dev": overall_row[4] if overall_row[4] else 0
                        }
                        
                    return {
                        "metric_name": metric_name,
                        "unit": unit,
                        "grouping": grouping,
                        "timeframe": timeframe,
                        "groups": groups,
                        "overall": overall
                    }
                else:
                    return {
                        "metric_name": metric_name,
                        "grouping": grouping,
                        "timeframe": timeframe,
                        "groups": [],
                        "overall": None,
                        "message": "No metrics found for the specified criteria"
                    }
                    
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
            return {"error": str(e)}
    
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
                await anyio.sleep(self.config["update_interval"])
                
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
                
            except anyio.get_cancelled_exc_class():
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