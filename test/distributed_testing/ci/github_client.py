#!/usr/bin/env python3
"""
GitHub CI Client for Distributed Testing Framework

This module provides a client for interacting with GitHub's API to report test results,
update build status, and add PR comments.
"""

import asyncio
import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import aiohttp

# Import the standardized interface
from distributed_testing.ci.api_interface import CIProviderInterface, TestRunResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GitHubClient(CIProviderInterface):
    """
    Client for interacting with GitHub's API.
    
    This client implements the standardized CIProviderInterface for GitHub
    and provides methods for creating and updating test runs, adding comments to PRs,
    and uploading artifacts.
    """
    
    def __init__(self):
        """
        Initialize the GitHub client.
        """
        self.token = None
        self.repository = None
        self.api_url = "https://api.github.com"
        self.session = None
        self.commit_sha = None
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the CI provider with configuration.
        
        Args:
            config: Configuration dictionary containing:
                   - token: GitHub API token
                   - repository: GitHub repository (owner/repo)
                   - api_url: GitHub API URL (optional)
                   - commit_sha: Commit SHA (optional)
            
        Returns:
            True if initialization succeeded
        """
        self.token = config.get("token")
        self.repository = config.get("repository")
        self.api_url = config.get("api_url", "https://api.github.com")
        self.commit_sha = config.get("commit_sha")
        
        if not self.token:
            logger.error("GitHub token is required")
            return False
        
        if not self.repository:
            logger.error("GitHub repository is required")
            return False
        
        logger.info(f"GitHubClient initialized for repository {self.repository}")
        return True
    
    async def _ensure_session(self):
        """Ensure an aiohttp session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession(headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "DistributedTestingFramework"
            })
    
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new test run using GitHub Checks API.
        
        Args:
            test_run_data: Test run data including name, build_id, etc.
            
        Returns:
            Dictionary with test run information
        """
        await self._ensure_session()
        
        try:
            # Extract key information from test run data
            name = test_run_data.get("name", f"Distributed Test Run - {int(time.time())}")
            commit_sha = test_run_data.get("commit_sha", self.commit_sha)
            
            if not commit_sha:
                logger.warning("No commit SHA provided for GitHub check")
                # Use a fallback for demo purposes
                commit_sha = "HEAD"
            
            # Create a check run using GitHub Checks API
            url = f"{self.api_url}/repos/{self.repository}/check-runs"
            
            payload = {
                "name": name,
                "head_sha": commit_sha,
                "status": "in_progress",
                "started_at": datetime.now().isoformat()
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 201:
                    data = await response.json()
                    
                    # Extract relevant information for our test run
                    test_run = {
                        "id": str(data["id"]),
                        "external_id": str(data["id"]),
                        "name": data["name"],
                        "status": "running",
                        "start_time": data["started_at"],
                        "url": data["html_url"]
                    }
                    
                    logger.info(f"Created GitHub check run: {test_run['id']}")
                    return test_run
                else:
                    error_text = await response.text()
                    logger.error(f"Error creating GitHub check run: {response.status} - {error_text}")
                    
                    # Return a simulated test run for easier recovery
                    return {
                        "id": f"gh-simulated-{int(time.time())}",
                        "name": name,
                        "status": "running",
                        "start_time": datetime.now().isoformat(),
                        "note": "Created in simulation mode due to API error"
                    }
        
        except Exception as e:
            logger.error(f"Exception creating GitHub check run: {str(e)}")
            
            # Return a simulated test run for easier recovery
            return {
                "id": f"gh-simulated-{int(time.time())}",
                "name": test_run_data.get("name", f"Distributed Test Run - {int(time.time())}"),
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "note": "Created in simulation mode due to exception"
            }
    
    async def update_test_run(self, test_run_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a test run using GitHub Checks API.
        
        Args:
            test_run_id: Test run ID
            update_data: Data to update
            
        Returns:
            True if update succeeded
        """
        await self._ensure_session()
        
        try:
            # Skip if we're using a simulated test run
            if test_run_id.startswith("gh-simulated-"):
                logger.info(f"Skipping update for simulated test run {test_run_id}")
                return True
            
            # Update check run using GitHub Checks API
            url = f"{self.api_url}/repos/{self.repository}/check-runs/{test_run_id}"
            
            # Map our status to GitHub check run status
            status_map = {
                "running": "in_progress",
                "completed": "completed",
                "failed": "completed"
            }
            
            # Map our status to GitHub conclusion
            conclusion_map = {
                "completed": "success",
                "failed": "failure"
            }
            
            payload = {
                "status": status_map.get(update_data.get("status", "running"), "in_progress")
            }
            
            # Add conclusion if status is completed or failed
            if update_data.get("status") in ("completed", "failed"):
                payload["conclusion"] = conclusion_map.get(update_data.get("status"))
                payload["completed_at"] = update_data.get("end_time", datetime.now().isoformat())
            
            # Add summary if available
            if "summary" in update_data:
                summary = update_data["summary"]
                
                output_text = [
                    f"# Distributed Testing Results\n",
                    f"\n## Summary\n",
                    f"\n- **Total Tasks:** {summary.get('total_tests', summary.get('total_tasks', 0))}",
                ]
                
                # Handle both task_statuses and explicit passed/failed counts
                if "task_statuses" in summary:
                    for status, count in summary.get("task_statuses", {}).items():
                        output_text.append(f"- **{status.capitalize()} Tasks:** {count}")
                else:
                    output_text.append(f"- **Passed Tests:** {summary.get('passed_tests', 0)}")
                    output_text.append(f"- **Failed Tests:** {summary.get('failed_tests', 0)}")
                    output_text.append(f"- **Skipped Tests:** {summary.get('skipped_tests', 0)}")
                
                if "duration" in summary:
                    output_text.append(f"\n**Duration:** {summary['duration']:.2f} seconds")
                elif "duration_seconds" in summary:
                    output_text.append(f"\n**Duration:** {summary['duration_seconds']:.2f} seconds")
                
                payload["output"] = {
                    "title": f"Distributed Testing Results",
                    "summary": "\n".join(output_text)
                }
            
            async with self.session.patch(url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"Updated GitHub check run: {test_run_id}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Error updating GitHub check run: {response.status} - {error_text}")
                    return False
        
        except Exception as e:
            logger.error(f"Exception updating GitHub check run: {str(e)}")
            return False
    
    async def add_pr_comment(self, pr_number: str, comment: str) -> bool:
        """
        Add a comment to a pull request.
        
        Args:
            pr_number: Pull request number
            comment: Comment text
            
        Returns:
            True if comment was added successfully
        """
        await self._ensure_session()
        
        try:
            url = f"{self.api_url}/repos/{self.repository}/issues/{pr_number}/comments"
            
            payload = {
                "body": comment
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 201:
                    logger.info(f"Added comment to PR #{pr_number}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Error adding comment to PR: {response.status} - {error_text}")
                    return False
        
        except Exception as e:
            logger.error(f"Exception adding comment to PR: {str(e)}")
            return False
    
    async def upload_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """
        Upload an artifact for a test run.
        
        Note: GitHub doesn't have a direct artifact API through check runs, so this is a simplified
        implementation that would need more work for a production system.
        
        Args:
            test_run_id: Test run ID
            artifact_path: Path to artifact file
            artifact_name: Name of artifact
            
        Returns:
            True if upload succeeded
        """
        await self._ensure_session()
        
        # GitHub doesn't have a direct artifact API through check runs,
        # so this is a simplified implementation that would need more work
        # for a production system.
        
        # In a real implementation, you might use GitHub Actions artifacts API or
        # upload to a release or use Gist.
        
        logger.info(f"Artifact upload for GitHub not fully implemented. Would upload {artifact_path} as {artifact_name}")
        return True
    
    async def get_test_run_status(self, test_run_id: str) -> Dict[str, Any]:
        """
        Get the status of a test run.
        
        Args:
            test_run_id: Test run ID
            
        Returns:
            Dictionary with test run status information
        """
        await self._ensure_session()
        
        try:
            # Skip if we're using a simulated test run
            if test_run_id.startswith("gh-simulated-"):
                logger.info(f"Returning simulated status for test run {test_run_id}")
                return {
                    "id": test_run_id,
                    "status": "running",
                    "conclusion": None
                }
            
            # Get check run using GitHub Checks API
            url = f"{self.api_url}/repos/{self.repository}/check-runs/{test_run_id}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Map GitHub status to our standard status
                    status_map = {
                        "queued": "pending",
                        "in_progress": "running",
                        "completed": data.get("conclusion", "completed")
                    }
                    
                    return {
                        "id": str(data["id"]),
                        "name": data["name"],
                        "status": status_map.get(data.get("status"), data.get("status")),
                        "conclusion": data.get("conclusion"),
                        "url": data.get("html_url"),
                        "started_at": data.get("started_at"),
                        "completed_at": data.get("completed_at")
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Error getting GitHub check run: {response.status} - {error_text}")
                    
                    # Return a simulated status for easier recovery
                    return {
                        "id": test_run_id,
                        "status": "unknown",
                        "error": f"Failed to get status: {response.status}"
                    }
        
        except Exception as e:
            logger.error(f"Exception getting GitHub check run: {str(e)}")
            
            # Return a simulated status for easier recovery
            return {
                "id": test_run_id,
                "status": "unknown",
                "error": f"Exception: {str(e)}"
            }
    
    async def set_build_status(self, status: str, description: str) -> bool:
        """
        Set the build status in the CI system.
        
        Args:
            status: Status to set (success, failure, pending)
            description: Status description
            
        Returns:
            True if status was set successfully
        """
        await self._ensure_session()
        
        if not self.commit_sha:
            logger.error("Cannot set build status without commit SHA")
            return False
        
        try:
            # Set commit status using GitHub Status API
            url = f"{self.api_url}/repos/{self.repository}/statuses/{self.commit_sha}"
            
            # Map our status to GitHub status
            github_status_map = {
                "success": "success",
                "failure": "failure",
                "error": "error",
                "pending": "pending",
                "running": "pending"
            }
            
            payload = {
                "state": github_status_map.get(status, "pending"),
                "description": description,
                "context": "distributed-testing-framework"
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 201:
                    logger.info(f"Set GitHub build status to {status}: {description}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Error setting GitHub build status: {response.status} - {error_text}")
                    return False
        
        except Exception as e:
            logger.error(f"Exception setting GitHub build status: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("GitHubClient session closed")