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
        
        This implementation uses GitHub Gists to store artifacts since GitHub
        doesn't have a direct artifact API through check runs.
        
        Args:
            test_run_id: Test run ID
            artifact_path: Path to artifact file
            artifact_name: Name of artifact
            
        Returns:
            True if upload succeeded
        """
        await self._ensure_session()
        
        try:
            # Skip if we're using a simulated test run
            if test_run_id.startswith("gh-simulated-"):
                logger.info(f"Skipping artifact upload for simulated test run {test_run_id}")
                return True
            
            # Check if file exists
            if not os.path.exists(artifact_path):
                logger.error(f"Artifact file not found: {artifact_path}")
                return False
            
            # Determine if file is binary or text
            is_binary = False
            try:
                with open(artifact_path, 'r', encoding='utf-8') as f:
                    f.read(1024)  # Try to read as text
            except UnicodeDecodeError:
                is_binary = True
            
            if is_binary:
                # For binary files, use GitHub Releases API
                return await self._upload_binary_artifact(test_run_id, artifact_path, artifact_name)
            else:
                # For text files, use GitHub Gists API
                return await self._upload_text_artifact(test_run_id, artifact_path, artifact_name)
        
        except Exception as e:
            logger.error(f"Exception uploading artifact to GitHub: {str(e)}")
            return False
    
    async def _upload_text_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """
        Upload a text artifact using GitHub Gists API.
        
        Args:
            test_run_id: Test run ID
            artifact_path: Path to artifact file
            artifact_name: Name of artifact
            
        Returns:
            True if upload succeeded
        """
        try:
            # Read file content
            with open(artifact_path, 'r', encoding='utf-8', errors='replace') as f:
                file_content = f.read()
            
            # Create a gist
            url = f"{self.api_url}/gists"
            
            payload = {
                "description": f"Artifact {artifact_name} for test run {test_run_id}",
                "public": False,
                "files": {
                    artifact_name: {
                        "content": file_content
                    }
                }
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 201:
                    data = await response.json()
                    gist_url = data.get("html_url")
                    raw_url = data.get("files", {}).get(artifact_name, {}).get("raw_url")
                    
                    if gist_url:
                        logger.info(f"Created Gist for artifact {artifact_name}: {gist_url}")
                        
                        # Store the artifact URL in the instance for later retrieval
                        if not hasattr(self, "_artifact_urls"):
                            self._artifact_urls = {}
                        
                        if test_run_id not in self._artifact_urls:
                            self._artifact_urls[test_run_id] = {}
                        
                        self._artifact_urls[test_run_id][artifact_name] = {
                            "url": gist_url,
                            "raw_url": raw_url
                        }
                        
                        # Update the check run to include a link to the artifact
                        await self._update_check_run_with_artifact(test_run_id, artifact_name, gist_url)
                        
                        return True
                    else:
                        logger.error("Failed to get Gist URL from response")
                        return False
                else:
                    error_text = await response.text()
                    logger.error(f"Error creating Gist for artifact: {response.status} - {error_text}")
                    return False
        
        except Exception as e:
            logger.error(f"Error uploading text artifact: {str(e)}")
            return False
    
    async def _upload_binary_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """
        Upload a binary artifact using GitHub Releases API.
        
        Args:
            test_run_id: Test run ID
            artifact_path: Path to artifact file
            artifact_name: Name of artifact
            
        Returns:
            True if upload succeeded
        """
        try:
            # Create a tag for the release
            tag_name = f"test-run-{test_run_id}"
            
            # Try to get the commit SHA if available
            commit_sha = self.commit_sha
            if not commit_sha:
                # Use default branch if no commit SHA available
                branch_url = f"{self.api_url}/repos/{self.repository}/branches/main"
                async with self.session.get(branch_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        commit_sha = data.get("commit", {}).get("sha")
                    else:
                        commit_sha = "HEAD"
            
            # Create a release
            release_url = f"{self.api_url}/repos/{self.repository}/releases"
            
            release_payload = {
                "tag_name": tag_name,
                "name": f"Test Run {test_run_id}",
                "body": f"Artifacts for test run {test_run_id}",
                "draft": False,
                "prerelease": True,
                "target_commitish": commit_sha
            }
            
            async with self.session.post(release_url, json=release_payload) as response:
                # Check if release creation was successful
                if response.status in (201, 200):
                    release_data = await response.json()
                    release_id = release_data.get("id")
                    upload_url = release_data.get("upload_url").split("{")[0]
                    
                    # Upload the asset
                    with open(artifact_path, "rb") as f:
                        content = f.read()
                    
                    # Set up headers for binary upload
                    headers = {
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(os.path.getsize(artifact_path))
                    }
                    
                    # Upload the asset
                    asset_url = f"{upload_url}?name={artifact_name}"
                    async with self.session.post(asset_url, data=content, headers=headers) as upload_response:
                        if upload_response.status == 201:
                            asset_data = await upload_response.json()
                            browser_download_url = asset_data.get("browser_download_url")
                            
                            if browser_download_url:
                                logger.info(f"Uploaded binary artifact {artifact_name} to {browser_download_url}")
                                
                                # Store the artifact URL for later retrieval
                                if not hasattr(self, "_artifact_urls"):
                                    self._artifact_urls = {}
                                
                                if test_run_id not in self._artifact_urls:
                                    self._artifact_urls[test_run_id] = {}
                                
                                self._artifact_urls[test_run_id][artifact_name] = {
                                    "url": browser_download_url,
                                    "release_id": release_id
                                }
                                
                                # Update the check run to include a link to the artifact
                                await self._update_check_run_with_artifact(test_run_id, artifact_name, browser_download_url)
                                
                                return True
                            else:
                                logger.error("Failed to get download URL from response")
                                return False
                        else:
                            error_text = await upload_response.text()
                            logger.error(f"Error uploading binary artifact: {upload_response.status} - {error_text}")
                            return False
                else:
                    error_text = await response.text()
                    logger.error(f"Error creating release: {response.status} - {error_text}")
                    return False
        
        except Exception as e:
            logger.error(f"Error uploading binary artifact: {str(e)}")
            return False
    
    async def _update_check_run_with_artifact(self, test_run_id: str, artifact_name: str, artifact_url: str) -> bool:
        """
        Update a check run to include a link to an artifact.
        
        Args:
            test_run_id: Test run ID
            artifact_name: Name of artifact
            artifact_url: URL to artifact
            
        Returns:
            True if update succeeded
        """
        try:
            # Skip for simulated test runs
            if test_run_id.startswith("gh-simulated-"):
                return True
            
            check_run_url = f"{self.api_url}/repos/{self.repository}/check-runs/{test_run_id}"
            
            # Get existing output
            async with self.session.get(check_run_url) as get_response:
                if get_response.status == 200:
                    check_data = await get_response.json()
                    existing_output = check_data.get("output", {})
                    
                    # Update output to include artifact link
                    output_title = existing_output.get("title", "Test Run Results")
                    output_summary = existing_output.get("summary", "")
                    
                    # Check if we already have an Artifacts section
                    if "## Artifacts" in output_summary:
                        # Add to existing Artifacts section
                        output_summary += f"\n- [{artifact_name}]({artifact_url})"
                    else:
                        # Add new Artifacts section
                        output_summary += f"\n\n## Artifacts\n\n- [{artifact_name}]({artifact_url})"
                    
                    updated_output = {
                        "title": output_title,
                        "summary": output_summary
                    }
                    
                    # Update the check run
                    update_payload = {
                        "output": updated_output
                    }
                    
                    async with self.session.patch(check_run_url, json=update_payload) as update_response:
                        if update_response.status == 200:
                            logger.info(f"Updated check run {test_run_id} with artifact link")
                            return True
                        else:
                            error_text = await update_response.text()
                            logger.error(f"Error updating check run with artifact link: {update_response.status} - {error_text}")
                            return False
                else:
                    error_text = await get_response.text()
                    logger.warning(f"Could not get check run to update with artifact link: {get_response.status} - {error_text}")
                    return False
            
        except Exception as e:
            logger.error(f"Error updating check run with artifact: {str(e)}")
            return False
            
    async def get_artifact_url(self, test_run_id: str, artifact_name: str) -> Optional[str]:
        """
        Get the URL for a test run artifact.
        
        Args:
            test_run_id: Test run ID
            artifact_name: Name of artifact
            
        Returns:
            URL to the artifact or None if not found
        """
        await self._ensure_session()
        
        try:
            # Skip for simulated test runs
            if test_run_id.startswith("gh-simulated-"):
                logger.warning(f"Cannot get artifact URL for simulated test run {test_run_id}")
                return None
            
            # Check if we have the URL cached
            if hasattr(self, "_artifact_urls") and test_run_id in self._artifact_urls and artifact_name in self._artifact_urls[test_run_id]:
                artifact_info = self._artifact_urls[test_run_id][artifact_name]
                return artifact_info.get("raw_url") or artifact_info.get("url")
            
            # Try to find the artifact URL from the check run
            check_run_url = f"{self.api_url}/repos/{self.repository}/check-runs/{test_run_id}"
            
            async with self.session.get(check_run_url) as response:
                if response.status == 200:
                    check_data = await response.json()
                    output = check_data.get("output", {})
                    summary = output.get("summary", "")
                    
                    # Look for the artifact link in the summary
                    import re
                    artifact_pattern = rf"\[{re.escape(artifact_name)}\]\((.*?)\)"
                    match = re.search(artifact_pattern, summary)
                    
                    if match:
                        return match.group(1)
                    else:
                        logger.warning(f"Artifact {artifact_name} not found in check run {test_run_id}")
                else:
                    logger.warning(f"Failed to get check run {test_run_id}")
            
            # Try to find the artifact in releases
            # First, try to get the release by tag name
            tag_name = f"test-run-{test_run_id}"
            releases_url = f"{self.api_url}/repos/{self.repository}/releases/tags/{tag_name}"
            
            async with self.session.get(releases_url) as response:
                if response.status == 200:
                    release_data = await response.json()
                    assets = release_data.get("assets", [])
                    
                    for asset in assets:
                        if asset.get("name") == artifact_name:
                            return asset.get("browser_download_url")
            
            # If we didn't find the artifact, return None
            logger.warning(f"Artifact {artifact_name} not found for test run {test_run_id}")
            return None
                
        except Exception as e:
            logger.error(f"Error getting artifact URL: {str(e)}")
            return None
    
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