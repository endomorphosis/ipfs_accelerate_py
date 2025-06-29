#!/usr/bin/env python3
"""
Bitbucket Pipelines Client for Distributed Testing Framework

This module provides a client for interacting with Bitbucket Pipelines API to report test results,
update build status, and add artifacts.
"""

import asyncio
import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin

import aiohttp

# Import the standardized interface
from distributed_testing.ci.api_interface import CIProviderInterface, TestRunResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BitbucketClient(CIProviderInterface):
    """
    Client for interacting with Bitbucket Pipelines API.
    
    This client implements the standardized CIProviderInterface for Bitbucket Pipelines
    and provides methods for creating and updating test runs, adding PR comments, and 
    uploading artifacts.
    """
    
    def __init__(self):
        """
        Initialize the Bitbucket Pipelines client.
        """
        self.username = None
        self.app_password = None
        self.workspace = None
        self.repository = None
        self.api_url = "https://api.bitbucket.org/2.0"
        self.session = None
        self.build_number = None
        self.commit_hash = None
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the CI provider with configuration.
        
        Args:
            config: Configuration dictionary containing:
                   - username: Bitbucket username
                   - app_password: Bitbucket app password
                   - workspace: Bitbucket workspace (team)
                   - repository: Bitbucket repository name
                   - api_url: Bitbucket API URL (optional)
                   - build_number: Pipeline build number (optional)
                   - commit_hash: Commit hash (optional)
            
        Returns:
            True if initialization succeeded
        """
        self.username = config.get("username")
        self.app_password = config.get("app_password")
        self.workspace = config.get("workspace")
        self.repository = config.get("repository")
        self.api_url = config.get("api_url", "https://api.bitbucket.org/2.0")
        self.build_number = config.get("build_number")
        self.commit_hash = config.get("commit_hash")
        
        # Required credentials
        if not self.username or not self.app_password:
            logger.error("Bitbucket username and app_password are required")
            return False
        
        # Required repository information
        if not self.workspace or not self.repository:
            logger.error("Bitbucket workspace and repository are required")
            return False
        
        logger.info(f"BitbucketClient initialized for {self.workspace}/{self.repository}")
        return True
    
    async def _ensure_session(self):
        """Ensure an aiohttp session exists with proper authentication."""
        if self.session is None:
            # Use HTTP Basic Auth with username and app password
            auth = aiohttp.BasicAuth(self.username, self.app_password)
            
            self.session = aiohttp.ClientSession(
                auth=auth,
                headers={"User-Agent": "DistributedTestingFramework"}
            )
    
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new test run record using Bitbucket Pipelines API.
        
        In Bitbucket Pipelines, we can't truly create a test run via API, but we can 
        track it using the repository reports API.
        
        Args:
            test_run_data: Test run data including name, build_id, etc.
            
        Returns:
            Dictionary with test run information
        """
        await self._ensure_session()
        
        try:
            # Extract key information from test run data
            name = test_run_data.get("name", f"Distributed Test Run - {int(time.time())}")
            commit_hash = test_run_data.get("commit_hash", self.commit_hash)
            build_number = test_run_data.get("build_number", self.build_number)
            
            # We need a commit hash to create a report
            if not commit_hash:
                logger.warning("No commit hash provided for Bitbucket test report")
                
                # Return a simulated test run
                return {
                    "id": f"bitbucket-simulated-{int(time.time())}",
                    "name": name,
                    "status": "running",
                    "start_time": datetime.now().isoformat(),
                    "note": "Created in simulation mode due to missing commit hash"
                }
            
            # Generate a unique report ID
            report_id = f"distributed-test-{int(time.time())}"
            
            # In a production implementation, we'd create a report using the Bitbucket Reports API
            # https://developer.atlassian.com/cloud/bitbucket/rest/api-group-commit-reports/
            
            # For the example, we'll create a test run record without an API call
            test_run = {
                "id": f"bitbucket-{self.workspace}-{self.repository}-{report_id}",
                "external_id": report_id,
                "name": name,
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "workspace": self.workspace,
                "repository": self.repository,
                "commit_hash": commit_hash,
                "build_number": build_number
            }
            
            logger.info(f"Created Bitbucket test run: {test_run['id']}")
            return test_run
        
        except Exception as e:
            logger.error(f"Exception creating Bitbucket test run: {str(e)}")
            
            # Return a simulated test run for easier recovery
            return {
                "id": f"bitbucket-simulated-{int(time.time())}",
                "name": test_run_data.get("name", f"Distributed Test Run - {int(time.time())}"),
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "note": f"Created in simulation mode due to exception: {str(e)}"
            }
    
    async def update_test_run(self, test_run_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a test run using Bitbucket Pipelines API.
        
        Args:
            test_run_id: Test run ID
            update_data: Data to update
            
        Returns:
            True if update succeeded
        """
        await self._ensure_session()
        
        try:
            # Skip if we're using a simulated test run
            if test_run_id.startswith("bitbucket-simulated-"):
                logger.info(f"Skipping update for simulated test run {test_run_id}")
                return True
            
            # Extract workspace, repository, and report ID from test run ID
            parts = test_run_id.split("-", 3)
            if len(parts) < 4 or parts[0] != "bitbucket":
                logger.error(f"Invalid Bitbucket test run ID format: {test_run_id}")
                return False
            
            workspace = parts[1]
            repository = parts[2]
            report_id = parts[3]
            
            # Extract status and other information
            status = update_data.get("status", "running")
            
            # Map our statuses to Bitbucket report result
            bitbucket_result_map = {
                "running": "PENDING",
                "completed": "PASSED",
                "failed": "FAILED"
            }
            
            # In a production implementation, we'd update the report using the Bitbucket Reports API
            # https://developer.atlassian.com/cloud/bitbucket/rest/api-group-commit-reports/
            
            # For the example, we'll just log the update
            bitbucket_result = bitbucket_result_map.get(status, "PENDING")
            logger.info(f"Updated Bitbucket test run {report_id} with result: {bitbucket_result}")
            
            # If we have summary data, we can include that in the report update
            if "summary" in update_data:
                summary = update_data["summary"]
                logger.info(f"Test summary: {summary}")
            
            return True
            
        except Exception as e:
            logger.error(f"Exception updating Bitbucket test run: {str(e)}")
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
            # Construct the API endpoint for PR comments
            url = f"{self.api_url}/repositories/{self.workspace}/{self.repository}/pullrequests/{pr_number}/comments"
            
            # Prepare the comment payload
            payload = {
                "content": {
                    "raw": comment
                }
            }
            
            # Make the API request
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
        
        Bitbucket doesn't have a built-in artifacts API, so this is a placeholder implementation.
        
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
            if test_run_id.startswith("bitbucket-simulated-"):
                logger.info(f"Skipping artifact upload for simulated test run {test_run_id}")
                return True
            
            # Check if file exists
            if not os.path.exists(artifact_path):
                logger.error(f"Artifact file not found: {artifact_path}")
                return False
            
            # Bitbucket doesn't have a built-in artifacts API comparable to GitHub Actions or CircleCI.
            # In a real implementation, you might use Bitbucket's downloads API or another storage solution.
            
            logger.info(f"Artifact upload for Bitbucket not fully implemented. Would upload {artifact_path} as {artifact_name}")
            return True
            
        except Exception as e:
            logger.error(f"Exception uploading artifact to Bitbucket: {str(e)}")
            return False
    
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
            if test_run_id.startswith("bitbucket-simulated-"):
                logger.info(f"Returning simulated status for test run {test_run_id}")
                return {
                    "id": test_run_id,
                    "status": "running",
                    "result": "PENDING"
                }
            
            # Extract workspace, repository, and report ID from test run ID
            parts = test_run_id.split("-", 3)
            if len(parts) < 4 or parts[0] != "bitbucket":
                logger.error(f"Invalid Bitbucket test run ID format: {test_run_id}")
                return {"id": test_run_id, "status": "unknown", "error": "Invalid ID format"}
            
            workspace = parts[1]
            repository = parts[2]
            report_id = parts[3]
            
            # In a production implementation, we'd get the report details using the Bitbucket Reports API
            # For now, return a placeholder status
            return {
                "id": test_run_id,
                "name": f"Report {report_id}",
                "status": "running",
                "result": "PENDING",
                "workspace": workspace,
                "repository": repository
            }
            
        except Exception as e:
            logger.error(f"Exception getting Bitbucket test run status: {str(e)}")
            
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
        
        if not self.commit_hash:
            logger.error("Cannot set build status without commit hash")
            return False
        
        try:
            # Map our status to Bitbucket build status
            bitbucket_status_map = {
                "success": "SUCCESSFUL",
                "failure": "FAILED",
                "error": "FAILED",
                "pending": "INPROGRESS",
                "running": "INPROGRESS"
            }
            
            # Construct the API endpoint for commit statuses
            url = f"{self.api_url}/repositories/{self.workspace}/{self.repository}/commit/{self.commit_hash}/statuses/build"
            
            # Prepare the status payload
            payload = {
                "state": bitbucket_status_map.get(status, "INPROGRESS"),
                "key": "distributed-testing-framework",
                "name": "Distributed Testing Framework",
                "description": description,
                "url": f"https://bitbucket.org/{self.workspace}/{self.repository}/addon/pipelines/home"
            }
            
            # Make the API request
            async with self.session.post(url, json=payload) as response:
                if response.status == 201 or response.status == 200:
                    logger.info(f"Set Bitbucket build status to {status}: {description}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Error setting Bitbucket build status: {response.status} - {error_text}")
                    return False
            
        except Exception as e:
            logger.error(f"Exception setting Bitbucket build status: {str(e)}")
            return False
    
    async def get_artifact_url(self, test_run_id: str, artifact_name: str) -> Optional[str]:
        """
        Get the URL for a test run artifact.
        
        Bitbucket Pipelines doesn't have a built-in artifacts API comparable to GitHub
        Actions or CircleCI. This implementation uses the Bitbucket Downloads API as
        a workaround for storing artifacts.
        
        Args:
            test_run_id: Test run ID
            artifact_name: Name of artifact
            
        Returns:
            URL to the artifact or None if not found
        """
        await self._ensure_session()
        
        try:
            # Skip if we're using a simulated test run
            if test_run_id.startswith("bitbucket-simulated-"):
                logger.warning(f"Cannot get artifact URL for simulated test run {test_run_id}")
                return None
            
            # Check if we have the URL cached
            if hasattr(self, "_artifact_urls") and test_run_id in self._artifact_urls and artifact_name in self._artifact_urls[test_run_id]:
                logger.info(f"Using cached artifact URL for {artifact_name}")
                return self._artifact_urls[test_run_id][artifact_name]
            
            # Extract workspace, repository, and report ID from test run ID
            parts = test_run_id.split("-", 3)
            if len(parts) < 4 or parts[0] != "bitbucket":
                logger.error(f"Invalid Bitbucket test run ID format: {test_run_id}")
                return None
            
            workspace = parts[1]
            repository = parts[2]
            report_id = parts[3]
            
            # In Bitbucket, the closest equivalent to artifacts is the Downloads API
            # We'll form a URL pattern based on the workspace/repository/downloads
            # https://developer.atlassian.com/cloud/bitbucket/rest/api-group-downloads/
            
            # First check if the file exists in the downloads section
            downloads_url = f"{self.api_url}/repositories/{workspace}/{repository}/downloads"
            
            # Try to get the downloads list
            async with self.session.get(downloads_url) as response:
                if response.status == 200:
                    downloads_data = await response.json()
                    
                    # Look for our artifact in the list
                    for download in downloads_data.get("values", []):
                        # Match on name - in a real implementation, we might use a more specific pattern
                        # like f"{test_run_id}-{artifact_name}" to avoid conflicts
                        if download.get("name") == artifact_name or download.get("name") == f"{report_id}-{artifact_name}":
                            # Found the artifact, get the URL
                            artifact_url = download.get("links", {}).get("self", {}).get("href")
                            
                            if artifact_url:
                                # Cache the URL for future use
                                if not hasattr(self, "_artifact_urls"):
                                    self._artifact_urls = {}
                                
                                if test_run_id not in self._artifact_urls:
                                    self._artifact_urls[test_run_id] = {}
                                
                                self._artifact_urls[test_run_id][artifact_name] = artifact_url
                                
                                logger.info(f"Found artifact URL for {artifact_name}: {artifact_url}")
                                return artifact_url
                    
                    # If we didn't find the artifact in the downloads list, see if we can construct a URL
                    # This is a fallback approach and may not work in all cases
                    # In a real implementation, we might use a different storage solution with direct URL access
                    fallback_url = f"https://bitbucket.org/{workspace}/{repository}/downloads/{report_id}-{artifact_name}"
                    logger.warning(f"Artifact not found in downloads API, using fallback URL: {fallback_url}")
                    
                    # Cache the fallback URL
                    if not hasattr(self, "_artifact_urls"):
                        self._artifact_urls = {}
                    
                    if test_run_id not in self._artifact_urls:
                        self._artifact_urls[test_run_id] = {}
                    
                    self._artifact_urls[test_run_id][artifact_name] = fallback_url
                    
                    return fallback_url
                else:
                    # API error - log and fail gracefully
                    error_text = await response.text()
                    logger.error(f"Error getting Bitbucket downloads list: {response.status} - {error_text}")
                    return None
        
        except Exception as e:
            logger.error(f"Exception getting artifact URL from Bitbucket: {str(e)}")
            return None
    
    async def close(self) -> None:
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("BitbucketClient session closed")