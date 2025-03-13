#!/usr/bin/env python3
"""
GitLab CI Client for Distributed Testing Framework

This module provides a client for interacting with GitLab's API to report test results,
update build status, and add MR comments.
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

class GitLabClient(CIProviderInterface):
    """
    Client for interacting with GitLab's API.
    
    This client implements the standardized CIProviderInterface for GitLab
    and provides methods for creating and updating test runs, adding comments to MRs,
    and uploading artifacts.
    """
    
    def __init__(self):
        """
        Initialize the GitLab client.
        """
        self.token = None
        self.project = None
        self.api_url = "https://gitlab.com/api/v4"
        self.session = None
        self.commit_sha = None
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the CI provider with configuration.
        
        Args:
            config: Configuration dictionary containing:
                   - token: GitLab API token
                   - project: GitLab project ID or URL-encoded path
                   - api_url: GitLab API URL (optional)
                   - commit_sha: Commit SHA (optional)
            
        Returns:
            True if initialization succeeded
        """
        self.token = config.get("token")
        self.project = config.get("project")
        self.api_url = config.get("api_url", "https://gitlab.com/api/v4")
        self.commit_sha = config.get("commit_sha")
        
        if not self.token:
            logger.error("GitLab token is required")
            return False
        
        if not self.project:
            logger.error("GitLab project is required")
            return False
        
        logger.info(f"GitLabClient initialized for project {self.project}")
        return True
    
    async def _ensure_session(self):
        """Ensure an aiohttp session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession(headers={
                "PRIVATE-TOKEN": self.token,
                "User-Agent": "DistributedTestingFramework"
            })
    
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new test run using GitLab CI API.
        
        Args:
            test_run_data: Test run data including name, build_id, etc.
            
        Returns:
            Dictionary with test run information
        """
        await self._ensure_session()
        
        try:
            # Extract key information from test run data
            name = test_run_data.get("name", f"Distributed Test Run - {int(time.time())}")
            commit_sha = test_run_data.get("commit_sha")
            pipeline_id = test_run_data.get("pipeline_id")
            
            # If pipeline_id is provided, we'll use that to create a job
            if pipeline_id:
                # In a real implementation, this would call the pipeline jobs API
                # or create a custom job, but GitLab doesn't allow creating jobs via API
                logger.info(f"Using pipeline ID {pipeline_id} for test run")
                test_run = {
                    "id": f"gl-pipeline-{pipeline_id}-{int(time.time())}",
                    "external_id": pipeline_id,
                    "name": name,
                    "status": "running",
                    "start_time": datetime.now().isoformat(),
                    "pipeline_id": pipeline_id
                }
                
                logger.info(f"Created GitLab test run for pipeline: {test_run['id']}")
                return test_run
            
            # If commit_sha is provided, we can use GitLab Commit Status API
            elif commit_sha:
                # Create commit status using GitLab API
                url = f"{self.api_url}/projects/{self.project}/statuses/{commit_sha}"
                
                payload = {
                    "state": "running",
                    "name": name,
                    "context": "distributed-testing",
                    "description": "Running distributed tests"
                }
                
                async with self.session.post(url, params=payload) as response:
                    if response.status == 201:
                        data = await response.json()
                        
                        # Extract relevant information for our test run
                        test_run = {
                            "id": f"gl-status-{commit_sha}-{int(time.time())}",
                            "external_id": data.get("id", "unknown"),
                            "name": name,
                            "status": "running",
                            "start_time": datetime.now().isoformat(),
                            "commit_sha": commit_sha
                        }
                        
                        logger.info(f"Created GitLab commit status: {test_run['id']}")
                        return test_run
                    else:
                        error_text = await response.text()
                        logger.error(f"Error creating GitLab commit status: {response.status} - {error_text}")
                        
                        # Return a simulated test run for easier recovery
                        return {
                            "id": f"gl-simulated-{int(time.time())}",
                            "name": name,
                            "status": "running",
                            "start_time": datetime.now().isoformat(),
                            "note": "Created in simulation mode due to API error"
                        }
            else:
                logger.warning("No pipeline ID or commit SHA provided for GitLab test run")
                
                # Return a simulated test run
                return {
                    "id": f"gl-simulated-{int(time.time())}",
                    "name": name,
                    "status": "running",
                    "start_time": datetime.now().isoformat(),
                    "note": "Created in simulation mode due to missing pipeline ID or commit SHA"
                }
        
        except Exception as e:
            logger.error(f"Exception creating GitLab test run: {str(e)}")
            
            # Return a simulated test run for easier recovery
            return {
                "id": f"gl-simulated-{int(time.time())}",
                "name": test_run_data.get("name", f"Distributed Test Run - {int(time.time())}"),
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "note": "Created in simulation mode due to exception"
            }
    
    async def update_test_run(self, test_run_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a test run using GitLab CI API.
        
        Args:
            test_run_id: Test run ID
            update_data: Data to update
            
        Returns:
            True if update succeeded
        """
        await self._ensure_session()
        
        try:
            # Skip if we're using a simulated test run
            if test_run_id.startswith("gl-simulated-"):
                logger.info(f"Skipping update for simulated test run {test_run_id}")
                return True
            
            # Extract status from update data
            status = update_data.get("status", "running")
            
            # Map our status to GitLab status
            status_map = {
                "running": "running",
                "completed": "success",
                "failed": "failed"
            }
            
            gitlab_status = status_map.get(status, "running")
            
            # If it's a pipeline-based test run
            if test_run_id.startswith("gl-pipeline-"):
                # In a real implementation, this would update the pipeline or job
                # but GitLab doesn't allow updating jobs via API
                logger.info(f"Would update GitLab pipeline for test run {test_run_id} with status {gitlab_status}")
                return True
            
            # If it's a commit status-based test run
            elif test_run_id.startswith("gl-status-"):
                # Extract commit SHA from test run ID
                parts = test_run_id.split("-")
                if len(parts) >= 3:
                    commit_sha = parts[2]
                    
                    # Update commit status using GitLab API
                    url = f"{self.api_url}/projects/{self.project}/statuses/{commit_sha}"
                    
                    payload = {
                        "state": gitlab_status,
                        "name": update_data.get("name", "Distributed Testing"),
                        "context": "distributed-testing",
                        "description": f"Tests {status} at {datetime.now().isoformat()}"
                    }
                    
                    # If we have a summary, add it to the description
                    if "summary" in update_data:
                        summary = update_data["summary"]
                        description_parts = [f"Tests {status}"]
                        
                        if "total_tasks" in summary:
                            description_parts.append(f"Total tasks: {summary['total_tasks']}")
                        
                        for task_status, count in summary.get("task_statuses", {}).items():
                            description_parts.append(f"{task_status.capitalize()}: {count}")
                        
                        if "duration" in summary:
                            description_parts.append(f"Duration: {summary['duration']:.2f}s")
                        
                        payload["description"] = " | ".join(description_parts)
                    
                    async with self.session.post(url, params=payload) as response:
                        if response.status == 201:
                            logger.info(f"Updated GitLab commit status for {test_run_id}")
                            return True
                        else:
                            error_text = await response.text()
                            logger.error(f"Error updating GitLab commit status: {response.status} - {error_text}")
                            return False
                else:
                    logger.error(f"Invalid test run ID format: {test_run_id}")
                    return False
            else:
                logger.warning(f"Unknown test run ID format: {test_run_id}")
                return False
        
        except Exception as e:
            logger.error(f"Exception updating GitLab test run: {str(e)}")
            return False
    
    async def add_pr_comment(self, pr_number: str, comment: str) -> bool:
        """
        Add a comment to a merge request.
        
        Args:
            pr_number: Merge request IID
            comment: Comment text
            
        Returns:
            True if comment was added successfully
        """
        await self._ensure_session()
        
        try:
            url = f"{self.api_url}/projects/{self.project}/merge_requests/{pr_number}/notes"
            
            payload = {
                "body": comment
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 201:
                    logger.info(f"Added comment to MR !{pr_number}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Error adding comment to MR: {response.status} - {error_text}")
                    return False
        
        except Exception as e:
            logger.error(f"Exception adding comment to MR: {str(e)}")
            return False
    
    async def upload_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """
        Upload an artifact for a test run.
        
        Note: GitLab doesn't allow uploading job artifacts via API directly. This is a simplified
        implementation that would need more work for a production system.
        
        Args:
            test_run_id: Test run ID
            artifact_path: Path to artifact file
            artifact_name: Name of artifact
            
        Returns:
            True if upload succeeded
        """
        await self._ensure_session()
        
        # In a real implementation, this might upload to a GitLab package registry,
        # or use the project file upload API.
        
        logger.info(f"Artifact upload for GitLab not fully implemented. Would upload {artifact_path} as {artifact_name}")
        return True
    
    async def close(self) -> None:
        """
        Close the CI provider and clean up resources.
        
        Returns:
            None
        """
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("GitLabClient session closed")
            
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
            if test_run_id.startswith("gl-simulated-"):
                logger.info(f"Returning simulated status for test run {test_run_id}")
                return {
                    "id": test_run_id,
                    "status": "running",
                    "conclusion": None
                }
            
            # If it's a pipeline-based test run
            if test_run_id.startswith("gl-pipeline-"):
                # Extract pipeline ID from test run ID
                parts = test_run_id.split("-")
                if len(parts) >= 3:
                    pipeline_id = parts[2]
                    
                    # Get pipeline status using GitLab API
                    url = f"{self.api_url}/projects/{self.project}/pipelines/{pipeline_id}"
                    
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Map GitLab status to our standard status
                            status_map = {
                                "pending": "pending",
                                "running": "running",
                                "success": "completed",
                                "failed": "failed",
                                "canceled": "cancelled",
                                "skipped": "skipped"
                            }
                            
                            return {
                                "id": test_run_id,
                                "name": f"Pipeline #{pipeline_id}",
                                "status": status_map.get(data.get("status"), data.get("status")),
                                "conclusion": data.get("status"),
                                "url": data.get("web_url"),
                                "started_at": data.get("created_at"),
                                "completed_at": data.get("finished_at")
                            }
                        else:
                            error_text = await response.text()
                            logger.error(f"Error getting GitLab pipeline: {response.status} - {error_text}")
                            
                            # Return a simulated status for easier recovery
                            return {
                                "id": test_run_id,
                                "status": "unknown",
                                "error": f"Failed to get status: {response.status}"
                            }
            
            # If it's a commit status-based test run
            elif test_run_id.startswith("gl-status-"):
                # Extract commit SHA from test run ID
                parts = test_run_id.split("-")
                if len(parts) >= 3:
                    commit_sha = parts[2]
                    
                    # Get commit status using GitLab API
                    url = f"{self.api_url}/projects/{self.project}/repository/commits/{commit_sha}/statuses"
                    
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Find the most recent status for our context
                            our_statuses = [
                                s for s in data 
                                if s.get("name", "").startswith("Distributed Test")
                            ]
                            
                            if our_statuses:
                                latest = our_statuses[0]
                                
                                # Map GitLab status to our standard status
                                status_map = {
                                    "pending": "pending",
                                    "running": "running",
                                    "success": "completed",
                                    "failed": "failed",
                                    "canceled": "cancelled"
                                }
                                
                                return {
                                    "id": test_run_id,
                                    "name": latest.get("name", "Commit Status"),
                                    "status": status_map.get(latest.get("status"), latest.get("status")),
                                    "description": latest.get("description"),
                                    "created_at": latest.get("created_at")
                                }
                            else:
                                logger.warning(f"No distributed testing statuses found for commit {commit_sha}")
                                return {
                                    "id": test_run_id,
                                    "status": "unknown",
                                    "error": "No matching statuses found"
                                }
                        else:
                            error_text = await response.text()
                            logger.error(f"Error getting GitLab commit statuses: {response.status} - {error_text}")
                            
                            # Return a simulated status for easier recovery
                            return {
                                "id": test_run_id,
                                "status": "unknown",
                                "error": f"Failed to get status: {response.status}"
                            }
                else:
                    logger.error(f"Invalid test run ID format: {test_run_id}")
                    return {
                        "id": test_run_id,
                        "status": "unknown",
                        "error": "Invalid test run ID format"
                    }
            else:
                logger.warning(f"Unknown test run ID format: {test_run_id}")
                return {
                    "id": test_run_id,
                    "status": "unknown",
                    "error": "Unknown test run ID format"
                }
        
        except Exception as e:
            logger.error(f"Exception getting GitLab test run status: {str(e)}")
            
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
            # Set commit status using GitLab API
            url = f"{self.api_url}/projects/{self.project}/statuses/{self.commit_sha}"
            
            # Map our status to GitLab status
            gitlab_status_map = {
                "success": "success",
                "failure": "failed",
                "error": "failed",
                "pending": "pending",
                "running": "running"
            }
            
            payload = {
                "state": gitlab_status_map.get(status, "pending"),
                "description": description,
                "name": "distributed-testing-framework",
                "context": "distributed-testing"
            }
            
            async with self.session.post(url, params=payload) as response:
                if response.status == 201:
                    logger.info(f"Set GitLab build status to {status}: {description}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Error setting GitLab build status: {response.status} - {error_text}")
                    return False
        
        except Exception as e:
            logger.error(f"Exception setting GitLab build status: {str(e)}")
            return False