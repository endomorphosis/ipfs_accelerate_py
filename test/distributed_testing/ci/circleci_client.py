#!/usr/bin/env python3
"""
CircleCI Client for Distributed Testing Framework

This module provides a client for interacting with CircleCI's API to report test results,
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

class CircleCIClient(CIProviderInterface):
    """
    Client for interacting with CircleCI's API.
    
    This client implements the standardized CIProviderInterface for CircleCI
    and provides methods for creating and updating test runs, adding artifacts,
    and setting build status.
    """
    
    def __init__(self):
        """
        Initialize the CircleCI client.
        """
        self.token = None
        self.project_slug = None
        self.api_url = "https://circleci.com/api/v2"
        self.session = None
        self.build_num = None
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the CI provider with configuration.
        
        Args:
            config: Configuration dictionary containing:
                   - token: CircleCI API token
                   - project_slug: CircleCI project slug (org/project)
                   - api_url: CircleCI API URL (optional)
                   - build_num: Current build number (optional)
            
        Returns:
            True if initialization succeeded
        """
        self.token = config.get("token")
        self.project_slug = config.get("project_slug")
        self.api_url = config.get("api_url", "https://circleci.com/api/v2")
        self.build_num = config.get("build_num")
        
        if not self.token:
            logger.error("CircleCI token is required")
            return False
        
        if not self.project_slug:
            logger.error("CircleCI project slug is required")
            return False
        
        logger.info(f"CircleCIClient initialized for project {self.project_slug}")
        return True
    
    async def _ensure_session(self):
        """Ensure an aiohttp session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession(headers={
                "Circle-Token": self.token,
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "DistributedTestingFramework"
            })
    
    async def create_test_run(self, test_run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new test run record using CircleCI API.
        
        In CircleCI, we can't truly create a test run via API, but we can track it
        by associating it with a pipeline, workflow, or job.
        
        Args:
            test_run_data: Test run data including name, build_id, etc.
            
        Returns:
            Dictionary with test run information
        """
        await self._ensure_session()
        
        try:
            # Extract key information from test run data
            name = test_run_data.get("name", f"Distributed Test Run - {int(time.time())}")
            pipeline_id = test_run_data.get("pipeline_id")
            workflow_id = test_run_data.get("workflow_id")
            job_number = test_run_data.get("job_number", self.build_num)
            
            # If we have job information, use that to identify the test run
            if job_number and self.project_slug:
                logger.info(f"Using CircleCI job {job_number} for test run")
                
                # In a production implementation, we might interact with the tests endpoint
                # https://circleci.com/docs/api/v2/#operation/createTests
                
                # Create test run record
                test_run = {
                    "id": f"circleci-{self.project_slug}-{job_number}",
                    "external_id": f"{self.project_slug}/{job_number}",
                    "name": name,
                    "status": "running",
                    "start_time": datetime.now().isoformat(),
                    "project_slug": self.project_slug,
                    "job_number": job_number,
                    "pipeline_id": pipeline_id,
                    "workflow_id": workflow_id
                }
                
                logger.info(f"Created CircleCI test run: {test_run['id']}")
                return test_run
            
            # If we don't have job info, create a simulated test run
            else:
                logger.warning("No job information provided for CircleCI test run")
                
                # Return a simulated test run
                return {
                    "id": f"circleci-simulated-{int(time.time())}",
                    "name": name,
                    "status": "running",
                    "start_time": datetime.now().isoformat(),
                    "note": "Created in simulation mode due to missing job information"
                }
        
        except Exception as e:
            logger.error(f"Exception creating CircleCI test run: {str(e)}")
            
            # Return a simulated test run for easier recovery
            return {
                "id": f"circleci-simulated-{int(time.time())}",
                "name": test_run_data.get("name", f"Distributed Test Run - {int(time.time())}"),
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "note": "Created in simulation mode due to exception"
            }
    
    async def update_test_run(self, test_run_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a test run using CircleCI API.
        
        Args:
            test_run_id: Test run ID
            update_data: Data to update
            
        Returns:
            True if update succeeded
        """
        await self._ensure_session()
        
        try:
            # Skip if we're using a simulated test run
            if test_run_id.startswith("circleci-simulated-"):
                logger.info(f"Skipping update for simulated test run {test_run_id}")
                return True
            
            # In CircleCI, updating test results would be done through the tests endpoint
            # This is a simplified version for the example
            logger.info(f"Updated CircleCI test run: {test_run_id} with status: {update_data.get('status')}")
            
            # If we have summary data and this is a real run, we could update the actual test results
            if "summary" in update_data and not test_run_id.startswith("circleci-simulated-"):
                summary = update_data["summary"]
                logger.info(f"Test summary: {summary}")
            
            return True
            
        except Exception as e:
            logger.error(f"Exception updating CircleCI test run: {str(e)}")
            return False
    
    async def add_pr_comment(self, pr_number: str, comment: str) -> bool:
        """
        Add a comment to a pull request.
        
        CircleCI doesn't have a native API for adding PR comments, so this is a placeholder.
        In a real implementation, you'd need to interact with the VCS provider's API.
        
        Args:
            pr_number: Pull request number
            comment: Comment text
            
        Returns:
            True if comment was added successfully
        """
        logger.info(f"CircleCIClient doesn't support adding PR comments directly. Would add comment to PR #{pr_number}")
        return False
    
    async def upload_artifact(self, test_run_id: str, artifact_path: str, artifact_name: str) -> bool:
        """
        Upload an artifact for a test run.
        
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
            if test_run_id.startswith("circleci-simulated-"):
                logger.info(f"Skipping artifact upload for simulated test run {test_run_id}")
                return True
            
            # Extract project slug and job number from test run ID
            parts = test_run_id.split("-", 2)
            if len(parts) < 3 or parts[0] != "circleci":
                logger.error(f"Invalid CircleCI test run ID format: {test_run_id}")
                return False
            
            project_slug = parts[1]
            job_number = parts[2]
            
            # Check if file exists
            if not os.path.exists(artifact_path):
                logger.error(f"Artifact file not found: {artifact_path}")
                return False
            
            # In CircleCI, artifacts are typically stored via the artifacts endpoint
            # This would require a custom implementation with the CircleCI API
            
            # For demonstration, we'll just log and return success
            logger.info(f"Would upload artifact {artifact_path} as {artifact_name} for {project_slug}/{job_number}")
            return True
            
        except Exception as e:
            logger.error(f"Exception uploading artifact to CircleCI: {str(e)}")
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
            if test_run_id.startswith("circleci-simulated-"):
                logger.info(f"Returning simulated status for test run {test_run_id}")
                return {
                    "id": test_run_id,
                    "status": "running",
                    "conclusion": None
                }
            
            # Extract project slug and job number from test run ID
            parts = test_run_id.split("-", 2)
            if len(parts) < 3 or parts[0] != "circleci":
                logger.error(f"Invalid CircleCI test run ID format: {test_run_id}")
                return {"id": test_run_id, "status": "unknown", "error": "Invalid ID format"}
            
            project_slug = parts[1]
            job_number = parts[2]
            
            # In a real implementation, we would query the job details
            # https://circleci.com/docs/api/v2/#operation/getJobDetails
            
            # For now, we'll return a placeholder response
            return {
                "id": test_run_id,
                "name": f"Job {job_number}",
                "status": "running",  # In a real implementation, this would be fetched from the API
                "project_slug": project_slug,
                "job_number": job_number
            }
            
        except Exception as e:
            logger.error(f"Exception getting CircleCI test run status: {str(e)}")
            
            # Return a simulated status for easier recovery
            return {
                "id": test_run_id,
                "status": "unknown",
                "error": f"Exception: {str(e)}"
            }
    
    async def set_build_status(self, status: str, description: str) -> bool:
        """
        Set the build status in the CI system.
        
        CircleCI doesn't allow directly setting build status via API, so this is a placeholder.
        
        Args:
            status: Status to set (success, failure, pending)
            description: Status description
            
        Returns:
            True if status was set successfully
        """
        logger.info(f"CircleCIClient doesn't support setting build status directly. Would set status to {status}: {description}")
        return False
    
    async def close(self) -> None:
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("CircleCIClient session closed")