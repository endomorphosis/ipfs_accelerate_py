#!/usr/bin/env python3
"""
TestRail Connector for Distributed Testing Framework

This module provides a connector for interacting with TestRail's API to create test runs,
test cases, submit test results, and fetch testing plans.
"""

import anyio
import base64
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import aiohttp

# Import the standardized interface
from test.tests.distributed.distributed_testing.external_systems.api_interface import (
    ExternalSystemInterface,
    ConnectorCapabilities,
    ExternalSystemResult,
    ExternalSystemFactory
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestRailConnector(ExternalSystemInterface):
    """
    Connector for interacting with TestRail's API.
    
    This connector implements the standardized ExternalSystemInterface for TestRail
    and provides methods for managing test cases, test runs, and results.
    """
    
    def __init__(self):
        """
        Initialize the TestRail connector.
        """
        self.url = None
        self.username = None
        self.api_key = None
        self.session = None
        self.project_id = None
        self.api_base = None
        self.rate_limit_sleep = 1.0  # seconds to sleep when rate limited
        
        # Cache for commonly used data
        self.cache = {
            "projects": [],
            "test_cases": {},
            "test_suites": {},
            "test_runs": {},
            "case_types": [],
            "case_priorities": [],
            "case_templates": []
        }
        
        # Capabilities
        self.capabilities = ConnectorCapabilities(
            supports_create=True,
            supports_update=True,
            supports_delete=True,
            supports_query=True,
            supports_batch_operations=False,
            supports_attachments=True,
            supports_comments=True,
            supports_custom_fields=True,
            supports_relationships=True,
            supports_history=True,
            item_types=["test_case", "test_run", "test_result", "test_plan", "milestone", "section"],
            query_operators=["=", "!=", "<", "<=", ">", ">=", "IN", "NOT IN"],
            max_batch_size=0,
            rate_limit=180,  # Common TestRail rate limit
            supports_test_plans=True,  # TestRail specific capability
            supports_milestones=True,  # TestRail specific capability
            supports_configurations=True,  # TestRail specific capability
            supports_requirements=False  # TestRail doesn't support requirements management
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the TestRail connector with configuration.
        
        Args:
            config: Configuration dictionary containing:
                   - url: TestRail server URL
                   - username: TestRail username
                   - api_key: TestRail API key
                   - project_id: Default TestRail project ID
            
        Returns:
            True if initialization succeeded
        """
        self.url = config.get("url")
        self.username = config.get("username")
        self.api_key = config.get("api_key")
        self.project_id = config.get("project_id")
        
        if not self.url:
            logger.error("TestRail URL is required")
            return False
        
        if not self.username:
            logger.error("TestRail username is required")
            return False
        
        if not self.api_key:
            logger.error("TestRail API key is required")
            return False
        
        # Normalize URL
        if self.url.endswith("/"):
            self.url = self.url[:-1]
        
        # Set API base URL
        self.api_base = f"{self.url}/index.php?/api/v2"
        
        logger.info(f"TestRailConnector initialized for server {self.url}")
        return True
    
    async def _ensure_session(self):
        """Ensure an aiohttp session exists with proper authentication."""
        if self.session is None:
            auth_str = f"{self.username}:{self.api_key}"
            auth = aiohttp.BasicAuth(self.username, self.api_key)
            
            self.session = aiohttp.ClientSession(
                auth=auth,
                headers={
                    "User-Agent": "DistributedTestingFramework",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
    
    async def connect(self) -> bool:
        """
        Establish connection to TestRail and validate credentials.
        
        Returns:
            True if connection succeeded
        """
        await self._ensure_session()
        
        try:
            # Get user info to validate credentials
            url = f"{self.api_base}/get_user_by_email&email={self.username}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Connected to TestRail as {data.get('name')} ({data.get('email')})")
                    
                    # Prefetch common metadata for better performance
                    await self._prefetch_metadata()
                    
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to connect to TestRail: {response.status} - {error_text}")
                    return False
                
        except Exception as e:
            logger.error(f"Exception connecting to TestRail: {str(e)}")
            return False
    
    async def is_connected(self) -> bool:
        """
        Check if the connector is currently connected to TestRail.
        
        Returns:
            True if connected
        """
        if self.session is None:
            return False
        
        try:
            # Simple check using get_projects API
            url = f"{self.api_base}/get_projects"
            
            async with self.session.get(url) as response:
                return response.status == 200
                
        except Exception:
            return False
    
    async def _prefetch_metadata(self):
        """Prefetch common metadata like projects, case types, etc."""
        try:
            # Get projects
            url = f"{self.api_base}/get_projects"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.cache["projects"] = data
                    logger.debug(f"Cached {len(data)} projects")
            
            # Get case types
            url = f"{self.api_base}/get_case_types"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.cache["case_types"] = data
                    logger.debug(f"Cached {len(data)} case types")
            
            # Get case priorities
            url = f"{self.api_base}/get_priorities"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.cache["case_priorities"] = data
                    logger.debug(f"Cached {len(data)} case priorities")
            
            # If project ID is provided, get project-specific data
            if self.project_id:
                # Get test suites for the project
                url = f"{self.api_base}/get_suites/{self.project_id}"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.cache["test_suites"][self.project_id] = data
                        logger.debug(f"Cached {len(data)} test suites for project {self.project_id}")
                
                # Get case templates for the project
                url = f"{self.api_base}/get_templates/{self.project_id}"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.cache["case_templates"] = data
                        logger.debug(f"Cached {len(data)} case templates")
            
        except Exception as e:
            logger.warning(f"Error prefetching TestRail metadata: {str(e)}")
    
    async def _handle_rate_limit(self, response):
        """
        Handle rate limiting by pausing when necessary.
        
        Args:
            response: The aiohttp response object
            
        Returns:
            The original response
        """
        if response.status == 429:  # Too Many Requests
            retry_after = response.headers.get("Retry-After")
            
            if retry_after:
                try:
                    seconds = int(retry_after)
                except ValueError:
                    seconds = self.rate_limit_sleep
            else:
                seconds = self.rate_limit_sleep
                
            logger.warning(f"TestRail rate limit hit, pausing for {seconds} seconds")
            await anyio.sleep(seconds)
        
        return response
    
    async def execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an operation on TestRail.
        
        Args:
            operation: The operation to execute
            params: Parameters for the operation
            
        Returns:
            Dictionary with operation result
        """
        await self._ensure_session()
        
        try:
            result = ExternalSystemResult(
                success=False,
                operation=operation,
                error_message="Operation not implemented",
                error_code="NOT_IMPLEMENTED"
            )
            
            # Map common operations to TestRail API calls
            if operation == "add_test_run":
                result = await self._add_test_run(params)
            elif operation == "add_test_case":
                result = await self._add_test_case(params)
            elif operation == "add_test_result":
                result = await self._add_test_result(params)
            elif operation == "add_test_plan":
                result = await self._add_test_plan(params)
            elif operation == "get_test_case":
                result = await self._get_test_case(params)
            elif operation == "get_test_run":
                result = await self._get_test_run(params)
            elif operation == "get_test_results":
                result = await self._get_test_results(params)
            elif operation == "get_test_plan":
                result = await self._get_test_plan(params)
            elif operation == "update_test_case":
                result = await self._update_test_case(params)
            elif operation == "update_test_run":
                result = await self._update_test_run(params)
            elif operation == "close_test_run":
                result = await self._close_test_run(params)
            elif operation == "delete_test_run":
                result = await self._delete_test_run(params)
            elif operation == "delete_test_case":
                result = await self._delete_test_case(params)
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Exception executing TestRail operation {operation}: {str(e)}")
            
            return ExternalSystemResult(
                success=False,
                operation=operation,
                error_message=str(e),
                error_code="EXCEPTION"
            ).to_dict()
    
    async def _add_test_run(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Create a test run in TestRail.
        
        Args:
            params: Parameters for creating a test run
            
        Returns:
            ExternalSystemResult with test run details
        """
        project_id = params.get("project_id", self.project_id)
        
        if not project_id:
            return ExternalSystemResult(
                success=False,
                operation="add_test_run",
                error_message="Project ID is required",
                error_code="MISSING_PROJECT_ID"
            )
        
        suite_id = params.get("suite_id")
        name = params.get("name")
        
        if not name:
            return ExternalSystemResult(
                success=False,
                operation="add_test_run",
                error_message="Name is required",
                error_code="MISSING_NAME"
            )
        
        # Prepare data
        data = {
            "name": name,
            "include_all": params.get("include_all", True)
        }
        
        # Add optional fields
        for field in ["description", "milestone_id", "assignedto_id"]:
            if field in params:
                data[field] = params[field]
        
        # Add case IDs if provided
        if "case_ids" in params:
            data["include_all"] = False
            data["case_ids"] = params["case_ids"]
        
        # API endpoint depends on whether suite is specified
        if suite_id:
            url = f"{self.api_base}/add_run/{project_id}/{suite_id}"
        else:
            url = f"{self.api_base}/add_run/{project_id}"
        
        try:
            async with self.session.post(url, json=data) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 201]:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="add_test_run",
                        result_data=data
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="add_test_run",
                        error_message=f"Failed to create test run: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="add_test_run",
                error_message=f"Exception creating test run: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _add_test_case(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Create a test case in TestRail.
        
        Args:
            params: Parameters for creating a test case
            
        Returns:
            ExternalSystemResult with test case details
        """
        project_id = params.get("project_id", self.project_id)
        section_id = params.get("section_id")
        title = params.get("title")
        
        if not project_id:
            return ExternalSystemResult(
                success=False,
                operation="add_test_case",
                error_message="Project ID is required",
                error_code="MISSING_PROJECT_ID"
            )
        
        if not section_id:
            return ExternalSystemResult(
                success=False,
                operation="add_test_case",
                error_message="Section ID is required",
                error_code="MISSING_SECTION_ID"
            )
        
        if not title:
            return ExternalSystemResult(
                success=False,
                operation="add_test_case",
                error_message="Title is required",
                error_code="MISSING_TITLE"
            )
        
        # Prepare data
        data = {
            "title": title
        }
        
        # Add optional fields
        for field in ["template_id", "type_id", "priority_id", "estimate", "milestone_id",
                      "refs", "custom_preconds", "custom_steps", "custom_expected"]:
            if field in params:
                data[field] = params[field]
        
        url = f"{self.api_base}/add_case/{section_id}"
        
        try:
            async with self.session.post(url, json=data) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 201]:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="add_test_case",
                        result_data=data
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="add_test_case",
                        error_message=f"Failed to create test case: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="add_test_case",
                error_message=f"Exception creating test case: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _add_test_result(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Add a test result in TestRail.
        
        Args:
            params: Parameters for adding a test result
            
        Returns:
            ExternalSystemResult with test result details
        """
        test_id = params.get("test_id")
        status_id = params.get("status_id")
        
        if not test_id:
            return ExternalSystemResult(
                success=False,
                operation="add_test_result",
                error_message="Test ID is required",
                error_code="MISSING_TEST_ID"
            )
        
        if status_id is None:
            return ExternalSystemResult(
                success=False,
                operation="add_test_result",
                error_message="Status ID is required",
                error_code="MISSING_STATUS_ID"
            )
        
        # Prepare data
        data = {
            "status_id": status_id
        }
        
        # Add optional fields
        for field in ["comment", "version", "elapsed", "defects", "assignedto_id",
                      "custom_step_results"]:
            if field in params:
                data[field] = params[field]
        
        url = f"{self.api_base}/add_result/{test_id}"
        
        try:
            async with self.session.post(url, json=data) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 201]:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="add_test_result",
                        result_data=data
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="add_test_result",
                        error_message=f"Failed to add test result: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="add_test_result",
                error_message=f"Exception adding test result: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _add_test_plan(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Create a test plan in TestRail.
        
        Args:
            params: Parameters for creating a test plan
            
        Returns:
            ExternalSystemResult with test plan details
        """
        project_id = params.get("project_id", self.project_id)
        name = params.get("name")
        
        if not project_id:
            return ExternalSystemResult(
                success=False,
                operation="add_test_plan",
                error_message="Project ID is required",
                error_code="MISSING_PROJECT_ID"
            )
        
        if not name:
            return ExternalSystemResult(
                success=False,
                operation="add_test_plan",
                error_message="Name is required",
                error_code="MISSING_NAME"
            )
        
        # Prepare data
        data = {
            "name": name
        }
        
        # Add optional fields
        for field in ["description", "milestone_id", "entries"]:
            if field in params:
                data[field] = params[field]
        
        url = f"{self.api_base}/add_plan/{project_id}"
        
        try:
            async with self.session.post(url, json=data) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 201]:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="add_test_plan",
                        result_data=data
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="add_test_plan",
                        error_message=f"Failed to create test plan: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="add_test_plan",
                error_message=f"Exception creating test plan: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_test_case(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get a test case from TestRail.
        
        Args:
            params: Parameters with case_id
            
        Returns:
            ExternalSystemResult with test case details
        """
        case_id = params.get("case_id")
        
        if not case_id:
            return ExternalSystemResult(
                success=False,
                operation="get_test_case",
                error_message="Case ID is required",
                error_code="MISSING_CASE_ID"
            )
        
        url = f"{self.api_base}/get_case/{case_id}"
        
        try:
            async with self.session.get(url) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="get_test_case",
                        result_data=data
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_test_case",
                        error_message=f"Failed to get test case: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_test_case",
                error_message=f"Exception getting test case: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_test_run(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get a test run from TestRail.
        
        Args:
            params: Parameters with run_id
            
        Returns:
            ExternalSystemResult with test run details
        """
        run_id = params.get("run_id")
        
        if not run_id:
            return ExternalSystemResult(
                success=False,
                operation="get_test_run",
                error_message="Run ID is required",
                error_code="MISSING_RUN_ID"
            )
        
        url = f"{self.api_base}/get_run/{run_id}"
        
        try:
            async with self.session.get(url) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="get_test_run",
                        result_data=data
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_test_run",
                        error_message=f"Failed to get test run: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_test_run",
                error_message=f"Exception getting test run: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_test_results(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get test results from TestRail.
        
        Args:
            params: Parameters with test_id or run_id
            
        Returns:
            ExternalSystemResult with test results
        """
        test_id = params.get("test_id")
        run_id = params.get("run_id")
        
        if not test_id and not run_id:
            return ExternalSystemResult(
                success=False,
                operation="get_test_results",
                error_message="Either Test ID or Run ID is required",
                error_code="MISSING_ID"
            )
        
        # Get results for a specific test
        if test_id:
            url = f"{self.api_base}/get_results/{test_id}"
        # Get results for all tests in a run
        else:
            url = f"{self.api_base}/get_results_for_run/{run_id}"
        
        try:
            async with self.session.get(url) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="get_test_results",
                        result_data=data
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_test_results",
                        error_message=f"Failed to get test results: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_test_results",
                error_message=f"Exception getting test results: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_test_plan(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get a test plan from TestRail.
        
        Args:
            params: Parameters with plan_id
            
        Returns:
            ExternalSystemResult with test plan details
        """
        plan_id = params.get("plan_id")
        
        if not plan_id:
            return ExternalSystemResult(
                success=False,
                operation="get_test_plan",
                error_message="Plan ID is required",
                error_code="MISSING_PLAN_ID"
            )
        
        url = f"{self.api_base}/get_plan/{plan_id}"
        
        try:
            async with self.session.get(url) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="get_test_plan",
                        result_data=data
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_test_plan",
                        error_message=f"Failed to get test plan: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_test_plan",
                error_message=f"Exception getting test plan: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _update_test_case(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Update a test case in TestRail.
        
        Args:
            params: Parameters for updating a test case
            
        Returns:
            ExternalSystemResult with updated test case details
        """
        case_id = params.get("case_id")
        
        if not case_id:
            return ExternalSystemResult(
                success=False,
                operation="update_test_case",
                error_message="Case ID is required",
                error_code="MISSING_CASE_ID"
            )
        
        # Prepare data
        data = {}
        
        # Add fields to update
        for field in ["title", "template_id", "type_id", "priority_id", "estimate", "milestone_id", 
                      "refs", "custom_preconds", "custom_steps", "custom_expected"]:
            if field in params:
                data[field] = params[field]
        
        if not data:
            return ExternalSystemResult(
                success=False,
                operation="update_test_case",
                error_message="No fields to update",
                error_code="NO_UPDATE_FIELDS"
            )
        
        url = f"{self.api_base}/update_case/{case_id}"
        
        try:
            async with self.session.post(url, json=data) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="update_test_case",
                        result_data=data
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="update_test_case",
                        error_message=f"Failed to update test case: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="update_test_case",
                error_message=f"Exception updating test case: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _update_test_run(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Update a test run in TestRail.
        
        Args:
            params: Parameters for updating a test run
            
        Returns:
            ExternalSystemResult with updated test run details
        """
        run_id = params.get("run_id")
        
        if not run_id:
            return ExternalSystemResult(
                success=False,
                operation="update_test_run",
                error_message="Run ID is required",
                error_code="MISSING_RUN_ID"
            )
        
        # Prepare data
        data = {}
        
        # Add fields to update
        for field in ["name", "description", "milestone_id", "include_all", "case_ids", "assignedto_id"]:
            if field in params:
                data[field] = params[field]
        
        if not data:
            return ExternalSystemResult(
                success=False,
                operation="update_test_run",
                error_message="No fields to update",
                error_code="NO_UPDATE_FIELDS"
            )
        
        url = f"{self.api_base}/update_run/{run_id}"
        
        try:
            async with self.session.post(url, json=data) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="update_test_run",
                        result_data=data
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="update_test_run",
                        error_message=f"Failed to update test run: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="update_test_run",
                error_message=f"Exception updating test run: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _close_test_run(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Close a test run in TestRail.
        
        Args:
            params: Parameters with run_id
            
        Returns:
            ExternalSystemResult with operation result
        """
        run_id = params.get("run_id")
        
        if not run_id:
            return ExternalSystemResult(
                success=False,
                operation="close_test_run",
                error_message="Run ID is required",
                error_code="MISSING_RUN_ID"
            )
        
        url = f"{self.api_base}/close_run/{run_id}"
        
        try:
            async with self.session.post(url, json={}) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="close_test_run",
                        result_data=data
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="close_test_run",
                        error_message=f"Failed to close test run: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="close_test_run",
                error_message=f"Exception closing test run: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _delete_test_run(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Delete a test run in TestRail.
        
        Args:
            params: Parameters with run_id
            
        Returns:
            ExternalSystemResult with operation result
        """
        run_id = params.get("run_id")
        
        if not run_id:
            return ExternalSystemResult(
                success=False,
                operation="delete_test_run",
                error_message="Run ID is required",
                error_code="MISSING_RUN_ID"
            )
        
        url = f"{self.api_base}/delete_run/{run_id}"
        
        try:
            async with self.session.post(url, json={}) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 204]:
                    return ExternalSystemResult(
                        success=True,
                        operation="delete_test_run",
                        result_data={"run_id": run_id}
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="delete_test_run",
                        error_message=f"Failed to delete test run: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="delete_test_run",
                error_message=f"Exception deleting test run: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _delete_test_case(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Delete a test case in TestRail.
        
        Args:
            params: Parameters with case_id
            
        Returns:
            ExternalSystemResult with operation result
        """
        case_id = params.get("case_id")
        
        if not case_id:
            return ExternalSystemResult(
                success=False,
                operation="delete_test_case",
                error_message="Case ID is required",
                error_code="MISSING_CASE_ID"
            )
        
        url = f"{self.api_base}/delete_case/{case_id}"
        
        try:
            async with self.session.post(url, json={}) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 204]:
                    return ExternalSystemResult(
                        success=True,
                        operation="delete_test_case",
                        result_data={"case_id": case_id}
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="delete_test_case",
                        error_message=f"Failed to delete test case: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="delete_test_case",
                error_message=f"Exception deleting test case: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query TestRail objects.
        
        Args:
            query_params: Query parameters
            
        Returns:
            List of query results
        """
        item_type = query_params.get("item_type", "test_case")
        project_id = query_params.get("project_id", self.project_id)
        
        if not project_id:
            logger.error("Project ID is required for TestRail queries")
            return []
        
        try:
            # Different query approaches based on item type
            if item_type == "test_case":
                # Get test cases for the project
                suite_id = query_params.get("suite_id")
                
                if not suite_id:
                    logger.error("Suite ID is required for querying test cases")
                    return []
                
                url = f"{self.api_base}/get_cases/{project_id}&suite_id={suite_id}"
                
                # Add filters if provided
                for filter_name in ["section_id", "created_after", "created_before", "updated_after", "updated_before", 
                                   "milestone_id", "priority_id", "type_id", "template_id"]:
                    if filter_name in query_params:
                        url += f"&{filter_name}={query_params[filter_name]}"
                
            elif item_type == "test_run":
                # Get test runs for the project
                url = f"{self.api_base}/get_runs/{project_id}"
                
                # Add filters if provided
                for filter_name in ["created_after", "created_before", "is_completed", "milestone_id", "suite_id"]:
                    if filter_name in query_params:
                        url += f"&{filter_name}={query_params[filter_name]}"
                
            elif item_type == "test_plan":
                # Get test plans for the project
                url = f"{self.api_base}/get_plans/{project_id}"
                
                # Add filters if provided
                for filter_name in ["created_after", "created_before", "is_completed", "milestone_id"]:
                    if filter_name in query_params:
                        url += f"&{filter_name}={query_params[filter_name]}"
                
            elif item_type == "milestone":
                # Get milestones for the project
                url = f"{self.api_base}/get_milestones/{project_id}"
                
                # Add filters if provided
                for filter_name in ["is_completed", "is_started"]:
                    if filter_name in query_params:
                        url += f"&{filter_name}={query_params[filter_name]}"
                
            else:
                logger.error(f"Unsupported item type for TestRail query: {item_type}")
                return []
            
            # Execute the query
            async with self.session.get(url) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    # For most APIs, this will be a list of objects
                    if isinstance(data, list):
                        return data
                    # Some APIs return an object with a key containing the list
                    elif isinstance(data, dict):
                        # Try to find a list in the response
                        for key, value in data.items():
                            if isinstance(value, list):
                                return value
                        # If no list is found, return the object as a single-item list
                        return [data]
                    else:
                        return []
                else:
                    logger.error(f"TestRail query failed: {response.status}")
                    return []
                
        except Exception as e:
            logger.error(f"Exception querying TestRail: {str(e)}")
            return []
    
    async def create_item(self, item_type: str, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an item in the TestRail system.
        
        Args:
            item_type: Type of item to create
            item_data: Item data
            
        Returns:
            Dictionary with created item details
        """
        if item_type == "test_case":
            result = await self._add_test_case(item_data)
        elif item_type == "test_run":
            result = await self._add_test_run(item_data)
        elif item_type == "test_result":
            result = await self._add_test_result(item_data)
        elif item_type == "test_plan":
            result = await self._add_test_plan(item_data)
        else:
            result = ExternalSystemResult(
                success=False,
                operation=f"create_{item_type}",
                error_message=f"Unsupported item type: {item_type}",
                error_code="UNSUPPORTED_ITEM_TYPE"
            )
        
        if result.success:
            return result.result_data
        else:
            raise Exception(f"Failed to create {item_type}: {result.error_message}")
    
    async def update_item(self, item_type: str, item_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an item in the TestRail system.
        
        Args:
            item_type: Type of item to update
            item_id: ID of the item to update
            update_data: Data to update
            
        Returns:
            True if update succeeded
        """
        if item_type == "test_case":
            update_data["case_id"] = item_id
            result = await self._update_test_case(update_data)
        elif item_type == "test_run":
            update_data["run_id"] = item_id
            result = await self._update_test_run(update_data)
        else:
            result = ExternalSystemResult(
                success=False,
                operation=f"update_{item_type}",
                error_message=f"Unsupported item type: {item_type}",
                error_code="UNSUPPORTED_ITEM_TYPE"
            )
        
        return result.success
    
    async def delete_item(self, item_type: str, item_id: str) -> bool:
        """
        Delete an item from the TestRail system.
        
        Args:
            item_type: Type of item to delete
            item_id: ID of the item to delete
            
        Returns:
            True if deletion succeeded
        """
        if item_type == "test_case":
            result = await self._delete_test_case({"case_id": item_id})
        elif item_type == "test_run":
            result = await self._delete_test_run({"run_id": item_id})
        else:
            result = ExternalSystemResult(
                success=False,
                operation=f"delete_{item_type}",
                error_message=f"Unsupported item type: {item_type}",
                error_code="UNSUPPORTED_ITEM_TYPE"
            )
        
        return result.success
    
    async def get_item(self, item_type: str, item_id: str) -> Dict[str, Any]:
        """
        Get an item from the TestRail system.
        
        Args:
            item_type: Type of item to get
            item_id: ID of the item to get
            
        Returns:
            Dictionary with item details
        """
        if item_type == "test_case":
            result = await self._get_test_case({"case_id": item_id})
        elif item_type == "test_run":
            result = await self._get_test_run({"run_id": item_id})
        elif item_type == "test_plan":
            result = await self._get_test_plan({"plan_id": item_id})
        elif item_type == "test_result":
            result = await self._get_test_results({"test_id": item_id})
        else:
            result = ExternalSystemResult(
                success=False,
                operation=f"get_{item_type}",
                error_message=f"Unsupported item type: {item_type}",
                error_code="UNSUPPORTED_ITEM_TYPE"
            )
        
        if result.success:
            return result.result_data
        else:
            raise Exception(f"Failed to get {item_type}: {result.error_message}")
    
    async def system_info(self) -> Dict[str, Any]:
        """
        Get information about the TestRail system.
        
        Returns:
            Dictionary with system information
        """
        await self._ensure_session()
        
        try:
            # Get server info from any endpoint that returns it
            url = f"{self.api_base}/get_projects"
            server_info = {"error": "Not available"}
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    # TestRail doesn't have a dedicated server info endpoint
                    server_info = {"status": "active"}
                else:
                    server_info = {"error": f"Failed to connect: {response.status}"}
            
            # Build system info
            return {
                "system_type": "testrail",
                "connected": await self.is_connected(),
                "server_url": self.url,
                "server_info": server_info,
                "api_version": "v2",
                "username": self.username,
                "project_id": self.project_id,
                "capabilities": self.capabilities.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Exception getting TestRail system info: {str(e)}")
            
            return {
                "system_type": "testrail",
                "connected": False,
                "server_url": self.url,
                "error": str(e),
                "capabilities": self.capabilities.to_dict()
            }
    
    async def close(self) -> None:
        """
        Close the connection to TestRail and clean up resources.
        
        Returns:
            None
        """
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("TestRail connection closed")


# Register with factory
ExternalSystemFactory.register_connector("testrail", TestRailConnector)