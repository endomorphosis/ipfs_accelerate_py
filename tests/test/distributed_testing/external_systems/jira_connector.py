#!/usr/bin/env python3
"""
JIRA Connector for Distributed Testing Framework

This module provides a connector for interacting with JIRA's API to create issues,
update statuses, add comments, and manage test results.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import aiohttp

# Import the standardized interface
from distributed_testing.external_systems.api_interface import (
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

class JiraConnector(ExternalSystemInterface):
    """
    Connector for interacting with JIRA's API.
    
    This connector implements the standardized ExternalSystemInterface for JIRA
    and provides methods for creating issues, updating statuses, adding comments,
    and managing test results.
    """
    
    def __init__(self):
        """
        Initialize the JIRA connector.
        """
        self.token = None
        self.email = None
        self.server_url = None
        self.project_key = None
        self.session = None
        self.user_info = None
        self.api_version = "3"
        self.rate_limit_sleep = 1.0  # seconds to sleep when rate limited
        
        # Field mappings
        self.field_mappings = {
            "summary": "summary",
            "description": "description",
            "issue_type": "issuetype",
            "priority": "priority",
            "assignee": "assignee",
            "reporter": "reporter",
            "labels": "labels",
            "components": "components",
            "fix_versions": "fixVersions",
            "affects_versions": "versions",
            "environment": "environment",
            "due_date": "duedate"
        }
        
        # Cache for issue types, priorities, etc.
        self.cache = {
            "issue_types": [],
            "priorities": [],
            "statuses": [],
            "fields": {},
            "projects": []
        }
        
        # Capabilities
        self.capabilities = ConnectorCapabilities(
            supports_create=True,
            supports_update=True,
            supports_delete=True,
            supports_query=True,
            supports_batch_operations=False,  # JIRA doesn't have true batch API
            supports_attachments=True,
            supports_comments=True,
            supports_custom_fields=True,
            supports_relationships=True,
            supports_history=True,
            item_types=["issue", "comment", "attachment", "worklog", "test_result"],
            query_operators=["=", "!=", "<", "<=", ">", ">=", "~", "!~", "IN", "NOT IN", "IS", "IS NOT"],
            max_batch_size=0,
            rate_limit=300,  # Default JIRA Cloud rate limit (300 requests per minute)
            supports_jql=True,  # JIRA specific capability
            supports_epics=True,  # JIRA specific capability
            supports_sprints=True  # JIRA specific capability
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the JIRA connector with configuration.
        
        Args:
            config: Configuration dictionary containing:
                   - email: JIRA account email address
                   - token: JIRA API token
                   - server_url: JIRA server URL
                   - project_key: Default JIRA project key
                   - api_version: API version (default: 3)
            
        Returns:
            True if initialization succeeded
        """
        self.email = config.get("email")
        self.token = config.get("token")
        self.server_url = config.get("server_url")
        self.project_key = config.get("project_key")
        self.api_version = config.get("api_version", "3")
        
        if not self.email:
            logger.error("JIRA email is required")
            return False
        
        if not self.token:
            logger.error("JIRA API token is required")
            return False
        
        if not self.server_url:
            logger.error("JIRA server URL is required")
            return False
        
        # Normalize server URL
        if self.server_url.endswith("/"):
            self.server_url = self.server_url[:-1]
        
        logger.info(f"JiraConnector initialized for server {self.server_url}")
        return True
    
    async def _ensure_session(self):
        """Ensure an aiohttp session exists with proper authentication."""
        if self.session is None:
            # Create auth header for Basic authentication
            auth = aiohttp.BasicAuth(self.email, self.token)
            
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
        Establish connection to JIRA and validate credentials.
        
        Returns:
            True if connection succeeded
        """
        await self._ensure_session()
        
        try:
            # Get current user info to validate credentials
            url = f"{self.server_url}/rest/api/{self.api_version}/myself"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.user_info = data
                    logger.info(f"Connected to JIRA as {data.get('displayName')} ({data.get('emailAddress')})")
                    
                    # Prefetch common metadata for better performance
                    await self._prefetch_metadata()
                    
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to connect to JIRA: {response.status} - {error_text}")
                    return False
                
        except Exception as e:
            logger.error(f"Exception connecting to JIRA: {str(e)}")
            return False
    
    async def is_connected(self) -> bool:
        """
        Check if the connector is currently connected to JIRA.
        
        Returns:
            True if connected
        """
        if self.session is None or self.user_info is None:
            return False
        
        try:
            # Lightweight check to see if session is still valid
            url = f"{self.server_url}/rest/api/{self.api_version}/myself"
            
            async with self.session.get(url) as response:
                return response.status == 200
                
        except Exception:
            return False
    
    async def _prefetch_metadata(self):
        """Prefetch common metadata like issue types, priorities, etc."""
        try:
            # Get issue types
            url = f"{self.server_url}/rest/api/{self.api_version}/issuetype"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.cache["issue_types"] = data
                    logger.debug(f"Cached {len(data)} issue types")
            
            # Get priorities
            url = f"{self.server_url}/rest/api/{self.api_version}/priority"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.cache["priorities"] = data
                    logger.debug(f"Cached {len(data)} priorities")
            
            # Get projects
            url = f"{self.server_url}/rest/api/{self.api_version}/project"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.cache["projects"] = data
                    logger.debug(f"Cached {len(data)} projects")
            
            # Get field information
            url = f"{self.server_url}/rest/api/{self.api_version}/field"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Convert to map from id to field info
                    self.cache["fields"] = {field["id"]: field for field in data}
                    logger.debug(f"Cached {len(data)} fields")
                    
        except Exception as e:
            logger.warning(f"Error prefetching JIRA metadata: {str(e)}")
    
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
                
            logger.warning(f"JIRA rate limit hit, pausing for {seconds} seconds")
            await asyncio.sleep(seconds)
        
        return response
    
    async def execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an operation on JIRA.
        
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
            
            # Map common operations to JIRA API calls
            if operation == "create_issue":
                result = await self._create_issue(params)
            elif operation == "update_issue":
                result = await self._update_issue(params)
            elif operation == "add_comment":
                result = await self._add_comment(params)
            elif operation == "transition_issue":
                result = await self._transition_issue(params)
            elif operation == "add_attachment":
                result = await self._add_attachment(params)
            elif operation == "add_worklog":
                result = await self._add_worklog(params)
            elif operation == "search_issues":
                result = await self._search_issues(params)
            elif operation == "get_issue":
                result = await self._get_issue(params)
            elif operation == "delete_issue":
                result = await self._delete_issue(params)
            elif operation == "get_issue_transitions":
                result = await self._get_issue_transitions(params)
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Exception executing JIRA operation {operation}: {str(e)}")
            
            return ExternalSystemResult(
                success=False,
                operation=operation,
                error_message=str(e),
                error_code="EXCEPTION"
            ).to_dict()
    
    async def _create_issue(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Create a JIRA issue.
        
        Args:
            params: Issue creation parameters
            
        Returns:
            ExternalSystemResult with issue details
        """
        project_key = params.get("project_key", self.project_key)
        
        if not project_key:
            return ExternalSystemResult(
                success=False,
                operation="create_issue",
                error_message="Project key is required",
                error_code="MISSING_PROJECT"
            )
        
        issue_type = params.get("issue_type", "Bug")
        summary = params.get("summary")
        
        if not summary:
            return ExternalSystemResult(
                success=False,
                operation="create_issue",
                error_message="Summary is required",
                error_code="MISSING_SUMMARY"
            )
        
        # Build issue fields
        fields = {
            "project": {"key": project_key},
            "issuetype": {"name": issue_type},
            "summary": summary
        }
        
        # Add description if provided
        if "description" in params:
            fields["description"] = params["description"]
        
        # Add other fields
        for key, value in params.items():
            if key in ["project_key", "issue_type", "summary", "description"]:
                continue
            
            # Map field name if needed
            field_name = self.field_mappings.get(key, key)
            
            # Handle specific field types
            if key == "labels":
                fields[field_name] = value if isinstance(value, list) else [value]
            elif key in ["components", "fix_versions", "affects_versions"]:
                fields[field_name] = [{"name": name} for name in value] if isinstance(value, list) else [{"name": value}]
            elif key == "assignee":
                fields[field_name] = {"name": value}
            else:
                fields[field_name] = value
        
        # Create issue
        url = f"{self.server_url}/rest/api/{self.api_version}/issue"
        
        try:
            payload = {"fields": fields}
            
            async with self.session.post(url, json=payload) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 201]:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="create_issue",
                        result_data={
                            "id": data.get("id"),
                            "key": data.get("key"),
                            "self": data.get("self")
                        }
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="create_issue",
                        error_message=f"Failed to create issue: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="create_issue",
                error_message=f"Exception creating issue: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _update_issue(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Update a JIRA issue.
        
        Args:
            params: Issue update parameters
            
        Returns:
            ExternalSystemResult with update result
        """
        issue_key = params.get("issue_key")
        
        if not issue_key:
            return ExternalSystemResult(
                success=False,
                operation="update_issue",
                error_message="Issue key is required",
                error_code="MISSING_ISSUE_KEY"
            )
        
        # Build fields to update
        fields = {}
        
        for key, value in params.items():
            if key == "issue_key":
                continue
            
            # Map field name if needed
            field_name = self.field_mappings.get(key, key)
            
            # Handle specific field types
            if key == "labels":
                fields[field_name] = value if isinstance(value, list) else [value]
            elif key in ["components", "fix_versions", "affects_versions"]:
                fields[field_name] = [{"name": name} for name in value] if isinstance(value, list) else [{"name": value}]
            elif key == "assignee":
                fields[field_name] = {"name": value}
            else:
                fields[field_name] = value
        
        # Update issue
        url = f"{self.server_url}/rest/api/{self.api_version}/issue/{issue_key}"
        
        try:
            payload = {"fields": fields}
            
            async with self.session.put(url, json=payload) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 204]:
                    return ExternalSystemResult(
                        success=True,
                        operation="update_issue",
                        result_data={"issue_key": issue_key}
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="update_issue",
                        error_message=f"Failed to update issue: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="update_issue",
                error_message=f"Exception updating issue: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _add_comment(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Add a comment to a JIRA issue.
        
        Args:
            params: Comment parameters
            
        Returns:
            ExternalSystemResult with comment details
        """
        issue_key = params.get("issue_key")
        
        if not issue_key:
            return ExternalSystemResult(
                success=False,
                operation="add_comment",
                error_message="Issue key is required",
                error_code="MISSING_ISSUE_KEY"
            )
        
        body = params.get("body")
        
        if not body:
            return ExternalSystemResult(
                success=False,
                operation="add_comment",
                error_message="Comment body is required",
                error_code="MISSING_BODY"
            )
        
        # Add comment
        url = f"{self.server_url}/rest/api/{self.api_version}/issue/{issue_key}/comment"
        
        try:
            # Handle different comment formats based on API version
            if self.api_version == "2":
                payload = {"body": body}
            else:
                # API v3 uses document format
                payload = {
                    "body": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": body
                                    }
                                ]
                            }
                        ]
                    }
                }
            
            async with self.session.post(url, json=payload) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 201]:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="add_comment",
                        result_data={
                            "id": data.get("id"),
                            "self": data.get("self")
                        }
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="add_comment",
                        error_message=f"Failed to add comment: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="add_comment",
                error_message=f"Exception adding comment: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _transition_issue(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Transition a JIRA issue to a new status.
        
        Args:
            params: Transition parameters
            
        Returns:
            ExternalSystemResult with transition result
        """
        issue_key = params.get("issue_key")
        
        if not issue_key:
            return ExternalSystemResult(
                success=False,
                operation="transition_issue",
                error_message="Issue key is required",
                error_code="MISSING_ISSUE_KEY"
            )
        
        transition_id = params.get("transition_id")
        transition_name = params.get("transition_name")
        
        if not transition_id and not transition_name:
            return ExternalSystemResult(
                success=False,
                operation="transition_issue",
                error_message="Either transition_id or transition_name is required",
                error_code="MISSING_TRANSITION"
            )
        
        # If only name is provided, find the ID
        if not transition_id and transition_name:
            # Get available transitions
            transitions = await self._get_issue_transitions({"issue_key": issue_key})
            
            if not transitions.success:
                return transitions
            
            # Find transition by name
            for transition in transitions.result_data.get("transitions", []):
                if transition.get("name") == transition_name:
                    transition_id = transition.get("id")
                    break
            
            if not transition_id:
                return ExternalSystemResult(
                    success=False,
                    operation="transition_issue",
                    error_message=f"Transition '{transition_name}' not found",
                    error_code="INVALID_TRANSITION"
                )
        
        # Transition issue
        url = f"{self.server_url}/rest/api/{self.api_version}/issue/{issue_key}/transitions"
        
        try:
            payload = {
                "transition": {
                    "id": transition_id
                }
            }
            
            # Add fields if provided
            if "fields" in params:
                payload["fields"] = params["fields"]
            
            async with self.session.post(url, json=payload) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 204]:
                    return ExternalSystemResult(
                        success=True,
                        operation="transition_issue",
                        result_data={"issue_key": issue_key, "transition_id": transition_id}
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="transition_issue",
                        error_message=f"Failed to transition issue: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="transition_issue",
                error_message=f"Exception transitioning issue: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_issue_transitions(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get available transitions for a JIRA issue.
        
        Args:
            params: Parameters with issue_key
            
        Returns:
            ExternalSystemResult with transitions data
        """
        issue_key = params.get("issue_key")
        
        if not issue_key:
            return ExternalSystemResult(
                success=False,
                operation="get_issue_transitions",
                error_message="Issue key is required",
                error_code="MISSING_ISSUE_KEY"
            )
        
        # Get transitions
        url = f"{self.server_url}/rest/api/{self.api_version}/issue/{issue_key}/transitions"
        
        try:
            async with self.session.get(url) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="get_issue_transitions",
                        result_data=data
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_issue_transitions",
                        error_message=f"Failed to get transitions: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_issue_transitions",
                error_message=f"Exception getting transitions: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _add_attachment(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Add an attachment to a JIRA issue.
        
        Args:
            params: Attachment parameters
            
        Returns:
            ExternalSystemResult with attachment details
        """
        issue_key = params.get("issue_key")
        
        if not issue_key:
            return ExternalSystemResult(
                success=False,
                operation="add_attachment",
                error_message="Issue key is required",
                error_code="MISSING_ISSUE_KEY"
            )
        
        file_path = params.get("file_path")
        
        if not file_path:
            return ExternalSystemResult(
                success=False,
                operation="add_attachment",
                error_message="File path is required",
                error_code="MISSING_FILE_PATH"
            )
        
        # Add attachment
        url = f"{self.server_url}/rest/api/{self.api_version}/issue/{issue_key}/attachments"
        
        try:
            # Create multipart form data
            with open(file_path, "rb") as f:
                file_data = f.read()
            
            file_name = params.get("file_name", file_path.split("/")[-1])
            
            # JIRA API requires a specific header for attachments
            headers = {"X-Atlassian-Token": "no-check"}
            
            # Create form data
            form = aiohttp.FormData()
            form.add_field("file", file_data, filename=file_name)
            
            async with self.session.post(url, data=form, headers=headers) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 201]:
                    data = await response.json()
                    
                    # Data is a list of attachments
                    if isinstance(data, list) and len(data) > 0:
                        attachment = data[0]
                        
                        return ExternalSystemResult(
                            success=True,
                            operation="add_attachment",
                            result_data={
                                "id": attachment.get("id"),
                                "filename": attachment.get("filename"),
                                "size": attachment.get("size"),
                                "content_type": attachment.get("mimeType"),
                                "created": attachment.get("created")
                            }
                        )
                    else:
                        return ExternalSystemResult(
                            success=True,
                            operation="add_attachment",
                            result_data={"message": "Attachment added"}
                        )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="add_attachment",
                        error_message=f"Failed to add attachment: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="add_attachment",
                error_message=f"Exception adding attachment: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _add_worklog(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Add a worklog entry to a JIRA issue.
        
        Args:
            params: Worklog parameters
            
        Returns:
            ExternalSystemResult with worklog details
        """
        issue_key = params.get("issue_key")
        
        if not issue_key:
            return ExternalSystemResult(
                success=False,
                operation="add_worklog",
                error_message="Issue key is required",
                error_code="MISSING_ISSUE_KEY"
            )
        
        time_spent = params.get("time_spent")
        
        if not time_spent:
            return ExternalSystemResult(
                success=False,
                operation="add_worklog",
                error_message="Time spent is required",
                error_code="MISSING_TIME_SPENT"
            )
        
        # Add worklog
        url = f"{self.server_url}/rest/api/{self.api_version}/issue/{issue_key}/worklog"
        
        try:
            payload = {
                "timeSpent": time_spent
            }
            
            # Add comment if provided
            if "comment" in params:
                if self.api_version == "2":
                    payload["comment"] = params["comment"]
                else:
                    # API v3 uses document format
                    payload["comment"] = {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": params["comment"]
                                    }
                                ]
                            }
                        ]
                    }
            
            # Add start date if provided
            if "started" in params:
                payload["started"] = params["started"]
            
            async with self.session.post(url, json=payload) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 201]:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="add_worklog",
                        result_data={
                            "id": data.get("id"),
                            "self": data.get("self"),
                            "time_spent": data.get("timeSpent"),
                            "time_spent_seconds": data.get("timeSpentSeconds")
                        }
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="add_worklog",
                        error_message=f"Failed to add worklog: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="add_worklog",
                error_message=f"Exception adding worklog: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _search_issues(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Search for JIRA issues using JQL.
        
        Args:
            params: Search parameters
            
        Returns:
            ExternalSystemResult with search results
        """
        jql = params.get("jql")
        
        if not jql:
            return ExternalSystemResult(
                success=False,
                operation="search_issues",
                error_message="JQL query is required",
                error_code="MISSING_JQL"
            )
        
        # Get pagination parameters
        start_at = params.get("start_at", 0)
        max_results = params.get("max_results", 50)
        
        # Fields to include
        fields = params.get("fields", ["summary", "status", "assignee", "issuetype"])
        
        # Search issues
        url = f"{self.server_url}/rest/api/{self.api_version}/search"
        
        try:
            payload = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": max_results,
                "fields": fields
            }
            
            async with self.session.post(url, json=payload) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="search_issues",
                        result_data=data
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="search_issues",
                        error_message=f"Failed to search issues: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="search_issues",
                error_message=f"Exception searching issues: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_issue(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get a JIRA issue by key.
        
        Args:
            params: Parameters with issue_key
            
        Returns:
            ExternalSystemResult with issue data
        """
        issue_key = params.get("issue_key")
        
        if not issue_key:
            return ExternalSystemResult(
                success=False,
                operation="get_issue",
                error_message="Issue key is required",
                error_code="MISSING_ISSUE_KEY"
            )
        
        # Get fields to include
        fields = params.get("fields", [])
        expand = params.get("expand", [])
        
        # Build query parameters
        query_params = {}
        
        if fields:
            query_params["fields"] = ",".join(fields)
        
        if expand:
            query_params["expand"] = ",".join(expand)
        
        # Get issue
        url = f"{self.server_url}/rest/api/{self.api_version}/issue/{issue_key}"
        
        try:
            async with self.session.get(url, params=query_params) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status == 200:
                    data = await response.json()
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="get_issue",
                        result_data=data
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_issue",
                        error_message=f"Failed to get issue: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_issue",
                error_message=f"Exception getting issue: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _delete_issue(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Delete a JIRA issue.
        
        Args:
            params: Parameters with issue_key
            
        Returns:
            ExternalSystemResult with deletion result
        """
        issue_key = params.get("issue_key")
        
        if not issue_key:
            return ExternalSystemResult(
                success=False,
                operation="delete_issue",
                error_message="Issue key is required",
                error_code="MISSING_ISSUE_KEY"
            )
        
        # Delete issue
        url = f"{self.server_url}/rest/api/{self.api_version}/issue/{issue_key}"
        
        try:
            async with self.session.delete(url) as response:
                response = await self._handle_rate_limit(response)
                
                if response.status in [200, 204]:
                    return ExternalSystemResult(
                        success=True,
                        operation="delete_issue",
                        result_data={"issue_key": issue_key}
                    )
                else:
                    error_text = await response.text()
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="delete_issue",
                        error_message=f"Failed to delete issue: {response.status} - {error_text}",
                        error_code=f"HTTP_{response.status}"
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="delete_issue",
                error_message=f"Exception deleting issue: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query JIRA issues.
        
        Args:
            query_params: Query parameters
            
        Returns:
            List of query results
        """
        # Convert query_params to JQL
        jql_parts = []
        
        for key, value in query_params.items():
            # Skip special parameters
            if key in ["start_at", "max_results", "fields", "expand"]:
                continue
            
            # Handle different operators
            if isinstance(value, dict):
                operator = value.get("operator", "=")
                value = value.get("value")
            else:
                operator = "="
            
            # Format value based on type
            if isinstance(value, list):
                formatted_value = f"({','.join([f'"{v}"' if isinstance(v, str) else str(v) for v in value])})"
                if operator == "=":
                    operator = "IN"
                elif operator == "!=":
                    operator = "NOT IN"
            elif isinstance(value, str):
                formatted_value = f'"{value}"'
            else:
                formatted_value = str(value)
            
            jql_parts.append(f"{key} {operator} {formatted_value}")
        
        # Combine parts with AND
        jql = " AND ".join(jql_parts)
        
        # Add project filter if not explicitly provided and default project is set
        if "project" not in query_params and self.project_key:
            jql = f"project = {self.project_key}" + (" AND " + jql if jql else "")
        
        # Get pagination parameters
        start_at = query_params.get("start_at", 0)
        max_results = query_params.get("max_results", 50)
        
        # Fields to include
        fields = query_params.get("fields", ["key", "summary", "status", "assignee", "issuetype"])
        
        # Get search results
        search_result = await self._search_issues({
            "jql": jql,
            "start_at": start_at,
            "max_results": max_results,
            "fields": fields
        })
        
        if search_result.success:
            # Extract issues
            return search_result.result_data.get("issues", [])
        else:
            logger.error(f"Error querying JIRA: {search_result.error_message}")
            return []
    
    async def create_item(self, item_type: str, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an item in the JIRA system.
        
        Args:
            item_type: Type of item to create
            item_data: Item data
            
        Returns:
            Dictionary with created item details
        """
        if item_type == "issue":
            result = await self._create_issue(item_data)
        elif item_type == "comment":
            result = await self._add_comment(item_data)
        elif item_type == "attachment":
            result = await self._add_attachment(item_data)
        elif item_type == "worklog":
            result = await self._add_worklog(item_data)
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
        Update an item in the JIRA system.
        
        Args:
            item_type: Type of item to update
            item_id: ID of the item to update
            update_data: Data to update
            
        Returns:
            True if update succeeded
        """
        if item_type == "issue":
            result = await self._update_issue({"issue_key": item_id, **update_data})
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
        Delete an item from the JIRA system.
        
        Args:
            item_type: Type of item to delete
            item_id: ID of the item to delete
            
        Returns:
            True if deletion succeeded
        """
        if item_type == "issue":
            result = await self._delete_issue({"issue_key": item_id})
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
        Get an item from the JIRA system.
        
        Args:
            item_type: Type of item to get
            item_id: ID of the item to get
            
        Returns:
            Dictionary with item details
        """
        if item_type == "issue":
            result = await self._get_issue({"issue_key": item_id})
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
        Get information about the JIRA system.
        
        Returns:
            Dictionary with system information
        """
        await self._ensure_session()
        
        try:
            # Get server info
            url = f"{self.server_url}/rest/api/{self.api_version}/serverInfo"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    server_info = await response.json()
                else:
                    server_info = {"error": f"Failed to get server info: {response.status}"}
            
            # Get user info
            user_info = self.user_info or {"error": "Not connected"}
            
            # Build system info
            return {
                "system_type": "jira",
                "connected": self.user_info is not None,
                "server_url": self.server_url,
                "server_info": server_info,
                "api_version": self.api_version,
                "user_info": user_info,
                "project_key": self.project_key,
                "capabilities": self.capabilities.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Exception getting JIRA system info: {str(e)}")
            
            return {
                "system_type": "jira",
                "connected": False,
                "server_url": self.server_url,
                "error": str(e),
                "capabilities": self.capabilities.to_dict()
            }
    
    async def close(self) -> None:
        """
        Close the connection to JIRA and clean up resources.
        
        Returns:
            None
        """
        if self.session:
            await self.session.close()
            self.session = None
            self.user_info = None
            logger.info("JIRA connection closed")


# Register with factory
ExternalSystemFactory.register_connector("jira", JiraConnector)