#!/usr/bin/env python3
"""
Slack Connector for Distributed Testing Framework

This module provides a connector for sending notifications to Slack using the Slack Web API.
"""

import anyio
import json
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

class SlackConnector(ExternalSystemInterface):
    """
    Connector for sending notifications to Slack.
    
    This connector implements the standardized ExternalSystemInterface for Slack
    and provides methods for sending messages, file uploads, and other Slack operations.
    """
    
    def __init__(self):
        """
        Initialize the Slack connector.
        """
        self.token = None
        self.default_channel = None
        self.session = None
        self.bot_info = None
        self.rate_limit_remaining = 1000
        self.rate_limit_reset = 0
        self.rate_limit_sleep = 1.0  # seconds to sleep when rate limited
        
        # Capabilities
        self.capabilities = ConnectorCapabilities(
            supports_create=True,
            supports_update=True,
            supports_delete=True,
            supports_query=True,
            supports_batch_operations=False,
            supports_attachments=True,
            supports_comments=False,
            supports_custom_fields=False,
            supports_relationships=False,
            supports_history=True,
            item_types=["message", "file", "thread", "channel"],
            query_operators=["=", "!=", "IN", "NOT IN"],
            max_batch_size=0,
            rate_limit=100,  # Default Slack tier rate limit (100 requests per minute for most methods)
            supports_reactions=True,
            supports_rich_formatting=True,
            supports_threads=True,
            supports_scheduled_messages=True
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the Slack connector with configuration.
        
        Args:
            config: Configuration dictionary containing:
                   - token: Slack API token (bot or user token)
                   - default_channel: Default channel to send messages to
            
        Returns:
            True if initialization succeeded
        """
        self.token = config.get("token")
        self.default_channel = config.get("default_channel")
        
        if not self.token:
            logger.error("Slack API token is required")
            return False
        
        logger.info("SlackConnector initialized")
        return True
    
    async def _ensure_session(self):
        """Ensure an aiohttp session exists with proper headers."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json; charset=utf-8",
                    "User-Agent": "DistributedTestingFramework/1.0"
                }
            )
    
    async def connect(self) -> bool:
        """
        Establish connection to Slack and validate token.
        
        Returns:
            True if connection succeeded
        """
        await self._ensure_session()
        
        try:
            # Call auth.test to validate token
            url = "https://slack.com/api/auth.test"
            
            async with self.session.post(url) as response:
                data = await response.json()
                
                if response.status == 200 and data.get("ok"):
                    # Store bot info
                    self.bot_info = data
                    logger.info(f"Connected to Slack as {data.get('user')} (team: {data.get('team')})")
                    
                    # Check rate limits from response headers
                    self._update_rate_limits(response)
                    
                    return True
                else:
                    logger.error(f"Failed to connect to Slack: {data.get('error', 'Unknown error')}")
                    return False
                
        except Exception as e:
            logger.error(f"Exception connecting to Slack: {str(e)}")
            return False
    
    async def is_connected(self) -> bool:
        """
        Check if the connector is currently connected to Slack.
        
        Returns:
            True if connected
        """
        if self.session is None or self.bot_info is None:
            return False
        
        try:
            # Lightweight check to see if token is still valid
            url = "https://slack.com/api/auth.test"
            
            async with self.session.post(url) as response:
                data = await response.json()
                
                # Update rate limits
                self._update_rate_limits(response)
                
                return response.status == 200 and data.get("ok", False)
                
        except Exception:
            return False
    
    def _update_rate_limits(self, response):
        """
        Update rate limit information from response headers.
        
        Args:
            response: The aiohttp response object
        """
        # Get rate limit headers
        remaining = response.headers.get("X-Rate-Limit-Remaining")
        reset = response.headers.get("X-Rate-Limit-Reset")
        
        if remaining is not None:
            try:
                self.rate_limit_remaining = int(remaining)
            except ValueError:
                pass
        
        if reset is not None:
            try:
                self.rate_limit_reset = int(reset)
            except ValueError:
                pass
    
    async def _handle_rate_limit(self, response):
        """
        Handle rate limiting by pausing when necessary.
        
        Args:
            response: The aiohttp response object
            
        Returns:
            The json data from the response
        """
        # Update rate limits
        self._update_rate_limits(response)
        
        # Get response data
        data = await response.json()
        
        # Check for rate limiting
        if data.get("error") == "ratelimited":
            # Calculate how long to wait
            retry_after = data.get("retry_after")
            
            if retry_after:
                seconds = float(retry_after)
            elif self.rate_limit_reset > 0:
                seconds = max(0, self.rate_limit_reset - time.time())
            else:
                seconds = self.rate_limit_sleep
                
            logger.warning(f"Slack rate limit hit, pausing for {seconds:.2f} seconds")
            await anyio.sleep(seconds)
        
        return data
    
    async def execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an operation on Slack.
        
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
            
            # Map common operations to Slack API calls
            if operation == "send_message":
                result = await self._send_message(params)
            elif operation == "update_message":
                result = await self._update_message(params)
            elif operation == "delete_message":
                result = await self._delete_message(params)
            elif operation == "upload_file":
                result = await self._upload_file(params)
            elif operation == "add_reaction":
                result = await self._add_reaction(params)
            elif operation == "create_channel":
                result = await self._create_channel(params)
            elif operation == "archive_channel":
                result = await self._archive_channel(params)
            elif operation == "get_channel_info":
                result = await self._get_channel_info(params)
            elif operation == "get_message_history":
                result = await self._get_message_history(params)
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Exception executing Slack operation {operation}: {str(e)}")
            
            return ExternalSystemResult(
                success=False,
                operation=operation,
                error_message=str(e),
                error_code="EXCEPTION"
            ).to_dict()
    
    async def _send_message(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Send a message to a Slack channel.
        
        Args:
            params: Message parameters
            
        Returns:
            ExternalSystemResult with message details
        """
        channel = params.get("channel", self.default_channel)
        
        if not channel:
            return ExternalSystemResult(
                success=False,
                operation="send_message",
                error_message="Channel is required",
                error_code="MISSING_CHANNEL"
            )
        
        text = params.get("text")
        blocks = params.get("blocks")
        
        if not text and not blocks:
            return ExternalSystemResult(
                success=False,
                operation="send_message",
                error_message="Either text or blocks is required",
                error_code="MISSING_CONTENT"
            )
        
        # Send message
        url = "https://slack.com/api/chat.postMessage"
        
        try:
            payload = {
                "channel": channel
            }
            
            # Add text if provided
            if text:
                payload["text"] = text
            
            # Add blocks if provided
            if blocks:
                payload["blocks"] = blocks
            
            # Add thread_ts if provided
            if "thread_ts" in params:
                payload["thread_ts"] = params["thread_ts"]
            
            # Add other parameters
            for key in ["as_user", "icon_emoji", "icon_url", "username", "unfurl_links", "unfurl_media"]:
                if key in params:
                    payload[key] = params[key]
            
            async with self.session.post(url, json=payload) as response:
                data = await self._handle_rate_limit(response)
                
                if data.get("ok"):
                    return ExternalSystemResult(
                        success=True,
                        operation="send_message",
                        result_data={
                            "channel": data.get("channel"),
                            "ts": data.get("ts"),
                            "message": data.get("message")
                        }
                    )
                else:
                    return ExternalSystemResult(
                        success=False,
                        operation="send_message",
                        error_message=f"Failed to send message: {data.get('error')}",
                        error_code=data.get("error", "UNKNOWN_ERROR")
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="send_message",
                error_message=f"Exception sending message: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _update_message(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Update a message in a Slack channel.
        
        Args:
            params: Message update parameters
            
        Returns:
            ExternalSystemResult with update result
        """
        channel = params.get("channel", self.default_channel)
        
        if not channel:
            return ExternalSystemResult(
                success=False,
                operation="update_message",
                error_message="Channel is required",
                error_code="MISSING_CHANNEL"
            )
        
        ts = params.get("ts")
        
        if not ts:
            return ExternalSystemResult(
                success=False,
                operation="update_message",
                error_message="Message timestamp is required",
                error_code="MISSING_TS"
            )
        
        text = params.get("text")
        blocks = params.get("blocks")
        
        if not text and not blocks:
            return ExternalSystemResult(
                success=False,
                operation="update_message",
                error_message="Either text or blocks is required",
                error_code="MISSING_CONTENT"
            )
        
        # Update message
        url = "https://slack.com/api/chat.update"
        
        try:
            payload = {
                "channel": channel,
                "ts": ts
            }
            
            # Add text if provided
            if text:
                payload["text"] = text
            
            # Add blocks if provided
            if blocks:
                payload["blocks"] = blocks
            
            # Add as_user if provided
            if "as_user" in params:
                payload["as_user"] = params["as_user"]
            
            async with self.session.post(url, json=payload) as response:
                data = await self._handle_rate_limit(response)
                
                if data.get("ok"):
                    return ExternalSystemResult(
                        success=True,
                        operation="update_message",
                        result_data={
                            "channel": data.get("channel"),
                            "ts": data.get("ts"),
                            "text": text
                        }
                    )
                else:
                    return ExternalSystemResult(
                        success=False,
                        operation="update_message",
                        error_message=f"Failed to update message: {data.get('error')}",
                        error_code=data.get("error", "UNKNOWN_ERROR")
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="update_message",
                error_message=f"Exception updating message: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _delete_message(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Delete a message from a Slack channel.
        
        Args:
            params: Message deletion parameters
            
        Returns:
            ExternalSystemResult with deletion result
        """
        channel = params.get("channel", self.default_channel)
        
        if not channel:
            return ExternalSystemResult(
                success=False,
                operation="delete_message",
                error_message="Channel is required",
                error_code="MISSING_CHANNEL"
            )
        
        ts = params.get("ts")
        
        if not ts:
            return ExternalSystemResult(
                success=False,
                operation="delete_message",
                error_message="Message timestamp is required",
                error_code="MISSING_TS"
            )
        
        # Delete message
        url = "https://slack.com/api/chat.delete"
        
        try:
            payload = {
                "channel": channel,
                "ts": ts
            }
            
            # Add as_user if provided
            if "as_user" in params:
                payload["as_user"] = params["as_user"]
            
            async with self.session.post(url, json=payload) as response:
                data = await self._handle_rate_limit(response)
                
                if data.get("ok"):
                    return ExternalSystemResult(
                        success=True,
                        operation="delete_message",
                        result_data={
                            "channel": data.get("channel"),
                            "ts": ts
                        }
                    )
                else:
                    return ExternalSystemResult(
                        success=False,
                        operation="delete_message",
                        error_message=f"Failed to delete message: {data.get('error')}",
                        error_code=data.get("error", "UNKNOWN_ERROR")
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="delete_message",
                error_message=f"Exception deleting message: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _upload_file(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Upload a file to Slack.
        
        Args:
            params: File upload parameters
            
        Returns:
            ExternalSystemResult with file details
        """
        channel = params.get("channel", self.default_channel)
        
        if not channel:
            return ExternalSystemResult(
                success=False,
                operation="upload_file",
                error_message="Channel is required",
                error_code="MISSING_CHANNEL"
            )
        
        file_path = params.get("file_path")
        content = params.get("content")
        
        if not file_path and not content:
            return ExternalSystemResult(
                success=False,
                operation="upload_file",
                error_message="Either file_path or content is required",
                error_code="MISSING_FILE_DATA"
            )
        
        # Upload file
        url = "https://slack.com/api/files.upload"
        
        try:
            # Create form data
            form = aiohttp.FormData()
            form.add_field("channels", channel)
            
            # Add file from path or content
            if file_path:
                with open(file_path, "rb") as f:
                    file_data = f.read()
                
                filename = params.get("filename", file_path.split("/")[-1])
                form.add_field("file", file_data, filename=filename)
            else:
                filename = params.get("filename", "file.txt")
                form.add_field("content", content)
                form.add_field("filename", filename)
            
            # Add optional fields
            for key in ["title", "initial_comment", "thread_ts"]:
                if key in params:
                    form.add_field(key, params[key])
            
            async with self.session.post(url, data=form) as response:
                data = await self._handle_rate_limit(response)
                
                if data.get("ok"):
                    file_data = data.get("file", {})
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="upload_file",
                        result_data={
                            "id": file_data.get("id"),
                            "name": file_data.get("name"),
                            "title": file_data.get("title"),
                            "mimetype": file_data.get("mimetype"),
                            "filetype": file_data.get("filetype"),
                            "size": file_data.get("size"),
                            "url_private": file_data.get("url_private"),
                            "permalink": file_data.get("permalink")
                        }
                    )
                else:
                    return ExternalSystemResult(
                        success=False,
                        operation="upload_file",
                        error_message=f"Failed to upload file: {data.get('error')}",
                        error_code=data.get("error", "UNKNOWN_ERROR")
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="upload_file",
                error_message=f"Exception uploading file: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _add_reaction(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Add a reaction to a message.
        
        Args:
            params: Reaction parameters
            
        Returns:
            ExternalSystemResult with reaction result
        """
        channel = params.get("channel", self.default_channel)
        
        if not channel:
            return ExternalSystemResult(
                success=False,
                operation="add_reaction",
                error_message="Channel is required",
                error_code="MISSING_CHANNEL"
            )
        
        ts = params.get("ts")
        
        if not ts:
            return ExternalSystemResult(
                success=False,
                operation="add_reaction",
                error_message="Message timestamp is required",
                error_code="MISSING_TS"
            )
        
        name = params.get("name")
        
        if not name:
            return ExternalSystemResult(
                success=False,
                operation="add_reaction",
                error_message="Reaction name is required",
                error_code="MISSING_NAME"
            )
        
        # Add reaction
        url = "https://slack.com/api/reactions.add"
        
        try:
            payload = {
                "channel": channel,
                "timestamp": ts,
                "name": name
            }
            
            async with self.session.post(url, json=payload) as response:
                data = await self._handle_rate_limit(response)
                
                if data.get("ok"):
                    return ExternalSystemResult(
                        success=True,
                        operation="add_reaction",
                        result_data={
                            "channel": channel,
                            "ts": ts,
                            "name": name
                        }
                    )
                else:
                    return ExternalSystemResult(
                        success=False,
                        operation="add_reaction",
                        error_message=f"Failed to add reaction: {data.get('error')}",
                        error_code=data.get("error", "UNKNOWN_ERROR")
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="add_reaction",
                error_message=f"Exception adding reaction: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _create_channel(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Create a Slack channel.
        
        Args:
            params: Channel creation parameters
            
        Returns:
            ExternalSystemResult with channel details
        """
        name = params.get("name")
        
        if not name:
            return ExternalSystemResult(
                success=False,
                operation="create_channel",
                error_message="Channel name is required",
                error_code="MISSING_NAME"
            )
        
        # Create channel
        url = "https://slack.com/api/conversations.create"
        
        try:
            payload = {
                "name": name
            }
            
            # Add is_private if provided
            if "is_private" in params:
                payload["is_private"] = params["is_private"]
            
            async with self.session.post(url, json=payload) as response:
                data = await self._handle_rate_limit(response)
                
                if data.get("ok"):
                    channel = data.get("channel", {})
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="create_channel",
                        result_data={
                            "id": channel.get("id"),
                            "name": channel.get("name"),
                            "is_channel": channel.get("is_channel"),
                            "is_private": channel.get("is_private"),
                            "created": channel.get("created")
                        }
                    )
                else:
                    return ExternalSystemResult(
                        success=False,
                        operation="create_channel",
                        error_message=f"Failed to create channel: {data.get('error')}",
                        error_code=data.get("error", "UNKNOWN_ERROR")
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="create_channel",
                error_message=f"Exception creating channel: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _archive_channel(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Archive a Slack channel.
        
        Args:
            params: Channel archiving parameters
            
        Returns:
            ExternalSystemResult with archiving result
        """
        channel = params.get("channel")
        
        if not channel:
            return ExternalSystemResult(
                success=False,
                operation="archive_channel",
                error_message="Channel is required",
                error_code="MISSING_CHANNEL"
            )
        
        # Archive channel
        url = "https://slack.com/api/conversations.archive"
        
        try:
            payload = {
                "channel": channel
            }
            
            async with self.session.post(url, json=payload) as response:
                data = await self._handle_rate_limit(response)
                
                if data.get("ok"):
                    return ExternalSystemResult(
                        success=True,
                        operation="archive_channel",
                        result_data={
                            "channel": channel
                        }
                    )
                else:
                    return ExternalSystemResult(
                        success=False,
                        operation="archive_channel",
                        error_message=f"Failed to archive channel: {data.get('error')}",
                        error_code=data.get("error", "UNKNOWN_ERROR")
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="archive_channel",
                error_message=f"Exception archiving channel: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_channel_info(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get information about a Slack channel.
        
        Args:
            params: Channel info parameters
            
        Returns:
            ExternalSystemResult with channel info
        """
        channel = params.get("channel")
        
        if not channel:
            return ExternalSystemResult(
                success=False,
                operation="get_channel_info",
                error_message="Channel is required",
                error_code="MISSING_CHANNEL"
            )
        
        # Get channel info
        url = "https://slack.com/api/conversations.info"
        
        try:
            async with self.session.get(url, params={"channel": channel}) as response:
                data = await self._handle_rate_limit(response)
                
                if data.get("ok"):
                    channel_info = data.get("channel", {})
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="get_channel_info",
                        result_data=channel_info
                    )
                else:
                    return ExternalSystemResult(
                        success=False,
                        operation="get_channel_info",
                        error_message=f"Failed to get channel info: {data.get('error')}",
                        error_code=data.get("error", "UNKNOWN_ERROR")
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_channel_info",
                error_message=f"Exception getting channel info: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_message_history(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get message history from a Slack channel.
        
        Args:
            params: Message history parameters
            
        Returns:
            ExternalSystemResult with message history
        """
        channel = params.get("channel", self.default_channel)
        
        if not channel:
            return ExternalSystemResult(
                success=False,
                operation="get_message_history",
                error_message="Channel is required",
                error_code="MISSING_CHANNEL"
            )
        
        # Get message history
        url = "https://slack.com/api/conversations.history"
        
        try:
            # Build query parameters
            query_params = {
                "channel": channel
            }
            
            # Add optional parameters
            for key in ["latest", "oldest", "inclusive", "limit", "cursor"]:
                if key in params:
                    query_params[key] = params[key]
            
            async with self.session.get(url, params=query_params) as response:
                data = await self._handle_rate_limit(response)
                
                if data.get("ok"):
                    return ExternalSystemResult(
                        success=True,
                        operation="get_message_history",
                        result_data=data
                    )
                else:
                    return ExternalSystemResult(
                        success=False,
                        operation="get_message_history",
                        error_message=f"Failed to get message history: {data.get('error')}",
                        error_code=data.get("error", "UNKNOWN_ERROR")
                    )
                    
        except Exception as e:
            return ExternalSystemResult(
                success=False,
                operation="get_message_history",
                error_message=f"Exception getting message history: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query Slack for data.
        
        Args:
            query_params: Query parameters
            
        Returns:
            List of query results
        """
        item_type = query_params.get("item_type", "message")
        
        if item_type == "message":
            # Query messages
            channel = query_params.get("channel", self.default_channel)
            
            if not channel:
                logger.error("Channel is required for message query")
                return []
            
            # Extract history parameters
            history_params = {
                "channel": channel
            }
            
            for key in ["latest", "oldest", "inclusive", "limit", "cursor"]:
                if key in query_params:
                    history_params[key] = query_params[key]
            
            # Get message history
            result = await self._get_message_history(history_params)
            
            if result.success:
                # Return messages
                return result.result_data.get("messages", [])
            else:
                logger.error(f"Error querying Slack messages: {result.error_message}")
                return []
                
        elif item_type == "channel":
            # Query channels
            url = "https://slack.com/api/conversations.list"
            
            try:
                # Build query parameters
                params = {}
                
                # Add optional parameters
                for key in ["types", "exclude_archived", "limit", "cursor"]:
                    if key in query_params:
                        params[key] = query_params[key]
                
                async with self.session.get(url, params=params) as response:
                    data = await self._handle_rate_limit(response)
                    
                    if data.get("ok"):
                        # Return channels
                        return data.get("channels", [])
                    else:
                        logger.error(f"Error listing channels: {data.get('error')}")
                        return []
                        
            except Exception as e:
                logger.error(f"Exception listing channels: {str(e)}")
                return []
                
        else:
            logger.error(f"Unsupported item type for query: {item_type}")
            return []
    
    async def create_item(self, item_type: str, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an item in Slack.
        
        Args:
            item_type: Type of item to create
            item_data: Item data
            
        Returns:
            Dictionary with created item details
        """
        if item_type == "message":
            result = await self._send_message(item_data)
        elif item_type == "file":
            result = await self._upload_file(item_data)
        elif item_type == "channel":
            result = await self._create_channel(item_data)
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
        Update an item in Slack.
        
        Args:
            item_type: Type of item to update
            item_id: ID of the item to update
            update_data: Data to update
            
        Returns:
            True if update succeeded
        """
        if item_type == "message":
            # For messages, item_id is the timestamp
            # Add the timestamp to the update data
            result = await self._update_message({**update_data, "ts": item_id})
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
        Delete an item from Slack.
        
        Args:
            item_type: Type of item to delete
            item_id: ID of the item to delete
            
        Returns:
            True if deletion succeeded
        """
        if item_type == "message":
            # For messages, item_id is the timestamp
            # Need to get the channel from the delete_data
            channel = self.default_channel
            result = await self._delete_message({"channel": channel, "ts": item_id})
        elif item_type == "channel":
            result = await self._archive_channel({"channel": item_id})
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
        Get an item from Slack.
        
        Args:
            item_type: Type of item to get
            item_id: ID of the item to get
            
        Returns:
            Dictionary with item details
        """
        if item_type == "channel":
            result = await self._get_channel_info({"channel": item_id})
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
        Get information about the Slack system.
        
        Returns:
            Dictionary with system information
        """
        await self._ensure_session()
        
        try:
            # Get team info
            url = "https://slack.com/api/team.info"
            
            async with self.session.get(url) as response:
                team_data = await self._handle_rate_limit(response)
                
                if not team_data.get("ok"):
                    team_data = {"error": team_data.get("error", "Unknown error")}
            
            # Get bot info
            bot_info = self.bot_info or {"error": "Not connected"}
            
            # Build system info
            return {
                "system_type": "slack",
                "connected": self.bot_info is not None,
                "team_info": team_data.get("team", {}),
                "bot_info": bot_info,
                "default_channel": self.default_channel,
                "capabilities": self.capabilities.to_dict(),
                "rate_limit_remaining": self.rate_limit_remaining,
                "rate_limit_reset": self.rate_limit_reset
            }
            
        except Exception as e:
            logger.error(f"Exception getting Slack system info: {str(e)}")
            
            return {
                "system_type": "slack",
                "connected": False,
                "error": str(e),
                "capabilities": self.capabilities.to_dict()
            }
    
    async def close(self) -> None:
        """
        Close the connection to Slack and clean up resources.
        
        Returns:
            None
        """
        if self.session:
            await self.session.close()
            self.session = None
            self.bot_info = None
            logger.info("Slack connection closed")


# Register with factory
ExternalSystemFactory.register_connector("slack", SlackConnector)