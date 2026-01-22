#!/usr/bin/env python3
"""
Microsoft Teams Connector for Distributed Testing Framework

This module provides a connector for sending notifications to Microsoft Teams channels
through webhook URLs and the Microsoft Graph API if more detailed integrations are needed.
"""

import asyncio
import logging
import json
import os
import re
import aiohttp
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

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

class MSTeamsConnector(ExternalSystemInterface):
    """
    Connector for sending notifications to Microsoft Teams.
    
    This connector implements the standardized ExternalSystemInterface for Microsoft Teams
    integration and provides methods for sending messages with various card formats.
    """
    
    def __init__(self):
        """
        Initialize the MS Teams connector.
        """
        self.webhook_url = None
        self.use_graph_api = False
        self.graph_api_token = None
        self.graph_api_endpoint = None
        self.tenant_id = None
        self.client_id = None
        self.client_secret = None
        self.default_channel = None
        self.default_team_id = None
        self.rate_limit = 30  # Default: 30 messages per minute
        self.rate_limit_sleep = 2.0  # Seconds to sleep when rate limited
        self.sent_count = 0
        self.connected = False
        self.session = None
        self.token_expires_at = 0
        self.access_token = None
        
        # Cache for message templates
        self.templates = {}
        
        # Capabilities
        self.capabilities = ConnectorCapabilities(
            supports_create=True,
            supports_update=False,  # Messages can't be updated after sending with webhook
            supports_delete=False,  # Messages can't be deleted with webhook
            supports_query=False,   # Can't query sent messages with webhook
            supports_batch_operations=True,
            supports_attachments=False,  # No direct attachment support with webhooks
            supports_comments=False,
            supports_custom_fields=False,
            supports_relationships=False,
            supports_history=False,
            item_types=["message", "adaptive_card"],
            query_operators=[],
            max_batch_size=10,
            rate_limit=30,  # 30 messages per minute
            supports_adaptive_cards=True,     # Teams specific capability
            supports_hero_cards=True,         # Teams specific capability
            supports_list_cards=True,         # Teams specific capability
            supports_connector_cards=True,    # Teams specific capability
            supports_action_buttons=True,     # Teams specific capability
            supports_mentions=True            # Teams specific capability
        )
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the MS Teams connector with configuration.
        
        Args:
            config: Configuration dictionary containing:
                   - webhook_url: Teams webhook URL (for webhook integration)
                   - use_graph_api: Whether to use Microsoft Graph API (optional, default False)
                   - graph_api_token: Microsoft Graph API token (optional)
                   - tenant_id: Microsoft Teams tenant ID (optional, for Graph API)
                   - client_id: Microsoft app client ID (optional, for Graph API)
                   - client_secret: Microsoft app client secret (optional, for Graph API)
                   - graph_api_endpoint: Microsoft Graph API endpoint (optional)
                   - default_channel: Default channel name (optional, for Graph API)
                   - default_team_id: Default team ID (optional, for Graph API)
                   - templates_dir: Directory containing message templates (optional)
            
        Returns:
            True if initialization succeeded
        """
        self.webhook_url = config.get("webhook_url")
        self.use_graph_api = config.get("use_graph_api", False)
        self.graph_api_token = config.get("graph_api_token")
        self.tenant_id = config.get("tenant_id")
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")
        self.graph_api_endpoint = config.get("graph_api_endpoint", "https://graph.microsoft.com/v1.0")
        self.default_channel = config.get("default_channel")
        self.default_team_id = config.get("default_team_id")
        self.rate_limit = config.get("rate_limit", self.rate_limit)
        
        # Optional: load message templates
        templates_dir = config.get("templates_dir")
        if templates_dir and os.path.isdir(templates_dir):
            self._load_templates(templates_dir)
        
        # Determine the mode and required parameters
        if self.use_graph_api:
            # Graph API mode
            if not self.graph_api_token and not (self.tenant_id and self.client_id and self.client_secret):
                logger.error("When using Graph API, either graph_api_token or (tenant_id, client_id, client_secret) are required")
                return False
            
            if not self.default_team_id:
                logger.warning("No default_team_id provided for Graph API, some operations may not work")
                
        else:
            # Webhook mode
            if not self.webhook_url:
                logger.error("webhook_url is required when not using Graph API")
                return False
        
        logger.info(f"MSTeamsConnector initialized with {'Graph API' if self.use_graph_api else 'Webhook'} mode")
        return True
    
    def _load_templates(self, templates_dir: str) -> None:
        """
        Load message templates from a directory.
        
        Args:
            templates_dir: Directory containing template files
        """
        try:
            for filename in os.listdir(templates_dir):
                if filename.endswith(".json"):
                    template_name = os.path.splitext(filename)[0]
                    with open(os.path.join(templates_dir, filename), 'r') as f:
                        content = f.read()
                        try:
                            # Parse JSON to ensure it's valid
                            json.loads(content)
                            self.templates[template_name] = content
                            logger.debug(f"Loaded Teams template: {template_name}")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Error parsing template {template_name}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error loading Teams templates: {str(e)}")
    
    async def connect(self) -> bool:
        """
        Establish connection to Microsoft Teams API.
        
        Returns:
            True if connection succeeded
        """
        try:
            self.session = aiohttp.ClientSession()
            
            if self.use_graph_api and not self.graph_api_token:
                # Get an access token from Microsoft identity platform
                if not await self._get_access_token():
                    logger.error("Failed to obtain Graph API access token")
                    return False
            
            # Test the connection
            if self.use_graph_api:
                connected = await self._test_graph_api_connection()
            else:
                connected = await self._test_webhook_connection()
            
            self.connected = connected
            return connected
            
        except Exception as e:
            logger.error(f"Exception connecting to Microsoft Teams: {str(e)}")
            if self.session:
                await self.session.close()
                self.session = None
            self.connected = False
            return False
    
    async def _get_access_token(self) -> bool:
        """
        Get an access token for Microsoft Graph API.
        
        Returns:
            True if successful
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        now = datetime.now().timestamp()
        if self.access_token and now < self.token_expires_at - 300:  # 5 minute buffer
            return True
            
        try:
            token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
            
            data = {
                "client_id": self.client_id,
                "scope": "https://graph.microsoft.com/.default",
                "client_secret": self.client_secret,
                "grant_type": "client_credentials"
            }
            
            async with self.session.post(token_url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    self.access_token = result.get("access_token")
                    # Calculate expiry time (token expires_in is in seconds)
                    self.token_expires_at = now + result.get("expires_in", 3600)
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get access token. Status: {response.status}, Error: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Exception getting access token: {str(e)}")
            return False
    
    async def _test_webhook_connection(self) -> bool:
        """
        Test the webhook connection by sending a small test message.
        
        Returns:
            True if successful
        """
        try:
            # Create a minimal test message
            test_message = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "summary": "Connection Test",
                "themeColor": "0078D7",
                "text": "_This is a connection test from the Distributed Testing Framework._",
                "potentialAction": [
                    {
                        "@type": "ActionCard",
                        "name": "Connection test successful",
                        "inputs": []
                    }
                ]
            }
            
            # Send test message but hide it from channel using very small/invisible text
            # Use minimal color and no title to make it less intrusive
            low_visibility_test = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "summary": "Connection Test",
                "themeColor": "EEEEEE",
                "text": "<span style='font-size:1px;color:#FFFFFF;'>Connection test from DTF</span>"
            }
            
            # Send the minimal message
            async with self.session.post(self.webhook_url, json=low_visibility_test) as response:
                if response.status == 200:
                    logger.debug("Teams webhook connection test successful")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Teams webhook connection test failed. Status: {response.status}, Error: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Exception testing Teams webhook connection: {str(e)}")
            return False
    
    async def _test_graph_api_connection(self) -> bool:
        """
        Test the Graph API connection by getting user/team information.
        
        Returns:
            True if successful
        """
        try:
            if not self.access_token and not await self._get_access_token():
                return False
                
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            # Try to get information about teams - this is a good permission test
            endpoint = f"{self.graph_api_endpoint}/me/joinedTeams"
            
            async with self.session.get(endpoint, headers=headers) as response:
                if response.status == 200:
                    logger.debug("Teams Graph API connection test successful")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Teams Graph API connection test failed. Status: {response.status}, Error: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Exception testing Teams Graph API connection: {str(e)}")
            return False
            
    async def is_connected(self) -> bool:
        """
        Check if the connector is currently connected to Microsoft Teams.
        
        Returns:
            True if connected
        """
        if not self.session:
            return False
            
        # For Graph API, check token validity
        if self.use_graph_api:
            if datetime.now().timestamp() > self.token_expires_at:
                if not await self._get_access_token():
                    self.connected = False
                    return False
                    
        # For simplicity, just check if the session is open
        return not self.session.closed
    
    async def _handle_rate_limit(self):
        """Handle rate limiting to avoid exceeding messaging limits."""
        self.sent_count += 1
        if self.sent_count >= self.rate_limit:
            # Reset counter and wait before allowing more sends
            wait_time = self.rate_limit_sleep
            logger.info(f"Rate limit reached, pausing for {wait_time} seconds")
            await asyncio.sleep(wait_time)
            self.sent_count = 0
    
    async def _ensure_connection(self):
        """Ensure connection is established."""
        if not self.connected or not self.session or self.session.closed:
            await self.connect()
            
        # For Graph API, ensure token is valid
        if self.use_graph_api and datetime.now().timestamp() > self.token_expires_at - 300:
            await self._get_access_token()
    
    async def execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an operation for Microsoft Teams integration.
        
        Args:
            operation: The operation to execute
            params: Parameters for the operation
            
        Returns:
            Dictionary with operation result
        """
        await self._ensure_connection()
        
        try:
            result = ExternalSystemResult(
                success=False,
                operation=operation,
                error_message="Operation not implemented",
                error_code="NOT_IMPLEMENTED"
            )
            
            # Map operations to Teams functions
            if operation == "send_message":
                result = await self._send_message(params)
            elif operation == "send_adaptive_card":
                result = await self._send_adaptive_card(params)
            elif operation == "send_batch_messages":
                result = await self._send_batch_messages(params)
            elif operation == "send_template_message":
                result = await self._send_template_message(params)
            elif operation == "get_channels" and self.use_graph_api:
                result = await self._get_channels(params)
            elif operation == "get_team_members" and self.use_graph_api:
                result = await self._get_team_members(params)
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Exception executing Teams operation {operation}: {str(e)}")
            
            return ExternalSystemResult(
                success=False,
                operation=operation,
                error_message=str(e),
                error_code="EXCEPTION"
            ).to_dict()
    
    async def _send_message(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Send a message to Microsoft Teams.
        
        Args:
            params: Parameters for sending a message
            
        Returns:
            ExternalSystemResult with send operation result
        """
        # Handle rate limiting
        await self._handle_rate_limit()
        
        if self.use_graph_api:
            return await self._send_message_graph(params)
        else:
            return await self._send_message_webhook(params)
    
    async def _send_message_webhook(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Send a message using webhook.
        
        Args:
            params: Message parameters
            
        Returns:
            ExternalSystemResult with operation result
        """
        webhook_url = params.get("webhook_url", self.webhook_url)
        text = params.get("text", "")
        title = params.get("title")
        subtitle = params.get("subtitle")
        theme_color = params.get("theme_color", "0078D7")  # Default Teams blue
        sections = params.get("sections", [])
        facts = params.get("facts", [])
        actions = params.get("actions", [])
        images = params.get("images", [])
        
        if not webhook_url:
            return ExternalSystemResult(
                success=False,
                operation="send_message",
                error_message="Webhook URL is required",
                error_code="MISSING_WEBHOOK_URL"
            )
        
        if not text and not sections:
            return ExternalSystemResult(
                success=False,
                operation="send_message",
                error_message="Either text or sections are required",
                error_code="MISSING_CONTENT"
            )
        
        # Create the Teams message card
        message_card = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": theme_color,
            "text": text
        }
        
        if title:
            message_card["title"] = title
            
        if subtitle:
            message_card["subtitle"] = subtitle
        
        # Add sections if provided
        if sections:
            message_card["sections"] = sections
        elif facts:
            # Create a default section with facts
            message_card["sections"] = [{
                "facts": [{"name": k, "value": v} for k, v in facts]
            }]
            
        # Add images if provided
        if images and not sections:
            if "sections" not in message_card:
                message_card["sections"] = [{}]
            message_card["sections"][0]["images"] = images
            
        # Add actions if provided
        if actions:
            message_card["potentialAction"] = actions
        
        try:
            async with self.session.post(webhook_url, json=message_card) as response:
                if response.status == 200:
                    logger.info(f"Message sent to Teams webhook successfully")
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="send_message",
                        result_data={
                            "message_type": "card",
                            "title": title,
                            "webhook_url": webhook_url
                        }
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to send message to Teams webhook. Status: {response.status}, Error: {error_text}")
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="send_message",
                        error_message=f"Failed to send message. Status: {response.status}, Error: {error_text}",
                        error_code="SEND_ERROR"
                    )
                    
        except Exception as e:
            logger.error(f"Exception sending message to Teams webhook: {str(e)}")
            
            return ExternalSystemResult(
                success=False,
                operation="send_message",
                error_message=f"Exception sending message: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _send_message_graph(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Send a message using Microsoft Graph API.
        
        Args:
            params: Message parameters
            
        Returns:
            ExternalSystemResult with operation result
        """
        team_id = params.get("team_id", self.default_team_id)
        channel_id = params.get("channel_id")
        channel_name = params.get("channel_name", self.default_channel)
        content = params.get("content", "")
        subject = params.get("subject")
        
        if not team_id:
            return ExternalSystemResult(
                success=False,
                operation="send_message",
                error_message="Team ID is required",
                error_code="MISSING_TEAM_ID"
            )
        
        if not channel_id and not channel_name:
            return ExternalSystemResult(
                success=False,
                operation="send_message",
                error_message="Either channel_id or channel_name is required",
                error_code="MISSING_CHANNEL"
            )
        
        # If only channel name is provided, find the channel ID
        if not channel_id and channel_name:
            channel_result = await self._get_channel_id(team_id, channel_name)
            if not channel_result.success:
                return channel_result
            channel_id = channel_result.result_data.get("channel_id")
        
        # Ensure access token is valid
        if not await self._get_access_token():
            return ExternalSystemResult(
                success=False,
                operation="send_message",
                error_message="Failed to obtain access token",
                error_code="AUTH_ERROR"
            )
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        # Create message payload
        message = {
            "body": {
                "content": content,
                "contentType": "html"  # Use HTML for rich formatting
            }
        }
        
        if subject:
            message["subject"] = subject
        
        endpoint = f"{self.graph_api_endpoint}/teams/{team_id}/channels/{channel_id}/messages"
        
        try:
            async with self.session.post(endpoint, headers=headers, json=message) as response:
                if response.status in (200, 201):
                    result = await response.json()
                    logger.info(f"Message sent to Teams channel successfully")
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="send_message",
                        result_data={
                            "message_id": result.get("id"),
                            "team_id": team_id,
                            "channel_id": channel_id
                        }
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to send message via Graph API. Status: {response.status}, Error: {error_text}")
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="send_message",
                        error_message=f"Failed to send message. Status: {response.status}, Error: {error_text}",
                        error_code="SEND_ERROR"
                    )
                    
        except Exception as e:
            logger.error(f"Exception sending message via Graph API: {str(e)}")
            
            return ExternalSystemResult(
                success=False,
                operation="send_message",
                error_message=f"Exception sending message: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_channel_id(self, team_id: str, channel_name: str) -> ExternalSystemResult:
        """
        Get channel ID by name using Graph API.
        
        Args:
            team_id: Team ID
            channel_name: Channel name
            
        Returns:
            ExternalSystemResult with channel ID
        """
        if not await self._get_access_token():
            return ExternalSystemResult(
                success=False,
                operation="get_channel_id",
                error_message="Failed to obtain access token",
                error_code="AUTH_ERROR"
            )
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        endpoint = f"{self.graph_api_endpoint}/teams/{team_id}/channels"
        
        try:
            async with self.session.get(endpoint, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    channels = result.get("value", [])
                    
                    # Find the channel by name (case insensitive)
                    for channel in channels:
                        if channel.get("displayName", "").lower() == channel_name.lower():
                            return ExternalSystemResult(
                                success=True,
                                operation="get_channel_id",
                                result_data={
                                    "channel_id": channel.get("id"),
                                    "display_name": channel.get("displayName")
                                }
                            )
                    
                    # Channel not found
                    return ExternalSystemResult(
                        success=False,
                        operation="get_channel_id",
                        error_message=f"Channel not found: {channel_name}",
                        error_code="CHANNEL_NOT_FOUND"
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get channels. Status: {response.status}, Error: {error_text}")
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_channel_id",
                        error_message=f"Failed to get channels. Status: {response.status}, Error: {error_text}",
                        error_code="API_ERROR"
                    )
                    
        except Exception as e:
            logger.error(f"Exception getting channels: {str(e)}")
            
            return ExternalSystemResult(
                success=False,
                operation="get_channel_id",
                error_message=f"Exception getting channels: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _send_adaptive_card(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Send an adaptive card to Microsoft Teams.
        
        Args:
            params: Parameters for sending an adaptive card
            
        Returns:
            ExternalSystemResult with send operation result
        """
        webhook_url = params.get("webhook_url", self.webhook_url)
        card_content = params.get("card_content", {})
        
        if not webhook_url:
            return ExternalSystemResult(
                success=False,
                operation="send_adaptive_card",
                error_message="Webhook URL is required",
                error_code="MISSING_WEBHOOK_URL"
            )
        
        if not card_content:
            return ExternalSystemResult(
                success=False,
                operation="send_adaptive_card",
                error_message="Card content is required",
                error_code="MISSING_CARD_CONTENT"
            )
        
        # Handle rate limiting
        await self._handle_rate_limit()
        
        # Create the message with adaptive card
        message = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": card_content
                }
            ]
        }
        
        try:
            async with self.session.post(webhook_url, json=message) as response:
                if response.status == 200:
                    logger.info(f"Adaptive card sent to Teams webhook successfully")
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="send_adaptive_card",
                        result_data={
                            "message_type": "adaptive_card",
                            "webhook_url": webhook_url
                        }
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to send adaptive card to Teams webhook. Status: {response.status}, Error: {error_text}")
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="send_adaptive_card",
                        error_message=f"Failed to send adaptive card. Status: {response.status}, Error: {error_text}",
                        error_code="SEND_ERROR"
                    )
                    
        except Exception as e:
            logger.error(f"Exception sending adaptive card to Teams webhook: {str(e)}")
            
            return ExternalSystemResult(
                success=False,
                operation="send_adaptive_card",
                error_message=f"Exception sending adaptive card: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _send_batch_messages(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Send multiple messages in a batch.
        
        Args:
            params: Parameters for batch message sending
            
        Returns:
            ExternalSystemResult with batch operation result
        """
        messages = params.get("messages", [])
        
        if not messages:
            return ExternalSystemResult(
                success=False,
                operation="send_batch_messages",
                error_message="Messages array is required",
                error_code="MISSING_MESSAGES"
            )
        
        results = []
        success_count = 0
        failure_count = 0
        
        for message in messages:
            result = await self._send_message(message)
            results.append({
                "success": result.success,
                "title": message.get("title", ""),
                "error": None if result.success else result.error_message
            })
            
            if result.success:
                success_count += 1
            else:
                failure_count += 1
        
        return ExternalSystemResult(
            success=success_count > 0,
            operation="send_batch_messages",
            result_data={
                "total": len(messages),
                "success_count": success_count,
                "failure_count": failure_count,
                "results": results
            }
        )
    
    async def _send_template_message(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Send a message using a template.
        
        Args:
            params: Parameters for template-based message
            
        Returns:
            ExternalSystemResult with operation result
        """
        template_name = params.get("template_name")
        template_data = params.get("template_data", {})
        
        if not template_name:
            return ExternalSystemResult(
                success=False,
                operation="send_template_message",
                error_message="Template name is required",
                error_code="MISSING_TEMPLATE"
            )
        
        if template_name not in self.templates:
            return ExternalSystemResult(
                success=False,
                operation="send_template_message",
                error_message=f"Template not found: {template_name}",
                error_code="TEMPLATE_NOT_FOUND"
            )
        
        # Apply template variables
        template_content = self.templates[template_name]
        message = self._apply_template_variables(template_content, template_data)
        
        try:
            # Parse the processed template
            card_content = json.loads(message)
            
            # Check if this is an adaptive card or a message card
            if "type" in card_content and card_content["type"] == "AdaptiveCard":
                return await self._send_adaptive_card({
                    "webhook_url": params.get("webhook_url", self.webhook_url),
                    "card_content": card_content
                })
            else:
                # Assume it's a message card
                return await self._send_message({
                    "webhook_url": params.get("webhook_url", self.webhook_url),
                    "text": params.get("text", ""),
                    "title": params.get("title", ""),
                    "theme_color": params.get("theme_color", "0078D7"),
                    **card_content  # Merge with other message parameters
                })
                
        except json.JSONDecodeError as e:
            return ExternalSystemResult(
                success=False,
                operation="send_template_message",
                error_message=f"Error parsing template: {str(e)}",
                error_code="TEMPLATE_ERROR"
            )
    
    def _apply_template_variables(self, template: str, data: Dict[str, Any]) -> str:
        """
        Apply variable substitution to a template.
        
        Args:
            template: Template string
            data: Dictionary of variable values
            
        Returns:
            Template with variables replaced
        """
        for key, value in data.items():
            # Simple variable substitution using {{variable}} format
            template = template.replace(f"{{{{${key}}}}}", json.dumps(value) if isinstance(value, (dict, list)) else str(value))
            template = template.replace(f"{{{{${key}|}}}}", "null" if value is None else json.dumps(value) if isinstance(value, (dict, list)) else f'"{value}"')
            template = template.replace(f"{{{{${key}:s}}}}", str(value))
        return template
    
    async def _get_channels(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get channels for a team using Graph API.
        
        Args:
            params: Parameters with team ID
            
        Returns:
            ExternalSystemResult with channels list
        """
        team_id = params.get("team_id", self.default_team_id)
        
        if not team_id:
            return ExternalSystemResult(
                success=False,
                operation="get_channels",
                error_message="Team ID is required",
                error_code="MISSING_TEAM_ID"
            )
        
        if not await self._get_access_token():
            return ExternalSystemResult(
                success=False,
                operation="get_channels",
                error_message="Failed to obtain access token",
                error_code="AUTH_ERROR"
            )
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        endpoint = f"{self.graph_api_endpoint}/teams/{team_id}/channels"
        
        try:
            async with self.session.get(endpoint, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    channels = result.get("value", [])
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="get_channels",
                        result_data={
                            "team_id": team_id,
                            "channels": channels
                        }
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get channels. Status: {response.status}, Error: {error_text}")
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_channels",
                        error_message=f"Failed to get channels. Status: {response.status}, Error: {error_text}",
                        error_code="API_ERROR"
                    )
                    
        except Exception as e:
            logger.error(f"Exception getting channels: {str(e)}")
            
            return ExternalSystemResult(
                success=False,
                operation="get_channels",
                error_message=f"Exception getting channels: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def _get_team_members(self, params: Dict[str, Any]) -> ExternalSystemResult:
        """
        Get members of a team using Graph API.
        
        Args:
            params: Parameters with team ID
            
        Returns:
            ExternalSystemResult with members list
        """
        team_id = params.get("team_id", self.default_team_id)
        
        if not team_id:
            return ExternalSystemResult(
                success=False,
                operation="get_team_members",
                error_message="Team ID is required",
                error_code="MISSING_TEAM_ID"
            )
        
        if not await self._get_access_token():
            return ExternalSystemResult(
                success=False,
                operation="get_team_members",
                error_message="Failed to obtain access token",
                error_code="AUTH_ERROR"
            )
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        endpoint = f"{self.graph_api_endpoint}/teams/{team_id}/members"
        
        try:
            async with self.session.get(endpoint, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    members = result.get("value", [])
                    
                    return ExternalSystemResult(
                        success=True,
                        operation="get_team_members",
                        result_data={
                            "team_id": team_id,
                            "members": members
                        }
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get team members. Status: {response.status}, Error: {error_text}")
                    
                    return ExternalSystemResult(
                        success=False,
                        operation="get_team_members",
                        error_message=f"Failed to get team members. Status: {response.status}, Error: {error_text}",
                        error_code="API_ERROR"
                    )
                    
        except Exception as e:
            logger.error(f"Exception getting team members: {str(e)}")
            
            return ExternalSystemResult(
                success=False,
                operation="get_team_members",
                error_message=f"Exception getting team members: {str(e)}",
                error_code="EXCEPTION"
            )
    
    async def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query is not supported for webhook-based Microsoft Teams integration.
        
        Args:
            query_params: Query parameters
            
        Returns:
            Empty list (webhook connector doesn't support queries)
        """
        logger.warning("Query operation is not supported for Teams webhook connector")
        return []
    
    async def create_item(self, item_type: str, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an item in the Teams system (send a message).
        
        Args:
            item_type: Type of item to create (must be "message" or "adaptive_card")
            item_data: Item data
            
        Returns:
            Dictionary with created item details
        """
        if item_type not in ["message", "adaptive_card"]:
            raise Exception(f"Unsupported item type: {item_type}. Only 'message' and 'adaptive_card' are supported.")
        
        if item_type == "message":
            result = await self._send_message(item_data)
        else:  # adaptive_card
            result = await self._send_adaptive_card(item_data)
        
        if result.success:
            return result.result_data
        else:
            raise Exception(f"Failed to send {item_type}: {result.error_message}")
    
    async def update_item(self, item_type: str, item_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update is not supported for webhook-based Microsoft Teams integration.
        
        Args:
            item_type: Type of item to update
            item_id: ID of the item to update
            update_data: Data to update
            
        Returns:
            Always False (webhook connector doesn't support updates)
        """
        logger.warning("Update operation is not supported for Teams webhook connector")
        return False
    
    async def delete_item(self, item_type: str, item_id: str) -> bool:
        """
        Delete is not supported for webhook-based Microsoft Teams integration.
        
        Args:
            item_type: Type of item to delete
            item_id: ID of the item to delete
            
        Returns:
            Always False (webhook connector doesn't support deletion)
        """
        logger.warning("Delete operation is not supported for Teams webhook connector")
        return False
    
    async def get_item(self, item_type: str, item_id: str) -> Dict[str, Any]:
        """
        Get item is not supported for webhook-based Microsoft Teams integration.
        
        Args:
            item_type: Type of item to get
            item_id: ID of the item to get
            
        Returns:
            Empty dictionary (webhook connector doesn't support item retrieval)
        """
        logger.warning("Get item operation is not supported for Teams webhook connector")
        return {}
    
    async def system_info(self) -> Dict[str, Any]:
        """
        Get information about the Microsoft Teams integration.
        
        Returns:
            Dictionary with system information
        """
        try:
            is_connected = await self.is_connected()
            
            info = {
                "system_type": "msteams",
                "connected": is_connected,
                "integration_mode": "Graph API" if self.use_graph_api else "Webhook",
                "capabilities": self.capabilities.to_dict(),
                "templates_count": len(self.templates)
            }
            
            if self.use_graph_api:
                info.update({
                    "graph_api_endpoint": self.graph_api_endpoint,
                    "default_team_id": self.default_team_id,
                    "default_channel": self.default_channel,
                    "token_valid": datetime.now().timestamp() < self.token_expires_at
                })
            else:
                # Mask webhook URL for security
                if self.webhook_url:
                    parts = self.webhook_url.split("/")
                    if len(parts) > 4:
                        masked_parts = parts[0:3] + ["***"] + [parts[-1]]
                        masked_url = "/".join(masked_parts)
                        info["webhook_url_masked"] = masked_url
            
            return info
            
        except Exception as e:
            logger.error(f"Exception getting Teams system info: {str(e)}")
            
            return {
                "system_type": "msteams",
                "connected": False,
                "integration_mode": "Graph API" if self.use_graph_api else "Webhook",
                "error": str(e),
                "capabilities": self.capabilities.to_dict()
            }
    
    async def close(self) -> None:
        """
        Close the connection and clean up resources.
        
        Returns:
            None
        """
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.warning(f"Error closing session: {str(e)}")
            finally:
                self.session = None
                self.connected = False
                logger.info("Microsoft Teams connection closed")


# Register with factory
ExternalSystemFactory.register_connector("msteams", MSTeamsConnector)