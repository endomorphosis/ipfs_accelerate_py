#!/usr/bin/env python3
"""
Discord Connector for External Systems Integration

This module implements a Discord connector for the External Systems Integration API,
allowing the distributed testing framework to send messages to Discord channels
via webhooks or the Discord Bot API.
"""

import aiohttp
import logging
import json
from typing import Dict, List, Any, Optional

from .api_interface import (
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

class DiscordConnector(ExternalSystemInterface):
    """
    Discord connector implementing the External System Interface.
    
    This connector allows sending messages to Discord channels via webhooks
    or the Discord Bot API.
    """
    
    def __init__(self):
        """Initialize the Discord connector."""
        self.initialized = False
        self.connected = False
        self.config = {}
        self.session = None
        self.webhook_url = None
        self.bot_token = None
        self.use_bot_api = False
        self.default_channel_id = None
        self.username = "Distributed Testing Framework"
        self.avatar_url = None
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the Discord connector with configuration.
        
        Args:
            config: Configuration dictionary containing Discord-specific settings
            
        Returns:
            True if initialization succeeded
        """
        self.config = config
        
        # Extract configuration
        self.webhook_url = config.get("webhook_url")
        self.bot_token = config.get("bot_token")
        self.use_bot_api = config.get("use_bot_api", False)
        self.default_channel_id = config.get("default_channel_id")
        self.username = config.get("username", "Distributed Testing Framework")
        self.avatar_url = config.get("avatar_url")
        
        # Validate configuration
        if not self.webhook_url and not self.bot_token:
            logger.error("Either webhook_url or bot_token must be provided")
            return False
            
        if self.use_bot_api and not self.bot_token:
            logger.error("bot_token is required when use_bot_api is true")
            return False
            
        if self.use_bot_api and not self.default_channel_id:
            logger.error("default_channel_id is required when use_bot_api is true")
            return False
        
        self.initialized = True
        logger.info("Discord connector initialized")
        return True
    
    async def connect(self) -> bool:
        """
        Establish connection to Discord.
        
        Returns:
            True if connection succeeded
        """
        if not self.initialized:
            logger.error("Discord connector not initialized")
            return False
        
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Test connection
            if self.use_bot_api:
                # Test Bot API connection
                async with self.session.get(
                    "https://discord.com/api/v10/users/@me",
                    headers={"Authorization": f"Bot {self.bot_token}"}
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to connect to Discord Bot API: {response.status}")
                        await self.session.close()
                        self.session = None
                        return False
            else:
                # Test Webhook connection (just check if URL is valid)
                if not self.webhook_url.startswith("https://discord.com/api/webhooks/"):
                    logger.error("Invalid Discord webhook URL")
                    await self.session.close()
                    self.session = None
                    return False
            
            self.connected = True
            logger.info("Connected to Discord")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Discord: {str(e)}")
            
            if self.session:
                await self.session.close()
                self.session = None
                
            return False
    
    async def is_connected(self) -> bool:
        """
        Check if the connector is currently connected to Discord.
        
        Returns:
            True if connected
        """
        return self.connected and self.session is not None
    
    async def execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an operation on Discord.
        
        Args:
            operation: The operation to execute
            params: Parameters for the operation
            
        Returns:
            Dictionary with operation result
        """
        if not await self.is_connected():
            logger.error("Not connected to Discord")
            return {"success": False, "error": "Not connected to Discord"}
        
        if operation == "send_message":
            return await self._send_message(params)
        elif operation == "send_embed":
            return await self._send_embed(params)
        else:
            logger.error(f"Unsupported operation: {operation}")
            return {"success": False, "error": f"Unsupported operation: {operation}"}
    
    async def _send_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to a Discord channel.
        
        Args:
            params: Parameters for the message
            
        Returns:
            Dictionary with operation result
        """
        channel_id = params.get("channel_id", self.default_channel_id)
        content = params.get("content", "")
        embeds = params.get("embeds", [])
        
        if not content and not embeds:
            logger.error("Either content or embeds must be provided")
            return {"success": False, "error": "Either content or embeds must be provided"}
        
        try:
            if self.use_bot_api:
                # Send via Bot API
                url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
                headers = {
                    "Authorization": f"Bot {self.bot_token}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "content": content,
                }
                
                if embeds:
                    payload["embeds"] = embeds
                
                async with self.session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Failed to send message via Bot API: {response.status} - {error_text}")
                        return {"success": False, "error": f"Failed to send message: {response.status} - {error_text}"}
                    
                    result = await response.json()
                    return {
                        "success": True,
                        "message_id": result.get("id"),
                        "timestamp": result.get("timestamp")
                    }
            else:
                # Send via Webhook
                payload = {}
                
                if content:
                    payload["content"] = content
                    
                if embeds:
                    payload["embeds"] = embeds
                    
                if self.username:
                    payload["username"] = self.username
                    
                if self.avatar_url:
                    payload["avatar_url"] = self.avatar_url
                
                async with self.session.post(self.webhook_url, json=payload) as response:
                    if response.status not in (200, 204):
                        error_text = await response.text()
                        logger.error(f"Failed to send message via webhook: {response.status} - {error_text}")
                        return {"success": False, "error": f"Failed to send message: {response.status} - {error_text}"}
                    
                    return {
                        "success": True
                    }
                    
        except Exception as e:
            logger.error(f"Error sending Discord message: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _send_embed(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an embed to a Discord channel.
        
        Args:
            params: Parameters for the embed
            
        Returns:
            Dictionary with operation result
        """
        channel_id = params.get("channel_id", self.default_channel_id)
        title = params.get("title", "")
        description = params.get("description", "")
        color = params.get("color", 0x3498db)  # Default blue
        fields = params.get("fields", [])
        footer = params.get("footer")
        thumbnail = params.get("thumbnail")
        image = params.get("image")
        
        # Create embed
        embed = {
            "title": title,
            "description": description,
            "color": color,
        }
        
        if fields:
            embed["fields"] = fields
            
        if footer:
            embed["footer"] = footer
            
        if thumbnail:
            embed["thumbnail"] = {"url": thumbnail}
            
        if image:
            embed["image"] = {"url": image}
        
        # Send as embed
        return await self._send_message({
            "channel_id": channel_id,
            "embeds": [embed]
        })
    
    async def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query Discord for data.
        
        Args:
            query_params: Query parameters
            
        Returns:
            List of query results
        """
        # Discord connector does not support querying
        logger.error("Discord connector does not support querying")
        return []
    
    async def create_item(self, item_type: str, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an item in Discord.
        
        Args:
            item_type: Type of item to create
            item_data: Item data
            
        Returns:
            Dictionary with created item details
        """
        if item_type == "message":
            return await self._send_message(item_data)
        elif item_type == "embed":
            return await self._send_embed(item_data)
        else:
            logger.error(f"Unsupported item type: {item_type}")
            return {"success": False, "error": f"Unsupported item type: {item_type}"}
    
    async def update_item(self, item_type: str, item_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an item in Discord.
        
        Args:
            item_type: Type of item to update
            item_id: ID of the item to update
            update_data: Data to update
            
        Returns:
            True if update succeeded
        """
        # Discord connector doesn't support updating items
        logger.error("Discord connector does not support updating items")
        return False
    
    async def delete_item(self, item_type: str, item_id: str) -> bool:
        """
        Delete an item from Discord.
        
        Args:
            item_type: Type of item to delete
            item_id: ID of the item to delete
            
        Returns:
            True if deletion succeeded
        """
        if not await self.is_connected():
            logger.error("Not connected to Discord")
            return False
            
        if not self.use_bot_api:
            logger.error("Deleting messages requires Bot API")
            return False
            
        if item_type != "message":
            logger.error(f"Unsupported item type: {item_type}")
            return False
            
        channel_id = self.default_channel_id
        
        # Extract channel ID if item_id contains it
        if ":" in item_id:
            parts = item_id.split(":")
            if len(parts) == 2:
                channel_id, message_id = parts
            else:
                message_id = item_id
        else:
            message_id = item_id
        
        try:
            # Delete message via Bot API
            url = f"https://discord.com/api/v10/channels/{channel_id}/messages/{message_id}"
            headers = {
                "Authorization": f"Bot {self.bot_token}"
            }
            
            async with self.session.delete(url, headers=headers) as response:
                return response.status == 204
                
        except Exception as e:
            logger.error(f"Error deleting Discord message: {str(e)}")
            return False
    
    async def get_item(self, item_type: str, item_id: str) -> Dict[str, Any]:
        """
        Get an item from Discord.
        
        Args:
            item_type: Type of item to get
            item_id: ID of the item to get
            
        Returns:
            Dictionary with item details
        """
        if not await self.is_connected():
            logger.error("Not connected to Discord")
            return {}
            
        if not self.use_bot_api:
            logger.error("Getting messages requires Bot API")
            return {}
            
        if item_type != "message":
            logger.error(f"Unsupported item type: {item_type}")
            return {}
            
        channel_id = self.default_channel_id
        
        # Extract channel ID if item_id contains it
        if ":" in item_id:
            parts = item_id.split(":")
            if len(parts) == 2:
                channel_id, message_id = parts
            else:
                message_id = item_id
        else:
            message_id = item_id
        
        try:
            # Get message via Bot API
            url = f"https://discord.com/api/v10/channels/{channel_id}/messages/{message_id}"
            headers = {
                "Authorization": f"Bot {self.bot_token}"
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status != 200:
                    return {}
                    
                return await response.json()
                
        except Exception as e:
            logger.error(f"Error getting Discord message: {str(e)}")
            return {}
    
    async def system_info(self) -> Dict[str, Any]:
        """
        Get information about Discord.
        
        Returns:
            Dictionary with system information
        """
        info = {
            "name": "Discord",
            "type": "notification",
            "version": "v10",
            "connected": await self.is_connected(),
            "capabilities": self.get_capabilities().to_dict()
        }
        
        if self.use_bot_api and await self.is_connected():
            try:
                # Get bot information
                url = "https://discord.com/api/v10/users/@me"
                headers = {
                    "Authorization": f"Bot {self.bot_token}"
                }
                
                async with self.session.get(url, headers=headers) as response:
                    if response.status == 200:
                        bot_info = await response.json()
                        info["bot_username"] = bot_info.get("username")
                        info["bot_id"] = bot_info.get("id")
                        
            except Exception as e:
                logger.error(f"Error getting Discord bot info: {str(e)}")
        
        return info
    
    async def close(self) -> None:
        """
        Close the connection to Discord and clean up resources.
        
        Returns:
            None
        """
        if self.session:
            await self.session.close()
            self.session = None
            
        self.connected = False
        logger.info("Discord connector closed")
    
    def get_capabilities(self) -> ConnectorCapabilities:
        """
        Get connector capabilities.
        
        Returns:
            ConnectorCapabilities instance
        """
        return ConnectorCapabilities(
            supports_create=True,
            supports_update=False,
            supports_delete=self.use_bot_api,
            supports_query=False,
            supports_batch_operations=False,
            supports_attachments=False,
            supports_comments=False,
            supports_custom_fields=False,
            supports_relationships=False,
            supports_history=False,
            item_types=["message", "embed"],
            query_operators=[],
            max_batch_size=0,
            rate_limit=50,  # Discord has a rate limit of ~50 requests per second
            supports_embeds=True,
            supports_files=self.use_bot_api,
            supports_reactions=self.use_bot_api
        )

# Register connector with the factory
ExternalSystemFactory.register_connector("discord", DiscordConnector)