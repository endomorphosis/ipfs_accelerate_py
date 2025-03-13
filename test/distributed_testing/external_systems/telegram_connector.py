#!/usr/bin/env python3
"""
Telegram Connector for External Systems Integration

This module implements a Telegram connector for the External Systems Integration API,
allowing the distributed testing framework to send messages to Telegram channels
via the Telegram Bot API.
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

class TelegramConnector(ExternalSystemInterface):
    """
    Telegram connector implementing the External System Interface.
    
    This connector allows sending messages to Telegram channels via the Telegram Bot API.
    """
    
    def __init__(self):
        """Initialize the Telegram connector."""
        self.initialized = False
        self.connected = False
        self.config = {}
        self.session = None
        self.bot_token = None
        self.default_chat_id = None
        self.api_base_url = None
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the Telegram connector with configuration.
        
        Args:
            config: Configuration dictionary containing Telegram-specific settings
            
        Returns:
            True if initialization succeeded
        """
        self.config = config
        
        # Extract configuration
        self.bot_token = config.get("bot_token")
        self.default_chat_id = config.get("default_chat_id")
        
        # Validate configuration
        if not self.bot_token:
            logger.error("bot_token is required")
            return False
            
        if not self.default_chat_id:
            logger.warning("default_chat_id is not provided, it will need to be specified with each message")
        
        # Setup API base URL
        self.api_base_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        self.initialized = True
        logger.info("Telegram connector initialized")
        return True
    
    async def connect(self) -> bool:
        """
        Establish connection to Telegram.
        
        Returns:
            True if connection succeeded
        """
        if not self.initialized:
            logger.error("Telegram connector not initialized")
            return False
        
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Test connection by getting bot info
            async with self.session.get(f"{self.api_base_url}/getMe") as response:
                if response.status != 200:
                    logger.error(f"Failed to connect to Telegram Bot API: {response.status}")
                    await self.session.close()
                    self.session = None
                    return False
                
                # Check if the response is valid
                data = await response.json()
                if not data.get("ok"):
                    error_msg = data.get("description", "Unknown error")
                    logger.error(f"Failed to get bot info: {error_msg}")
                    await self.session.close()
                    self.session = None
                    return False
            
            self.connected = True
            logger.info("Connected to Telegram")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Telegram: {str(e)}")
            
            if self.session:
                await self.session.close()
                self.session = None
                
            return False
    
    async def is_connected(self) -> bool:
        """
        Check if the connector is currently connected to Telegram.
        
        Returns:
            True if connected
        """
        return self.connected and self.session is not None
    
    async def execute_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an operation on Telegram.
        
        Args:
            operation: The operation to execute
            params: Parameters for the operation
            
        Returns:
            Dictionary with operation result
        """
        if not await self.is_connected():
            logger.error("Not connected to Telegram")
            return {"success": False, "error": "Not connected to Telegram"}
        
        if operation == "send_message":
            return await self._send_message(params)
        elif operation == "send_photo":
            return await self._send_photo(params)
        elif operation == "send_document":
            return await self._send_document(params)
        else:
            logger.error(f"Unsupported operation: {operation}")
            return {"success": False, "error": f"Unsupported operation: {operation}"}
    
    async def _send_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a text message to a Telegram chat.
        
        Args:
            params: Parameters for the message
            
        Returns:
            Dictionary with operation result
        """
        chat_id = params.get("chat_id", self.default_chat_id)
        text = params.get("text", "")
        parse_mode = params.get("parse_mode", "HTML")  # HTML or Markdown
        disable_web_page_preview = params.get("disable_web_page_preview", False)
        disable_notification = params.get("disable_notification", False)
        
        if not chat_id:
            logger.error("chat_id is required")
            return {"success": False, "error": "chat_id is required"}
            
        if not text:
            logger.error("text is required")
            return {"success": False, "error": "text is required"}
        
        try:
            data = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": disable_web_page_preview,
                "disable_notification": disable_notification
            }
            
            async with self.session.post(f"{self.api_base_url}/sendMessage", json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to send message: {response.status} - {error_text}")
                    return {"success": False, "error": f"Failed to send message: {response.status} - {error_text}"}
                
                result = await response.json()
                
                if not result.get("ok"):
                    error_msg = result.get("description", "Unknown error")
                    logger.error(f"Failed to send message: {error_msg}")
                    return {"success": False, "error": f"Failed to send message: {error_msg}"}
                
                message = result.get("result", {})
                
                return {
                    "success": True,
                    "message_id": message.get("message_id"),
                    "date": message.get("date")
                }
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _send_photo(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a photo to a Telegram chat.
        
        Args:
            params: Parameters for the photo
            
        Returns:
            Dictionary with operation result
        """
        chat_id = params.get("chat_id", self.default_chat_id)
        photo_url = params.get("photo_url")
        caption = params.get("caption", "")
        parse_mode = params.get("parse_mode", "HTML")
        disable_notification = params.get("disable_notification", False)
        
        if not chat_id:
            logger.error("chat_id is required")
            return {"success": False, "error": "chat_id is required"}
            
        if not photo_url:
            logger.error("photo_url is required")
            return {"success": False, "error": "photo_url is required"}
        
        try:
            data = {
                "chat_id": chat_id,
                "photo": photo_url,
                "caption": caption,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification
            }
            
            async with self.session.post(f"{self.api_base_url}/sendPhoto", json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to send photo: {response.status} - {error_text}")
                    return {"success": False, "error": f"Failed to send photo: {response.status} - {error_text}"}
                
                result = await response.json()
                
                if not result.get("ok"):
                    error_msg = result.get("description", "Unknown error")
                    logger.error(f"Failed to send photo: {error_msg}")
                    return {"success": False, "error": f"Failed to send photo: {error_msg}"}
                
                message = result.get("result", {})
                
                return {
                    "success": True,
                    "message_id": message.get("message_id"),
                    "date": message.get("date")
                }
                
        except Exception as e:
            logger.error(f"Error sending Telegram photo: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _send_document(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a document to a Telegram chat.
        
        Args:
            params: Parameters for the document
            
        Returns:
            Dictionary with operation result
        """
        chat_id = params.get("chat_id", self.default_chat_id)
        document_url = params.get("document_url")
        caption = params.get("caption", "")
        parse_mode = params.get("parse_mode", "HTML")
        disable_notification = params.get("disable_notification", False)
        
        if not chat_id:
            logger.error("chat_id is required")
            return {"success": False, "error": "chat_id is required"}
            
        if not document_url:
            logger.error("document_url is required")
            return {"success": False, "error": "document_url is required"}
        
        try:
            data = {
                "chat_id": chat_id,
                "document": document_url,
                "caption": caption,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification
            }
            
            async with self.session.post(f"{self.api_base_url}/sendDocument", json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Failed to send document: {response.status} - {error_text}")
                    return {"success": False, "error": f"Failed to send document: {response.status} - {error_text}"}
                
                result = await response.json()
                
                if not result.get("ok"):
                    error_msg = result.get("description", "Unknown error")
                    logger.error(f"Failed to send document: {error_msg}")
                    return {"success": False, "error": f"Failed to send document: {error_msg}"}
                
                message = result.get("result", {})
                
                return {
                    "success": True,
                    "message_id": message.get("message_id"),
                    "date": message.get("date")
                }
                
        except Exception as e:
            logger.error(f"Error sending Telegram document: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def query(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Query Telegram for data.
        
        Args:
            query_params: Query parameters
            
        Returns:
            List of query results
        """
        # Only support querying for updates
        if query_params.get("type") == "updates":
            try:
                offset = query_params.get("offset", 0)
                limit = query_params.get("limit", 100)
                timeout = query_params.get("timeout", 0)
                allowed_updates = query_params.get("allowed_updates", [])
                
                data = {
                    "offset": offset,
                    "limit": limit,
                    "timeout": timeout
                }
                
                if allowed_updates:
                    data["allowed_updates"] = allowed_updates
                
                async with self.session.post(f"{self.api_base_url}/getUpdates", json=data) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get updates: {response.status}")
                        return []
                    
                    result = await response.json()
                    
                    if not result.get("ok"):
                        error_msg = result.get("description", "Unknown error")
                        logger.error(f"Failed to get updates: {error_msg}")
                        return []
                    
                    return result.get("result", [])
                    
            except Exception as e:
                logger.error(f"Error querying Telegram updates: {str(e)}")
                return []
        else:
            logger.error("Telegram connector only supports querying for updates")
            return []
    
    async def create_item(self, item_type: str, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an item in Telegram.
        
        Args:
            item_type: Type of item to create
            item_data: Item data
            
        Returns:
            Dictionary with created item details
        """
        if item_type == "message":
            return await self._send_message(item_data)
        elif item_type == "photo":
            return await self._send_photo(item_data)
        elif item_type == "document":
            return await self._send_document(item_data)
        else:
            logger.error(f"Unsupported item type: {item_type}")
            return {"success": False, "error": f"Unsupported item type: {item_type}"}
    
    async def update_item(self, item_type: str, item_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an item in Telegram.
        
        Args:
            item_type: Type of item to update
            item_id: ID of the item to update
            update_data: Data to update
            
        Returns:
            True if update succeeded
        """
        if not await self.is_connected():
            logger.error("Not connected to Telegram")
            return False
            
        if item_type != "message":
            logger.error(f"Unsupported item type: {item_type}")
            return False
        
        try:
            # Parse message_id and chat_id from item_id (format: "chat_id:message_id")
            if ":" in item_id:
                parts = item_id.split(":")
                if len(parts) != 2:
                    logger.error(f"Invalid item_id format: {item_id}")
                    return False
                
                chat_id, message_id = parts
            else:
                # Use default chat_id
                chat_id = self.default_chat_id
                message_id = item_id
            
            if not chat_id:
                logger.error("chat_id is required")
                return False
            
            # Prepare data for editing
            text = update_data.get("text")
            
            if not text:
                logger.error("text is required for message update")
                return False
                
            parse_mode = update_data.get("parse_mode", "HTML")
            disable_web_page_preview = update_data.get("disable_web_page_preview", False)
            
            data = {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": disable_web_page_preview
            }
            
            async with self.session.post(f"{self.api_base_url}/editMessageText", json=data) as response:
                if response.status != 200:
                    logger.error(f"Failed to update message: {response.status}")
                    return False
                
                result = await response.json()
                
                if not result.get("ok"):
                    error_msg = result.get("description", "Unknown error")
                    logger.error(f"Failed to update message: {error_msg}")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error updating Telegram message: {str(e)}")
            return False
    
    async def delete_item(self, item_type: str, item_id: str) -> bool:
        """
        Delete an item from Telegram.
        
        Args:
            item_type: Type of item to delete
            item_id: ID of the item to delete
            
        Returns:
            True if deletion succeeded
        """
        if not await self.is_connected():
            logger.error("Not connected to Telegram")
            return False
            
        if item_type != "message":
            logger.error(f"Unsupported item type: {item_type}")
            return False
        
        try:
            # Parse message_id and chat_id from item_id (format: "chat_id:message_id")
            if ":" in item_id:
                parts = item_id.split(":")
                if len(parts) != 2:
                    logger.error(f"Invalid item_id format: {item_id}")
                    return False
                
                chat_id, message_id = parts
            else:
                # Use default chat_id
                chat_id = self.default_chat_id
                message_id = item_id
            
            if not chat_id:
                logger.error("chat_id is required")
                return False
            
            data = {
                "chat_id": chat_id,
                "message_id": message_id
            }
            
            async with self.session.post(f"{self.api_base_url}/deleteMessage", json=data) as response:
                if response.status != 200:
                    logger.error(f"Failed to delete message: {response.status}")
                    return False
                
                result = await response.json()
                
                if not result.get("ok"):
                    error_msg = result.get("description", "Unknown error")
                    logger.error(f"Failed to delete message: {error_msg}")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error deleting Telegram message: {str(e)}")
            return False
    
    async def get_item(self, item_type: str, item_id: str) -> Dict[str, Any]:
        """
        Get an item from Telegram.
        
        Args:
            item_type: Type of item to get
            item_id: ID of the item to get
            
        Returns:
            Dictionary with item details
        """
        # Telegram doesn't provide a direct way to get a message by ID
        logger.error("Telegram API doesn't support getting messages by ID")
        return {}
    
    async def system_info(self) -> Dict[str, Any]:
        """
        Get information about Telegram.
        
        Returns:
            Dictionary with system information
        """
        info = {
            "name": "Telegram",
            "type": "notification",
            "connected": await self.is_connected(),
            "capabilities": self.get_capabilities().to_dict()
        }
        
        if await self.is_connected():
            try:
                async with self.session.get(f"{self.api_base_url}/getMe") as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if result.get("ok"):
                            bot_info = result.get("result", {})
                            info["bot_username"] = bot_info.get("username")
                            info["bot_id"] = bot_info.get("id")
                            info["bot_name"] = bot_info.get("first_name")
                            info["is_bot"] = bot_info.get("is_bot")
                            
            except Exception as e:
                logger.error(f"Error getting Telegram bot info: {str(e)}")
        
        return info
    
    async def close(self) -> None:
        """
        Close the connection to Telegram and clean up resources.
        
        Returns:
            None
        """
        if self.session:
            await self.session.close()
            self.session = None
            
        self.connected = False
        logger.info("Telegram connector closed")
    
    def get_capabilities(self) -> ConnectorCapabilities:
        """
        Get connector capabilities.
        
        Returns:
            ConnectorCapabilities instance
        """
        return ConnectorCapabilities(
            supports_create=True,
            supports_update=True,
            supports_delete=True,
            supports_query=True,
            supports_batch_operations=False,
            supports_attachments=True,
            supports_comments=False,
            supports_custom_fields=False,
            supports_relationships=False,
            supports_history=False,
            item_types=["message", "photo", "document"],
            query_operators=[],
            max_batch_size=0,
            rate_limit=30,  # Telegram has a rate limit of ~30 messages per second
            supports_html_formatting=True,
            supports_markdown_formatting=True,
            supports_inline_buttons=True,
            supports_file_uploads=True
        )

# Register connector with the factory
ExternalSystemFactory.register_connector("telegram", TelegramConnector)