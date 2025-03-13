"""
WebSocket Handlers for Monitoring Dashboard

This module provides WebSocket handlers for the monitoring dashboard, enabling real-time
communication between clients and the dashboard.
"""

import asyncio
import json
import logging
from typing import Dict, List, Set, Any, Optional

import aiohttp
from aiohttp import web
import websockets

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manager for WebSocket connections and message distribution."""
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        self.connections = {}
        self.connection_topics = {}
        self.topic_connections = {}
        
        logger.info("WebSocket manager initialized")
    
    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle a WebSocket connection request.
        
        Args:
            request: WebSocket request
            
        Returns:
            WebSocket response
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Generate connection ID
        conn_id = id(ws)
        self.connections[conn_id] = ws
        self.connection_topics[conn_id] = set()
        
        logger.info(f"WebSocket connection established: {conn_id}")
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self.handle_message(conn_id, msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket connection {conn_id} closed with error: {ws.exception()}")
        finally:
            # Cleanup when connection is closed
            await self.remove_connection(conn_id)
        
        return ws
    
    async def handle_message(self, conn_id: int, message: str):
        """Handle a WebSocket message.
        
        Args:
            conn_id: Connection ID
            message: Message data
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type", "unknown")
            
            if msg_type == "subscribe":
                topic = data.get("topic")
                if topic:
                    await self.subscribe(conn_id, topic)
            
            elif msg_type == "unsubscribe":
                topic = data.get("topic")
                if topic:
                    await self.unsubscribe(conn_id, topic)
            
            elif msg_type == "e2e_test_monitoring_init":
                # Handle test monitoring initialization
                test_id = data.get("test_id")
                if test_id:
                    topic = f"e2e_test_monitoring:{test_id}"
                    await self.subscribe(conn_id, topic)
                    
                    # Broadcast initialization to subscribers
                    await self.broadcast(topic, data)
                    
                    logger.info(f"E2E test monitoring initialized for test {test_id}")
            
            elif msg_type == "e2e_test_monitoring_update":
                # Handle test monitoring update
                test_data = data.get("data", {})
                test_id = test_data.get("test_id")
                if test_id:
                    topic = f"e2e_test_monitoring:{test_id}"
                    # Broadcast update to subscribers
                    await self.broadcast(topic, data)
            
            else:
                logger.warning(f"Unknown message type: {msg_type}")
        
        except json.JSONDecodeError:
            logger.error(f"Failed to parse WebSocket message: {message}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def subscribe(self, conn_id: int, topic: str):
        """Subscribe a connection to a topic.
        
        Args:
            conn_id: Connection ID
            topic: Topic to subscribe to
        """
        if conn_id not in self.connections:
            return
        
        self.connection_topics[conn_id].add(topic)
        
        if topic not in self.topic_connections:
            self.topic_connections[topic] = set()
        
        self.topic_connections[topic].add(conn_id)
        
        logger.debug(f"Connection {conn_id} subscribed to topic: {topic}")
        
        # Send confirmation
        await self.send_to_connection(conn_id, {
            "type": "subscription_confirmed",
            "topic": topic
        })
    
    async def unsubscribe(self, conn_id: int, topic: str):
        """Unsubscribe a connection from a topic.
        
        Args:
            conn_id: Connection ID
            topic: Topic to unsubscribe from
        """
        if conn_id not in self.connections:
            return
        
        if topic in self.connection_topics[conn_id]:
            self.connection_topics[conn_id].remove(topic)
        
        if topic in self.topic_connections and conn_id in self.topic_connections[topic]:
            self.topic_connections[topic].remove(conn_id)
            
            # Clean up empty topic sets
            if not self.topic_connections[topic]:
                del self.topic_connections[topic]
        
        logger.debug(f"Connection {conn_id} unsubscribed from topic: {topic}")
        
        # Send confirmation
        await self.send_to_connection(conn_id, {
            "type": "unsubscription_confirmed",
            "topic": topic
        })
    
    async def remove_connection(self, conn_id: int):
        """Remove a connection and clean up subscriptions.
        
        Args:
            conn_id: Connection ID
        """
        if conn_id not in self.connections:
            return
        
        # Remove from connections
        if conn_id in self.connections:
            del self.connections[conn_id]
        
        # Remove from topics
        if conn_id in self.connection_topics:
            topics = list(self.connection_topics[conn_id])
            for topic in topics:
                if topic in self.topic_connections and conn_id in self.topic_connections[topic]:
                    self.topic_connections[topic].remove(conn_id)
                    
                    # Clean up empty topic sets
                    if not self.topic_connections[topic]:
                        del self.topic_connections[topic]
            
            del self.connection_topics[conn_id]
        
        logger.info(f"WebSocket connection removed: {conn_id}")
    
    async def broadcast(self, topic: str, message: Any):
        """Broadcast a message to all subscribers of a topic.
        
        Args:
            topic: Topic to broadcast to
            message: Message to broadcast
        """
        if topic not in self.topic_connections:
            return
        
        # Convert message to JSON if it's not a string
        if not isinstance(message, str):
            message = json.dumps(message)
        
        # Get connection IDs for this topic
        conn_ids = list(self.topic_connections[topic])
        
        # Send message to all connections
        for conn_id in conn_ids:
            if conn_id in self.connections:
                try:
                    await self.connections[conn_id].send_str(message)
                except Exception as e:
                    logger.error(f"Failed to send message to connection {conn_id}: {e}")
                    # Remove problematic connection
                    await self.remove_connection(conn_id)
    
    async def send_to_connection(self, conn_id: int, message: Any):
        """Send a message to a specific connection.
        
        Args:
            conn_id: Connection ID
            message: Message to send
        """
        if conn_id not in self.connections:
            return
        
        # Convert message to JSON if it's not a string
        if not isinstance(message, str):
            message = json.dumps(message)
        
        try:
            await self.connections[conn_id].send_str(message)
        except Exception as e:
            logger.error(f"Failed to send message to connection {conn_id}: {e}")
            # Remove problematic connection
            await self.remove_connection(conn_id)

def setup_websocket_routes(app: web.Application, websocket_manager: WebSocketManager):
    """Set up WebSocket routes for the application.
    
    Args:
        app: aiohttp application
        websocket_manager: WebSocket manager instance
    """
    app.router.add_get('/ws', websocket_manager.handle_websocket)
    app.router.add_get('/ws/e2e-test-monitoring', websocket_manager.handle_websocket)
    
    logger.info("WebSocket routes configured")