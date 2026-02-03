"""
WebSocket handler for HF Model Server

Provides real-time bidirectional communication for:
- Streaming inference responses
- Real-time model status updates
- Backend health monitoring
- Queue status updates
"""

import asyncio
import json
import logging
import time
from typing import Dict, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import asdict

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {
            "inference": set(),
            "status": set(),
            "backend": set(),
            "queue": set(),
        }
    
    async def connect(self, client_id: str, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket client connected: {client_id}")
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            # Remove from all subscriptions
            for topic in self.subscriptions.values():
                topic.discard(client_id)
            logger.info(f"WebSocket client disconnected: {client_id}")
    
    def subscribe(self, client_id: str, topic: str):
        """Subscribe a client to a topic"""
        if topic in self.subscriptions:
            self.subscriptions[topic].add(client_id)
            logger.debug(f"Client {client_id} subscribed to {topic}")
    
    def unsubscribe(self, client_id: str, topic: str):
        """Unsubscribe a client from a topic"""
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(client_id)
            logger.debug(f"Client {client_id} unsubscribed from {topic}")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """Send a message to a specific client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_to_topic(self, message: Dict[str, Any], topic: str):
        """Broadcast a message to all subscribers of a topic"""
        if topic not in self.subscriptions:
            logger.warning(f"Unknown topic: {topic}")
            return
        
        disconnected = []
        for client_id in self.subscriptions[topic]:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to {client_id}: {e}")
                    disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)
    
    async def broadcast_all(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)


class WebSocketInferenceHandler:
    """Handles WebSocket-based inference requests"""
    
    def __init__(self, connection_manager: ConnectionManager, backend_manager=None):
        self.connection_manager = connection_manager
        self.backend_manager = backend_manager
        self.active_inference_tasks: Dict[str, asyncio.Task] = {}
    
    async def handle_client(self, websocket: WebSocket, client_id: str):
        """Handle WebSocket client connection"""
        await self.connection_manager.connect(client_id, websocket)
        
        try:
            # Send welcome message
            await self.connection_manager.send_personal_message({
                "type": "connection",
                "status": "connected",
                "client_id": client_id,
                "timestamp": time.time()
            }, client_id)
            
            # Message handling loop
            while True:
                try:
                    data = await websocket.receive_json()
                    await self.handle_message(client_id, data)
                except WebSocketDisconnect:
                    logger.info(f"Client {client_id} disconnected")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from {client_id}: {e}")
                    await self.connection_manager.send_personal_message({
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": time.time()
                    }, client_id)
                except Exception as e:
                    logger.error(f"Error handling message from {client_id}: {e}")
                    await self.connection_manager.send_personal_message({
                        "type": "error",
                        "message": str(e),
                        "timestamp": time.time()
                    }, client_id)
        
        finally:
            # Cancel any active inference tasks for this client
            for task_id, task in list(self.active_inference_tasks.items()):
                if task_id.startswith(client_id):
                    task.cancel()
                    del self.active_inference_tasks[task_id]
            
            self.connection_manager.disconnect(client_id)
    
    async def handle_message(self, client_id: str, data: Dict[str, Any]):
        """Handle incoming WebSocket message"""
        msg_type = data.get("type")
        
        if msg_type == "subscribe":
            # Subscribe to topics
            topics = data.get("topics", [])
            for topic in topics:
                self.connection_manager.subscribe(client_id, topic)
            
            await self.connection_manager.send_personal_message({
                "type": "subscribed",
                "topics": topics,
                "timestamp": time.time()
            }, client_id)
        
        elif msg_type == "unsubscribe":
            # Unsubscribe from topics
            topics = data.get("topics", [])
            for topic in topics:
                self.connection_manager.unsubscribe(client_id, topic)
            
            await self.connection_manager.send_personal_message({
                "type": "unsubscribed",
                "topics": topics,
                "timestamp": time.time()
            }, client_id)
        
        elif msg_type == "inference":
            # Handle inference request
            await self.handle_inference_request(client_id, data)
        
        elif msg_type == "status":
            # Send status information
            await self.send_status(client_id, data.get("details", "summary"))
        
        elif msg_type == "ping":
            # Respond to ping
            await self.connection_manager.send_personal_message({
                "type": "pong",
                "timestamp": time.time()
            }, client_id)
        
        else:
            logger.warning(f"Unknown message type from {client_id}: {msg_type}")
            await self.connection_manager.send_personal_message({
                "type": "error",
                "message": f"Unknown message type: {msg_type}",
                "timestamp": time.time()
            }, client_id)
    
    async def handle_inference_request(self, client_id: str, data: Dict[str, Any]):
        """Handle an inference request via WebSocket"""
        request_id = data.get("request_id", f"{client_id}_{int(time.time() * 1000)}")
        
        try:
            # Extract inference parameters
            model = data.get("model")
            task = data.get("task", "text-generation")
            inputs = data.get("inputs")
            parameters = data.get("parameters", {})
            stream = data.get("stream", False)
            
            if not model or not inputs:
                await self.connection_manager.send_personal_message({
                    "type": "error",
                    "request_id": request_id,
                    "message": "Missing required fields: model and inputs",
                    "timestamp": time.time()
                }, client_id)
                return
            
            # Send acknowledgment
            await self.connection_manager.send_personal_message({
                "type": "inference_started",
                "request_id": request_id,
                "model": model,
                "task": task,
                "timestamp": time.time()
            }, client_id)
            
            # Create inference task
            task_id = f"{client_id}_{request_id}"
            inference_task = asyncio.create_task(
                self.run_inference(client_id, request_id, model, task, inputs, parameters, stream)
            )
            self.active_inference_tasks[task_id] = inference_task
            
            # Wait for completion
            try:
                await inference_task
            finally:
                if task_id in self.active_inference_tasks:
                    del self.active_inference_tasks[task_id]
        
        except Exception as e:
            logger.error(f"Error handling inference request: {e}")
            await self.connection_manager.send_personal_message({
                "type": "inference_error",
                "request_id": request_id,
                "error": str(e),
                "timestamp": time.time()
            }, client_id)
    
    async def run_inference(
        self,
        client_id: str,
        request_id: str,
        model: str,
        task: str,
        inputs: Any,
        parameters: Dict[str, Any],
        stream: bool
    ):
        """Run the actual inference"""
        try:
            # TODO: Implement actual inference using backend_manager
            # For now, send a mock response
            
            if stream:
                # Simulate streaming response
                tokens = ["This ", "is ", "a ", "mock ", "streaming ", "response."]
                for i, token in enumerate(tokens):
                    await self.connection_manager.send_personal_message({
                        "type": "inference_chunk",
                        "request_id": request_id,
                        "chunk": token,
                        "index": i,
                        "timestamp": time.time()
                    }, client_id)
                    await asyncio.sleep(0.1)  # Simulate processing time
                
                # Send completion
                await self.connection_manager.send_personal_message({
                    "type": "inference_complete",
                    "request_id": request_id,
                    "timestamp": time.time()
                }, client_id)
            else:
                # Non-streaming response
                await asyncio.sleep(0.5)  # Simulate processing time
                
                await self.connection_manager.send_personal_message({
                    "type": "inference_result",
                    "request_id": request_id,
                    "result": "This is a mock response.",
                    "model": model,
                    "task": task,
                    "timestamp": time.time()
                }, client_id)
        
        except asyncio.CancelledError:
            logger.info(f"Inference cancelled for request {request_id}")
            await self.connection_manager.send_personal_message({
                "type": "inference_cancelled",
                "request_id": request_id,
                "timestamp": time.time()
            }, client_id)
            raise
        
        except Exception as e:
            logger.error(f"Inference error for request {request_id}: {e}")
            await self.connection_manager.send_personal_message({
                "type": "inference_error",
                "request_id": request_id,
                "error": str(e),
                "timestamp": time.time()
            }, client_id)
    
    async def send_status(self, client_id: str, detail_level: str = "summary"):
        """Send status information to client"""
        status = {
            "type": "status",
            "timestamp": time.time(),
            "detail_level": detail_level
        }
        
        if self.backend_manager:
            # Add backend manager status
            status["backend_status"] = self.backend_manager.get_backend_status_report()
        
        await self.connection_manager.send_personal_message(status, client_id)


# Global connection manager instance
_global_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance"""
    global _global_connection_manager
    if _global_connection_manager is None:
        _global_connection_manager = ConnectionManager()
    return _global_connection_manager
