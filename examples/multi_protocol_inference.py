"""
Multi-Protocol Inference Example

This example demonstrates using the unified inference service with:
- HTTP/REST API
- WebSocket real-time communication
- libp2p distributed inference (if enabled)
- MCP tool integration

Run this after starting the unified inference service.
"""

import asyncio
import aiohttp
import websockets
import json
import time
from typing import Dict, Any


class MultiProtocolInferenceClient:
    """Client that can use multiple protocols for inference"""
    
    def __init__(
        self,
        http_base_url: str = "http://localhost:8000",
        websocket_url: str = "ws://localhost:8000/ws/example_client",
        client_id: str = "example_client"
    ):
        self.http_base_url = http_base_url
        self.websocket_url = websocket_url
        self.client_id = client_id
        self.websocket = None
    
    # -------------------------------------------------------------------------
    # HTTP/REST API Methods
    # -------------------------------------------------------------------------
    
    async def http_check_health(self) -> Dict[str, Any]:
        """Check server health via HTTP"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.http_base_url}/health") as resp:
                return await resp.json()
    
    async def http_list_models(self) -> Dict[str, Any]:
        """List available models via HTTP"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.http_base_url}/v1/models") as resp:
                return await resp.json()
    
    async def http_get_status(self) -> Dict[str, Any]:
        """Get server status via HTTP"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.http_base_url}/status") as resp:
                return await resp.json()
    
    async def http_text_completion(
        self,
        prompt: str,
        model: str = "gpt2",
        max_tokens: int = 50
    ) -> Dict[str, Any]:
        """Run text completion via HTTP"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens
            }
            async with session.post(
                f"{self.http_base_url}/v1/completions",
                json=payload
            ) as resp:
                return await resp.json()
    
    async def http_chat_completion(
        self,
        messages: list,
        model: str = "gpt2"
    ) -> Dict[str, Any]:
        """Run chat completion via HTTP"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": model,
                "messages": messages
            }
            async with session.post(
                f"{self.http_base_url}/v1/chat/completions",
                json=payload
            ) as resp:
                return await resp.json()
    
    # -------------------------------------------------------------------------
    # WebSocket Methods
    # -------------------------------------------------------------------------
    
    async def websocket_connect(self):
        """Connect to WebSocket"""
        print(f"Connecting to WebSocket: {self.websocket_url}")
        self.websocket = await websockets.connect(self.websocket_url)
        print("WebSocket connected")
    
    async def websocket_disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            print("WebSocket disconnected")
    
    async def websocket_send(self, message: Dict[str, Any]):
        """Send message via WebSocket"""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        await self.websocket.send(json.dumps(message))
    
    async def websocket_receive(self) -> Dict[str, Any]:
        """Receive message from WebSocket"""
        if not self.websocket:
            raise RuntimeError("WebSocket not connected")
        message = await self.websocket.recv()
        return json.loads(message)
    
    async def websocket_subscribe(self, topics: list):
        """Subscribe to WebSocket topics"""
        await self.websocket_send({
            "type": "subscribe",
            "topics": topics
        })
        # Wait for confirmation
        response = await self.websocket_receive()
        return response
    
    async def websocket_inference(
        self,
        model: str,
        inputs: Any,
        task: str = "text-generation",
        stream: bool = False,
        parameters: Dict[str, Any] = None
    ):
        """Run inference via WebSocket"""
        request_id = f"req_{int(time.time() * 1000)}"
        
        await self.websocket_send({
            "type": "inference",
            "request_id": request_id,
            "model": model,
            "task": task,
            "inputs": inputs,
            "stream": stream,
            "parameters": parameters or {}
        })
        
        # Collect responses
        results = []
        while True:
            response = await self.websocket_receive()
            
            if response["type"] == "inference_started":
                print(f"Inference started: {response['request_id']}")
            
            elif response["type"] == "inference_chunk" and stream:
                print(f"Chunk: {response['chunk']}", end="", flush=True)
                results.append(response["chunk"])
            
            elif response["type"] == "inference_result" and not stream:
                print(f"Result: {response['result']}")
                results.append(response["result"])
                break
            
            elif response["type"] == "inference_complete":
                print("\nInference complete")
                break
            
            elif response["type"] == "inference_error":
                print(f"Error: {response['error']}")
                break
        
        return results
    
    async def websocket_get_status(self):
        """Get status via WebSocket"""
        await self.websocket_send({
            "type": "status",
            "details": "full"
        })
        
        response = await self.websocket_receive()
        return response
    
    # -------------------------------------------------------------------------
    # MCP Tool Methods (if available)
    # -------------------------------------------------------------------------
    
    def mcp_list_backends(self) -> Dict[str, Any]:
        """List inference backends via MCP tool"""
        try:
            from ipfs_accelerate_py.mcp.tools.backend_management import (
                list_inference_backends
            )
            return list_inference_backends()
        except ImportError:
            return {"error": "MCP tools not available"}
    
    def mcp_get_backend_status(self) -> Dict[str, Any]:
        """Get backend status via MCP tool"""
        try:
            from ipfs_accelerate_py.mcp.tools.backend_management import (
                get_backend_status
            )
            return get_backend_status()
        except ImportError:
            return {"error": "MCP tools not available"}
    
    def mcp_select_backend(
        self,
        task: str,
        model: str = None
    ) -> Dict[str, Any]:
        """Select backend via MCP tool"""
        try:
            from ipfs_accelerate_py.mcp.tools.backend_management import (
                select_backend_for_inference
            )
            return select_backend_for_inference(task=task, model=model)
        except ImportError:
            return {"error": "MCP tools not available"}


async def run_http_examples(client: MultiProtocolInferenceClient):
    """Run HTTP/REST API examples"""
    print("\n" + "="*70)
    print("HTTP/REST API EXAMPLES")
    print("="*70)
    
    # Health check
    print("\n1. Health Check:")
    health = await client.http_check_health()
    print(f"   Status: {health}")
    
    # List models
    print("\n2. List Models:")
    models = await client.http_list_models()
    print(f"   Available models: {len(models.get('data', []))}")
    for model in models.get('data', [])[:3]:
        print(f"   - {model['id']}")
    
    # Get status
    print("\n3. Server Status:")
    status = await client.http_get_status()
    print(f"   Status: {status.get('status')}")
    print(f"   Uptime: {status.get('uptime_seconds', 0):.1f}s")
    
    # Text completion
    print("\n4. Text Completion:")
    result = await client.http_text_completion(
        prompt="The quick brown fox",
        model="gpt2",
        max_tokens=20
    )
    print(f"   Generated: {result.get('choices', [{}])[0].get('text', 'N/A')}")
    
    # Chat completion
    print("\n5. Chat Completion:")
    result = await client.http_chat_completion(
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ],
        model="gpt2"
    )
    reply = result.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')
    print(f"   Reply: {reply}")


async def run_websocket_examples(client: MultiProtocolInferenceClient):
    """Run WebSocket examples"""
    print("\n" + "="*70)
    print("WEBSOCKET EXAMPLES")
    print("="*70)
    
    # Connect
    print("\n1. Connecting to WebSocket...")
    await client.websocket_connect()
    
    # Subscribe to topics
    print("\n2. Subscribing to Topics:")
    topics = ["inference", "status"]
    response = await client.websocket_subscribe(topics)
    print(f"   Subscribed to: {response.get('topics', [])}")
    
    # Non-streaming inference
    print("\n3. Non-Streaming Inference:")
    await client.websocket_inference(
        model="gpt2",
        inputs="Hello, world!",
        task="text-generation",
        stream=False,
        parameters={"max_length": 50}
    )
    
    # Streaming inference
    print("\n4. Streaming Inference:")
    print("   Generated: ", end="")
    await client.websocket_inference(
        model="gpt2",
        inputs="Once upon a time",
        task="text-generation",
        stream=True,
        parameters={"max_length": 50}
    )
    
    # Get status via WebSocket
    print("\n5. Get Status via WebSocket:")
    status = await client.websocket_get_status()
    print(f"   Backends: {status.get('backend_status', {}).get('total_backends', 0)}")
    
    # Disconnect
    print("\n6. Disconnecting...")
    await client.websocket_disconnect()


def run_mcp_examples(client: MultiProtocolInferenceClient):
    """Run MCP tool examples"""
    print("\n" + "="*70)
    print("MCP TOOL EXAMPLES")
    print("="*70)
    
    # List backends
    print("\n1. List Inference Backends:")
    backends = client.mcp_list_backends()
    if "error" not in backends:
        print(f"   Total backends: {backends.get('total_backends', 0)}")
        for backend in backends.get('backends', [])[:3]:
            print(f"   - {backend['name']} ({backend['type']})")
    else:
        print(f"   {backends['error']}")
    
    # Get backend status
    print("\n2. Backend Status:")
    status = client.mcp_get_backend_status()
    if "error" not in status:
        print(f"   Total requests: {status.get('total_requests', 0)}")
        print(f"   Successful: {status.get('total_successful', 0)}")
        print(f"   Failed: {status.get('total_failed', 0)}")
    else:
        print(f"   {status['error']}")
    
    # Select backend
    print("\n3. Select Backend for Task:")
    backend = client.mcp_select_backend(
        task="text-generation",
        model="gpt2"
    )
    if "error" not in backend:
        print(f"   Selected: {backend.get('name')} ({backend.get('type')})")
        print(f"   Endpoint: {backend.get('endpoint', 'N/A')}")
    else:
        print(f"   {backend.get('error')}")


async def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("MULTI-PROTOCOL INFERENCE CLIENT EXAMPLES")
    print("="*70)
    print("\nThis example demonstrates using the unified inference service via:")
    print("- HTTP/REST API")
    print("- WebSocket real-time communication")
    print("- MCP tool integration")
    print("\nMake sure the unified inference service is running:")
    print("  python -m ipfs_accelerate_py.unified_inference_service")
    print("="*70)
    
    # Create client
    client = MultiProtocolInferenceClient()
    
    # Run examples
    try:
        # HTTP examples
        await run_http_examples(client)
        
        # WebSocket examples
        await run_websocket_examples(client)
        
        # MCP examples (synchronous)
        run_mcp_examples(client)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the server is running:")
        print("  python -m ipfs_accelerate_py.unified_inference_service")
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
