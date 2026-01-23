"""
GitHub Copilot SDK Python Wrapper

This module provides Python wrappers for GitHub Copilot SDK,
enabling agentic AI features with caching and retry logic.
"""

import asyncio
import json
import logging
import os
import time
import random
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# Import cache from github_cli
try:
    from ipfs_accelerate_py.github_cli.cache import GitHubAPICache, get_global_cache
except ImportError:
    # Fallback if running standalone
    GitHubAPICache = None
    get_global_cache = None

# Try to import the Copilot SDK
try:
    from copilot import CopilotClient as _CopilotClient
    from copilot import define_tool, Tool
    from copilot.types import SessionConfig, MessageOptions
    HAVE_COPILOT_SDK = True
except ImportError:
    logger.warning("GitHub Copilot SDK not installed. Install with: pip install github-copilot-sdk")
    HAVE_COPILOT_SDK = False
    _CopilotClient = None
    define_tool = None
    Tool = None
    SessionConfig = None
    MessageOptions = None


class CopilotSDK:
    """Python wrapper for GitHub Copilot SDK with caching and retry."""
    
    def __init__(
        self,
        cli_path: Optional[str] = None,
        cli_url: Optional[str] = None,
        model: str = "gpt-4o",
        enable_cache: bool = True,
        cache: Optional['GitHubAPICache'] = None,
        cache_ttl: int = 300,
        log_level: str = "info",
        auto_start: bool = True,
        auto_restart: bool = True,
    ):
        """
        Initialize Copilot SDK wrapper.
        
        Args:
            cli_path: Path to copilot CLI executable (optional)
            cli_url: URL of existing CLI server (e.g., "localhost:8080")
            model: Default model to use (default: "gpt-4o")
            enable_cache: Whether to enable response caching
            cache: Custom cache instance (uses global cache if None)
            cache_ttl: Default cache TTL in seconds (default: 5 minutes)
            log_level: Log level for SDK (default: "info")
            auto_start: Whether to automatically start the CLI server
            auto_restart: Whether to automatically restart on crash
        """
        if not HAVE_COPILOT_SDK:
            raise ImportError(
                "GitHub Copilot SDK is not installed. "
                "Install with: pip install github-copilot-sdk"
            )
        
        self.cli_path = cli_path or os.environ.get("COPILOT_CLI_PATH", "copilot")
        self.cli_url = cli_url
        self.default_model = model
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.log_level = log_level
        self.auto_start = auto_start
        self.auto_restart = auto_restart
        
        # Set up cache (share with GitHub CLI cache)
        if enable_cache and GitHubAPICache:
            self.cache = cache if cache is not None else get_global_cache()
        else:
            self.cache = None
        
        # SDK client and session
        self._client = None
        self._active_sessions = {}
        self._tools = {}
        
        # Event loop for async operations
        self._loop = None
        self._loop_thread = None
    
    def _get_or_create_loop(self):
        """Get or create an event loop for async operations."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create a new one
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop
    
    def _run_async(self, coro):
        """Run an async coroutine and return the result."""
        try:
            loop = asyncio.get_running_loop()
            # Already in async context, just await
            return coro
        except RuntimeError:
            # Not in async context, run in new loop
            loop = self._get_or_create_loop()
            return loop.run_until_complete(coro)
    
    async def _get_or_create_client(self):
        """Get or create the Copilot client."""
        if self._client is None:
            options = {
                "log_level": self.log_level,
                "auto_start": self.auto_start,
                "auto_restart": self.auto_restart,
            }
            
            if self.cli_url:
                options["cli_url"] = self.cli_url
            elif self.cli_path:
                options["cli_path"] = self.cli_path
            
            self._client = _CopilotClient(options)
            await self._client.start()
            logger.info("Copilot SDK client started successfully")
        
        return self._client
    
    async def _create_session_async(
        self,
        model: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        streaming: bool = False,
        **kwargs
    ) -> Any:
        """Create a new Copilot session (async)."""
        client = await self._get_or_create_client()
        
        session_config = {
            "model": model or self.default_model,
            "streaming": streaming,
        }
        
        if tools:
            session_config["tools"] = tools
        
        # Add any additional config
        session_config.update(kwargs)
        
        session = await client.create_session(session_config)
        session_id = id(session)
        self._active_sessions[session_id] = session
        
        logger.info(f"Created Copilot session {session_id} with model {session_config['model']}")
        return session
    
    def create_session(
        self,
        model: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        streaming: bool = False,
        **kwargs
    ) -> Any:
        """
        Create a new Copilot session.
        
        Args:
            model: Model to use (default: self.default_model)
            tools: List of tools to register with the session
            streaming: Whether to enable streaming responses
            **kwargs: Additional session configuration
            
        Returns:
            Session object
        """
        return self._run_async(
            self._create_session_async(model, tools, streaming, **kwargs)
        )
    
    async def _send_message_async(
        self,
        session: Any,
        prompt: str,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a message to a session (async)."""
        # Check cache first
        cache_key = f"{id(session)}:{prompt}"
        if use_cache and self.cache:
            cached_result = self.cache.get("copilot_sdk_message", key=cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached copilot SDK response for: {prompt[:50]}...")
                return cached_result
        
        # Send message
        await session.send({"prompt": prompt, **kwargs})
        
        # Wait for response (collect events)
        messages = []
        done = asyncio.Event()
        
        def on_event(event):
            if event.type.value == "assistant.message":
                messages.append({
                    "type": "message",
                    "content": event.data.content,
                    "timestamp": time.time()
                })
            elif event.type.value == "assistant.reasoning":
                messages.append({
                    "type": "reasoning",
                    "content": event.data.content,
                    "timestamp": time.time()
                })
            elif event.type.value == "session.idle":
                done.set()
        
        session.on(on_event)
        await done.wait()
        
        # Build response
        response = {
            "prompt": prompt,
            "messages": messages,
            "success": True,
            "model": session._config.get("model") if hasattr(session, "_config") else self.default_model,
            "timestamp": time.time()
        }
        
        # Cache successful results
        if use_cache and self.cache:
            self.cache.put("copilot_sdk_message", response, ttl=self.cache_ttl, key=cache_key)
        
        return response
    
    def send_message(
        self,
        session: Any,
        prompt: str,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a message to a Copilot session.
        
        Args:
            session: Session object from create_session()
            prompt: Message to send
            use_cache: Whether to use cached results
            **kwargs: Additional message options
            
        Returns:
            Dict with response messages and metadata
        """
        return self._run_async(
            self._send_message_async(session, prompt, use_cache, **kwargs)
        )
    
    async def _stream_message_async(
        self,
        session: Any,
        prompt: str,
        on_chunk: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Stream a message response (async)."""
        # Send message
        await session.send({"prompt": prompt, **kwargs})
        
        # Collect streaming chunks
        chunks = []
        messages = []
        done = asyncio.Event()
        
        def on_event(event):
            if event.type.value == "assistant.message_delta":
                chunk = event.data.delta_content or ""
                if chunk:
                    chunks.append(chunk)
                    if on_chunk:
                        on_chunk(chunk)
            elif event.type.value == "assistant.message":
                messages.append({
                    "type": "message",
                    "content": event.data.content,
                    "timestamp": time.time()
                })
            elif event.type.value == "assistant.reasoning":
                messages.append({
                    "type": "reasoning",
                    "content": event.data.content,
                    "timestamp": time.time()
                })
            elif event.type.value == "session.idle":
                done.set()
        
        session.on(on_event)
        await done.wait()
        
        return {
            "prompt": prompt,
            "chunks": chunks,
            "messages": messages,
            "full_text": "".join(chunks),
            "success": True,
            "timestamp": time.time()
        }
    
    def stream_message(
        self,
        session: Any,
        prompt: str,
        on_chunk: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Stream a message response from a Copilot session.
        
        Args:
            session: Session object from create_session()
            prompt: Message to send
            on_chunk: Callback function for each chunk of text
            **kwargs: Additional message options
            
        Returns:
            Dict with chunks, messages and metadata
        """
        return self._run_async(
            self._stream_message_async(session, prompt, on_chunk, **kwargs)
        )
    
    async def _destroy_session_async(self, session: Any) -> Dict[str, Any]:
        """Destroy a session (async)."""
        try:
            await session.destroy()
            session_id = id(session)
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
            
            return {
                "success": True,
                "message": f"Session {session_id} destroyed",
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error destroying session: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def destroy_session(self, session: Any) -> Dict[str, Any]:
        """
        Destroy a Copilot session.
        
        Args:
            session: Session object to destroy
            
        Returns:
            Dict with success status
        """
        return self._run_async(self._destroy_session_async(session))
    
    async def _stop_async(self) -> Dict[str, Any]:
        """Stop the Copilot client (async)."""
        try:
            # Destroy all active sessions
            for session in list(self._active_sessions.values()):
                try:
                    await session.destroy()
                except Exception as e:
                    logger.warning(f"Error destroying session: {e}")
            
            self._active_sessions.clear()
            
            # Stop client
            if self._client:
                await self._client.stop()
                self._client = None
            
            return {
                "success": True,
                "message": "Copilot SDK stopped",
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Error stopping Copilot SDK: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop the Copilot client and cleanup resources.
        
        Returns:
            Dict with success status
        """
        return self._run_async(self._stop_async())
    
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable
    ) -> Any:
        """
        Register a tool for use in sessions.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: JSON schema for parameters
            handler: Callable to handle tool invocations
            
        Returns:
            Tool object
        """
        if not HAVE_COPILOT_SDK or not Tool:
            raise ImportError("Copilot SDK not available")
        
        tool = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler
        )
        
        self._tools[name] = tool
        logger.info(f"Registered tool: {name}")
        return tool
    
    def get_tools(self) -> Dict[str, Any]:
        """Get all registered tools."""
        return self._tools.copy()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
