#!/usr/bin/env python3
"""
API Client for IPFS Accelerate Test Suite

This module provides a client for interacting with the Test Suite API,
allowing for test execution, monitoring, and result retrieval.
"""

import os
import sys
import json
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ApiClient:
    """
    Client for interacting with the IPFS Accelerate API.
    
    This class provides methods for running tests, checking status,
    and retrieving results through the API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        
        # Import requests lazily to avoid hard dependency
        try:
            import requests
            self.session = requests.Session()
            
            # Set up authentication if provided
            if api_key:
                self.session.headers.update({
                    "Authorization": f"Bearer {api_key}"
                })
            
            # Set default headers
            self.session.headers.update({
                "Content-Type": "application/json",
                "Accept": "application/json"
            })
            
            self._requests_available = True
        except ImportError:
            logger.warning("requests package not available, HTTP methods will not work")
            self._requests_available = False
    
    def run_test(self, 
                model_name: str, 
                hardware: List[str] = ["cpu"], 
                test_type: str = "basic",
                timeout: int = 300,
                save_results: bool = True) -> Dict[str, Any]:
        """
        Run a test through the API.
        
        Args:
            model_name: Name of the model to test
            hardware: List of hardware platforms to test on
            test_type: Type of test to run (basic, comprehensive, fault_tolerance)
            timeout: Timeout in seconds
            save_results: Whether to save results to disk
            
        Returns:
            Dict containing the API response
            
        Raises:
            RuntimeError: If requests package is not available
            ConnectionError: If the API server cannot be reached
            Exception: For other API errors
        """
        if not self._requests_available:
            raise RuntimeError("requests package not available")
        
        # Prepare request data
        request_data = {
            "model_name": model_name,
            "hardware": hardware,
            "test_type": test_type,
            "timeout": timeout,
            "save_results": save_results
        }
        
        # Send request
        try:
            response = self.session.post(
                f"{self.base_url}/api/test/run",
                json=request_data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error running test: {e}")
            raise
    
    def get_test_status(self, run_id: str) -> Dict[str, Any]:
        """
        Get the status of a test run.
        
        Args:
            run_id: ID of the test run
            
        Returns:
            Dict containing the test status
            
        Raises:
            RuntimeError: If requests package is not available
            ConnectionError: If the API server cannot be reached
            Exception: For other API errors
        """
        if not self._requests_available:
            raise RuntimeError("requests package not available")
        
        try:
            response = self.session.get(f"{self.base_url}/api/test/status/{run_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting test status: {e}")
            raise
    
    def get_test_results(self, run_id: str) -> Dict[str, Any]:
        """
        Get the results of a test run.
        
        Args:
            run_id: ID of the test run
            
        Returns:
            Dict containing the test results
            
        Raises:
            RuntimeError: If requests package is not available
            ConnectionError: If the API server cannot be reached
            Exception: For other API errors
        """
        if not self._requests_available:
            raise RuntimeError("requests package not available")
        
        try:
            response = self.session.get(f"{self.base_url}/api/test/results/{run_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting test results: {e}")
            raise
    
    def monitor_test(self, run_id: str, poll_interval: float = 1.0, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Monitor a test run until completion.
        
        Args:
            run_id: ID of the test run
            poll_interval: Interval in seconds between status checks
            timeout: Optional timeout in seconds
            
        Returns:
            Dict containing the final test results
            
        Raises:
            RuntimeError: If requests package is not available
            TimeoutError: If the timeout is exceeded
            Exception: For API errors
        """
        if not self._requests_available:
            raise RuntimeError("requests package not available")
        
        start_time = time.time()
        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Test monitoring timed out after {timeout} seconds")
            
            # Get current status
            status = self.get_test_status(run_id)
            
            # Print progress update
            logger.info(f"Test {run_id} - Status: {status['status']}, Progress: {status['progress']:.1%}, "
                     f"Step: {status['current_step']}")
            
            # Check if completed or failed
            if status["status"] in ["completed", "failed"]:
                # Get full results
                return self.get_test_results(run_id)
            
            # Wait before next check
            time.sleep(poll_interval)
    
    def close(self):
        """Close the HTTP session."""
        if self._requests_available:
            self.session.close()

class AsyncApiClient:
    """
    Asynchronous client for interacting with the IPFS Accelerate API.
    
    This class provides async methods for running tests, checking status,
    and retrieving results through the API.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize the async API client.
        
        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        self._websocket = None
        
        # Import aiohttp lazily to avoid hard dependency
        try:
            import aiohttp
            self._aiohttp_available = True
        except ImportError:
            logger.warning("aiohttp package not available, async methods will not work")
            self._aiohttp_available = False
    
    async def run_test(self, 
                      model_name: str, 
                      hardware: List[str] = ["cpu"], 
                      test_type: str = "basic",
                      timeout: int = 300,
                      save_results: bool = True) -> Dict[str, Any]:
        """
        Run a test through the API asynchronously.
        
        Args:
            model_name: Name of the model to test
            hardware: List of hardware platforms to test on
            test_type: Type of test to run (basic, comprehensive, fault_tolerance)
            timeout: Timeout in seconds
            save_results: Whether to save results to disk
            
        Returns:
            Dict containing the API response
            
        Raises:
            RuntimeError: If aiohttp package is not available
            ConnectionError: If the API server cannot be reached
            Exception: For other API errors
        """
        if not self._aiohttp_available:
            raise RuntimeError("aiohttp package not available")
        
        import aiohttp
        
        # Prepare request data
        request_data = {
            "model_name": model_name,
            "hardware": hardware,
            "test_type": test_type,
            "timeout": timeout,
            "save_results": save_results
        }
        
        # Send request
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/test/run",
                    json=request_data,
                    headers=headers
                ) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        raise Exception(f"API error: {response.status} - {error_text}")
                    
                    return await response.json()
        except Exception as e:
            logger.error(f"Error running test: {e}")
            raise
    
    async def get_test_status(self, run_id: str) -> Dict[str, Any]:
        """
        Get the status of a test run asynchronously.
        
        Args:
            run_id: ID of the test run
            
        Returns:
            Dict containing the test status
            
        Raises:
            RuntimeError: If aiohttp package is not available
            ConnectionError: If the API server cannot be reached
            Exception: For other API errors
        """
        if not self._aiohttp_available:
            raise RuntimeError("aiohttp package not available")
        
        import aiohttp
        
        try:
            headers = {"Accept": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/test/status/{run_id}",
                    headers=headers
                ) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        raise Exception(f"API error: {response.status} - {error_text}")
                    
                    return await response.json()
        except Exception as e:
            logger.error(f"Error getting test status: {e}")
            raise
    
    async def get_test_results(self, run_id: str) -> Dict[str, Any]:
        """
        Get the results of a test run asynchronously.
        
        Args:
            run_id: ID of the test run
            
        Returns:
            Dict containing the test results
            
        Raises:
            RuntimeError: If aiohttp package is not available
            ConnectionError: If the API server cannot be reached
            Exception: For other API errors
        """
        if not self._aiohttp_available:
            raise RuntimeError("aiohttp package not available")
        
        import aiohttp
        
        try:
            headers = {"Accept": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/test/results/{run_id}",
                    headers=headers
                ) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        raise Exception(f"API error: {response.status} - {error_text}")
                    
                    return await response.json()
        except Exception as e:
            logger.error(f"Error getting test results: {e}")
            raise
    
    async def connect_websocket(self, run_id: str) -> None:
        """
        Connect to the WebSocket for real-time updates.
        
        Args:
            run_id: ID of the test run
            
        Raises:
            RuntimeError: If aiohttp package is not available
            ConnectionError: If the WebSocket connection fails
            Exception: For other errors
        """
        if not self._aiohttp_available:
            raise RuntimeError("aiohttp package not available")
        
        import aiohttp
        
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            session = aiohttp.ClientSession()
            self._websocket = await session.ws_connect(
                f"{self.base_url}/api/test/ws/{run_id}",
                headers=headers
            )
            logger.info(f"WebSocket connected for test {run_id}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            raise
    
    async def receive_updates(self, callback=None):
        """
        Receive updates from the WebSocket connection.
        
        Args:
            callback: Optional callback function to process updates
            
        Yields:
            Dict containing updates from the WebSocket
            
        Raises:
            RuntimeError: If WebSocket is not connected
            Exception: For WebSocket errors
        """
        if not self._websocket:
            raise RuntimeError("WebSocket not connected")
        
        try:
            async for message in self._websocket:
                if message.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(message.data)
                    
                    # Call callback if provided
                    if callback:
                        callback(data)
                    
                    yield data
                    
                elif message.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket connection closed")
                    break
                    
                elif message.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {message.data}")
                    break
        except Exception as e:
            logger.error(f"Error receiving WebSocket updates: {e}")
            raise
        finally:
            await self.close_websocket()
    
    async def close_websocket(self):
        """Close the WebSocket connection."""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
    
    async def monitor_test_ws(self, run_id: str, callback=None) -> Dict[str, Any]:
        """
        Monitor a test run using WebSockets until completion.
        
        Args:
            run_id: ID of the test run
            callback: Optional callback function for updates
            
        Returns:
            Dict containing the final test results
            
        Raises:
            RuntimeError: If aiohttp package is not available
            Exception: For connection errors
        """
        if not self._aiohttp_available:
            raise RuntimeError("aiohttp package not available")
        
        try:
            # Connect to WebSocket
            await self.connect_websocket(run_id)
            
            # Get updates until completion
            async for update in self.receive_updates(callback):
                if update["status"] in ["completed", "failed"]:
                    # Get full results
                    return await self.get_test_results(run_id)
        except Exception as e:
            logger.error(f"Error monitoring test via WebSocket: {e}")
            raise
        finally:
            await self.close_websocket()

# Example usage
async def example_async():
    """Example of using the async API client."""
    client = AsyncApiClient()
    
    # Run a test
    run_response = await client.run_test("bert-base-uncased")
    run_id = run_response["run_id"]
    
    # Define update callback
    def update_callback(data):
        print(f"Progress: {data['progress']:.1%} - {data['current_step']}")
    
    # Monitor via WebSocket
    results = await client.monitor_test_ws(run_id, callback=update_callback)
    
    print("Test results:", json.dumps(results, indent=2))

def example_sync():
    """Example of using the synchronous API client."""
    client = ApiClient()
    
    # Run a test
    run_response = client.run_test("bert-base-uncased")
    run_id = run_response["run_id"]
    
    # Monitor until completion
    results = client.monitor_test(run_id)
    
    print("Test results:", json.dumps(results, indent=2))
    
    # Clean up
    client.close()

if __name__ == "__main__":
    # Run synchronous example
    print("Running synchronous example:")
    example_sync()
    
    # Run async example
    print("\nRunning asynchronous example:")
    asyncio.run(example_async())