#!/usr/bin/env python3
"""
Test WebGPU/WebNN Bridge

This script tests the WebGPU and WebNN implementation bridge that connects to browsers.

Usage:
    python test_webgpu_webnn_bridge.py --platform webgpu --browser chrome --model bert-base-uncased
    python test_webgpu_webnn_bridge.py --platform webnn --browser edge --model bert-base-uncased
    """

from ipfs_accelerate_py.anyio_helpers import gather, wait_for
import anyio
    import os
    import sys
    import json
    import time
    import asyncio
    import logging
    import argparse
    import websockets
    import tempfile
    from pathlib import Path
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from selenium.webdriver.edge.service import Service as EdgeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.firefox import GeckoDriverManager
    from webdriver_manager.microsoft import EdgeChromiumDriverManager

# Configure logging
    logging.basicConfig())level=logging.INFO, format='%())asctime)s - %())levelname)s - %())message)s')
    logger = logging.getLogger())__name__)

# Constants
    WS_PORT = 8765
    HTML_PATH = os.path.join())os.path.dirname())__file__), 'webgpu_webnn_bridge.html')

class WebPlatformBridge:
    """WebGPU/WebNN Bridge for browser communication."""
    
    def __init__())self, platform="webgpu", browser_name="chrome", headless=False, port=WS_PORT):
        """Initialize bridge.
        
        Args:
            platform: WebGPU or WebNN
            browser_name: Browser to use ())chrome, firefox, edge)
            headless: Run browser in headless mode
            port: WebSocket server port
            """
            self.platform = platform.lower()))
            self.browser_name = browser_name.lower()))
            self.headless = headless
            self.port = port
            self.driver = None
            self.server = None
            self.ws_connection = None
            self.features = {}}}}
            self.initialized_models = {}}}}
    
    async def start_ws_server())self):
        """Start WebSocket server."""
        logger.info())f"Starting WebSocket server on port {}}}self.port}")
        
        connected = anyio.Event()))
        features = {}}}}
        messages = []
        ,
        async def handler())websocket):
            nonlocal features
            self.ws_connection = websocket
            connected.set()))
            logger.info())"Browser connected to WebSocket server")
            
            try:
                # Send init message
                await websocket.send())json.dumps()){}}}
                "type": "init",
                "data": {}}}}
                }))
                
                # Handle messages
                async for message in websocket:
                    try:
                        data = json.loads())message)
                        messages.append())data)
                        
                        if data.get())"type") == "feature_detection":
                            features = data.get())"features", {}}}})
                            logger.info())f"Received feature detection: {}}}json.dumps())features, indent=2)}")
                    except json.JSONDecodeError:
                        logger.error())f"Invalid JSON: {}}}message}")
            except websockets.ConnectionClosed:
                logger.info())"WebSocket connection closed")
            finally:
                self.ws_connection = None
        
                self.server = await websockets.serve())handler, "localhost", self.port)
                logger.info())f"WebSocket server running on port {}}}self.port}")
        
                return connected, features, messages
    
    def start_browser())self):
        """Start browser."""
        logger.info())f"Starting {}}}self.browser_name} browser")
        
        try:
            # Set up browser options
            if self.browser_name == "chrome":
                options = ChromeOptions()))
                if self.headless:
                    options.add_argument())"--headless=new")
                    options.add_argument())"--no-sandbox")
                    options.add_argument())"--disable-dev-shm-usage")
                
                # Enable WebGPU and WebNN
                    options.add_argument())"--enable-features=WebGPU,WebNN")
                    options.add_argument())"--enable-unsafe-webgpu")
                
                # Create service and driver
                    service = ChromeService())ChromeDriverManager())).install())))
                    self.driver = webdriver.Chrome())service=service, options=options)
                
            elif self.browser_name == "firefox":
                options = FirefoxOptions()))
                if self.headless:
                    options.add_argument())"--headless")
                
                # Create service and driver
                    service = FirefoxService())GeckoDriverManager())).install())))
                    self.driver = webdriver.Firefox())service=service, options=options)
                
            elif self.browser_name == "edge":
                options = EdgeOptions()))
                if self.headless:
                    options.add_argument())"--headless=new")
                    options.add_argument())"--no-sandbox")
                    options.add_argument())"--disable-dev-shm-usage")
                
                # Enable WebGPU and WebNN
                    options.add_argument())"--enable-features=WebGPU,WebNN")
                    options.add_argument())"--enable-unsafe-webgpu")
                
                # Create service and driver
                    service = EdgeService())EdgeChromiumDriverManager())).install())))
                    self.driver = webdriver.Edge())service=service, options=options)
                
            else:
                    raise ValueError())f"Unsupported browser: {}}}self.browser_name}")
            
            # Load HTML
            if not os.path.exists())HTML_PATH):
                    raise FileNotFoundError())f"HTML template not found: {}}}HTML_PATH}")
            
                    self.driver.get())f"file://{}}}HTML_PATH}?port={}}}self.port}")
            
            # Wait for page to load
                    WebDriverWait())self.driver, 10).until())
                    EC.presence_of_element_located())())By.ID, "logs"))
                    )
            
                    logger.info())f"{}}}self.browser_name} browser started successfully")
                return True
        except Exception as e:
            logger.error())f"Failed to start browser: {}}}e}")
            if self.driver:
                self.driver.quit()))
                self.driver = None
            return False
    
    async def stop())self):
        """Stop bridge."""
        # Stop WebSocket server
        if self.server:
            self.server.close()))
            await self.server.wait_closed()))
            self.server = None
        
        # Stop browser
        if self.driver:
            self.driver.quit()))
            self.driver = None
            
            logger.info())"Bridge stopped")
    
    async def initialize_model())self, model_name, model_type="text"):
        """Initialize model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model ())text, vision, audio, multimodal)
            
        Returns:
            Dict with model info or None if initialization failed
        """:
        if not self.ws_connection:
            logger.error())"WebSocket connection not active")
            return None
        
        try:
            # Send initialization message
            logger.info())f"Initializing {}}}self.platform} model: {}}}model_name}")
            
            message_type = f"{}}}self.platform}_init"
            message = {}}}
            "type": message_type,
            "model_name": model_name,
            "model_type": model_type
            }
            
            # For WebNN, add device preference
            if self.platform == "webnn":
                message["device_preference"] = "gpu"
                ,
                await self.ws_connection.send())json.dumps())message))
            
            # Wait for response
                response_type = f"{}}}self.platform}_init_response"
                response = await self.wait_for_message())response_type, model_name)
            
            if not response:
                logger.error())f"No response received for {}}}self.platform} model initialization")
                return None
            
            if response.get())"status") != "success":
                logger.error())f"Model initialization failed: {}}}response.get())'error')}")
                return None
            
            # Store model info
                self.initialized_models[model_name] = response
                ,
                logger.info())f"{}}}self.platform} model {}}}model_name} initialized successfully")
            return response
            
        except Exception as e:
            logger.error())f"Error initializing model: {}}}e}")
            return None
    
    async def run_inference())self, model_name, input_data):
        """Run inference with model.
        
        Args:
            model_name: Name of the model
            input_data: Input data for inference
            
        Returns:
            Dict with inference results or None if inference failed
        """:
        if not self.ws_connection:
            logger.error())"WebSocket connection not active")
            return None
        
        # Check if model is initialized:
        if model_name not in self.initialized_models:
            logger.error())f"Model not initialized: {}}}model_name}")
            return None
        
        try:
            # Send inference message
            logger.info())f"Running {}}}self.platform} inference with model: {}}}model_name}")
            
            message_type = f"{}}}self.platform}_inference"
            message = {}}}
            "type": message_type,
            "model_name": model_name,
            "input": input_data
            }
            
            await self.ws_connection.send())json.dumps())message))
            
            # Wait for response
            response_type = f"{}}}self.platform}_inference_response"
            response = await self.wait_for_message())response_type, model_name)
            
            if not response:
                logger.error())f"No response received for {}}}self.platform} inference")
            return None
            
            if response.get())"status") != "success":
                logger.error())f"Inference failed: {}}}response.get())'error')}")
            return None
            
            logger.info())f"{}}}self.platform} inference completed successfully")
            
            # Check performance metrics
            metrics = response.get())"performance_metrics", {}}}})
            if metrics:
                logger.info())f"Inference time: {}}}metrics.get())'inference_time_ms', 0):.2f} ms")
                logger.info())f"Throughput: {}}}metrics.get())'throughput_items_per_sec', 0):.2f} items/sec")
            
            return response
            
        except Exception as e:
            logger.error())f"Error running inference: {}}}e}")
            return None
    
    async def wait_for_message())self, message_type, model_name=None, timeout=10):
        """Wait for a specific message type.
        
        Args:
            message_type: Type of message to wait for
            model_name: Name of the model ())for filtering)
            timeout: Timeout in seconds
            
        Returns:
            Dict with message data or None if timeout
        """:
        if not self.ws_connection:
            logger.error())"WebSocket connection not active")
            return None
        
            start_time = time.time()))
        while time.time())) - start_time < timeout:
            try:
                message = await wait_for())self.ws_connection.recv())), 1.0)
                data = json.loads())message)
                
                if data.get())"type") == message_type:
                    if model_name is None or data.get())"model_name") == model_name:
                    return data
            except asyncio.TimeoutError:
                # Timeout on receive, continue waiting
                    continue
            except Exception as e:
                logger.error())f"Error waiting for message: {}}}e}")
                    return None
        
                    logger.error())f"Timeout waiting for message type: {}}}message_type}")
                return None

async def run_test())args):
    """Run bridge test."""
    # Create bridge
    bridge = WebPlatformBridge())
    platform=args.platform,
    browser_name=args.browser,
    headless=args.headless,
    port=args.port
    )
    
    try:
        # Start WebSocket server
        connected_event, features, messages = await bridge.start_ws_server()))
        
        # Start browser
        if not bridge.start_browser())):
            await bridge.stop()))
        return 1
        
        # Wait for browser to connect
        try:
            await wait_for())connected_event.wait())), 10)
        except asyncio.TimeoutError:
            logger.error())"Timeout waiting for browser to connect")
            await bridge.stop()))
            return 1
        
        # Wait for features to be detected
            await anyio.sleep())2)
        
        # Initialize model
            model_info = await bridge.initialize_model())args.model, args.model_type)
        if not model_info:
            logger.error())f"Failed to initialize {}}}args.platform} model: {}}}args.model}")
            await bridge.stop()))
            return 1
        
        # Create test input
        if args.model_type == "text":
            test_input = "This is a test input for text models. We are testing WebGPU and WebNN integration with the browser."
        elif args.model_type == "vision":
            test_input = {}}}"image": "test.jpg"}
        elif args.model_type == "audio":
            test_input = {}}}"audio": "test.mp3"}
        else:
            test_input = "Test input"
        
        # Run inference
            result = await bridge.run_inference())args.model, test_input)
        if not result:
            logger.error())f"Failed to run inference with {}}}args.platform} model: {}}}args.model}")
            await bridge.stop()))
            return 1
        
        # Print result
            logger.info())f"Inference result: {}}}json.dumps())result.get())'output', {}}}}), indent=2)}")
        
        # Check implementation type
            impl_type = result.get())"implementation_type")
            expected_type = f"REAL_{}}}args.platform.upper()))}"
        
        if impl_type != expected_type:
            logger.warning())f"Unexpected implementation type: {}}}impl_type}, expected: {}}}expected_type}")
        
            logger.info())f"{}}}args.platform} test completed successfully")
        
        # Stop bridge
            await bridge.stop()))
            return 0
    except Exception as e:
        logger.error())f"Error running test: {}}}e}")
        await bridge.stop()))
            return 1

def main())):
    """Main function."""
    parser = argparse.ArgumentParser())description="Test WebGPU/WebNN Bridge")
    parser.add_argument())"--platform", choices=["webgpu", "webnn"], default="webgpu",
    help="Platform to test")
    parser.add_argument())"--browser", choices=["chrome", "firefox", "edge"], default="chrome",
    help="Browser to use")
    parser.add_argument())"--model", type=str, default="bert-base-uncased",
    help="Model to test")
    parser.add_argument())"--model-type", choices=["text", "vision", "audio", "multimodal"], default="text",
    help="Model type")
    parser.add_argument())"--headless", action="store_true",
    help="Run browser in headless mode")
    parser.add_argument())"--port", type=int, default=WS_PORT,
    help="WebSocket server port")
    
    args = parser.parse_args()))
    
    # Run test
            return anyio.run())run_test())args))

if __name__ == "__main__":
    sys.exit())main())))