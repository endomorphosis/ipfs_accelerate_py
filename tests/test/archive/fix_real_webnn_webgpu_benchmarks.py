#!/usr/bin/env python3
"""
Fix Real WebNN/WebGPU Benchmarks

This script resolves issues with real WebNN/WebGPU benchmarks by:
1. Setting up proper environment variables to enforce real implementations
2. Fixing WebSocket connection issues in the browser bridge
3. Resolving browser compatibility issues
4. Integrating Firefox-specific optimizations for audio models
5. Implementing proper detection of real vs simulated implementations

Usage:
    python fix_real_webnn_webgpu_benchmarks.py --browser chrome --platform webgpu
    python fix_real_webnn_webgpu_benchmarks.py --browser edge --platform webnn
    python fix_real_webnn_webgpu_benchmarks.py --model whisper --browser firefox --optimize-audio
"""

import os
import sys
import time
import json
import logging
import argparse
import tempfile
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import selenium
    HAS_SELENIUM = True
except ImportError:
    logger.warning("Selenium not installed. Run: pip install selenium")
    HAS_SELENIUM = False

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    logger.warning("Websockets not installed. Run: pip install websockets")
    HAS_WEBSOCKETS = False

# Constants
SUPPORTED_BROWSERS = ["chrome", "firefox", "edge", "safari"]
SUPPORTED_PLATFORMS = ["webnn", "webgpu"]
SUPPORTED_MODELS = ["bert", "t5", "vit", "clip", "whisper", "wav2vec2", "clap", "llama", "all"]

class WebBridgeValidator:
    """Validates WebSocket bridge between Python and browser."""
    
    def __init__(self):
        """Initialize WebBridgeValidator."""
        self.websocket_server_process = None
        self.temp_port = self._find_free_port()
    
    def _find_free_port(self) -> int:
        """Find a free port for WebSocket server."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def check_websocket_library(self) -> bool:
        """Check if websockets library is installed and functioning."""
        if not HAS_WEBSOCKETS:
            logger.error("WebSockets library not installed. Run: pip install websockets")
            return False
            
        try:
            import websockets
            logger.info("WebSockets library is installed and accessible")
            return True
        except ImportError as e:
            logger.error(f"Error importing websockets: {e}")
            return False
    
    def start_test_websocket_server(self) -> bool:
        """Start a test WebSocket server to verify functionality."""
        if not HAS_WEBSOCKETS:
            return False
            
        try:
            # Create a temporary file for the WebSocket server
            fd, path = tempfile.mkstemp(suffix='.py')
            with os.fdopen(fd, 'w') as f:
                f.write(f"""
import asyncio
import websockets
import sys

async def handler(websocket, path):
    await websocket.send('{{"status": "connected"}}')
    try:
        async for message in websocket:
            await websocket.send('{{"echo": ' + message + '}}')
    except Exception as e:
        print(f"Error: {{e}}", file=sys.stderr)

async def main():
    async with websockets.serve(handler, "localhost", {self.temp_port}):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
""")
            
            # Start the WebSocket server as a separate process
            server_process = subprocess.Popen([sys.executable, path], 
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
            self.websocket_server_process = server_process
            logger.info(f"WebSocket test server started on port {self.temp_port}")
            
            # Give it a moment to start
            time.sleep(1)
            
            # Check if it's running
            if server_process.poll() is not None:
                stdout, stderr = server_process.communicate()
                logger.error(f"WebSocket server failed to start: {stderr.decode()}")
                return False
                
            logger.info("WebSocket server successfully started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return False
    
    def stop_test_websocket_server(self) -> None:
        """Stop the test WebSocket server."""
        if self.websocket_server_process:
            logger.info("Stopping WebSocket test server")
            self.websocket_server_process.terminate()
            self.websocket_server_process = None
    
    def run_browser_websocket_test(self, browser: str) -> bool:
        """Test WebSocket connection from browser to server."""
        if not HAS_SELENIUM:
            logger.error("Selenium not installed. Run: pip install selenium")
            return False
            
        if not self.websocket_server_process:
            logger.error("WebSocket server not running")
            return False
            
        try:
            # Create a temporary HTML file for testing
            fd, path = tempfile.mkstemp(suffix='.html')
            with os.fdopen(fd, 'w') as f:
                f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Test</title>
</head>
<body>
    <h1>WebSocket Test</h1>
    <div id="status">Connecting...</div>
    
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            const statusDiv = document.getElementById("status");
            
            try {{
                const socket = new WebSocket("ws://localhost:{self.temp_port}");
                
                socket.onopen = function(e) {{
                    statusDiv.textContent = "Connected!";
                    
                    // Send a test message
                    socket.send(JSON.stringify({{ "message": "Hello from browser!" }}));
                }};
                
                socket.onmessage = function(event) {{
                    statusDiv.textContent = "Received: " + event.data;
                    
                    // Signal success to Selenium
                    document.title = "WebSocket Success";
                }};
                
                socket.onclose = function(event) {{
                    statusDiv.textContent = "Connection closed";
                }};
                
                socket.onerror = function(error) {{
                    statusDiv.textContent = "Error: " + error.message;
                }};
            }} catch (e) {{
                statusDiv.textContent = "Error initializing WebSocket: " + e.message;
            }}
        }});
    </script>
</body>
</html>
""")
            
            # Initialize browser driver
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service as ChromeService
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.firefox.service import Service as FirefoxService
            from selenium.webdriver.firefox.options import Options as FirefoxOptions
            from selenium.webdriver.edge.service import Service as EdgeService
            from selenium.webdriver.edge.options import Options as EdgeOptions
            from selenium.webdriver.safari.service import Service as SafariService
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            driver = None
            
            try:
                # Initialize the appropriate browser
                if browser == "chrome":
                    options = ChromeOptions()
                    options.add_argument("--no-sandbox")
                    options.add_argument("--disable-dev-shm-usage")
                    service = ChromeService()
                    driver = webdriver.Chrome(service=service, options=options)
                    
                elif browser == "firefox":
                    options = FirefoxOptions()
                    service = FirefoxService()
                    driver = webdriver.Firefox(service=service, options=options)
                    
                elif browser == "edge":
                    options = EdgeOptions()
                    options.add_argument("--no-sandbox")
                    options.add_argument("--disable-dev-shm-usage")
                    service = EdgeService()
                    driver = webdriver.Edge(service=service, options=options)
                    
                elif browser == "safari":
                    service = SafariService()
                    driver = webdriver.Safari(service=service)
                
                else:
                    logger.error(f"Unsupported browser: {browser}")
                    return False
                
                # Load the test page
                file_url = f"file://{path}"
                logger.info(f"Opening WebSocket test page in {browser}: {file_url}")
                driver.get(file_url)
                
                # Wait for WebSocket connection to succeed
                try:
                    WebDriverWait(driver, 5).until(EC.title_is("WebSocket Success"))
                    logger.info(f"WebSocket test successful in {browser}!")
                    return True
                except Exception as e:
                    logger.error(f"WebSocket test failed in {browser}: {e}")
                    # Get console logs if possible
                    if browser == "chrome":
                        logs = driver.get_log('browser')
                        for log in logs:
                            logger.error(f"Browser log: {log}")
                    return False
                    
            finally:
                if driver:
                    driver.quit()
                
                # Remove the temporary file
                os.unlink(path)
                
        except Exception as e:
            logger.error(f"Failed to run browser WebSocket test: {e}")
            return False

    def validate_websocket_bridge(self, browser: str) -> bool:
        """Full validation of WebSocket bridge between Python and browser."""
        if not self.check_websocket_library():
            return False
            
        if not self.start_test_websocket_server():
            return False
            
        try:
            return self.run_browser_websocket_test(browser)
        finally:
            self.stop_test_websocket_server()

class RealImplementationValidator:
    """Validates real WebNN/WebGPU implementation vs. simulation."""
    
    def __init__(self, browser: str):
        """Initialize RealImplementationValidator."""
        self.browser = browser
        self.temp_files = []
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Failed to remove temporary file {file_path}: {e}")
    
    def create_detection_script(self, platform: str) -> str:
        """Create a temporary script to detect real implementation."""
        # Create a temporary Python script for real implementation detection
        fd, path = tempfile.mkstemp(suffix='.py')
        with os.fdopen(fd, 'w') as f:
            f.write(f"""
#!/usr/bin/env python3
\"\"\"
Real {platform.upper()} Implementation Detection Script
\"\"\"

import os
import json
import sys
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Force real implementation
os.environ["WEBNN_SIMULATION"] = "0"
os.environ["WEBGPU_SIMULATION"] = "0"
os.environ["USE_BROWSER_AUTOMATION"] = "1"

# Import the implementation modules
sys.path.append(str(Path(__file__).resolve().parent))
try:
    from run_real_webgpu_webnn_fixed import WebImplementation
except ImportError as e:
    logger.error(f"Error importing WebImplementation: {{e}}")
    sys.exit(1)

def main():
    \"\"\"Check if real {platform.upper()} is available.\"\"\"
    impl = WebImplementation(platform="{platform}", browser="{self.browser}", headless=False)
    
    # Start the implementation
    logger.info(f"Starting {platform.upper()} implementation with {{self.browser}} browser")
    start_success = impl.start(allow_simulation=False)
    
    if not start_success:
        logger.error(f"Failed to start {platform.upper()} implementation: Real hardware not available")
        print(json.dumps({{"status": "error", "real": False, "reason": "Implementation failed to start"}}))
        impl.stop()
        return 1
    
    # Detect features
    features = impl.features or {{}}
    
    # Determine if this is a real implementation
    is_real = False
    simulation_status = impl.simulation_mode
    
    if not simulation_status:
        if "{platform}" == "webgpu" and features.get("webgpu", False):
            is_real = True
        elif "{platform}" == "webnn" and features.get("webnn", False):
            is_real = True
    
    # Get adapter info for WebGPU
    adapter_info = None
    if "{platform}" == "webgpu" and features.get("webgpu_adapter", None):
        adapter_info = features.get("webgpu_adapter", None)
    
    # Get backend info for WebNN
    backend_info = None
    if "{platform}" == "webnn" and features.get("webnn_backend", None):
        backend_info = features.get("webnn_backend", None)
    
    # Return detection results
    result = {{
        "status": "success" if start_success else "error",
        "real": is_real,
        "simulation": simulation_status,
        "platform": "{platform}",
        "browser": "{self.browser}",
        "features": features,
        "adapter_info": adapter_info,
        "backend_info": backend_info
    }}
    
    print(json.dumps(result))
    
    # Stop the implementation
    impl.stop()
    return 0 if is_real else 1

if __name__ == "__main__":
    sys.exit(main())
""")
        
        # Make the script executable
        os.chmod(path, 0o755)
        self.temp_files.append(path)
        return path
    
    def check_real_implementation(self, platform: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if real implementation is available."""
        if platform not in SUPPORTED_PLATFORMS:
            logger.error(f"Unsupported platform: {platform}")
            return False, {"status": "error", "reason": f"Unsupported platform: {platform}"}
        
        script_path = self.create_detection_script(platform)
        
        try:
            # Run the detection script
            logger.info(f"Running {platform.upper()} implementation detection script")
            process = subprocess.Popen([sys.executable, script_path], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=60)  # 60 second timeout
            
            # Log stderr if needed
            if stderr:
                logger.debug(f"Detection script stderr: {stderr.decode()}")
            
            # Parse the result
            if process.returncode != 0:
                logger.warning(f"Detection script failed with code {process.returncode}")
                if stdout:
                    try:
                        result = json.loads(stdout.decode().strip())
                        return False, result
                    except json.JSONDecodeError:
                        pass
                return False, {"status": "error", "reason": "Detection script failed"}
            
            # Parse the JSON output
            try:
                result = json.loads(stdout.decode().strip())
                return result.get("real", False), result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse detection script output: {e}")
                return False, {"status": "error", "reason": "Failed to parse detection script output"}
                
        except subprocess.TimeoutExpired:
            logger.error("Detection script timed out")
            return False, {"status": "error", "reason": "Detection script timed out"}
        except Exception as e:
            logger.error(f"Failed to run detection script: {e}")
            return False, {"status": "error", "reason": str(e)}
        finally:
            self.cleanup()

def fix_websocket_bridge():
    """Fix WebSocket bridge between Python and browser."""
    logger.info("Checking WebSocket bridge between Python and browser")
    
    # Check if WebSocket is working properly
    bridge_validator = WebBridgeValidator()
    
    # Check WebSocket library
    if not bridge_validator.check_websocket_library():
        logger.error("WebSocket library check failed")
        print("To fix, run: pip install websockets==11.0.3")
        return False
    
    # Test WebSocket server
    if not bridge_validator.start_test_websocket_server():
        logger.error("WebSocket server test failed")
        print("To fix WebSocket server issues, check for port conflicts or firewall issues")
        return False
    
    # Stop the test server
    bridge_validator.stop_test_websocket_server()
    
    logger.info("WebSocket bridge is working correctly")
    return True

def fix_browser_detection():
    """Fix browser detection and compatibility issues."""
    logger.info("Checking browser detection and compatibility")
    
    if not HAS_SELENIUM:
        logger.error("Selenium not installed. Run: pip install selenium")
        print("To fix browser detection, run: pip install selenium==4.16.0")
        return False
    
    try:
        from selenium import webdriver
        logger.info("Selenium is installed correctly")
        return True
    except Exception as e:
        logger.error(f"Failed to import selenium webdriver: {e}")
        return False

def create_implementation_fix_script(browser: str, platform: str, model: str, optimize_audio: bool) -> str:
    """Create a script to fix real implementation issues."""
    # Create a temporary Python script for implementation fixes
    fd, path = tempfile.mkstemp(suffix='.py')
    
    # Determine model_type based on model name
    model_type = "text"
    if model in ["vit", "clip", "detr"]:
        model_type = "vision"
    elif model in ["whisper", "wav2vec2", "clap"]:
        model_type = "audio"
    elif model in ["llava", "llava_next", "xclip"]:
        model_type = "multimodal"
    
    with os.fdopen(fd, 'w') as f:
        f.write(f"""
#!/usr/bin/env python3
\"\"\"
Fix Real {platform.upper()} Implementation for {browser.capitalize()} Browser with {model.upper()} Model
\"\"\"

import os
import sys
import json
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Force real implementation
os.environ["WEBNN_SIMULATION"] = "0"
os.environ["WEBGPU_SIMULATION"] = "0"
os.environ["USE_BROWSER_AUTOMATION"] = "1"

# Set up specialized flags for Firefox audio optimizations
use_firefox_audio_optimizations = {str(optimize_audio and browser == 'firefox' and model_type == 'audio').lower()}
if use_firefox_audio_optimizations:
    os.environ["USE_FIREFOX_WEBGPU"] = "1"
    os.environ["MOZ_WEBGPU_ADVANCED_COMPUTE"] = "1"
    os.environ["WEBGPU_COMPUTE_SHADERS_ENABLED"] = "1"
    logger.info("Enabled Firefox audio optimizations (256x1x1 workgroup size)")

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent))

# Import the implementation module
try:
    from run_real_webgpu_webnn_fixed import WebImplementation
except ImportError as e:
    logger.error(f"Error importing WebImplementation: {{e}}")
    sys.exit(1)

def main():
    \"\"\"Fix and run real {platform.upper()} implementation.\"\"\"
    # Create implementation
    impl = WebImplementation(platform="{platform}", browser="{browser}", headless=False)
    
    # Set up 4-bit optimization
    impl.set_quantization(bits=4, mixed=True)
    
    # Start implementation
    logger.info(f"Starting real {platform.upper()} implementation with {browser} browser")
    start_time = time.time()
    
    if not impl.start(allow_simulation=False):
        logger.error(f"Failed to start {platform.upper()} implementation: Real hardware not available")
        return 1
    
    startup_time = time.time() - start_time
    logger.info(f"Implementation started in {{startup_time:.2f}} seconds")
    
    # Check if using real implementation
    is_simulation = impl.simulation_mode
    if is_simulation:
        logger.error(f"Running in simulation mode - real {platform.upper()} hardware not detected")
        impl.stop()
        return 1
    
    # Initialize model
    logger.info(f"Initializing model: {model}")
    model_init_time = time.time()
    model_result = impl.init_model("{model}", "{model_type}")
    model_init_time = time.time() - model_init_time
    
    if not model_result or model_result.get("status") != "success":
        logger.error(f"Failed to initialize model: {model}")
        impl.stop()
        return 1
    
    logger.info(f"Model initialized in {{model_init_time:.2f}} seconds")
    
    # Run inference
    logger.info(f"Running inference with model: {model}")
    
    # Create input data based on model type
    if "{model_type}" == "text":
        input_data = "This is a test input for real {platform.upper()} implementation."
    elif "{model_type}" == "vision":
        input_data = {{"image": "test.jpg"}}
    elif "{model_type}" == "audio":
        input_data = {{"audio": "test.mp3"}}
    elif "{model_type}" == "multimodal":
        input_data = {{"image": "test.jpg", "text": "What's in this image?"}}
    
    inference_time = time.time()
    inference_result = impl.run_inference("{model}", input_data)
    inference_time = time.time() - inference_time
    
    if not inference_result or inference_result.get("status") != "success":
        logger.error(f"Failed to run inference with model: {model}")
        impl.stop()
        return 1
    
    logger.info(f"Inference completed in {{inference_time:.2f}} seconds")
    
    # Output performance metrics
    metrics = inference_result.get("performance_metrics", {{}})
    implementation_type = inference_result.get("implementation_type", "UNKNOWN")
    is_real = inference_result.get("is_real_implementation", False) and not inference_result.get("is_simulation", True)
    
    print("=" * 80)
    print(f"REAL {platform.upper()} BENCHMARK RESULTS ({browser.upper()} - {model.upper()})")
    print("=" * 80)
    print(f"Implementation: {'REAL HARDWARE' if is_real else 'SIMULATION'}")
    print(f"Implementation Type: {implementation_type}")
    print(f"Latency: {metrics.get('inference_time_ms', 'N/A')} ms")
    print(f"Throughput: {metrics.get('throughput_items_per_sec', 'N/A')} items/sec")
    print(f"Memory Usage: {metrics.get('memory_usage_mb', 'N/A')} MB")
    if use_firefox_audio_optimizations:
        print("Firefox Audio Optimizations: ENABLED (256x1x1 workgroup size)")
    print(f"Model Init Time: {model_init_time:.2f} seconds")
    print(f"Inference Time: {inference_time:.2f} seconds")
    print(f"Startup Time: {startup_time:.2f} seconds")
    print("=" * 80)
    
    # Save results to file
    result_obj = {{
        "platform": "{platform}",
        "browser": "{browser}",
        "model": "{model}",
        "model_type": "{model_type}",
        "is_real": is_real,
        "simulation": not is_real,
        "implementation_type": implementation_type,
        "performance_metrics": metrics,
        "model_init_time": model_init_time,
        "inference_time": inference_time,
        "startup_time": startup_time,
        "firefox_audio_optimizations": use_firefox_audio_optimizations
    }}
    
    with open(f"{platform}_{browser}_{model}_benchmark_results.json", "w") as f:
        json.dump(result_obj, f, indent=2)
    
    # Stop implementation
    impl.stop()
    
    return 0 if is_real else 1

if __name__ == "__main__":
    sys.exit(main())
""")
    
    # Make the script executable
    os.chmod(path, 0o755)
    return path

def fix_real_implementation(browser: str, platform: str, model: str, optimize_audio: bool = False) -> bool:
    """Fix real implementation issues for WebNN/WebGPU."""
    logger.info(f"Fixing real {platform.upper()} implementation for {browser} browser with {model} model")
    
    # Validate web socket bridge
    if not fix_websocket_bridge():
        logger.error("WebSocket bridge validation failed")
        return False
    
    # Validate browser detection
    if not fix_browser_detection():
        logger.error("Browser detection validation failed")
        return False
    
    # Validate real implementation
    validator = RealImplementationValidator(browser)
    is_real, details = validator.check_real_implementation(platform)
    
    if not is_real:
        logger.warning(f"Real {platform.upper()} implementation not detected on {browser}")
        if details.get("simulation", True):
            logger.error("Running in simulation mode - real hardware not detected")
            logger.error("This could be due to:")
            logger.error("  1. Missing hardware support (GPU doesn't support WebGPU)")
            logger.error("  2. Browser doesn't support the required features")
            logger.error("  3. Browser flags not properly set")
            
            # Browser-specific recommendations
            if browser == "chrome":
                logger.info("For Chrome, ensure WebGPU is enabled:")
                logger.info("  Run with: --enable-features=WebGPU,WebGPUDeveloperFeatures")
            elif browser == "firefox":
                logger.info("For Firefox, ensure WebGPU is enabled:")
                logger.info("  Set dom.webgpu.enabled=true in about:config")
            elif browser == "edge":
                logger.info("For Edge, ensure WebGPU is enabled:")
                logger.info("  Run with: --enable-features=WebGPU,WebGPUDeveloperFeatures")
            
            if platform == "webnn":
                logger.info("For WebNN support:")
                logger.info("  Edge has the best WebNN support")
                logger.info("  Set --enable-features=WebNN in browser flags")
        
        # Create and run the implementation fix script anyway
        fix_script = create_implementation_fix_script(browser, platform, model, optimize_audio)
        
        logger.info(f"Created implementation fix script: {fix_script}")
        logger.info(f"Run the script to apply fixes:")
        print(f"\npython {fix_script}\n")
        
        logger.warning("Running in SIMULATION mode - real hardware not detected")
        return False
    
    logger.info(f"Real {platform.upper()} implementation detected on {browser}!")
    
    # Create implementation fix script
    fix_script = create_implementation_fix_script(browser, platform, model, optimize_audio)
    
    logger.info(f"Created implementation fix script: {fix_script}")
    logger.info(f"Run the script to use real implementation:")
    print(f"\npython {fix_script}\n")
    
    return True

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Fix real WebNN/WebGPU benchmark issues")
    
    # Browser options
    parser.add_argument("--browser", choices=SUPPORTED_BROWSERS, default="chrome",
                        help="Browser to use for testing")
    
    # Platform options
    parser.add_argument("--platform", choices=SUPPORTED_PLATFORMS, default="webgpu",
                        help="Platform to test (webnn or webgpu)")
    
    # Model options
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="bert",
                        help="Model to test")
    
    # Optimization options
    parser.add_argument("--optimize-audio", action="store_true",
                        help="Enable audio optimizations for Firefox (256x1x1 workgroup size)")
    
    # Validation options
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate, don't create fix scripts")
    
    # Debug options
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check requirements
    if not HAS_SELENIUM:
        logger.error("Selenium not installed. Run: pip install selenium")
        return 1
    
    if not HAS_WEBSOCKETS:
        logger.error("Websockets not installed. Run: pip install websockets")
        return 1
    
    # Fix real implementation
    if args.validate_only:
        validator = RealImplementationValidator(args.browser)
        is_real, details = validator.check_real_implementation(args.platform)
        
        print("=" * 80)
        print(f"REAL {args.platform.upper()} IMPLEMENTATION VALIDATION ({args.browser.upper()})")
        print("=" * 80)
        print(f"Real Implementation: {'YES' if is_real else 'NO'}")
        print(f"Simulation Mode: {'NO' if is_real else 'YES'}")
        
        if args.platform == "webgpu" and details.get("adapter_info"):
            adapter = details.get("adapter_info")
            print(f"WebGPU Adapter: {adapter.get('vendor', 'Unknown')} - {adapter.get('architecture', 'Unknown')}")
        
        if args.platform == "webnn" and details.get("backend_info"):
            backend = details.get("backend_info")
            print(f"WebNN Backend: {backend}")
        
        print("=" * 80)
        
        return 0 if is_real else 1
    else:
        success = fix_real_implementation(args.browser, args.platform, args.model, args.optimize_audio)
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())