#!/usr/bin/env python3
"""
Real Browser Test for Selenium Browser Bridge

This script performs a simple test to verify that the Selenium Browser Bridge
can launch and interact with a real browser. It checks WebGPU and WebNN capabilities
and demonstrates the browser recovery mechanisms.

Usage:
    python run_real_browser_test.py [--browser chrome] [--model bert-base-uncased] [--platform webgpu]
"""

import os
import sys
import time
import json
import anyio
import argparse
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("real_browser_test")

# Set more verbose logging if environment variable is set
if os.environ.get("SELENIUM_BRIDGE_LOG_LEVEL", "").upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)

# Import the browser bridge
try:
    from selenium_browser_bridge import (
        BrowserConfiguration, SeleniumBrowserBridge, SELENIUM_AVAILABLE
    )
except ImportError:
    logger.error("Error importing selenium_browser_bridge. Make sure it exists at the expected path.")
    SELENIUM_AVAILABLE = False

# Import recovery strategies
try:
    from browser_recovery_strategies import (
        BrowserType, ModelType, detect_browser_type, detect_model_type
    )
    RECOVERY_AVAILABLE = True
except ImportError:
    logger.error("Error importing browser_recovery_strategies. Make sure it exists at the expected path.")
    RECOVERY_AVAILABLE = False

class RealBrowserTest:
    """Simple test for real browser integration with Selenium."""
    
    def __init__(self, browser_name: str = "chrome", model_name: str = "bert-base-uncased",
                 platform: str = "webgpu", headless: bool = False, allow_simulation: bool = True):
        """
        Initialize the real browser test.
        
        Args:
            browser_name: Browser name (chrome, firefox, edge)
            model_name: Model name to test
            platform: Platform to use (webgpu, webnn)
            headless: Whether to run in headless mode
            allow_simulation: Whether to allow simulation fallback
        """
        self.browser_name = browser_name
        self.model_name = model_name
        self.platform = platform
        self.headless = headless
        self.allow_simulation = allow_simulation
        
        # Attempt to detect model type if recovery strategies are available
        self.model_type = None
        if RECOVERY_AVAILABLE:
            self.model_type = detect_model_type(model_name)
            self.browser_type = detect_browser_type(browser_name)
        
        # Create browser configuration
        self.config = self._create_browser_config()
        
        logger.info(f"Initialized real browser test with browser={browser_name}, model={model_name}, platform={platform}")
    
    def _create_browser_config(self) -> BrowserConfiguration:
        """
        Create browser configuration based on model and browser type.
        
        Returns:
            Browser configuration
        """
        # Basic configuration
        config = BrowserConfiguration(
            browser_name=self.browser_name,
            platform=self.platform,
            headless=self.headless,
            timeout=30
        )
        
        # Apply optimizations if model type is available
        if self.model_type:
            # Text models
            if self.model_type == ModelType.TEXT:
                if self.browser_type == BrowserType.EDGE and self.platform == "webnn":
                    # Edge is best for WebNN with text models
                    config.optimize_for = "latency"
                    config.max_batch_size = 1
                    logger.info("Applying text model optimizations for Edge with WebNN")
            
            # Audio models
            elif self.model_type == ModelType.AUDIO and self.browser_type == BrowserType.FIREFOX:
                # Firefox works best for audio models with compute shaders
                config.compute_shaders = True
                config.custom_prefs = {"dom.webgpu.workgroup_size": "256,1,1"}
                logger.info("Applying audio model optimizations for Firefox with compute shaders")
            
            # Vision models
            elif self.model_type == ModelType.VISION and self.browser_type == BrowserType.CHROME:
                # Chrome works well for vision models
                config.shader_precompilation = True
                config.custom_args = ["--enable-zero-copy"]
                logger.info("Applying vision model optimizations for Chrome")
            
            # Multimodal models
            elif self.model_type == ModelType.MULTIMODAL:
                # Enable parallel loading for multimodal models
                config.parallel_loading = True
                logger.info("Applying multimodal optimizations with parallel loading")
        
        return config
    
    async def run_test(self) -> Dict[str, Any]:
        """
        Run the real browser test.
        
        Returns:
            Dictionary with test results
        """
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium not available. Cannot run test.")
            return {
                "success": False,
                "error": "Selenium not available",
                "browser": self.browser_name,
                "model": self.model_name,
                "platform": self.platform
            }
        
        # Create browser bridge
        bridge = SeleniumBrowserBridge(self.config)
        
        try:
            # Launch browser
            logger.info(f"Launching {self.browser_name} browser...")
            launch_success = await bridge.launch(allow_simulation=self.allow_simulation)
            
            if not launch_success:
                logger.error(f"Failed to launch {self.browser_name} browser")
                return {
                    "success": False,
                    "error": f"Failed to launch {self.browser_name} browser",
                    "browser": self.browser_name,
                    "model": self.model_name,
                    "platform": self.platform
                }
            
            # Check if we're in simulation mode
            simulation_mode = getattr(bridge, 'simulation_mode', False)
            logger.info(f"Browser launched successfully. Simulation mode: {simulation_mode}")
            
            # Check platform support
            logger.info(f"Checking platform support for {self.platform}...")
            platform_support = await bridge._check_platform_support() 
            logger.info(f"Platform support: {platform_support}")
            
            # Run a simple test
            logger.info(f"Running test with model {self.model_name}...")
            result = await bridge.run_test(
                model_name=self.model_name,
                input_data="This is a test input"
            )
            
            # Add platform support info to result
            result["platform_support"] = platform_support
            result["simulation_mode"] = simulation_mode
            result["browser"] = self.browser_name
            result["model"] = self.model_name
            result["platform"] = self.platform
            
            # Get bridge metrics
            metrics = bridge.get_metrics()
            result["metrics"] = metrics
            
            return result
            
        except Exception as e:
            logger.error(f"Error during test: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "browser": self.browser_name,
                "model": self.model_name,
                "platform": self.platform
            }
            
        finally:
            # Close browser
            if bridge:
                try:
                    logger.info("Closing browser...")
                    await bridge.close()
                except Exception as e:
                    logger.warning(f"Error closing browser: {str(e)}")
    
    def print_test_header(self):
        """Print test header information."""
        print("=" * 80)
        print("Real Browser Test with Selenium Browser Bridge")
        print("=" * 80)
        print(f"Browser:       {self.browser_name}")
        print(f"Model:         {self.model_name}")
        print(f"Platform:      {self.platform}")
        print(f"Headless:      {self.headless}")
        print(f"Allow Simulation: {self.allow_simulation}")
        if self.model_type:
            print(f"Model Type:    {self.model_type.value}")
        print("=" * 80)
    
    def print_test_results(self, results: Dict[str, Any]):
        """
        Print test results.
        
        Args:
            results: Test results dictionary
        """
        print("\n" + "=" * 80)
        print("Test Results")
        print("=" * 80)
        
        # Basic info
        success = results.get("success", False)
        if success:
            print("✅ Test Succeeded")
        else:
            print("❌ Test Failed")
            if "error" in results:
                print(f"Error: {results['error']}")
        
        # Platform support
        platform_support = results.get("platform_support", {})
        if platform_support:
            print("\nPlatform Support:")
            for platform, supported in platform_support.items():
                status = "✅ Supported" if supported else "❌ Not Supported"
                print(f"  {platform}: {status}")
        
        # Simulation mode
        simulation_mode = results.get("simulation_mode", False)
        print(f"\nSimulation Mode: {'✅ Enabled' if simulation_mode else '❌ Disabled'}")
        
        # Execution info
        if "execution_time_ms" in results:
            print(f"Execution Time: {results['execution_time_ms']:.2f} ms")
        
        if "result" in results:
            print(f"Result: {results['result']}")
        
        # Model info
        if "model_type" in results:
            print(f"Detected Model Type: {results['model_type']}")
        
        print("=" * 80)
        
        # Print metrics if available
        metrics = results.get("metrics", {})
        if metrics:
            print("\nBrowser Metrics:")
            print(f"Script Executions: {metrics.get('script_executions', 0)}")
            
            # Print circuit breaker metrics if available
            circuit_metrics = metrics.get("circuit_breaker", {})
            if circuit_metrics:
                print("\nCircuit Breaker Metrics:")
                for key, value in circuit_metrics.items():
                    print(f"  {key}: {value}")
            
            print("=" * 80)
        

async def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test real browser integration with Selenium Browser Bridge")
    parser.add_argument("--browser", default="chrome", choices=["chrome", "firefox", "edge"], 
                        help="Browser to test (chrome, firefox, edge)")
    parser.add_argument("--model", default="bert-base-uncased", 
                        help="Model name to test")
    parser.add_argument("--platform", default="webgpu", choices=["webgpu", "webnn"], 
                        help="Platform to test (webgpu, webnn)")
    parser.add_argument("--no-headless", action="store_true", 
                        help="Run browser in visible mode (not headless)")
    parser.add_argument("--no-simulation", action="store_true", 
                        help="Disable simulation fallback")
    parser.add_argument("--save-results", type=str, 
                        help="Save results to JSON file")
    args = parser.parse_args()
    
    # Create and run test
    test = RealBrowserTest(
        browser_name=args.browser,
        model_name=args.model,
        platform=args.platform,
        headless=not args.no_headless,
        allow_simulation=not args.no_simulation
    )
    
    # Print test header
    test.print_test_header()
    
    # Run test
    results = await test.run_test()
    
    # Print results
    test.print_test_results(results)
    
    # Save results if requested
    if args.save_results:
        try:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.save_results}")
        except Exception as e:
            print(f"\nError saving results: {str(e)}")
    
    # Return exit code
    return 0 if results.get("success", False) else 1

if __name__ == "__main__":
    exit_code = anyio.run(main())
    sys.exit(exit_code)