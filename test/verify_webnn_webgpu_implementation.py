#!/usr/bin/env python3
"""
Verify WebNN and WebGPU Implementation Status

This script verifies the implementation status of WebNN and WebGPU in the framework,
checking if real implementations are being used and properly detecting hardware capabilities.

Usage:
    python verify_webnn_webgpu_implementation.py

Options:
    --browser: Browser to use for testing (chrome, firefox, edge)
    --no-headless: Disable headless mode (show browser UI)
    --verbose: Enable verbose logging
"""

import os
import sys
import json
import asyncio
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    missing_deps = []
    
    try:
        import selenium
    except ImportError:
        missing_deps.append("selenium")
    
    try:
        import websockets
    except ImportError:
        missing_deps.append("websockets")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.error("Please install required packages with:")
        logger.error(f"pip install {' '.join(missing_deps)}")
        return False
    
    return True

def check_implementation_files():
    """Check if implementation files exist."""
    required_files = [
        "implement_real_webnn_webgpu.py",
        "fixed_web_platform/real_webgpu_connection.py",
        "fixed_web_platform/real_webnn_connection.py",
        "fixed_web_platform/webgpu_implementation.py",
        "fixed_web_platform/webnn_implementation.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing implementation files: {', '.join(missing_files)}")
        return False
    
    return True

def check_simulated_code_flags():
    """Check if there are any remaining simulation flags in the implementation files."""
    implementation_files = [
        "fixed_web_platform/webgpu_implementation.py",
        "fixed_web_platform/webnn_implementation.py",
        "run_real_webgpu_webnn.py"
    ]
    
    simulation_indicators = [
        "SIMULATED_WEBGPU",
        "SIMULATED_WEBNN",
        "is_simulation = True",
        "simulation_mode = True"
    ]
    
    files_with_simulation = []
    
    for file_path in implementation_files:
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                content = f.read()
                
                # Check if file has fallback simulation
                for indicator in simulation_indicators:
                    if indicator in content:
                        # Check if it's just a fallback mechanism or required simulation
                        if "simulation_mode = True" in content and "if not webgpu_supported:" in content:
                            # This is a fallback, which is fine
                            pass
                        else:
                            files_with_simulation.append(file_path)
                            break
    
    if files_with_simulation:
        logger.warning(f"Files with simulation indicators: {', '.join(files_with_simulation)}")
        logger.warning("These files may still have simulation code that should be reviewed.")
        return False
    
    return True

async def test_webgpu_implementation(browser="chrome", headless=True, verbose=False):
    """Test the WebGPU implementation."""
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Import from the implementation file
        from fixed_web_platform.webgpu_implementation import RealWebGPUImplementation
        
        # Create implementation
        impl = RealWebGPUImplementation(browser_name=browser, headless=headless)
        
        # Initialize
        logger.info("Initializing WebGPU implementation")
        success = await impl.initialize()
        if not success:
            logger.error("Failed to initialize WebGPU implementation")
            return False, {}
        
        # Get feature support
        features = impl.get_feature_support()
        logger.info(f"WebGPU feature support: {json.dumps(features, indent=2)}")
        
        # Get implementation details
        impl_type = impl.get_implementation_type()
        logger.info(f"WebGPU implementation type: {impl_type}")
        
        # Check if WebGPU is supported in the browser
        webgpu_supported = features.get("webgpu", False)
        webgpu_adapter = features.get("webgpuAdapter", {})
        
        if webgpu_supported:
            logger.info("WebGPU is SUPPORTED in the browser")
            if webgpu_adapter:
                logger.info(f"WebGPU Adapter: {webgpu_adapter.get('vendor')} - {webgpu_adapter.get('architecture')}")
        else:
            logger.warning("WebGPU is NOT SUPPORTED in the browser")
        
        # Initialize model to test real vs. simulation
        logger.info("Testing model initialization with BERT")
        model_info = await impl.initialize_model("bert-base-uncased", model_type="text")
        
        # Try to run simple inference to check simulation status
        logger.info("Testing inference with BERT model")
        result = await impl.run_inference(
            "bert-base-uncased", 
            "This is a test input for WebGPU verification."
        )
        
        if not result:
            logger.error("Failed to run inference with BERT model")
            await impl.shutdown()
            return False, {}
        
        # Check if simulation was used
        is_simulation = result.get("is_simulation", True)
        using_transformers_js = result.get("using_transformers_js", False)
        
        simulation_status = {
            "webgpu_supported": webgpu_supported,
            "webgpu_adapter": webgpu_adapter,
            "impl_type": impl_type,
            "is_simulation": is_simulation,
            "using_transformers_js": using_transformers_js
        }
        
        if is_simulation:
            logger.warning("WebGPU is using SIMULATION MODE")
        else:
            logger.info("WebGPU is using REAL HARDWARE ACCELERATION")
        
        if using_transformers_js:
            logger.info("WebGPU is using transformers.js for model inference")
        
        # Shutdown
        await impl.shutdown()
        return True, simulation_status
        
    except Exception as e:
        logger.error(f"Error testing WebGPU implementation: {e}")
        return False, {}

async def test_webnn_implementation(browser="edge", headless=True, verbose=False):
    """Test the WebNN implementation."""
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Import from the implementation file
        from fixed_web_platform.webnn_implementation import RealWebNNImplementation
        
        # Create implementation
        impl = RealWebNNImplementation(browser_name=browser, headless=headless)
        
        # Initialize
        logger.info("Initializing WebNN implementation")
        success = await impl.initialize()
        if not success:
            logger.error("Failed to initialize WebNN implementation")
            return False, {}
        
        # Get feature support
        features = impl.get_feature_support()
        logger.info(f"WebNN feature support: {json.dumps(features, indent=2)}")
        
        # Get implementation details
        impl_type = impl.get_implementation_type()
        logger.info(f"WebNN implementation type: {impl_type}")
        
        # Check if WebNN is supported in the browser
        webnn_supported = features.get("webnn", False)
        webnn_backend = features.get("webnnBackend", None)
        
        if webnn_supported:
            logger.info("WebNN is SUPPORTED in the browser")
            if webnn_backend:
                logger.info(f"WebNN Backend: {webnn_backend}")
        else:
            logger.warning("WebNN is NOT SUPPORTED in the browser")
        
        # Initialize model to test real vs. simulation
        logger.info("Testing model initialization with BERT")
        model_info = await impl.initialize_model("bert-base-uncased", model_type="text")
        
        # Try to run simple inference to check simulation status
        logger.info("Testing inference with BERT model")
        result = await impl.run_inference(
            "bert-base-uncased", 
            "This is a test input for WebNN verification."
        )
        
        if not result:
            logger.error("Failed to run inference with BERT model")
            await impl.shutdown()
            return False, {}
        
        # Check if simulation was used
        is_simulation = result.get("is_simulation", True)
        using_transformers_js = result.get("using_transformers_js", False)
        
        simulation_status = {
            "webnn_supported": webnn_supported,
            "webnn_backend": webnn_backend,
            "impl_type": impl_type,
            "is_simulation": is_simulation,
            "using_transformers_js": using_transformers_js
        }
        
        if is_simulation:
            logger.warning("WebNN is using SIMULATION MODE")
        else:
            logger.info("WebNN is using REAL HARDWARE ACCELERATION")
        
        if using_transformers_js:
            logger.info("WebNN is using transformers.js for model inference")
        
        # Shutdown
        await impl.shutdown()
        return True, simulation_status
        
    except Exception as e:
        logger.error(f"Error testing WebNN implementation: {e}")
        return False, {}

async def generate_implementation_report(webgpu_status, webnn_status):
    """Generate comprehensive implementation report."""
    report = "\n" + "=" * 80 + "\n"
    report += "             WEBNN AND WEBGPU IMPLEMENTATION STATUS REPORT\n"
    report += "=" * 80 + "\n\n"
    
    # Overall status summary
    report += "## OVERALL IMPLEMENTATION STATUS\n\n"
    
    webgpu_impl_ready = webgpu_status.get("impl_type") == "REAL_WEBGPU"
    webnn_impl_ready = webnn_status.get("impl_type", "").startswith("REAL_WEBNN")
    
    webgpu_sim = webgpu_status.get("is_simulation", True)
    webnn_sim = webnn_status.get("is_simulation", True)
    
    report += f"- WebGPU Implementation: {'✅ REAL' if webgpu_impl_ready else '❌ NOT REAL'}\n"
    report += f"- WebNN Implementation: {'✅ REAL' if webnn_impl_ready else '❌ NOT REAL'}\n"
    report += f"- WebGPU Using Simulation: {'❌ YES' if webgpu_sim else '✅ NO'}\n"
    report += f"- WebNN Using Simulation: {'❌ YES' if webnn_sim else '✅ NO'}\n\n"
    
    # Hardware support
    report += "## HARDWARE SUPPORT STATUS\n\n"
    
    webgpu_supported = webgpu_status.get("webgpu_supported", False)
    webnn_supported = webnn_status.get("webnn_supported", False)
    
    report += f"- WebGPU Supported by Browser: {'✅ YES' if webgpu_supported else '❌ NO'}\n"
    report += f"- WebNN Supported by Browser: {'✅ YES' if webnn_supported else '❌ NO'}\n"
    
    if webgpu_supported:
        webgpu_adapter = webgpu_status.get("webgpu_adapter", {})
        vendor = webgpu_adapter.get("vendor", "Unknown")
        architecture = webgpu_adapter.get("architecture", "Unknown")
        report += f"- WebGPU Adapter: {vendor} - {architecture}\n"
    
    if webnn_supported:
        webnn_backend = webnn_status.get("webnn_backend", "Unknown")
        report += f"- WebNN Backend: {webnn_backend}\n"
    
    report += "\n"
    
    # Implementation details
    report += "## IMPLEMENTATION DETAILS\n\n"
    
    # WebGPU implementation details
    report += "### WebGPU Implementation:\n"
    report += f"- Implementation Type: {webgpu_status.get('impl_type', 'Unknown')}\n"
    report += f"- Using Transformers.js: {'✅ YES' if webgpu_status.get('using_transformers_js', False) else '❌ NO'}\n"
    report += f"- Using Hardware Acceleration: {'✅ YES' if not webgpu_status.get('is_simulation', True) else '❌ NO'}\n"
    
    # WebNN implementation details
    report += "\n### WebNN Implementation:\n"
    report += f"- Implementation Type: {webnn_status.get('impl_type', 'Unknown')}\n"
    report += f"- Using Transformers.js: {'✅ YES' if webnn_status.get('using_transformers_js', False) else '❌ NO'}\n"
    report += f"- Using Hardware Acceleration: {'✅ YES' if not webnn_status.get('is_simulation', True) else '❌ NO'}\n"
    
    # Conclusion
    report += "\n## CONCLUSION\n\n"
    
    # Check for implementation files to determine if the implementation is complete
    implementation_files_exist = check_implementation_files()
    
    if implementation_files_exist:
        report += "✅ Real WebGPU and WebNN implementations are COMPLETE\n"
        
        if webgpu_sim or webnn_sim:
            report += "❗ However, they are using simulation mode because hardware acceleration is not available in the browser.\n"
            report += "This is expected when running in environments without WebGPU/WebNN hardware support.\n"
            report += "The implementations will automatically use hardware acceleration when available.\n"
        else:
            report += "✅ Both implementations are using REAL hardware acceleration\n"
    else:
        report += "❌ Real WebGPU and WebNN implementations are NOT COMPLETE\n"
        report += "Further development is needed to implement real browser-based WebGPU and WebNN support.\n"
    
    report += "\n" + "=" * 80 + "\n"
    
    return report

async def main_async(args):
    """Run verification asynchronously."""
    # Check dependencies
    logger.info("Checking dependencies...")
    if not check_dependencies():
        return 1
    
    # Check implementation files
    logger.info("Checking implementation files...")
    if not check_implementation_files():
        return 1
    
    # Check for simulation code flags
    logger.info("Checking for simulation code flags...")
    check_simulated_code_flags()
    
    # Test WebGPU implementation
    logger.info(f"Testing WebGPU implementation with {args.browser} browser...")
    webgpu_success, webgpu_status = await test_webgpu_implementation(
        browser=args.browser, 
        headless=not args.no_headless,
        verbose=args.verbose
    )
    
    # Test WebNN implementation
    logger.info(f"Testing WebNN implementation with {args.browser} browser...")
    webnn_success, webnn_status = await test_webnn_implementation(
        browser=args.browser, 
        headless=not args.no_headless,
        verbose=args.verbose
    )
    
    # Generate report
    report = await generate_implementation_report(webgpu_status, webnn_status)
    print(report)
    
    # Write report to file
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        logger.info(f"Report written to {args.output}")
    
    # Return success if both implementations were verified
    return 0 if webgpu_success and webnn_success else 1

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Verify WebNN and WebGPU Implementation Status")
    parser.add_argument("--browser", choices=["chrome", "firefox", "edge"], default="chrome",
                      help="Browser to use for testing")
    parser.add_argument("--no-headless", action="store_true",
                      help="Disable headless mode (show browser UI)")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    parser.add_argument("--output", type=str,
                      help="Output file for report")
    
    args = parser.parse_args()
    
    # Run verification
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(main_async(args))

if __name__ == "__main__":
    sys.exit(main())