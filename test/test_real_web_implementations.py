#!/usr/bin/env python3
"""
Test Real WebNN and WebGPU Implementations

This script tests the real WebNN and WebGPU implementations
with a simple BERT model.

Usage:
    python test_real_web_implementations.py --platform webgpu
    python test_real_web_implementations.py --platform webnn
    python test_real_web_implementations.py --platform both
    """

    import os
    import sys
    import json
    import anyio
    import argparse
    import logging
    from pathlib import Path

# Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

# Add parent directory to path so we can import from fixed_web_platform
    sys.path.insert(0, str(Path(__file__).parent))

# Import implementations
try:
    from fixed_web_platform.webgpu_implementation import RealWebGPUImplementation
    from fixed_web_platform.webnn_implementation import RealWebNNImplementation
    HAS_IMPLEMENTATIONS = True
except ImportError:
    logger.error("Failed to import WebGPU/WebNN implementations")
    logger.error("Make sure to run fix_webnn_webgpu_implementations.py --fix first")
    HAS_IMPLEMENTATIONS = False

async def test_webgpu(browser_name="chrome", headless=False):
    """Test WebGPU implementation.
    
    Args:
        browser_name: Browser to use
        headless: Whether to run in headless mode
        """
        logger.info("===== Testing WebGPU Implementation =====")
        logger.info(f"\1{browser_name}\3")
        logger.info(f"\1{headless}\3")
        logger.info("=======================================")
    
    # Create implementation
        webgpu_impl = RealWebGPUImplementation(browser_name=browser_name, headless=headless)
    
    try:
        # Initialize
        logger.info("\nInitializing WebGPU implementation...")
        success = await webgpu_impl.initialize()
        if not success:
            logger.error("Failed to initialize WebGPU implementation")
        return False
        
        # Get feature support
        logger.info("\nWebGPU feature support:")
        features = webgpu_impl.get_feature_support()
        if features:
            for key, value in features.items():
                logger.info(f"\1{value}\3")
        else:
            logger.info("  No feature information available")
        
        # Initialize model
            model_name = "bert-base-uncased"
            logger.info(f"\1{model_name}\3")
        
            model_info = await webgpu_impl.initialize_model(model_name, model_type="text")
        if not model_info:
            logger.error(f"\1{model_name}\3")
            await webgpu_impl.shutdown()
            return False
        
            logger.info("Model initialization successful:")
            logger.info(f"\1{model_info.get('status', 'unknown')}\3")
            is_real = not model_info.get('is_simulation', True)
            logger.info(f"\1{is_real}\3")
        
        # Run inference
            logger.info("\nRunning inference...")
            input_text = "This is a test input for WebGPU implementation."
        
            result = await webgpu_impl.run_inference(model_name, input_text)
        if not result:
            logger.error("Failed to run inference")
            await webgpu_impl.shutdown()
            return False
        
        # Check implementation details
            impl_details = result.get("_implementation_details", {})
            is_simulation = impl_details.get("is_simulation", True)
            using_transformers = impl_details.get("using_transformers_js", False)
        
            logger.info("\nInference results:")
            logger.info(f"\1{result.get('status', 'unknown')}\3")
            logger.info(f"\1{not is_simulation}\3")
            logger.info(f"\1{using_transformers}\3")
        
        if 'performance_metrics' in result:
            metrics = result['performance_metrics'],,
            logger.info("\nPerformance metrics:")
            logger.info(f"  Inference time: {metrics.get('inference_time_ms', 0):.2f} ms")
            logger.info(f"  Throughput: {metrics.get('throughput_items_per_sec', 0):.2f} items/sec")
        
        # Shutdown
            await webgpu_impl.shutdown()
            logger.info("\nWebGPU implementation test completed successfully")
        
            return True
    
    except Exception as e:
        logger.error(f"\1{e}\3")
        await webgpu_impl.shutdown()
            return False

async def test_webnn(browser_name="chrome", headless=False):
    """Test WebNN implementation.
    
    Args:
        browser_name: Browser to use
        headless: Whether to run in headless mode
        """
        logger.info("===== Testing WebNN Implementation =====")
        logger.info(f"\1{browser_name}\3")
        logger.info(f"\1{headless}\3")
        logger.info("=======================================")
    
    # Create implementation
        webnn_impl = RealWebNNImplementation(browser_name=browser_name, headless=headless)
    
    try:
        # Initialize
        logger.info("\nInitializing WebNN implementation...")
        success = await webnn_impl.initialize()
        if not success:
            logger.error("Failed to initialize WebNN implementation")
        return False
        
        # Get feature support
        logger.info("\nWebNN feature support:")
        features = webnn_impl.get_feature_support()
        if features:
            for key, value in features.items():
                logger.info(f"\1{value}\3")
        else:
            logger.info("  No feature information available")
        
        # Get backend info
            logger.info("\nWebNN backend info:")
            backend_info = webnn_impl.get_backend_info()
        if backend_info:
            for key, value in backend_info.items():
                logger.info(f"\1{value}\3")
        else:
            logger.info("  No backend information available")
        
        # Initialize model
            model_name = "bert-base-uncased"
            logger.info(f"\1{model_name}\3")
        
            model_info = await webnn_impl.initialize_model(model_name, model_type="text")
        if not model_info:
            logger.error(f"\1{model_name}\3")
            await webnn_impl.shutdown()
            return False
        
            logger.info("Model initialization successful:")
            logger.info(f"\1{model_info.get('status', 'unknown')}\3")
            is_real = not model_info.get('is_simulation', True)
            logger.info(f"\1{is_real}\3")
        
        # Run inference
            logger.info("\nRunning inference...")
            input_text = "This is a test input for WebNN implementation."
        
            result = await webnn_impl.run_inference(model_name, input_text)
        if not result:
            logger.error("Failed to run inference")
            await webnn_impl.shutdown()
            return False
        
        # Check implementation details
            impl_details = result.get("_implementation_details", {})
            is_simulation = impl_details.get("is_simulation", True)
            using_transformers = impl_details.get("using_transformers_js", False)
        
            logger.info("\nInference results:")
            logger.info(f"\1{result.get('status', 'unknown')}\3")
            logger.info(f"\1{not is_simulation}\3")
            logger.info(f"\1{using_transformers}\3")
        
        if 'performance_metrics' in result:
            metrics = result['performance_metrics'],,
            logger.info("\nPerformance metrics:")
            logger.info(f"  Inference time: {metrics.get('inference_time_ms', 0):.2f} ms")
            logger.info(f"  Throughput: {metrics.get('throughput_items_per_sec', 0):.2f} items/sec")
        
        # Shutdown
            await webnn_impl.shutdown()
            logger.info("\nWebNN implementation test completed successfully")
        
            return True
    
    except Exception as e:
        logger.error(f"\1{e}\3")
        await webnn_impl.shutdown()
            return False

async def main_async(args):
    """Main async function."""
    if not HAS_IMPLEMENTATIONS:
        logger.error("WebGPU/WebNN implementations not available")
        logger.error("Please run fix_webnn_webgpu_implementations.py --fix first")
    return False
    
    if args.platform == "webgpu" or args.platform == "both":
        # Test WebGPU
        webgpu_success = await test_webgpu(browser_name=args.browser, headless=args.headless)
        if not webgpu_success and args.platform == "webgpu":
        return False
    
    if args.platform == "webnn" or args.platform == "both":
        # Test WebNN
        webnn_success = await test_webnn(browser_name=args.browser, headless=args.headless)
        if not webnn_success and args.platform == "webnn":
        return False
    
        return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Real WebNN and WebGPU Implementations")
    parser.add_argument("--platform", choices=["webgpu", "webnn", "both"], default="both",
    help="Platform to test")
    parser.add_argument("--browser", choices=["chrome", "firefox", "edge"], default="chrome",
    help="Browser to use")
    parser.add_argument("--headless", action="store_true",
    help="Run browser in headless mode")
    
    args = parser.parse_args()
    
    # Run async main
    success = anyio.run(main_async, args)
    
        return 0 if success else 1
:
if __name__ == "__main__":
    sys.exit(main())