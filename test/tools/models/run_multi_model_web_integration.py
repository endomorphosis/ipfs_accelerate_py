#!/usr/bin/env python3
"""
Demonstration script for the Multi-Model Web Integration system.

This script demonstrates the complete integration between the predictive performance
system, web resource pooling, and empirical validation - providing a comprehensive
example of using WebNN/WebGPU acceleration with performance prediction and validation.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_multi_model_web_integration")

# Add the parent directory to the Python path for imports
parent_dir = Path(__file__).parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import the necessary modules
try:
    from ipfs_accelerate_py.predictive_performance.multi_model_web_integration import MultiModelWebIntegration
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure the predictive_performance module is available")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Model Web Integration Demo with WebNN/WebGPU Acceleration"
    )
    
    # Model configuration
    parser.add_argument(
        "--models", 
        type=str, 
        default="bert-base-uncased,vit-base-patch16-224",
        help="Comma-separated list of models to run (default: bert-base-uncased,vit-base-patch16-224)"
    )
    
    # Browser configuration
    parser.add_argument(
        "--browser",
        type=str,
        choices=["chrome", "firefox", "edge", "safari", "auto"],
        default="auto",
        help="Browser to use for execution (default: auto for automatic selection)"
    )
    
    # Hardware platform
    parser.add_argument(
        "--platform",
        type=str,
        choices=["webgpu", "webnn", "cpu", "auto"],
        default="auto",
        help="Hardware platform to use (default: auto for automatic selection)"
    )
    
    # Execution strategy
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["parallel", "sequential", "batched", "auto"],
        default="auto",
        help="Execution strategy to use (default: auto for automatic recommendation)"
    )
    
    # Optimization goal
    parser.add_argument(
        "--optimize",
        type=str,
        choices=["latency", "throughput", "memory"],
        default="latency",
        help="Optimization goal (default: latency)"
    )
    
    # Tensor sharing
    parser.add_argument(
        "--tensor-sharing",
        action="store_true",
        help="Enable tensor sharing between models"
    )
    
    # Empirical validation
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable empirical validation of predictions"
    )
    
    # Compare strategies
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Compare different execution strategies"
    )
    
    # Browser detection
    parser.add_argument(
        "--detect-browsers",
        action="store_true",
        help="Detect available browsers and their capabilities"
    )
    
    # Database path
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to database file for storing results"
    )
    
    # Repetitions
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions for each execution (default: 1)"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def create_model_configs(model_names: List[str]) -> List[Dict[str, Any]]:
    """
    Create model configurations from model names.
    
    Args:
        model_names: List of model names
        
    Returns:
        List of model configurations
    """
    model_configs = []
    
    for name in model_names:
        # Determine model type based on name
        if any(x in name.lower() for x in ["bert", "t5", "gpt", "llama", "roberta", "bart"]):
            model_type = "text_embedding"
        elif any(x in name.lower() for x in ["vit", "resnet", "efficientnet", "convnext"]):
            model_type = "vision"
        elif any(x in name.lower() for x in ["whisper", "wav2vec", "hubert"]):
            model_type = "audio"
        elif any(x in name.lower() for x in ["clip", "blip"]):
            model_type = "multimodal"
        else:
            model_type = "text_embedding"  # Default
        
        # Create configuration
        config = {
            "model_name": name,
            "model_type": model_type,
            "batch_size": 1
        }
        
        model_configs.append(config)
    
    return model_configs


def get_hardware_platform(platform: str, browser: Optional[str] = None) -> str:
    """
    Get the hardware platform to use based on the platform and browser.
    
    Args:
        platform: Specified platform
        browser: Specified browser (if any)
        
    Returns:
        Hardware platform to use
    """
    if platform != "auto":
        return platform
    
    # Auto select based on browser
    if browser == "edge":
        return "webnn"  # Edge has good WebNN support
    elif browser in ["chrome", "firefox"]:
        return "webgpu"  # Chrome and Firefox have good WebGPU support
    else:
        return "webgpu"  # Default to WebGPU


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse model names
    model_names = [name.strip() for name in args.models.split(",")]
    model_configs = create_model_configs(model_names)
    
    logger.info(f"Running with {len(model_configs)} models: {', '.join(model_names)}")
    
    # Determine browser
    browser = None if args.browser == "auto" else args.browser
    
    # Determine hardware platform
    hardware_platform = get_hardware_platform(args.platform, browser)
    
    # Determine execution strategy
    execution_strategy = None if args.strategy == "auto" else args.strategy
    
    # Create browser preferences
    browser_preferences = {
        "text_embedding": "edge",     # Edge has best WebNN support for text models
        "text_generation": "chrome",  # Chrome has good all-around support
        "vision": "chrome",           # Chrome works well for vision models
        "audio": "firefox",           # Firefox has best audio compute shader performance
        "multimodal": "chrome"        # Chrome has best balance for multimodal
    }
    
    # Create and initialize integration
    integration = MultiModelWebIntegration(
        max_connections=4,
        browser_preferences=browser_preferences,
        enable_validation=args.validate,
        enable_tensor_sharing=args.tensor_sharing,
        enable_strategy_optimization=True,
        db_path=args.db_path,
        validation_interval=5,
        refinement_interval=20,
        browser_capability_detection=args.detect_browsers,
        verbose=args.verbose
    )
    
    success = integration.initialize()
    if not success:
        logger.error("Failed to initialize integration")
        sys.exit(1)
    
    try:
        # Detect browsers if requested
        if args.detect_browsers:
            logger.info("Detecting browser capabilities")
            capabilities = integration.get_browser_capabilities()
            
            logger.info(f"Detected {len(capabilities)} browsers with capabilities:")
            for browser_name, caps in capabilities.items():
                logger.info(f"  {browser_name}:")
                logger.info(f"    WebGPU: {caps.get('webgpu', False)}")
                logger.info(f"    WebNN: {caps.get('webnn', False)}")
                logger.info(f"    Compute Shader: {caps.get('compute_shader', False)}")
                logger.info(f"    Memory Limit: {caps.get('memory_limit', 'unknown')} MB")
                logger.info(f"    Concurrent Model Limit: {caps.get('concurrent_model_limit', 'unknown')}")
        
        # Get optimal browser if auto-selection
        if browser is None:
            # Use first model's type for browser selection
            if model_configs:
                model_type = model_configs[0].get("model_type", "text_embedding")
                browser = integration.get_optimal_browser(model_type)
                logger.info(f"Auto-selected optimal browser for {model_type}: {browser}")
        
        # Get optimal strategy if auto-selection
        if execution_strategy is None:
            execution_strategy = integration.get_optimal_strategy(
                model_configs=model_configs,
                browser=browser,
                hardware_platform=hardware_platform,
                optimization_goal=args.optimize
            )
            logger.info(f"Auto-selected optimal execution strategy: {execution_strategy}")
        
        # Compare strategies if requested
        if args.compare_strategies:
            logger.info(f"Comparing execution strategies for {len(model_configs)} models")
            
            comparison = integration.compare_strategies(
                model_configs=model_configs,
                hardware_platform=hardware_platform,
                browser=browser,
                optimization_goal=args.optimize
            )
            
            logger.info("Strategy comparison results:")
            logger.info(f"  Best strategy: {comparison.get('best_strategy', 'unknown')}")
            logger.info(f"  Recommended strategy: {comparison.get('recommended_strategy', 'unknown')}")
            logger.info(f"  Recommendation accuracy: {comparison.get('recommendation_accuracy', False)}")
            
            # Print detailed results for each strategy
            if "strategy_results" in comparison:
                logger.info("  Performance by strategy:")
                for strategy, metrics in comparison["strategy_results"].items():
                    logger.info(f"    {strategy}:")
                    logger.info(f"      Throughput: {metrics.get('throughput', 0):.2f} items/sec")
                    logger.info(f"      Latency: {metrics.get('latency', 0):.2f} ms")
                    logger.info(f"      Memory: {metrics.get('memory_usage', 0):.2f} MB")
            
            # Print optimization impact
            if "optimization_impact" in comparison:
                impact = comparison["optimization_impact"]
                if "improvement_percent" in impact:
                    logger.info(f"  Optimization impact: {impact.get('improvement_percent', 0):.1f}% improvement")
        
        # Execute the models
        logger.info(f"Executing {len(model_configs)} models with {execution_strategy} strategy")
        
        total_time = 0
        avg_throughput = 0
        avg_latency = 0
        
        for i in range(args.repetitions):
            logger.info(f"Execution {i+1}/{args.repetitions}")
            
            start_time = time.time()
            
            result = integration.execute_models(
                model_configs=model_configs,
                hardware_platform=hardware_platform,
                execution_strategy=execution_strategy,
                optimization_goal=args.optimize,
                browser=browser,
                validate_predictions=args.validate,
                return_detailed_metrics=args.verbose
            )
            
            execution_time = time.time() - start_time
            total_time += execution_time
            
            if result.get("success", False):
                logger.info(f"  Execution successful in {execution_time:.2f} seconds")
                logger.info(f"  Strategy: {result.get('execution_strategy', 'unknown')}")
                
                # Log performance metrics
                throughput = result.get("throughput", 0)
                latency = result.get("latency", 0)
                memory = result.get("memory_usage", 0)
                
                avg_throughput += throughput
                avg_latency += latency
                
                logger.info(f"  Throughput: {throughput:.2f} items/sec")
                logger.info(f"  Latency: {latency:.2f} ms")
                logger.info(f"  Memory usage: {memory:.2f} MB")
                
                # Log predicted vs actual if validation enabled
                if args.validate:
                    predicted_throughput = result.get("predicted_throughput", 0)
                    predicted_latency = result.get("predicted_latency", 0)
                    predicted_memory = result.get("predicted_memory", 0)
                    
                    # Calculate prediction errors
                    throughput_error = abs(1 - (throughput / predicted_throughput if predicted_throughput > 0 else 1))
                    latency_error = abs(1 - (latency / predicted_latency if predicted_latency > 0 else 1))
                    memory_error = abs(1 - (memory / predicted_memory if predicted_memory > 0 else 1))
                    
                    logger.info("  Prediction validation:")
                    logger.info(f"    Throughput: {predicted_throughput:.2f} predicted vs {throughput:.2f} actual ({throughput_error:.2%} error)")
                    logger.info(f"    Latency: {predicted_latency:.2f} predicted vs {latency:.2f} actual ({latency_error:.2%} error)")
                    logger.info(f"    Memory: {predicted_memory:.2f} predicted vs {memory:.2f} actual ({memory_error:.2%} error)")
            else:
                logger.error(f"  Execution failed: {result.get('error', 'unknown error')}")
        
        # Print average results
        if args.repetitions > 1:
            avg_throughput /= args.repetitions
            avg_latency /= args.repetitions
            avg_time = total_time / args.repetitions
            
            logger.info(f"Average results over {args.repetitions} repetitions:")
            logger.info(f"  Execution time: {avg_time:.2f} seconds")
            logger.info(f"  Throughput: {avg_throughput:.2f} items/sec")
            logger.info(f"  Latency: {avg_latency:.2f} ms")
        
        # Get validation metrics
        if args.validate:
            logger.info("Validation metrics:")
            metrics = integration.get_validation_metrics()
            
            logger.info(f"  Validation count: {metrics.get('validation_count', 0)}")
            
            if "error_rates" in metrics:
                error_rates = metrics["error_rates"]
                for metric, value in error_rates.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {metric}: {value:.2%}")
            
            # Print database metrics if available
            if "database" in metrics:
                db_metrics = metrics["database"]
                logger.info("  Database metrics:")
                logger.info(f"    Total validations: {db_metrics.get('validation_count', 0)}")
                logger.info(f"    Refinement count: {db_metrics.get('refinement_count', 0)}")
        
        # Get execution statistics
        logger.info("Execution statistics:")
        stats = integration.get_execution_statistics()
        
        logger.info(f"  Total executions: {stats['total_executions']}")
        logger.info(f"  Browser executions: {stats['browser_executions']}")
        logger.info(f"  Strategy executions: {stats['strategy_executions']}")
    
    finally:
        # Close the integration
        integration.close()
        logger.info("Multi-Model Web Integration demo completed")


if __name__ == "__main__":
    main()