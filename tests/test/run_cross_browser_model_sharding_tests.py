#!/usr/bin/env python3
"""
Comprehensive Cross-Browser Model Sharding Test Runner

This script runs end-to-end tests for fault-tolerant cross-browser model sharding
across all sharding strategies and browsers, providing detailed metrics and analysis.

Usage:
    python run_cross_browser_model_sharding_tests.py --all-models
    python run_cross_browser_model_sharding_tests.py --model llama-7b --browsers chrome,firefox,edge --comprehensive
    python run_cross_browser_model_sharding_tests.py --model whisper-tiny --browsers firefox,chrome --fault-level high
"""

import os
import sys
import asyncio
import argparse
import logging
import datetime
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cross_browser_tests.log')
    ]
)
logger = logging.getLogger(__name__)

# Define model types to test
MODEL_TYPES = {
    "text": ["bert-base-uncased", "t5-small"],
    "vision": ["vit-base-patch16-224"],
    "audio": ["whisper-tiny"],
    "multimodal": ["clip-vit-base-patch32"]
}

# Define validation test parameters
SHARDING_STRATEGIES = ["optimal", "layer", "component_based", "attention_feedforward"]
FAULT_TOLERANCE_LEVELS = ["medium", "high"]
RECOVERY_STRATEGIES = ["progressive", "coordinated"]
BROWSERS = ["chrome", "firefox", "edge"]

async def run_validation_test(
    model: str,
    browsers: List[str],
    strategy: Optional[str] = None,
    fault_level: Optional[str] = None,
    recovery_strategy: Optional[str] = None,
    comprehensive: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a validation test for a specific configuration.

    Args:
        model: Model name to test
        browsers: List of browsers to use
        strategy: Sharding strategy
        fault_level: Fault tolerance level
        recovery_strategy: Recovery strategy
        comprehensive: Whether to run comprehensive tests
        output_dir: Directory to store test results

    Returns:
        Dictionary with test results and status
    """
    logger.info(f"Running validation test for {model} with {strategy if strategy else 'default'} strategy")

    # Build command-line arguments
    cmd = ["python", "test_fault_tolerant_cross_browser_model_sharding_validation.py"]
    cmd.extend(["--model", model])
    cmd.extend(["--browsers", ",".join(browsers)])

    if strategy:
        cmd.extend(["--strategy", strategy])

    if fault_level:
        cmd.extend(["--fault-level", fault_level])

    if recovery_strategy:
        cmd.extend(["--recovery-strategy", recovery_strategy])

    if comprehensive:
        cmd.append("--comprehensive")

    # Add output path if provided
    if output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model.replace('/', '_')}_{strategy or 'default'}_{timestamp}.json"
        output_path = os.path.join(output_dir, filename)
        cmd.extend(["--output", output_path])

    # Add timeout for large models
    if "llama" in model and "70b" in model:
        cmd.extend(["--timeout", "600"])  # 10-minute timeout for very large models
    elif any(large in model for large in ["large", "xl", "13b"]):
        cmd.extend(["--timeout", "300"])  # 5-minute timeout for large models

    # Execute command
    logger.info(f"Executing: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False  # Don't raise exception on non-zero exit code
        )

        # Check return code
        exit_code = result.returncode
        status = "passed" if exit_code == 0 else "failed" if exit_code == 1 else "error"

        logger.info(f"Test completed with status: {status}")

        # Return test results
        return {
            "model": model,
            "browsers": browsers,
            "strategy": strategy,
            "fault_level": fault_level,
            "recovery_strategy": recovery_strategy,
            "comprehensive": comprehensive,
            "status": status,
            "exit_code": exit_code,
            "output": result.stdout,
            "error": result.stderr,
            "output_path": output_path if output_dir else None,
            "command": " ".join(cmd),
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error running test: {e}")
        return {
            "model": model,
            "browsers": browsers,
            "strategy": strategy,
            "fault_level": fault_level,
            "recovery_strategy": recovery_strategy,
            "comprehensive": comprehensive,
            "status": "error",
            "error": str(e),
            "command": " ".join(cmd),
            "timestamp": datetime.datetime.now().isoformat()
        }

async def run_model_tests(
    model: str,
    browsers: List[str],
    strategies: Optional[List[str]] = None,
    fault_levels: Optional[List[str]] = None,
    recovery_strategies: Optional[List[str]] = None,
    comprehensive: bool = False,
    output_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Run tests for a specific model with all specified configurations.

    Args:
        model: Model name to test
        browsers: List of browsers to use
        strategies: List of sharding strategies
        fault_levels: List of fault tolerance levels
        recovery_strategies: List of recovery strategies
        comprehensive: Whether to run comprehensive tests
        output_dir: Directory to store test results

    Returns:
        List of dictionaries with test results
    """
    logger.info(f"Running tests for model {model}")

    # Use default values if not provided
    strategies = strategies or ["optimal"]
    fault_levels = fault_levels or ["medium"]
    recovery_strategies = recovery_strategies or ["progressive"]

    # If comprehensive, use all options
    if comprehensive:
        strategies = SHARDING_STRATEGIES
        fault_levels = FAULT_TOLERANCE_LEVELS
        recovery_strategies = RECOVERY_STRATEGIES

    # Create list to store results
    results = []

    # Run tests for each combination
    for strategy in strategies:
        for fault_level in fault_levels:
            for recovery_strategy in recovery_strategies:
                result = await run_validation_test(
                    model=model,
                    browsers=browsers,
                    strategy=strategy,
                    fault_level=fault_level,
                    recovery_strategy=recovery_strategy,
                    comprehensive=comprehensive,
                    output_dir=output_dir
                )
                results.append(result)

    return results

async def run_batch_tests(args) -> Dict[str, Any]:
    """
    Run a batch of tests based on command-line arguments.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary with batch test results
    """
    # Create timestamp for this batch
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine models to test
    models_to_test = []

    if args.all_models:
        # Add one model from each type
        for model_type, model_list in MODEL_TYPES.items():
            models_to_test.append(model_list[0])
    elif args.model_type:
        # Add all models of specified type
        if args.model_type in MODEL_TYPES:
            models_to_test.extend(MODEL_TYPES[args.model_type])
        else:
            logger.error(f"Unknown model type: {args.model_type}")
            return {"status": "error", "message": f"Unknown model type: {args.model_type}"}
    elif args.model:
        # Add specified model
        models_to_test.append(args.model)
    else:
        # Default to testing BERT
        models_to_test.append("bert-base-uncased")

    # Parse browser list
    browsers = args.browsers.split(',')

    # Determine strategies, fault levels, and recovery strategies
    strategies = SHARDING_STRATEGIES if args.all_strategies else ([args.strategy] if args.strategy else None)
    fault_levels = FAULT_TOLERANCE_LEVELS if args.all_fault_levels else ([args.fault_level] if args.fault_level else None)
    recovery_strategies = RECOVERY_STRATEGIES if args.all_recovery_strategies else ([args.recovery_strategy] if args.recovery_strategy else None)

    # Create output directory if needed
    output_dir = None
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, f"cross_browser_tests_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    # Create batch results container
    batch_results = {
        "timestamp": timestamp,
        "command_args": vars(args),
        "models_tested": models_to_test,
        "browsers": browsers,
        "strategies": strategies,
        "fault_levels": fault_levels,
        "recovery_strategies": recovery_strategies,
        "results": [],
        "summary": {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "error_tests": 0
        }
    }

    # Run tests for each model
    total_models = len(models_to_test)
    for i, model in enumerate(models_to_test, 1):
        logger.info(f"Testing model {i}/{total_models}: {model}")

        model_results = await run_model_tests(
            model=model,
            browsers=browsers,
            strategies=strategies,
            fault_levels=fault_levels,
            recovery_strategies=recovery_strategies,
            comprehensive=args.comprehensive,
            output_dir=output_dir
        )

        # Add to batch results
        batch_results["results"].extend(model_results)

        # Update summary
        for result in model_results:
            batch_results["summary"]["total_tests"] += 1
            if result["status"] == "passed":
                batch_results["summary"]["passed_tests"] += 1
            elif result["status"] == "failed":
                batch_results["summary"]["failed_tests"] += 1
            else:
                batch_results["summary"]["error_tests"] += 1

    # Calculate pass rate
    total_tests = batch_results["summary"]["total_tests"]
    passed_tests = batch_results["summary"]["passed_tests"]
    batch_results["summary"]["pass_rate"] = passed_tests / total_tests if total_tests > 0 else 0

    # Generate overall status
    if batch_results["summary"]["error_tests"] > 0:
        batch_results["status"] = "error"
    elif batch_results["summary"]["failed_tests"] > 0:
        batch_results["status"] = "failed"
    else:
        batch_results["status"] = "passed"

    # Save batch results if output directory is specified
    if output_dir:
        batch_results_path = os.path.join(output_dir, "batch_results.json")
        with open(batch_results_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        logger.info(f"Batch results saved to {batch_results_path}")

    return batch_results

def print_batch_summary(batch_results: Dict[str, Any]) -> None:
    """
    Print summary of batch test results.

    Args:
        batch_results: Batch test results
    """
    summary = batch_results["summary"]
    status = batch_results["status"]

    print("\n" + "="*80)
    print(f"CROSS-BROWSER MODEL SHARDING TESTS SUMMARY")
    print("="*80)
    print(f"Models tested: {', '.join(batch_results['models_tested'])}")
    print(f"Browsers: {', '.join(batch_results['browsers'])}")
    
    if batch_results.get("strategies"):
        print(f"Strategies: {', '.join(batch_results['strategies'])}")
    
    if batch_results.get("fault_levels"):
        print(f"Fault levels: {', '.join(batch_results['fault_levels'])}")
    
    if batch_results.get("recovery_strategies"):
        print(f"Recovery strategies: {', '.join(batch_results['recovery_strategies'])}")
    
    print(f"Total tests: {summary['total_tests']}")
    print("-"*80)
    print(f"Passed: {summary['passed_tests']} ({summary['pass_rate']:.1%})")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Errors: {summary['error_tests']}")
    print("-"*80)
    
    # Set color for status
    if status == "passed":
        status_color = "\033[92m"  # Green
    elif status == "failed":
        status_color = "\033[91m"  # Red
    else:
        status_color = "\033[93m"  # Yellow
    
    print(f"Overall status: {status_color}{status}\033[0m")
    print("="*80)
    
    # Print results for each model
    if batch_results.get("results"):
        # Group results by model
        model_results = {}
        for result in batch_results["results"]:
            model = result["model"]
            if model not in model_results:
                model_results[model] = []
            model_results[model].append(result)
        
        print("\nRESULTS BY MODEL:")
        print("-"*80)
        
        for model, results in model_results.items():
            passed = sum(1 for r in results if r["status"] == "passed")
            total = len(results)
            pass_rate = passed / total if total > 0 else 0
            
            # Set color based on pass rate
            if pass_rate >= 0.9:
                status_color = "\033[92m"  # Green
            elif pass_rate >= 0.7:
                status_color = "\033[93m"  # Yellow
            else:
                status_color = "\033[91m"  # Red
            
            print(f"{model}: {status_color}{passed}/{total} ({pass_rate:.1%})\033[0m")
            
            # Print details for each test
            for result in results:
                strategy = result.get("strategy", "default")
                fault_level = result.get("fault_level", "default")
                recovery_strategy = result.get("recovery_strategy", "default")
                status = result["status"]
                
                # Set color for status
                if status == "passed":
                    status_color = "\033[92m"  # Green
                elif status == "failed":
                    status_color = "\033[91m"  # Red
                else:
                    status_color = "\033[93m"  # Yellow
                
                print(f"  - {strategy} + {fault_level} + {recovery_strategy}: {status_color}{status}\033[0m")
            
            print("-"*80)
    
    print("\n")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Cross-Browser Model Sharding Test Runner")
    
    # Model selection options
    parser.add_argument("--model", type=str,
                      help="Model name to test")
    parser.add_argument("--model-type", type=str, choices=list(MODEL_TYPES.keys()),
                      help="Test all models of a specific type")
    parser.add_argument("--all-models", action="store_true",
                      help="Test one model from each type")
    parser.add_argument("--list-models", action="store_true",
                      help="List available models by type and exit")
    
    # Browser options
    parser.add_argument("--browsers", type=str, default="chrome,firefox,edge",
                      help="Comma-separated list of browsers to use")
    
    # Test configuration options
    parser.add_argument("--strategy", type=str, choices=SHARDING_STRATEGIES,
                      help="Sharding strategy to use")
    parser.add_argument("--all-strategies", action="store_true",
                      help="Test all sharding strategies")
    parser.add_argument("--fault-level", type=str, choices=FAULT_TOLERANCE_LEVELS,
                      help="Fault tolerance level to use")
    parser.add_argument("--all-fault-levels", action="store_true",
                      help="Test all fault tolerance levels")
    parser.add_argument("--recovery-strategy", type=str, choices=RECOVERY_STRATEGIES,
                      help="Recovery strategy to use")
    parser.add_argument("--all-recovery-strategies", action="store_true",
                      help="Test all recovery strategies")
    
    # Test execution options
    parser.add_argument("--comprehensive", action="store_true",
                      help="Run comprehensive tests with all options")
    parser.add_argument("--output-dir", type=str,
                      help="Directory to store test results")
    parser.add_argument("--analyze", action="store_true",
                      help="Analyze test results after running tests")
    parser.add_argument("--visualize", action="store_true",
                      help="Generate visualizations of test results")
    
    # Parse arguments
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("Available models by type:")
        for model_type, models in MODEL_TYPES.items():
            print(f"{model_type}:")
            for model in models:
                print(f"  - {model}")
        return 0
    
    # Set comprehensive mode if specified
    if args.comprehensive:
        args.all_strategies = True
        args.all_fault_levels = True
        args.all_recovery_strategies = True
        logger.info("Running in comprehensive test mode")
    
    # Run tests
    try:
        logger.info("Starting cross-browser model sharding tests")
        batch_results = asyncio.run(run_batch_tests(args))
        
        # Print summary
        print_batch_summary(batch_results)
        
        # Run analysis if requested
        if args.analyze and args.output_dir:
            logger.info("Analyzing test results")
            
            # Import metrics collector here to avoid circular imports
            sys.path.append(str(Path(__file__).parent))
            from fixed_web_platform.cross_browser_metrics_collector import MetricsCollector
            
            # Create metrics collector with database in output directory
            db_path = os.path.join(args.output_dir, "cross_browser_metrics.duckdb")
            collector = MetricsCollector(db_path=db_path)
            
            # Import test results
            for result in batch_results["results"]:
                if "output_path" in result and result["output_path"]:
                    try:
                        with open(result["output_path"], 'r') as f:
                            test_result = json.load(f)
                            asyncio.run(collector.record_test_result(test_result))
                    except Exception as e:
                        logger.error(f"Error importing test result: {e}")
            
            # Run analysis
            analysis = asyncio.run(collector.analyze_fault_tolerance_performance())
            
            # Save analysis
            analysis_path = os.path.join(args.output_dir, "analysis.json")
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Analysis saved to {analysis_path}")
            
            # Generate visualizations if requested
            if args.visualize:
                logger.info("Generating visualizations")
                
                # Generate recovery time visualization
                asyncio.run(collector.generate_fault_tolerance_visualization(
                    output_path=os.path.join(args.output_dir, "recovery_time.png"),
                    metric="recovery_time"
                ))
                
                # Generate success rate visualization
                asyncio.run(collector.generate_fault_tolerance_visualization(
                    output_path=os.path.join(args.output_dir, "success_rate.png"),
                    metric="success_rate"
                ))
                
                # Generate performance impact visualization
                asyncio.run(collector.generate_fault_tolerance_visualization(
                    output_path=os.path.join(args.output_dir, "performance_impact.png"),
                    metric="performance_impact"
                ))
                
                logger.info(f"Visualizations saved to {args.output_dir}")
            
            # Close collector
            collector.close()
        
        # Determine exit code based on overall status
        if batch_results["status"] == "passed":
            logger.info("All tests passed")
            return 0
        elif batch_results["status"] == "failed":
            logger.error("Some tests failed")
            return 1
        else:
            logger.error("Tests encountered errors")
            return 2
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())