#!/usr/bin/env python3
"""
Advanced Fault-Tolerant Cross-Browser Model Sharding Validation Tool

This script provides comprehensive end-to-end testing and validation of the fault tolerance
features in the Cross-Browser Model Sharding system, with detailed metrics collection and analysis.

Usage:
    python test_fault_tolerant_cross_browser_model_sharding_validation.py --model llama-7b --browsers chrome,firefox,edge --all-strategies
    python test_fault_tolerant_cross_browser_model_sharding_validation.py --model whisper-tiny --browsers firefox,chrome --strategy optimal --fault-level high
    python test_fault_tolerant_cross_browser_model_sharding_validation.py --model bert-base-uncased --browsers edge,chrome --comprehensive
"""

import os
import sys
import json
import time
import anyio
import argparse
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).resolve().parent))

# Import required modules
try:
    from cross_browser_model_sharding import CrossBrowserModelShardingManager
    from fixed_web_platform.fault_tolerance_validation import FaultToleranceValidator
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Required modules not available: {e}")
    MODULES_AVAILABLE = False

# Define model families and configurations
MODEL_FAMILIES = {
    "bert": {
        "sizes": ["base-uncased", "large-uncased"],
        "model_type": "text",
        "components": ["embedding", "transformer_layers", "pooler"]
    },
    "whisper": {
        "sizes": ["tiny", "small", "base", "large"],
        "model_type": "audio",
        "components": ["audio_encoder", "text_decoder", "lm_head"]
    },
    "vit": {
        "sizes": ["base-patch16-224", "large-patch16-224"],
        "model_type": "vision",
        "components": ["embedding", "transformer", "head"]
    },
    "clip": {
        "sizes": ["vit-base-patch32", "vit-large-patch14"],
        "model_type": "multimodal",
        "components": ["vision_encoder", "text_encoder", "projection"]
    },
    "llama": {
        "sizes": ["7b", "13b", "70b"],
        "model_type": "text",
        "components": ["embedding", "attention", "feedforward", "lm_head"]
    },
    "t5": {
        "sizes": ["small", "base", "large"],
        "model_type": "text",
        "components": ["encoder", "decoder", "lm_head"]
    }
}

# Define common test parameters
DEFAULT_SHARDING_STRATEGIES = ["optimal", "layer", "component_based", "attention_feedforward"]
DEFAULT_FAULT_TOLERANCE_LEVELS = ["medium", "high"]
DEFAULT_RECOVERY_STRATEGIES = ["progressive", "coordinated"]
DEFAULT_TEST_SCENARIOS = [
    "connection_lost", 
    "browser_crash", 
    "component_timeout", 
    "multi_browser_failure"
]

async def run_validation_for_configuration(
    model_name: str, 
    model_type: str,
    shards: int,
    browsers: List[str],
    strategy: str,
    fault_tolerance_level: str,
    recovery_strategy: str,
    test_scenarios: List[str],
    timeout: int,
    comprehensive: bool
) -> Dict[str, Any]:
    """
    Run fault tolerance validation for a specific configuration.
    
    Args:
        model_name: Name of the model to test
        model_type: Type of model (text, vision, audio, multimodal)
        shards: Number of shards to create
        browsers: List of browsers to use
        strategy: Sharding strategy
        fault_tolerance_level: Fault tolerance level
        recovery_strategy: Recovery strategy
        test_scenarios: List of test scenarios to run
        timeout: Timeout in seconds
        comprehensive: Whether to run comprehensive tests
        
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Testing configuration: {model_name}, {strategy}, {fault_tolerance_level}, {recovery_strategy}")
    
    # Create model configuration
    model_config = {
        "model_type": model_type,
        "enable_fault_tolerance": True,
        "fault_tolerance_level": fault_tolerance_level,
        "recovery_strategy": recovery_strategy,
        "timeout": timeout
    }
    
    # Create test result container
    result = {
        "model_name": model_name,
        "model_type": model_type,
        "shards": shards,
        "browsers": browsers,
        "strategy": strategy,
        "fault_tolerance_level": fault_tolerance_level,
        "recovery_strategy": recovery_strategy,
        "start_time": datetime.datetime.now().isoformat(),
        "status": "initialized",
        "validation_results": None,
        "analysis": None
    }
    
    try:
        # Create sharding manager
        manager = CrossBrowserModelShardingManager(
            model_name=model_name,
            browsers=browsers,
            shard_type=strategy,
            num_shards=shards,
            model_config=model_config
        )
        
        # Initialize sharding
        logger.info(f"Initializing sharding for {model_name}")
        start_time = time.time()
        initialized = await manager.initialize()
        init_time = time.time() - start_time
        
        if not initialized:
            logger.error(f"Failed to initialize sharding for {model_name}")
            result["status"] = "initialization_failed"
            result["error"] = "Failed to initialize sharding"
            result["end_time"] = datetime.datetime.now().isoformat()
            return result
        
        logger.info(f"Sharding initialized in {init_time:.2f}s")
        result["initialization_time"] = init_time
        
        # Configure validator
        validator_config = {
            "fault_tolerance_level": fault_tolerance_level,
            "recovery_strategy": recovery_strategy,
            "test_scenarios": test_scenarios
        }
        
        # If comprehensive, add more test scenarios
        if comprehensive:
            validator_config["test_scenarios"] = DEFAULT_TEST_SCENARIOS
        
        # Create validator
        validator = FaultToleranceValidator(manager, validator_config)
        
        # Run validation
        logger.info(f"Running fault tolerance validation for {model_name}")
        start_time = time.time()
        validation_results = await validator.validate_fault_tolerance()
        validation_time = time.time() - start_time
        
        logger.info(f"Validation completed in {validation_time:.2f}s with status: {validation_results.get('validation_status', 'unknown')}")
        result["validation_time"] = validation_time
        result["validation_results"] = validation_results
        
        # Analyze results
        analysis = validator.analyze_results(validation_results)
        result["analysis"] = analysis
        
        # Update result status
        result["status"] = validation_results.get("validation_status", "unknown")
        
        # Shutdown manager
        await manager.shutdown()
        
        logger.info(f"Testing completed for configuration: {model_name}, {strategy}, {fault_tolerance_level}, {recovery_strategy}")
        
        result["end_time"] = datetime.datetime.now().isoformat()
        return result
        
    except Exception as e:
        logger.error(f"Error testing configuration: {e}")
        import traceback
        traceback.print_exc()
        
        result["status"] = "error"
        result["error"] = str(e)
        result["error_traceback"] = traceback.format_exc()
        result["end_time"] = datetime.datetime.now().isoformat()
        
        return result

async def run_validation_tests(args) -> Dict[str, Any]:
    """
    Run validation tests based on command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with overall test results
    """
    # Split model name into family and size if in format family-size
    model_parts = args.model.split('-', 1)
    if len(model_parts) == 2 and model_parts[0] in MODEL_FAMILIES:
        model_family = model_parts[0]
        model_size = model_parts[1]
    else:
        # Try to infer family from model name
        model_family = None
        model_size = None
        for family in MODEL_FAMILIES:
            if family in args.model.lower():
                model_family = family
                break
        
        if not model_family:
            model_family = args.default_family
            model_size = "base"
    
    # Get model type
    if args.model_type:
        model_type = args.model_type
    elif model_family in MODEL_FAMILIES:
        model_type = MODEL_FAMILIES[model_family]["model_type"]
    else:
        model_type = "text"  # Default
    
    # Parse browser list
    browsers = args.browsers.split(',')
    
    # Determine strategies to test
    if args.all_strategies:
        strategies = DEFAULT_SHARDING_STRATEGIES
    elif args.strategy:
        strategies = [args.strategy]
    else:
        strategies = ["optimal"]  # Default
    
    # Determine fault tolerance levels to test
    if args.all_fault_levels:
        fault_levels = DEFAULT_FAULT_TOLERANCE_LEVELS
    elif args.fault_level:
        fault_levels = [args.fault_level]
    else:
        fault_levels = ["medium"]  # Default
    
    # Determine recovery strategies to test
    if args.all_recovery_strategies:
        recovery_strategies = DEFAULT_RECOVERY_STRATEGIES
    elif args.recovery_strategy:
        recovery_strategies = [args.recovery_strategy]
    else:
        recovery_strategies = ["progressive"]  # Default
    
    # Create overall results container
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "command_args": vars(args),
        "model": args.model,
        "model_type": model_type,
        "browsers": browsers,
        "shards": args.shards,
        "strategies_tested": strategies,
        "fault_levels_tested": fault_levels,
        "recovery_strategies_tested": recovery_strategies,
        "total_combinations": len(strategies) * len(fault_levels) * len(recovery_strategies),
        "results": [],
        "summary": {
            "total_tests": 0,
            "successful_tests": 0,
            "passed_validations": 0,
            "warning_validations": 0,
            "failed_validations": 0,
            "error_tests": 0
        }
    }
    
    # Run tests for each combination
    total_combinations = len(strategies) * len(fault_levels) * len(recovery_strategies)
    completed = 0
    
    logger.info(f"Running {total_combinations} test combinations")
    
    for strategy in strategies:
        for fault_level in fault_levels:
            for recovery_strategy in recovery_strategies:
                # Build test scenarios list
                if args.comprehensive or args.all_scenarios:
                    test_scenarios = DEFAULT_TEST_SCENARIOS
                else:
                    # Adjust based on fault tolerance level
                    if fault_level == "high":
                        test_scenarios = ["connection_lost", "browser_crash", "component_timeout", "multi_browser_failure"]
                    elif fault_level == "medium":
                        test_scenarios = ["connection_lost", "browser_crash", "component_timeout"]
                    else:
                        test_scenarios = ["connection_lost", "component_timeout"]
                
                # Run validation for this configuration
                test_result = await run_validation_for_configuration(
                    model_name=args.model,
                    model_type=model_type,
                    shards=args.shards,
                    browsers=browsers,
                    strategy=strategy,
                    fault_tolerance_level=fault_level,
                    recovery_strategy=recovery_strategy,
                    test_scenarios=test_scenarios,
                    timeout=args.timeout,
                    comprehensive=args.comprehensive
                )
                
                # Add to results
                results["results"].append(test_result)
                
                # Update summary
                results["summary"]["total_tests"] += 1
                
                if test_result["status"] == "error":
                    results["summary"]["error_tests"] += 1
                elif test_result["status"] in ["initialization_failed", "failed"]:
                    results["summary"]["failed_validations"] += 1
                elif test_result["status"] == "warning":
                    results["summary"]["warning_validations"] += 1
                elif test_result["status"] == "passed":
                    results["summary"]["passed_validations"] += 1
                    results["summary"]["successful_tests"] += 1
                
                completed += 1
                logger.info(f"Completed {completed}/{total_combinations} combinations")
                
                # Check if we should stop on first failure
                if args.stop_on_failure and test_result["status"] in ["error", "failed", "initialization_failed"]:
                    logger.warning("Stopping testing due to failure and --stop-on-failure flag")
                    break
            
            if args.stop_on_failure and results["summary"]["error_tests"] + results["summary"]["failed_validations"] > 0:
                break
        
        if args.stop_on_failure and results["summary"]["error_tests"] + results["summary"]["failed_validations"] > 0:
            break
    
    # Add pass rate to summary
    results["summary"]["pass_rate"] = results["summary"]["passed_validations"] / results["summary"]["total_tests"] if results["summary"]["total_tests"] > 0 else 0
    
    # Generate overall status
    if results["summary"]["error_tests"] > 0:
        results["overall_status"] = "error"
    elif results["summary"]["failed_validations"] > 0:
        results["overall_status"] = "failed"
    elif results["summary"]["warning_validations"] > 0:
        results["overall_status"] = "warning"
    else:
        results["overall_status"] = "passed"
    
    return results

def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save test results to a file.
    
    Args:
        results: Test results
        output_path: Path to output file
    """
    try:
        # Create parent directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def print_summary(results: Dict[str, Any]) -> None:
    """
    Print summary of test results.
    
    Args:
        results: Test results
    """
    summary = results["summary"]
    overall_status = results["overall_status"]
    
    print("\n" + "="*80)
    print(f"FAULT TOLERANCE VALIDATION SUMMARY")
    print("="*80)
    print(f"Model: {results['model']}")
    print(f"Browsers: {', '.join(results['browsers'])}")
    print(f"Strategies tested: {', '.join(results['strategies_tested'])}")
    print(f"Fault levels tested: {', '.join(results['fault_levels_tested'])}")
    print(f"Recovery strategies tested: {', '.join(results['recovery_strategies_tested'])}")
    print(f"Total test combinations: {results['total_combinations']}")
    print(f"Completed tests: {summary['total_tests']}")
    print("-"*80)
    print(f"Passed: {summary['passed_validations']} ({summary['pass_rate']:.1%})")
    print(f"Warnings: {summary['warning_validations']}")
    print(f"Failed: {summary['failed_validations']}")
    print(f"Errors: {summary['error_tests']}")
    print("-"*80)
    
    # Set color for overall status
    if overall_status == "passed":
        status_color = "\033[92m"  # Green
    elif overall_status == "warning":
        status_color = "\033[93m"  # Yellow
    else:
        status_color = "\033[91m"  # Red
    
    print(f"Overall status: {status_color}{overall_status}\033[0m")
    print("="*80)
    
    # Print results for each combination
    if results.get("results"):
        print("\nTEST COMBINATION RESULTS:")
        print("-"*80)
        
        for i, result in enumerate(results["results"], 1):
            # Set color for status
            if result["status"] == "passed":
                status_color = "\033[92m"  # Green
            elif result["status"] == "warning":
                status_color = "\033[93m"  # Yellow
            else:
                status_color = "\033[91m"  # Red
            
            print(f"{i}. {result['strategy']} + {result['fault_tolerance_level']} + {result['recovery_strategy']}: {status_color}{result['status']}\033[0m")
            
            # If there was an error, print it
            if result["status"] == "error" and "error" in result:
                print(f"   Error: {result['error']}")
            
            # Print analysis if available
            if "analysis" in result and result["analysis"]:
                # Strengths
                if "strengths" in result["analysis"] and result["analysis"]["strengths"]:
                    print("   Strengths:")
                    for strength in result["analysis"]["strengths"][:2]:  # Show first 2
                        print(f"   ✓ {strength}")
                
                # Weaknesses
                if "weaknesses" in result["analysis"] and result["analysis"]["weaknesses"]:
                    print("   Weaknesses:")
                    for weakness in result["analysis"]["weaknesses"][:2]:  # Show first 2
                        print(f"   ✗ {weakness}")
                
                # Recovery time if available
                if "avg_recovery_time_ms" in result["analysis"]:
                    print(f"   Average recovery time: {result['analysis']['avg_recovery_time_ms']:.2f}ms")
            
            print("-"*80)
    
    print("\n")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Fault-Tolerant Cross-Browser Model Sharding Validation")
    
    # Model selection options
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                      help="Model name to test")
    parser.add_argument("--model-type", type=str, choices=["text", "vision", "audio", "multimodal"],
                      help="Type of model (will be auto-detected if not specified)")
    parser.add_argument("--default-family", type=str, default="bert",
                      help="Default model family if cannot be detected")
    parser.add_argument("--list-models", action="store_true",
                      help="List supported model families and exit")
    
    # Browser options
    parser.add_argument("--browsers", type=str, default="chrome,firefox,edge",
                      help="Comma-separated list of browsers to use")
    
    # Sharding options
    parser.add_argument("--shards", type=int, default=3,
                      help="Number of shards to create")
    parser.add_argument("--strategy", type=str, choices=DEFAULT_SHARDING_STRATEGIES,
                      help="Sharding strategy to use")
    parser.add_argument("--all-strategies", action="store_true",
                      help="Test all sharding strategies")
    
    # Fault tolerance options
    parser.add_argument("--fault-level", type=str, choices=DEFAULT_FAULT_TOLERANCE_LEVELS,
                      help="Fault tolerance level to use")
    parser.add_argument("--all-fault-levels", action="store_true",
                      help="Test all fault tolerance levels")
    parser.add_argument("--recovery-strategy", type=str, choices=DEFAULT_RECOVERY_STRATEGIES,
                      help="Recovery strategy to use")
    parser.add_argument("--all-recovery-strategies", action="store_true",
                      help="Test all recovery strategies")
    
    # Testing options
    parser.add_argument("--all-scenarios", action="store_true",
                      help="Test all failure scenarios")
    parser.add_argument("--timeout", type=int, default=300,
                      help="Timeout in seconds for tests")
    parser.add_argument("--stop-on-failure", action="store_true",
                      help="Stop testing on first failure")
    parser.add_argument("--comprehensive", action="store_true",
                      help="Run comprehensive tests with all options")
    
    # Output options
    parser.add_argument("--output", type=str,
                      help="Path to output file for test results (JSON)")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # List models if requested
    if args.list_models:
        print("Supported model families:")
        for family, details in MODEL_FAMILIES.items():
            sizes = ", ".join(details["sizes"])
            print(f"  - {family} ({details['model_type']}): {sizes}")
        return 0
    
    # Check if required modules are available
    if not MODULES_AVAILABLE:
        logger.error("Required modules not available")
        return 1
    
    # Configure for comprehensive testing
    if args.comprehensive:
        args.all_strategies = True
        args.all_fault_levels = True
        args.all_recovery_strategies = True
        args.all_scenarios = True
        logger.info("Running in comprehensive test mode")
    
    # Run tests
    try:
        logger.info(f"Starting fault tolerance validation for {args.model}")
        results = anyio.run(run_validation_tests(args))
        
        # Print summary
        print_summary(results)
        
        # Save results if output path specified
        if args.output:
            save_results(results, args.output)
        
        # Determine exit code based on overall status
        if results["overall_status"] == "passed":
            logger.info("All tests passed")
            return 0
        elif results["overall_status"] == "warning":
            logger.warning("Tests completed with warnings")
            return 0  # Still consider warnings as success for automation
        elif results["overall_status"] == "failed":
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