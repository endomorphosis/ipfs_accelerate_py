#!/usr/bin/env python3
"""
Comprehensive End-to-End Testing for Fault-Tolerant Cross-Browser Model Sharding

This script orchestrates comprehensive end-to-end testing across all sharding strategies
for the Fault-Tolerant Cross-Browser Model Sharding component. It tests various models,
sharding strategies, recovery approaches, and failure scenarios to ensure complete
coverage and robust fault tolerance.

Usage:
    # Run comprehensive tests across all strategies
    python run_comprehensive_ft_sharding_tests.py --comprehensive
    
    # Test specific model with all sharding strategies
    python run_comprehensive_ft_sharding_tests.py --model bert-base-uncased --all-strategies
    
    # Test with specific recovery strategy across all models
    python run_comprehensive_ft_sharding_tests.py --recovery-strategy coordinated --all-models
    
    # Test specific failure mode
    python run_comprehensive_ft_sharding_tests.py --failure-mode browser_crash
"""

import os
import sys
import argparse
import subprocess
import logging
import json
import datetime
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OUTPUT_DIR = "./test_results/comprehensive_ft_sharding"
MODEL_LIST = [
    {"name": "bert-base-uncased", "type": "text"},
    {"name": "vit-base-patch16-224", "type": "vision"},
    {"name": "whisper-tiny", "type": "audio"},
    {"name": "clip-vit-base-patch32", "type": "multimodal"}
]
SHARDING_STRATEGIES = ["layer", "attention_feedforward", "component", "hybrid", "pipeline"]
RECOVERY_STRATEGIES = ["simple", "progressive", "parallel", "coordinated"]
FAULT_TOLERANCE_LEVELS = ["medium", "high"]
FAILURE_MODES = ["connection_lost", "browser_crash", "memory_pressure", "component_timeout"]
BROWSER_COMBINATIONS = {
    "minimal": "chrome",
    "standard": "chrome,firefox",
    "extended": "chrome,firefox,edge",
    "comprehensive": "chrome,firefox,edge,safari"
}

def main():
    """Main entry point for the comprehensive test runner"""
    parser = argparse.ArgumentParser(
        description="Comprehensive End-to-End Testing for Fault-Tolerant Cross-Browser Model Sharding"
    )
    
    # Model selection options
    parser.add_argument("--model", type=str,
                      help="Specific model to test")
    parser.add_argument("--all-models", action="store_true",
                      help="Test all representative models")
    
    # Sharding strategy options
    parser.add_argument("--sharding-strategy", type=str, choices=SHARDING_STRATEGIES,
                      help="Specific sharding strategy to test")
    parser.add_argument("--all-strategies", action="store_true",
                      help="Test all sharding strategies")
    
    # Fault tolerance options
    parser.add_argument("--fault-tolerance-level", type=str, choices=FAULT_TOLERANCE_LEVELS,
                      help="Fault tolerance level to test")
    parser.add_argument("--recovery-strategy", type=str, choices=RECOVERY_STRATEGIES,
                      help="Recovery strategy to test")
    parser.add_argument("--failure-mode", type=str, choices=FAILURE_MODES,
                      help="Failure mode to test")
    
    # Browser options
    parser.add_argument("--browser-combination", type=str, choices=list(BROWSER_COMBINATIONS.keys()),
                      default="standard",
                      help="Browser combination to use")
    
    # Test scope options
    parser.add_argument("--comprehensive", action="store_true",
                      help="Run comprehensive tests with many combinations")
    parser.add_argument("--quick", action="store_true",
                      help="Run a quick test with minimal combinations")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                      help="Directory for output files")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create timestamp for output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"test_run_{timestamp}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure file logging
    log_file = os.path.join(output_dir, f"comprehensive_test_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine models to test
    models_to_test = []
    if args.all_models or args.comprehensive:
        # Use all models
        models_to_test = MODEL_LIST
    elif args.model:
        # Find model type for the specified model
        model_type = next((m["type"] for m in MODEL_LIST if m["name"] == args.model), "text")
        models_to_test = [{"name": args.model, "type": model_type}]
    else:
        # Default to BERT
        models_to_test = [{"name": "bert-base-uncased", "type": "text"}]
    
    # Determine sharding strategies to test
    strategies_to_test = []
    if args.all_strategies or args.comprehensive:
        # Use all strategies
        strategies_to_test = SHARDING_STRATEGIES
    elif args.sharding_strategy:
        # Use specified strategy
        strategies_to_test = [args.sharding_strategy]
    elif args.quick:
        # Use minimal subset for quick tests
        strategies_to_test = ["layer", "component"]
    else:
        # Default to layer-based sharding
        strategies_to_test = ["layer"]
    
    # Determine recovery strategies to test
    recovery_strategies_to_test = []
    if args.comprehensive:
        # Use all recovery strategies
        recovery_strategies_to_test = RECOVERY_STRATEGIES
    elif args.recovery_strategy:
        # Use specified recovery strategy
        recovery_strategies_to_test = [args.recovery_strategy]
    elif args.quick:
        # Use minimal subset for quick tests
        recovery_strategies_to_test = ["progressive"]
    else:
        # Default to progressive and coordinated for good coverage
        recovery_strategies_to_test = ["progressive", "coordinated"]
    
    # Determine fault tolerance levels to test
    ft_levels_to_test = []
    if args.comprehensive:
        # Use all fault tolerance levels
        ft_levels_to_test = FAULT_TOLERANCE_LEVELS
    elif args.fault_tolerance_level:
        # Use specified fault tolerance level
        ft_levels_to_test = [args.fault_tolerance_level]
    elif args.quick:
        # Use minimal subset for quick tests
        ft_levels_to_test = ["medium"]
    else:
        # Default to high for better testing
        ft_levels_to_test = ["high"]
    
    # Determine failure modes to test
    failure_modes_to_test = []
    if args.comprehensive:
        # Use all failure modes
        failure_modes_to_test = FAILURE_MODES
    elif args.failure_mode:
        # Use specified failure mode
        failure_modes_to_test = [args.failure_mode]
    elif args.quick:
        # Use minimal subset for quick tests
        failure_modes_to_test = ["browser_crash"]
    else:
        # Default to most common scenarios
        failure_modes_to_test = ["connection_lost", "browser_crash"]
    
    # Get browser combination
    browser_combo = BROWSER_COMBINATIONS[args.browser_combination]
    
    # Build test matrix
    test_matrix = []
    
    # Determine test scope based on flags
    if args.comprehensive:
        # Full combinatorial testing for comprehensive coverage
        for model in models_to_test:
            for strategy in strategies_to_test:
                for recovery in recovery_strategies_to_test:
                    for ft_level in ft_levels_to_test:
                        for failure in failure_modes_to_test:
                            test_matrix.append({
                                "model": model["name"],
                                "model_type": model["type"],
                                "strategy": strategy,
                                "recovery": recovery,
                                "ft_level": ft_level,
                                "failure": failure
                            })
    elif args.quick:
        # Minimal testing for quick results
        model = models_to_test[0]
        strategy = strategies_to_test[0]
        recovery = recovery_strategies_to_test[0]
        ft_level = ft_levels_to_test[0]
        failure = failure_modes_to_test[0]
        
        test_matrix.append({
            "model": model["name"],
            "model_type": model["type"],
            "strategy": strategy,
            "recovery": recovery,
            "ft_level": ft_level,
            "failure": failure
        })
    else:
        # Balanced testing with some key combinations
        for model in models_to_test:
            for strategy in strategies_to_test:
                for recovery in recovery_strategies_to_test:
                    # Use the first fault tolerance level and failure mode
                    test_matrix.append({
                        "model": model["name"],
                        "model_type": model["type"],
                        "strategy": strategy,
                        "recovery": recovery,
                        "ft_level": ft_levels_to_test[0],
                        "failure": failure_modes_to_test[0]
                    })
    
    # Log test plan
    total_tests = len(test_matrix)
    logger.info(f"Preparing to run {total_tests} comprehensive fault tolerance tests")
    logger.info(f"Models: {[m['name'] for m in models_to_test]}")
    logger.info(f"Strategies: {strategies_to_test}")
    logger.info(f"Recovery strategies: {recovery_strategies_to_test}")
    logger.info(f"Fault tolerance levels: {ft_levels_to_test}")
    logger.info(f"Failure modes: {failure_modes_to_test}")
    logger.info(f"Browser combination: {browser_combo}")
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Run all tests
    results = []
    start_time = time.time()
    completed = 0
    
    for test_config in test_matrix:
        # Build command for test
        cmd = ["python", "run_web_resource_pool_fault_tolerance_test.py", "--cross-browser-sharding"]
        
        # Add model parameters
        cmd.extend(["--model", test_config["model"]])
        
        # Add sharding strategy
        cmd.extend(["--sharding-strategy", test_config["strategy"]])
        
        # Add fault tolerance parameters
        cmd.append("--fault-tolerance")
        cmd.extend(["--fault-tolerance-level", test_config["ft_level"]])
        cmd.extend(["--recovery-strategy", test_config["recovery"]])
        cmd.append("--simulate-failure")
        cmd.extend(["--failure-type", test_config["failure"]])
        
        # Add browser combination
        cmd.extend(["--browsers", browser_combo])
        
        # Add additional testing options
        cmd.append("--performance-test")
        cmd.append("--resource-pool")
        
        # Add output directory
        test_name = f"{test_config['model']}_{test_config['strategy']}_{test_config['recovery']}_{test_config['ft_level']}_{test_config['failure']}"
        test_output_dir = os.path.join(output_dir, test_name)
        cmd.extend(["--output-dir", test_output_dir])
        
        if args.verbose:
            cmd.append("--verbose")
        
        # Log test execution
        logger.info(f"Running test {completed+1}/{total_tests}: {test_name}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Execute test
        try:
            # Create test output directory
            os.makedirs(test_output_dir, exist_ok=True)
            
            # Run the test
            start_test_time = time.time()
            process = subprocess.run(cmd, check=False, capture_output=True, text=True)
            end_test_time = time.time()
            test_duration = end_test_time - start_test_time
            
            # Log test output
            if process.stdout:
                with open(os.path.join(test_output_dir, "stdout.log"), "w") as f:
                    f.write(process.stdout)
                
                if args.verbose:
                    logger.debug(f"Standard output: {process.stdout}")
            
            if process.stderr:
                with open(os.path.join(test_output_dir, "stderr.log"), "w") as f:
                    f.write(process.stderr)
                
                logger.warning(f"Standard error: {process.stderr}")
            
            # Create result object
            result = {
                "test_name": test_name,
                "model": test_config["model"],
                "model_type": test_config["model_type"],
                "sharding_strategy": test_config["strategy"],
                "recovery_strategy": test_config["recovery"],
                "fault_tolerance_level": test_config["ft_level"],
                "failure_mode": test_config["failure"],
                "exit_code": process.returncode,
                "status": "success" if process.returncode == 0 else "failure",
                "duration_seconds": test_duration,
                "output_dir": test_output_dir
            }
            
            # Add to results
            results.append(result)
            
            # Log completion
            logger.info(f"Test {completed+1}/{total_tests} completed with status: {result['status']}")
            logger.info(f"Duration: {test_duration:.2f}s")
            
        except Exception as e:
            # Log error
            logger.error(f"Error running test {test_name}: {e}")
            
            # Add error result
            results.append({
                "test_name": test_name,
                "model": test_config["model"],
                "model_type": test_config["model_type"],
                "sharding_strategy": test_config["strategy"],
                "recovery_strategy": test_config["recovery"],
                "fault_tolerance_level": test_config["ft_level"],
                "failure_mode": test_config["failure"],
                "exit_code": -1,
                "status": "error",
                "error": str(e),
                "output_dir": test_output_dir
            })
        
        # Increment completed counter
        completed += 1
        
        # Log progress
        elapsed_time = time.time() - start_time
        tests_remaining = total_tests - completed
        if completed > 0 and tests_remaining > 0:
            time_per_test = elapsed_time / completed
            estimated_remaining = time_per_test * tests_remaining
            logger.info(f"Progress: {completed}/{total_tests} tests completed ({completed/total_tests*100:.1f}%)")
            logger.info(f"Elapsed time: {elapsed_time/60:.1f} minutes, Estimated remaining: {estimated_remaining/60:.1f} minutes")
    
    # Calculate total duration
    total_duration = time.time() - start_time
    
    # Generate summary
    summary = {
        "timestamp": timestamp,
        "total_tests": total_tests,
        "successful_tests": sum(1 for r in results if r["status"] == "success"),
        "failed_tests": sum(1 for r in results if r["status"] == "failure"),
        "error_tests": sum(1 for r in results if r["status"] == "error"),
        "total_duration_seconds": total_duration
    }
    
    # Group results by various dimensions
    by_model = {}
    by_strategy = {}
    by_recovery = {}
    by_failure = {}
    
    for result in results:
        # By model
        model = result["model"]
        by_model.setdefault(model, {"total": 0, "success": 0, "failure": 0, "error": 0})
        by_model[model]["total"] += 1
        if result["status"] == "success":
            by_model[model]["success"] += 1
        elif result["status"] == "failure":
            by_model[model]["failure"] += 1
        else:
            by_model[model]["error"] += 1
        
        # By strategy
        strategy = result["sharding_strategy"]
        by_strategy.setdefault(strategy, {"total": 0, "success": 0, "failure": 0, "error": 0})
        by_strategy[strategy]["total"] += 1
        if result["status"] == "success":
            by_strategy[strategy]["success"] += 1
        elif result["status"] == "failure":
            by_strategy[strategy]["failure"] += 1
        else:
            by_strategy[strategy]["error"] += 1
        
        # By recovery strategy
        recovery = result["recovery_strategy"]
        by_recovery.setdefault(recovery, {"total": 0, "success": 0, "failure": 0, "error": 0})
        by_recovery[recovery]["total"] += 1
        if result["status"] == "success":
            by_recovery[recovery]["success"] += 1
        elif result["status"] == "failure":
            by_recovery[recovery]["failure"] += 1
        else:
            by_recovery[recovery]["error"] += 1
        
        # By failure mode
        failure = result["failure_mode"]
        by_failure.setdefault(failure, {"total": 0, "success": 0, "failure": 0, "error": 0})
        by_failure[failure]["total"] += 1
        if result["status"] == "success":
            by_failure[failure]["success"] += 1
        elif result["status"] == "failure":
            by_failure[failure]["failure"] += 1
        else:
            by_failure[failure]["error"] += 1
    
    # Calculate success rates
    for category in [by_model, by_strategy, by_recovery, by_failure]:
        for item in category.values():
            item["success_rate"] = item["success"] / item["total"] if item["total"] > 0 else 0
    
    # Add grouped results to summary
    summary["by_model"] = by_model
    summary["by_strategy"] = by_strategy
    summary["by_recovery"] = by_recovery
    summary["by_failure"] = by_failure
    
    # Calculate overall success rate
    summary["success_rate"] = summary["successful_tests"] / summary["total_tests"] if summary["total_tests"] > 0 else 0
    
    # Determine overall status
    if summary["error_tests"] > 0:
        summary["overall_status"] = "error"
    elif summary["failed_tests"] > 0:
        if summary["success_rate"] >= 0.9:
            # Consider it a warning if success rate is still high
            summary["overall_status"] = "warning"
        else:
            summary["overall_status"] = "failure"
    else:
        summary["overall_status"] = "success"
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"detailed_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary to console
    print("\n" + "="*80)
    print(f"COMPREHENSIVE FAULT-TOLERANT CROSS-BROWSER MODEL SHARDING TEST SUMMARY")
    print("="*80)
    print(f"Total tests:     {summary['total_tests']}")
    print(f"Successful:      {summary['successful_tests']} ({summary['success_rate']*100:.1f}%)")
    print(f"Failed:          {summary['failed_tests']}")
    print(f"Errors:          {summary['error_tests']}")
    print(f"Total duration:  {summary['total_duration_seconds']/60:.1f} minutes")
    print("-"*80)
    
    # Print results by category
    print("\nRESULTS BY SHARDING STRATEGY:")
    for strategy, data in by_strategy.items():
        print(f"  {strategy.ljust(20)}: {data['success_rate']*100:.1f}% success ({data['success']}/{data['total']})")
    
    print("\nRESULTS BY RECOVERY STRATEGY:")
    for recovery, data in by_recovery.items():
        print(f"  {recovery.ljust(15)}: {data['success_rate']*100:.1f}% success ({data['success']}/{data['total']})")
    
    print("\nRESULTS BY FAILURE MODE:")
    for failure, data in by_failure.items():
        print(f"  {failure.ljust(25)}: {data['success_rate']*100:.1f}% success ({data['success']}/{data['total']})")
    
    print("\nRESULTS BY MODEL:")
    for model, data in by_model.items():
        print(f"  {model.ljust(20)}: {data['success_rate']*100:.1f}% success ({data['success']}/{data['total']})")
    
    # Print overall status
    status_color = ""
    reset_color = ""
    if summary["overall_status"] == "success":
        status_color = "\033[92m"  # Green
    elif summary["overall_status"] == "warning":
        status_color = "\033[93m"  # Yellow
    else:
        status_color = "\033[91m"  # Red
    
    print("-"*80)
    print(f"Overall status: {status_color}{summary['overall_status'].upper()}{reset_color}")
    print(f"Results available in: {output_dir}")
    print("="*80)
    
    # Log completion
    logger.info(f"All tests completed in {total_duration/60:.1f} minutes")
    logger.info(f"Results available in: {output_dir}")
    
    # Return appropriate exit code
    if summary["overall_status"] == "success":
        return 0
    elif summary["overall_status"] == "warning":
        return 0  # Consider warnings as successful for automation
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())