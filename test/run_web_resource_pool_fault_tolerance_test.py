#!/usr/bin/env python3
"""
Enhanced WebGPU/WebNN Resource Pool Fault Tolerance Test Suite

This script provides a comprehensive framework for testing fault tolerance in WebGPU/WebNN
resource pool implementations, especially for cross-browser model sharding. It supports 
various testing modes including single tests, targeted scenarios, and comprehensive 
end-to-end testing across all sharding strategies.

Usage:
    # Run basic test with mock implementation (no real browsers needed)
    python run_web_resource_pool_fault_tolerance_test.py --mock

    # Run comprehensive test with mock implementation
    python run_web_resource_pool_fault_tolerance_test.py --mock --comprehensive
    
    # Run cross-browser model sharding test with all sharding strategies
    python run_web_resource_pool_fault_tolerance_test.py --cross-browser-sharding --all-strategies
    
    # Test specific sharding strategy with specific recovery approach
    python run_web_resource_pool_fault_tolerance_test.py --model bert-base-uncased --sharding-strategy layer --recovery-strategy coordinated
    
    # Run comprehensive end-to-end test across all strategies
    python run_web_resource_pool_fault_tolerance_test.py --e2e --comprehensive
"""

import os
import sys
import subprocess
import argparse
import logging
import anyio
import json
import time
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for sharding strategies and recovery mechanisms
SHARDING_STRATEGIES = ["layer", "attention_feedforward", "component", "hybrid", "pipeline"]
RECOVERY_STRATEGIES = ["simple", "progressive", "parallel", "coordinated"]
FAILURE_MODES = ["connection_lost", "browser_crash", "memory_pressure", "component_timeout", 
                "multiple_browser_failures", "staggered_failures", "browser_reload"]
BROWSER_COMBINATIONS = {
    "minimal": "chrome",
    "standard": "chrome,firefox",
    "extended": "chrome,firefox,edge",
    "comprehensive": "chrome,firefox,edge,safari"
}

class FaultToleranceTestRunner:
    """Enhanced test runner for WebGPU/WebNN Resource Pool Fault Tolerance tests"""
    
    def __init__(self, args):
        """Initialize the test runner with command-line arguments"""
        self.args = args
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = args.output_dir or f"./fault_tolerance_test_results_{self.timestamp}"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set verbose logging if requested
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Create log file
        log_path = os.path.join(self.output_dir, f"test_run_{self.timestamp}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        logger.info(f"Initialized WebGPU/WebNN Resource Pool Fault Tolerance Test Runner")
        logger.info(f"Results will be saved to: {self.output_dir}")
    
    async def run_tests(self):
        """Run all tests based on command-line arguments"""
        start_time = time.time()
        results = []
        
        # Determine which test mode to run
        if self.args.e2e:
            # Run end-to-end tests
            logger.info("Running end-to-end tests across all sharding strategies")
            e2e_results = await self._run_e2e_tests()
            results.extend(e2e_results)
        
        elif self.args.cross_browser_sharding:
            # Run cross-browser model sharding tests
            logger.info("Running cross-browser model sharding tests")
            sharding_results = await self._run_cross_browser_sharding_tests()
            results.extend(sharding_results)
        
        else:
            # Run standard integration tests (original behavior)
            logger.info("Running standard integration tests")
            integration_results = await self._run_integration_tests()
            results.append(integration_results)
        
        # Generate and save summary
        total_duration = time.time() - start_time
        summary = self._generate_summary(results, total_duration)
        
        summary_file = os.path.join(self.output_dir, f"test_summary_{self.timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary to console
        self._print_summary(summary)
        
        logger.info(f"All tests completed in {total_duration:.2f}s. Summary saved to {summary_file}")
        
        # Return success if all tests passed
        return 0 if summary["overall_status"] == "success" else 1
    
    async def _run_integration_tests(self):
        """Run standard integration tests (original behavior)"""
        # Build command for standard integration test
        cmd = ["python", "test_web_resource_pool_fault_tolerance_integration.py"]
        
        # Add arguments
        cmd.extend(["--model", self.args.model])
        cmd.extend(["--browsers", self.args.browsers])
        cmd.extend(["--output-dir", self.output_dir])
        
        if self.args.mock:
            cmd.append("--mock")
        
        if self.args.basic:
            cmd.append("--basic")
        
        if self.args.comparative:
            cmd.append("--comparative")
            
        if self.args.stress_test:
            cmd.append("--stress-test")
            cmd.extend(["--iterations", str(self.args.iterations)])
            
        if self.args.resource_pool:
            cmd.append("--resource-pool")
            
        if self.args.comprehensive:
            cmd.append("--comprehensive")
            
        if self.args.verbose:
            cmd.append("--verbose")
        
        # Log the command
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Update the command to use the fixed mock implementation
        if self.args.mock:
            # Add environment variable to use fixed mock implementation
            os.environ["USE_FIXED_MOCK"] = "1"
        
        # Execute the integration test
        try:
            # Use asyncio subprocess for better integration with other async tests
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            stdout = stdout.decode() if stdout else ""
            stderr = stderr.decode() if stderr else ""
            
            # Log output
            if stdout:
                logger.debug(f"Standard output: {stdout}")
            if stderr:
                logger.warning(f"Standard error: {stderr}")
            
            result = {
                "test_type": "integration",
                "command": " ".join(cmd),
                "exit_code": process.returncode,
                "status": "success" if process.returncode == 0 else "failure",
                "output_dir": self.output_dir
            }
            
            logger.info(f"Integration test completed with exit code: {process.returncode}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running integration test: {e}")
            return {
                "test_type": "integration",
                "command": " ".join(cmd),
                "exit_code": 1,
                "status": "error",
                "error": str(e)
            }
    
    async def _run_cross_browser_sharding_tests(self):
        """Run cross-browser model sharding tests"""
        results = []
        
        # Get sharding strategies to test
        strategies = []
        if self.args.all_strategies:
            strategies = SHARDING_STRATEGIES
        elif self.args.sharding_strategy:
            strategies = [self.args.sharding_strategy]
        else:
            # Default to layer-based sharding
            strategies = ["layer"]
        
        # Get recovery strategies to test
        recovery_strategies = []
        if self.args.all_recovery_strategies:
            recovery_strategies = RECOVERY_STRATEGIES
        elif self.args.recovery_strategy:
            recovery_strategies = [self.args.recovery_strategy]
        else:
            # Default to progressive recovery
            recovery_strategies = ["progressive"]
        
        # Get browsers to test
        browsers = self.args.browsers
        if self.args.browser_combination:
            if self.args.browser_combination in BROWSER_COMBINATIONS:
                browsers = BROWSER_COMBINATIONS[self.args.browser_combination]
            
        # Determine test mode (comprehensive or specific)
        if self.args.comprehensive:
            # Test all combinations for comprehensive coverage
            for strategy in strategies:
                for recovery_strategy in recovery_strategies:
                    # Build command for comprehensive test
                    cmd = ["python", "test_fault_tolerant_cross_browser_model_sharding.py"]
                    
                    cmd.extend(["--model", self.args.model])
                    cmd.extend(["--type", strategy])
                    cmd.extend(["--fault-tolerance", "--fault-tolerance-level", "high", "--recovery-strategy", recovery_strategy])
                    cmd.append("--simulate-failure")
                    cmd.append("--performance-test")
                    cmd.append("--resource-pool-integration")
                    cmd.append("--use-performance-history")
                    cmd.extend(["--output", os.path.join(self.output_dir, f"sharding_{strategy}_{recovery_strategy}_{self.timestamp}.json")])
                    
                    if self.args.verbose:
                        cmd.append("--verbose")
                    
                    # Log the command
                    logger.info(f"Running cross-browser sharding test with {strategy} strategy and {recovery_strategy} recovery")
                    
                    # Execute the test
                    try:
                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE
                        )
                        
                        stdout, stderr = await process.communicate()
                        stdout = stdout.decode() if stdout else ""
                        stderr = stderr.decode() if stderr else ""
                        
                        # Log output
                        if stdout:
                            logger.debug(f"Standard output: {stdout}")
                        if stderr:
                            logger.warning(f"Standard error: {stderr}")
                        
                        result = {
                            "test_type": "cross_browser_sharding",
                            "strategy": strategy,
                            "recovery_strategy": recovery_strategy,
                            "command": " ".join(cmd),
                            "exit_code": process.returncode,
                            "status": "success" if process.returncode == 0 else "failure"
                        }
                        
                        results.append(result)
                        logger.info(f"Sharding test with {strategy} strategy and {recovery_strategy} recovery completed with exit code: {process.returncode}")
                        
                    except Exception as e:
                        logger.error(f"Error running sharding test with {strategy} strategy: {e}")
                        results.append({
                            "test_type": "cross_browser_sharding",
                            "strategy": strategy,
                            "recovery_strategy": recovery_strategy,
                            "exit_code": 1,
                            "status": "error",
                            "error": str(e)
                        })
        else:
            # Run a single test with specified parameters
            strategy = strategies[0]
            recovery_strategy = recovery_strategies[0]
            
            # Build command for specific test
            cmd = ["python", "test_fault_tolerant_cross_browser_model_sharding.py"]
            
            cmd.extend(["--model", self.args.model])
            cmd.extend(["--type", strategy])
            
            # Add fault tolerance parameters if requested
            if self.args.fault_tolerance:
                cmd.extend(["--fault-tolerance", "--fault-tolerance-level", self.args.fault_tolerance_level or "medium"])
                cmd.extend(["--recovery-strategy", recovery_strategy])
                
                if self.args.simulate_failure:
                    cmd.append("--simulate-failure")
                    if self.args.failure_type:
                        cmd.extend(["--failure-type", self.args.failure_type])
            
            # Add performance testing if requested
            if self.args.performance_test:
                cmd.append("--performance-test")
                cmd.extend(["--iterations", str(self.args.iterations or 10)])
            
            if self.args.stress_test:
                cmd.append("--stress-test")
            
            # Add resource pool integration if requested
            if self.args.resource_pool:
                cmd.append("--resource-pool-integration")
            
            # Add output file
            cmd.extend(["--output", os.path.join(self.output_dir, f"sharding_{strategy}_{recovery_strategy}_{self.timestamp}.json")])
            
            if self.args.verbose:
                cmd.append("--verbose")
            
            # Log the command
            logger.info(f"Running cross-browser sharding test with {strategy} strategy")
            
            # Execute the test
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                stdout = stdout.decode() if stdout else ""
                stderr = stderr.decode() if stderr else ""
                
                # Log output
                if stdout:
                    logger.debug(f"Standard output: {stdout}")
                if stderr:
                    logger.warning(f"Standard error: {stderr}")
                
                result = {
                    "test_type": "cross_browser_sharding",
                    "strategy": strategy,
                    "recovery_strategy": recovery_strategy,
                    "command": " ".join(cmd),
                    "exit_code": process.returncode,
                    "status": "success" if process.returncode == 0 else "failure"
                }
                
                results.append(result)
                logger.info(f"Sharding test with {strategy} strategy completed with exit code: {process.returncode}")
                
            except Exception as e:
                logger.error(f"Error running sharding test with {strategy} strategy: {e}")
                results.append({
                    "test_type": "cross_browser_sharding",
                    "strategy": strategy,
                    "recovery_strategy": recovery_strategy,
                    "exit_code": 1,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    async def _run_e2e_tests(self):
        """Run comprehensive end-to-end tests across all sharding strategies"""
        results = []
        
        # Build command for the test suite runner
        cmd = ["python", "run_cross_browser_model_sharding_test_suite.py"]
        
        # Basic parameters
        if self.args.model:
            cmd.extend(["--models", self.args.model])
        
        if self.args.browsers:
            cmd.extend(["--browsers", self.args.browsers])
        elif self.args.browser_combination:
            cmd.extend(["--browser-combination", self.args.browser_combination])
        
        # Add strategy parameters
        if self.args.all_strategies:
            cmd.append("--all-sharding-strategies")
        elif self.args.sharding_strategy:
            cmd.extend(["--sharding-strategies", self.args.sharding_strategy])
        
        # Add fault tolerance parameters
        if self.args.all_recovery_strategies:
            cmd.append("--all-recovery-strategies")
        elif self.args.recovery_strategy:
            cmd.extend(["--recovery-strategies", self.args.recovery_strategy])
        
        if self.args.fault_tolerance_level:
            cmd.extend(["--fault-tolerance-levels", self.args.fault_tolerance_level])
        
        # Comprehensive testing
        if self.args.comprehensive:
            cmd.append("--comprehensive")
        
        # Add specific test scenarios
        if self.args.fault_tolerance_only:
            cmd.append("--fault-tolerance-only")
        
        if self.args.performance_test:
            cmd.append("--performance-only")
        
        # Configure output directory
        e2e_output_dir = os.path.join(self.output_dir, f"e2e_tests_{self.timestamp}")
        cmd.extend(["--output-dir", e2e_output_dir])
        
        # Set concurrency level
        if self.args.concurrent_tests:
            cmd.extend(["--concurrent-tests", str(self.args.concurrent_tests)])
        
        if self.args.verbose:
            cmd.append("--verbose")
        
        # Log the command
        logger.info(f"Running end-to-end test suite: {' '.join(cmd)}")
        
        # Execute the test suite
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            stdout = stdout.decode() if stdout else ""
            stderr = stderr.decode() if stderr else ""
            
            # Log output
            if stdout:
                logger.debug(f"Standard output: {stdout}")
            if stderr:
                logger.warning(f"Standard error: {stderr}")
            
            # Try to parse results summary if available
            summary_path = os.path.join(e2e_output_dir, f"test_suite_summary_{self.timestamp}.json")
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r') as f:
                        e2e_summary = json.load(f)
                        
                    # Create detailed results
                    for model, data in e2e_summary.get("by_model", {}).items():
                        for strategy, strategy_data in e2e_summary.get("by_strategy", {}).items():
                            results.append({
                                "test_type": "e2e",
                                "model": model,
                                "strategy": strategy,
                                "success_rate": strategy_data.get("success_rate", 0),
                                "status": "success" if strategy_data.get("success_rate", 0) > 0.9 else "failure"
                            })
                except Exception as e:
                    logger.error(f"Error parsing E2E test summary: {e}")
            
            # Add overall result
            results.append({
                "test_type": "e2e_overall",
                "command": " ".join(cmd),
                "exit_code": process.returncode,
                "output_dir": e2e_output_dir,
                "status": "success" if process.returncode == 0 else "failure"
            })
            
            logger.info(f"End-to-end test suite completed with exit code: {process.returncode}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running end-to-end test suite: {e}")
            return [{
                "test_type": "e2e_overall",
                "command": " ".join(cmd),
                "exit_code": 1,
                "status": "error",
                "error": str(e)
            }]
    
    def _generate_summary(self, results, total_duration):
        """Generate a summary of all test results"""
        summary = {
            "timestamp": self.timestamp,
            "total_tests": len(results),
            "successful_tests": sum(1 for r in results if r["status"] == "success"),
            "failed_tests": sum(1 for r in results if r["status"] == "failure"),
            "error_tests": sum(1 for r in results if r["status"] == "error"),
            "total_duration_seconds": total_duration,
            "by_test_type": {},
            "by_strategy": {},
            "by_recovery_strategy": {}
        }
        
        # Group results by test type
        for result in results:
            test_type = result["test_type"]
            summary["by_test_type"].setdefault(test_type, {"total": 0, "success": 0, "failure": 0, "error": 0})
            summary["by_test_type"][test_type]["total"] += 1
            
            if result["status"] == "success":
                summary["by_test_type"][test_type]["success"] += 1
            elif result["status"] == "failure":
                summary["by_test_type"][test_type]["failure"] += 1
            else:
                summary["by_test_type"][test_type]["error"] += 1
        
        # Group results by strategy (for sharding tests)
        for result in results:
            if "strategy" in result:
                strategy = result["strategy"]
                summary["by_strategy"].setdefault(strategy, {"total": 0, "success": 0, "failure": 0, "error": 0})
                summary["by_strategy"][strategy]["total"] += 1
                
                if result["status"] == "success":
                    summary["by_strategy"][strategy]["success"] += 1
                elif result["status"] == "failure":
                    summary["by_strategy"][strategy]["failure"] += 1
                else:
                    summary["by_strategy"][strategy]["error"] += 1
        
        # Group results by recovery strategy (for sharding tests)
        for result in results:
            if "recovery_strategy" in result:
                recovery = result["recovery_strategy"]
                summary["by_recovery_strategy"].setdefault(recovery, {"total": 0, "success": 0, "failure": 0, "error": 0})
                summary["by_recovery_strategy"][recovery]["total"] += 1
                
                if result["status"] == "success":
                    summary["by_recovery_strategy"][recovery]["success"] += 1
                elif result["status"] == "failure":
                    summary["by_recovery_strategy"][recovery]["failure"] += 1
                else:
                    summary["by_recovery_strategy"][recovery]["error"] += 1
        
        # Calculate success rates
        for category in [summary["by_test_type"], summary["by_strategy"], summary["by_recovery_strategy"]]:
            for item in category.values():
                item["success_rate"] = item["success"] / item["total"] if item["total"] > 0 else 0
        
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
        
        return summary
    
    def _print_summary(self, summary):
        """Print a formatted summary to the console"""
        print("\n" + "="*80)
        print(f"WEBGPU/WEBNN RESOURCE POOL FAULT TOLERANCE TEST SUMMARY")
        print("="*80)
        print(f"Total tests:     {summary['total_tests']}")
        print(f"Successful:      {summary['successful_tests']} ({summary['success_rate']*100:.1f}%)")
        print(f"Failed:          {summary['failed_tests']}")
        print(f"Errors:          {summary['error_tests']}")
        print(f"Total duration:  {summary['total_duration_seconds']/60:.1f} minutes")
        print("-"*80)
        
        # Print results by test type
        print("\nRESULTS BY TEST TYPE:")
        for test_type, data in summary["by_test_type"].items():
            print(f"  {test_type.ljust(20)}: {data['success_rate']*100:.1f}% success ({data['success']}/{data['total']})")
        
        # Print results by strategy if available
        if summary["by_strategy"]:
            print("\nRESULTS BY SHARDING STRATEGY:")
            for strategy, data in summary["by_strategy"].items():
                print(f"  {strategy.ljust(20)}: {data['success_rate']*100:.1f}% success ({data['success']}/{data['total']})")
        
        # Print results by recovery strategy if available
        if summary["by_recovery_strategy"]:
            print("\nRESULTS BY RECOVERY STRATEGY:")
            for strategy, data in summary["by_recovery_strategy"].items():
                print(f"  {strategy.ljust(15)}: {data['success_rate']*100:.1f}% success ({data['success']}/{data['total']})")
        
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
        print(f"Results available in: {self.output_dir}")
        print("="*80)

async def async_main(args):
    """Async main entry point"""
    # Create test runner
    runner = FaultToleranceTestRunner(args)
    
    # Run tests
    try:
        return await runner.run_tests()
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced WebGPU/WebNN Resource Pool Fault Tolerance Test Suite"
    )
    
    # Test mode selection
    mode_group = parser.add_argument_group("Test Modes")
    mode_group.add_argument("--cross-browser-sharding", action="store_true",
                          help="Run cross-browser model sharding tests")
    mode_group.add_argument("--e2e", action="store_true",
                          help="Run comprehensive end-to-end tests")
    
    # Basic options (original behavior)
    basic_group = parser.add_argument_group("Basic Options")
    basic_group.add_argument("--model", type=str, default="bert-base-uncased",
                           help="Model name to test")
    basic_group.add_argument("--browsers", type=str, default="chrome,firefox,edge",
                           help="Comma-separated list of browsers to use")
    basic_group.add_argument("--mock", action="store_true",
                           help="Use mock implementation for testing without browsers")
    basic_group.add_argument("--basic", action="store_true",
                           help="Run basic integration test")
    basic_group.add_argument("--comparative", action="store_true",
                           help="Run comparative integration test")
    basic_group.add_argument("--stress-test", action="store_true",
                           help="Run stress test integration")
    basic_group.add_argument("--resource-pool", action="store_true",
                           help="Test integration with resource pool")
    
    # Enhanced sharding options
    sharding_group = parser.add_argument_group("Sharding Options")
    sharding_group.add_argument("--sharding-strategy", type=str, choices=SHARDING_STRATEGIES,
                              help="Specific sharding strategy to test")
    sharding_group.add_argument("--all-strategies", action="store_true",
                              help="Test all sharding strategies")
    
    # Fault tolerance options
    fault_group = parser.add_argument_group("Fault Tolerance Options")
    fault_group.add_argument("--fault-tolerance", action="store_true",
                           help="Enable fault tolerance testing")
    fault_group.add_argument("--fault-tolerance-level", type=str, 
                           choices=["low", "medium", "high", "critical"],
                           help="Fault tolerance level to test")
    fault_group.add_argument("--fault-tolerance-only", action="store_true",
                           help="Run only fault tolerance test scenarios")
    fault_group.add_argument("--recovery-strategy", type=str, choices=RECOVERY_STRATEGIES,
                           help="Specific recovery strategy to test")
    fault_group.add_argument("--all-recovery-strategies", action="store_true",
                           help="Test all recovery strategies")
    fault_group.add_argument("--failure-type", type=str, choices=FAILURE_MODES,
                           help="Specific failure mode to test")
    fault_group.add_argument("--simulate-failure", action="store_true",
                           help="Simulate browser failure during tests")
    
    # Browser options
    browser_group = parser.add_argument_group("Browser Options")
    browser_group.add_argument("--browser-combination", type=str, choices=list(BROWSER_COMBINATIONS.keys()),
                             help="Predefined browser combination to use")
    
    # Performance testing options
    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument("--performance-test", action="store_true",
                          help="Run performance tests")
    perf_group.add_argument("--iterations", type=int, default=5,
                          help="Number of iterations for performance/stress testing")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-dir", type=str,
                            help="Directory for output files")
    output_group.add_argument("--verbose", action="store_true",
                            help="Enable verbose logging")
    
    # Advanced options
    advanced_group = parser.add_argument_group("Advanced Options")
    advanced_group.add_argument("--comprehensive", action="store_true",
                              help="Run comprehensive tests with all options")
    advanced_group.add_argument("--concurrent-tests", type=int, default=1,
                              help="Number of tests to run concurrently (for e2e testing)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Special behavior for comprehensive flag
    if args.comprehensive:
        # If e2e mode is active, don't set other options that might conflict
        if not args.e2e and not args.cross_browser_sharding:
            # For legacy mode, set all the original flags
            args.basic = True
            args.comparative = True
            args.stress_test = True
            args.resource_pool = True
    
    # Run tests with asyncio event loop
    return anyio.run(async_main(args))

if __name__ == "__main__":
    sys.exit(main())