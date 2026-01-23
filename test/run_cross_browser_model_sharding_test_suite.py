#!/usr/bin/env python3
"""
Comprehensive Cross-Browser Model Sharding Test Suite Runner

This script executes a comprehensive test suite for validating the Fault-Tolerant
Cross-Browser Model Sharding implementation across multiple browsers, model types,
and failure scenarios. It focuses on advanced fault tolerance validation, 
comprehensive metrics collection, and end-to-end testing across all sharding strategies.

Usage:
    python run_cross_browser_model_sharding_test_suite.py --comprehensive
    python run_cross_browser_model_sharding_test_suite.py --fault-tolerance-level high --models bert,whisper,vit
    python run_cross_browser_model_sharding_test_suite.py --sharding-strategies layer,component --browsers chrome,firefox,edge
"""

import os
import sys
import json
import time
import logging
import argparse
import anyio
import concurrent.futures
import datetime
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))

# Import test script
try:
    from test_fault_tolerant_cross_browser_model_sharding import main as run_single_test
    TEST_SCRIPT_AVAILABLE = True
except ImportError as e:
    logger.error(f"Could not import test_fault_tolerant_cross_browser_model_sharding: {e}")
    TEST_SCRIPT_AVAILABLE = False

# Import DuckDB integration if available
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("DuckDB not available, database integration will be disabled")
    DUCKDB_AVAILABLE = False

# Constants
DEFAULT_OUTPUT_DIR = "test_results/cross_browser_model_sharding"
DEFAULT_DB_PATH = "./benchmark_db.duckdb"
TEST_TIMEOUT = 1800  # 30 minutes per test

# Model mappings
MODEL_MAPPING = {
    "text": ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
    "vision": ["vit-base-patch16-224", "resnet-50"],
    "audio": ["whisper-tiny", "wav2vec2-base"],
    "multimodal": ["clip-vit-base-patch32"]
}

# Sharding strategies
SHARDING_STRATEGIES = ["layer", "attention_feedforward", "component"]

# Fault tolerance levels
FAULT_TOLERANCE_LEVELS = ["low", "medium", "high", "critical"]

# Recovery strategies
RECOVERY_STRATEGIES = ["simple", "progressive", "parallel", "coordinated"]

# Browser combinations
BROWSER_COMBINATIONS = {
    "minimal": ["chrome"],
    "basic": ["chrome", "firefox"],
    "standard": ["chrome", "firefox", "edge"],
    "comprehensive": ["chrome", "firefox", "edge", "safari"]
}

# Test scenario definitions
BASIC_TEST_SCENARIOS = [
    {"name": "base_functionality", "fault_tolerance": False, "simulate_failure": False, "performance_test": False},
    {"name": "basic_fault_tolerance", "fault_tolerance": True, "simulate_failure": False, "performance_test": False}
]

FAULT_TOLERANCE_TEST_SCENARIOS = [
    {"name": "connection_failure", "fault_tolerance": True, "simulate_failure": True, "failure_type": "connection_lost"},
    {"name": "browser_crash", "fault_tolerance": True, "simulate_failure": True, "failure_type": "browser_crash"},
    {"name": "memory_pressure", "fault_tolerance": True, "simulate_failure": True, "failure_type": "memory_pressure"},
    {"name": "operation_timeout", "fault_tolerance": True, "simulate_failure": True, "failure_type": "timeout"},
    {"name": "cascade_failures", "fault_tolerance": True, "simulate_failure": True, "cascade_failures": True}
]

PERFORMANCE_TEST_SCENARIOS = [
    {"name": "performance_baseline", "fault_tolerance": False, "performance_test": True, "iterations": 10},
    {"name": "performance_with_fault_tolerance", "fault_tolerance": True, "performance_test": True, "iterations": 10},
    {"name": "stress_test", "fault_tolerance": True, "stress_test": True, "concurrent_requests": 20, "stress_duration": 120}
]

COMPREHENSIVE_TEST_SCENARIOS = BASIC_TEST_SCENARIOS + FAULT_TOLERANCE_TEST_SCENARIOS + PERFORMANCE_TEST_SCENARIOS

class TestSuiteRunner:
    """Class to manage the cross-browser model sharding test suite"""
    
    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.db_conn = None
        self.test_results = []
        self.failed_tests = []
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure logging to file
        self.log_file = os.path.join(self.output_dir, f"test_suite_{self.timestamp}.log")
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        # Initialize database connection if available
        if DUCKDB_AVAILABLE and not args.no_db:
            self._init_database()
            
    def _init_database(self):
        """Initialize database connection and tables"""
        try:
            db_path = self.args.db_path or DEFAULT_DB_PATH
            self.db_conn = duckdb.connect(db_path)
            
            # Create test_runs table if it doesn't exist
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS cross_browser_model_sharding_test_runs (
                    run_id VARCHAR,
                    timestamp TIMESTAMP,
                    model_name VARCHAR,
                    model_type VARCHAR,
                    shard_count INT,
                    shard_type VARCHAR,
                    browsers VARCHAR,
                    fault_tolerance_level VARCHAR,
                    recovery_strategy VARCHAR,
                    scenario_name VARCHAR,
                    status VARCHAR,
                    duration_seconds DOUBLE,
                    error_message VARCHAR
                )
            """)
            
            # Create test_metrics table if it doesn't exist
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS cross_browser_model_sharding_test_metrics (
                    run_id VARCHAR,
                    model_name VARCHAR,
                    initialization_time_ms DOUBLE,
                    avg_inference_time_ms DOUBLE,
                    memory_usage_mb DOUBLE,
                    browser_allocation JSON,
                    recovery_rate DOUBLE,
                    recovery_time_ms DOUBLE,
                    throughput_tokens_per_second DOUBLE,
                    success_rate DOUBLE,
                    raw_metrics JSON
                )
            """)
            
            logger.info(f"Database connection initialized with path: {db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            self.db_conn = None
    
    def _store_test_result(self, test_result):
        """Store test result in database"""
        if not self.db_conn:
            return
            
        try:
            run_id = test_result.get("run_id", f"run_{int(time.time())}")
            timestamp = test_result.get("timestamp", datetime.datetime.now())
            model_name = test_result.get("model_name", "")
            model_type = test_result.get("model_type", "")
            shard_count = test_result.get("shard_count", 0)
            shard_type = test_result.get("shard_type", "")
            browsers = ','.join(test_result.get("browsers", []))
            fault_tolerance_level = test_result.get("fault_tolerance_level", "")
            recovery_strategy = test_result.get("recovery_strategy", "")
            scenario_name = test_result.get("scenario_name", "")
            status = test_result.get("status", "unknown")
            duration_seconds = test_result.get("duration_seconds", 0.0)
            error_message = test_result.get("error_message", "")
            
            # Insert into test_runs table
            self.db_conn.execute("""
                INSERT INTO cross_browser_model_sharding_test_runs
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, timestamp, model_name, model_type, shard_count, shard_type,
                browsers, fault_tolerance_level, recovery_strategy, scenario_name,
                status, duration_seconds, error_message
            ))
            
            # If metrics available, insert into test_metrics table
            if "metrics" in test_result:
                metrics = test_result["metrics"]
                initialization_time_ms = metrics.get("initialization_time_ms", 0.0)
                avg_inference_time_ms = metrics.get("avg_inference_time_ms", 0.0)
                memory_usage_mb = metrics.get("memory_usage_mb", 0.0)
                browser_allocation = json.dumps(metrics.get("browser_allocation", {}))
                recovery_rate = metrics.get("recovery_rate", 0.0)
                recovery_time_ms = metrics.get("recovery_time_ms", 0.0)
                throughput = metrics.get("throughput_tokens_per_second", 0.0)
                success_rate = metrics.get("success_rate", 0.0)
                raw_metrics = json.dumps(metrics)
                
                self.db_conn.execute("""
                    INSERT INTO cross_browser_model_sharding_test_metrics
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, model_name, initialization_time_ms, avg_inference_time_ms,
                    memory_usage_mb, browser_allocation, recovery_rate, recovery_time_ms,
                    throughput, success_rate, raw_metrics
                ))
            
            logger.debug(f"Stored test result in database: {run_id}")
        except Exception as e:
            logger.error(f"Error storing test result in database: {e}")
    
    async def run_test(self, model_name, model_type, shard_type, browsers, scenario, fault_tolerance_level, recovery_strategy):
        """Run a single test with the specified parameters"""
        run_id = f"run_{model_name}_{shard_type}_{fault_tolerance_level}_{scenario['name']}_{int(time.time())}"
        
        # Create command line arguments for the test
        cmd_args = [
            "--model", model_name,
            "--model-type", model_type,
            "--shards", str(min(len(browsers) + 1, 4)),  # Set sensible shard count based on browsers
            "--type", shard_type
        ]
        
        # Add scenario-specific arguments
        if scenario.get("fault_tolerance", False):
            cmd_args.extend(["--fault-tolerance", "--fault-tolerance-level", fault_tolerance_level, "--recovery-strategy", recovery_strategy])
        
        if scenario.get("simulate_failure", False):
            cmd_args.append("--simulate-failure")
            if "failure_type" in scenario:
                cmd_args.extend(["--failure-type", scenario["failure_type"]])
            if scenario.get("cascade_failures", False):
                cmd_args.append("--cascade-failures")
        
        if scenario.get("performance_test", False):
            cmd_args.append("--performance-test")
            cmd_args.extend(["--iterations", str(scenario.get("iterations", 10))])
        
        if scenario.get("stress_test", False):
            cmd_args.append("--stress-test")
            cmd_args.extend(["--concurrent-requests", str(scenario.get("concurrent_requests", 20))])
            cmd_args.extend(["--stress-duration", str(scenario.get("stress_duration", 60))])
        
        # Add resource pool integration if enabled
        if self.args.resource_pool_integration:
            cmd_args.append("--resource-pool-integration")
        
        # Add performance history if enabled
        if self.args.use_performance_history:
            cmd_args.append("--use-performance-history")
        
        # Add database path if available
        if self.args.db_path and not self.args.no_db:
            cmd_args.extend(["--db-path", self.args.db_path])
        
        # Add output file
        output_file = os.path.join(self.output_dir, f"{run_id}_result.json")
        cmd_args.extend(["--output", output_file])
        
        # Add verbose flag if enabled
        if self.args.verbose:
            cmd_args.append("--verbose")
        
        # Log test start
        logger.info(f"Starting test: {run_id}")
        logger.info(f"Parameters: model={model_name}, type={model_type}, shard_type={shard_type}, browsers={browsers}, scenario={scenario['name']}, ft_level={fault_tolerance_level}, recovery={recovery_strategy}")
        
        # Prepare test result object
        test_result = {
            "run_id": run_id,
            "timestamp": datetime.datetime.now(),
            "model_name": model_name,
            "model_type": model_type,
            "shard_count": min(len(browsers) + 1, 4),
            "shard_type": shard_type,
            "browsers": browsers,
            "fault_tolerance_level": fault_tolerance_level,
            "recovery_strategy": recovery_strategy,
            "scenario_name": scenario["name"],
            "command_args": cmd_args,
            "status": "running",
            "start_time": time.time()
        }
        
        try:
            # Run the test with timeout protection
            process = await asyncio.create_subprocess_exec(
                sys.executable, "test_fault_tolerant_cross_browser_model_sharding.py", 
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await # TODO: Replace with anyio.fail_after - asyncio.wait_for(process.communicate(), timeout=TEST_TIMEOUT)
                stdout = stdout.decode() if stdout else ""
                stderr = stderr.decode() if stderr else ""
                
                # Check if output file was created
                if os.path.exists(output_file):
                    with open(output_file, 'r') as f:
                        result_data = json.load(f)
                        
                    # Extract relevant metrics
                    if result_data["status"] == "completed":
                        test_result["status"] = "success"
                        
                        # Extract basic metrics
                        metrics = {}
                        
                        # Extract initialization metrics
                        if "initialization_metrics" in result_data:
                            initialization_metrics = result_data["initialization_metrics"]
                            metrics["initialization_time_ms"] = initialization_metrics.get("initialization_time_ms", 0)
                        
                        # Extract final metrics
                        if "final_metrics" in result_data:
                            final_metrics = result_data["final_metrics"]
                            
                            if "browser_allocation" in final_metrics:
                                metrics["browser_allocation"] = final_metrics["browser_allocation"]
                            
                            if "avg_inference_time_ms" in final_metrics:
                                metrics["avg_inference_time_ms"] = final_metrics["avg_inference_time_ms"]
                                
                            if "memory_usage_mb" in final_metrics:
                                metrics["memory_usage_mb"] = final_metrics["memory_usage_mb"]
                                
                            if "throughput_tokens_per_second" in final_metrics:
                                metrics["throughput_tokens_per_second"] = final_metrics["throughput_tokens_per_second"]
                        
                        # Extract recovery metrics
                        if "recovery_metrics" in result_data:
                            recovery_metrics = result_data["recovery_metrics"]
                            
                            if "recovery_success_rate" in recovery_metrics:
                                metrics["recovery_rate"] = recovery_metrics["recovery_success_rate"]
                                
                            if "recovery_time_ms" in recovery_metrics:
                                metrics["recovery_time_ms"] = recovery_metrics["recovery_time_ms"]
                        
                        # Extract performance statistics
                        if "phases" in result_data and "performance_testing" in result_data["phases"]:
                            performance_phase = result_data["phases"]["performance_testing"]
                            
                            if "statistics" in performance_phase:
                                statistics = performance_phase["statistics"]
                                
                                if "avg_execution_time" in statistics:
                                    metrics["avg_execution_time"] = statistics["avg_execution_time"]
                                    
                                if "successful_iterations" in statistics and "total_iterations" in statistics:
                                    metrics["success_rate"] = statistics["successful_iterations"] / statistics["total_iterations"]
                                    
                        # Save raw metrics for reference
                        metrics["raw_result_data"] = result_data
                        
                        test_result["metrics"] = metrics
                    else:
                        test_result["status"] = "failed"
                        test_result["error_message"] = result_data.get("error", "Unknown error")
                        self.failed_tests.append(test_result)
                else:
                    test_result["status"] = "failed"
                    test_result["error_message"] = "No output file created"
                    test_result["stdout"] = stdout
                    test_result["stderr"] = stderr
                    self.failed_tests.append(test_result)
            except asyncio.TimeoutError:
                # Handle timeout
                test_result["status"] = "timeout"
                test_result["error_message"] = f"Test exceeded timeout ({TEST_TIMEOUT}s)"
                self.failed_tests.append(test_result)
                
                # Try to terminate the process
                process.terminate()
                try:
                    await # TODO: Replace with anyio.fail_after - asyncio.wait_for(process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    process.kill()
        except Exception as e:
            # Handle unexpected errors
            test_result["status"] = "error"
            test_result["error_message"] = str(e)
            test_result["traceback"] = traceback.format_exc()
            self.failed_tests.append(test_result)
        finally:
            # Calculate duration
            test_result["end_time"] = time.time()
            test_result["duration_seconds"] = test_result["end_time"] - test_result["start_time"]
            
            # Log test completion
            logger.info(f"Completed test: {run_id}, status: {test_result['status']}, duration: {test_result['duration_seconds']:.2f}s")
            
            # Store test result
            self.test_results.append(test_result)
            self._store_test_result(test_result)
            
            return test_result
    
    async def run_test_suite(self):
        """Run the complete test suite based on command-line arguments"""
        if not TEST_SCRIPT_AVAILABLE:
            logger.error("Test script not available, cannot run test suite")
            return 1
        
        # Determine models to test
        models_to_test = []
        if self.args.all_models:
            # Add all models from MODEL_MAPPING
            for model_type, model_list in MODEL_MAPPING.items():
                for model in model_list:
                    models_to_test.append((model, model_type))
        elif self.args.model_families:
            # Add models from specified families
            for family in self.args.model_families:
                if family in MODEL_MAPPING:
                    for model in MODEL_MAPPING[family]:
                        models_to_test.append((model, family))
                else:
                    logger.warning(f"Unknown model family: {family}")
        elif self.args.models:
            # Add specified models with automatic type detection
            for model in self.args.models:
                model_type = self._detect_model_type(model)
                models_to_test.append((model, model_type))
        else:
            # Default to bert-base-uncased
            models_to_test.append(("bert-base-uncased", "text"))
            
        # Determine sharding strategies to test
        strategies_to_test = []
        if self.args.all_sharding_strategies:
            strategies_to_test = SHARDING_STRATEGIES
        elif self.args.sharding_strategies:
            strategies_to_test = self.args.sharding_strategies
        else:
            # Default to layer-based
            strategies_to_test = ["layer"]
            
        # Determine fault tolerance levels to test
        ft_levels_to_test = []
        if self.args.all_fault_tolerance_levels:
            ft_levels_to_test = FAULT_TOLERANCE_LEVELS
        elif self.args.fault_tolerance_levels:
            ft_levels_to_test = self.args.fault_tolerance_levels
        else:
            # Default to medium
            ft_levels_to_test = ["medium"]
            
        # Determine recovery strategies to test
        strategies = []
        if self.args.all_recovery_strategies:
            strategies = RECOVERY_STRATEGIES
        elif self.args.recovery_strategies:
            strategies = self.args.recovery_strategies
        else:
            # Default to progressive
            strategies = ["progressive"]
            
        # Determine browser combinations to test
        browser_sets_to_test = []
        if self.args.browser_combination:
            if self.args.browser_combination in BROWSER_COMBINATIONS:
                browser_sets_to_test = [BROWSER_COMBINATIONS[self.args.browser_combination]]
            else:
                logger.warning(f"Unknown browser combination: {self.args.browser_combination}")
                browser_sets_to_test = [BROWSER_COMBINATIONS["standard"]]
        elif self.args.browsers:
            browser_sets_to_test = [self.args.browsers]
        else:
            # Default to standard
            browser_sets_to_test = [BROWSER_COMBINATIONS["standard"]]
            
        # Determine test scenarios to run
        scenarios_to_test = []
        if self.args.comprehensive:
            scenarios_to_test = COMPREHENSIVE_TEST_SCENARIOS
        elif self.args.fault_tolerance_only:
            scenarios_to_test = FAULT_TOLERANCE_TEST_SCENARIOS
        elif self.args.performance_only:
            scenarios_to_test = PERFORMANCE_TEST_SCENARIOS
        else:
            # Default to basic scenarios
            scenarios_to_test = BASIC_TEST_SCENARIOS
            
        # Calculate total number of tests
        total_tests = len(models_to_test) * len(strategies_to_test) * len(ft_levels_to_test) * len(strategies) * len(browser_sets_to_test) * len(scenarios_to_test)
        logger.info(f"Preparing to run {total_tests} tests:")
        logger.info(f"  Models: {[m[0] for m in models_to_test]}")
        logger.info(f"  Sharding Strategies: {strategies_to_test}")
        logger.info(f"  Fault Tolerance Levels: {ft_levels_to_test}")
        logger.info(f"  Recovery Strategies: {strategies}")
        logger.info(f"  Browser Sets: {browser_sets_to_test}")
        logger.info(f"  Scenarios: {[s['name'] for s in scenarios_to_test]}")
        
        # Create test matrix
        test_matrix = []
        for model_name, model_type in models_to_test:
            for shard_type in strategies_to_test:
                for ft_level in ft_levels_to_test:
                    for recovery_strategy in strategies:
                        for browsers in browser_sets_to_test:
                            for scenario in scenarios_to_test:
                                test_matrix.append((model_name, model_type, shard_type, browsers, scenario, ft_level, recovery_strategy))
        
        # Run tests based on concurrency setting
        completed_tests = 0
        start_time = time.time()
        
        if self.args.concurrent_tests > 1:
            # Run tests concurrently
            semaphore = asyncio.Semaphore(self.args.concurrent_tests)
            
            async def run_with_semaphore(test_params):
                async with semaphore:
                    return await self.run_test(*test_params)
            
            tasks = [run_with_semaphore(params) for params in test_matrix]
            results = await # TODO: Replace with task group - asyncio.gather(*tasks)
            
            for result in results:
                completed_tests += 1
                logger.info(f"Progress: {completed_tests}/{total_tests} tests completed ({completed_tests/total_tests*100:.1f}%)")
        else:
            # Run tests sequentially
            for test_params in test_matrix:
                await self.run_test(*test_params)
                completed_tests += 1
                logger.info(f"Progress: {completed_tests}/{total_tests} tests completed ({completed_tests/total_tests*100:.1f}%)")
        
        # Calculate total duration
        total_duration = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(total_tests, total_duration)
        
        # Save summary to file
        summary_file = os.path.join(self.output_dir, f"test_suite_summary_{self.timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Test suite complete. Summary saved to {summary_file}")
        
        # Return success if no critical tests failed
        return 0 if len(self.failed_tests) == 0 else 1
    
    def _detect_model_type(self, model_name):
        """Detect model type based on model name"""
        model_name_lower = model_name.lower()
        
        if any(keyword in model_name_lower for keyword in ["bert", "roberta", "distil", "gpt", "llama", "t5"]):
            return "text"
        elif any(keyword in model_name_lower for keyword in ["vit", "resnet", "deit", "convnext"]):
            return "vision"
        elif any(keyword in model_name_lower for keyword in ["whisper", "wav2vec", "hubert", "clap"]):
            return "audio"
        elif any(keyword in model_name_lower for keyword in ["clip", "flava", "blip", "llava"]):
            return "multimodal"
        else:
            # Default to text if can't determine
            return "text"
    
    def _generate_summary(self, total_tests, total_duration):
        """Generate test suite summary"""
        successful_tests = [t for t in self.test_results if t["status"] == "success"]
        failed_tests = [t for t in self.test_results if t["status"] in ["failed", "error"]]
        timeout_tests = [t for t in self.test_results if t["status"] == "timeout"]
        
        # Group results by various dimensions
        by_model = {}
        by_strategy = {}
        by_ft_level = {}
        by_scenario = {}
        
        for test in self.test_results:
            # By model
            model = test["model_name"]
            by_model.setdefault(model, {"total": 0, "success": 0, "failed": 0, "timeout": 0})
            by_model[model]["total"] += 1
            if test["status"] == "success":
                by_model[model]["success"] += 1
            elif test["status"] == "timeout":
                by_model[model]["timeout"] += 1
            else:
                by_model[model]["failed"] += 1
                
            # By strategy
            strategy = test["shard_type"]
            by_strategy.setdefault(strategy, {"total": 0, "success": 0, "failed": 0, "timeout": 0})
            by_strategy[strategy]["total"] += 1
            if test["status"] == "success":
                by_strategy[strategy]["success"] += 1
            elif test["status"] == "timeout":
                by_strategy[strategy]["timeout"] += 1
            else:
                by_strategy[strategy]["failed"] += 1
                
            # By fault tolerance level
            ft_level = test["fault_tolerance_level"]
            by_ft_level.setdefault(ft_level, {"total": 0, "success": 0, "failed": 0, "timeout": 0})
            by_ft_level[ft_level]["total"] += 1
            if test["status"] == "success":
                by_ft_level[ft_level]["success"] += 1
            elif test["status"] == "timeout":
                by_ft_level[ft_level]["timeout"] += 1
            else:
                by_ft_level[ft_level]["failed"] += 1
                
            # By scenario
            scenario = test["scenario_name"]
            by_scenario.setdefault(scenario, {"total": 0, "success": 0, "failed": 0, "timeout": 0})
            by_scenario[scenario]["total"] += 1
            if test["status"] == "success":
                by_scenario[scenario]["success"] += 1
            elif test["status"] == "timeout":
                by_scenario[scenario]["timeout"] += 1
            else:
                by_scenario[scenario]["failed"] += 1
        
        # Calculate performance metrics if available
        performance_metrics = {}
        for test in successful_tests:
            if "metrics" in test:
                metrics = test["metrics"]
                model = test["model_name"]
                performance_metrics.setdefault(model, {
                    "avg_inference_time_ms": [],
                    "memory_usage_mb": [],
                    "throughput": [],
                    "recovery_rate": [],
                    "recovery_time_ms": []
                })
                
                if "avg_inference_time_ms" in metrics:
                    performance_metrics[model]["avg_inference_time_ms"].append(metrics["avg_inference_time_ms"])
                if "memory_usage_mb" in metrics:
                    performance_metrics[model]["memory_usage_mb"].append(metrics["memory_usage_mb"])
                if "throughput_tokens_per_second" in metrics:
                    performance_metrics[model]["throughput"].append(metrics["throughput_tokens_per_second"])
                if "recovery_rate" in metrics:
                    performance_metrics[model]["recovery_rate"].append(metrics["recovery_rate"])
                if "recovery_time_ms" in metrics:
                    performance_metrics[model]["recovery_time_ms"].append(metrics["recovery_time_ms"])
        
        # Calculate average metrics
        avg_performance_metrics = {}
        for model, metrics in performance_metrics.items():
            avg_performance_metrics[model] = {}
            for metric_name, values in metrics.items():
                if values:
                    avg_performance_metrics[model][metric_name] = sum(values) / len(values)
        
        # Create summary
        summary = {
            "timestamp": self.timestamp,
            "total_tests": total_tests,
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "timeout_tests": len(timeout_tests),
            "success_rate": len(successful_tests) / total_tests if total_tests > 0 else 0,
            "total_duration_seconds": total_duration,
            "average_test_duration": sum(t["duration_seconds"] for t in self.test_results) / len(self.test_results) if self.test_results else 0,
            "by_model": by_model,
            "by_strategy": by_strategy,
            "by_fault_tolerance_level": by_ft_level,
            "by_scenario": by_scenario,
            "performance_metrics": avg_performance_metrics,
            "failed_test_details": [
                {
                    "run_id": t["run_id"],
                    "model_name": t["model_name"],
                    "shard_type": t["shard_type"],
                    "scenario_name": t["scenario_name"],
                    "fault_tolerance_level": t["fault_tolerance_level"],
                    "status": t["status"],
                    "error_message": t.get("error_message", "")
                } for t in failed_tests
            ]
        }
        
        return summary

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Comprehensive Cross-Browser Model Sharding Test Suite Runner")
    
    # Test scope options
    parser.add_argument("--all-models", action="store_true",
                       help="Test all supported models")
    parser.add_argument("--model-families", nargs="+", choices=list(MODEL_MAPPING.keys()),
                       help="Specific model families to test")
    parser.add_argument("--models", nargs="+",
                       help="Specific models to test (comma-separated)")
    
    # Sharding strategy options
    parser.add_argument("--all-sharding-strategies", action="store_true",
                       help="Test all sharding strategies")
    parser.add_argument("--sharding-strategies", nargs="+", choices=SHARDING_STRATEGIES,
                       help="Specific sharding strategies to test (comma-separated)")
    
    # Fault tolerance options
    parser.add_argument("--all-fault-tolerance-levels", action="store_true",
                       help="Test all fault tolerance levels")
    parser.add_argument("--fault-tolerance-levels", nargs="+", choices=FAULT_TOLERANCE_LEVELS,
                       help="Specific fault tolerance levels to test (comma-separated)")
    
    # Recovery strategy options
    parser.add_argument("--all-recovery-strategies", action="store_true",
                       help="Test all recovery strategies")
    parser.add_argument("--recovery-strategies", nargs="+", choices=RECOVERY_STRATEGIES,
                       help="Specific recovery strategies to test (comma-separated)")
    
    # Browser options
    parser.add_argument("--browser-combination", choices=list(BROWSER_COMBINATIONS.keys()),
                       help="Predefined browser combination to use")
    parser.add_argument("--browsers", nargs="+",
                       help="Specific browsers to test (comma-separated)")
    
    # Test scenario options
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive test suite with all scenarios")
    parser.add_argument("--fault-tolerance-only", action="store_true",
                       help="Run only fault tolerance test scenarios")
    parser.add_argument("--performance-only", action="store_true",
                       help="Run only performance test scenarios")
    
    # Integration options
    parser.add_argument("--resource-pool-integration", action="store_true",
                       help="Enable resource pool integration")
    parser.add_argument("--use-performance-history", action="store_true",
                       help="Enable browser performance history tracking")
    
    # Concurrency options
    parser.add_argument("--concurrent-tests", type=int, default=1,
                       help="Number of tests to run concurrently")
    
    # Output options
    parser.add_argument("--output-dir", type=str,
                       help=f"Directory to store test results (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--db-path", type=str,
                       help=f"Path to DuckDB database (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--no-db", action="store_true",
                       help="Disable database integration")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Parse models if comma-separated
    if args.models and len(args.models) == 1 and ',' in args.models[0]:
        args.models = args.models[0].split(',')
        
    # Parse strategies if comma-separated
    if args.sharding_strategies and len(args.sharding_strategies) == 1 and ',' in args.sharding_strategies[0]:
        args.sharding_strategies = args.sharding_strategies[0].split(',')
        
    # Parse fault tolerance levels if comma-separated
    if args.fault_tolerance_levels and len(args.fault_tolerance_levels) == 1 and ',' in args.fault_tolerance_levels[0]:
        args.fault_tolerance_levels = args.fault_tolerance_levels[0].split(',')
        
    # Parse recovery strategies if comma-separated
    if args.recovery_strategies and len(args.recovery_strategies) == 1 and ',' in args.recovery_strategies[0]:
        args.recovery_strategies = args.recovery_strategies[0].split(',')
        
    # Parse browsers if comma-separated
    if args.browsers and len(args.browsers) == 1 and ',' in args.browsers[0]:
        args.browsers = args.browsers[0].split(',')
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create test suite runner
    runner = TestSuiteRunner(args)
    
    # Run test suite
    try:
        exit_code = await runner.run_test_suite()
        return exit_code
    except KeyboardInterrupt:
        logger.info("Test suite interrupted by user")
        return 130

if __name__ == "__main__":
    try:
        exit_code = anyio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        traceback.print_exc()
        sys.exit(1)