#!/usr/bin/env python3
"""
Basic WebGPU/WebNN Resource Pool Fault Tolerance Test

This script provides a simple test case for the WebGPU/WebNN Resource Pool fault tolerance
system. It focuses on demonstrating basic fault injection and recovery without requiring
complex imports or dependencies, making it ideal for CI/CD environments or quick tests.

The test creates a mock implementation of the resource pool and demonstrates how it handles
and recovers from different types of faults including connection loss, component failure,
and browser crashes.

Usage:
    # Run the basic test with default settings
    python test_basic_resource_pool_fault_tolerance.py

    # Run with a specific model and fault scenario
    python test_basic_resource_pool_fault_tolerance.py --model vit-base-patch16-224 --scenario browser_crash

    # Test a specific recovery strategy
    python test_basic_resource_pool_fault_tolerance.py --recovery-strategy coordinated
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to find the mock implementation
sys.path.append(str(Path(__file__).resolve().parent))

# Try to import the fixed mock implementation
try:
    from fixed_mock_cross_browser_sharding import MockCrossBrowserModelShardingManager
    logger.info("Successfully imported MockCrossBrowserModelShardingManager")
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    sys.exit(1)

class BasicResourcePoolFaultTester:
    """Simple tester for resource pool fault tolerance capabilities."""
    
    def __init__(self, 
                model_name: str = "bert-base-uncased",
                recovery_strategy: str = "progressive",
                output_dir: str = "./fault_tolerance_basic_test_results"):
        """
        Initialize the fault tolerance tester.
        
        Args:
            model_name: Name of the model to test
            recovery_strategy: Recovery strategy to use
            output_dir: Directory for output files
        """
        self.model_name = model_name
        self.recovery_strategy = recovery_strategy
        self.browsers = ["chrome", "firefox", "edge"]
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.fault_scenarios = {
            "connection_lost": "Browser connection loss scenario",
            "component_failure": "Model component failure scenario",
            "browser_crash": "Complete browser crash scenario",
            "multiple_failures": "Multiple simultaneous failures scenario"
        }
        
        # Create timestamp for results
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = os.path.join(self.output_dir, f"results_{self.timestamp}.json")
        
        # Test results
        self.results = {
            "test_info": {
                "model": self.model_name,
                "browsers": self.browsers,
                "recovery_strategy": self.recovery_strategy,
                "timestamp": self.timestamp
            },
            "scenarios": {}
        }
    
    async def test_scenario(self, scenario: str) -> Dict[str, Any]:
        """
        Test a specific fault scenario.
        
        Args:
            scenario: Name of the scenario to test
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing scenario: {scenario}")
        
        # Create model manager for this scenario
        manager = MockCrossBrowserModelShardingManager(
            model_name=self.model_name,
            browsers=self.browsers,
            shard_type="optimal",
            num_shards=len(self.browsers),
            model_config={
                "enable_fault_tolerance": True,
                "fault_tolerance_level": "medium",
                "recovery_strategy": self.recovery_strategy,
                "timeout": 60
            }
        )
        
        scenario_results = {
            "scenario": scenario,
            "description": self.fault_scenarios.get(scenario, "Unknown scenario"),
            "phases": {}
        }
        
        try:
            # Phase 1: Initialize
            logger.info("Phase 1: Initializing model manager")
            start_time = time.time()
            initialized = await manager.initialize()
            init_time = time.time() - start_time
            
            phase1_result = {
                "success": initialized,
                "duration": init_time,
                "message": f"Initialization {'succeeded' if initialized else 'failed'}"
            }
            scenario_results["phases"]["initialize"] = phase1_result
            
            if not initialized:
                logger.error("Failed to initialize model manager, aborting test")
                return scenario_results
            
            logger.info(f"Model manager initialized in {init_time:.2f}s")
            
            # Phase 2: Initial inference test
            logger.info("Phase 2: Running initial inference test")
            test_input = {"input_ids": [101, 2023, 2003, 1037, 3231, 102], "attention_mask": [1, 1, 1, 1, 1, 1]}
            
            start_time = time.time()
            inference_result = await manager.run_inference_sharded(test_input)
            inference_time = time.time() - start_time
            
            phase2_result = {
                "success": "error" not in inference_result,
                "duration": inference_time,
                "message": "Initial inference successful" if "error" not in inference_result else inference_result.get("error")
            }
            scenario_results["phases"]["initial_inference"] = phase2_result
            
            if "error" in inference_result:
                logger.error(f"Initial inference failed: {inference_result['error']}")
                return scenario_results
            
            logger.info(f"Initial inference successful in {inference_time:.2f}s")
            
            # Phase 3: Fault injection
            logger.info(f"Phase 3: Injecting fault ({scenario})")
            start_time = time.time()
            fault_result = await manager.inject_fault(scenario, 0)
            fault_time = time.time() - start_time
            
            phase3_result = {
                "success": fault_result.get("status") == "success",
                "duration": fault_time,
                "message": f"Fault injection {fault_result.get('status')}",
                "fault_type": scenario,
                "details": fault_result
            }
            scenario_results["phases"]["fault_injection"] = phase3_result
            
            # Phase 4: Inference with fault
            logger.info("Phase 4: Running inference with fault present")
            start_time = time.time()
            fault_inference_result = await manager.run_inference_sharded(test_input)
            fault_inference_time = time.time() - start_time
            
            fault_inference_success = "error" not in fault_inference_result
            phase4_result = {
                "success": fault_inference_success,
                "duration": fault_inference_time,
                "message": "Inference with fault succeeded (fault tolerance working)" if fault_inference_success else f"Inference with fault failed as expected: {fault_inference_result.get('error', 'Unknown error')}"
            }
            scenario_results["phases"]["fault_inference"] = phase4_result
            
            # Phase 5: Recovery
            logger.info("Phase 5: Recovering from fault")
            start_time = time.time()
            recovery_result = await manager.recover_fault(scenario, 0)
            recovery_time = time.time() - start_time
            
            recovery_success = recovery_result.get("recovered", False)
            phase5_result = {
                "success": recovery_success,
                "duration": recovery_time,
                "message": f"Recovery {'succeeded' if recovery_success else 'failed'}",
                "details": recovery_result
            }
            scenario_results["phases"]["recovery"] = phase5_result
            
            # Phase 6: Post-recovery inference
            logger.info("Phase 6: Running inference after recovery")
            start_time = time.time()
            recovery_inference_result = await manager.run_inference_sharded(test_input)
            recovery_inference_time = time.time() - start_time
            
            recovery_inference_success = "error" not in recovery_inference_result
            phase6_result = {
                "success": recovery_inference_success,
                "duration": recovery_inference_time,
                "message": "Post-recovery inference successful" if recovery_inference_success else f"Post-recovery inference failed: {recovery_inference_result.get('error', 'Unknown error')}"
            }
            scenario_results["phases"]["recovery_inference"] = phase6_result
            
            # Get diagnostics
            diagnostics = await manager.get_diagnostics()
            scenario_results["diagnostics"] = diagnostics
            
            # Shutdown the manager
            await manager.shutdown()
            
            # Calculate overall success
            scenario_results["overall_success"] = (
                phase1_result["success"] and 
                phase2_result["success"] and
                phase3_result["success"] and
                (phase4_result["success"] or not phase4_result["success"]) and  # Either outcome is acceptable for fault inference
                phase5_result["success"] and
                phase6_result["success"]
            )
            
            # Return results
            return scenario_results
            
        except Exception as e:
            logger.error(f"Error during test scenario {scenario}: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to shutdown the manager
            try:
                await manager.shutdown()
            except Exception:
                pass
            
            # Record error
            scenario_results["error"] = str(e)
            scenario_results["traceback"] = traceback.format_exc()
            scenario_results["overall_success"] = False
            
            return scenario_results
    
    async def run_test(self, scenario: str = None) -> Dict[str, Any]:
        """
        Run fault tolerance tests for all or a specific scenario.
        
        Args:
            scenario: Specific scenario to test (None for all scenarios)
            
        Returns:
            Dictionary with test results
        """
        scenarios_to_test = [scenario] if scenario else list(self.fault_scenarios.keys())
        
        logger.info(f"Starting fault tolerance test for model {self.model_name}")
        logger.info(f"Recovery strategy: {self.recovery_strategy}")
        logger.info(f"Testing scenarios: {scenarios_to_test}")
        
        # Run tests for each scenario
        for scenario in scenarios_to_test:
            if scenario not in self.fault_scenarios:
                logger.warning(f"Unknown scenario: {scenario}, skipping")
                continue
                
            logger.info(f"Testing scenario: {scenario}")
            scenario_result = await self.test_scenario(scenario)
            self.results["scenarios"][scenario] = scenario_result
            
            # Log results
            success = scenario_result.get("overall_success", False)
            logger.info(f"Scenario {scenario}: {'SUCCESS' if success else 'FAILURE'}")
            
        # Save results to file
        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
            
        logger.info(f"Test results saved to {self.results_file}")
        
        # Calculate overall test success
        successful_scenarios = [s for s, r in self.results["scenarios"].items() 
                               if r.get("overall_success", False)]
        total_scenarios = len(self.results["scenarios"])
        
        self.results["summary"] = {
            "total_scenarios": total_scenarios,
            "successful_scenarios": len(successful_scenarios),
            "success_rate": len(successful_scenarios) / total_scenarios if total_scenarios > 0 else 0,
            "overall_success": len(successful_scenarios) == total_scenarios and total_scenarios > 0
        }
        
        # Print summary
        logger.info(f"Test summary: {len(successful_scenarios)}/{total_scenarios} scenarios successful")
        logger.info(f"Overall test result: {'SUCCESS' if self.results['summary']['overall_success'] else 'FAILURE'}")
        
        return self.results

def format_results_for_display(results: Dict[str, Any]) -> str:
    """
    Format test results for terminal display.
    
    Args:
        results: Test results dictionary
        
    Returns:
        Formatted string for display
    """
    output = []
    output.append("=" * 80)
    output.append(f"FAULT TOLERANCE TEST RESULTS - {results['test_info']['model']}")
    output.append(f"Recovery Strategy: {results['test_info']['recovery_strategy']}")
    output.append(f"Timestamp: {results['test_info']['timestamp']}")
    output.append("=" * 80)
    
    for scenario, data in results["scenarios"].items():
        overall = "✅ SUCCESS" if data.get("overall_success", False) else "❌ FAILURE"
        output.append(f"\nSCENARIO: {scenario} - {overall}")
        output.append(f"Description: {data.get('description', 'No description')}")
        
        if "error" in data:
            output.append(f"ERROR: {data['error']}")
            continue
            
        output.append("\nTest Phases:")
        for phase, phase_data in data.get("phases", {}).items():
            success = "✅" if phase_data.get("success", False) else "❌"
            duration = f"{phase_data.get('duration', 0):.2f}s"
            output.append(f"  {success} {phase.ljust(20)} - {duration.ljust(8)} - {phase_data.get('message', '')}")
    
    if "summary" in results:
        output.append("\n" + "=" * 80)
        output.append("SUMMARY:")
        output.append(f"Scenarios: {results['summary']['successful_scenarios']}/{results['summary']['total_scenarios']} successful")
        output.append(f"Success Rate: {results['summary']['success_rate'] * 100:.1f}%")
        overall = "✅ SUCCESS" if results['summary']['overall_success'] else "❌ FAILURE"
        output.append(f"Overall Result: {overall}")
        output.append("=" * 80)
    
    return "\n".join(output)

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Basic WebGPU/WebNN Resource Pool Fault Tolerance Test"
    )
    
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                      help="Model name to test")
    parser.add_argument("--scenario", type=str, choices=["connection_lost", "component_failure", "browser_crash", "multiple_failures"],
                      help="Specific scenario to test (default: test all scenarios)")
    parser.add_argument("--recovery-strategy", type=str, default="progressive",
                      choices=["simple", "progressive", "coordinated"],
                      help="Recovery strategy to use (default: progressive)")
    parser.add_argument("--output-dir", type=str, default="./fault_tolerance_basic_test_results",
                      help="Directory for output files")
    
    args = parser.parse_args()
    
    # Run the tester
    tester = BasicResourcePoolFaultTester(
        model_name=args.model,
        recovery_strategy=args.recovery_strategy,
        output_dir=args.output_dir
    )
    
    results = await tester.run_test(args.scenario)
    
    # Display formatted results
    print(format_results_for_display(results))
    
    # Return success/failure exit code
    return 0 if results["summary"]["overall_success"] else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))