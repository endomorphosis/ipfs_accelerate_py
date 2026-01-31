#!/usr/bin/env python3
"""
Advanced Fault Tolerance Validation and Visualization Runner

This script provides a command-line interface for running the advanced fault tolerance
validation system with visualization capabilities for the WebGPU/WebNN Resource Pool
Integration project.

Usage:
    # Standard validation
    python run_advanced_fault_tolerance_visualization.py --model bert-base-uncased --browsers chrome,firefox,edge
    
    # Comparative validation
    python run_advanced_fault_tolerance_visualization.py --model llama-7b --comparative --output-dir ./reports
    
    # Stress test
    python run_advanced_fault_tolerance_visualization.py --model whisper-tiny --stress-test --iterations 10
"""

import os
import sys
import json
import time
import anyio
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

async def run_validation():
    """Run fault tolerance validation based on command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced Fault Tolerance Validation and Visualization"
    )
    
    # Basic options
    parser.add_argument("--model", type=str, required=True,
                      help="Model name to test")
    parser.add_argument("--browsers", type=str, default="chrome,firefox,edge",
                      help="Comma-separated list of browsers to use")
    parser.add_argument("--output-dir", type=str, default="./fault_tolerance_reports",
                      help="Directory for output files")
    
    # Validation options
    parser.add_argument("--fault-level", type=str, default="medium",
                      choices=["low", "medium", "high", "critical"],
                      help="Fault tolerance level to validate")
    parser.add_argument("--recovery-strategy", type=str, default="progressive",
                      choices=["simple", "progressive", "parallel", "coordinated"],
                      help="Recovery strategy to validate")
    parser.add_argument("--test-scenarios", type=str,
                      help="Comma-separated list of test scenarios")
    
    # Test modes
    parser.add_argument("--comparative", action="store_true",
                      help="Run comparative validation across multiple configurations")
    parser.add_argument("--stress-test", action="store_true",
                      help="Run stress test validation with multiple iterations")
    parser.add_argument("--iterations", type=int, default=5,
                      help="Number of iterations for stress testing")
    
    # Report options
    parser.add_argument("--report-name", type=str, 
                      help="Name of the report file (defaults to model_fault_report.html)")
    parser.add_argument("--ci-compatible", action="store_true",
                      help="Generate CI-compatible report with embedded images")
    
    # Debugging options
    parser.add_argument("--mock", action="store_true",
                      help="Use mock implementation for testing without actual browsers")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Set default report name if not specified
    if not args.report_name:
        model_name = args.model.replace('/', '_')
        if args.comparative:
            args.report_name = f"{model_name}_comparative_report.html"
        elif args.stress_test:
            args.report_name = f"{model_name}_stress_test_report.html"
        else:
            args.report_name = f"{model_name}_fault_report.html"
    
    # Prepare browsers list
    browsers = args.browsers.split(',')
    
    # Prepare test scenarios
    test_scenarios = None
    if args.test_scenarios:
        test_scenarios = args.test_scenarios.split(',')
    
    try:
        # Import necessary modules
        try:
            from test.web_platform.fault_tolerance_visualization_integration import FaultToleranceValidationSystem
            
            if args.mock:
                from mock_cross_browser_sharding import MockCrossBrowserModelShardingManager as ModelManager
            else:
                from cross_browser_model_sharding import CrossBrowserModelShardingManager as ModelManager
                
            modules_available = True
        except ImportError as e:
            logger.error(f"Required modules not available: {e}")
            modules_available = False
            
        if not modules_available:
            logger.error("Cannot proceed without required modules")
            return 1
        
        # Create output directory (use absolute path)
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean up any existing nested directories with the same name
        nested_dir = os.path.join(output_dir, os.path.basename(output_dir))
        if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
            import shutil
            try:
                shutil.rmtree(nested_dir)
                logger.info(f"Removed nested directory: {nested_dir}")
            except Exception as e:
                logger.warning(f"Could not remove nested directory: {e}")
                
        # Update args with absolute path
        args.output_dir = output_dir
        
        # Create model configuration
        model_config = {
            "enable_fault_tolerance": True,
            "fault_tolerance_level": args.fault_level,
            "recovery_strategy": args.recovery_strategy,
            "timeout": 300
        }
        
        # Create mock implementation if requested
        if args.mock:
            logger.info("Using mock implementation for testing")
            manager = ModelManager(
                model_name=args.model,
                browsers=browsers,
                shard_type="optimal",
                num_shards=len(browsers),
                model_config=model_config
            )
        else:
            # Create actual model manager
            manager = ModelManager(
                model_name=args.model,
                browsers=browsers,
                shard_type="optimal",
                num_shards=len(browsers),
                model_config=model_config
            )
        
        # Initialize model manager
        logger.info(f"Initializing model manager for {args.model}")
        start_time = time.time()
        initialized = await manager.initialize()
        init_time = time.time() - start_time
        
        if not initialized:
            logger.error(f"Failed to initialize model manager for {args.model}")
            return 1
        
        logger.info(f"Model manager initialized in {init_time:.2f}s")
        
        # Create validation system
        validation_system = FaultToleranceValidationSystem(
            model_manager=manager,
            output_dir=args.output_dir
        )
        
        # Run validation based on mode
        if args.comparative:
            logger.info("Running comparative validation")
            results = await validation_system.run_comparative_validation(
                strategies=["simple", "progressive", "coordinated"],
                levels=[args.fault_level],
                test_scenarios=test_scenarios,
                report_prefix=args.model.replace('-', '_')
            )
            
            summary_path = os.path.join(args.output_dir, f"{args.model.replace('-', '_')}_comparative_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Comparative validation completed. Summary saved to: {summary_path}")
            
        elif args.stress_test:
            logger.info(f"Running stress test validation with {args.iterations} iterations")
            results = await validation_system.run_stress_test_validation(
                iterations=args.iterations,
                fault_tolerance_level=args.fault_level,
                recovery_strategy=args.recovery_strategy,
                test_scenarios=test_scenarios,
                report_name=args.report_name
            )
            
            summary_path = os.path.join(args.output_dir, f"{args.model.replace('-', '_')}_stress_test_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Stress test validation completed. Summary saved to: {summary_path}")
            
        else:
            logger.info("Running standard validation")
            results = await validation_system.run_validation_with_visualization(
                fault_tolerance_level=args.fault_level,
                recovery_strategy=args.recovery_strategy,
                test_scenarios=test_scenarios,
                generate_report=True,
                report_name=args.report_name,
                ci_compatible=args.ci_compatible
            )
            
            validation_status = results.get("validation_results", {}).get("validation_status", "unknown")
            logger.info(f"Validation completed with status: {validation_status}")
            
            if "report" in results.get("visualizations", {}):
                report_path = results["visualizations"]["report"]
                logger.info(f"Report generated at: {report_path}")
        
        # Shutdown model manager
        await manager.shutdown()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(anyio.run(run_validation()))