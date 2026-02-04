#!/usr/bin/env python3
"""
Command-line interface for the refactored benchmark suite.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import List, Optional

from test.tools.skills.refactored_benchmark_suite import ModelBenchmark, BenchmarkSuite
from test.tools.skills.refactored_benchmark_suite.utils.logging import setup_logger

# Configure logger
logger = setup_logger("benchmark_cli")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="HuggingFace Model Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument(
        "--model", type=str, nargs="+",
        help="HuggingFace model ID(s) to benchmark"
    )
    model_group.add_argument(
        "--task", type=str,
        help="Model task (e.g., text-generation, image-classification)"
    )
    model_group.add_argument(
        "--suite", type=str, choices=["text-classification", "text-generation", "text2text-generation", "image-classification", "popular-models"],
        help="Run a predefined benchmark suite"
    )
    model_group.add_argument(
        "--config", type=str,
        help="Path to benchmark configuration file"
    )
    
    # Hardware options
    hardware_group = parser.add_argument_group("Hardware Options")
    hardware_group.add_argument(
        "--hardware", type=str, nargs="+", default=["cpu"],
        help="Hardware platforms to benchmark on"
    )
    hardware_group.add_argument(
        "--all-hardware", action="store_true",
        help="Benchmark on all available hardware platforms"
    )
    
    # Benchmark parameters
    param_group = parser.add_argument_group("Benchmark Parameters")
    param_group.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4],
        help="Batch sizes to benchmark"
    )
    param_group.add_argument(
        "--sequence-lengths", type=int, nargs="+", default=[16, 32, 64],
        help="Sequence lengths to benchmark"
    )
    param_group.add_argument(
        "--metrics", type=str, nargs="+", default=["latency", "throughput", "memory"],
        help="Metrics to collect"
    )
    param_group.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup iterations"
    )
    param_group.add_argument(
        "--iterations", type=int, default=20,
        help="Number of benchmark iterations"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output-dir", type=str, default="benchmark_results",
        help="Directory to save results"
    )
    output_group.add_argument(
        "--export-formats", type=str, nargs="+", default=["json"],
        choices=["json", "csv", "markdown"],
        help="Export formats"
    )
    output_group.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to disk"
    )
    
    # Visualization options
    viz_group = parser.add_argument_group("Visualization Options")
    viz_group.add_argument(
        "--plot", action="store_true",
        help="Generate plots of benchmark results"
    )
    viz_group.add_argument(
        "--dashboard", action="store_true",
        help="Generate interactive dashboard of benchmark results"
    )
    
    # HuggingFace Hub options
    hub_group = parser.add_argument_group("HuggingFace Hub Options")
    hub_group.add_argument(
        "--publish-to-hub", action="store_true",
        help="Publish results to HuggingFace Hub"
    )
    hub_group.add_argument(
        "--token", type=str,
        help="HuggingFace Hub token for publishing"
    )
    
    # Misc options
    misc_group = parser.add_argument_group("Miscellaneous Options")
    misc_group.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase verbosity (can be specified multiple times)"
    )
    misc_group.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress non-error output"
    )
    
    return parser.parse_args()

def get_log_level(verbose: int, quiet: bool) -> int:
    """Get the logging level based on verbosity."""
    if quiet:
        return logging.ERROR
    elif verbose == 0:
        return logging.INFO
    elif verbose == 1:
        return logging.DEBUG
    else:
        return logging.DEBUG

def validate_args(args) -> bool:
    """Validate command-line arguments."""
    if not args.model and not args.suite and not args.config:
        logger.error("Error: At least one of --model, --suite, or --config must be specified")
        return False
    return True

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set log level
    log_level = get_log_level(args.verbose, args.quiet)
    logger.setLevel(log_level)
    
    # Validate arguments
    if not validate_args(args):
        return 1
    
    # Create output directory
    if not args.no_save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Common config parameters
        common_config = {
            "batch_sizes": args.batch_sizes,
            "sequence_lengths": args.sequence_lengths,
            "hardware": args.hardware,
            "metrics": args.metrics,
            "warmup_iterations": args.warmup,
            "test_iterations": args.iterations,
            "save_results": not args.no_save,
            "output_dir": args.output_dir
        }
        
        # Task-specific parameter
        if args.task:
            common_config["task"] = args.task
        
        # Run benchmarks
        if args.suite:
            # Run a predefined suite
            suite = BenchmarkSuite.from_predefined_suite(args.suite, **common_config)
            suite_results = suite.run()
            
            # Process results
            for model_id, results in suite_results.items():
                # Export results
                for export_format in args.export_formats:
                    if export_format == "json":
                        results.export_to_json()
                    elif export_format == "csv":
                        results.export_to_csv()
                    elif export_format == "markdown":
                        results.export_to_markdown()
                
                # Generate plots
                if args.plot:
                    results.plot_latency_comparison()
                    results.plot_throughput_scaling()
                    results.plot_memory_usage()
                
                # Publish to Hub
                if args.publish_to_hub:
                    results.publish_to_hub(token=args.token)
            
            # Generate dashboard
            if args.dashboard:
                try:
                    from test.tools.skills.refactored_benchmark_suite.visualizers.dashboard import generate_dashboard
                    dashboard_path = generate_dashboard(list(suite_results.values()), output_dir=args.output_dir)
                    logger.info(f"Generated dashboard: {dashboard_path}")
                except ImportError:
                    logger.error("Dashboard generation requires dash and plotly. Install with 'pip install dash plotly'")
                except Exception as e:
                    logger.error(f"Error generating dashboard: {e}")
        
        elif args.config:
            # Run from config file
            try:
                from test.tools.skills.refactored_benchmark_suite.config.benchmark_config import create_benchmark_configs_from_file
                
                # Load configurations
                configs = create_benchmark_configs_from_file(args.config)
                
                if not configs:
                    logger.error(f"No valid configurations found in {args.config}")
                    return 1
                
                # Run benchmarks for each configuration
                results = {}
                for config in configs:
                    model_id = config.pop("model_id")
                    logger.info(f"Benchmarking model {model_id} from config file")
                    
                    benchmark = ModelBenchmark(model_id=model_id, **config)
                    model_results = benchmark.run()
                    results[model_id] = model_results
                    
                    # Export results
                    for export_format in args.export_formats:
                        if export_format == "json":
                            model_results.export_to_json()
                        elif export_format == "csv":
                            model_results.export_to_csv()
                        elif export_format == "markdown":
                            model_results.export_to_markdown()
                    
                    # Generate plots
                    if args.plot:
                        model_results.plot_latency_comparison()
                        model_results.plot_throughput_scaling()
                        model_results.plot_memory_usage()
                    
                    # Publish to Hub
                    if args.publish_to_hub:
                        model_results.publish_to_hub(token=args.token)
                        
                # Generate dashboard for all results
                if args.dashboard and results:
                    try:
                        from test.tools.skills.refactored_benchmark_suite.visualizers.dashboard import generate_dashboard
                        dashboard_path = generate_dashboard(list(results.values()), output_dir=args.output_dir)
                        logger.info(f"Generated dashboard: {dashboard_path}")
                    except ImportError:
                        logger.error("Dashboard generation requires dash and plotly. Install with 'pip install dash plotly'")
                    except Exception as e:
                        logger.error(f"Error generating dashboard: {e}")
                
            except Exception as e:
                logger.error(f"Error running benchmarks from config file: {e}")
                return 1
        
        else:
            # Run individual models
            for model_id in args.model:
                logger.info(f"Benchmarking model: {model_id}")
                benchmark = ModelBenchmark(model_id=model_id, **common_config)
                results = benchmark.run()
                
                # Export results
                for export_format in args.export_formats:
                    if export_format == "json":
                        results.export_to_json()
                    elif export_format == "csv":
                        results.export_to_csv()
                    elif export_format == "markdown":
                        results.export_to_markdown()
                
                # Generate plots
                if args.plot:
                    results.plot_latency_comparison()
                    results.plot_throughput_scaling()
                    results.plot_memory_usage()
                
                # Publish to Hub
                if args.publish_to_hub:
                    results.publish_to_hub(token=args.token)
        
        logger.info("Benchmark complete")
        return 0
    
    except Exception as e:
        logger.error(f"Error during benchmark: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())