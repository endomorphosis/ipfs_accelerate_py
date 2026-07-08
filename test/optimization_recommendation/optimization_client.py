#!/usr/bin/env python3
"""
Optimization Recommendation Client

This module provides a client for interacting with the Hardware Optimization Analyzer.
It allows easy access to optimization recommendations through a simple API.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("optimization_client")

# Import analyzer
try:
    from test.optimization_recommendation.hardware_optimization_analyzer import (
        HardwareOptimizationAnalyzer
    )
    ANALYZER_AVAILABLE = True
except ImportError:
    logger.warning("HardwareOptimizationAnalyzer not available")
    ANALYZER_AVAILABLE = False

try:
    from test.api_client.predictive_performance_client import (
        HardwarePlatform,
        PrecisionType,
        ModelMode
    )
    ENUMS_AVAILABLE = True
except ImportError:
    logger.warning("PredictivePerformanceClient enums not available")
    ENUMS_AVAILABLE = False

class OptimizationClient:
    """Client for accessing hardware optimization recommendations."""
    
    def __init__(
        self,
        benchmark_db_path: str = "benchmark_db.duckdb",
        predictive_api_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the client.
        
        Args:
            benchmark_db_path: Path to benchmark DuckDB database
            predictive_api_url: URL of the Predictive Performance API
            api_key: Optional API key for authenticated endpoints
            config: Optional configuration dictionary
        """
        self.benchmark_db_path = benchmark_db_path
        self.predictive_api_url = predictive_api_url
        self.api_key = api_key
        self.config = config or {}
        
        # Initialize analyzer
        self.analyzer = None
        if ANALYZER_AVAILABLE:
            try:
                self.analyzer = HardwareOptimizationAnalyzer(
                    benchmark_db_path=benchmark_db_path,
                    predictive_api_url=predictive_api_url,
                    api_key=api_key,
                    config=config
                )
                logger.info(f"Initialized optimization analyzer with benchmark DB at {benchmark_db_path}")
            except Exception as e:
                logger.error(f"Error initializing optimization analyzer: {e}")
    
    def get_recommendations(
        self,
        model_name: str,
        hardware_platform: Union[str, "HardwarePlatform"],
        model_family: Optional[str] = None,
        batch_size: Optional[int] = None,
        current_precision: Optional[Union[str, "PrecisionType"]] = None
    ) -> Dict[str, Any]:
        """
        Get optimization recommendations for a model on specific hardware.
        
        Args:
            model_name: Name of the model
            hardware_platform: Hardware platform
            model_family: Optional model family
            batch_size: Optional batch size
            current_precision: Optional current precision being used
            
        Returns:
            Dictionary with optimization recommendations
        """
        if not self.analyzer:
            return {"error": "Optimization analyzer not available"}
        
        # Convert enums if needed
        if ENUMS_AVAILABLE and isinstance(hardware_platform, HardwarePlatform):
            hardware_platform = hardware_platform.value
        
        if ENUMS_AVAILABLE and isinstance(current_precision, PrecisionType):
            current_precision = current_precision.value
        
        # Get recommendations
        return self.analyzer.get_optimization_recommendations(
            model_name=model_name,
            hardware_platform=hardware_platform,
            model_family=model_family,
            batch_size=batch_size,
            current_precision=current_precision
        )
    
    def analyze_performance(
        self,
        model_name: str,
        hardware_platform: Union[str, "HardwarePlatform"],
        batch_size: Optional[int] = None,
        days: int = 90,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze performance data for a model on specific hardware.
        
        Args:
            model_name: Name of the model
            hardware_platform: Hardware platform
            batch_size: Optional batch size filter
            days: Number of days to look back
            limit: Maximum number of records to analyze
            
        Returns:
            Dictionary with performance analysis
        """
        if not self.analyzer:
            return {"error": "Optimization analyzer not available"}
        
        # Convert enums if needed
        if ENUMS_AVAILABLE and isinstance(hardware_platform, HardwarePlatform):
            hardware_platform = hardware_platform.value
        
        # Get analysis
        return self.analyzer.analyze_performance_data(
            model_name=model_name,
            hardware_platform=hardware_platform,
            batch_size=batch_size,
            days=days,
            limit=limit
        )
    
    def get_available_strategies(
        self,
        hardware_platform: Union[str, "HardwarePlatform"]
    ) -> List[Dict[str, Any]]:
        """
        Get available optimization strategies for a hardware platform.
        
        Args:
            hardware_platform: Hardware platform
            
        Returns:
            List of optimization strategies
        """
        if not self.analyzer:
            return []
        
        # Convert enums if needed
        if ENUMS_AVAILABLE and isinstance(hardware_platform, HardwarePlatform):
            hardware_platform = hardware_platform.value
        
        # Get strategies
        return self.analyzer.get_optimization_strategies(hardware_platform)
    
    def generate_report(
        self,
        model_names: List[str],
        hardware_platforms: List[Union[str, "HardwarePlatform"]],
        batch_size: Optional[int] = None,
        current_precision: Optional[Union[str, "PrecisionType"]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive optimization report for multiple models and hardware platforms.
        
        Args:
            model_names: List of model names
            hardware_platforms: List of hardware platforms
            batch_size: Optional batch size filter
            current_precision: Optional current precision
            
        Returns:
            Dictionary with optimization report
        """
        if not self.analyzer:
            return {"error": "Optimization analyzer not available"}
        
        # Convert enums if needed
        if ENUMS_AVAILABLE:
            hardware_platforms = [
                hp.value if isinstance(hp, HardwarePlatform) else hp 
                for hp in hardware_platforms
            ]
            
        if ENUMS_AVAILABLE and isinstance(current_precision, PrecisionType):
            current_precision = current_precision.value
        
        # Generate report
        return self.analyzer.generate_optimization_report(
            model_names=model_names,
            hardware_platforms=hardware_platforms,
            batch_size=batch_size,
            current_precision=current_precision
        )
    
    def close(self):
        """Close connections."""
        if self.analyzer:
            self.analyzer.close()

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hardware Optimization Client")
    parser.add_argument("--benchmark-db", type=str, default="benchmark_db.duckdb", 
                      help="Path to benchmark DuckDB database")
    parser.add_argument("--api-url", type=str, default="http://localhost:8080",
                      help="URL of the Predictive Performance API")
    parser.add_argument("--api-key", type=str, help="API key for authenticated endpoints")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--hardware", type=str, help="Hardware platform")
    parser.add_argument("--family", type=str, help="Model family")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "int8", "int4"], 
                      help="Current precision")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive report")
    parser.add_argument("--models", type=str, nargs="+", help="List of models for report")
    parser.add_argument("--hardware-platforms", type=str, nargs="+", 
                      help="List of hardware platforms for report")
    parser.add_argument("--strategies", action="store_true", help="List available strategies")
    parser.add_argument("--output", type=str, help="Path to write JSON output")
    
    args = parser.parse_args()
    
    # Create client
    client = OptimizationClient(
        benchmark_db_path=args.benchmark_db,
        predictive_api_url=args.api_url,
        api_key=args.api_key
    )
    
    try:
        output = None
        
        # List available strategies
        if args.strategies and args.hardware:
            print(f"Listing optimization strategies for {args.hardware}...")
            strategies = client.get_available_strategies(args.hardware)
            print(f"Found {len(strategies)} strategies")
            output = strategies
            
        # Generate report for multiple models and hardware
        elif args.report:
            if not args.models or not args.hardware_platforms:
                print("ERROR: --models and --hardware-platforms are required for report generation")
                return 1
            
            print(f"Generating optimization report for {len(args.models)} models and {len(args.hardware_platforms)} hardware platforms...")
            report = client.generate_report(
                model_names=args.models,
                hardware_platforms=args.hardware_platforms,
                batch_size=args.batch_size,
                current_precision=args.precision
            )
            
            print(f"Report generated with {len(report.get('top_recommendations', []))} top recommendations")
            output = report
            
        # Get recommendations for specific model and hardware
        elif args.model and args.hardware:
            print(f"Getting optimization recommendations for {args.model} on {args.hardware}...")
            
            recommendations = client.get_recommendations(
                model_name=args.model,
                hardware_platform=args.hardware,
                model_family=args.family,
                batch_size=args.batch_size,
                current_precision=args.precision
            )
            
            num_recommendations = len(recommendations.get("recommendations", []))
            print(f"Found {num_recommendations} optimization recommendations")
            
            output = recommendations
            
        else:
            print("ERROR: Please specify --strategies and --hardware, or --model and --hardware, or --report with --models and --hardware-platforms")
            return 1
        
        # Write output to file
        if args.output and output:
            with open(args.output, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Output written to {args.output}")
        elif output:
            print(json.dumps(output, indent=2))
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return 1
    
    finally:
        # Close connections
        client.close()

if __name__ == "__main__":
    sys.exit(main())