#!/usr/bin/env python3
"""
Hardware Optimization Recommendation Demo

This script demonstrates how to use the hardware optimization recommendation system
to get performance improvement suggestions for specific model and hardware combinations.
It leverages the Benchmark API and Predictive Performance API data to provide
data-driven optimization recommendations.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("optimization_demo")

# Import components
try:
    from test.optimization_recommendation.optimization_client import OptimizationClient
    from test.api_client.predictive_performance_client import (
        PredictivePerformanceClient,
        HardwarePlatform,
        PrecisionType
    )
    from test.integration.benchmark_predictive_performance_bridge import (
        BenchmarkPredictivePerformanceBridge
    )
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Required components not available: {e}")
    COMPONENTS_AVAILABLE = False

def print_table(rows, header=None, padding=2):
    """Print a formatted table."""
    if not rows:
        return
    
    # Add header if provided
    if header:
        rows = [header] + rows
    
    # Calculate column widths
    col_widths = []
    for i in range(len(rows[0])):
        col_widths.append(max(len(str(row[i])) for row in rows) + padding)
    
    # Print header with separator
    if header:
        print("".join(str(rows[0][i]).ljust(col_widths[i]) for i in range(len(rows[0]))))
        print("".join("-" * col_widths[i] for i in range(len(col_widths))))
        rows = rows[1:]
    
    # Print rows
    for row in rows:
        print("".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))))

def generate_sample_data(client, num_models=3):
    """Generate sample data for demonstration."""
    print(f"Generating sample data with {num_models} models...")
    result = client.generate_sample_data(num_models=num_models, wait=True)
    
    if "error" in result:
        print(f"Error generating sample data: {result['error']}")
        return False
    
    print("Sample data generated successfully!")
    return True

def sync_benchmark_data(bridge, limit=50):
    """Synchronize benchmark data with predictive performance system."""
    print(f"Synchronizing up to {limit} benchmark results...")
    result = bridge.sync_recent_results(limit=limit)
    
    if not result.get("success", False):
        print(f"Error synchronizing benchmark data: {result.get('message', 'Unknown error')}")
        return False
    
    print(f"Synchronized {result['synced']} benchmark results ({result['failed']} failed)")
    return True

def demonstrate_optimization_workflow(
    benchmark_db_path="benchmark_db.duckdb",
    api_url="http://localhost:8080",
    api_key=None,
    model_name="bert-base-uncased",
    hardware="cuda"
):
    """
    Demonstrate complete optimization workflow.
    
    Args:
        benchmark_db_path: Path to benchmark database
        api_url: API base URL
        api_key: Optional API key
        model_name: Model to analyze
        hardware: Hardware platform to analyze
    """
    if not COMPONENTS_AVAILABLE:
        print("ERROR: Required components not available. Please install the necessary packages.")
        return False
    
    print("=" * 80)
    print(f"HARDWARE OPTIMIZATION RECOMMENDATION DEMO")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Hardware: {hardware}")
    print("-" * 80)
    
    # Initialize clients
    try:
        pp_client = PredictivePerformanceClient(base_url=api_url, api_key=api_key)
        bridge = BenchmarkPredictivePerformanceBridge(
            benchmark_db_path=benchmark_db_path,
            predictive_api_url=api_url,
            api_key=api_key
        )
        opt_client = OptimizationClient(
            benchmark_db_path=benchmark_db_path,
            predictive_api_url=api_url,
            api_key=api_key
        )
    except Exception as e:
        print(f"Error initializing clients: {e}")
        return False
    
    # Check connections
    connection_status = bridge.check_connections()
    if not connection_status["benchmark_db"]:
        print(f"ERROR: Could not connect to benchmark database at {benchmark_db_path}")
        return False
    
    if not connection_status["predictive_api"]:
        print(f"ERROR: Could not connect to Predictive Performance API at {api_url}")
        # Try to generate sample data
        print("Attempting to generate sample data...")
        generate_sample_data(pp_client)
    
    # Step 1: Ensure we have data by synchronizing benchmark data
    print("\nSTEP 1: Synchronizing benchmark data")
    sync_benchmark_data(bridge)
    
    # Step 2: Analyze performance data
    print("\nSTEP 2: Analyzing performance data")
    print(f"Analyzing {model_name} on {hardware}...")
    analysis = opt_client.analyze_performance(
        model_name=model_name,
        hardware_platform=hardware
    )
    
    if "error" in analysis:
        print(f"Error analyzing performance: {analysis['error']}")
        # Try a different model if specified model has no data
        if model_name == "bert-base-uncased":
            print("Trying with a different model...")
            model_name = "gpt2"
            print(f"Analyzing {model_name} on {hardware}...")
            analysis = opt_client.analyze_performance(
                model_name=model_name,
                hardware_platform=hardware
            )
    
    # Display analysis summary
    data_points = analysis.get("data_points", 0)
    print(f"Found {data_points} data points for analysis")
    
    if data_points > 0:
        # Print current performance metrics
        print("\nCurrent Performance Metrics:")
        metrics = []
        for metric in ["throughput", "latency", "memory_usage"]:
            if metric in analysis:
                mean = analysis[metric].get("mean", "N/A")
                if isinstance(mean, (int, float)):
                    mean = f"{mean:.2f}"
                metrics.append([metric.capitalize(), mean])
        
        print_table(metrics, ["Metric", "Mean Value"])
    else:
        print("No performance data available. Using simulated data for demo purposes.")
    
    # Step 3: Get optimization recommendations
    print("\nSTEP 3: Getting optimization recommendations")
    recommendations = opt_client.get_recommendations(
        model_name=model_name,
        hardware_platform=hardware
    )
    
    if "error" in recommendations:
        print(f"Error getting recommendations: {recommendations['error']}")
        return False
    
    # Display recommendations
    recs = recommendations.get("recommendations", [])
    print(f"Found {len(recs)} optimization recommendations")
    
    if recs:
        print("\nTop 3 Recommendations:")
        rows = []
        for i, rec in enumerate(recs[:3]):
            name = rec.get("name", "Unknown")
            description = rec.get("description", "")
            throughput_imp = rec.get("expected_improvements", {}).get("throughput_improvement", 0)
            latency_red = rec.get("expected_improvements", {}).get("latency_reduction", 0)
            confidence = rec.get("confidence", 0)
            
            rows.append([
                f"{i+1}. {name}",
                description,
                f"{throughput_imp*100:.1f}%",
                f"{latency_red*100:.1f}%",
                f"{confidence*100:.1f}%"
            ])
        
        print_table(rows, ["Recommendation", "Description", "Throughput ↑", "Latency ↓", "Confidence"])
        
        # Show detailed implementation for the top recommendation
        if recs:
            top_rec = recs[0]
            print("\nImplementation Details for Top Recommendation:")
            print(f"Name: {top_rec.get('name')}")
            print(f"Description: {top_rec.get('description')}")
            print("\nConfiguration Parameters:")
            for k, v in top_rec.get("configuration", {}).items():
                print(f"  - {k}: {v}")
            
            print("\nImplementation Guide:")
            print(f"{top_rec.get('implementation')}")
            
            print("\nExpected Improvements:")
            imp = top_rec.get("expected_improvements", {})
            print(f"  - Throughput: +{imp.get('throughput_improvement', 0)*100:.1f}%")
            print(f"  - Latency: -{imp.get('latency_reduction', 0)*100:.1f}%")
            print(f"  - Memory Usage: -{imp.get('memory_reduction', 0)*100:.1f}%")
    
    # Step 4: Generate comprehensive report
    print("\nSTEP 4: Generating comprehensive report")
    # Get a few model names for the report
    try:
        models = pp_client.list_measurements(limit=5)
        model_names = list(set([m["model_name"] for m in models.get("results", [])]))
        if not model_names:
            model_names = ["bert-base-uncased", "gpt2", "vit-base-patch16-224"]
    except:
        model_names = ["bert-base-uncased", "gpt2", "vit-base-patch16-224"]
    
    hardware_platforms = ["cuda", "cpu"]
    if hardware not in hardware_platforms:
        hardware_platforms.append(hardware)
    
    print(f"Generating report for {len(model_names)} models and {len(hardware_platforms)} hardware platforms...")
    report = opt_client.generate_report(
        model_names=model_names,
        hardware_platforms=hardware_platforms
    )
    
    if "error" in report:
        print(f"Error generating report: {report['error']}")
    else:
        # Display report summary
        print("\nReport Summary:")
        print(f"Generated at: {report.get('generated_at')}")
        print(f"Models analyzed: {report.get('models')}")
        print(f"Hardware platforms: {report.get('hardware_platforms')}")
        
        # Display top recommendations
        top_recs = report.get("top_recommendations", [])
        if top_recs:
            print(f"\nTop {len(top_recs)} Global Recommendations:")
            rows = []
            for i, rec in enumerate(top_recs[:5]):
                model = rec.get("model_name", "Unknown")
                hw = rec.get("hardware_platform", "Unknown")
                name = rec.get("recommendation", {}).get("name", "Unknown")
                throughput_imp = rec.get("recommendation", {}).get("expected_improvements", {}).get("throughput_improvement", 0)
                
                rows.append([
                    i+1,
                    model,
                    hw,
                    name,
                    f"{throughput_imp*100:.1f}%"
                ])
            
            print_table(rows, ["#", "Model", "Hardware", "Recommendation", "Throughput ↑"])
    
    # Clean up
    opt_client.close()
    
    print("\nOptimization recommendation demo completed successfully!")
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hardware Optimization Demo")
    parser.add_argument("--benchmark-db", type=str, default="benchmark_db.duckdb", 
                      help="Path to benchmark DuckDB database")
    parser.add_argument("--api-url", type=str, default="http://localhost:8080",
                      help="URL of the Predictive Performance API")
    parser.add_argument("--api-key", type=str, help="API key for authenticated endpoints")
    parser.add_argument("--model", type=str, default="bert-base-uncased", 
                      help="Model to analyze")
    parser.add_argument("--hardware", type=str, default="cuda",
                      help="Hardware platform to analyze")
    parser.add_argument("--sample-data", action="store_true",
                      help="Generate sample data before running demo")
    
    args = parser.parse_args()
    
    # Generate sample data if requested
    if args.sample_data:
        try:
            pp_client = PredictivePerformanceClient(base_url=args.api_url, api_key=args.api_key)
            generate_sample_data(pp_client)
        except Exception as e:
            print(f"Error generating sample data: {e}")
    
    # Run demonstration
    demonstrate_optimization_workflow(
        benchmark_db_path=args.benchmark_db,
        api_url=args.api_url,
        api_key=args.api_key,
        model_name=args.model,
        hardware=args.hardware
    )

if __name__ == "__main__":
    main()