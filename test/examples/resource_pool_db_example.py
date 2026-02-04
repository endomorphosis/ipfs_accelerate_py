#!/usr/bin/env python3
"""
WebGPU/WebNN Resource Pool Database Integration Example

This example demonstrates how to use the DuckDB integration with the WebGPU/WebNN
Resource Pool to store performance metrics, generate reports, and visualize performance data.

Usage:
    python resource_pool_db_example.py

Options:
    --db-path PATH      Path to database file (default: use environment variable or default)
    --report-format FMT Report format: json, html, markdown (default: html)
    --output-dir DIR    Output directory for reports (default: ./reports)
    --visualize         Create visualizations
    --days DAYS         Number of days to include in reports (default: 30)
    --model MODEL       Specific model to analyze (optional)
    --browser BROWSER   Specific browser to analyze (optional)
"""

import os
import sys
import argparse
import anyio
import json
from pathlib import Path
from datetime import datetime

# Check fixed_web_platform path
script_dir = Path(__file__).parent
root_dir = script_dir.parent

# Add root directory to path to allow importing modules
sys.path.insert(0, str(root_dir))

try:
    from test.tests.web.web_platform.resource_pool_bridge_integration import ResourcePoolBridgeIntegration
except ImportError:
    print("Error: Could not import ResourcePoolBridgeIntegration. Make sure the path is correct.")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="WebGPU/WebNN Resource Pool Database Integration Example")
    parser.add_argument("--db-path", type=str, help="Path to database file")
    parser.add_argument("--report-format", type=str, default="html", 
                        choices=["json", "html", "markdown"], 
                        help="Report format (json, html, markdown)")
    parser.add_argument("--output-dir", type=str, default="./reports", 
                        help="Output directory for reports")
    parser.add_argument("--visualize", action="store_true", 
                        help="Create visualizations")
    parser.add_argument("--days", type=int, default=30, 
                        help="Number of days to include in reports")
    parser.add_argument("--model", type=str, 
                        help="Specific model to analyze")
    parser.add_argument("--browser", type=str,
                        help="Specific browser to analyze")
    return parser.parse_args()

async def run_example(args):
    """Run the resource pool database integration example."""
    
    # Set up database path (using argument, environment variable, or default)
    db_path = args.db_path
    if not db_path:
        db_path = os.environ.get("BENCHMARK_DB_PATH", "benchmark_db.duckdb")
    
    print(f"Using database: {db_path}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create resource pool with database integration
    pool = ResourcePoolBridgeIntegration(
        # These are mock connections for simulation/example purposes
        browser_connections={
            "conn_1": {
                "browser": "chrome",
                "platform": "webgpu",
                "active": True,
                "is_simulation": True,
                "loaded_models": set(),
                "resource_usage": {
                    "memory_mb": 500,
                    "cpu_percent": 20,
                    "gpu_percent": 30
                },
                "bridge": None
            },
            "conn_2": {
                "browser": "firefox",
                "platform": "webgpu",
                "active": True,
                "is_simulation": True,
                "loaded_models": set(),
                "resource_usage": {
                    "memory_mb": 450,
                    "cpu_percent": 15,
                    "gpu_percent": 40
                },
                "bridge": None
            },
            "conn_3": {
                "browser": "edge",
                "platform": "webnn",
                "active": True,
                "is_simulation": True,
                "loaded_models": set(),
                "resource_usage": {
                    "memory_mb": 350,
                    "cpu_percent": 10,
                    "gpu_percent": 20
                },
                "bridge": None
            }
        },
        max_connections=4,
        browser_preferences={
            'audio': 'firefox',     # Firefox for audio models
            'vision': 'chrome',     # Chrome for vision models
            'text_embedding': 'edge' # Edge for embedding models
        },
        adaptive_scaling=True,
        db_path=db_path,
        enable_tensor_sharing=True,
        enable_ultra_low_precision=True
    )
    
    # Initialize
    print("Initializing resource pool with database integration...")
    success = await pool.initialize()
    if not success:
        print("Failed to initialize resource pool")
        return
    
    try:
        # Simulate using the resource pool with different models
        print("\nSimulating model usage to generate metrics data...")
        
        # Text model (BERT) on Edge browser using WebNN
        print("  - Loading text embedding model (BERT) on Edge with WebNN...")
        text_conn_id, text_conn = await pool.get_connection(
            model_type="text_embedding", 
            model_name="bert-base-uncased",
            platform="webnn",
            browser="edge"
        )
        
        # Vision model (ViT) on Chrome browser using WebGPU
        print("  - Loading vision model (ViT) on Chrome with WebGPU...")
        vision_conn_id, vision_conn = await pool.get_connection(
            model_type="vision", 
            model_name="vit-base",
            platform="webgpu",
            browser="chrome"
        )
        
        # Audio model (Whisper) on Firefox browser using WebGPU with compute shaders
        print("  - Loading audio model (Whisper) on Firefox with compute shaders...")
        audio_conn_id, audio_conn = await pool.get_connection(
            model_type="audio", 
            model_name="whisper-tiny",
            platform="webgpu",
            browser="firefox"
        )
        
        # Simulate model inference and release connections with performance metrics
        
        # BERT inference
        print("  - Running BERT inference and storing metrics...")
        await pool.release_connection(
            text_conn_id, 
            success=True, 
            metrics={
                "model_name": "bert-base-uncased",
                "model_type": "text_embedding",
                "inference_time_ms": 25.8,
                "throughput": 38.7,
                "memory_mb": 380,
                "response_time_ms": 28.0,
                "compute_shader_optimized": False,
                "precompile_shaders": True,
                "parallel_loading": False,
                "mixed_precision": False,
                "precision_bits": 16,
                "initialization_time_ms": 120.5,
                "batch_size": 1,
                "params": "110M",
                "resource_usage": {
                    "memory_mb": 380,
                    "cpu_percent": 20,
                    "gpu_percent": 25
                }
            }
        )
        
        # ViT inference
        print("  - Running ViT inference and storing metrics...")
        await pool.release_connection(
            vision_conn_id, 
            success=True, 
            metrics={
                "model_name": "vit-base",
                "model_type": "vision",
                "inference_time_ms": 85.3,
                "throughput": 11.7,
                "memory_mb": 520,
                "response_time_ms": 90.0,
                "compute_shader_optimized": False,
                "precompile_shaders": True,
                "parallel_loading": True,
                "mixed_precision": False,
                "precision_bits": 16,
                "initialization_time_ms": 240.5,
                "batch_size": 1,
                "params": "86M",
                "resource_usage": {
                    "memory_mb": 520,
                    "cpu_percent": 30,
                    "gpu_percent": 45
                }
            }
        )
        
        # Whisper inference
        print("  - Running Whisper inference and storing metrics...")
        await pool.release_connection(
            audio_conn_id, 
            success=True, 
            metrics={
                "model_name": "whisper-tiny",
                "model_type": "audio",
                "inference_time_ms": 120.5,
                "throughput": 8.3,
                "memory_mb": 450,
                "response_time_ms": 125.0,
                "compute_shader_optimized": True,
                "precompile_shaders": True,
                "parallel_loading": False,
                "mixed_precision": False,
                "precision_bits": 16,
                "initialization_time_ms": 180.5,
                "batch_size": 1,
                "params": "39M",
                "resource_usage": {
                    "memory_mb": 450,
                    "cpu_percent": 25,
                    "gpu_percent": 40
                }
            }
        )
        
        # Check if database integration is working
        if not hasattr(pool, 'db_integration') or not pool.db_integration:
            print("\nWARNING: Database integration is not available.")
            return
        
        # Generate performance report
        print("\nGenerating performance report...")
        report = pool.get_performance_report(
            model_name=args.model,
            browser=args.browser,
            days=args.days,
            output_format=args.report_format
        )
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_part = f"_{args.model}" if args.model else ""
        browser_part = f"_{args.browser}" if args.browser else ""
        
        report_filename = f"performance_report{model_part}{browser_part}_{timestamp}.{args.report_format}"
        report_path = output_dir / report_filename
        
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Performance report saved to {report_path}")
        
        # Create visualization if requested
        if args.visualize:
            print("\nCreating performance visualizations...")
            
            # Generate visualization for throughput and latency
            vis_filename = f"performance_chart{model_part}{browser_part}_{timestamp}.png"
            vis_path = output_dir / vis_filename
            
            success = pool.create_performance_visualization(
                model_name=args.model,
                metrics=['throughput', 'latency', 'memory'],
                days=args.days,
                output_file=str(vis_path)
            )
            
            if success:
                print(f"Visualization saved to {vis_path}")
            else:
                print("Failed to create visualization")
        
        # Print summary statistics
        print("\nResource Pool Statistics:")
        stats = pool.get_stats()
        
        # Format and display key stats
        print(f"  - Database integration: {stats.get('database_integration', {}).get('enabled', False)}")
        print(f"  - Database path: {stats.get('database_integration', {}).get('db_path', 'unknown')}")
        
        # Get browser distribution
        browser_dist = stats.get('browser_distribution', {})
        print("\nBrowser Distribution:")
        for browser, count in browser_dist.items():
            print(f"  - {browser}: {count} connection(s)")
        
        # Get model distribution
        model_dist = stats.get('model_connections', {}).get('model_distribution', {})
        print("\nModel Distribution:")
        for model_type, count in model_dist.items():
            print(f"  - {model_type}: {count} model(s)")
        
    finally:
        # Clean up
        print("\nClosing resource pool and database connection...")
        await pool.close()
        print("Cleanup complete")

def main():
    """Main function to run the example."""
    args = parse_args()
    anyio.run(run_example(args))
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()