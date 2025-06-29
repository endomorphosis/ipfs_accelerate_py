#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace Model Hub Benchmark Publisher

This script publishes hardware compatibility and performance benchmarks to 
HuggingFace Model Hub model cards, creating standardized performance badges
and detailed benchmark information.

Features:
- Extracts benchmark data from hardware_compatibility_matrix.duckdb
- Formats metrics according to Model Hub metadata requirements
- Publishes metrics to model cards using the HuggingFace Hub API
- Generates standardized performance badges for models
- Outputs markdown performance tables for model card inclusion
"""

import os
import sys
import json
import time
import argparse
import tempfile
import logging
from datetime import datetime
from pathlib import Path

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("Warning: DuckDB not available. Please install with 'pip install duckdb'")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: Pandas not available. Please install with 'pip install pandas'")

try:
    from huggingface_hub import (
        HfApi, 
        ModelCard, 
        ModelCardData, 
        CardData,
        login
    )
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    print("Warning: Hugging Face Hub not available. Please install with 'pip install huggingface_hub'")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"hf_benchmark_publish_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hardware platform info for badges
HARDWARE_DISPLAY_NAMES = {
    "cpu": "CPU",
    "cuda": "CUDA",
    "mps": "MPS",
    "openvino": "OpenVINO",
    "webnn": "WebNN",
    "webgpu": "WebGPU",
}

HARDWARE_COLORS = {
    "cpu": "blue",
    "cuda": "orange",
    "mps": "green",
    "openvino": "red",
    "webnn": "purple",
    "webgpu": "brown",
}

# Badge templates
BADGE_TEMPLATE = "https://img.shields.io/badge/{label}-{value}-{color}"
BENCHMARK_BADGE_TEMPLATE = "https://img.shields.io/badge/{hardware}%20Inference-{time_ms}ms-{color}"

def extract_benchmarks_from_db(db_path="hardware_compatibility_matrix.duckdb"):
    """
    Extract performance benchmark data from the DuckDB database
    
    Args:
        db_path: Path to the DuckDB database file
        
    Returns:
        pandas DataFrame with benchmark results
    """
    if not DUCKDB_AVAILABLE or not PANDAS_AVAILABLE:
        logger.error("DuckDB and/or Pandas not available, cannot extract benchmarks")
        return None
    
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return None
    
    try:
        # Connect to database
        conn = duckdb.connect(db_path)
        
        # Query for latest benchmark results for each model and hardware
        df = conn.execute("""
            WITH ranked_results AS (
                SELECT
                    model_id,
                    model_type,
                    hardware,
                    success,
                    load_time,
                    inference_time,
                    memory_usage,
                    timestamp,
                    ROW_NUMBER() OVER (PARTITION BY model_id, hardware ORDER BY timestamp DESC) as rn
                FROM hardware_results
                WHERE success = TRUE
            )
            SELECT
                model_id,
                model_type,
                hardware,
                load_time,
                inference_time,
                memory_usage,
                timestamp
            FROM ranked_results
            WHERE rn = 1
            ORDER BY model_id, hardware
        """).fetchdf()
        
        conn.close()
        
        if df.empty:
            logger.warning("No benchmark results found in database")
            return None
        
        logger.info(f"Extracted {len(df)} benchmark results for {df['model_id'].nunique()} models")
        return df
    
    except Exception as e:
        logger.error(f"Error querying database: {e}")
        return None

def format_benchmark_for_model_card(model_benchmarks):
    """
    Format benchmark data for inclusion in model card
    
    Args:
        model_benchmarks: DataFrame with benchmark results for a specific model
        
    Returns:
        dict with formatted benchmark data and markdown table
    """
    if model_benchmarks.empty:
        return None
    
    # Sort by hardware platform
    model_benchmarks = model_benchmarks.sort_values("hardware")
    
    # Create markdown table
    markdown_table = "## Performance Benchmarks\n\n"
    markdown_table += "| Hardware | Inference Time (ms) | Memory Usage (MB) | Load Time (s) |\n"
    markdown_table += "|----------|---------------------|-------------------|---------------|\n"
    
    # Add badge markdown
    badges_markdown = ""
    
    # Metadata dictionary
    metadata = {
        "benchmarks": []
    }
    
    # Add rows to table and metadata
    for _, row in model_benchmarks.iterrows():
        hardware = row["hardware"]
        hw_display = HARDWARE_DISPLAY_NAMES.get(hardware, hardware)
        
        # Convert to ms for display
        inference_ms = row["inference_time"] * 1000 if pd.notna(row["inference_time"]) else None
        
        # Format values
        inference_str = f"{inference_ms:.1f}" if inference_ms is not None else "N/A"
        memory_str = f"{row['memory_usage']:.1f}" if pd.notna(row["memory_usage"]) else "N/A"
        load_str = f"{row['load_time']:.2f}" if pd.notna(row["load_time"]) else "N/A"
        
        # Add to table
        markdown_table += f"| {hw_display} | {inference_str} | {memory_str} | {load_str} |\n"
        
        # Add to metadata
        metadata["benchmarks"].append({
            "hardware": hardware,
            "inference_time_ms": float(f"{inference_ms:.1f}") if inference_ms is not None else None,
            "memory_usage_mb": float(f"{row['memory_usage']:.1f}") if pd.notna(row["memory_usage"]) else None,
            "load_time_s": float(f"{row['load_time']:.2f}") if pd.notna(row["load_time"]) else None,
            "timestamp": row["timestamp"].isoformat() if isinstance(row["timestamp"], datetime) else str(row["timestamp"])
        })
        
        # Create badge for this hardware
        if inference_ms is not None:
            color = HARDWARE_COLORS.get(hardware, "blue")
            badge_url = BENCHMARK_BADGE_TEMPLATE.format(
                hardware=hw_display,
                time_ms=f"{inference_ms:.1f}",
                color=color
            )
            badges_markdown += f"![{hw_display} Benchmark]({badge_url}) "
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown_table += f"\n*Benchmarks last updated: {timestamp}*\n\n"
    markdown_table += "*Measured with IPFS Accelerate Testing Framework*\n\n"
    
    # Add badges section
    markdown_table = badges_markdown + "\n\n" + markdown_table
    
    # Add CPU vs GPU speedup if available
    cpu_time = None
    gpu_time = None
    
    for benchmark in metadata["benchmarks"]:
        if benchmark["hardware"] == "cpu" and benchmark["inference_time_ms"] is not None:
            cpu_time = benchmark["inference_time_ms"]
        elif benchmark["hardware"] == "cuda" and benchmark["inference_time_ms"] is not None:
            gpu_time = benchmark["inference_time_ms"]
    
    if cpu_time is not None and gpu_time is not None and gpu_time > 0:
        speedup = cpu_time / gpu_time
        markdown_table += f"\n**CPU to GPU Speedup: {speedup:.1f}x**\n"
        metadata["cpu_gpu_speedup"] = float(f"{speedup:.1f}")
    
    # Add methodology
    markdown_table += """
## Benchmark Methodology

These benchmarks were collected using the IPFS Accelerate Testing Framework:

- **Inference Time**: Average time for a single forward pass
- **Memory Usage**: Peak memory consumption during inference
- **Load Time**: Time to load model from disk and initialize

Measurements performed in a controlled environment with warm-up passes to ensure stability.
"""

    return {
        "markdown": markdown_table,
        "metadata": metadata
    }

def update_model_card(model_id, benchmark_data, token=None, commit_message="Update performance benchmarks"):
    """
    Update the model card for a specific model with benchmark information
    
    Args:
        model_id: The Hugging Face model ID (e.g., "bert-base-uncased")
        benchmark_data: Dict with markdown and metadata for benchmarks
        token: HF API token for authentication
        commit_message: Custom commit message for the update
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    if not HF_HUB_AVAILABLE:
        logger.error("Hugging Face Hub not available, cannot update model card")
        return False
    
    if not benchmark_data:
        logger.error(f"No benchmark data provided for model {model_id}")
        return False
    
    try:
        # Initialize API
        api = HfApi()
        
        # Login if token provided
        if token:
            login(token)
        
        logger.info(f"Updating model card for {model_id}")
        
        # Try to get existing model card
        try:
            model_card = ModelCard.load(model_id)
            logger.info(f"Loaded existing model card for {model_id}")
        except Exception as e:
            logger.warning(f"Could not load existing model card for {model_id}: {e}")
            # Create new model card
            model_card = ModelCard(ModelCardData())
        
        # Add benchmarks to model card metadata
        if model_card.data:
            # Add benchmarks to existing data
            model_card.data.to_dict()["benchmark"] = benchmark_data["metadata"]
        else:
            # Create new data with benchmarks
            card_data = {"benchmark": benchmark_data["metadata"]}
            model_card.data = ModelCardData.from_dict(card_data)
        
        # Add benchmark section to model card content
        benchmark_section = benchmark_data["markdown"]
        
        if model_card.text:
            # Try to update existing benchmark section
            if "## Performance Benchmarks" in model_card.text:
                # Replace existing section
                parts = model_card.text.split("## Performance Benchmarks")
                before = parts[0]
                
                # Find the next section
                after_parts = parts[1].split("\n## ")
                if len(after_parts) > 1:
                    after = "\n## " + "\n## ".join(after_parts[1:])
                else:
                    after = ""
                
                model_card.text = before + benchmark_section + after
            else:
                # Add new section at the end
                model_card.text += "\n\n" + benchmark_section
        else:
            # Create new model card text
            model_card.text = f"# {model_id}\n\n{benchmark_section}"
        
        # Save card to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            model_card.save(f.name)
            temp_path = f.name
        
        # Push to hub
        api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo="README.md",
            repo_id=model_id,
            repo_type="model",
            commit_message=commit_message
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        logger.info(f"Successfully updated model card for {model_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error updating model card for {model_id}: {e}")
        return False

def publish_benchmarks_to_hub(benchmarks_df, token=None, dry_run=False, limit=None):
    """
    Publish benchmarks to model cards on the HF Hub
    
    Args:
        benchmarks_df: DataFrame with benchmark results
        token: HF API token for authentication
        dry_run: If True, don't actually update cards, just generate data
        limit: Maximum number of models to update
        
    Returns:
        dict: Summary of update results
    """
    if benchmarks_df is None or benchmarks_df.empty:
        logger.error("No benchmark data available")
        return {
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "total": 0
        }
    
    # Get unique model IDs
    model_ids = benchmarks_df["model_id"].unique()
    
    if limit:
        model_ids = model_ids[:limit]
    
    logger.info(f"Publishing benchmarks for {len(model_ids)} models" + (" (DRY RUN)" if dry_run else ""))
    
    results = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "total": len(model_ids),
        "models": {}
    }
    
    for model_id in model_ids:
        logger.info(f"Processing model: {model_id}")
        
        # Get benchmarks for this model
        model_benchmarks = benchmarks_df[benchmarks_df["model_id"] == model_id].copy()
        
        if model_benchmarks.empty:
            logger.warning(f"No benchmarks found for model {model_id}")
            results["skipped"] += 1
            results["models"][model_id] = "skipped"
            continue
        
        # Format benchmarks for model card
        benchmark_data = format_benchmark_for_model_card(model_benchmarks)
        
        if benchmark_data is None:
            logger.warning(f"Failed to format benchmarks for model {model_id}")
            results["failed"] += 1
            results["models"][model_id] = "failed"
            continue
        
        # In dry run mode, just print the markdown
        if dry_run:
            logger.info(f"Benchmark markdown for {model_id} (DRY RUN):")
            logger.info(benchmark_data["markdown"])
            results["success"] += 1
            results["models"][model_id] = "dry_run"
            continue
        
        # Update model card on Hub
        success = update_model_card(
            model_id=model_id,
            benchmark_data=benchmark_data,
            token=token,
            commit_message="Update performance benchmarks from IPFS Accelerate Testing Framework"
        )
        
        if success:
            results["success"] += 1
            results["models"][model_id] = "success"
        else:
            results["failed"] += 1
            results["models"][model_id] = "failed"
        
        # Sleep briefly to avoid rate limits
        time.sleep(1)
    
    # Log summary
    logger.info(f"Benchmark publishing complete: {results['success']} succeeded, {results['failed']} failed, {results['skipped']} skipped")
    
    return results

def save_local_benchmarks(benchmarks_df, output_dir="benchmark_reports"):
    """
    Save benchmark data locally as markdown files
    
    Args:
        benchmarks_df: DataFrame with benchmark results
        output_dir: Directory to save reports
        
    Returns:
        int: Number of reports generated
    """
    if benchmarks_df is None or benchmarks_df.empty:
        logger.error("No benchmark data available")
        return 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique model IDs
    model_ids = benchmarks_df["model_id"].unique()
    
    logger.info(f"Generating local benchmark reports for {len(model_ids)} models")
    
    reports_generated = 0
    
    for model_id in model_ids:
        # Get benchmarks for this model
        model_benchmarks = benchmarks_df[benchmarks_df["model_id"] == model_id].copy()
        
        if model_benchmarks.empty:
            continue
        
        # Format benchmarks for model card
        benchmark_data = format_benchmark_for_model_card(model_benchmarks)
        
        if benchmark_data is None:
            continue
        
        # Create safe filename
        safe_name = model_id.replace("/", "__")
        output_file = os.path.join(output_dir, f"{safe_name}_benchmark.md")
        
        # Save markdown to file
        with open(output_file, "w") as f:
            f.write(f"# Performance Benchmarks for {model_id}\n\n")
            f.write(benchmark_data["markdown"])
        
        # Save metadata to JSON
        json_file = os.path.join(output_dir, f"{safe_name}_benchmark.json")
        with open(json_file, "w") as f:
            json.dump(benchmark_data["metadata"], f, indent=2)
        
        reports_generated += 1
    
    logger.info(f"Generated {reports_generated} local benchmark reports in {output_dir}")
    return reports_generated

def main():
    parser = argparse.ArgumentParser(description="HuggingFace Model Hub Benchmark Publisher")
    parser.add_argument("--db", type=str, default="hardware_compatibility_matrix.duckdb",
                       help="Path to DuckDB database with benchmark results")
    parser.add_argument("--token", type=str,
                       help="HuggingFace API token (or use HF_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Generate benchmark data but don't publish to Hub")
    parser.add_argument("--local", action="store_true",
                       help="Save benchmark reports locally instead of publishing to Hub")
    parser.add_argument("--output-dir", type=str, default="benchmark_reports",
                       help="Directory to save local benchmark reports")
    parser.add_argument("--limit", type=int,
                       help="Limit the number of models to process")
    parser.add_argument("--model", type=str,
                       help="Process only the specified model ID")
    args = parser.parse_args()
    
    # Check for required packages
    missing_packages = []
    if not DUCKDB_AVAILABLE:
        missing_packages.append("duckdb")
    if not PANDAS_AVAILABLE:
        missing_packages.append("pandas")
    if not args.local and not HF_HUB_AVAILABLE:
        missing_packages.append("huggingface_hub")
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error(f"Please install with: pip install {' '.join(missing_packages)}")
        return 1
    
    # Check for database
    if not os.path.exists(args.db):
        logger.error(f"Database file not found: {args.db}")
        return 1
    
    # Get API token from args or env
    token = args.token or os.environ.get("HF_TOKEN")
    if not token and not args.dry_run and not args.local:
        logger.warning("No HuggingFace API token provided. Using anonymous access (may fail for private repos).")
    
    # Extract benchmarks from database
    logger.info(f"Extracting benchmarks from {args.db}")
    benchmarks_df = extract_benchmarks_from_db(args.db)
    
    if benchmarks_df is None or benchmarks_df.empty:
        logger.error("Failed to extract benchmarks from database")
        return 1
    
    # Filter by model if specified
    if args.model:
        benchmarks_df = benchmarks_df[benchmarks_df["model_id"] == args.model]
        if benchmarks_df.empty:
            logger.error(f"No benchmarks found for model {args.model}")
            return 1
    
    # Save locally or publish to Hub
    if args.local:
        reports_count = save_local_benchmarks(benchmarks_df, args.output_dir)
        if reports_count == 0:
            logger.warning("No benchmark reports were generated")
            return 1
    else:
        results = publish_benchmarks_to_hub(
            benchmarks_df=benchmarks_df,
            token=token,
            dry_run=args.dry_run,
            limit=args.limit
        )
        
        if results["success"] == 0 and results["total"] > 0:
            logger.warning("No benchmarks were successfully published")
            return 1
    
    logger.info("Benchmark publishing completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())