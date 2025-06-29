#!/usr/bin/env python
"""
Simulation analysis and visualization tools for hardware benchmarks.

This module provides tools to analyze and visualize the simulation status
of hardware platforms in benchmark results. It helps users understand which
results are from real hardware vs. simulated environments.

Implementation date: April 10, 2025
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
import tempfile
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Third-party imports
try:
    import duckdb
    import pandas as pd
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Visualization dependencies not available. Install pandas and matplotlib for visualization.")
    VISUALIZATION_AVAILABLE = False

def get_db_connection(db_path=None):
    """Get a connection to the benchmark database"""
    if not db_path:
        db_path = os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
    
    logger.info(f"Connecting to database: {db_path}")
    return duckdb.connect(db_path)

def get_simulation_statistics(conn) -> Dict[str, Any]:
    """Get statistics on simulated vs. real hardware results in the database"""
    try:
        # Check if simulation columns exist
        has_simulation_columns = conn.execute("""
        SELECT 
            COUNT(*) > 0 as has_column
        FROM pragma_table_info('test_results')
        WHERE name = 'is_simulated'
        """).fetchone()[0]
        
        if not has_simulation_columns:
            logger.warning("Database does not have simulation tracking columns. Run apply_simulation_detection_fixes.py first.")
            return {
                "error": "Database schema missing simulation columns"
            }
        
        # Get overall statistics
        overall_stats = conn.execute("""
        SELECT 
            COUNT(*) as total_results,
            COUNT(CASE WHEN is_simulated = TRUE THEN 1 END) as simulated_results,
            COUNT(CASE WHEN is_simulated = FALSE THEN 1 END) as real_results,
            COUNT(CASE WHEN is_simulated IS NULL THEN 1 END) as unknown_results,
            ROUND(COUNT(CASE WHEN is_simulated = TRUE THEN 1 END) * 100.0 / 
                  NULLIF(COUNT(*), 0), 2) as simulated_percent
        FROM test_results
        """).fetchone()
        
        if not overall_stats:
            return {
                "error": "No test results found in database"
            }
        
        # Get statistics by hardware type
        hardware_stats = conn.execute("""
        SELECT 
            h.hardware_type,
            COUNT(*) as total_results,
            COUNT(CASE WHEN tr.is_simulated = TRUE THEN 1 END) as simulated_results,
            COUNT(CASE WHEN tr.is_simulated = FALSE THEN 1 END) as real_results,
            COUNT(CASE WHEN tr.is_simulated IS NULL THEN 1 END) as unknown_results,
            ROUND(COUNT(CASE WHEN tr.is_simulated = TRUE THEN 1 END) * 100.0 / 
                  NULLIF(COUNT(*), 0), 2) as simulated_percent
        FROM test_results tr
        JOIN hardware_platforms h ON tr.hardware_id = h.hardware_id
        GROUP BY h.hardware_type
        ORDER BY simulated_percent DESC
        """).fetchall()
        
        # Get performance statistics
        performance_stats = conn.execute("""
        SELECT 
            h.hardware_type,
            ROUND(AVG(CASE WHEN pr.is_simulated = TRUE THEN pr.average_latency_ms END), 2) as simulated_avg_latency,
            ROUND(AVG(CASE WHEN pr.is_simulated = FALSE THEN pr.average_latency_ms END), 2) as real_avg_latency,
            ROUND(AVG(CASE WHEN pr.is_simulated = TRUE THEN pr.throughput_items_per_second END), 2) as simulated_avg_throughput,
            ROUND(AVG(CASE WHEN pr.is_simulated = FALSE THEN pr.throughput_items_per_second END), 2) as real_avg_throughput
        FROM performance_results pr
        JOIN hardware_platforms h ON pr.hardware_id = h.hardware_id
        GROUP BY h.hardware_type
        HAVING simulated_avg_latency IS NOT NULL OR real_avg_latency IS NOT NULL
        """).fetchall()
        
        # Get simulation reasons
        simulation_reasons = conn.execute("""
        SELECT 
            simulation_reason,
            COUNT(*) as count
        FROM test_results
        WHERE is_simulated = TRUE
        GROUP BY simulation_reason
        ORDER BY count DESC
        """).fetchall()
        
        # Get error categories for simulated hardware
        error_categories = conn.execute("""
        SELECT 
            error_category,
            COUNT(*) as count
        FROM test_results
        WHERE is_simulated = TRUE
        AND success = FALSE
        GROUP BY error_category
        ORDER BY count DESC
        """).fetchall()
        
        # Compile statistics
        stats = {
            "overall": {
                "total_results": overall_stats[0],
                "simulated_results": overall_stats[1],
                "real_results": overall_stats[2],
                "unknown_results": overall_stats[3],
                "simulated_percent": overall_stats[4]
            },
            "hardware": [
                {
                    "hardware_type": row[0],
                    "total_results": row[1],
                    "simulated_results": row[2],
                    "real_results": row[3],
                    "unknown_results": row[4],
                    "simulated_percent": row[5]
                }
                for row in hardware_stats
            ],
            "performance": [
                {
                    "hardware_type": row[0],
                    "simulated_avg_latency": row[1],
                    "real_avg_latency": row[2],
                    "simulated_avg_throughput": row[3],
                    "real_avg_throughput": row[4]
                }
                for row in performance_stats
            ],
            "simulation_reasons": [
                {
                    "reason": row[0],
                    "count": row[1]
                }
                for row in simulation_reasons
            ],
            "error_categories": [
                {
                    "category": row[0],
                    "count": row[1]
                }
                for row in error_categories
            ]
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting simulation statistics: {str(e)}")
        return {
            "error": f"Failed to get simulation statistics: {str(e)}"
        }

def analyze_simulation_impact(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the impact of simulation on benchmark results"""
    if "error" in stats:
        return {
            "error": stats["error"]
        }
    
    # Compile analysis
    analysis = {
        "summary": "",
        "impact": [],
        "recommendations": []
    }
    
    # Generate summary
    overall = stats["overall"]
    if overall["simulated_results"] == 0:
        analysis["summary"] = "No simulated hardware detected. All results are from real hardware."
    else:
        simulated_percent = overall["simulated_percent"]
        analysis["summary"] = f"{simulated_percent}% of benchmark results are from simulated hardware. "
        
        if simulated_percent > 50:
            analysis["summary"] += "Most results are simulated, which may not reflect real-world performance."
        elif simulated_percent > 20:
            analysis["summary"] += "A significant portion of results are simulated, which may affect overall analysis."
        else:
            analysis["summary"] += "Only a small portion of results are simulated, with minimal impact on overall analysis."
    
    # Check each hardware type
    for hw in stats["hardware"]:
        if hw["simulated_results"] > 0:
            # Hardware has simulated results
            hw_name = hw["hardware_type"]
            sim_percent = hw["simulated_percent"]
            
            impact = {
                "hardware_type": hw_name,
                "simulated_percent": sim_percent,
                "description": ""
            }
            
            if sim_percent == 100:
                impact["description"] = f"All {hw_name} results are simulated, not reflecting actual hardware performance."
                analysis["recommendations"].append(f"Acquire real {hw_name} hardware for accurate benchmarks.")
            elif sim_percent > 80:
                impact["description"] = f"Most {hw_name} results ({sim_percent}%) are simulated, severely limiting accuracy."
                analysis["recommendations"].append(f"Prioritize obtaining real {hw_name} hardware for testing.")
            elif sim_percent > 50:
                impact["description"] = f"Majority of {hw_name} results ({sim_percent}%) are simulated, reducing reliability."
                analysis["recommendations"].append(f"Consider obtaining real {hw_name} hardware for more accurate results.")
            else:
                impact["description"] = f"Some {hw_name} results ({sim_percent}%) are simulated but real hardware data is available."
                analysis["recommendations"].append(f"Flag simulated {hw_name} results in reports or exclude from key analyses.")
            
            analysis["impact"].append(impact)
    
    # Check performance differences
    for perf in stats["performance"]:
        hw_name = perf["hardware_type"]
        sim_latency = perf["simulated_avg_latency"]
        real_latency = perf["real_avg_latency"]
        sim_throughput = perf["simulated_avg_throughput"]
        real_throughput = perf["real_avg_throughput"]
        
        # If we have both simulated and real data, compare them
        if sim_latency is not None and real_latency is not None:
            latency_diff_percent = abs((sim_latency - real_latency) / real_latency * 100)
            
            if latency_diff_percent > 30:
                analysis["impact"].append({
                    "hardware_type": hw_name,
                    "metric": "latency",
                    "description": f"Simulated {hw_name} latency differs by {latency_diff_percent:.1f}% from real hardware."
                })
        
        if sim_throughput is not None and real_throughput is not None:
            throughput_diff_percent = abs((sim_throughput - real_throughput) / real_throughput * 100)
            
            if throughput_diff_percent > 30:
                analysis["impact"].append({
                    "hardware_type": hw_name,
                    "metric": "throughput",
                    "description": f"Simulated {hw_name} throughput differs by {throughput_diff_percent:.1f}% from real hardware."
                })
    
    # Add general recommendations
    if overall["simulated_results"] > 0:
        analysis["recommendations"].append("Clearly label simulated results in benchmark reports.")
        analysis["recommendations"].append("Consider separating simulated and real hardware results in analyses.")
        
        if overall["simulated_percent"] > 50:
            analysis["recommendations"].append("Prioritize obtaining real hardware for more accurate benchmarks.")
            analysis["recommendations"].append("Use caution when making decisions based on current benchmark data.")
    
    # Deduplicate recommendations
    analysis["recommendations"] = list(set(analysis["recommendations"]))
    
    return analysis

def generate_simulation_report(stats: Dict[str, Any], analysis: Dict[str, Any], format: str = "text") -> str:
    """Generate a report on simulation status in the benchmark database"""
    if "error" in stats:
        return f"Error: {stats['error']}"
    
    if format == "json":
        report_data = {
            "statistics": stats,
            "analysis": analysis,
            "generated_at": datetime.datetime.now().isoformat()
        }
        return json.dumps(report_data, indent=2)
    
    elif format == "markdown":
        # Generate markdown report
        lines = []
        lines.append("# Simulation Analysis Report")
        lines.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        lines.append("## Summary")
        lines.append(analysis["summary"])
        lines.append("")
        
        # Overall statistics
        overall = stats["overall"]
        lines.append("## Overall Statistics")
        lines.append(f"- **Total benchmark results:** {overall['total_results']}")
        lines.append(f"- **Simulated results:** {overall['simulated_results']} ({overall['simulated_percent']}%)")
        lines.append(f"- **Real hardware results:** {overall['real_results']} ({100 - overall['simulated_percent']}%)")
        if overall['unknown_results'] > 0:
            lines.append(f"- **Unknown status:** {overall['unknown_results']}")
        lines.append("")
        
        # Hardware-specific statistics
        lines.append("## Hardware-Specific Statistics")
        lines.append("| Hardware Type | Total Results | Simulated | Real | Unknown | Simulated % |")
        lines.append("|--------------|--------------|-----------|------|---------|------------|")
        for hw in stats["hardware"]:
            lines.append(f"| {hw['hardware_type']} | {hw['total_results']} | {hw['simulated_results']} | {hw['real_results']} | {hw['unknown_results']} | {hw['simulated_percent']}% |")
        lines.append("")
        
        # Performance comparison
        if stats["performance"]:
            lines.append("## Performance Comparison (Simulated vs. Real)")
            lines.append("| Hardware Type | Simulated Latency (ms) | Real Latency (ms) | Simulated Throughput | Real Throughput |")
            lines.append("|--------------|------------------------|-------------------|---------------------|-----------------|")
            for perf in stats["performance"]:
                sim_latency = perf["simulated_avg_latency"] or "N/A"
                real_latency = perf["real_avg_latency"] or "N/A"
                sim_throughput = perf["simulated_avg_throughput"] or "N/A"
                real_throughput = perf["real_avg_throughput"] or "N/A"
                lines.append(f"| {perf['hardware_type']} | {sim_latency} | {real_latency} | {sim_throughput} | {real_throughput} |")
            lines.append("")
        
        # Simulation impact
        if analysis["impact"]:
            lines.append("## Simulation Impact")
            for impact in analysis["impact"]:
                if "metric" in impact:
                    lines.append(f"- **{impact['hardware_type']} ({impact['metric']}):** {impact['description']}")
                else:
                    lines.append(f"- **{impact['hardware_type']}:** {impact['description']}")
            lines.append("")
        
        # Recommendations
        if analysis["recommendations"]:
            lines.append("## Recommendations")
            for rec in analysis["recommendations"]:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Simulation reasons
        if stats["simulation_reasons"]:
            lines.append("## Simulation Reasons")
            for reason in stats["simulation_reasons"]:
                lines.append(f"- **{reason['reason']}:** {reason['count']} results")
            lines.append("")
        
        # Error categories
        if stats["error_categories"]:
            lines.append("## Error Categories for Simulated Hardware")
            for cat in stats["error_categories"]:
                lines.append(f"- **{cat['category'] or 'uncategorized'}:** {cat['count']} errors")
            lines.append("")
        
        return "\n".join(lines)
    
    else:  # text format
        # Generate plain text report
        lines = []
        lines.append("SIMULATION ANALYSIS REPORT")
        lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("-" * 80)
        
        lines.append("SUMMARY")
        lines.append(analysis["summary"])
        lines.append("")
        
        # Overall statistics
        overall = stats["overall"]
        lines.append("OVERALL STATISTICS")
        lines.append(f"Total benchmark results: {overall['total_results']}")
        lines.append(f"Simulated results: {overall['simulated_results']} ({overall['simulated_percent']}%)")
        lines.append(f"Real hardware results: {overall['real_results']} ({100 - overall['simulated_percent']}%)")
        if overall['unknown_results'] > 0:
            lines.append(f"Unknown status: {overall['unknown_results']}")
        lines.append("")
        
        # Hardware-specific statistics
        lines.append("HARDWARE-SPECIFIC STATISTICS")
        for hw in stats["hardware"]:
            lines.append(f"{hw['hardware_type']}:")
            lines.append(f"  Total results: {hw['total_results']}")
            lines.append(f"  Simulated: {hw['simulated_results']} ({hw['simulated_percent']}%)")
            lines.append(f"  Real: {hw['real_results']}")
            if hw['unknown_results'] > 0:
                lines.append(f"  Unknown: {hw['unknown_results']}")
        lines.append("")
        
        # Performance comparison
        if stats["performance"]:
            lines.append("PERFORMANCE COMPARISON (SIMULATED VS. REAL)")
            for perf in stats["performance"]:
                lines.append(f"{perf['hardware_type']}:")
                if perf["simulated_avg_latency"] is not None:
                    lines.append(f"  Simulated latency: {perf['simulated_avg_latency']} ms")
                if perf["real_avg_latency"] is not None:
                    lines.append(f"  Real latency: {perf['real_avg_latency']} ms")
                if perf["simulated_avg_throughput"] is not None:
                    lines.append(f"  Simulated throughput: {perf['simulated_avg_throughput']} items/s")
                if perf["real_avg_throughput"] is not None:
                    lines.append(f"  Real throughput: {perf['real_avg_throughput']} items/s")
            lines.append("")
        
        # Simulation impact
        if analysis["impact"]:
            lines.append("SIMULATION IMPACT")
            for impact in analysis["impact"]:
                if "metric" in impact:
                    lines.append(f"- {impact['hardware_type']} ({impact['metric']}): {impact['description']}")
                else:
                    lines.append(f"- {impact['hardware_type']}: {impact['description']}")
            lines.append("")
        
        # Recommendations
        if analysis["recommendations"]:
            lines.append("RECOMMENDATIONS")
            for i, rec in enumerate(analysis["recommendations"], 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        # Simulation reasons
        if stats["simulation_reasons"]:
            lines.append("SIMULATION REASONS")
            for reason in stats["simulation_reasons"]:
                lines.append(f"- {reason['reason']}: {reason['count']} results")
            lines.append("")
        
        # Error categories
        if stats["error_categories"]:
            lines.append("ERROR CATEGORIES FOR SIMULATED HARDWARE")
            for cat in stats["error_categories"]:
                lines.append(f"- {cat['category'] or 'uncategorized'}: {cat['count']} errors")
            lines.append("")
        
        return "\n".join(lines)

def visualize_simulation_status(stats: Dict[str, Any], output_path: Optional[str] = None) -> Optional[str]:
    """Visualize simulation status in the benchmark database"""
    if not VISUALIZATION_AVAILABLE:
        logger.error("Visualization dependencies not available. Install pandas and matplotlib for visualization.")
        return None
    
    if "error" in stats:
        logger.error(f"Cannot visualize due to error: {stats['error']}")
        return None
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Simulation Status Analysis", fontsize=16)
    
    # 1. Overall simulation status pie chart (top left)
    overall = stats["overall"]
    labels = ["Real Hardware", "Simulated", "Unknown"]
    sizes = [overall["real_results"], overall["simulated_results"], overall["unknown_results"]]
    
    # Remove zero values
    non_zero_labels = [label for label, size in zip(labels, sizes) if size > 0]
    non_zero_sizes = [size for size in sizes if size > 0]
    
    axs[0, 0].pie(non_zero_sizes, labels=non_zero_labels, autopct='%1.1f%%', 
               shadow=True, startangle=90)
    axs[0, 0].set_title("Overall Simulation Status")
    
    # 2. Hardware-specific simulation status (top right)
    hardware_df = pd.DataFrame(stats["hardware"])
    hardware_df = hardware_df.sort_values(by="simulated_percent", ascending=False)
    
    # Only show hardware types with at least one result
    hardware_df = hardware_df[hardware_df["total_results"] > 0]
    
    if not hardware_df.empty:
        x = hardware_df["hardware_type"]
        
        bottom = np.zeros(len(x))
        
        # Plot stacked bars
        p1 = axs[0, 1].bar(x, hardware_df["real_results"], label='Real', color='green', alpha=0.7)
        p2 = axs[0, 1].bar(x, hardware_df["simulated_results"], bottom=hardware_df["real_results"], 
                        label='Simulated', color='orange', alpha=0.7)
        
        if hardware_df["unknown_results"].sum() > 0:
            p3 = axs[0, 1].bar(x, hardware_df["unknown_results"], 
                            bottom=hardware_df["real_results"] + hardware_df["simulated_results"],
                            label='Unknown', color='gray', alpha=0.7)
        
        axs[0, 1].set_title("Simulation Status by Hardware Type")
        axs[0, 1].set_xlabel("Hardware Type")
        axs[0, 1].set_ylabel("Number of Results")
        axs[0, 1].legend()
        
        # Rotate x-axis labels for better readability
        plt.setp(axs[0, 1].get_xticklabels(), rotation=45, ha="right")
    else:
        axs[0, 1].text(0.5, 0.5, "No hardware data available", 
                    horizontalalignment='center', verticalalignment='center')
    
    # 3. Performance comparison (bottom left)
    if stats["performance"]:
        perf_df = pd.DataFrame(stats["performance"])
        perf_df = perf_df.dropna(subset=["simulated_avg_latency", "real_avg_latency"], how="all")
        
        if not perf_df.empty:
            # Prepare data for grouped bar chart
            x = np.arange(len(perf_df))
            width = 0.35
            
            # Get non-null values for latency
            sim_latency = perf_df["simulated_avg_latency"].fillna(0)
            real_latency = perf_df["real_avg_latency"].fillna(0)
            
            # Plot bars
            axs[1, 0].bar(x - width/2, sim_latency, width, label='Simulated', color='orange', alpha=0.7)
            axs[1, 0].bar(x + width/2, real_latency, width, label='Real', color='green', alpha=0.7)
            
            axs[1, 0].set_title("Latency Comparison (Simulated vs. Real)")
            axs[1, 0].set_xlabel("Hardware Type")
            axs[1, 0].set_ylabel("Average Latency (ms)")
            axs[1, 0].set_xticks(x)
            axs[1, 0].set_xticklabels(perf_df["hardware_type"])
            axs[1, 0].legend()
            
            # Rotate x-axis labels for better readability
            plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha="right")
        else:
            axs[1, 0].text(0.5, 0.5, "No comparative performance data available", 
                        horizontalalignment='center', verticalalignment='center')
    else:
        axs[1, 0].text(0.5, 0.5, "No performance data available", 
                    horizontalalignment='center', verticalalignment='center')
    
    # 4. Simulation reasons (bottom right)
    if stats["simulation_reasons"]:
        reason_df = pd.DataFrame(stats["simulation_reasons"])
        
        # Simplify reasons for better display
        reason_df["reason"] = reason_df["reason"].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)
        
        # Plot horizontal bar chart
        reason_df = reason_df.sort_values(by="count")
        axs[1, 1].barh(reason_df["reason"], reason_df["count"], color='orange', alpha=0.7)
        axs[1, 1].set_title("Common Simulation Reasons")
        axs[1, 1].set_xlabel("Count")
    else:
        axs[1, 1].text(0.5, 0.5, "No simulation reason data available", 
                    horizontalalignment='center', verticalalignment='center')
    
    # Adjust layout and save if output path provided
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path)
        logger.info(f"Visualization saved to {output_path}")
        return output_path
    else:
        # Create a temporary file for the visualization
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            plt.savefig(temp_file.name)
            logger.info(f"Visualization saved to temporary file: {temp_file.name}")
            return temp_file.name

def main():
    """Main function to analyze simulation status in benchmark results"""
    parser = argparse.ArgumentParser(description="Analyze simulation status in benchmark results")
    parser.add_argument("--db-path", help="Path to the benchmark database")
    parser.add_argument("--output", help="Output file for the report")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text",
                      help="Output format for the report")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization of simulation status")
    parser.add_argument("--viz-output", help="Output file for visualization (PNG)")
    args = parser.parse_args()
    
    # Connect to the database
    try:
        conn = get_db_connection(args.db_path)
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        return 1
    
    # Get simulation statistics
    stats = get_simulation_statistics(conn)
    
    if "error" in stats:
        logger.error(f"Error getting simulation statistics: {stats['error']}")
        return 1
    
    # Analyze the impact of simulation
    analysis = analyze_simulation_impact(stats)
    
    # Generate the report
    report = generate_simulation_report(stats, analysis, args.format)
    
    # Output the report
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        logger.info(f"Report written to {args.output}")
    else:
        print(report)
    
    # Generate visualization if requested
    if args.visualize and VISUALIZATION_AVAILABLE:
        viz_path = visualize_simulation_status(stats, args.viz_output)
        if viz_path:
            logger.info(f"Visualization saved to {viz_path}")
        else:
            logger.error("Failed to generate visualization")
            return 1
    
    return 0

if __name__ == "__main__":
    try:
        import numpy as np
        sys.exit(main())
    except ImportError:
        logger.error("Required package numpy not found. Please install it with pip install numpy.")
        sys.exit(1)