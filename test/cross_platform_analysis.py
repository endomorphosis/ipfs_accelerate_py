#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Platform Mobile Analysis Tool

This module provides comprehensive analysis capabilities for comparing performance
metrics between Android and iOS devices using data from the benchmark database.
It generates reports, visualizations, and optimization recommendations to help
understand cross-platform model performance.

Features:
    - Cross-platform performance comparison (Android vs iOS)
    - Model performance analysis across different mobile hardware
    - Battery impact comparison
    - Thermal behavior analysis
    - Hardware compatibility scoring
    - Optimization recommendations for mobile deployment
    - Report generation in multiple formats
    - Visualization support

Date: April 2025
"""

import os
import sys
import json
import logging
import argparse
import datetime
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Local imports for database access
try:
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI, get_db_connection
    DUCKDB_AVAILABLE = True
except ImportError:
    logger.warning("Could not import benchmark_db_api. Database functionality will be limited.")
    DUCKDB_AVAILABLE = False


class CrossPlatformAnalyzer:
    """
    Analyzes and compares benchmark results across different mobile platforms.
    
    Provides methods for comparing performance metrics between Android and iOS
    devices, identifying optimization opportunities, and generating
    comprehensive reports.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the cross-platform analyzer.
        
        Args:
            db_path: Path to the benchmark database
        """
        self.db_path = db_path
        self.db_api = None
        
        # Initialize database connection
        if DUCKDB_AVAILABLE:
            try:
                self.db_api = BenchmarkDBAPI(db_path)
                logger.info(f"Connected to database: {db_path}")
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
        else:
            logger.error("DuckDB support is required for cross-platform analysis")
    
    def get_platform_comparison(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get cross-platform performance comparison data.
        
        Args:
            model_name: Optional model name to filter by
            
        Returns:
            List of comparison results
        """
        if not self.db_api:
            logger.error("Database connection not available")
            return []
        
        try:
            # Build query
            query = """
            WITH android_metrics AS (
                SELECT
                    m.id AS model_id,
                    m.model_name,
                    m.model_family,
                    AVG(ab.average_latency_ms) AS avg_latency_ms,
                    AVG(ab.throughput_items_per_second) AS avg_throughput,
                    AVG(ab.battery_impact_percent) AS avg_battery_impact,
                    AVG(ab.temperature_max_celsius) AS avg_temperature,
                    COUNT(ab.id) AS benchmark_count
                FROM
                    models m
                JOIN
                    android_benchmark_results ab ON m.id = ab.model_id
                GROUP BY
                    m.id, m.model_name, m.model_family
            ),
            ios_metrics AS (
                SELECT
                    m.id AS model_id,
                    m.model_name,
                    m.model_family,
                    AVG(ib.average_latency_ms) AS avg_latency_ms,
                    AVG(ib.throughput_items_per_second) AS avg_throughput,
                    AVG(ib.battery_impact_percent) AS avg_battery_impact,
                    AVG(ib.temperature_max_celsius) AS avg_temperature,
                    COUNT(ib.id) AS benchmark_count
                FROM
                    models m
                JOIN
                    ios_benchmark_results ib ON m.id = ib.model_id
                GROUP BY
                    m.id, m.model_name, m.model_family
            )
            SELECT
                COALESCE(a.model_id, i.model_id) AS model_id,
                COALESCE(a.model_name, i.model_name) AS model_name,
                COALESCE(a.model_family, i.model_family) AS model_family,
                
                -- Android metrics
                a.avg_latency_ms AS android_latency_ms,
                a.avg_throughput AS android_throughput,
                a.avg_battery_impact AS android_battery_impact,
                a.avg_temperature AS android_temperature,
                a.benchmark_count AS android_benchmark_count,
                
                -- iOS metrics
                i.avg_latency_ms AS ios_latency_ms,
                i.avg_throughput AS ios_throughput,
                i.avg_battery_impact AS ios_battery_impact,
                i.avg_temperature AS ios_temperature,
                i.benchmark_count AS ios_benchmark_count,
                
                -- Comparison metrics
                CASE 
                    WHEN a.avg_throughput > 0 AND i.avg_throughput > 0
                    THEN i.avg_throughput / a.avg_throughput
                    ELSE NULL
                END AS ios_android_throughput_ratio,
                
                CASE 
                    WHEN a.avg_latency_ms > 0 AND i.avg_latency_ms > 0
                    THEN a.avg_latency_ms / i.avg_latency_ms
                    ELSE NULL
                END AS android_ios_latency_ratio,
                
                CASE 
                    WHEN a.avg_battery_impact > 0 AND i.avg_battery_impact > 0
                    THEN i.avg_battery_impact / a.avg_battery_impact
                    ELSE NULL
                END AS ios_android_battery_ratio
                
            FROM
                android_metrics a
            FULL OUTER JOIN
                ios_metrics i ON a.model_id = i.model_id
            WHERE
                (a.benchmark_count > 0 OR i.benchmark_count > 0)
            """
            
            # Add model filter if specified
            if model_name:
                query += f" AND (a.model_name = '{model_name}' OR i.model_name = '{model_name}')"
            
            # Execute query
            results = self.db_api.execute_query(query)
            
            # Convert to list of dictionaries
            comparison = []
            for row in results:
                result = {}
                for i, col in enumerate(results.description):
                    result[col[0]] = row[i]
                comparison.append(result)
            
            return comparison
        
        except Exception as e:
            logger.error(f"Error getting platform comparison: {e}")
            return []
    
    def get_device_comparison(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get device comparison data for Android and iOS.
        
        Returns:
            Dictionary with Android and iOS device lists
        """
        if not self.db_api:
            logger.error("Database connection not available")
            return {"android": [], "ios": []}
        
        try:
            # Get Android devices
            android_query = """
            SELECT
                device_model,
                chipset,
                accelerator,
                COUNT(id) AS benchmark_count,
                AVG(average_latency_ms) AS avg_latency_ms,
                AVG(throughput_items_per_second) AS avg_throughput,
                AVG(battery_impact_percent) AS avg_battery_impact,
                AVG(temperature_max_celsius) AS avg_temperature,
                AVG(CASE WHEN throttling_detected THEN 1 ELSE 0 END) AS throttling_frequency,
                MAX(created_at) AS last_benchmark
            FROM
                android_benchmark_results
            GROUP BY
                device_model, chipset, accelerator
            ORDER BY
                avg_throughput DESC
            """
            
            android_results = self.db_api.execute_query(android_query)
            
            # Get iOS devices
            ios_query = """
            SELECT
                device_model,
                chipset,
                accelerator,
                COUNT(id) AS benchmark_count,
                AVG(average_latency_ms) AS avg_latency_ms,
                AVG(throughput_items_per_second) AS avg_throughput,
                AVG(battery_impact_percent) AS avg_battery_impact,
                AVG(temperature_max_celsius) AS avg_temperature,
                AVG(CASE WHEN throttling_detected THEN 1 ELSE 0 END) AS throttling_frequency,
                MAX(created_at) AS last_benchmark
            FROM
                ios_benchmark_results
            GROUP BY
                device_model, chipset, accelerator
            ORDER BY
                avg_throughput DESC
            """
            
            ios_results = self.db_api.execute_query(ios_query)
            
            # Convert to dictionaries
            android_devices = []
            for row in android_results:
                device = {}
                for i, col in enumerate(android_results.description):
                    device[col[0]] = row[i]
                android_devices.append(device)
            
            ios_devices = []
            for row in ios_results:
                device = {}
                for i, col in enumerate(ios_results.description):
                    device[col[0]] = row[i]
                ios_devices.append(device)
            
            return {
                "android": android_devices,
                "ios": ios_devices
            }
        
        except Exception as e:
            logger.error(f"Error getting device comparison: {e}")
            return {"android": [], "ios": []}
    
    def get_model_performance(self, model_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get model performance data across platforms.
        
        Args:
            model_name: Optional model name to filter by
            
        Returns:
            Dictionary with Android and iOS performance data
        """
        if not self.db_api:
            logger.error("Database connection not available")
            return {"android": [], "ios": []}
        
        try:
            # Build Android query
            android_query = """
            SELECT
                m.model_name,
                m.model_family,
                ab.device_model,
                ab.chipset,
                ab.accelerator,
                ab.average_latency_ms,
                ab.throughput_items_per_second,
                ab.battery_impact_percent,
                ab.temperature_max_celsius,
                ab.throttling_detected,
                ab.memory_peak_mb,
                ab.created_at
            FROM
                models m
            JOIN
                android_benchmark_results ab ON m.id = ab.model_id
            """
            
            # Add model filter if specified
            if model_name:
                android_query += f" WHERE m.model_name = '{model_name}'"
            
            android_query += " ORDER BY ab.created_at DESC"
            
            # Execute Android query
            android_results = self.db_api.execute_query(android_query)
            
            # Build iOS query
            ios_query = """
            SELECT
                m.model_name,
                m.model_family,
                ib.device_model,
                ib.chipset,
                ib.accelerator,
                ib.average_latency_ms,
                ib.throughput_items_per_second,
                ib.battery_impact_percent,
                ib.temperature_max_celsius,
                ib.throttling_detected,
                ib.memory_peak_mb,
                ib.created_at
            FROM
                models m
            JOIN
                ios_benchmark_results ib ON m.id = ib.model_id
            """
            
            # Add model filter if specified
            if model_name:
                ios_query += f" WHERE m.model_name = '{model_name}'"
            
            ios_query += " ORDER BY ib.created_at DESC"
            
            # Execute iOS query
            ios_results = self.db_api.execute_query(ios_query)
            
            # Convert to dictionaries
            android_data = []
            for row in android_results:
                result = {}
                for i, col in enumerate(android_results.description):
                    result[col[0]] = row[i]
                android_data.append(result)
            
            ios_data = []
            for row in ios_results:
                result = {}
                for i, col in enumerate(ios_results.description):
                    result[col[0]] = row[i]
                ios_data.append(result)
            
            return {
                "android": android_data,
                "ios": ios_data
            }
        
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {"android": [], "ios": []}
    
    def analyze_cross_platform_performance(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze cross-platform performance metrics.
        
        Args:
            model_name: Optional model name to filter by
            
        Returns:
            Dictionary with analysis results
        """
        # Get comparison data
        comparisons = self.get_platform_comparison(model_name)
        
        if not comparisons:
            return {"status": "error", "message": "No comparison data available"}
        
        # Calculate platform-specific metrics
        android_metrics = {
            "model_count": sum(1 for c in comparisons if c.get("android_benchmark_count", 0) > 0),
            "avg_latency": np.mean([c.get("android_latency_ms", 0) for c in comparisons if c.get("android_latency_ms")]),
            "avg_throughput": np.mean([c.get("android_throughput", 0) for c in comparisons if c.get("android_throughput")]),
            "avg_battery_impact": np.mean([c.get("android_battery_impact", 0) for c in comparisons if c.get("android_battery_impact")]),
            "avg_temperature": np.mean([c.get("android_temperature", 0) for c in comparisons if c.get("android_temperature")])
        }
        
        ios_metrics = {
            "model_count": sum(1 for c in comparisons if c.get("ios_benchmark_count", 0) > 0),
            "avg_latency": np.mean([c.get("ios_latency_ms", 0) for c in comparisons if c.get("ios_latency_ms")]),
            "avg_throughput": np.mean([c.get("ios_throughput", 0) for c in comparisons if c.get("ios_throughput")]),
            "avg_battery_impact": np.mean([c.get("ios_battery_impact", 0) for c in comparisons if c.get("ios_battery_impact")]),
            "avg_temperature": np.mean([c.get("ios_temperature", 0) for c in comparisons if c.get("ios_temperature")])
        }
        
        # Calculate cross-platform metrics
        cross_platform_metrics = {
            "model_count": sum(1 for c in comparisons if c.get("android_benchmark_count", 0) > 0 and c.get("ios_benchmark_count", 0) > 0),
            "avg_throughput_ratio": np.mean([c.get("ios_android_throughput_ratio", 0) for c in comparisons if c.get("ios_android_throughput_ratio")]),
            "avg_latency_ratio": np.mean([c.get("android_ios_latency_ratio", 0) for c in comparisons if c.get("android_ios_latency_ratio")]),
            "avg_battery_ratio": np.mean([c.get("ios_android_battery_ratio", 0) for c in comparisons if c.get("ios_android_battery_ratio")])
        }
        
        # Calculate model-specific metrics
        model_metrics = []
        
        for comparison in comparisons:
            model_name = comparison.get("model_name", "Unknown")
            model_family = comparison.get("model_family", "Unknown")
            
            has_android = comparison.get("android_benchmark_count", 0) > 0
            has_ios = comparison.get("ios_benchmark_count", 0) > 0
            
            if has_android and has_ios:
                throughput_ratio = comparison.get("ios_android_throughput_ratio", 1.0)
                latency_ratio = comparison.get("android_ios_latency_ratio", 1.0)
                battery_ratio = comparison.get("ios_android_battery_ratio", 1.0)
                
                # Calculate performance score (higher is better for iOS)
                performance_score = np.mean([
                    throughput_ratio if throughput_ratio else 1.0,
                    latency_ratio if latency_ratio else 1.0
                ])
                
                # Calculate efficiency score (lower is better for iOS)
                efficiency_score = battery_ratio if battery_ratio else 1.0
                
                # Determine platform recommendation
                if performance_score > 1.2:
                    platform_recommendation = "iOS"
                elif performance_score < 0.8:
                    platform_recommendation = "Android"
                else:
                    # If performance is similar, choose based on battery efficiency
                    platform_recommendation = "Android" if efficiency_score > 1.1 else "iOS" if efficiency_score < 0.9 else "Either"
                
                model_metrics.append({
                    "model_name": model_name,
                    "model_family": model_family,
                    "android_throughput": comparison.get("android_throughput", 0),
                    "ios_throughput": comparison.get("ios_throughput", 0),
                    "throughput_ratio": throughput_ratio,
                    "android_latency": comparison.get("android_latency_ms", 0),
                    "ios_latency": comparison.get("ios_latency_ms", 0),
                    "latency_ratio": latency_ratio,
                    "android_battery": comparison.get("android_battery_impact", 0),
                    "ios_battery": comparison.get("ios_battery_impact", 0),
                    "battery_ratio": battery_ratio,
                    "performance_score": performance_score,
                    "efficiency_score": efficiency_score,
                    "platform_recommendation": platform_recommendation
                })
        
        # Sort by performance score (descending)
        model_metrics.sort(key=lambda x: x.get("performance_score", 0), reverse=True)
        
        # Generate optimization recommendations
        optimization_recommendations = self._generate_recommendations(comparisons)
        
        # Prepare result
        result = {
            "status": "success",
            "android_metrics": android_metrics,
            "ios_metrics": ios_metrics,
            "cross_platform_metrics": cross_platform_metrics,
            "model_metrics": model_metrics,
            "recommendations": optimization_recommendations,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return result
    
    def _generate_recommendations(self, comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate optimization recommendations based on comparison data.
        
        Args:
            comparisons: List of model comparison data
            
        Returns:
            Dictionary with recommendations
        """
        # Count models with different performance characteristics
        ios_faster_count = sum(1 for c in comparisons 
                              if c.get("ios_android_throughput_ratio", 1) > 1.2 
                              and c.get("android_benchmark_count", 0) > 0 
                              and c.get("ios_benchmark_count", 0) > 0)
        
        android_faster_count = sum(1 for c in comparisons 
                                  if c.get("ios_android_throughput_ratio", 1) < 0.8 
                                  and c.get("android_benchmark_count", 0) > 0 
                                  and c.get("ios_benchmark_count", 0) > 0)
        
        similar_count = sum(1 for c in comparisons 
                           if 0.8 <= c.get("ios_android_throughput_ratio", 1) <= 1.2 
                           and c.get("android_benchmark_count", 0) > 0 
                           and c.get("ios_benchmark_count", 0) > 0)
        
        # Identify model families with platform preferences
        ios_preferred_families = {}
        android_preferred_families = {}
        
        for comparison in comparisons:
            if (comparison.get("ios_android_throughput_ratio", 1) > 1.2 
                and comparison.get("android_benchmark_count", 0) > 0 
                and comparison.get("ios_benchmark_count", 0) > 0):
                
                family = comparison.get("model_family", "Unknown")
                ios_preferred_families[family] = ios_preferred_families.get(family, 0) + 1
            
            elif (comparison.get("ios_android_throughput_ratio", 1) < 0.8 
                  and comparison.get("android_benchmark_count", 0) > 0 
                  and comparison.get("ios_benchmark_count", 0) > 0):
                
                family = comparison.get("model_family", "Unknown")
                android_preferred_families[family] = android_preferred_families.get(family, 0) + 1
        
        # Generate model family recommendations
        family_recommendations = []
        
        for family, count in ios_preferred_families.items():
            if count >= 2:  # At least 2 models in the family prefer iOS
                family_recommendations.append({
                    "family": family,
                    "platform": "iOS",
                    "reason": f"{count} models in this family perform better on iOS"
                })
        
        for family, count in android_preferred_families.items():
            if count >= 2:  # At least 2 models in the family prefer Android
                family_recommendations.append({
                    "family": family,
                    "platform": "Android",
                    "reason": f"{count} models in this family perform better on Android"
                })
        
        # Generate battery impact recommendations
        battery_recommendations = []
        
        for comparison in comparisons:
            if (comparison.get("android_benchmark_count", 0) > 0 
                and comparison.get("ios_benchmark_count", 0) > 0):
                
                android_battery = comparison.get("android_battery_impact", 0)
                ios_battery = comparison.get("ios_battery_impact", 0)
                
                if android_battery > 5 and ios_battery > 5:
                    battery_recommendations.append({
                        "model": comparison.get("model_name", "Unknown"),
                        "issue": "High battery impact on both platforms",
                        "recommendation": "Consider model optimization or smaller variant"
                    })
                elif android_battery > 5 and ios_battery < 3:
                    battery_recommendations.append({
                        "model": comparison.get("model_name", "Unknown"),
                        "issue": "High battery impact on Android only",
                        "recommendation": "Prefer iOS deployment or optimize Android implementation"
                    })
                elif ios_battery > 5 and android_battery < 3:
                    battery_recommendations.append({
                        "model": comparison.get("model_name", "Unknown"),
                        "issue": "High battery impact on iOS only",
                        "recommendation": "Prefer Android deployment or optimize iOS implementation"
                    })
        
        # Generate thermal impact recommendations
        thermal_recommendations = []
        
        for comparison in comparisons:
            if (comparison.get("android_benchmark_count", 0) > 0 
                and comparison.get("ios_benchmark_count", 0) > 0):
                
                android_temp = comparison.get("android_temperature", 0)
                ios_temp = comparison.get("ios_temperature", 0)
                
                if android_temp > 40 and ios_temp > 40:
                    thermal_recommendations.append({
                        "model": comparison.get("model_name", "Unknown"),
                        "issue": "High thermal impact on both platforms",
                        "recommendation": "Implement cooling periods and throttling detection"
                    })
                elif android_temp > 45 and ios_temp < 35:
                    thermal_recommendations.append({
                        "model": comparison.get("model_name", "Unknown"),
                        "issue": "High thermal impact on Android only",
                        "recommendation": "Prefer iOS deployment or implement Android cooling"
                    })
                elif ios_temp > 45 and android_temp < 35:
                    thermal_recommendations.append({
                        "model": comparison.get("model_name", "Unknown"),
                        "issue": "High thermal impact on iOS only",
                        "recommendation": "Prefer Android deployment or implement iOS cooling"
                    })
        
        # Generate general recommendations
        general_recommendations = []
        
        if ios_faster_count > android_faster_count and ios_faster_count > similar_count:
            general_recommendations.append({
                "type": "Performance",
                "recommendation": "Most models perform better on iOS. Prioritize iOS deployment for performance-critical applications."
            })
        elif android_faster_count > ios_faster_count and android_faster_count > similar_count:
            general_recommendations.append({
                "type": "Performance",
                "recommendation": "Most models perform better on Android. Prioritize Android deployment for performance-critical applications."
            })
        else:
            general_recommendations.append({
                "type": "Performance",
                "recommendation": "Performance is similar across platforms. Choose based on specific model needs or user base."
            })
        
        # Add quantization recommendation
        general_recommendations.append({
            "type": "Optimization",
            "recommendation": "Apply INT8 quantization for all mobile deployments to improve performance and reduce battery impact."
        })
        
        # Add batch size recommendation
        general_recommendations.append({
            "type": "Configuration",
            "recommendation": "Use small batch sizes (1-4) on mobile devices to reduce memory usage and improve responsiveness."
        })
        
        # Add neural engine recommendation
        general_recommendations.append({
            "type": "Hardware",
            "recommendation": "For iOS, prioritize devices with Neural Engine (iPhone 11+ or iPad Pro 3rd gen+) for optimal performance."
        })
        
        # Compile all recommendations
        recommendations = {
            "general": general_recommendations,
            "model_family": family_recommendations,
            "battery": battery_recommendations,
            "thermal": thermal_recommendations
        }
        
        return recommendations
    
    def generate_visualization(self, analysis_result: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate performance comparison visualizations.
        
        Args:
            analysis_result: Result from analyze_cross_platform_performance
            output_path: Optional path to save the visualization
            
        Returns:
            Path to the saved visualization
        """
        if not analysis_result or analysis_result.get("status") != "success":
            logger.error("Invalid analysis result")
            return ""
        
        try:
            # Create output directory if needed
            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            else:
                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    output_path = f.name
            
            # Get model metrics
            model_metrics = analysis_result.get("model_metrics", [])
            
            if not model_metrics:
                logger.error("No model metrics available for visualization")
                return ""
            
            # Prepare data
            model_names = [m.get("model_name", "Unknown") for m in model_metrics]
            android_throughput = [m.get("android_throughput", 0) for m in model_metrics]
            ios_throughput = [m.get("ios_throughput", 0) for m in model_metrics]
            
            # Create figure and axes
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot throughput comparison
            x = np.arange(len(model_names))
            width = 0.35
            
            ax1.bar(x - width/2, android_throughput, width, label='Android')
            ax1.bar(x + width/2, ios_throughput, width, label='iOS')
            
            ax1.set_title('Throughput Comparison (items/second)')
            ax1.set_ylabel('Throughput (items/s)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names, rotation=45, ha='right')
            ax1.legend()
            
            # Plot throughput ratio (iOS/Android)
            throughput_ratios = [m.get("throughput_ratio", 1) for m in model_metrics]
            colors = ['green' if r > 1 else 'red' for r in throughput_ratios]
            
            ax2.bar(x, throughput_ratios, color=colors)
            ax2.axhline(y=1, color='black', linestyle='--')
            
            ax2.set_title('iOS/Android Throughput Ratio')
            ax2.set_ylabel('Ratio')
            ax2.set_xticks(x)
            ax2.set_xticklabels(model_names, rotation=45, ha='right')
            
            # Add explanatory text
            ax2.text(0.02, 0.95, 'Green = iOS faster, Red = Android faster', 
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Visualization saved to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return ""
    
    def generate_comparison_report(self, model_name: Optional[str] = None, format: str = "markdown") -> Dict[str, str]:
        """
        Generate a comprehensive cross-platform comparison report.
        
        Args:
            model_name: Optional model name to filter by
            format: Report format (markdown, html)
            
        Returns:
            Dictionary with report content and path
        """
        # Analyze performance
        analysis = self.analyze_cross_platform_performance(model_name)
        
        if analysis.get("status") != "success":
            return {"status": "error", "message": "Analysis failed", "content": "", "path": ""}
        
        # Generate visualization
        viz_path = self.generate_visualization(analysis)
        
        # Generate report
        if format == "html":
            content = self._generate_html_report(analysis, viz_path)
            report_ext = "html"
        else:
            content = self._generate_markdown_report(analysis, viz_path)
            report_ext = "md"
        
        # Save report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = f"_{model_name}" if model_name else ""
        report_path = f"cross_platform_report{model_suffix}_{timestamp}.{report_ext}"
        
        with open(report_path, "w") as f:
            f.write(content)
        
        return {
            "status": "success",
            "content": content,
            "path": report_path,
            "visualization": viz_path
        }
    
    def _generate_markdown_report(self, analysis: Dict[str, Any], viz_path: str) -> str:
        """
        Generate a markdown report from analysis results.
        
        Args:
            analysis: Analysis results
            viz_path: Path to visualization file
            
        Returns:
            Markdown report
        """
        # Extract metrics
        android_metrics = analysis.get("android_metrics", {})
        ios_metrics = analysis.get("ios_metrics", {})
        cross_metrics = analysis.get("cross_platform_metrics", {})
        model_metrics = analysis.get("model_metrics", [])
        recommendations = analysis.get("recommendations", {})
        
        # Generate report
        report = "# Cross-Platform Mobile Performance Analysis\n\n"
        report += f"Generated: {datetime.datetime.now().isoformat()}\n\n"
        
        # Summary
        report += "## Summary\n\n"
        
        report += "### Platform Support\n\n"
        report += f"- **Android Models**: {android_metrics.get('model_count', 0)}\n"
        report += f"- **iOS Models**: {ios_metrics.get('model_count', 0)}\n"
        report += f"- **Cross-Platform Models**: {cross_metrics.get('model_count', 0)}\n\n"
        
        report += "### Performance Overview\n\n"
        report += "| Metric | Android | iOS | Ratio (iOS/Android) |\n"
        report += "|--------|---------|-----|--------------------|\n"
        
        # Add performance metrics
        report += f"| Throughput (items/s) | {android_metrics.get('avg_throughput', 0):.2f} | {ios_metrics.get('avg_throughput', 0):.2f} | {cross_metrics.get('avg_throughput_ratio', 0):.2f}x |\n"
        report += f"| Latency (ms) | {android_metrics.get('avg_latency', 0):.2f} | {ios_metrics.get('avg_latency', 0):.2f} | {1/cross_metrics.get('avg_latency_ratio', 1):.2f}x |\n"
        report += f"| Battery Impact (%) | {android_metrics.get('avg_battery_impact', 0):.2f} | {ios_metrics.get('avg_battery_impact', 0):.2f} | {cross_metrics.get('avg_battery_ratio', 0):.2f}x |\n"
        report += f"| Temperature (°C) | {android_metrics.get('avg_temperature', 0):.2f} | {ios_metrics.get('avg_temperature', 0):.2f} | - |\n\n"
        
        # Platform recommendation
        if cross_metrics.get('avg_throughput_ratio', 1) > 1.2:
            report += "**Overall Performance Recommendation**: iOS platform typically provides better performance\n\n"
        elif cross_metrics.get('avg_throughput_ratio', 1) < 0.8:
            report += "**Overall Performance Recommendation**: Android platform typically provides better performance\n\n"
        else:
            report += "**Overall Performance Recommendation**: Both platforms provide similar performance\n\n"
        
        # Model comparison
        report += "## Model-Specific Comparison\n\n"
        report += "| Model | Android Throughput | iOS Throughput | Ratio (iOS/Android) | Recommended Platform |\n"
        report += "|-------|-------------------|----------------|---------------------|----------------------|\n"
        
        for metric in model_metrics:
            model_name = metric.get("model_name", "Unknown")
            android_throughput = metric.get("android_throughput", 0)
            ios_throughput = metric.get("ios_throughput", 0)
            throughput_ratio = metric.get("throughput_ratio", 0)
            platform = metric.get("platform_recommendation", "Either")
            
            report += f"| {model_name} | {android_throughput:.2f} | {ios_throughput:.2f} | {throughput_ratio:.2f}x | {platform} |\n"
        
        # Optimization recommendations
        report += "\n## Optimization Recommendations\n\n"
        
        # General recommendations
        report += "### General Recommendations\n\n"
        for rec in recommendations.get("general", []):
            report += f"- **{rec.get('type', '')}**: {rec.get('recommendation', '')}\n"
        
        # Model family recommendations
        if recommendations.get("model_family"):
            report += "\n### Model Family Recommendations\n\n"
            for rec in recommendations.get("model_family", []):
                report += f"- **{rec.get('family', '')}**: Prefer {rec.get('platform', '')} - {rec.get('reason', '')}\n"
        
        # Battery recommendations
        if recommendations.get("battery"):
            report += "\n### Battery Optimization\n\n"
            for rec in recommendations.get("battery", []):
                report += f"- **{rec.get('model', '')}**: {rec.get('issue', '')} - {rec.get('recommendation', '')}\n"
        
        # Thermal recommendations
        if recommendations.get("thermal"):
            report += "\n### Thermal Management\n\n"
            for rec in recommendations.get("thermal", []):
                report += f"- **{rec.get('model', '')}**: {rec.get('issue', '')} - {rec.get('recommendation', '')}\n"
        
        # Implementation guidelines
        report += "\n## Implementation Guidelines\n\n"
        
        report += "### Android\n\n"
        report += "- Use TFLite with NNAPI for hardware acceleration\n"
        report += "- Apply INT8 quantization for better performance\n"
        report += "- Target API level 30+ for best NNAPI support\n"
        report += "- Implement thermal throttling detection\n"
        report += "- Consider DSP acceleration for audio models\n\n"
        
        report += "### iOS\n\n"
        report += "- Use Core ML with Neural Engine support\n"
        report += "- Target iOS 14+ for best Core ML performance\n"
        report += "- Ensure Metal Performance Shaders fallback\n"
        report += "- Implement power mode detection\n"
        report += "- Use Core ML Converters for ONNX models\n\n"
        
        # Add visualization reference
        if viz_path:
            report += "\n## Performance Visualization\n\n"
            report += f"See attached visualization: {os.path.basename(viz_path)}\n\n"
        
        return report
    
    def _generate_html_report(self, analysis: Dict[str, Any], viz_path: str) -> str:
        """
        Generate an HTML report from analysis results.
        
        Args:
            analysis: Analysis results
            viz_path: Path to visualization file
            
        Returns:
            HTML report
        """
        # Extract metrics
        android_metrics = analysis.get("android_metrics", {})
        ios_metrics = analysis.get("ios_metrics", {})
        cross_metrics = analysis.get("cross_platform_metrics", {})
        model_metrics = analysis.get("model_metrics", [])
        recommendations = analysis.get("recommendations", {})
        
        # Generate HTML
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Cross-Platform Mobile Performance Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #333;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .ios-faster {
            color: green;
        }
        .android-faster {
            color: red;
        }
        .recommendation {
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 4px solid #4CAF50;
            margin-bottom: 20px;
        }
        .visualization {
            max-width: 100%;
            height: auto;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>Cross-Platform Mobile Performance Analysis</h1>
    <p>Generated: """ + datetime.datetime.now().isoformat() + """</p>
    
    <h2>Summary</h2>
    
    <h3>Platform Support</h3>
    <ul>
        <li><strong>Android Models</strong>: """ + str(android_metrics.get('model_count', 0)) + """</li>
        <li><strong>iOS Models</strong>: """ + str(ios_metrics.get('model_count', 0)) + """</li>
        <li><strong>Cross-Platform Models</strong>: """ + str(cross_metrics.get('model_count', 0)) + """</li>
    </ul>
    
    <h3>Performance Overview</h3>
    <table>
        <tr>
            <th>Metric</th>
            <th>Android</th>
            <th>iOS</th>
            <th>Ratio (iOS/Android)</th>
        </tr>
        <tr>
            <td>Throughput (items/s)</td>
            <td>""" + f"{android_metrics.get('avg_throughput', 0):.2f}" + """</td>
            <td>""" + f"{ios_metrics.get('avg_throughput', 0):.2f}" + """</td>
            <td class=""" + ("ios-faster" if cross_metrics.get('avg_throughput_ratio', 1) > 1 else "android-faster") + """>""" + f"{cross_metrics.get('avg_throughput_ratio', 0):.2f}x" + """</td>
        </tr>
        <tr>
            <td>Latency (ms)</td>
            <td>""" + f"{android_metrics.get('avg_latency', 0):.2f}" + """</td>
            <td>""" + f"{ios_metrics.get('avg_latency', 0):.2f}" + """</td>
            <td class=""" + ("android-faster" if cross_metrics.get('avg_latency_ratio', 1) > 1 else "ios-faster") + """>""" + f"{1/cross_metrics.get('avg_latency_ratio', 1):.2f}x" + """</td>
        </tr>
        <tr>
            <td>Battery Impact (%)</td>
            <td>""" + f"{android_metrics.get('avg_battery_impact', 0):.2f}" + """</td>
            <td>""" + f"{ios_metrics.get('avg_battery_impact', 0):.2f}" + """</td>
            <td class=""" + ("android-faster" if cross_metrics.get('avg_battery_ratio', 1) < 1 else "ios-faster") + """>""" + f"{cross_metrics.get('avg_battery_ratio', 0):.2f}x" + """</td>
        </tr>
        <tr>
            <td>Temperature (°C)</td>
            <td>""" + f"{android_metrics.get('avg_temperature', 0):.2f}" + """</td>
            <td>""" + f"{ios_metrics.get('avg_temperature', 0):.2f}" + """</td>
            <td>-</td>
        </tr>
    </table>
    """
        
        # Platform recommendation
        if cross_metrics.get('avg_throughput_ratio', 1) > 1.2:
            html += """<div class="recommendation">
        <p><strong>Overall Performance Recommendation</strong>: iOS platform typically provides better performance</p>
    </div>"""
        elif cross_metrics.get('avg_throughput_ratio', 1) < 0.8:
            html += """<div class="recommendation">
        <p><strong>Overall Performance Recommendation</strong>: Android platform typically provides better performance</p>
    </div>"""
        else:
            html += """<div class="recommendation">
        <p><strong>Overall Performance Recommendation</strong>: Both platforms provide similar performance</p>
    </div>"""
        
        # Model comparison
        html += """
    <h2>Model-Specific Comparison</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Android Throughput</th>
            <th>iOS Throughput</th>
            <th>Ratio (iOS/Android)</th>
            <th>Recommended Platform</th>
        </tr>
    """
        
        for metric in model_metrics:
            model_name = metric.get("model_name", "Unknown")
            android_throughput = metric.get("android_throughput", 0)
            ios_throughput = metric.get("ios_throughput", 0)
            throughput_ratio = metric.get("throughput_ratio", 0)
            platform = metric.get("platform_recommendation", "Either")
            
            ratio_class = "ios-faster" if throughput_ratio > 1 else "android-faster"
            
            html += f"""
        <tr>
            <td>{model_name}</td>
            <td>{android_throughput:.2f}</td>
            <td>{ios_throughput:.2f}</td>
            <td class="{ratio_class}">{throughput_ratio:.2f}x</td>
            <td>{platform}</td>
        </tr>"""
        
        html += """
    </table>
    
    <h2>Optimization Recommendations</h2>
    
    <h3>General Recommendations</h3>
    <ul>
    """
        
        # General recommendations
        for rec in recommendations.get("general", []):
            html += f"""    <li><strong>{rec.get('type', '')}</strong>: {rec.get('recommendation', '')}</li>
    """
        
        html += """</ul>"""
        
        # Model family recommendations
        if recommendations.get("model_family"):
            html += """
    <h3>Model Family Recommendations</h3>
    <ul>
    """
            for rec in recommendations.get("model_family", []):
                html += f"""    <li><strong>{rec.get('family', '')}</strong>: Prefer {rec.get('platform', '')} - {rec.get('reason', '')}</li>
    """
            html += """</ul>"""
        
        # Battery recommendations
        if recommendations.get("battery"):
            html += """
    <h3>Battery Optimization</h3>
    <ul>
    """
            for rec in recommendations.get("battery", []):
                html += f"""    <li><strong>{rec.get('model', '')}</strong>: {rec.get('issue', '')} - {rec.get('recommendation', '')}</li>
    """
            html += """</ul>"""
        
        # Thermal recommendations
        if recommendations.get("thermal"):
            html += """
    <h3>Thermal Management</h3>
    <ul>
    """
            for rec in recommendations.get("thermal", []):
                html += f"""    <li><strong>{rec.get('model', '')}</strong>: {rec.get('issue', '')} - {rec.get('recommendation', '')}</li>
    """
            html += """</ul>"""
        
        # Implementation guidelines
        html += """
    <h2>Implementation Guidelines</h2>
    
    <h3>Android</h3>
    <ul>
        <li>Use TFLite with NNAPI for hardware acceleration</li>
        <li>Apply INT8 quantization for better performance</li>
        <li>Target API level 30+ for best NNAPI support</li>
        <li>Implement thermal throttling detection</li>
        <li>Consider DSP acceleration for audio models</li>
    </ul>
    
    <h3>iOS</h3>
    <ul>
        <li>Use Core ML with Neural Engine support</li>
        <li>Target iOS 14+ for best Core ML performance</li>
        <li>Ensure Metal Performance Shaders fallback</li>
        <li>Implement power mode detection</li>
        <li>Use Core ML Converters for ONNX models</li>
    </ul>
    """
        
        # Add visualization
        if viz_path:
            viz_base64 = self._get_image_base64(viz_path)
            if viz_base64:
                html += f"""
    <h2>Performance Visualization</h2>
    <img src="data:image/png;base64,{viz_base64}" alt="Performance Comparison" class="visualization">
    """
        
        html += """
</body>
</html>
"""
        
        return html
    
    def _get_image_base64(self, image_path: str) -> str:
        """
        Convert image to base64 for embedding in HTML.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image
        """
        import base64
        
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return ""


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Cross-Platform Mobile Analysis Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Performance comparison command
    compare_parser = subparsers.add_parser("compare", help="Compare performance across platforms")
    compare_parser.add_argument("--db-path", required=True, help="Path to benchmark database")
    compare_parser.add_argument("--model", help="Optional model name to filter by")
    compare_parser.add_argument("--output", help="Path to save the report")
    compare_parser.add_argument("--format", choices=["markdown", "html"], default="markdown", help="Report format")
    
    # Generate visualization command
    visualize_parser = subparsers.add_parser("visualize", help="Generate performance visualization")
    visualize_parser.add_argument("--db-path", required=True, help="Path to benchmark database")
    visualize_parser.add_argument("--model", help="Optional model name to filter by")
    visualize_parser.add_argument("--output", required=True, help="Path to save the visualization")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze cross-platform performance")
    analyze_parser.add_argument("--db-path", required=True, help="Path to benchmark database")
    analyze_parser.add_argument("--model", help="Optional model name to filter by")
    analyze_parser.add_argument("--output", help="Path to save the analysis result")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        analyzer = CrossPlatformAnalyzer(args.db_path)
        
        if args.command == "compare":
            report = analyzer.generate_comparison_report(args.model, args.format)
            
            if report.get("status") == "success":
                if args.output:
                    with open(args.output, "w") as f:
                        f.write(report.get("content", ""))
                    print(f"Report saved to: {args.output}")
                else:
                    print(report.get("content", ""))
                
                print(f"Visualization saved to: {report.get('visualization', 'None')}")
                return 0
            else:
                print(f"Error: {report.get('message', 'Failed to generate report')}")
                return 1
        
        elif args.command == "visualize":
            analysis = analyzer.analyze_cross_platform_performance(args.model)
            
            if analysis.get("status") == "success":
                viz_path = analyzer.generate_visualization(analysis, args.output)
                
                if viz_path:
                    print(f"Visualization saved to: {viz_path}")
                    return 0
                else:
                    print("Failed to generate visualization")
                    return 1
            else:
                print(f"Error: {analysis.get('message', 'Analysis failed')}")
                return 1
        
        elif args.command == "analyze":
            analysis = analyzer.analyze_cross_platform_performance(args.model)
            
            if analysis.get("status") == "success":
                if args.output:
                    with open(args.output, "w") as f:
                        json.dump(analysis, f, indent=2)
                    print(f"Analysis saved to: {args.output}")
                else:
                    print(json.dumps(analysis, indent=2))
                
                return 0
            else:
                print(f"Error: {analysis.get('message', 'Analysis failed')}")
                return 1
        
        else:
            parser.print_help()
            return 1
    
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())