#!/usr/bin/env python3
"""
Validation Dashboard for the Benchmark Validation System.

This module provides a comprehensive dashboard for visualizing and analyzing
benchmark validation results. It integrates with the Monitoring Dashboard
system for distributed testing and the Advanced Visualization System.

The ValidationDashboard component offers the following key features:
1. Interactive dashboards with multiple validation visualizations
2. Comparison dashboards for analyzing multiple sets of validation results
3. Integration with the Distributed Testing Monitoring Dashboard
4. Export capabilities to multiple formats (HTML, Markdown, JSON)
5. Embedding capabilities for integration with other systems
6. Dashboard management (listing, updating, deleting)
7. Basic HTML dashboards with fallback when advanced visualization is unavailable

Usage:
    from duckdb_api.benchmark_validation.visualization import ValidationDashboard
    
    # Create dashboard instance
    dashboard = ValidationDashboard(config={...})
    
    # Create a dashboard
    dashboard_path = dashboard.create_dashboard(
        validation_results=validation_results,
        dashboard_name="my_dashboard"
    )
    
    # Create a comparison dashboard
    comparison_path = dashboard.create_comparison_dashboard(
        validation_results_sets={
            "baseline": baseline_results,
            "experiment": experiment_results
        }
    )
    
    # Register with monitoring dashboard
    dashboard.register_with_monitoring_dashboard(
        dashboard_name="my_dashboard"
    )

Examples:
    See duckdb_api.benchmark_validation.examples.dashboard_example for a comprehensive
    demonstration of dashboard functionality.
"""

import os
import json
import logging
import datetime
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmark_validation_dashboard")

# Import base classes
from duckdb_api.benchmark_validation.core.base import (
    ValidationLevel,
    ValidationStatus,
    BenchmarkResult,
    ValidationResult
)

# Import reporter for visualization capabilities
from duckdb_api.benchmark_validation.visualization.reporter import ValidationReporterImpl

# Import visualization dependencies if available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("Pandas not available. Some dashboard features will be limited.")
    PANDAS_AVAILABLE = False

# Import Advanced Visualization System components if available
try:
    from duckdb_api.visualization.advanced_visualization.viz_customizable_dashboard import (
        CustomizableDashboard
    )
    from duckdb_api.visualization.advanced_visualization.export_integration import (
        ExportIntegration
    )
    ADVANCED_VIZ_AVAILABLE = True
except ImportError:
    logger.warning("Advanced visualization components not available. Falling back to basic visualizations.")
    ADVANCED_VIZ_AVAILABLE = False

# Import Monitoring Dashboard integration if available
try:
    from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_visualization_integration import (
        VisualizationDashboardIntegration
    )
    MONITORING_DASHBOARD_AVAILABLE = True
except ImportError:
    logger.warning("Monitoring Dashboard integration not available.")
    MONITORING_DASHBOARD_AVAILABLE = False


class ValidationDashboard:
    """
    Dashboard component for visualizing and analyzing benchmark validation results.
    
    This class provides a comprehensive dashboard for visualization and analysis of
    benchmark validation results with integration capabilities for the monitoring dashboard.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the validation dashboard.
        
        Args:
            config: Configuration options for the dashboard
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "dashboard_name": "benchmark_validation_dashboard",
            "dashboard_title": "Benchmark Validation Dashboard",
            "dashboard_description": "Comprehensive visualization of benchmark validation results",
            "output_directory": "output",
            "dashboard_directory": "dashboards",
            "monitoring_integration": True,
            "theme": "light",
            "auto_refresh": True,
            "refresh_interval": 300,  # 5 minutes
            "max_results": 1000,
            "default_view": "summary"
        }
        
        # Apply default config values if not specified
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Set up directories
        self.output_dir = self.config["output_directory"]
        self.dashboard_dir = os.path.join(self.output_dir, self.config["dashboard_directory"])
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dashboard_dir, exist_ok=True)
        
        # Initialize reporter for visualization capabilities
        self.reporter = ValidationReporterImpl(config={
            "output_directory": self.output_dir,
            "theme": self.config["theme"],
            "include_visualizations": True
        })
        
        # Initialize dashboard components
        self.dashboard_instance = None
        self.monitoring_integration = None
        self.validation_results_cache = {}
        self.dashboard_components = {}
        self.dashboard_config = {
            "name": self.config["dashboard_name"],
            "title": self.config["dashboard_title"],
            "description": self.config["dashboard_description"],
            "theme": self.config["theme"],
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "components": []
        }
        
        # Initialize visualization system if available
        if ADVANCED_VIZ_AVAILABLE:
            self._initialize_advanced_visualization()
        
        # Initialize monitoring dashboard integration if available and enabled
        if MONITORING_DASHBOARD_AVAILABLE and self.config["monitoring_integration"]:
            self._initialize_monitoring_integration()
        
        logger.info("Validation Dashboard initialized")
    
    def _initialize_advanced_visualization(self):
        """Initialize the Advanced Visualization System."""
        try:
            # Initialize customizable dashboard
            self.dashboard_instance = CustomizableDashboard(
                output_dir=self.dashboard_dir,
                theme=self.config["theme"]
            )
            
            logger.info("Advanced Visualization System initialized")
        except Exception as e:
            logger.error(f"Error initializing Advanced Visualization System: {e}")
            self.dashboard_instance = None
    
    def _initialize_monitoring_integration(self):
        """Initialize integration with the Monitoring Dashboard."""
        try:
            # Initialize integration with monitoring dashboard
            self.monitoring_integration = VisualizationDashboardIntegration(
                dashboard_dir=self.dashboard_dir,
                integration_dir=os.path.join(self.dashboard_dir, "monitor_integration")
            )
            
            logger.info("Monitoring Dashboard integration initialized")
        except Exception as e:
            logger.error(f"Error initializing Monitoring Dashboard integration: {e}")
            self.monitoring_integration = None
    
    def create_dashboard(
        self,
        validation_results: List[ValidationResult],
        dashboard_name: Optional[str] = None,
        dashboard_title: Optional[str] = None,
        dashboard_description: Optional[str] = None,
        components: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Create a dashboard for validation results.
        
        Args:
            validation_results: List of validation results
            dashboard_name: Optional custom name for dashboard
            dashboard_title: Optional title for dashboard
            dashboard_description: Optional description for dashboard
            components: Optional custom component configurations
            
        Returns:
            Path to the created dashboard
        """
        if not validation_results:
            logger.error("No validation results provided")
            return None
        
        # Generate a unique name if not provided
        if not dashboard_name:
            dashboard_name = f"{self.config['dashboard_name']}_{datetime.datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
        
        # Use default title and description if not provided
        if not dashboard_title:
            dashboard_title = f"Benchmark Validation Dashboard - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        if not dashboard_description:
            dashboard_description = f"Dashboard for {len(validation_results)} validation results generated on {datetime.datetime.now().strftime('%Y-%m-%d')}"
        
        # Cache validation results for component generation
        self.validation_results_cache[dashboard_name] = validation_results
        
        # Create dashboard components if not provided
        if not components:
            components = self._generate_dashboard_components(validation_results)
        
        dashboard_path = None
        
        # Create dashboard using Advanced Visualization System if available
        if ADVANCED_VIZ_AVAILABLE and self.dashboard_instance:
            try:
                dashboard_path = self.dashboard_instance.create_dashboard(
                    dashboard_name=dashboard_name,
                    title=dashboard_title,
                    description=dashboard_description,
                    components=components
                )
                
                logger.info(f"Created dashboard at {dashboard_path}")
                
                # Register with monitoring dashboard if integration is available
                if MONITORING_DASHBOARD_AVAILABLE and self.monitoring_integration:
                    try:
                        embedded_dashboard = self.monitoring_integration.create_embedded_dashboard(
                            name=dashboard_name,
                            page="validation",
                            template="overview",
                            title=dashboard_title,
                            description=dashboard_description,
                            position="below",
                            components=components
                        )
                        
                        logger.info(f"Dashboard registered with monitoring system: {embedded_dashboard['name']}")
                    except Exception as e:
                        logger.error(f"Error registering dashboard with monitoring system: {e}")
                
                return dashboard_path
            
            except Exception as e:
                logger.error(f"Error creating dashboard with Advanced Visualization: {e}")
        
        # Fallback to basic HTML dashboard if Advanced Visualization not available
        try:
            report_content = self.reporter.generate_report(
                validation_results=validation_results,
                report_format="html",
                include_visualizations=True
            )
            
            dashboard_path = os.path.join(self.dashboard_dir, f"{dashboard_name}.html")
            
            with open(dashboard_path, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Created basic HTML dashboard at {dashboard_path}")
            return dashboard_path
        
        except Exception as e:
            logger.error(f"Error creating basic HTML dashboard: {e}")
            return None
    
    def _generate_dashboard_components(self, validation_results: List[ValidationResult]) -> List[Dict[str, Any]]:
        """
        Generate dashboard components based on validation results.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            List of component configurations
        """
        components = []
        
        # Only generate advanced components if we have the necessary libraries
        if ADVANCED_VIZ_AVAILABLE and PANDAS_AVAILABLE:
            # 1. Summary metrics component
            summary_metrics = self._calculate_summary_metrics(validation_results)
            components.append({
                "type": "metrics",
                "title": "Validation Summary",
                "config": {
                    "metrics": summary_metrics
                },
                "width": 2,
                "height": 1
            })
            
            # 2. Confidence score distribution
            components.append({
                "type": "confidence_distribution",
                "title": "Confidence Score Distribution",
                "config": {
                    "bin_count": 20,
                    "show_status_breakdown": True
                },
                "width": 1,
                "height": 1
            })
            
            # 3. Status by benchmark type
            components.append({
                "type": "status_by_benchmark_type",
                "title": "Validation Status by Benchmark Type",
                "config": {
                    "show_percentage": True,
                    "sort_by": "count"
                },
                "width": 1,
                "height": 1
            })
            
            # 4. Hardware-model validation heatmap
            components.append({
                "type": "validation_heatmap",
                "title": "Validation Status Heatmap",
                "config": {
                    "metric": "confidence_score",
                    "row_group": "model_family",
                    "col_group": "hardware_type",
                    "color_scale": "YlGnBu"
                },
                "width": 2,
                "height": 1
            })
            
            # 5. Time series if we have timestamp data
            has_timestamps = all(hasattr(val_result, 'timestamp') for val_result in validation_results)
            if has_timestamps:
                components.append({
                    "type": "time_series",
                    "title": "Validation Trends Over Time",
                    "config": {
                        "metrics": ["confidence_score", "validation_time"],
                        "group_by": "benchmark_type",
                        "time_range": 90  # 90 days
                    },
                    "width": 2,
                    "height": 1
                })
            
            # 6. Detailed results table
            components.append({
                "type": "results_table",
                "title": "Detailed Validation Results",
                "config": {
                    "page_size": 10,
                    "enable_search": True,
                    "enable_filtering": True,
                    "columns": [
                        "id", "model_id", "hardware_id", "benchmark_type", 
                        "validation_level", "status", "confidence_score", "issues"
                    ]
                },
                "width": 2,
                "height": 1
            })
            
            # 7. Validation issues breakdown
            components.append({
                "type": "issues_breakdown",
                "title": "Common Validation Issues",
                "config": {
                    "max_issues": 10,
                    "group_by_type": True
                },
                "width": 1,
                "height": 1
            })
            
            # 8. Recommendations if any
            has_recommendations = any(
                hasattr(val_result, 'recommendations') and val_result.recommendations 
                for val_result in validation_results
            )
            
            if has_recommendations:
                components.append({
                    "type": "recommendations",
                    "title": "Validation Recommendations",
                    "config": {
                        "max_recommendations": 10,
                        "group_by_type": True
                    },
                    "width": 1,
                    "height": 1
                })
        
        return components
    
    def _calculate_summary_metrics(self, validation_results: List[ValidationResult]) -> List[Dict[str, Any]]:
        """
        Calculate summary metrics for the dashboard.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            List of summary metrics
        """
        summary_metrics = []
        
        # 1. Total validation results
        summary_metrics.append({
            "name": "Total Results",
            "value": len(validation_results),
            "format": "number",
            "color": "info"
        })
        
        # 2. Valid results count
        valid_count = sum(1 for r in validation_results if r.status == ValidationStatus.VALID)
        valid_percent = round(valid_count / len(validation_results) * 100) if validation_results else 0
        
        summary_metrics.append({
            "name": "Valid Results",
            "value": valid_count,
            "percentage": valid_percent,
            "format": "number",
            "color": "success"
        })
        
        # 3. Invalid results count
        invalid_count = sum(1 for r in validation_results if r.status == ValidationStatus.INVALID)
        invalid_percent = round(invalid_count / len(validation_results) * 100) if validation_results else 0
        
        summary_metrics.append({
            "name": "Invalid Results",
            "value": invalid_count,
            "percentage": invalid_percent,
            "format": "number",
            "color": "danger"
        })
        
        # 4. Warning results count
        warning_count = sum(1 for r in validation_results if r.status == ValidationStatus.WARNING)
        warning_percent = round(warning_count / len(validation_results) * 100) if validation_results else 0
        
        summary_metrics.append({
            "name": "Warnings",
            "value": warning_count,
            "percentage": warning_percent,
            "format": "number",
            "color": "warning"
        })
        
        # 5. Average confidence score
        avg_confidence = sum(r.confidence_score for r in validation_results) / len(validation_results) if validation_results else 0
        
        summary_metrics.append({
            "name": "Avg Confidence",
            "value": avg_confidence,
            "format": "percentage",
            "color": "info"
        })
        
        # 6. Benchmark types count
        benchmark_types_set = set(r.benchmark_result.benchmark_type.name for r in validation_results)
        
        summary_metrics.append({
            "name": "Benchmark Types",
            "value": len(benchmark_types_set),
            "format": "number",
            "color": "secondary"
        })
        
        return summary_metrics
    
    def update_dashboard(
        self,
        dashboard_name: str,
        validation_results: Optional[List[ValidationResult]] = None,
        dashboard_title: Optional[str] = None,
        dashboard_description: Optional[str] = None,
        components: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Update an existing dashboard.
        
        Args:
            dashboard_name: Name of the dashboard to update
            validation_results: Optional new validation results
            dashboard_title: Optional new title
            dashboard_description: Optional new description
            components: Optional new component configurations
            
        Returns:
            Path to the updated dashboard
        """
        # Check if dashboard exists in Advanced Visualization System
        dashboard_path = None
        
        if ADVANCED_VIZ_AVAILABLE and self.dashboard_instance:
            try:
                # Update validation results cache if provided
                if validation_results:
                    self.validation_results_cache[dashboard_name] = validation_results
                
                # Generate new components if validation results provided and components not specified
                if validation_results and not components:
                    components = self._generate_dashboard_components(validation_results)
                
                # Update dashboard
                dashboard_path = self.dashboard_instance.update_dashboard(
                    dashboard_name=dashboard_name,
                    title=dashboard_title,
                    description=dashboard_description,
                    components=components
                )
                
                if dashboard_path:
                    logger.info(f"Updated dashboard at {dashboard_path}")
                    
                    # Update in monitoring dashboard if integrated
                    if MONITORING_DASHBOARD_AVAILABLE and self.monitoring_integration:
                        try:
                            updated_dashboard = self.monitoring_integration.update_embedded_dashboard(
                                name=dashboard_name,
                                title=dashboard_title,
                                description=dashboard_description
                            )
                            
                            if updated_dashboard:
                                logger.info(f"Updated dashboard in monitoring system: {updated_dashboard['name']}")
                        except Exception as e:
                            logger.error(f"Error updating dashboard in monitoring system: {e}")
                    
                    return dashboard_path
                else:
                    logger.error(f"Dashboard '{dashboard_name}' not found in Advanced Visualization System")
            
            except Exception as e:
                logger.error(f"Error updating dashboard with Advanced Visualization: {e}")
        
        # Fallback to recreating basic HTML dashboard
        if dashboard_name in self.validation_results_cache:
            validation_results_to_use = validation_results or self.validation_results_cache[dashboard_name]
            
            try:
                report_content = self.reporter.generate_report(
                    validation_results=validation_results_to_use,
                    report_format="html",
                    include_visualizations=True
                )
                
                dashboard_path = os.path.join(self.dashboard_dir, f"{dashboard_name}.html")
                
                with open(dashboard_path, 'w') as f:
                    f.write(report_content)
                
                logger.info(f"Updated basic HTML dashboard at {dashboard_path}")
                return dashboard_path
            
            except Exception as e:
                logger.error(f"Error updating basic HTML dashboard: {e}")
                return None
        else:
            logger.error(f"Dashboard '{dashboard_name}' not found in validation results cache")
            return None
    
    def export_dashboard(
        self,
        dashboard_name: str,
        export_format: str = "html",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export a dashboard to a specific format.
        
        Args:
            dashboard_name: Name of the dashboard to export
            export_format: Format to export (html, pdf, png)
            output_path: Optional custom output path
            
        Returns:
            Path to the exported file
        """
        if not output_path:
            output_dir = os.path.join(self.output_dir, "exports")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{dashboard_name}.{export_format}")
        
        # Use Advanced Visualization System if available
        if ADVANCED_VIZ_AVAILABLE and self.dashboard_instance:
            try:
                export_path = self.dashboard_instance.export_dashboard(
                    dashboard_name=dashboard_name,
                    format=export_format,
                    output_path=output_path
                )
                
                if export_path:
                    logger.info(f"Exported dashboard to {export_path}")
                    return export_path
                else:
                    logger.error(f"Failed to export dashboard '{dashboard_name}'")
            
            except Exception as e:
                logger.error(f"Error exporting dashboard: {e}")
        
        # Fallback to basic export using the reporter
        if dashboard_name in self.validation_results_cache:
            validation_results = self.validation_results_cache[dashboard_name]
            
            try:
                if export_format == "html":
                    report_content = self.reporter.generate_report(
                        validation_results=validation_results,
                        report_format="html",
                        include_visualizations=True
                    )
                    
                    with open(output_path, 'w') as f:
                        f.write(report_content)
                    
                    logger.info(f"Exported basic HTML dashboard to {output_path}")
                    return output_path
                
                elif export_format == "markdown" or export_format == "md":
                    report_content = self.reporter.generate_report(
                        validation_results=validation_results,
                        report_format="markdown",
                        include_visualizations=False
                    )
                    
                    with open(output_path, 'w') as f:
                        f.write(report_content)
                    
                    logger.info(f"Exported markdown dashboard to {output_path}")
                    return output_path
                
                elif export_format == "json":
                    report_content = self.reporter.generate_report(
                        validation_results=validation_results,
                        report_format="json",
                        include_visualizations=False
                    )
                    
                    with open(output_path, 'w') as f:
                        f.write(report_content)
                    
                    logger.info(f"Exported JSON dashboard to {output_path}")
                    return output_path
                
                else:
                    logger.error(f"Unsupported export format: {export_format}")
                    return None
            
            except Exception as e:
                logger.error(f"Error exporting dashboard: {e}")
                return None
        else:
            logger.error(f"Dashboard '{dashboard_name}' not found in validation results cache")
            return None
    
    def list_dashboards(self) -> List[Dict[str, Any]]:
        """
        List all available dashboards.
        
        Returns:
            List of dashboard details
        """
        dashboards = []
        
        # Get dashboards from Advanced Visualization System if available
        if ADVANCED_VIZ_AVAILABLE and self.dashboard_instance:
            try:
                advanced_dashboards = self.dashboard_instance.list_dashboards()
                dashboards.extend(advanced_dashboards)
            except Exception as e:
                logger.error(f"Error listing dashboards from Advanced Visualization: {e}")
        
        # Get dashboards from monitoring integration if available
        if MONITORING_DASHBOARD_AVAILABLE and self.monitoring_integration:
            try:
                embedded_dashboards = self.monitoring_integration.get_embedded_dashboards_for_page("validation")
                for name, details in embedded_dashboards.items():
                    # Check if we have it from Advanced Visualization already to avoid duplicates
                    if not any(d.get("name") == name for d in dashboards):
                        dashboards.append({
                            "name": name,
                            "title": details.get("title", name),
                            "description": details.get("description", ""),
                            "created_at": details.get("created_at", ""),
                            "updated_at": details.get("updated_at", ""),
                            "path": details.get("path", ""),
                            "source": "monitoring"
                        })
            except Exception as e:
                logger.error(f"Error listing dashboards from Monitoring Dashboard: {e}")
        
        # Look for basic HTML dashboards in dashboard directory
        try:
            for file_name in os.listdir(self.dashboard_dir):
                if file_name.endswith(".html") and not any(d.get("name") == file_name[:-5] for d in dashboards):
                    file_path = os.path.join(self.dashboard_dir, file_name)
                    
                    # Get file modification time
                    try:
                        mtime = os.path.getmtime(file_path)
                        modified_time = datetime.datetime.fromtimestamp(mtime).isoformat()
                    except:
                        modified_time = ""
                    
                    dashboards.append({
                        "name": file_name[:-5],
                        "title": file_name[:-5].replace("_", " ").title(),
                        "description": f"Basic HTML Dashboard ({file_name})",
                        "created_at": "",
                        "updated_at": modified_time,
                        "path": file_path,
                        "source": "basic"
                    })
        except Exception as e:
            logger.error(f"Error listing basic HTML dashboards: {e}")
        
        return dashboards
    
    def get_dashboard_url(self, dashboard_name: str, base_url: Optional[str] = None) -> str:
        """
        Get the URL for a dashboard.
        
        Args:
            dashboard_name: Name of the dashboard
            base_url: Optional base URL for the dashboard server
            
        Returns:
            URL to the dashboard
        """
        # Try to get the embedded dashboard URL first if monitoring integration is available
        if MONITORING_DASHBOARD_AVAILABLE and self.monitoring_integration:
            try:
                embedded_dashboard = self.monitoring_integration.get_embedded_dashboard(dashboard_name)
                if embedded_dashboard:
                    return f"{base_url or ''}/validation?dashboard={dashboard_name}"
            except Exception as e:
                logger.error(f"Error getting dashboard URL from monitoring integration: {e}")
        
        # Get relative path from Advanced Visualization System
        if ADVANCED_VIZ_AVAILABLE and self.dashboard_instance:
            try:
                dashboards = self.dashboard_instance.list_dashboards()
                for dashboard in dashboards:
                    if dashboard.get("name") == dashboard_name:
                        path = dashboard.get("path", "")
                        if path:
                            # Convert to relative path for URL
                            rel_path = os.path.relpath(path, start=self.dashboard_dir)
                            return f"{base_url or ''}/{self.config['dashboard_directory']}/{rel_path}"
            except Exception as e:
                logger.error(f"Error getting dashboard URL from Advanced Visualization: {e}")
        
        # Fallback to basic HTML path
        html_path = os.path.join(self.dashboard_dir, f"{dashboard_name}.html")
        if os.path.exists(html_path):
            rel_path = os.path.relpath(html_path, start=self.dashboard_dir)
            return f"{base_url or ''}/{self.config['dashboard_directory']}/{rel_path}"
        
        logger.error(f"Dashboard '{dashboard_name}' not found")
        return ""
    
    def get_dashboard_iframe_html(self, dashboard_name: str, width: str = "100%", height: str = "800px") -> str:
        """
        Get HTML to embed dashboard using an iframe.
        
        Args:
            dashboard_name: Dashboard name
            width: iframe width
            height: iframe height
        
        Returns:
            HTML string with iframe code
        """
        # Try to get iframe HTML from monitoring integration first
        if MONITORING_DASHBOARD_AVAILABLE and self.monitoring_integration:
            try:
                iframe_html = self.monitoring_integration.get_dashboard_iframe_html(
                    name=dashboard_name,
                    width=width,
                    height=height
                )
                
                if iframe_html:
                    return iframe_html
            except Exception as e:
                logger.error(f"Error getting iframe HTML from monitoring integration: {e}")
        
        # Fallback to direct iframe
        dashboards = self.list_dashboards()
        for dashboard in dashboards:
            if dashboard.get("name") == dashboard_name:
                path = dashboard.get("path", "")
                if path:
                    # Create iframe HTML
                    return f"""
                    <div class="embedded-dashboard">
                        <iframe src="{self.get_dashboard_url(dashboard_name)}" 
                                width="{width}" height="{height}" frameborder="0">
                        </iframe>
                    </div>
                    """
        
        logger.error(f"Dashboard '{dashboard_name}' not found")
        return ""
    
    def delete_dashboard(self, dashboard_name: str) -> bool:
        """
        Delete a dashboard.
        
        Args:
            dashboard_name: Name of the dashboard to delete
            
        Returns:
            True if successful, False otherwise
        """
        success = False
        
        # Delete from Advanced Visualization System if available
        if ADVANCED_VIZ_AVAILABLE and self.dashboard_instance:
            try:
                deleted = self.dashboard_instance.delete_dashboard(dashboard_name)
                if deleted:
                    success = True
                    logger.info(f"Deleted dashboard '{dashboard_name}' from Advanced Visualization")
            except Exception as e:
                logger.error(f"Error deleting dashboard from Advanced Visualization: {e}")
        
        # Remove from monitoring integration if available
        if MONITORING_DASHBOARD_AVAILABLE and self.monitoring_integration:
            try:
                removed = self.monitoring_integration.remove_embedded_dashboard(dashboard_name)
                if removed:
                    success = True
                    logger.info(f"Removed dashboard '{dashboard_name}' from monitoring integration")
            except Exception as e:
                logger.error(f"Error removing dashboard from monitoring integration: {e}")
        
        # Remove from validation results cache
        if dashboard_name in self.validation_results_cache:
            del self.validation_results_cache[dashboard_name]
            success = True
        
        # Delete basic HTML file if it exists
        html_path = os.path.join(self.dashboard_dir, f"{dashboard_name}.html")
        if os.path.exists(html_path):
            try:
                os.remove(html_path)
                success = True
                logger.info(f"Deleted basic HTML dashboard at {html_path}")
            except Exception as e:
                logger.error(f"Error deleting basic HTML dashboard: {e}")
        
        return success
    
    def create_comparison_dashboard(
        self,
        validation_results_sets: Dict[str, List[ValidationResult]],
        dashboard_name: Optional[str] = None,
        dashboard_title: Optional[str] = None,
        dashboard_description: Optional[str] = None
    ) -> str:
        """
        Create a comparison dashboard for multiple sets of validation results.
        
        Args:
            validation_results_sets: Dictionary mapping set names to validation results
            dashboard_name: Optional custom name for dashboard
            dashboard_title: Optional title for dashboard
            dashboard_description: Optional description for dashboard
            
        Returns:
            Path to the created dashboard
        """
        if not validation_results_sets:
            logger.error("No validation results provided")
            return None
        
        # Generate a unique name if not provided
        if not dashboard_name:
            dashboard_name = f"comparison_dashboard_{datetime.datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
        
        # Use default title and description if not provided
        if not dashboard_title:
            dashboard_title = f"Benchmark Validation Comparison - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        if not dashboard_description:
            set_names = ", ".join(validation_results_sets.keys())
            dashboard_description = f"Comparison of validation results across multiple sets: {set_names}"
        
        # Generate comparison components
        components = self._generate_comparison_components(validation_results_sets)
        
        # Create dashboard using components
        return self.create_dashboard(
            validation_results=list(validation_results_sets.values())[0],  # Just use first set for base creation
            dashboard_name=dashboard_name,
            dashboard_title=dashboard_title,
            dashboard_description=dashboard_description,
            components=components
        )
    
    def _generate_comparison_components(
        self,
        validation_results_sets: Dict[str, List[ValidationResult]]
    ) -> List[Dict[str, Any]]:
        """
        Generate comparison dashboard components based on multiple sets of validation results.
        
        Args:
            validation_results_sets: Dictionary mapping set names to validation results
            
        Returns:
            List of component configurations
        """
        components = []
        
        # Only generate advanced components if we have the necessary libraries
        if ADVANCED_VIZ_AVAILABLE and PANDAS_AVAILABLE:
            # 1. Summary comparison component
            summary_metrics = {}
            for set_name, validation_results in validation_results_sets.items():
                summary_metrics[set_name] = self._calculate_summary_metrics(validation_results)
            
            components.append({
                "type": "comparison_metrics",
                "title": "Validation Summary Comparison",
                "config": {
                    "metrics_sets": summary_metrics
                },
                "width": 2,
                "height": 1
            })
            
            # 2. Side-by-side confidence distribution
            components.append({
                "type": "comparison_confidence_distribution",
                "title": "Confidence Score Distribution Comparison",
                "config": {
                    "bin_count": 20,
                    "show_status_breakdown": True,
                    "sets": list(validation_results_sets.keys())
                },
                "width": 2,
                "height": 1
            })
            
            # 3. Status comparison by benchmark type
            components.append({
                "type": "comparison_status_by_benchmark_type",
                "title": "Validation Status by Benchmark Type",
                "config": {
                    "show_percentage": True,
                    "sort_by": "count",
                    "sets": list(validation_results_sets.keys())
                },
                "width": 2,
                "height": 1
            })
            
            # 4. Multi-heatmap comparison
            components.append({
                "type": "comparison_heatmap",
                "title": "Validation Status Heatmap Comparison",
                "config": {
                    "metric": "confidence_score",
                    "row_group": "model_family",
                    "col_group": "hardware_type",
                    "color_scale": "YlGnBu",
                    "sets": list(validation_results_sets.keys())
                },
                "width": 2,
                "height": 1
            })
            
            # 5. Detailed comparison table
            components.append({
                "type": "comparison_table",
                "title": "Detailed Results Comparison",
                "config": {
                    "page_size": 10,
                    "enable_search": True,
                    "enable_filtering": True,
                    "sets": list(validation_results_sets.keys())
                },
                "width": 2,
                "height": 1
            })
        
        return components
    
    def embed_dashboard(
        self,
        dashboard_name: str,
        parent_dashboard_name: str,
        position: str = "below",
        section: Optional[str] = None
    ) -> bool:
        """
        Embed a dashboard within another dashboard.
        
        Args:
            dashboard_name: Name of the dashboard to embed
            parent_dashboard_name: Name of the parent dashboard
            position: Position to embed (above, below, tab)
            section: Optional section name in the parent dashboard
            
        Returns:
            True if successful, False otherwise
        """
        if not ADVANCED_VIZ_AVAILABLE or not self.dashboard_instance:
            logger.error("Advanced Visualization System not available")
            return False
        
        try:
            # Get iframe HTML for the dashboard
            iframe_html = self.get_dashboard_iframe_html(dashboard_name)
            
            if not iframe_html:
                logger.error(f"Could not get iframe HTML for dashboard '{dashboard_name}'")
                return False
            
            # Embed dashboard in parent dashboard
            embedded = self.dashboard_instance.embed_html_in_dashboard(
                dashboard_name=parent_dashboard_name,
                html_content=iframe_html,
                position=position,
                section=section
            )
            
            if embedded:
                logger.info(f"Embedded dashboard '{dashboard_name}' in '{parent_dashboard_name}'")
                return True
            else:
                logger.error(f"Failed to embed dashboard '{dashboard_name}' in '{parent_dashboard_name}'")
                return False
        
        except Exception as e:
            logger.error(f"Error embedding dashboard: {e}")
            return False
    
    def register_with_monitoring_dashboard(
        self,
        dashboard_name: str,
        page: str = "validation",
        position: str = "below"
    ) -> bool:
        """
        Register a dashboard with the monitoring dashboard.
        
        Args:
            dashboard_name: Name of the dashboard to register
            page: Page in the monitoring dashboard to embed
            position: Position to embed (above, below, tab)
            
        Returns:
            True if successful, False otherwise
        """
        if not MONITORING_DASHBOARD_AVAILABLE or not self.monitoring_integration:
            logger.error("Monitoring Dashboard integration not available")
            return False
        
        try:
            # Get dashboard details
            dashboards = self.list_dashboards()
            dashboard_details = None
            
            for dashboard in dashboards:
                if dashboard.get("name") == dashboard_name:
                    dashboard_details = dashboard
                    break
            
            if not dashboard_details:
                logger.error(f"Dashboard '{dashboard_name}' not found")
                return False
            
            # Register with monitoring dashboard
            embedded_dashboard = self.monitoring_integration.create_embedded_dashboard(
                name=dashboard_name,
                page=page,
                template="overview",
                title=dashboard_details.get("title", dashboard_name),
                description=dashboard_details.get("description", ""),
                position=position
            )
            
            if embedded_dashboard:
                logger.info(f"Dashboard registered with monitoring system: {embedded_dashboard['name']}")
                return True
            else:
                logger.error(f"Failed to register dashboard with monitoring system")
                return False
        
        except Exception as e:
            logger.error(f"Error registering dashboard with monitoring system: {e}")
            return False