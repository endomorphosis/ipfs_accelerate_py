"""
Comparative Report Generator for Simulation Validation Results.

This module provides functionality for generating comparative reports that
highlight differences between multiple validation results, such as different
simulation versions, calibration states, or hardware configurations.
"""

import os
import json
import datetime
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import uuid

from .report_generator import ReportGenerator, ReportFormat, ReportType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simulation_validation.reporting.comparative")

class ComparativeReportGenerator(ReportGenerator):
    """
    Generates comparative reports from multiple simulation validation results.
    
    This class extends the base ReportGenerator to provide specialized functionality
    for creating reports that compare and contrast multiple validation results,
    highlighting improvements, regressions, and key differences.
    """
    
    def __init__(
        self,
        output_dir: str = "reports",
        templates_dir: Optional[str] = None,
        default_format: ReportFormat = ReportFormat.HTML,
        include_visualizations: bool = True,
        include_timestamps: bool = True,
        include_version_info: bool = True,
        custom_css: Optional[str] = None,
        custom_logo: Optional[str] = None,
        company_name: Optional[str] = None,
        custom_templates: Optional[Dict[str, str]] = None,
        highlight_improvements: bool = True,
        highlight_regressions: bool = True,
        improvement_threshold: float = 5.0,  # Percentage improvement to highlight
        regression_threshold: float = 5.0,  # Percentage regression to highlight
        include_statistical_significance: bool = True,
        significance_level: float = 0.05,
        include_trend_analysis: bool = True,
        trend_window: int = 5  # Number of versions to include in trend analysis
    ):
        """
        Initialize the comparative report generator.
        
        Args:
            output_dir: Directory to store generated reports
            templates_dir: Directory containing report templates
            default_format: Default report format
            include_visualizations: Whether to include visualizations in reports
            include_timestamps: Whether to include timestamps in reports
            include_version_info: Whether to include version info in reports
            custom_css: Path to custom CSS file for HTML reports
            custom_logo: Path to custom logo image for reports
            company_name: Company name to include in reports
            custom_templates: Dict mapping template names to template file paths
            highlight_improvements: Whether to highlight improvements
            highlight_regressions: Whether to highlight regressions
            improvement_threshold: Threshold for highlighting improvements
            regression_threshold: Threshold for highlighting regressions
            include_statistical_significance: Whether to include significance tests
            significance_level: Significance level for statistical tests
            include_trend_analysis: Whether to include trend analysis
            trend_window: Number of versions to include in trend analysis
        """
        super().__init__(
            output_dir=output_dir,
            templates_dir=templates_dir,
            default_format=default_format,
            include_visualizations=include_visualizations,
            include_timestamps=include_timestamps,
            include_version_info=include_version_info,
            custom_css=custom_css,
            custom_logo=custom_logo,
            company_name=company_name,
            custom_templates=custom_templates
        )
        
        # Comparative report specific settings
        self.highlight_improvements = highlight_improvements
        self.highlight_regressions = highlight_regressions
        self.improvement_threshold = improvement_threshold
        self.regression_threshold = regression_threshold
        self.include_statistical_significance = include_statistical_significance
        self.significance_level = significance_level
        self.include_trend_analysis = include_trend_analysis
        self.trend_window = trend_window
        
        logger.info(f"Comparative report generator initialized with improvement threshold {improvement_threshold}% and regression threshold {regression_threshold}%")
    
    def generate_comparative_report(
        self,
        validation_results_before: Dict[str, Any],
        validation_results_after: Dict[str, Any],
        output_format: Optional[ReportFormat] = None,
        output_filename: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        visualization_paths: Optional[Dict[str, str]] = None,
        before_label: str = "Before",
        after_label: str = "After",
        version_info: Optional[Dict[str, Any]] = None,
        historical_results: Optional[List[Dict[str, Any]]] = None,
        include_details: bool = True,
        metrics_to_compare: Optional[List[str]] = None,
        comparison_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comparative report from two sets of validation results.
        
        Args:
            validation_results_before: Validation results for the baseline
            validation_results_after: Validation results for the new version
            output_format: Output format for the report
            output_filename: Filename for the generated report
            title: Report title
            description: Report description
            metadata: Additional metadata to include in the report
            visualization_paths: Paths to visualization files to include
            before_label: Label for the baseline results
            after_label: Label for the new results
            version_info: Information about the versions being compared
            historical_results: Historical validation results for trend analysis
            include_details: Whether to include detailed result comparisons
            metrics_to_compare: List of metrics to include in the comparison
            comparison_notes: Additional notes to include in the report
        
        Returns:
            Dictionary with information about the generated report
        """
        # Use default title if not provided
        if not title:
            title = "Simulation Validation Comparative Report"
        
        # Create comparison data
        comparison_data = self._create_comparison_data(
            validation_results_before=validation_results_before,
            validation_results_after=validation_results_after,
            before_label=before_label,
            after_label=after_label,
            metrics_to_compare=metrics_to_compare,
            include_details=include_details
        )
        
        # Add version info to comparison data
        if version_info:
            comparison_data["versions"] = version_info
        
        # Add trend analysis if requested and historical results are provided
        if self.include_trend_analysis and historical_results:
            comparison_data["trend_analysis"] = self._create_trend_analysis(
                historical_results=historical_results,
                current_results=validation_results_after,
                trend_window=self.trend_window
            )
        
        # Prepare template variables
        template_vars = {}
        
        # Add before and after labels
        template_vars["before_label"] = before_label
        template_vars["after_label"] = after_label
        
        # Add comparison notes if provided
        if comparison_notes:
            template_vars["comparison_notes"] = comparison_notes
        
        # Add comparative specific content
        template_vars["improvement_threshold"] = self.improvement_threshold
        template_vars["regression_threshold"] = self.regression_threshold
        template_vars["comparative_report_html"] = self._generate_comparative_report_html(
            comparison_data=comparison_data,
            before_label=before_label,
            after_label=after_label,
            comparison_notes=comparison_notes
        )
        template_vars["comparative_report_markdown"] = self._generate_comparative_report_markdown(
            comparison_data=comparison_data,
            before_label=before_label,
            after_label=after_label,
            comparison_notes=comparison_notes
        )
        template_vars["comparative_report_text"] = self._generate_comparative_report_text(
            comparison_data=comparison_data,
            before_label=before_label,
            after_label=after_label,
            comparison_notes=comparison_notes
        )
        
        # Generate the comparative report using the base class method
        return super().generate_report(
            validation_results=validation_results_after,  # Use the "after" results as the main results
            report_type=ReportType.COMPARATIVE_REPORT,
            output_format=output_format,
            output_filename=output_filename,
            title=title,
            description=description,
            metadata=metadata,
            template_vars=template_vars,
            visualization_paths=visualization_paths,
            comparative_data=comparison_data,
            show_improvement=self.highlight_improvements
        )
    
    def _create_comparison_data(
        self,
        validation_results_before: Dict[str, Any],
        validation_results_after: Dict[str, Any],
        before_label: str = "Before",
        after_label: str = "After",
        metrics_to_compare: Optional[List[str]] = None,
        include_details: bool = True
    ) -> Dict[str, Any]:
        """
        Create comparison data from two sets of validation results.
        
        Args:
            validation_results_before: Validation results for the baseline
            validation_results_after: Validation results for the new version
            before_label: Label for the baseline results
            after_label: Label for the new results
            metrics_to_compare: List of metrics to include in the comparison
            include_details: Whether to include detailed result comparisons
            
        Returns:
            Dictionary with comparison data
        """
        # Initialize comparison data
        comparison_data = {
            "summary": {
                "before_label": before_label,
                "after_label": after_label
            },
            "comparison": {},
            "details": {
                "hardware": {},
                "model": {}
            },
            "significant_changes": {
                "improvements": [],
                "regressions": []
            }
        }
        
        # Default metrics to compare
        if metrics_to_compare is None:
            metrics_to_compare = ["mape", "rmse", "correlation", "status"]
        
        # Create overall comparison
        overall_before = validation_results_before.get("overall", {})
        overall_after = validation_results_after.get("overall", {})
        
        # Compare metrics
        for metric in metrics_to_compare:
            if metric in overall_before and metric in overall_after:
                before_data = overall_before[metric]
                after_data = overall_after[metric]
                
                # Check if the metric is a dictionary with statistics
                if isinstance(before_data, dict) and "mean" in before_data and isinstance(after_data, dict) and "mean" in after_data:
                    before_value = before_data["mean"]
                    after_value = after_data["mean"]
                    
                    # Calculate improvement percentage (for numeric metrics)
                    try:
                        if before_value != 0:
                            improvement = ((before_value - after_value) / before_value) * 100
                            improvement_direction = 1  # Positive means improved (lower is better)
                        else:
                            improvement = 0
                            improvement_direction = 0
                        
                        # Adjust improvement direction for metrics where higher is better
                        if metric in ["correlation", "ranking"]:
                            improvement = -improvement  # Flip the sign
                            improvement_direction = -1  # Negative means improved (higher is better)
                        
                        comparison_data["comparison"][metric] = {
                            "before": before_value,
                            "after": after_value,
                            "improvement": improvement,
                            "improvement_direction": improvement_direction,
                            "is_significant_improvement": improvement >= self.improvement_threshold and improvement_direction == 1,
                            "is_significant_regression": improvement <= -self.regression_threshold and improvement_direction == 1
                        }
                        
                        # Adjust significance for metrics where higher is better
                        if metric in ["correlation", "ranking"]:
                            comparison_data["comparison"][metric]["is_significant_improvement"] = improvement >= self.improvement_threshold and improvement_direction == -1
                            comparison_data["comparison"][metric]["is_significant_regression"] = improvement <= -self.regression_threshold and improvement_direction == -1
                        
                        # Add to significant changes list if applicable
                        metric_name = metric.upper()
                        if comparison_data["comparison"][metric]["is_significant_improvement"]:
                            comparison_data["significant_changes"]["improvements"].append({
                                "metric": metric_name,
                                "before": before_value,
                                "after": after_value,
                                "improvement": improvement
                            })
                        elif comparison_data["comparison"][metric]["is_significant_regression"]:
                            comparison_data["significant_changes"]["regressions"].append({
                                "metric": metric_name,
                                "before": before_value,
                                "after": after_value,
                                "improvement": improvement
                            })
                    except (TypeError, ValueError) as e:
                        # Handle non-numeric metrics
                        comparison_data["comparison"][metric] = {
                            "before": before_value,
                            "after": after_value,
                            "improvement": None
                        }
                else:
                    # Handle non-statistical metrics (e.g., status)
                    comparison_data["comparison"][metric] = {
                        "before": before_data,
                        "after": after_data,
                        "improvement": "N/A"
                    }
        
        # Add detailed comparisons if requested
        if include_details:
            # Compare hardware results
            hardware_before = validation_results_before.get("hardware_results", {})
            hardware_after = validation_results_after.get("hardware_results", {})
            
            # Get union of hardware keys
            all_hardware = set(hardware_before.keys()) | set(hardware_after.keys())
            
            for hw_id in all_hardware:
                hw_before = hardware_before.get(hw_id, {})
                hw_after = hardware_after.get(hw_id, {})
                
                comparison_data["details"]["hardware"][hw_id] = {
                    "metrics": {}
                }
                
                # Compare metrics for this hardware
                for metric in metrics_to_compare:
                    if metric in hw_before and metric in hw_after:
                        before_data = hw_before[metric]
                        after_data = hw_after[metric]
                        
                        # Check if the metric is a dictionary with statistics
                        if isinstance(before_data, dict) and "mean" in before_data and isinstance(after_data, dict) and "mean" in after_data:
                            before_value = before_data["mean"]
                            after_value = after_data["mean"]
                            
                            # Calculate improvement percentage
                            try:
                                if before_value != 0:
                                    improvement = ((before_value - after_value) / before_value) * 100
                                    improvement_direction = 1  # Positive means improved (lower is better)
                                else:
                                    improvement = 0
                                    improvement_direction = 0
                                
                                # Adjust improvement direction for metrics where higher is better
                                if metric in ["correlation", "ranking"]:
                                    improvement = -improvement  # Flip the sign
                                    improvement_direction = -1  # Negative means improved (higher is better)
                                
                                comparison_data["details"]["hardware"][hw_id]["metrics"][metric] = {
                                    "before": before_value,
                                    "after": after_value,
                                    "improvement": improvement,
                                    "improvement_direction": improvement_direction,
                                    "is_significant_improvement": improvement >= self.improvement_threshold and improvement_direction == 1,
                                    "is_significant_regression": improvement <= -self.regression_threshold and improvement_direction == 1
                                }
                                
                                # Adjust significance for metrics where higher is better
                                if metric in ["correlation", "ranking"]:
                                    comparison_data["details"]["hardware"][hw_id]["metrics"][metric]["is_significant_improvement"] = improvement >= self.improvement_threshold and improvement_direction == -1
                                    comparison_data["details"]["hardware"][hw_id]["metrics"][metric]["is_significant_regression"] = improvement <= -self.regression_threshold and improvement_direction == -1
                            except (TypeError, ValueError) as e:
                                # Handle non-numeric metrics
                                comparison_data["details"]["hardware"][hw_id]["metrics"][metric] = {
                                    "before": before_value,
                                    "after": after_value,
                                    "improvement": None
                                }
                        else:
                            # Handle non-statistical metrics (e.g., status)
                            comparison_data["details"]["hardware"][hw_id]["metrics"][metric] = {
                                "before": before_data,
                                "after": after_data,
                                "improvement": "N/A"
                            }
            
            # Compare model results
            model_before = validation_results_before.get("model_results", {})
            model_after = validation_results_after.get("model_results", {})
            
            # Get union of model keys
            all_models = set(model_before.keys()) | set(model_after.keys())
            
            for model_id in all_models:
                model_before_data = model_before.get(model_id, {})
                model_after_data = model_after.get(model_id, {})
                
                comparison_data["details"]["model"][model_id] = {
                    "metrics": {}
                }
                
                # Compare metrics for this model
                for metric in metrics_to_compare:
                    if metric in model_before_data and metric in model_after_data:
                        before_data = model_before_data[metric]
                        after_data = model_after_data[metric]
                        
                        # Check if the metric is a dictionary with statistics
                        if isinstance(before_data, dict) and "mean" in before_data and isinstance(after_data, dict) and "mean" in after_data:
                            before_value = before_data["mean"]
                            after_value = after_data["mean"]
                            
                            # Calculate improvement percentage
                            try:
                                if before_value != 0:
                                    improvement = ((before_value - after_value) / before_value) * 100
                                    improvement_direction = 1  # Positive means improved (lower is better)
                                else:
                                    improvement = 0
                                    improvement_direction = 0
                                
                                # Adjust improvement direction for metrics where higher is better
                                if metric in ["correlation", "ranking"]:
                                    improvement = -improvement  # Flip the sign
                                    improvement_direction = -1  # Negative means improved (higher is better)
                                
                                comparison_data["details"]["model"][model_id]["metrics"][metric] = {
                                    "before": before_value,
                                    "after": after_value,
                                    "improvement": improvement,
                                    "improvement_direction": improvement_direction,
                                    "is_significant_improvement": improvement >= self.improvement_threshold and improvement_direction == 1,
                                    "is_significant_regression": improvement <= -self.regression_threshold and improvement_direction == 1
                                }
                                
                                # Adjust significance for metrics where higher is better
                                if metric in ["correlation", "ranking"]:
                                    comparison_data["details"]["model"][model_id]["metrics"][metric]["is_significant_improvement"] = improvement >= self.improvement_threshold and improvement_direction == -1
                                    comparison_data["details"]["model"][model_id]["metrics"][metric]["is_significant_regression"] = improvement <= -self.regression_threshold and improvement_direction == -1
                            except (TypeError, ValueError) as e:
                                # Handle non-numeric metrics
                                comparison_data["details"]["model"][model_id]["metrics"][metric] = {
                                    "before": before_value,
                                    "after": after_value,
                                    "improvement": None
                                }
                        else:
                            # Handle non-statistical metrics (e.g., status)
                            comparison_data["details"]["model"][model_id]["metrics"][metric] = {
                                "before": before_data,
                                "after": after_data,
                                "improvement": "N/A"
                            }
        
        # Add statistical significance if requested
        if self.include_statistical_significance:
            comparison_data["statistical_significance"] = self._calculate_statistical_significance(
                validation_results_before=validation_results_before,
                validation_results_after=validation_results_after,
                metrics_to_compare=metrics_to_compare
            )
        
        return comparison_data
    
    def _calculate_statistical_significance(
        self,
        validation_results_before: Dict[str, Any],
        validation_results_after: Dict[str, Any],
        metrics_to_compare: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance of changes between two sets of validation results.
        
        This method determines if the observed differences are statistically significant
        using appropriate statistical tests. For simplicity, it currently uses a basic
        approach of comparing means and standard deviations. In a real implementation,
        this would be replaced with proper hypothesis testing.
        
        Args:
            validation_results_before: Validation results for the baseline
            validation_results_after: Validation results for the new version
            metrics_to_compare: List of metrics to include in the comparison
            
        Returns:
            Dictionary with statistical significance results
        """
        # Initialize significance data
        significance_data = {}
        
        # Extract overall metrics
        overall_before = validation_results_before.get("overall", {})
        overall_after = validation_results_after.get("overall", {})
        
        # Check significance for each metric
        for metric in metrics_to_compare:
            if metric in overall_before and metric in overall_after:
                before_data = overall_before[metric]
                after_data = overall_after[metric]
                
                # Check if the metric is a dictionary with statistics
                if isinstance(before_data, dict) and "mean" in before_data and isinstance(after_data, dict) and "mean" in after_data:
                    before_mean = before_data["mean"]
                    after_mean = after_data["mean"]
                    before_std = before_data.get("std_dev", 0)
                    after_std = after_data.get("std_dev", 0)
                    before_size = before_data.get("sample_size", 30)  # Default to 30 if not provided
                    after_size = after_data.get("sample_size", 30)  # Default to 30 if not provided
                    
                    # Perform a simplified significance test
                    try:
                        import math
                        
                        # Calculate standard error
                        se_before = before_std / math.sqrt(before_size)
                        se_after = after_std / math.sqrt(after_size)
                        
                        # Calculate standard error of the difference
                        se_diff = math.sqrt(se_before**2 + se_after**2)
                        
                        # Calculate t-statistic
                        if se_diff > 0:
                            t_stat = (after_mean - before_mean) / se_diff
                        else:
                            t_stat = 0
                        
                        # Calculate approximate p-value (simplified approach)
                        from scipy import stats
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), before_size + after_size - 2))
                        
                        # Determine significance
                        is_significant = p_value < self.significance_level
                        
                        significance_data[metric] = {
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "degrees_of_freedom": before_size + after_size - 2,
                            "is_significant": is_significant,
                            "significance_level": self.significance_level
                        }
                    except (ImportError, TypeError, ValueError) as e:
                        # If scipy is not available or other error occurs, use a very simple approach
                        difference = abs(after_mean - before_mean)
                        pooled_std = (before_std + after_std) / 2
                        
                        if pooled_std > 0:
                            effect_size = difference / pooled_std
                        else:
                            effect_size = 0
                        
                        # Arbitrary threshold for significance
                        is_significant = effect_size > 0.5
                        
                        significance_data[metric] = {
                            "difference": difference,
                            "effect_size": effect_size,
                            "is_significant": is_significant,
                            "note": "Simple effect size calculation (scipy not available or error occurred)"
                        }
        
        return significance_data
    
    def _create_trend_analysis(
        self,
        historical_results: List[Dict[str, Any]],
        current_results: Dict[str, Any],
        trend_window: int = 5
    ) -> Dict[str, Any]:
        """
        Create trend analysis data from historical validation results.
        
        Args:
            historical_results: List of historical validation results
            current_results: Current validation results
            trend_window: Number of versions to include in trend analysis
            
        Returns:
            Dictionary with trend analysis data
        """
        # Initialize trend analysis data
        trend_data = {
            "metrics": {},
            "versions": []
        }
        
        # Sort historical results by timestamp (newest first)
        sorted_results = sorted(historical_results, key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Limit to trend window
        recent_results = sorted_results[:trend_window-1]  # -1 to make room for current results
        
        # Add current results to the list
        all_results = [current_results] + recent_results
        
        # Extract version information
        for result in all_results:
            metadata = result.get("metadata", {})
            trend_data["versions"].append({
                "version": metadata.get("version", "Unknown"),
                "timestamp": metadata.get("timestamp", "Unknown"),
                "description": metadata.get("description", "")
            })
        
        # Extract metrics for trend analysis
        metrics_to_track = ["mape", "rmse", "correlation"]
        
        for metric in metrics_to_track:
            trend_data["metrics"][metric] = []
            
            for result in all_results:
                overall = result.get("overall", {})
                
                if metric in overall:
                    metric_data = overall[metric]
                    
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        trend_data["metrics"][metric].append(metric_data["mean"])
                    else:
                        trend_data["metrics"][metric].append(None)
                else:
                    trend_data["metrics"][metric].append(None)
        
        # Calculate trends
        for metric, values in trend_data["metrics"].items():
            # Filter out None values
            filtered_values = [v for v in values if v is not None]
            
            if len(filtered_values) >= 2:
                # Calculate simple linear regression (slope)
                # In a real implementation, this would use proper statistical methods
                try:
                    x = list(range(len(filtered_values)))
                    y = filtered_values
                    
                    # Calculate means
                    mean_x = sum(x) / len(x)
                    mean_y = sum(y) / len(y)
                    
                    # Calculate slope
                    numerator = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y))
                    denominator = sum((x_i - mean_x)**2 for x_i in x)
                    
                    if denominator != 0:
                        slope = numerator / denominator
                    else:
                        slope = 0
                    
                    # Determine trend direction
                    if abs(slope) < 0.001:
                        trend = "stable"
                    elif slope < 0:
                        trend = "improving"  # Negative slope means decreasing error (for most metrics)
                    else:
                        trend = "worsening"  # Positive slope means increasing error (for most metrics)
                    
                    # Adjust trend direction for metrics where higher is better
                    if metric in ["correlation", "ranking"]:
                        if trend == "improving":
                            trend = "worsening"
                        elif trend == "worsening":
                            trend = "improving"
                    
                    trend_data["metrics"][metric + "_trend"] = {
                        "slope": slope,
                        "direction": trend
                    }
                except (TypeError, ValueError) as e:
                    trend_data["metrics"][metric + "_trend"] = {
                        "slope": 0,
                        "direction": "unknown",
                        "error": str(e)
                    }
            else:
                trend_data["metrics"][metric + "_trend"] = {
                    "slope": 0,
                    "direction": "insufficient data"
                }
        
        return trend_data
    
    def _generate_comparative_report_html(
        self,
        comparison_data: Dict[str, Any],
        before_label: str = "Before",
        after_label: str = "After",
        comparison_notes: Optional[str] = None
    ) -> str:
        """Generate HTML for comparative report."""
        html = ""
        
        # Add summary section
        html += "<div class='section summary'>"
        html += "<h2>Comparison Summary</h2>"
        
        # Add significant improvements and regressions
        improvements = comparison_data["significant_changes"]["improvements"]
        regressions = comparison_data["significant_changes"]["regressions"]
        
        if self.highlight_improvements and improvements:
            html += "<div class='improvements'>"
            html += "<h3>Significant Improvements</h3>"
            html += "<ul>"
            
            for improvement in improvements:
                html += f"<li><span class='improvement-metric'>{improvement['metric']}</span>: "
                html += f"<span class='improvement-value'>{improvement['before']:.4f} → {improvement['after']:.4f}</span> "
                html += f"<span class='improvement-percentage improvement-positive'>({improvement['improvement']:.2f}% improvement)</span></li>"
            
            html += "</ul>"
            html += "</div>"
        
        if self.highlight_regressions and regressions:
            html += "<div class='regressions'>"
            html += "<h3>Significant Regressions</h3>"
            html += "<ul>"
            
            for regression in regressions:
                html += f"<li><span class='regression-metric'>{regression['metric']}</span>: "
                html += f"<span class='regression-value'>{regression['before']:.4f} → {regression['after']:.4f}</span> "
                html += f"<span class='improvement-percentage improvement-negative'>({regression['improvement']:.2f}% change)</span></li>"
            
            html += "</ul>"
            html += "</div>"
        
        # Add comparison notes if provided
        if comparison_notes:
            html += "<div class='comparison-notes'>"
            html += "<h3>Notes</h3>"
            html += f"<p>{comparison_notes}</p>"
            html += "</div>"
        
        html += "</div>"
        
        # Add overall comparison
        html += "<div class='section overall-comparison'>"
        html += "<h2>Overall Metrics Comparison</h2>"
        
        html += "<table class='comparison-table'>"
        html += f"<tr><th>Metric</th><th>{before_label}</th><th>{after_label}</th><th>Change</th></tr>"
        
        for metric, data in comparison_data["comparison"].items():
            before_value = data["before"]
            after_value = data["after"]
            improvement = data.get("improvement")
            
            if improvement is not None and improvement != "N/A":
                # For numeric improvements, add appropriate styling
                if data.get("is_significant_improvement", False):
                    change_class = "improvement-positive"
                    change_text = f"+{improvement:.2f}%"
                elif data.get("is_significant_regression", False):
                    change_class = "improvement-negative"
                    change_text = f"{improvement:.2f}%"
                else:
                    change_class = "improvement-neutral"
                    change_text = f"{improvement:.2f}%"
            else:
                # For non-numeric improvements, just show the change
                change_class = ""
                if before_value == after_value:
                    change_text = "No change"
                else:
                    change_text = f"{before_value} → {after_value}"
            
            html += f"<tr><td>{metric.upper()}</td>"
            html += f"<td>{before_value}</td>"
            html += f"<td>{after_value}</td>"
            html += f"<td class='{change_class}'>{change_text}</td></tr>"
        
        html += "</table>"
        html += "</div>"
        
        # Add statistical significance if included
        if "statistical_significance" in comparison_data:
            html += "<div class='section statistical-significance'>"
            html += "<h2>Statistical Significance</h2>"
            
            html += "<table class='statistics-table'>"
            html += "<tr><th>Metric</th><th>T-Statistic</th><th>P-Value</th><th>Significant</th></tr>"
            
            for metric, sig_data in comparison_data["statistical_significance"].items():
                is_significant = sig_data.get("is_significant", False)
                significance_class = "positive" if is_significant else "neutral"
                
                html += f"<tr><td>{metric.upper()}</td>"
                
                if "t_statistic" in sig_data:
                    html += f"<td>{sig_data['t_statistic']:.4f}</td>"
                    html += f"<td>{sig_data['p_value']:.4f}</td>"
                else:
                    html += f"<td>N/A</td><td>N/A</td>"
                
                html += f"<td class='{significance_class}'>{is_significant}</td></tr>"
            
            html += "</table>"
            html += "</div>"
        
        # Add detailed comparisons if included
        if "details" in comparison_data and comparison_data["details"].get("hardware") and comparison_data["details"].get("model"):
            # Hardware details
            html += "<div class='section hardware-comparison'>"
            html += "<h2>Hardware-Specific Comparisons</h2>"
            
            for hw_id, hw_data in comparison_data["details"]["hardware"].items():
                html += f"<h3>Hardware: {hw_id}</h3>"
                
                html += "<table class='comparison-table'>"
                html += f"<tr><th>Metric</th><th>{before_label}</th><th>{after_label}</th><th>Change</th></tr>"
                
                for metric, data in hw_data["metrics"].items():
                    before_value = data["before"]
                    after_value = data["after"]
                    improvement = data.get("improvement")
                    
                    if improvement is not None and improvement != "N/A":
                        # For numeric improvements, add appropriate styling
                        if data.get("is_significant_improvement", False):
                            change_class = "improvement-positive"
                            change_text = f"+{improvement:.2f}%"
                        elif data.get("is_significant_regression", False):
                            change_class = "improvement-negative"
                            change_text = f"{improvement:.2f}%"
                        else:
                            change_class = "improvement-neutral"
                            change_text = f"{improvement:.2f}%"
                    else:
                        # For non-numeric improvements, just show the change
                        change_class = ""
                        if before_value == after_value:
                            change_text = "No change"
                        else:
                            change_text = f"{before_value} → {after_value}"
                    
                    html += f"<tr><td>{metric.upper()}</td>"
                    html += f"<td>{before_value}</td>"
                    html += f"<td>{after_value}</td>"
                    html += f"<td class='{change_class}'>{change_text}</td></tr>"
                
                html += "</table>"
            
            html += "</div>"
            
            # Model details
            html += "<div class='section model-comparison'>"
            html += "<h2>Model-Specific Comparisons</h2>"
            
            for model_id, model_data in comparison_data["details"]["model"].items():
                html += f"<h3>Model: {model_id}</h3>"
                
                html += "<table class='comparison-table'>"
                html += f"<tr><th>Metric</th><th>{before_label}</th><th>{after_label}</th><th>Change</th></tr>"
                
                for metric, data in model_data["metrics"].items():
                    before_value = data["before"]
                    after_value = data["after"]
                    improvement = data.get("improvement")
                    
                    if improvement is not None and improvement != "N/A":
                        # For numeric improvements, add appropriate styling
                        if data.get("is_significant_improvement", False):
                            change_class = "improvement-positive"
                            change_text = f"+{improvement:.2f}%"
                        elif data.get("is_significant_regression", False):
                            change_class = "improvement-negative"
                            change_text = f"{improvement:.2f}%"
                        else:
                            change_class = "improvement-neutral"
                            change_text = f"{improvement:.2f}%"
                    else:
                        # For non-numeric improvements, just show the change
                        change_class = ""
                        if before_value == after_value:
                            change_text = "No change"
                        else:
                            change_text = f"{before_value} → {after_value}"
                    
                    html += f"<tr><td>{metric.upper()}</td>"
                    html += f"<td>{before_value}</td>"
                    html += f"<td>{after_value}</td>"
                    html += f"<td class='{change_class}'>{change_text}</td></tr>"
                
                html += "</table>"
            
            html += "</div>"
        
        # Add trend analysis if included
        if "trend_analysis" in comparison_data:
            html += "<div class='section trend-analysis'>"
            html += "<h2>Trend Analysis</h2>"
            
            # Add version information
            html += "<div class='versions'>"
            html += "<h3>Versions Analyzed</h3>"
            
            html += "<table class='versions-table'>"
            html += "<tr><th>Version</th><th>Date</th><th>Description</th></tr>"
            
            for version_info in comparison_data["trend_analysis"]["versions"]:
                html += f"<tr><td>{version_info['version']}</td>"
                html += f"<td>{version_info['timestamp']}</td>"
                html += f"<td>{version_info['description']}</td></tr>"
            
            html += "</table>"
            html += "</div>"
            
            # Add metric trends
            html += "<div class='metric-trends'>"
            html += "<h3>Metric Trends</h3>"
            
            html += "<table class='trends-table'>"
            html += "<tr><th>Metric</th><th>Trend</th><th>Direction</th></tr>"
            
            for metric in ["mape", "rmse", "correlation"]:
                trend_key = metric + "_trend"
                if trend_key in comparison_data["trend_analysis"]["metrics"]:
                    trend_data = comparison_data["trend_analysis"]["metrics"][trend_key]
                    trend_direction = trend_data["direction"]
                    
                    # Set appropriate class for trend direction
                    if trend_direction == "improving":
                        direction_class = "improvement-positive"
                    elif trend_direction == "worsening":
                        direction_class = "improvement-negative"
                    else:
                        direction_class = "improvement-neutral"
                    
                    html += f"<tr><td>{metric.upper()}</td>"
                    html += f"<td>{trend_data['slope']:.6f}</td>"
                    html += f"<td class='{direction_class}'>{trend_direction}</td></tr>"
            
            html += "</table>"
            html += "</div>"
            
            html += "</div>"
        
        return html
    
    def _generate_comparative_report_markdown(
        self,
        comparison_data: Dict[str, Any],
        before_label: str = "Before",
        after_label: str = "After",
        comparison_notes: Optional[str] = None
    ) -> str:
        """Generate Markdown for comparative report."""
        markdown = ""
        
        # Add summary section
        markdown += "## Comparison Summary\n\n"
        
        # Add significant improvements and regressions
        improvements = comparison_data["significant_changes"]["improvements"]
        regressions = comparison_data["significant_changes"]["regressions"]
        
        if self.highlight_improvements and improvements:
            markdown += "### Significant Improvements\n\n"
            
            for improvement in improvements:
                markdown += f"- **{improvement['metric']}**: "
                markdown += f"{improvement['before']:.4f} → {improvement['after']:.4f} "
                markdown += f"(🔼 {improvement['improvement']:.2f}% improvement)\n"
            
            markdown += "\n"
        
        if self.highlight_regressions and regressions:
            markdown += "### Significant Regressions\n\n"
            
            for regression in regressions:
                markdown += f"- **{regression['metric']}**: "
                markdown += f"{regression['before']:.4f} → {regression['after']:.4f} "
                markdown += f"(🔽 {regression['improvement']:.2f}% change)\n"
            
            markdown += "\n"
        
        # Add comparison notes if provided
        if comparison_notes:
            markdown += "### Notes\n\n"
            markdown += f"{comparison_notes}\n\n"
        
        # Add overall comparison
        markdown += "## Overall Metrics Comparison\n\n"
        
        markdown += f"| Metric | {before_label} | {after_label} | Change |\n"
        markdown += "| ------ | ------- | ------ | ------ |\n"
        
        for metric, data in comparison_data["comparison"].items():
            before_value = data["before"]
            after_value = data["after"]
            improvement = data.get("improvement")
            
            if improvement is not None and improvement != "N/A":
                # For numeric improvements, add appropriate symbol
                if data.get("is_significant_improvement", False):
                    change_symbol = "🔼"
                    change_text = f"+{improvement:.2f}%"
                elif data.get("is_significant_regression", False):
                    change_symbol = "🔽"
                    change_text = f"{improvement:.2f}%"
                else:
                    change_symbol = "◯"
                    change_text = f"{improvement:.2f}%"
                
                change_field = f"{change_symbol} {change_text}"
            else:
                # For non-numeric improvements, just show the change
                if before_value == after_value:
                    change_field = "No change"
                else:
                    change_field = f"{before_value} → {after_value}"
            
            markdown += f"| {metric.upper()} | {before_value} | {after_value} | {change_field} |\n"
        
        markdown += "\n"
        
        # Add statistical significance if included
        if "statistical_significance" in comparison_data:
            markdown += "## Statistical Significance\n\n"
            
            markdown += "| Metric | T-Statistic | P-Value | Significant |\n"
            markdown += "| ------ | ----------- | ------- | ----------- |\n"
            
            for metric, sig_data in comparison_data["statistical_significance"].items():
                is_significant = sig_data.get("is_significant", False)
                significance_symbol = "✅" if is_significant else "❌"
                
                if "t_statistic" in sig_data:
                    t_stat = f"{sig_data['t_statistic']:.4f}"
                    p_value = f"{sig_data['p_value']:.4f}"
                else:
                    t_stat = "N/A"
                    p_value = "N/A"
                
                markdown += f"| {metric.upper()} | {t_stat} | {p_value} | {significance_symbol} {is_significant} |\n"
            
            markdown += "\n"
        
        # Add detailed comparisons if included
        if "details" in comparison_data and comparison_data["details"].get("hardware") and comparison_data["details"].get("model"):
            # Hardware details
            markdown += "## Hardware-Specific Comparisons\n\n"
            
            for hw_id, hw_data in comparison_data["details"]["hardware"].items():
                markdown += f"### Hardware: {hw_id}\n\n"
                
                markdown += f"| Metric | {before_label} | {after_label} | Change |\n"
                markdown += "| ------ | ------- | ------ | ------ |\n"
                
                for metric, data in hw_data["metrics"].items():
                    before_value = data["before"]
                    after_value = data["after"]
                    improvement = data.get("improvement")
                    
                    if improvement is not None and improvement != "N/A":
                        # For numeric improvements, add appropriate symbol
                        if data.get("is_significant_improvement", False):
                            change_symbol = "🔼"
                            change_text = f"+{improvement:.2f}%"
                        elif data.get("is_significant_regression", False):
                            change_symbol = "🔽"
                            change_text = f"{improvement:.2f}%"
                        else:
                            change_symbol = "◯"
                            change_text = f"{improvement:.2f}%"
                        
                        change_field = f"{change_symbol} {change_text}"
                    else:
                        # For non-numeric improvements, just show the change
                        if before_value == after_value:
                            change_field = "No change"
                        else:
                            change_field = f"{before_value} → {after_value}"
                    
                    markdown += f"| {metric.upper()} | {before_value} | {after_value} | {change_field} |\n"
                
                markdown += "\n"
            
            # Model details
            markdown += "## Model-Specific Comparisons\n\n"
            
            for model_id, model_data in comparison_data["details"]["model"].items():
                markdown += f"### Model: {model_id}\n\n"
                
                markdown += f"| Metric | {before_label} | {after_label} | Change |\n"
                markdown += "| ------ | ------- | ------ | ------ |\n"
                
                for metric, data in model_data["metrics"].items():
                    before_value = data["before"]
                    after_value = data["after"]
                    improvement = data.get("improvement")
                    
                    if improvement is not None and improvement != "N/A":
                        # For numeric improvements, add appropriate symbol
                        if data.get("is_significant_improvement", False):
                            change_symbol = "🔼"
                            change_text = f"+{improvement:.2f}%"
                        elif data.get("is_significant_regression", False):
                            change_symbol = "🔽"
                            change_text = f"{improvement:.2f}%"
                        else:
                            change_symbol = "◯"
                            change_text = f"{improvement:.2f}%"
                        
                        change_field = f"{change_symbol} {change_text}"
                    else:
                        # For non-numeric improvements, just show the change
                        if before_value == after_value:
                            change_field = "No change"
                        else:
                            change_field = f"{before_value} → {after_value}"
                    
                    markdown += f"| {metric.upper()} | {before_value} | {after_value} | {change_field} |\n"
                
                markdown += "\n"
        
        # Add trend analysis if included
        if "trend_analysis" in comparison_data:
            markdown += "## Trend Analysis\n\n"
            
            # Add version information
            markdown += "### Versions Analyzed\n\n"
            
            markdown += "| Version | Date | Description |\n"
            markdown += "| ------- | ---- | ----------- |\n"
            
            for version_info in comparison_data["trend_analysis"]["versions"]:
                markdown += f"| {version_info['version']} | {version_info['timestamp']} | {version_info['description']} |\n"
            
            markdown += "\n"
            
            # Add metric trends
            markdown += "### Metric Trends\n\n"
            
            markdown += "| Metric | Trend | Direction |\n"
            markdown += "| ------ | ----- | --------- |\n"
            
            for metric in ["mape", "rmse", "correlation"]:
                trend_key = metric + "_trend"
                if trend_key in comparison_data["trend_analysis"]["metrics"]:
                    trend_data = comparison_data["trend_analysis"]["metrics"][trend_key]
                    trend_direction = trend_data["direction"]
                    
                    # Set appropriate symbol for trend direction
                    if trend_direction == "improving":
                        direction_symbol = "🔼"
                    elif trend_direction == "worsening":
                        direction_symbol = "🔽"
                    else:
                        direction_symbol = "◯"
                    
                    markdown += f"| {metric.upper()} | {trend_data['slope']:.6f} | {direction_symbol} {trend_direction} |\n"
            
            markdown += "\n"
        
        return markdown
    
    def _generate_comparative_report_text(
        self,
        comparison_data: Dict[str, Any],
        before_label: str = "Before",
        after_label: str = "After",
        comparison_notes: Optional[str] = None
    ) -> str:
        """Generate text for comparative report."""
        text = ""
        
        # Add summary section
        text += "COMPARISON SUMMARY\n"
        text += "==================\n\n"
        
        # Add significant improvements and regressions
        improvements = comparison_data["significant_changes"]["improvements"]
        regressions = comparison_data["significant_changes"]["regressions"]
        
        if self.highlight_improvements and improvements:
            text += "Significant Improvements:\n"
            text += "-------------------------\n\n"
            
            for improvement in improvements:
                text += f"* {improvement['metric']}: "
                text += f"{improvement['before']:.4f} → {improvement['after']:.4f} "
                text += f"({improvement['improvement']:.2f}% improvement)\n"
            
            text += "\n"
        
        if self.highlight_regressions and regressions:
            text += "Significant Regressions:\n"
            text += "-----------------------\n\n"
            
            for regression in regressions:
                text += f"* {regression['metric']}: "
                text += f"{regression['before']:.4f} → {regression['after']:.4f} "
                text += f"({regression['improvement']:.2f}% change)\n"
            
            text += "\n"
        
        # Add comparison notes if provided
        if comparison_notes:
            text += "Notes:\n"
            text += "------\n\n"
            text += f"{comparison_notes}\n\n"
        
        # Add overall comparison
        text += "OVERALL METRICS COMPARISON\n"
        text += "==========================\n\n"
        
        text += f"Metric | {before_label} | {after_label} | Change\n"
        text += "-" * 50 + "\n"
        
        for metric, data in comparison_data["comparison"].items():
            before_value = data["before"]
            after_value = data["after"]
            improvement = data.get("improvement")
            
            if improvement is not None and improvement != "N/A":
                # For numeric improvements, add appropriate indicator
                if data.get("is_significant_improvement", False):
                    change_text = f"+{improvement:.2f}% (Significant Improvement)"
                elif data.get("is_significant_regression", False):
                    change_text = f"{improvement:.2f}% (Significant Regression)"
                else:
                    change_text = f"{improvement:.2f}%"
            else:
                # For non-numeric improvements, just show the change
                if before_value == after_value:
                    change_text = "No change"
                else:
                    change_text = f"{before_value} → {after_value}"
            
            text += f"{metric.upper()} | {before_value} | {after_value} | {change_text}\n"
        
        text += "\n"
        
        # Add statistical significance if included
        if "statistical_significance" in comparison_data:
            text += "STATISTICAL SIGNIFICANCE\n"
            text += "========================\n\n"
            
            text += "Metric | T-Statistic | P-Value | Significant\n"
            text += "-" * 50 + "\n"
            
            for metric, sig_data in comparison_data["statistical_significance"].items():
                is_significant = sig_data.get("is_significant", False)
                significance_indicator = "Yes" if is_significant else "No"
                
                if "t_statistic" in sig_data:
                    t_stat = f"{sig_data['t_statistic']:.4f}"
                    p_value = f"{sig_data['p_value']:.4f}"
                else:
                    t_stat = "N/A"
                    p_value = "N/A"
                
                text += f"{metric.upper()} | {t_stat} | {p_value} | {significance_indicator}\n"
            
            text += "\n"
        
        # Add detailed comparisons if included (limited to conserve space in text format)
        if "details" in comparison_data and comparison_data["details"].get("hardware") and comparison_data["details"].get("model"):
            # Hardware details (just highlight significant changes)
            text += "HARDWARE-SPECIFIC SIGNIFICANT CHANGES\n"
            text += "=====================================\n\n"
            
            for hw_id, hw_data in comparison_data["details"]["hardware"].items():
                has_significant_changes = False
                
                for metric, data in hw_data["metrics"].items():
                    if data.get("is_significant_improvement", False) or data.get("is_significant_regression", False):
                        has_significant_changes = True
                        break
                
                if has_significant_changes:
                    text += f"Hardware: {hw_id}\n"
                    text += "-" * (len(hw_id) + 10) + "\n\n"
                    
                    for metric, data in hw_data["metrics"].items():
                        if data.get("is_significant_improvement", False) or data.get("is_significant_regression", False):
                            before_value = data["before"]
                            after_value = data["after"]
                            improvement = data.get("improvement")
                            
                            if data.get("is_significant_improvement", False):
                                change_indicator = "(Significant Improvement)"
                            else:
                                change_indicator = "(Significant Regression)"
                            
                            text += f"{metric.upper()}: {before_value} → {after_value} ({improvement:.2f}%) {change_indicator}\n"
                    
                    text += "\n"
            
            # Model details (just highlight significant changes)
            text += "MODEL-SPECIFIC SIGNIFICANT CHANGES\n"
            text += "==================================\n\n"
            
            for model_id, model_data in comparison_data["details"]["model"].items():
                has_significant_changes = False
                
                for metric, data in model_data["metrics"].items():
                    if data.get("is_significant_improvement", False) or data.get("is_significant_regression", False):
                        has_significant_changes = True
                        break
                
                if has_significant_changes:
                    text += f"Model: {model_id}\n"
                    text += "-" * (len(model_id) + 7) + "\n\n"
                    
                    for metric, data in model_data["metrics"].items():
                        if data.get("is_significant_improvement", False) or data.get("is_significant_regression", False):
                            before_value = data["before"]
                            after_value = data["after"]
                            improvement = data.get("improvement")
                            
                            if data.get("is_significant_improvement", False):
                                change_indicator = "(Significant Improvement)"
                            else:
                                change_indicator = "(Significant Regression)"
                            
                            text += f"{metric.upper()}: {before_value} → {after_value} ({improvement:.2f}%) {change_indicator}\n"
                    
                    text += "\n"
        
        # Add trend analysis if included
        if "trend_analysis" in comparison_data:
            text += "TREND ANALYSIS\n"
            text += "==============\n\n"
            
            # Add metric trends
            text += "Metric Trends:\n"
            text += "-------------\n\n"
            
            for metric in ["mape", "rmse", "correlation"]:
                trend_key = metric + "_trend"
                if trend_key in comparison_data["trend_analysis"]["metrics"]:
                    trend_data = comparison_data["trend_analysis"]["metrics"][trend_key]
                    trend_direction = trend_data["direction"]
                    
                    text += f"{metric.upper()}: Slope {trend_data['slope']:.6f} ({trend_direction})\n"
            
            text += "\n"
            
            # Add version information
            text += "Versions Analyzed:\n"
            text += "-----------------\n\n"
            
            for version_info in comparison_data["trend_analysis"]["versions"]:
                text += f"Version: {version_info['version']}\n"
                text += f"Date: {version_info['timestamp']}\n"
                text += f"Description: {version_info['description']}\n\n"
            
            text += "\n"
        
        return text

# Example usage
if __name__ == "__main__":
    # Create a comparative report generator
    generator = ComparativeReportGenerator(
        output_dir="reports",
        highlight_improvements=True,
        highlight_regressions=True,
        improvement_threshold=5.0,
        regression_threshold=5.0,
        include_statistical_significance=True,
        include_trend_analysis=True
    )
    
    # Create sample validation results (before)
    validation_results_before = {
        "overall": {
            "mape": {
                "mean": 8.45,
                "std_dev": 1.85,
                "min": 3.2,
                "max": 12.1,
                "sample_size": 50
            },
            "rmse": {
                "mean": 0.0312,
                "std_dev": 0.0092,
                "min": 0.0145,
                "max": 0.0523,
                "sample_size": 50
            },
            "correlation": {
                "mean": 0.86,
                "std_dev": 0.09,
                "min": 0.74,
                "max": 0.95,
                "sample_size": 50
            },
            "status": "pass"
        },
        "hardware_results": {
            "rtx3080": {
                "mape": {"mean": 7.82, "std_dev": 1.7},
                "rmse": {"mean": 0.0287, "std_dev": 0.008},
                "correlation": {"mean": 0.88, "std_dev": 0.08},
                "status": "pass"
            },
            "a100": {
                "mape": {"mean": 6.91, "std_dev": 1.5},
                "rmse": {"mean": 0.0256, "std_dev": 0.007},
                "correlation": {"mean": 0.91, "std_dev": 0.07},
                "status": "pass"
            }
        },
        "model_results": {
            "bert-base-uncased": {
                "mape": {"mean": 9.12, "std_dev": 1.9},
                "rmse": {"mean": 0.0341, "std_dev": 0.009},
                "correlation": {"mean": 0.85, "std_dev": 0.08},
                "status": "pass"
            },
            "vit-base-patch16-224": {
                "mape": {"mean": 8.67, "std_dev": 1.8},
                "rmse": {"mean": 0.0321, "std_dev": 0.008},
                "correlation": {"mean": 0.87, "std_dev": 0.07},
                "status": "pass"
            }
        }
    }
    
    # Create sample validation results (after)
    validation_results_after = {
        "overall": {
            "mape": {
                "mean": 5.32,
                "std_dev": 1.23,
                "min": 2.1,
                "max": 8.7,
                "sample_size": 50
            },
            "rmse": {
                "mean": 0.0245,
                "std_dev": 0.0078,
                "min": 0.0123,
                "max": 0.0456,
                "sample_size": 50
            },
            "correlation": {
                "mean": 0.92,
                "std_dev": 0.08,
                "min": 0.85,
                "max": 0.98,
                "sample_size": 50
            },
            "status": "pass"
        },
        "hardware_results": {
            "rtx3080": {
                "mape": {"mean": 4.21, "std_dev": 1.1},
                "rmse": {"mean": 0.0201, "std_dev": 0.005},
                "correlation": {"mean": 0.94, "std_dev": 0.06},
                "status": "pass"
            },
            "a100": {
                "mape": {"mean": 3.56, "std_dev": 0.9},
                "rmse": {"mean": 0.0189, "std_dev": 0.004},
                "correlation": {"mean": 0.96, "std_dev": 0.05},
                "status": "pass"
            }
        },
        "model_results": {
            "bert-base-uncased": {
                "mape": {"mean": 6.78, "std_dev": 1.5},
                "rmse": {"mean": 0.0312, "std_dev": 0.008},
                "correlation": {"mean": 0.91, "std_dev": 0.07},
                "status": "pass"
            },
            "vit-base-patch16-224": {
                "mape": {"mean": 5.92, "std_dev": 1.3},
                "rmse": {"mean": 0.0278, "std_dev": 0.007},
                "correlation": {"mean": 0.93, "std_dev": 0.06},
                "status": "pass"
            }
        }
    }
    
    # Sample version information
    version_info = [
        {
            "version": "1.0.0",
            "date": "2025-07-15",
            "description": "Initial version"
        },
        {
            "version": "1.1.0",
            "date": "2025-07-29",
            "description": "Improved calibration"
        }
    ]
    
    # Sample comparison notes
    comparison_notes = "The improvements are primarily due to enhanced calibration of simulation parameters based on extensive hardware testing data."
    
    # Generate comparative report
    report = generator.generate_comparative_report(
        validation_results_before=validation_results_before,
        validation_results_after=validation_results_after,
        output_format=ReportFormat.HTML,
        title="Simulation Validation Comparative Report",
        description="Comparison of simulation validation results before and after calibration",
        before_label="Before Calibration",
        after_label="After Calibration",
        version_info=version_info,
        comparison_notes=comparison_notes
    )
    
    print(f"Comparative report generated: {report['path']}")