"""
Executive Summary Generator for Simulation Validation Results.

This module provides functionality for generating executive summaries from
simulation validation results, focusing on high-level information for
executive stakeholders.
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
logger = logging.getLogger("simulation_validation.reporting.executive")

class ExecutiveSummaryGenerator(ReportGenerator):
    """
    Generates executive summaries from simulation validation results.
    
    This class extends the base ReportGenerator to provide specialized
    functionality for creating executive summaries that focus on high-level
    information and key metrics for executive stakeholders.
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
        key_metrics: Optional[List[str]] = None,
        include_recommendations: bool = True,
        max_visualization_count: int = 3,
        executive_level: str = "c-suite"  # c-suite, director, manager
    ):
        """
        Initialize the executive summary generator.
        
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
            key_metrics: List of key metrics to highlight in the summary
            include_recommendations: Whether to include recommendations
            max_visualization_count: Maximum number of visualizations to include
            executive_level: Target executive level (c-suite, director, manager)
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
        
        # Executive summary specific settings
        self.key_metrics = key_metrics or ["mape", "rmse", "status"]
        self.include_recommendations = include_recommendations
        self.max_visualization_count = max_visualization_count
        self.executive_level = executive_level
        
        logger.info(f"Executive summary generator initialized for {executive_level} level")
    
    def generate_executive_summary(
        self,
        validation_results: Dict[str, Any],
        output_format: Optional[ReportFormat] = None,
        output_filename: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        visualization_paths: Optional[Dict[str, str]] = None,
        comparative_data: Optional[Dict[str, Any]] = None,
        business_impact: Optional[str] = None,
        strategic_implications: Optional[str] = None,
        next_steps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate an executive summary from validation results.
        
        Args:
            validation_results: The validation results to include in the report
            output_format: Output format for the report
            output_filename: Filename for the generated report
            title: Report title
            description: Report description
            metadata: Additional metadata to include in the report
            visualization_paths: Paths to visualization files to include
            comparative_data: Data for comparative analysis
            business_impact: Business impact statement
            strategic_implications: Strategic implications statement
            next_steps: List of recommended next steps
        
        Returns:
            Dictionary with information about the generated report
        """
        # Use default title if not provided
        if not title:
            title = "Simulation Validation Executive Summary"
        
        # Prepare template variables
        template_vars = {}
        
        # Add business impact if provided
        if business_impact:
            template_vars["business_impact"] = business_impact
        
        # Add strategic implications if provided
        if strategic_implications:
            template_vars["strategic_implications"] = strategic_implications
        
        # Add next steps if provided and include_recommendations is True
        if next_steps and self.include_recommendations:
            template_vars["next_steps"] = next_steps
        
        # Limit visualizations to max_visualization_count
        limited_viz_paths = None
        if visualization_paths:
            # Sort by importance (assumed to be in order)
            limited_viz_paths = dict(list(visualization_paths.items())[:self.max_visualization_count])
        
        # Add executive level-specific content
        template_vars["executive_level"] = self.executive_level
        template_vars["key_highlights"] = self._extract_key_highlights(validation_results)
        template_vars["executive_summary_html"] = self._generate_executive_summary_html(
            validation_results, comparative_data, business_impact, strategic_implications, next_steps
        )
        template_vars["executive_summary_markdown"] = self._generate_executive_summary_markdown(
            validation_results, comparative_data, business_impact, strategic_implications, next_steps
        )
        template_vars["executive_summary_text"] = self._generate_executive_summary_text(
            validation_results, comparative_data, business_impact, strategic_implications, next_steps
        )
        
        # Generate the executive summary using the base class method
        return super().generate_report(
            validation_results=validation_results,
            report_type=ReportType.EXECUTIVE_SUMMARY,
            output_format=output_format,
            output_filename=output_filename,
            title=title,
            description=description,
            metadata=metadata,
            template_vars=template_vars,
            visualization_paths=limited_viz_paths,
            comparative_data=comparative_data,
            show_improvement=True
        )
    
    def _extract_key_highlights(self, validation_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key highlights from validation results based on executive level."""
        highlights = []
        
        # Extract overall status
        overall = validation_results.get("overall", {})
        if "status" in overall:
            status = overall["status"]
            highlights.append({
                "title": "Overall Validation Status",
                "value": status.upper(),
                "is_positive": status == "pass",
                "priority": 1
            })
        
        # Extract key metrics based on executive level
        if self.executive_level == "c-suite":
            # For C-suite, focus on high-level business impact metrics
            if "mape" in overall:
                highlights.append({
                    "title": "Average Error Rate",
                    "value": f"{overall['mape']['mean']:.2f}%",
                    "is_positive": overall["mape"]["mean"] < 10.0,  # Arbitrary threshold
                    "priority": 2
                })
            
            # Add any business-critical model performance
            model_results = validation_results.get("model_results", {})
            critical_models = self._get_critical_models(model_results)
            if critical_models:
                for model, data in critical_models.items():
                    highlights.append({
                        "title": f"Critical Model: {model}",
                        "value": f"{data.get('mape', {}).get('mean', 0):.2f}% error",
                        "is_positive": data.get("status") == "pass",
                        "priority": 3
                    })
        
        elif self.executive_level == "director":
            # For Director level, include more detailed metrics
            if "mape" in overall:
                highlights.append({
                    "title": "MAPE",
                    "value": f"{overall['mape']['mean']:.2f}% (±{overall['mape'].get('std_dev', 0):.2f}%)",
                    "is_positive": overall["mape"]["mean"] < 10.0,  # Arbitrary threshold
                    "priority": 2
                })
            
            if "rmse" in overall:
                highlights.append({
                    "title": "RMSE",
                    "value": f"{overall['rmse']['mean']:.4f}",
                    "is_positive": overall["rmse"]["mean"] < 0.05,  # Arbitrary threshold
                    "priority": 3
                })
            
            # Add hardware platform status
            hardware_results = validation_results.get("hardware_results", {})
            if hardware_results:
                pass_count = sum(1 for hw in hardware_results.values() if hw.get("status") == "pass")
                total_count = len(hardware_results)
                highlights.append({
                    "title": "Hardware Platform Status",
                    "value": f"{pass_count}/{total_count} passing",
                    "is_positive": pass_count == total_count,
                    "priority": 4
                })
        
        elif self.executive_level == "manager":
            # For Manager level, include even more detailed metrics
            if "mape" in overall:
                highlights.append({
                    "title": "MAPE",
                    "value": f"{overall['mape']['mean']:.2f}% (±{overall['mape'].get('std_dev', 0):.2f}%)",
                    "is_positive": overall["mape"]["mean"] < 10.0,  # Arbitrary threshold
                    "priority": 2
                })
            
            if "rmse" in overall:
                highlights.append({
                    "title": "RMSE",
                    "value": f"{overall['rmse']['mean']:.4f}",
                    "is_positive": overall["rmse"]["mean"] < 0.05,  # Arbitrary threshold
                    "priority": 3
                })
            
            # Add hardware platform status
            hardware_results = validation_results.get("hardware_results", {})
            if hardware_results:
                for hw_id, hw_data in hardware_results.items():
                    highlights.append({
                        "title": f"Hardware: {hw_id}",
                        "value": f"{hw_data.get('mape', {}).get('mean', 0):.2f}% error",
                        "is_positive": hw_data.get("status") == "pass",
                        "priority": 5
                    })
            
            # Add model status
            model_results = validation_results.get("model_results", {})
            if model_results:
                for model_id, model_data in list(model_results.items())[:3]:  # Limit to top 3
                    highlights.append({
                        "title": f"Model: {model_id}",
                        "value": f"{model_data.get('mape', {}).get('mean', 0):.2f}% error",
                        "is_positive": model_data.get("status") == "pass",
                        "priority": 6
                    })
        
        # Sort highlights by priority
        highlights.sort(key=lambda x: x["priority"])
        
        return highlights
    
    def _get_critical_models(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get critical models from model results.
        
        For now, just takes the top 1-2 models. In a real-world scenario,
        this would use domain knowledge to identify business-critical models.
        """
        # Sort models by MAPE (lower is better)
        sorted_models = sorted(
            model_results.items(),
            key=lambda x: x[1].get("mape", {}).get("mean", float("inf"))
        )
        
        # Take top 1-2 models
        critical_models = {}
        for i, (model_id, model_data) in enumerate(sorted_models):
            if i >= 2:  # Limit to 2 critical models
                break
            critical_models[model_id] = model_data
        
        return critical_models
    
    def _generate_executive_summary_html(
        self,
        validation_results: Dict[str, Any],
        comparative_data: Optional[Dict[str, Any]] = None,
        business_impact: Optional[str] = None,
        strategic_implications: Optional[str] = None,
        next_steps: Optional[List[str]] = None
    ) -> str:
        """Generate HTML for executive summary."""
        html = ""
        
        # Add key highlights
        key_highlights = self._extract_key_highlights(validation_results)
        
        html += "<div class='key-highlights'>"
        html += "<h3>Key Highlights</h3>"
        
        html += "<div class='highlights-grid'>"
        for highlight in key_highlights:
            status_class = "positive" if highlight["is_positive"] else "negative"
            html += f"<div class='highlight-card {status_class}'>"
            html += f"<div class='highlight-title'>{highlight['title']}</div>"
            html += f"<div class='highlight-value'>{highlight['value']}</div>"
            html += "</div>"
        html += "</div>"
        html += "</div>"
        
        # Add business impact if provided
        if business_impact:
            html += "<div class='section'>"
            html += "<h3>Business Impact</h3>"
            html += f"<p>{business_impact}</p>"
            html += "</div>"
        
        # Add strategic implications if provided
        if strategic_implications:
            html += "<div class='section'>"
            html += "<h3>Strategic Implications</h3>"
            html += f"<p>{strategic_implications}</p>"
            html += "</div>"
        
        # Add next steps if provided
        if next_steps:
            html += "<div class='section'>"
            html += "<h3>Recommended Next Steps</h3>"
            html += "<ul>"
            for step in next_steps:
                html += f"<li>{step}</li>"
            html += "</ul>"
            html += "</div>"
        
        # Add comparative data if provided
        if comparative_data and "comparison" in comparative_data:
            html += "<div class='section'>"
            html += "<h3>Performance Improvement</h3>"
            html += "<table class='comparison-table'>"
            html += "<tr><th>Metric</th><th>Before</th><th>After</th><th>Improvement</th></tr>"
            
            for metric, data in comparative_data["comparison"].items():
                if metric not in self.key_metrics:
                    continue
                
                before = data.get("before", "N/A")
                after = data.get("after", "N/A")
                improvement = data.get("improvement", 0)
                
                improvement_class = "positive" if improvement > 0 else "negative"
                html += f"<tr>"
                html += f"<td>{metric}</td><td>{before}</td><td>{after}</td>"
                html += f"<td class='{improvement_class}'>{improvement:.2f}%</td>"
                html += f"</tr>"
            
            html += "</table>"
            html += "</div>"
        
        return html
    
    def _generate_executive_summary_markdown(
        self,
        validation_results: Dict[str, Any],
        comparative_data: Optional[Dict[str, Any]] = None,
        business_impact: Optional[str] = None,
        strategic_implications: Optional[str] = None,
        next_steps: Optional[List[str]] = None
    ) -> str:
        """Generate Markdown for executive summary."""
        markdown = ""
        
        # Add key highlights
        key_highlights = self._extract_key_highlights(validation_results)
        
        markdown += "## Key Highlights\n\n"
        
        for highlight in key_highlights:
            symbol = "✅" if highlight["is_positive"] else "❌"
            markdown += f"- **{highlight['title']}**: {symbol} {highlight['value']}\n"
        
        markdown += "\n"
        
        # Add business impact if provided
        if business_impact:
            markdown += "## Business Impact\n\n"
            markdown += f"{business_impact}\n\n"
        
        # Add strategic implications if provided
        if strategic_implications:
            markdown += "## Strategic Implications\n\n"
            markdown += f"{strategic_implications}\n\n"
        
        # Add next steps if provided
        if next_steps:
            markdown += "## Recommended Next Steps\n\n"
            for step in next_steps:
                markdown += f"- {step}\n"
            markdown += "\n"
        
        # Add comparative data if provided
        if comparative_data and "comparison" in comparative_data:
            markdown += "## Performance Improvement\n\n"
            markdown += "| Metric | Before | After | Improvement |\n"
            markdown += "| ------ | ------ | ----- | ----------- |\n"
            
            for metric, data in comparative_data["comparison"].items():
                if metric not in self.key_metrics:
                    continue
                
                before = data.get("before", "N/A")
                after = data.get("after", "N/A")
                improvement = data.get("improvement", 0)
                
                symbol = "🔼" if improvement > 0 else "🔽"
                markdown += f"| {metric} | {before} | {after} | {symbol} {improvement:.2f}% |\n"
            
            markdown += "\n"
        
        return markdown
    
    def _generate_executive_summary_text(
        self,
        validation_results: Dict[str, Any],
        comparative_data: Optional[Dict[str, Any]] = None,
        business_impact: Optional[str] = None,
        strategic_implications: Optional[str] = None,
        next_steps: Optional[List[str]] = None
    ) -> str:
        """Generate text for executive summary."""
        text = ""
        
        # Add key highlights
        key_highlights = self._extract_key_highlights(validation_results)
        
        text += "KEY HIGHLIGHTS\n"
        text += "==============\n\n"
        
        for highlight in key_highlights:
            symbol = "+" if highlight["is_positive"] else "-"
            text += f"[{symbol}] {highlight['title']}: {highlight['value']}\n"
        
        text += "\n"
        
        # Add business impact if provided
        if business_impact:
            text += "BUSINESS IMPACT\n"
            text += "===============\n\n"
            text += f"{business_impact}\n\n"
        
        # Add strategic implications if provided
        if strategic_implications:
            text += "STRATEGIC IMPLICATIONS\n"
            text += "======================\n\n"
            text += f"{strategic_implications}\n\n"
        
        # Add next steps if provided
        if next_steps:
            text += "RECOMMENDED NEXT STEPS\n"
            text += "======================\n\n"
            for i, step in enumerate(next_steps, 1):
                text += f"{i}. {step}\n"
            text += "\n"
        
        # Add comparative data if provided
        if comparative_data and "comparison" in comparative_data:
            text += "PERFORMANCE IMPROVEMENT\n"
            text += "=======================\n\n"
            
            for metric, data in comparative_data["comparison"].items():
                if metric not in self.key_metrics:
                    continue
                
                before = data.get("before", "N/A")
                after = data.get("after", "N/A")
                improvement = data.get("improvement", 0)
                
                text += f"{metric}:\n"
                text += f"  Before: {before}\n"
                text += f"  After: {after}\n"
                text += f"  Improvement: {improvement:.2f}%\n\n"
        
        return text

# Example usage
if __name__ == "__main__":
    # Create an executive summary generator
    generator = ExecutiveSummaryGenerator(
        output_dir="reports",
        executive_level="director"
    )
    
    # Create sample validation results
    validation_results = {
        "overall": {
            "mape": {
                "mean": 5.32,
                "std_dev": 1.23,
                "min": 2.1,
                "max": 8.7
            },
            "rmse": {
                "mean": 0.0245,
                "std_dev": 0.0078,
                "min": 0.0123,
                "max": 0.0456
            },
            "status": "pass"
        },
        "hardware_results": {
            "rtx3080": {
                "mape": {"mean": 4.21},
                "rmse": {"mean": 0.0201},
                "status": "pass"
            },
            "a100": {
                "mape": {"mean": 3.56},
                "rmse": {"mean": 0.0189},
                "status": "pass"
            }
        },
        "model_results": {
            "bert-base-uncased": {
                "mape": {"mean": 6.78},
                "rmse": {"mean": 0.0312},
                "status": "pass"
            },
            "vit-base-patch16-224": {
                "mape": {"mean": 5.92},
                "rmse": {"mean": 0.0278},
                "status": "pass"
            }
        }
    }
    
    # Sample comparative data
    comparative_data = {
        "comparison": {
            "mape": {
                "before": "8.45%",
                "after": "5.32%",
                "improvement": 37.04
            },
            "rmse": {
                "before": "0.0312",
                "after": "0.0245",
                "improvement": 21.47
            }
        },
        "versions": [
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
    }
    
    # Sample business impact and next steps
    business_impact = "The improved simulation accuracy reduces hardware testing costs by an estimated 45% while maintaining confidence in performance predictions."
    strategic_implications = "These improvements enable faster time-to-market for new hardware configurations and better resource allocation for critical models."
    next_steps = [
        "Extend calibration to cover 3 additional hardware platforms",
        "Implement automated recalibration triggered by drift detection",
        "Present findings to Hardware Strategy team for integration with roadmap"
    ]
    
    # Generate executive summary
    report = generator.generate_executive_summary(
        validation_results=validation_results,
        output_format=ReportFormat.HTML,
        title="Simulation Validation Executive Summary",
        description="Executive summary of simulation validation results",
        comparative_data=comparative_data,
        business_impact=business_impact,
        strategic_implications=strategic_implications,
        next_steps=next_steps
    )
    
    print(f"Executive summary generated: {report['path']}")