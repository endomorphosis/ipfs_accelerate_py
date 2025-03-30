"""
Technical Report Generator for Simulation Validation Results.

This module provides functionality for generating detailed technical reports
from simulation validation results, including comprehensive statistical analysis
and method descriptions.
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
logger = logging.getLogger("simulation_validation.reporting.technical")

class TechnicalReportGenerator(ReportGenerator):
    """
    Generates detailed technical reports from simulation validation results.
    
    This class extends the base ReportGenerator to provide specialized
    functionality for creating technical reports with comprehensive statistical
    analysis, detailed method descriptions, and in-depth validation results.
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
        include_methodology: bool = True,
        include_statistical_analysis: bool = True,
        include_raw_data: bool = False,
        include_detailed_validation: bool = True,
        detailed_metrics: Optional[List[str]] = None,
        confidence_level: float = 0.95
    ):
        """
        Initialize the technical report generator.
        
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
            include_methodology: Whether to include methodology documentation
            include_statistical_analysis: Whether to include statistical analysis
            include_raw_data: Whether to include raw data tables
            include_detailed_validation: Whether to include detailed validation
            detailed_metrics: List of metrics to include detailed analysis for
            confidence_level: Confidence level for statistical intervals
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
        
        # Technical report specific settings
        self.include_methodology = include_methodology
        self.include_statistical_analysis = include_statistical_analysis
        self.include_raw_data = include_raw_data
        self.include_detailed_validation = include_detailed_validation
        self.detailed_metrics = detailed_metrics or ["mape", "rmse", "correlation", "ranking"]
        self.confidence_level = confidence_level
        
        logger.info(f"Technical report generator initialized with confidence level {confidence_level}")
    
    def generate_technical_report(
        self,
        validation_results: Dict[str, Any],
        output_format: Optional[ReportFormat] = None,
        output_filename: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        visualization_paths: Optional[Dict[str, str]] = None,
        comparative_data: Optional[Dict[str, Any]] = None,
        methodology_description: Optional[str] = None,
        statistical_notes: Optional[str] = None,
        raw_data: Optional[Dict[str, Any]] = None,
        limitations: Optional[List[str]] = None,
        appendices: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a technical report from validation results.
        
        Args:
            validation_results: The validation results to include in the report
            output_format: Output format for the report
            output_filename: Filename for the generated report
            title: Report title
            description: Report description
            metadata: Additional metadata to include in the report
            visualization_paths: Paths to visualization files to include
            comparative_data: Data for comparative analysis
            methodology_description: Custom methodology description
            statistical_notes: Notes on statistical analysis
            raw_data: Raw data to include in the report
            limitations: List of limitations and caveats
            appendices: Dict mapping appendix names to content
        
        Returns:
            Dictionary with information about the generated report
        """
        # Use default title if not provided
        if not title:
            title = "Simulation Validation Technical Report"
        
        # Prepare template variables
        template_vars = {}
        
        # Add methodology description if provided and include_methodology is True
        if methodology_description and self.include_methodology:
            template_vars["methodology_description"] = methodology_description
        
        # Add statistical notes if provided and include_statistical_analysis is True
        if statistical_notes and self.include_statistical_analysis:
            template_vars["statistical_notes"] = statistical_notes
        
        # Add raw data if provided and include_raw_data is True
        if raw_data and self.include_raw_data:
            template_vars["raw_data"] = raw_data
        
        # Add limitations if provided
        if limitations:
            template_vars["limitations"] = limitations
        
        # Add appendices if provided
        if appendices:
            template_vars["appendices"] = appendices
        
        # Add technical specific content
        template_vars["confidence_level"] = self.confidence_level
        template_vars["technical_report_html"] = self._generate_technical_report_html(
            validation_results, 
            raw_data, 
            methodology_description, 
            statistical_notes, 
            limitations, 
            appendices
        )
        template_vars["technical_report_markdown"] = self._generate_technical_report_markdown(
            validation_results, 
            raw_data, 
            methodology_description, 
            statistical_notes, 
            limitations, 
            appendices
        )
        template_vars["technical_report_text"] = self._generate_technical_report_text(
            validation_results, 
            raw_data, 
            methodology_description, 
            statistical_notes, 
            limitations, 
            appendices
        )
        template_vars["detailed_statistics"] = self._generate_detailed_statistics(validation_results)
        
        # Add the technical detailed visualizations section
        if visualization_paths and self.include_visualizations:
            template_vars["technical_visualizations_html"] = self._generate_technical_visualizations_html(
                visualization_paths
            )
            template_vars["technical_visualizations_markdown"] = self._generate_technical_visualizations_markdown(
                visualization_paths
            )
        
        # Generate the technical report using the base class method
        return super().generate_report(
            validation_results=validation_results,
            report_type=ReportType.TECHNICAL_REPORT,
            output_format=output_format,
            output_filename=output_filename,
            title=title,
            description=description,
            metadata=metadata,
            template_vars=template_vars,
            visualization_paths=visualization_paths,
            comparative_data=comparative_data,
            show_improvement=True
        )
    
    def _generate_technical_report_html(
        self,
        validation_results: Dict[str, Any],
        raw_data: Optional[Dict[str, Any]] = None,
        methodology_description: Optional[str] = None,
        statistical_notes: Optional[str] = None,
        limitations: Optional[List[str]] = None,
        appendices: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate HTML for technical report."""
        html = ""
        
        # Add methodology section if include_methodology is True
        if self.include_methodology:
            html += "<div class='section methodology'>"
            html += "<h2>Methodology</h2>"
            
            if methodology_description:
                html += f"<div class='subsection'>{methodology_description}</div>"
            else:
                html += self._get_default_methodology_html()
            
            html += "</div>"
        
        # Add detailed validation results if include_detailed_validation is True
        if self.include_detailed_validation:
            html += "<div class='section detailed-validation'>"
            html += "<h2>Detailed Validation Results</h2>"
            
            # Overall results
            overall = validation_results.get("overall", {})
            if overall:
                html += "<div class='subsection'>"
                html += "<h3>Overall Statistics</h3>"
                html += "<table class='statistics-table'>"
                html += "<tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th>"
                html += f"<th>{int(self.confidence_level * 100)}% CI Lower</th>"
                html += f"<th>{int(self.confidence_level * 100)}% CI Upper</th></tr>"
                
                # Add rows for each metric
                for metric in self.detailed_metrics:
                    if metric in overall:
                        metric_data = overall[metric]
                        html += f"<tr><td>{metric.upper()}</td>"
                        html += f"<td>{metric_data.get('mean', 'N/A')}</td>"
                        html += f"<td>{metric_data.get('std_dev', 'N/A')}</td>"
                        html += f"<td>{metric_data.get('min', 'N/A')}</td>"
                        html += f"<td>{metric_data.get('max', 'N/A')}</td>"
                        
                        # Add confidence intervals if available
                        if "ci_lower" in metric_data and "ci_upper" in metric_data:
                            html += f"<td>{metric_data['ci_lower']}</td>"
                            html += f"<td>{metric_data['ci_upper']}</td>"
                        else:
                            # Calculate approximate CI if not provided
                            mean = metric_data.get('mean', 0)
                            std_dev = metric_data.get('std_dev', 0)
                            sample_size = metric_data.get('sample_size', 30)  # Default to 30 if not provided
                            
                            # z-score for the given confidence level
                            import math
                            z_score = 1.96  # Approximation for 95% CI
                            
                            if self.confidence_level == 0.99:
                                z_score = 2.576
                            elif self.confidence_level == 0.90:
                                z_score = 1.645
                            
                            # Calculate CI
                            margin = z_score * std_dev / math.sqrt(sample_size)
                            ci_lower = mean - margin
                            ci_upper = mean + margin
                            
                            html += f"<td>{ci_lower:.6f}</td>"
                            html += f"<td>{ci_upper:.6f}</td>"
                        
                        html += "</tr>"
                
                html += "</table>"
                html += "</div>"
            
            # Hardware results
            hardware_results = validation_results.get("hardware_results", {})
            if hardware_results:
                html += "<div class='subsection'>"
                html += "<h3>Hardware-Specific Results</h3>"
                
                for hw_id, hw_data in hardware_results.items():
                    html += f"<h4>Hardware: {hw_id}</h4>"
                    html += "<table class='statistics-table'>"
                    html += "<tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>Status</th></tr>"
                    
                    for metric in self.detailed_metrics:
                        if metric in hw_data:
                            metric_data = hw_data[metric]
                            html += f"<tr><td>{metric.upper()}</td>"
                            html += f"<td>{metric_data.get('mean', 'N/A')}</td>"
                            html += f"<td>{metric_data.get('std_dev', 'N/A')}</td>"
                            
                            # Add status if available
                            if "status" in hw_data:
                                status_class = "positive" if hw_data["status"] == "pass" else "negative"
                                html += f"<td class='{status_class}'>{hw_data['status'].upper()}</td>"
                            else:
                                html += "<td>N/A</td>"
                            
                            html += "</tr>"
                    
                    html += "</table>"
                
                html += "</div>"
            
            # Model results
            model_results = validation_results.get("model_results", {})
            if model_results:
                html += "<div class='subsection'>"
                html += "<h3>Model-Specific Results</h3>"
                
                for model_id, model_data in model_results.items():
                    html += f"<h4>Model: {model_id}</h4>"
                    html += "<table class='statistics-table'>"
                    html += "<tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>Status</th></tr>"
                    
                    for metric in self.detailed_metrics:
                        if metric in model_data:
                            metric_data = model_data[metric]
                            html += f"<tr><td>{metric.upper()}</td>"
                            html += f"<td>{metric_data.get('mean', 'N/A')}</td>"
                            html += f"<td>{metric_data.get('std_dev', 'N/A')}</td>"
                            
                            # Add status if available
                            if "status" in model_data:
                                status_class = "positive" if model_data["status"] == "pass" else "negative"
                                html += f"<td class='{status_class}'>{model_data['status'].upper()}</td>"
                            else:
                                html += "<td>N/A</td>"
                            
                            html += "</tr>"
                    
                    html += "</table>"
                
                html += "</div>"
            
            html += "</div>"
        
        # Add statistical analysis if include_statistical_analysis is True
        if self.include_statistical_analysis:
            html += "<div class='section statistical-analysis'>"
            html += "<h2>Statistical Analysis</h2>"
            
            if statistical_notes:
                html += f"<div class='subsection'>{statistical_notes}</div>"
            else:
                html += self._get_default_statistical_analysis_html()
            
            html += self._generate_detailed_statistics_html(validation_results)
            
            html += "</div>"
        
        # Add raw data if include_raw_data is True and raw_data is provided
        if self.include_raw_data and raw_data:
            html += "<div class='section raw-data'>"
            html += "<h2>Raw Data</h2>"
            
            # Add raw data tables
            for data_name, data in raw_data.items():
                html += f"<h3>{data_name}</h3>"
                
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    # Table data
                    html += "<table class='raw-data-table'>"
                    
                    # Add headers
                    html += "<tr>"
                    for header in data[0].keys():
                        html += f"<th>{header}</th>"
                    html += "</tr>"
                    
                    # Add rows
                    for row in data:
                        html += "<tr>"
                        for value in row.values():
                            html += f"<td>{value}</td>"
                        html += "</tr>"
                    
                    html += "</table>"
                else:
                    # JSON data
                    html += f"<pre>{json.dumps(data, indent=2)}</pre>"
            
            html += "</div>"
        
        # Add limitations if provided
        if limitations:
            html += "<div class='section limitations'>"
            html += "<h2>Limitations and Caveats</h2>"
            html += "<ul>"
            
            for limitation in limitations:
                html += f"<li>{limitation}</li>"
            
            html += "</ul>"
            html += "</div>"
        
        # Add appendices if provided
        if appendices:
            html += "<div class='section appendices'>"
            html += "<h2>Appendices</h2>"
            
            for appendix_name, appendix_content in appendices.items():
                html += f"<h3>{appendix_name}</h3>"
                html += f"<div>{appendix_content}</div>"
            
            html += "</div>"
        
        return html
    
    def _generate_technical_report_markdown(
        self,
        validation_results: Dict[str, Any],
        raw_data: Optional[Dict[str, Any]] = None,
        methodology_description: Optional[str] = None,
        statistical_notes: Optional[str] = None,
        limitations: Optional[List[str]] = None,
        appendices: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate Markdown for technical report."""
        markdown = ""
        
        # Add methodology section if include_methodology is True
        if self.include_methodology:
            markdown += "## Methodology\n\n"
            
            if methodology_description:
                markdown += f"{methodology_description}\n\n"
            else:
                markdown += self._get_default_methodology_markdown()
            
            markdown += "\n"
        
        # Add detailed validation results if include_detailed_validation is True
        if self.include_detailed_validation:
            markdown += "## Detailed Validation Results\n\n"
            
            # Overall results
            overall = validation_results.get("overall", {})
            if overall:
                markdown += "### Overall Statistics\n\n"
                markdown += "| Metric | Mean | Std Dev | Min | Max "
                markdown += f"| {int(self.confidence_level * 100)}% CI Lower "
                markdown += f"| {int(self.confidence_level * 100)}% CI Upper |\n"
                markdown += "| ------ | ---- | ------- | --- | --- | ------- | ------- |\n"
                
                # Add rows for each metric
                for metric in self.detailed_metrics:
                    if metric in overall:
                        metric_data = overall[metric]
                        markdown += f"| {metric.upper()} "
                        markdown += f"| {metric_data.get('mean', 'N/A')} "
                        markdown += f"| {metric_data.get('std_dev', 'N/A')} "
                        markdown += f"| {metric_data.get('min', 'N/A')} "
                        markdown += f"| {metric_data.get('max', 'N/A')} "
                        
                        # Add confidence intervals if available
                        if "ci_lower" in metric_data and "ci_upper" in metric_data:
                            markdown += f"| {metric_data['ci_lower']} "
                            markdown += f"| {metric_data['ci_upper']} |"
                        else:
                            # Calculate approximate CI if not provided
                            mean = metric_data.get('mean', 0)
                            std_dev = metric_data.get('std_dev', 0)
                            sample_size = metric_data.get('sample_size', 30)  # Default to 30 if not provided
                            
                            # z-score for the given confidence level
                            import math
                            z_score = 1.96  # Approximation for 95% CI
                            
                            if self.confidence_level == 0.99:
                                z_score = 2.576
                            elif self.confidence_level == 0.90:
                                z_score = 1.645
                            
                            # Calculate CI
                            margin = z_score * std_dev / math.sqrt(sample_size)
                            ci_lower = mean - margin
                            ci_upper = mean + margin
                            
                            markdown += f"| {ci_lower:.6f} "
                            markdown += f"| {ci_upper:.6f} |"
                        
                        markdown += "\n"
                
                markdown += "\n"
            
            # Hardware results
            hardware_results = validation_results.get("hardware_results", {})
            if hardware_results:
                markdown += "### Hardware-Specific Results\n\n"
                
                for hw_id, hw_data in hardware_results.items():
                    markdown += f"#### Hardware: {hw_id}\n\n"
                    markdown += "| Metric | Mean | Std Dev | Status |\n"
                    markdown += "| ------ | ---- | ------- | ------ |\n"
                    
                    for metric in self.detailed_metrics:
                        if metric in hw_data:
                            metric_data = hw_data[metric]
                            markdown += f"| {metric.upper()} "
                            markdown += f"| {metric_data.get('mean', 'N/A')} "
                            markdown += f"| {metric_data.get('std_dev', 'N/A')} "
                            
                            # Add status if available
                            if "status" in hw_data:
                                status_symbol = "✅" if hw_data["status"] == "pass" else "❌"
                                markdown += f"| {status_symbol} {hw_data['status'].upper()} |"
                            else:
                                markdown += "| N/A |"
                            
                            markdown += "\n"
                    
                    markdown += "\n"
            
            # Model results
            model_results = validation_results.get("model_results", {})
            if model_results:
                markdown += "### Model-Specific Results\n\n"
                
                for model_id, model_data in model_results.items():
                    markdown += f"#### Model: {model_id}\n\n"
                    markdown += "| Metric | Mean | Std Dev | Status |\n"
                    markdown += "| ------ | ---- | ------- | ------ |\n"
                    
                    for metric in self.detailed_metrics:
                        if metric in model_data:
                            metric_data = model_data[metric]
                            markdown += f"| {metric.upper()} "
                            markdown += f"| {metric_data.get('mean', 'N/A')} "
                            markdown += f"| {metric_data.get('std_dev', 'N/A')} "
                            
                            # Add status if available
                            if "status" in model_data:
                                status_symbol = "✅" if model_data["status"] == "pass" else "❌"
                                markdown += f"| {status_symbol} {model_data['status'].upper()} |"
                            else:
                                markdown += "| N/A |"
                            
                            markdown += "\n"
                    
                    markdown += "\n"
        
        # Add statistical analysis if include_statistical_analysis is True
        if self.include_statistical_analysis:
            markdown += "## Statistical Analysis\n\n"
            
            if statistical_notes:
                markdown += f"{statistical_notes}\n\n"
            else:
                markdown += self._get_default_statistical_analysis_markdown()
            
            markdown += self._generate_detailed_statistics_markdown(validation_results)
            
            markdown += "\n"
        
        # Add raw data if include_raw_data is True and raw_data is provided
        if self.include_raw_data and raw_data:
            markdown += "## Raw Data\n\n"
            
            # Add raw data tables
            for data_name, data in raw_data.items():
                markdown += f"### {data_name}\n\n"
                
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    # Table data
                    # Add headers
                    markdown += "| "
                    for header in data[0].keys():
                        markdown += f"{header} | "
                    markdown += "\n"
                    
                    # Add separator
                    markdown += "| "
                    for _ in range(len(data[0].keys())):
                        markdown += "--- | "
                    markdown += "\n"
                    
                    # Add rows
                    for row in data:
                        markdown += "| "
                        for value in row.values():
                            markdown += f"{value} | "
                        markdown += "\n"
                else:
                    # JSON data
                    markdown += "```json\n"
                    markdown += json.dumps(data, indent=2)
                    markdown += "\n```\n"
                
                markdown += "\n"
        
        # Add limitations if provided
        if limitations:
            markdown += "## Limitations and Caveats\n\n"
            
            for limitation in limitations:
                markdown += f"- {limitation}\n"
            
            markdown += "\n"
        
        # Add appendices if provided
        if appendices:
            markdown += "## Appendices\n\n"
            
            for appendix_name, appendix_content in appendices.items():
                markdown += f"### {appendix_name}\n\n"
                markdown += f"{appendix_content}\n\n"
        
        return markdown
    
    def _generate_technical_report_text(
        self,
        validation_results: Dict[str, Any],
        raw_data: Optional[Dict[str, Any]] = None,
        methodology_description: Optional[str] = None,
        statistical_notes: Optional[str] = None,
        limitations: Optional[List[str]] = None,
        appendices: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate text for technical report."""
        text = ""
        
        # Add methodology section if include_methodology is True
        if self.include_methodology:
            text += "METHODOLOGY\n"
            text += "===========\n\n"
            
            if methodology_description:
                text += f"{methodology_description}\n\n"
            else:
                text += self._get_default_methodology_text()
            
            text += "\n"
        
        # Add detailed validation results if include_detailed_validation is True
        if self.include_detailed_validation:
            text += "DETAILED VALIDATION RESULTS\n"
            text += "===========================\n\n"
            
            # Overall results
            overall = validation_results.get("overall", {})
            if overall:
                text += "Overall Statistics:\n"
                text += "------------------\n\n"
                
                # Add rows for each metric
                for metric in self.detailed_metrics:
                    if metric in overall:
                        metric_data = overall[metric]
                        text += f"{metric.upper()}:\n"
                        text += f"  Mean: {metric_data.get('mean', 'N/A')}\n"
                        text += f"  Std Dev: {metric_data.get('std_dev', 'N/A')}\n"
                        text += f"  Min: {metric_data.get('min', 'N/A')}\n"
                        text += f"  Max: {metric_data.get('max', 'N/A')}\n"
                        
                        # Add confidence intervals if available
                        if "ci_lower" in metric_data and "ci_upper" in metric_data:
                            text += f"  {int(self.confidence_level * 100)}% CI: [{metric_data['ci_lower']}, {metric_data['ci_upper']}]\n"
                        
                        text += "\n"
            
            # Hardware results
            hardware_results = validation_results.get("hardware_results", {})
            if hardware_results:
                text += "Hardware-Specific Results:\n"
                text += "--------------------------\n\n"
                
                for hw_id, hw_data in hardware_results.items():
                    text += f"Hardware: {hw_id}\n"
                    text += "-" * (len(hw_id) + 10) + "\n"
                    
                    for metric in self.detailed_metrics:
                        if metric in hw_data:
                            metric_data = hw_data[metric]
                            text += f"{metric.upper()}:\n"
                            text += f"  Mean: {metric_data.get('mean', 'N/A')}\n"
                            text += f"  Std Dev: {metric_data.get('std_dev', 'N/A')}\n"
                            
                            # Add status if available
                            if "status" in hw_data:
                                text += f"  Status: {hw_data['status'].upper()}\n"
                            
                            text += "\n"
                    
                    text += "\n"
            
            # Model results
            model_results = validation_results.get("model_results", {})
            if model_results:
                text += "Model-Specific Results:\n"
                text += "----------------------\n\n"
                
                for model_id, model_data in model_results.items():
                    text += f"Model: {model_id}\n"
                    text += "-" * (len(model_id) + 7) + "\n"
                    
                    for metric in self.detailed_metrics:
                        if metric in model_data:
                            metric_data = model_data[metric]
                            text += f"{metric.upper()}:\n"
                            text += f"  Mean: {metric_data.get('mean', 'N/A')}\n"
                            text += f"  Std Dev: {metric_data.get('std_dev', 'N/A')}\n"
                            
                            # Add status if available
                            if "status" in model_data:
                                text += f"  Status: {model_data['status'].upper()}\n"
                            
                            text += "\n"
                    
                    text += "\n"
        
        # Add statistical analysis if include_statistical_analysis is True
        if self.include_statistical_analysis:
            text += "STATISTICAL ANALYSIS\n"
            text += "====================\n\n"
            
            if statistical_notes:
                text += f"{statistical_notes}\n\n"
            else:
                text += self._get_default_statistical_analysis_text()
            
            text += self._generate_detailed_statistics_text(validation_results)
            
            text += "\n"
        
        # Add raw data if include_raw_data is True and raw_data is provided
        if self.include_raw_data and raw_data:
            text += "RAW DATA\n"
            text += "========\n\n"
            
            # Add raw data tables
            for data_name, data in raw_data.items():
                text += f"{data_name}:\n"
                text += "-" * len(data_name) + "\n\n"
                
                # Just summarize the data in text format
                if isinstance(data, list):
                    text += f"[List with {len(data)} items]\n\n"
                elif isinstance(data, dict):
                    text += f"[Dictionary with {len(data)} keys]\n\n"
                else:
                    text += f"{data}\n\n"
        
        # Add limitations if provided
        if limitations:
            text += "LIMITATIONS AND CAVEATS\n"
            text += "=======================\n\n"
            
            for i, limitation in enumerate(limitations, 1):
                text += f"{i}. {limitation}\n"
            
            text += "\n"
        
        # Add appendices if provided
        if appendices:
            text += "APPENDICES\n"
            text += "==========\n\n"
            
            for appendix_name, appendix_content in appendices.items():
                text += f"{appendix_name}:\n"
                text += "-" * len(appendix_name) + "\n\n"
                text += f"{appendix_content}\n\n"
        
        return text
    
    def _generate_technical_visualizations_html(
        self,
        visualization_paths: Dict[str, str]
    ) -> str:
        """Generate HTML for technical visualizations section."""
        html = "<div class='technical-visualizations'>"
        html += "<h2>Visualization Analysis</h2>"
        
        for viz_name, viz_path in visualization_paths.items():
            if os.path.exists(viz_path):
                html += "<div class='viz-container'>"
                html += f"<h3>{viz_name}</h3>"
                
                # Determine visualization type based on file extension
                ext = os.path.splitext(viz_path)[1].lower()
                
                if ext in ['.png', '.jpg', '.jpeg', '.gif']:
                    # Image file
                    html += f"<img src='{viz_path}' alt='{viz_name}' class='technical-viz' />"
                elif ext == '.html':
                    # HTML visualization (likely interactive)
                    html += f"<iframe src='{viz_path}' width='100%' height='600px' frameborder='0'></iframe>"
                elif ext == '.svg':
                    # SVG file
                    html += f"<img src='{viz_path}' alt='{viz_name}' class='technical-viz' />"
                else:
                    # Unknown type
                    html += f"<p>Visualization available at: {viz_path}</p>"
                
                # Add technical interpretation
                html += "<div class='viz-interpretation'>"
                html += f"<h4>Technical Interpretation: {viz_name}</h4>"
                html += f"<p>{self._generate_visualization_interpretation(viz_name)}</p>"
                html += "</div>"
                
                html += "</div>"
        
        html += "</div>"
        return html
    
    def _generate_technical_visualizations_markdown(
        self,
        visualization_paths: Dict[str, str]
    ) -> str:
        """Generate Markdown for technical visualizations section."""
        markdown = "## Visualization Analysis\n\n"
        
        for viz_name, viz_path in visualization_paths.items():
            if os.path.exists(viz_path):
                markdown += f"### {viz_name}\n\n"
                
                # Determine visualization type based on file extension
                ext = os.path.splitext(viz_path)[1].lower()
                
                if ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                    # Image file
                    markdown += f"![{viz_name}]({viz_path})\n\n"
                else:
                    # Non-image file
                    markdown += f"Visualization available at: [{viz_path}]({viz_path})\n\n"
                
                # Add technical interpretation
                markdown += f"**Technical Interpretation: {viz_name}**\n\n"
                markdown += f"{self._generate_visualization_interpretation(viz_name)}\n\n"
        
        return markdown
    
    def _generate_visualization_interpretation(self, viz_name: str) -> str:
        """
        Generate a technical interpretation for a visualization.
        
        This is a placeholder method that would be replaced with actual
        interpretation logic in a real implementation. Currently, it
        returns a generic interpretation based on the visualization name.
        """
        if "error" in viz_name.lower():
            return "This visualization shows the error distribution across different configurations. Areas with higher error values may indicate potential issues with calibration or model assumptions."
        elif "distribution" in viz_name.lower():
            return "This visualization displays the statistical distribution of values, allowing for analysis of central tendency, spread, and potential outliers."
        elif "comparison" in viz_name.lower():
            return "This comparative visualization highlights differences between configurations or versions, enabling direct assessment of relative performance and improvements."
        elif "correlation" in viz_name.lower():
            return "This correlation analysis visualizes relationships between different metrics, helping identify dependencies and potential causal factors affecting simulation accuracy."
        elif "time" in viz_name.lower() or "series" in viz_name.lower():
            return "This time series visualization tracks changes over time, enabling trend analysis and identification of temporal patterns or drift in accuracy metrics."
        else:
            return "This technical visualization provides detailed insights into the validation results, helping to identify patterns, outliers, and potential areas for further investigation."
    
    def _generate_detailed_statistics(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed statistics from validation results.
        
        This method extracts and computes additional statistical measures
        beyond those directly available in the validation results.
        """
        statistics = {
            "summary": {},
            "distributions": {},
            "correlations": {},
            "significance_tests": {}
        }
        
        # Extract overall statistics
        overall = validation_results.get("overall", {})
        if overall:
            statistics["summary"]["overall"] = {
                "metrics": {},
                "sample_size": overall.get("sample_size", 0),
                "pass_rate": overall.get("pass_rate", 0.0)
            }
            
            # Add metrics
            for metric in self.detailed_metrics:
                if metric in overall:
                    statistics["summary"]["overall"]["metrics"][metric] = overall[metric]
        
        # Extract hardware statistics
        hardware_results = validation_results.get("hardware_results", {})
        if hardware_results:
            statistics["summary"]["hardware"] = {}
            
            for hw_id, hw_data in hardware_results.items():
                statistics["summary"]["hardware"][hw_id] = {
                    "metrics": {},
                    "sample_size": hw_data.get("sample_size", 0),
                    "pass_rate": hw_data.get("pass_rate", 0.0)
                }
                
                # Add metrics
                for metric in self.detailed_metrics:
                    if metric in hw_data:
                        statistics["summary"]["hardware"][hw_id]["metrics"][metric] = hw_data[metric]
        
        # Extract model statistics
        model_results = validation_results.get("model_results", {})
        if model_results:
            statistics["summary"]["model"] = {}
            
            for model_id, model_data in model_results.items():
                statistics["summary"]["model"][model_id] = {
                    "metrics": {},
                    "sample_size": model_data.get("sample_size", 0),
                    "pass_rate": model_data.get("pass_rate", 0.0)
                }
                
                # Add metrics
                for metric in self.detailed_metrics:
                    if metric in model_data:
                        statistics["summary"]["model"][model_id]["metrics"][metric] = model_data[metric]
        
        # Add distribution information if available
        if "distributions" in validation_results:
            statistics["distributions"] = validation_results["distributions"]
        
        # Add correlation information if available
        if "correlations" in validation_results:
            statistics["correlations"] = validation_results["correlations"]
        
        # Add significance test results if available
        if "significance_tests" in validation_results:
            statistics["significance_tests"] = validation_results["significance_tests"]
        
        return statistics
    
    def _generate_detailed_statistics_html(self, validation_results: Dict[str, Any]) -> str:
        """Generate HTML for detailed statistics."""
        statistics = self._generate_detailed_statistics(validation_results)
        
        html = "<div class='detailed-statistics'>"
        
        # Add distribution analysis if available
        if statistics["distributions"]:
            html += "<div class='subsection'>"
            html += "<h3>Distribution Analysis</h3>"
            
            for dist_name, dist_data in statistics["distributions"].items():
                html += f"<h4>{dist_name}</h4>"
                
                if "histogram" in dist_data:
                    # Histogram data
                    hist = dist_data["histogram"]
                    html += "<div class='histogram-container'>"
                    
                    # Simple ASCII histogram
                    max_count = max(hist["counts"])
                    bar_width = 100  # Max percentage width
                    
                    for i, (bin_start, count) in enumerate(zip(hist["bins"], hist["counts"])):
                        bin_end = hist["bins"][i + 1] if i + 1 < len(hist["bins"]) else None
                        bin_label = f"{bin_start:.2f} - {bin_end:.2f}" if bin_end else f"{bin_start:.2f}+"
                        
                        percent_width = (count / max_count) * bar_width
                        
                        html += f"<div class='histogram-row'>"
                        html += f"<span class='bin-label'>{bin_label}</span>"
                        html += f"<div class='histogram-bar' style='width: {percent_width}%'></div>"
                        html += f"<span class='bin-count'>{count}</span>"
                        html += f"</div>"
                    
                    html += "</div>"
                
                if "statistics" in dist_data:
                    # Distribution statistics
                    stats = dist_data["statistics"]
                    html += "<table class='statistics-table'>"
                    html += "<tr><th>Statistic</th><th>Value</th></tr>"
                    
                    for stat_name, stat_value in stats.items():
                        html += f"<tr><td>{stat_name}</td><td>{stat_value}</td></tr>"
                    
                    html += "</table>"
            
            html += "</div>"
        
        # Add correlation analysis if available
        if statistics["correlations"]:
            html += "<div class='subsection'>"
            html += "<h3>Correlation Analysis</h3>"
            
            for corr_name, corr_data in statistics["correlations"].items():
                html += f"<h4>{corr_name}</h4>"
                
                if isinstance(corr_data, dict) and "matrix" in corr_data:
                    # Correlation matrix
                    matrix = corr_data["matrix"]
                    variables = corr_data.get("variables", [f"Var{i+1}" for i in range(len(matrix))])
                    
                    html += "<table class='correlation-matrix'>"
                    
                    # Header row
                    html += "<tr><th></th>"
                    for var in variables:
                        html += f"<th>{var}</th>"
                    html += "</tr>"
                    
                    # Data rows
                    for i, row in enumerate(matrix):
                        html += f"<tr><th>{variables[i]}</th>"
                        
                        for val in row:
                            # Color-code correlation values
                            if val > 0.7:
                                cell_class = "high-positive"
                            elif val > 0.3:
                                cell_class = "medium-positive"
                            elif val > -0.3:
                                cell_class = "neutral"
                            elif val > -0.7:
                                cell_class = "medium-negative"
                            else:
                                cell_class = "high-negative"
                            
                            html += f"<td class='{cell_class}'>{val:.3f}</td>"
                        
                        html += "</tr>"
                    
                    html += "</table>"
                else:
                    # Simple correlation data
                    html += "<table class='statistics-table'>"
                    html += "<tr><th>Variable 1</th><th>Variable 2</th><th>Correlation</th></tr>"
                    
                    for corr in corr_data:
                        html += f"<tr><td>{corr['var1']}</td><td>{corr['var2']}</td><td>{corr['value']:.3f}</td></tr>"
                    
                    html += "</table>"
            
            html += "</div>"
        
        # Add significance test results if available
        if statistics["significance_tests"]:
            html += "<div class='subsection'>"
            html += "<h3>Statistical Significance Tests</h3>"
            
            for test_name, test_data in statistics["significance_tests"].items():
                html += f"<h4>{test_name}</h4>"
                
                html += "<table class='statistics-table'>"
                html += "<tr><th>Parameter</th><th>Value</th></tr>"
                
                for param_name, param_value in test_data.items():
                    if param_name == "is_significant":
                        result_class = "positive" if param_value else "negative"
                        html += f"<tr><td>Is Significant</td><td class='{result_class}'>{param_value}</td></tr>"
                    else:
                        html += f"<tr><td>{param_name}</td><td>{param_value}</td></tr>"
                
                html += "</table>"
            
            html += "</div>"
        
        html += "</div>"
        return html
    
    def _generate_detailed_statistics_markdown(self, validation_results: Dict[str, Any]) -> str:
        """Generate Markdown for detailed statistics."""
        statistics = self._generate_detailed_statistics(validation_results)
        
        markdown = ""
        
        # Add distribution analysis if available
        if statistics["distributions"]:
            markdown += "### Distribution Analysis\n\n"
            
            for dist_name, dist_data in statistics["distributions"].items():
                markdown += f"#### {dist_name}\n\n"
                
                if "statistics" in dist_data:
                    # Distribution statistics
                    stats = dist_data["statistics"]
                    markdown += "| Statistic | Value |\n"
                    markdown += "| --------- | ----- |\n"
                    
                    for stat_name, stat_value in stats.items():
                        markdown += f"| {stat_name} | {stat_value} |\n"
                    
                    markdown += "\n"
            
            markdown += "\n"
        
        # Add correlation analysis if available
        if statistics["correlations"]:
            markdown += "### Correlation Analysis\n\n"
            
            for corr_name, corr_data in statistics["correlations"].items():
                markdown += f"#### {corr_name}\n\n"
                
                if isinstance(corr_data, dict) and "matrix" in corr_data:
                    # Correlation matrix
                    matrix = corr_data["matrix"]
                    variables = corr_data.get("variables", [f"Var{i+1}" for i in range(len(matrix))])
                    
                    # Header row
                    markdown += "| | "
                    for var in variables:
                        markdown += f"{var} | "
                    markdown += "\n"
                    
                    # Separator row
                    markdown += "| --- | "
                    for _ in variables:
                        markdown += "--- | "
                    markdown += "\n"
                    
                    # Data rows
                    for i, row in enumerate(matrix):
                        markdown += f"| **{variables[i]}** | "
                        
                        for val in row:
                            markdown += f"{val:.3f} | "
                        
                        markdown += "\n"
                    
                    markdown += "\n"
                else:
                    # Simple correlation data
                    markdown += "| Variable 1 | Variable 2 | Correlation |\n"
                    markdown += "| ---------- | ---------- | ----------- |\n"
                    
                    for corr in corr_data:
                        markdown += f"| {corr['var1']} | {corr['var2']} | {corr['value']:.3f} |\n"
                    
                    markdown += "\n"
            
            markdown += "\n"
        
        # Add significance test results if available
        if statistics["significance_tests"]:
            markdown += "### Statistical Significance Tests\n\n"
            
            for test_name, test_data in statistics["significance_tests"].items():
                markdown += f"#### {test_name}\n\n"
                
                markdown += "| Parameter | Value |\n"
                markdown += "| --------- | ----- |\n"
                
                for param_name, param_value in test_data.items():
                    if param_name == "is_significant":
                        symbol = "✅" if param_value else "❌"
                        markdown += f"| Is Significant | {symbol} {param_value} |\n"
                    else:
                        markdown += f"| {param_name} | {param_value} |\n"
                
                markdown += "\n"
            
            markdown += "\n"
        
        return markdown
    
    def _generate_detailed_statistics_text(self, validation_results: Dict[str, Any]) -> str:
        """Generate text for detailed statistics."""
        statistics = self._generate_detailed_statistics(validation_results)
        
        text = ""
        
        # Add distribution analysis if available
        if statistics["distributions"]:
            text += "Distribution Analysis:\n"
            text += "----------------------\n\n"
            
            for dist_name, dist_data in statistics["distributions"].items():
                text += f"{dist_name}:\n"
                text += "-" * len(dist_name) + "\n\n"
                
                if "statistics" in dist_data:
                    # Distribution statistics
                    stats = dist_data["statistics"]
                    
                    for stat_name, stat_value in stats.items():
                        text += f"{stat_name}: {stat_value}\n"
                    
                    text += "\n"
            
            text += "\n"
        
        # Add correlation analysis if available
        if statistics["correlations"]:
            text += "Correlation Analysis:\n"
            text += "---------------------\n\n"
            
            for corr_name, corr_data in statistics["correlations"].items():
                text += f"{corr_name}:\n"
                text += "-" * len(corr_name) + "\n\n"
                
                if isinstance(corr_data, dict) and "matrix" in corr_data:
                    # Correlation matrix
                    matrix = corr_data["matrix"]
                    variables = corr_data.get("variables", [f"Var{i+1}" for i in range(len(matrix))])
                    
                    # Just mention matrix dimensions in text format
                    text += f"Correlation matrix of dimension {len(matrix)}x{len(matrix)}\n"
                    text += f"Variables: {', '.join(variables)}\n\n"
                else:
                    # Simple correlation data
                    for corr in corr_data:
                        text += f"{corr['var1']} - {corr['var2']}: {corr['value']:.3f}\n"
                    
                    text += "\n"
            
            text += "\n"
        
        # Add significance test results if available
        if statistics["significance_tests"]:
            text += "Statistical Significance Tests:\n"
            text += "-------------------------------\n\n"
            
            for test_name, test_data in statistics["significance_tests"].items():
                text += f"{test_name}:\n"
                text += "-" * len(test_name) + "\n\n"
                
                for param_name, param_value in test_data.items():
                    text += f"{param_name}: {param_value}\n"
                
                text += "\n"
            
            text += "\n"
        
        return text
    
    def _get_default_methodology_html(self) -> str:
        """Get default HTML for methodology section."""
        return """
        <p>The simulation validation methodology follows a standardized process to assess the accuracy of
        simulation results compared to real hardware measurements. The validation process consists of the
        following steps:</p>
        
        <ol>
            <li><strong>Data Collection</strong>: Performance metrics are collected from both simulation and hardware
            under controlled conditions.</li>
            <li><strong>Data Alignment</strong>: Simulation and hardware results are aligned to ensure fair comparison.</li>
            <li><strong>Metric Calculation</strong>: Error metrics including MAPE, RMSE, and correlation coefficients are calculated.</li>
            <li><strong>Statistical Analysis</strong>: Comprehensive statistical analysis is performed to assess significance and reliability.</li>
            <li><strong>Validation Assessment</strong>: Results are evaluated against established thresholds to determine validation status.</li>
        </ol>
        
        <p>Validation is performed across multiple hardware platforms and model types to ensure comprehensive coverage.
        The validation status is determined based on a combination of error metrics and statistical significance tests.</p>
        """
    
    def _get_default_methodology_markdown(self) -> str:
        """Get default Markdown for methodology section."""
        return """
The simulation validation methodology follows a standardized process to assess the accuracy of
simulation results compared to real hardware measurements. The validation process consists of the
following steps:

1. **Data Collection**: Performance metrics are collected from both simulation and hardware
   under controlled conditions.
2. **Data Alignment**: Simulation and hardware results are aligned to ensure fair comparison.
3. **Metric Calculation**: Error metrics including MAPE, RMSE, and correlation coefficients are calculated.
4. **Statistical Analysis**: Comprehensive statistical analysis is performed to assess significance and reliability.
5. **Validation Assessment**: Results are evaluated against established thresholds to determine validation status.

Validation is performed across multiple hardware platforms and model types to ensure comprehensive coverage.
The validation status is determined based on a combination of error metrics and statistical significance tests.
"""
    
    def _get_default_methodology_text(self) -> str:
        """Get default text for methodology section."""
        return """
The simulation validation methodology follows a standardized process to assess the accuracy of
simulation results compared to real hardware measurements. The validation process consists of the
following steps:

1. Data Collection: Performance metrics are collected from both simulation and hardware
   under controlled conditions.
2. Data Alignment: Simulation and hardware results are aligned to ensure fair comparison.
3. Metric Calculation: Error metrics including MAPE, RMSE, and correlation coefficients are calculated.
4. Statistical Analysis: Comprehensive statistical analysis is performed to assess significance and reliability.
5. Validation Assessment: Results are evaluated against established thresholds to determine validation status.

Validation is performed across multiple hardware platforms and model types to ensure comprehensive coverage.
The validation status is determined based on a combination of error metrics and statistical significance tests.
"""
    
    def _get_default_statistical_analysis_html(self) -> str:
        """Get default HTML for statistical analysis section."""
        return """
        <p>The statistical analysis of validation results includes multiple metrics to provide a comprehensive
        assessment of simulation accuracy:</p>
        
        <ul>
            <li><strong>Mean Absolute Percentage Error (MAPE)</strong>: Measures the average percentage error between
            simulation and hardware results, providing an intuitive measure of accuracy.</li>
            <li><strong>Root Mean Square Error (RMSE)</strong>: Provides a measure of absolute error, giving more
            weight to larger errors due to the squaring operation.</li>
            <li><strong>Correlation Coefficient</strong>: Assesses the linear relationship between simulation and
            hardware results, indicating how well the simulation captures trends.</li>
            <li><strong>Ranking Preservation</strong>: Measures how well the simulation preserves the relative
            ranking of different configurations, which is crucial for decision-making.</li>
        </ul>
        
        <p>Confidence intervals are calculated using bootstrap resampling to provide robust estimates of
        uncertainty. Statistical significance tests are performed to determine if the observed differences
        between simulation and hardware results are statistically significant.</p>
        """
    
    def _get_default_statistical_analysis_markdown(self) -> str:
        """Get default Markdown for statistical analysis section."""
        return """
The statistical analysis of validation results includes multiple metrics to provide a comprehensive
assessment of simulation accuracy:

- **Mean Absolute Percentage Error (MAPE)**: Measures the average percentage error between
  simulation and hardware results, providing an intuitive measure of accuracy.
- **Root Mean Square Error (RMSE)**: Provides a measure of absolute error, giving more
  weight to larger errors due to the squaring operation.
- **Correlation Coefficient**: Assesses the linear relationship between simulation and
  hardware results, indicating how well the simulation captures trends.
- **Ranking Preservation**: Measures how well the simulation preserves the relative
  ranking of different configurations, which is crucial for decision-making.

Confidence intervals are calculated using bootstrap resampling to provide robust estimates of
uncertainty. Statistical significance tests are performed to determine if the observed differences
between simulation and hardware results are statistically significant.
"""
    
    def _get_default_statistical_analysis_text(self) -> str:
        """Get default text for statistical analysis section."""
        return """
The statistical analysis of validation results includes multiple metrics to provide a comprehensive
assessment of simulation accuracy:

- Mean Absolute Percentage Error (MAPE): Measures the average percentage error between
  simulation and hardware results, providing an intuitive measure of accuracy.
- Root Mean Square Error (RMSE): Provides a measure of absolute error, giving more
  weight to larger errors due to the squaring operation.
- Correlation Coefficient: Assesses the linear relationship between simulation and
  hardware results, indicating how well the simulation captures trends.
- Ranking Preservation: Measures how well the simulation preserves the relative
  ranking of different configurations, which is crucial for decision-making.

Confidence intervals are calculated using bootstrap resampling to provide robust estimates of
uncertainty. Statistical significance tests are performed to determine if the observed differences
between simulation and hardware results are statistically significant.
"""

# Example usage
if __name__ == "__main__":
    # Create a technical report generator
    generator = TechnicalReportGenerator(
        output_dir="reports",
        include_statistical_analysis=True,
        include_raw_data=True,
        confidence_level=0.95
    )
    
    # Create sample validation results
    validation_results = {
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
        },
        "distributions": {
            "MAPE Distribution": {
                "histogram": {
                    "bins": [0, 2, 4, 6, 8, 10],
                    "counts": [5, 18, 15, 8, 4]
                },
                "statistics": {
                    "skewness": 0.45,
                    "kurtosis": 2.3,
                    "shapiro_p_value": 0.12
                }
            }
        },
        "correlations": {
            "Metric Correlations": {
                "matrix": [
                    [1.0, 0.8, 0.6],
                    [0.8, 1.0, 0.7],
                    [0.6, 0.7, 1.0]
                ],
                "variables": ["MAPE", "RMSE", "Correlation"]
            }
        },
        "significance_tests": {
            "Simulation vs Hardware t-test": {
                "t_statistic": 1.45,
                "p_value": 0.15,
                "degrees_of_freedom": 49,
                "is_significant": False
            }
        }
    }
    
    # Sample raw data
    raw_data = {
        "Simulation vs Hardware Results": [
            {"hardware": "rtx3080", "model": "bert-base-uncased", "metric": "throughput", "simulation": 120.5, "hardware": 115.2, "error": 4.6},
            {"hardware": "rtx3080", "model": "vit-base-patch16-224", "metric": "throughput", "simulation": 85.3, "hardware": 82.1, "error": 3.9},
            {"hardware": "a100", "model": "bert-base-uncased", "metric": "throughput", "simulation": 180.2, "hardware": 175.8, "error": 2.5},
            {"hardware": "a100", "model": "vit-base-patch16-224", "metric": "throughput", "simulation": 130.6, "hardware": 126.3, "error": 3.4}
        ]
    }
    
    # Sample limitations
    limitations = [
        "The validation results are specific to the hardware configurations tested and may not generalize to all hardware variants.",
        "Extreme workloads outside the tested range may exhibit different error characteristics.",
        "The simulation model does not account for thermal throttling effects under sustained load.",
        "Network infrastructure variations were not tested and may affect distributed inference scenarios."
    ]
    
    # Generate technical report
    report = generator.generate_technical_report(
        validation_results=validation_results,
        output_format=ReportFormat.HTML,
        title="Simulation Validation Technical Report",
        description="Detailed technical analysis of simulation validation results",
        raw_data=raw_data,
        limitations=limitations
    )
    
    print(f"Technical report generated: {report['path']}")