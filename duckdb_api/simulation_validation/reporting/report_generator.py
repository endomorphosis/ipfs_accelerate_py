"""
Report Generator for Simulation Validation Results.

This module provides the base functionality for generating reports from
simulation validation results. It supports multiple output formats and
customization options.
"""

import os
import json
import datetime
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
import time
import uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simulation_validation.reporting")

class ReportFormat(Enum):
    """Supported report output formats."""
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"  # Requires optional dependencies
    JSON = "json"
    TEXT = "text"

class ReportType(Enum):
    """Types of reports that can be generated."""
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_REPORT = "technical_report"
    COMPARATIVE_REPORT = "comparative_report"
    COMPREHENSIVE_REPORT = "comprehensive_report"

class ReportGenerator:
    """
    Base class for generating reports from simulation validation results.
    
    This class provides the foundation for creating reports in multiple formats,
    with customization options and template support.
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
        custom_templates: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the report generator.
        
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
        """
        self.output_dir = output_dir
        self.templates_dir = templates_dir or os.path.join(
            os.path.dirname(__file__), "templates"
        )
        self.default_format = default_format
        self.include_visualizations = include_visualizations
        self.include_timestamps = include_timestamps
        self.include_version_info = include_version_info
        self.custom_css = custom_css
        self.custom_logo = custom_logo
        self.company_name = company_name
        self.custom_templates = custom_templates or {}
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Ensure templates directory exists
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir, exist_ok=True)
            logger.warning(f"Templates directory {self.templates_dir} created.")
            
        # Initialize report history for tracking
        self.report_history = []
        
        logger.info(f"Report generator initialized with output directory: {output_dir}")
    
    def generate_report(
        self,
        validation_results: Dict[str, Any],
        report_type: ReportType = ReportType.COMPREHENSIVE_REPORT,
        output_format: Optional[ReportFormat] = None,
        output_filename: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        template_vars: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        sections: Optional[List[str]] = None,
        visualization_paths: Optional[Dict[str, str]] = None,
        comparative_data: Optional[Dict[str, Any]] = None,
        show_improvement: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a report based on validation results.
        
        Args:
            validation_results: The validation results to include in the report
            report_type: Type of report to generate
            output_format: Output format for the report
            output_filename: Filename for the generated report
            title: Report title
            description: Report description
            metadata: Additional metadata to include in the report
            template_vars: Variables to pass to the template
            version: Report version
            sections: Specific sections to include in the report
            visualization_paths: Paths to visualization files to include
            comparative_data: Data for comparative analysis
            show_improvement: Whether to show improvement metrics
        
        Returns:
            Dictionary with information about the generated report
        """
        # Use default format if not specified
        format_to_use = output_format or self.default_format
        
        # Generate default filename if not provided
        if not output_filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            format_extension = self._get_format_extension(format_to_use)
            report_type_str = report_type.value
            output_filename = f"{report_type_str}_{timestamp}.{format_extension}"
        
        # Ensure file has correct extension
        if not self._has_correct_extension(output_filename, format_to_use):
            format_extension = self._get_format_extension(format_to_use)
            output_filename = f"{os.path.splitext(output_filename)[0]}.{format_extension}"
        
        # Full path to output file
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Prepare report metadata
        report_id = str(uuid.uuid4())
        report_metadata = self._prepare_metadata(
            title=title,
            description=description,
            metadata=metadata,
            version=version,
            report_type=report_type,
            output_format=format_to_use
        )
        
        # Load template for the report type and format
        template = self._load_template(report_type, format_to_use)
        
        # Prepare template variables
        variables = self._prepare_template_variables(
            validation_results=validation_results,
            report_type=report_type,
            metadata=report_metadata,
            template_vars=template_vars,
            sections=sections,
            visualization_paths=visualization_paths,
            comparative_data=comparative_data,
            show_improvement=show_improvement
        )
        
        # Render template
        rendered_content = self._render_template(template, variables)
        
        # Process rendered content based on format
        processed_content = self._process_content_for_format(
            rendered_content, format_to_use, output_path
        )
        
        # Write report to file
        self._write_report(processed_content, output_path)
        
        # Create report entry for history
        report_entry = {
            "id": report_id,
            "type": report_type.value,
            "format": format_to_use.value,
            "path": output_path,
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": report_metadata,
            "filename": output_filename
        }
        
        # Add to report history
        self.report_history.append(report_entry)
        
        logger.info(f"Generated {report_type.value} report: {output_path}")
        
        return report_entry
    
    def _get_format_extension(self, format_type: ReportFormat) -> str:
        """Get the file extension for a format type."""
        extensions = {
            ReportFormat.HTML: "html",
            ReportFormat.MARKDOWN: "md",
            ReportFormat.PDF: "pdf",
            ReportFormat.JSON: "json",
            ReportFormat.TEXT: "txt"
        }
        return extensions.get(format_type, "html")
    
    def _has_correct_extension(self, filename: str, format_type: ReportFormat) -> bool:
        """Check if filename has the correct extension for the format."""
        expected_ext = self._get_format_extension(format_type)
        file_ext = os.path.splitext(filename)[1].lstrip('.')
        return file_ext.lower() == expected_ext.lower()
    
    def _prepare_metadata(
        self,
        title: Optional[str],
        description: Optional[str],
        metadata: Optional[Dict[str, Any]],
        version: Optional[str],
        report_type: ReportType,
        output_format: ReportFormat
    ) -> Dict[str, Any]:
        """Prepare metadata for the report."""
        # Default title if not provided
        if not title:
            title = f"Simulation Validation {report_type.value.replace('_', ' ').title()}"
        
        # Prepare metadata dict
        result = {
            "title": title,
            "description": description or "Simulation validation report",
            "report_type": report_type.value,
            "format": output_format.value,
            "version": version or "1.0.0",
            "generated_at": datetime.datetime.now().isoformat(),
            "generator": "Simulation Validation Reporting Framework"
        }
        
        # Add company name if provided
        if self.company_name:
            result["company"] = self.company_name
        
        # Add custom metadata
        if metadata:
            result.update(metadata)
        
        return result
    
    def _load_template(self, report_type: ReportType, format_type: ReportFormat) -> str:
        """
        Load the template for a specific report type and format.
        
        First checks for custom template, then falls back to default.
        """
        # Check for custom template
        template_key = f"{report_type.value}_{format_type.value}"
        if template_key in self.custom_templates:
            template_path = self.custom_templates[template_key]
            if os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    return f.read()
            logger.warning(f"Custom template {template_path} not found.")
        
        # Default template path
        default_template_path = os.path.join(
            self.templates_dir,
            f"{report_type.value}_{format_type.value}.template"
        )
        
        # Check if default template exists
        if os.path.exists(default_template_path):
            with open(default_template_path, 'r') as f:
                return f.read()
        
        # If template doesn't exist, use a default template based on format
        logger.warning(f"Template {default_template_path} not found. Using default template.")
        
        if format_type == ReportFormat.HTML:
            return self._get_default_html_template(report_type)
        elif format_type == ReportFormat.MARKDOWN:
            return self._get_default_markdown_template(report_type)
        elif format_type == ReportFormat.JSON:
            return "{{content_json}}"
        elif format_type == ReportFormat.TEXT:
            return self._get_default_text_template(report_type)
        else:
            logger.error(f"No default template available for format {format_type.value}")
            return "No template available"
    
    def _get_default_html_template(self, report_type: ReportType) -> str:
        """Get a default HTML template for a report type."""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{metadata.title}}</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .header { border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 20px; }
        .footer { border-top: 1px solid #eee; margin-top: 30px; padding-top: 10px; font-size: 0.8em; color: #777; }
        .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .visualization { margin: 20px 0; text-align: center; }
        .visualization img { max-width: 100%; height: auto; }
        .improvement-positive { color: green; font-weight: bold; }
        .improvement-negative { color: red; font-weight: bold; }
        .metadata { font-size: 0.8em; color: #666; margin-top: 5px; }
        .section { margin-bottom: 30px; }
        {{custom_css}}
    </style>
</head>
<body>
    <div class="header">
        <h1>{{metadata.title}}</h1>
        <div class="metadata">
            <p>Generated: {{metadata.generated_at}}</p>
            <p>Version: {{metadata.version}}</p>
            {% if metadata.company %}
            <p>Company: {{metadata.company}}</p>
            {% endif %}
        </div>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        {{summary_html}}
    </div>
    
    <div class="section">
        <h2>Validation Results</h2>
        {{validation_results_html}}
    </div>
    
    {% if visualizations %}
    <div class="section">
        <h2>Visualizations</h2>
        {% for viz in visualizations %}
        <div class="visualization">
            <h3>{{viz.title}}</h3>
            <img src="{{viz.path}}" alt="{{viz.title}}" />
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    {% if comparative_data %}
    <div class="section">
        <h2>Comparative Analysis</h2>
        {{comparative_analysis_html}}
    </div>
    {% endif %}
    
    <div class="footer">
        <p>Generated by Simulation Validation Reporting Framework</p>
        <p>Report ID: {{metadata.report_id}}</p>
    </div>
</body>
</html>"""
    
    def _get_default_markdown_template(self, report_type: ReportType) -> str:
        """Get a default Markdown template for a report type."""
        return """# {{metadata.title}}

*Generated: {{metadata.generated_at}}*  
*Version: {{metadata.version}}*  
{% if metadata.company %}*Company: {{metadata.company}}*{% endif %}

## Summary

{{summary_markdown}}

## Validation Results

{{validation_results_markdown}}

{% if visualizations %}
## Visualizations

{% for viz in visualizations %}
### {{viz.title}}

![{{viz.title}}]({{viz.path}})

{% endfor %}
{% endif %}

{% if comparative_data %}
## Comparative Analysis

{{comparative_analysis_markdown}}
{% endif %}

---

Generated by Simulation Validation Reporting Framework  
Report ID: {{metadata.report_id}}"""
    
    def _get_default_text_template(self, report_type: ReportType) -> str:
        """Get a default text template for a report type."""
        return """{{metadata.title}}
Generated: {{metadata.generated_at}}
Version: {{metadata.version}}
{% if metadata.company %}Company: {{metadata.company}}{% endif %}

====== SUMMARY ======

{{summary_text}}

====== VALIDATION RESULTS ======

{{validation_results_text}}

{% if comparative_data %}
====== COMPARATIVE ANALYSIS ======

{{comparative_analysis_text}}
{% endif %}

---
Generated by Simulation Validation Reporting Framework
Report ID: {{metadata.report_id}}"""
    
    def _prepare_template_variables(
        self,
        validation_results: Dict[str, Any],
        report_type: ReportType,
        metadata: Dict[str, Any],
        template_vars: Optional[Dict[str, Any]] = None,
        sections: Optional[List[str]] = None,
        visualization_paths: Optional[Dict[str, str]] = None,
        comparative_data: Optional[Dict[str, Any]] = None,
        show_improvement: bool = True
    ) -> Dict[str, Any]:
        """Prepare variables for template rendering."""
        # Initialize variables dict with metadata
        variables = {
            "metadata": metadata,
            "report_type": report_type.value,
            "validation_results": validation_results
        }
        
        # Add custom CSS if provided
        if self.custom_css and os.path.exists(self.custom_css):
            with open(self.custom_css, 'r') as f:
                variables["custom_css"] = f.read()
        else:
            variables["custom_css"] = ""
        
        # Add custom logo if provided
        if self.custom_logo and os.path.exists(self.custom_logo):
            variables["custom_logo"] = self.custom_logo
        
        # Add template variables if provided
        if template_vars:
            variables.update(template_vars)
        
        # Add visualizations if provided and include_visualizations is True
        if visualization_paths and self.include_visualizations:
            visualizations = []
            for viz_name, viz_path in visualization_paths.items():
                if os.path.exists(viz_path):
                    visualizations.append({
                        "title": viz_name,
                        "path": viz_path
                    })
            variables["visualizations"] = visualizations
        else:
            variables["visualizations"] = []
        
        # Add comparative data if provided
        if comparative_data:
            variables["comparative_data"] = comparative_data
            variables["show_improvement"] = show_improvement
        
        # Add sections if provided
        if sections:
            variables["sections"] = sections
        
        # Add content for different formats
        variables["content_json"] = json.dumps({
            "metadata": metadata,
            "validation_results": validation_results,
            "comparative_data": comparative_data if comparative_data else None
        }, indent=2)
        
        # Generate HTML, Markdown, and text summaries
        variables["summary_html"] = self._generate_summary_html(validation_results)
        variables["summary_markdown"] = self._generate_summary_markdown(validation_results)
        variables["summary_text"] = self._generate_summary_text(validation_results)
        
        # Generate HTML, Markdown, and text validation results
        variables["validation_results_html"] = self._generate_validation_results_html(validation_results)
        variables["validation_results_markdown"] = self._generate_validation_results_markdown(validation_results)
        variables["validation_results_text"] = self._generate_validation_results_text(validation_results)
        
        # Generate comparative analysis if provided
        if comparative_data:
            variables["comparative_analysis_html"] = self._generate_comparative_analysis_html(
                comparative_data, show_improvement
            )
            variables["comparative_analysis_markdown"] = self._generate_comparative_analysis_markdown(
                comparative_data, show_improvement
            )
            variables["comparative_analysis_text"] = self._generate_comparative_analysis_text(
                comparative_data, show_improvement
            )
        
        return variables
    
    def _render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """
        Render a template with variables.
        
        This implementation uses a simple string format-based approach.
        For a more robust solution, consider using a template engine like Jinja2.
        """
        # Simple template rendering - replace variable placeholders
        result = template
        
        # Replace placeholders with corresponding values
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            if isinstance(value, str) and placeholder in result:
                result = result.replace(placeholder, value)
        
        return result
    
    def _process_content_for_format(
        self, 
        content: str, 
        format_type: ReportFormat,
        output_path: str
    ) -> Union[str, bytes]:
        """Process rendered content based on output format."""
        if format_type == ReportFormat.PDF:
            try:
                # PDF generation requires optional dependencies
                import weasyprint
                
                # Convert content to PDF
                pdf_bytes = weasyprint.HTML(string=content).write_pdf()
                return pdf_bytes
            except ImportError:
                logger.error("WeasyPrint not installed. Cannot generate PDF.")
                # Fallback to HTML
                return content
        
        # For other formats, return as is
        return content
    
    def _write_report(self, content: Union[str, bytes], output_path: str) -> None:
        """Write report content to file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write content to file
        if isinstance(content, bytes):
            with open(output_path, 'wb') as f:
                f.write(content)
        else:
            with open(output_path, 'w') as f:
                f.write(content)
    
    def _generate_summary_html(self, validation_results: Dict[str, Any]) -> str:
        """Generate HTML summary from validation results."""
        # Extract relevant information from validation results
        overall = validation_results.get("overall", {})
        
        html = "<table>"
        html += "<tr><th>Metric</th><th>Value</th></tr>"
        
        if "mape" in overall:
            html += f"<tr><td>MAPE</td><td>{overall['mape']['mean']:.2f}%</td></tr>"
        
        if "rmse" in overall:
            html += f"<tr><td>RMSE</td><td>{overall['rmse']['mean']:.4f}</td></tr>"
        
        if "status" in overall:
            status_class = "improvement-positive" if overall["status"] == "pass" else "improvement-negative"
            html += f"<tr><td>Status</td><td class='{status_class}'>{overall['status'].upper()}</td></tr>"
        
        html += "</table>"
        
        return html
    
    def _generate_summary_markdown(self, validation_results: Dict[str, Any]) -> str:
        """Generate Markdown summary from validation results."""
        # Extract relevant information from validation results
        overall = validation_results.get("overall", {})
        
        markdown = "| Metric | Value |\n| ------ | ----- |\n"
        
        if "mape" in overall:
            markdown += f"| MAPE | {overall['mape']['mean']:.2f}% |\n"
        
        if "rmse" in overall:
            markdown += f"| RMSE | {overall['rmse']['mean']:.4f} |\n"
        
        if "status" in overall:
            status_symbol = "✅" if overall["status"] == "pass" else "❌"
            markdown += f"| Status | {status_symbol} {overall['status'].upper()} |\n"
        
        return markdown
    
    def _generate_summary_text(self, validation_results: Dict[str, Any]) -> str:
        """Generate text summary from validation results."""
        # Extract relevant information from validation results
        overall = validation_results.get("overall", {})
        
        text = ""
        
        if "mape" in overall:
            text += f"MAPE: {overall['mape']['mean']:.2f}%\n"
        
        if "rmse" in overall:
            text += f"RMSE: {overall['rmse']['mean']:.4f}\n"
        
        if "status" in overall:
            text += f"Status: {overall['status'].upper()}\n"
        
        return text
    
    def _generate_validation_results_html(self, validation_results: Dict[str, Any]) -> str:
        """Generate HTML for validation results."""
        html = ""
        
        # Add overall stats
        overall = validation_results.get("overall", {})
        if overall:
            html += "<h3>Overall Statistics</h3>"
            html += "<table>"
            html += "<tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th></tr>"
            
            metrics = ["mape", "rmse", "correlation"]
            for metric in metrics:
                if metric in overall:
                    metric_data = overall[metric]
                    html += f"<tr><td>{metric.upper()}</td>"
                    html += f"<td>{metric_data.get('mean', 'N/A')}</td>"
                    html += f"<td>{metric_data.get('std_dev', 'N/A')}</td>"
                    html += f"<td>{metric_data.get('min', 'N/A')}</td>"
                    html += f"<td>{metric_data.get('max', 'N/A')}</td></tr>"
            
            html += "</table>"
        
        # Add hardware stats
        hardware_results = validation_results.get("hardware_results", {})
        if hardware_results:
            html += "<h3>Hardware Results</h3>"
            html += "<table>"
            html += "<tr><th>Hardware</th><th>MAPE</th><th>RMSE</th><th>Status</th></tr>"
            
            for hw_id, hw_data in hardware_results.items():
                status_class = "improvement-positive" if hw_data.get("status") == "pass" else "improvement-negative"
                html += f"<tr><td>{hw_id}</td>"
                html += f"<td>{hw_data.get('mape', {}).get('mean', 'N/A'):.2f}%</td>"
                html += f"<td>{hw_data.get('rmse', {}).get('mean', 'N/A'):.4f}</td>"
                html += f"<td class='{status_class}'>{hw_data.get('status', 'N/A').upper()}</td></tr>"
            
            html += "</table>"
        
        # Add model stats
        model_results = validation_results.get("model_results", {})
        if model_results:
            html += "<h3>Model Results</h3>"
            html += "<table>"
            html += "<tr><th>Model</th><th>MAPE</th><th>RMSE</th><th>Status</th></tr>"
            
            for model_id, model_data in model_results.items():
                status_class = "improvement-positive" if model_data.get("status") == "pass" else "improvement-negative"
                html += f"<tr><td>{model_id}</td>"
                html += f"<td>{model_data.get('mape', {}).get('mean', 'N/A'):.2f}%</td>"
                html += f"<td>{model_data.get('rmse', {}).get('mean', 'N/A'):.4f}</td>"
                html += f"<td class='{status_class}'>{model_data.get('status', 'N/A').upper()}</td></tr>"
            
            html += "</table>"
        
        return html
    
    def _generate_validation_results_markdown(self, validation_results: Dict[str, Any]) -> str:
        """Generate Markdown for validation results."""
        markdown = ""
        
        # Add overall stats
        overall = validation_results.get("overall", {})
        if overall:
            markdown += "### Overall Statistics\n\n"
            markdown += "| Metric | Mean | Std Dev | Min | Max |\n"
            markdown += "| ------ | ---- | ------- | --- | --- |\n"
            
            metrics = ["mape", "rmse", "correlation"]
            for metric in metrics:
                if metric in overall:
                    metric_data = overall[metric]
                    markdown += f"| {metric.upper()} | {metric_data.get('mean', 'N/A')} | "
                    markdown += f"{metric_data.get('std_dev', 'N/A')} | "
                    markdown += f"{metric_data.get('min', 'N/A')} | "
                    markdown += f"{metric_data.get('max', 'N/A')} |\n"
            
            markdown += "\n"
        
        # Add hardware stats
        hardware_results = validation_results.get("hardware_results", {})
        if hardware_results:
            markdown += "### Hardware Results\n\n"
            markdown += "| Hardware | MAPE | RMSE | Status |\n"
            markdown += "| -------- | ---- | ---- | ------ |\n"
            
            for hw_id, hw_data in hardware_results.items():
                status_symbol = "✅" if hw_data.get("status") == "pass" else "❌"
                markdown += f"| {hw_id} | "
                markdown += f"{hw_data.get('mape', {}).get('mean', 'N/A'):.2f}% | "
                markdown += f"{hw_data.get('rmse', {}).get('mean', 'N/A'):.4f} | "
                markdown += f"{status_symbol} {hw_data.get('status', 'N/A').upper()} |\n"
            
            markdown += "\n"
        
        # Add model stats
        model_results = validation_results.get("model_results", {})
        if model_results:
            markdown += "### Model Results\n\n"
            markdown += "| Model | MAPE | RMSE | Status |\n"
            markdown += "| ----- | ---- | ---- | ------ |\n"
            
            for model_id, model_data in model_results.items():
                status_symbol = "✅" if model_data.get("status") == "pass" else "❌"
                markdown += f"| {model_id} | "
                markdown += f"{model_data.get('mape', {}).get('mean', 'N/A'):.2f}% | "
                markdown += f"{model_data.get('rmse', {}).get('mean', 'N/A'):.4f} | "
                markdown += f"{status_symbol} {model_data.get('status', 'N/A').upper()} |\n"
            
            markdown += "\n"
        
        return markdown
    
    def _generate_validation_results_text(self, validation_results: Dict[str, Any]) -> str:
        """Generate text for validation results."""
        text = ""
        
        # Add overall stats
        overall = validation_results.get("overall", {})
        if overall:
            text += "Overall Statistics:\n"
            text += "-----------------\n"
            
            metrics = ["mape", "rmse", "correlation"]
            for metric in metrics:
                if metric in overall:
                    metric_data = overall[metric]
                    text += f"{metric.upper()}:\n"
                    text += f"  Mean: {metric_data.get('mean', 'N/A')}\n"
                    text += f"  Std Dev: {metric_data.get('std_dev', 'N/A')}\n"
                    text += f"  Min: {metric_data.get('min', 'N/A')}\n"
                    text += f"  Max: {metric_data.get('max', 'N/A')}\n"
            
            text += "\n"
        
        # Add hardware stats
        hardware_results = validation_results.get("hardware_results", {})
        if hardware_results:
            text += "Hardware Results:\n"
            text += "----------------\n"
            
            for hw_id, hw_data in hardware_results.items():
                text += f"{hw_id}:\n"
                text += f"  MAPE: {hw_data.get('mape', {}).get('mean', 'N/A'):.2f}%\n"
                text += f"  RMSE: {hw_data.get('rmse', {}).get('mean', 'N/A'):.4f}\n"
                text += f"  Status: {hw_data.get('status', 'N/A').upper()}\n"
                text += "\n"
        
        # Add model stats
        model_results = validation_results.get("model_results", {})
        if model_results:
            text += "Model Results:\n"
            text += "-------------\n"
            
            for model_id, model_data in model_results.items():
                text += f"{model_id}:\n"
                text += f"  MAPE: {model_data.get('mape', {}).get('mean', 'N/A'):.2f}%\n"
                text += f"  RMSE: {model_data.get('rmse', {}).get('mean', 'N/A'):.4f}\n"
                text += f"  Status: {model_data.get('status', 'N/A').upper()}\n"
                text += "\n"
        
        return text
    
    def _generate_comparative_analysis_html(
        self, 
        comparative_data: Dict[str, Any],
        show_improvement: bool = True
    ) -> str:
        """Generate HTML for comparative analysis."""
        html = ""
        
        # Add comparison table
        if "comparison" in comparative_data:
            comparison = comparative_data["comparison"]
            
            html += "<table>"
            html += "<tr><th>Metric</th><th>Before</th><th>After</th>"
            if show_improvement:
                html += "<th>Improvement</th>"
            html += "</tr>"
            
            for metric, data in comparison.items():
                before = data.get("before", "N/A")
                after = data.get("after", "N/A")
                
                html += f"<tr><td>{metric}</td><td>{before}</td><td>{after}</td>"
                
                if show_improvement and "improvement" in data:
                    improvement = data["improvement"]
                    improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
                    html += f"<td class='{improvement_class}'>{improvement:.2f}%</td>"
                
                html += "</tr>"
            
            html += "</table>"
        
        # Add version information
        if "versions" in comparative_data:
            versions = comparative_data["versions"]
            
            html += "<h3>Version Information</h3>"
            html += "<table>"
            html += "<tr><th>Version</th><th>Date</th><th>Description</th></tr>"
            
            for version in versions:
                html += f"<tr><td>{version.get('version', 'N/A')}</td>"
                html += f"<td>{version.get('date', 'N/A')}</td>"
                html += f"<td>{version.get('description', 'N/A')}</td></tr>"
            
            html += "</table>"
        
        return html
    
    def _generate_comparative_analysis_markdown(
        self, 
        comparative_data: Dict[str, Any],
        show_improvement: bool = True
    ) -> str:
        """Generate Markdown for comparative analysis."""
        markdown = ""
        
        # Add comparison table
        if "comparison" in comparative_data:
            comparison = comparative_data["comparison"]
            
            markdown += "| Metric | Before | After |"
            if show_improvement:
                markdown += " Improvement |"
            markdown += "\n"
            
            markdown += "| ------ | ------ | ----- |"
            if show_improvement:
                markdown += " ----------- |"
            markdown += "\n"
            
            for metric, data in comparison.items():
                before = data.get("before", "N/A")
                after = data.get("after", "N/A")
                
                markdown += f"| {metric} | {before} | {after} |"
                
                if show_improvement and "improvement" in data:
                    improvement = data["improvement"]
                    symbol = "🔼" if improvement > 0 else "🔽"
                    markdown += f" {symbol} {improvement:.2f}% |"
                
                markdown += "\n"
            
            markdown += "\n"
        
        # Add version information
        if "versions" in comparative_data:
            versions = comparative_data["versions"]
            
            markdown += "### Version Information\n\n"
            markdown += "| Version | Date | Description |\n"
            markdown += "| ------- | ---- | ----------- |\n"
            
            for version in versions:
                markdown += f"| {version.get('version', 'N/A')} | "
                markdown += f"{version.get('date', 'N/A')} | "
                markdown += f"{version.get('description', 'N/A')} |\n"
            
            markdown += "\n"
        
        return markdown
    
    def _generate_comparative_analysis_text(
        self, 
        comparative_data: Dict[str, Any],
        show_improvement: bool = True
    ) -> str:
        """Generate text for comparative analysis."""
        text = ""
        
        # Add comparison table
        if "comparison" in comparative_data:
            comparison = comparative_data["comparison"]
            
            text += "Metric Comparison:\n"
            text += "-----------------\n"
            
            for metric, data in comparison.items():
                before = data.get("before", "N/A")
                after = data.get("after", "N/A")
                
                text += f"{metric}:\n"
                text += f"  Before: {before}\n"
                text += f"  After: {after}\n"
                
                if show_improvement and "improvement" in data:
                    improvement = data["improvement"]
                    text += f"  Improvement: {improvement:.2f}%\n"
                
                text += "\n"
        
        # Add version information
        if "versions" in comparative_data:
            versions = comparative_data["versions"]
            
            text += "Version Information:\n"
            text += "-------------------\n"
            
            for version in versions:
                text += f"Version: {version.get('version', 'N/A')}\n"
                text += f"Date: {version.get('date', 'N/A')}\n"
                text += f"Description: {version.get('description', 'N/A')}\n"
                text += "\n"
        
        return text
    
    def get_reports(
        self,
        report_type: Optional[ReportType] = None,
        format_type: Optional[ReportFormat] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get report history filtered by various criteria.
        
        Args:
            report_type: Filter by report type
            format_type: Filter by format type
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            limit: Maximum number of reports to return
            
        Returns:
            List of report entries matching the criteria
        """
        # Convert dates to datetime objects if provided
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.datetime.fromisoformat(start_date)
            except ValueError:
                logger.warning(f"Invalid start date format: {start_date}")
        
        if end_date:
            try:
                end_datetime = datetime.datetime.fromisoformat(end_date)
            except ValueError:
                logger.warning(f"Invalid end date format: {end_date}")
        
        # Filter reports
        filtered_reports = []
        for report in self.report_history:
            # Apply report type filter
            if report_type and report["type"] != report_type.value:
                continue
            
            # Apply format type filter
            if format_type and report["format"] != format_type.value:
                continue
            
            # Apply date filters
            if start_datetime or end_datetime:
                try:
                    report_datetime = datetime.datetime.fromisoformat(report["timestamp"])
                    
                    if start_datetime and report_datetime < start_datetime:
                        continue
                    
                    if end_datetime and report_datetime > end_datetime:
                        continue
                except (ValueError, KeyError):
                    logger.warning(f"Invalid timestamp in report: {report.get('timestamp')}")
                    continue
            
            filtered_reports.append(report)
        
        # Sort by timestamp (newest first) and apply limit
        filtered_reports.sort(key=lambda x: x["timestamp"], reverse=True)
        return filtered_reports[:limit]
    
    def save_report_history(self, output_path: str) -> bool:
        """
        Save report history to a JSON file.
        
        Args:
            output_path: Path to save the history file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.report_history, f, indent=2)
            logger.info(f"Report history saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving report history: {e}")
            return False
    
    def load_report_history(self, input_path: str) -> bool:
        """
        Load report history from a JSON file.
        
        Args:
            input_path: Path to load history from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(input_path, 'r') as f:
                self.report_history = json.load(f)
            logger.info(f"Report history loaded from {input_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading report history: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Create a report generator
    generator = ReportGenerator(output_dir="reports")
    
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
    
    # Generate reports in different formats
    html_report = generator.generate_report(
        validation_results=validation_results,
        report_type=ReportType.COMPREHENSIVE_REPORT,
        output_format=ReportFormat.HTML,
        title="Simulation Validation Report",
        description="Comprehensive validation of simulation results"
    )
    
    markdown_report = generator.generate_report(
        validation_results=validation_results,
        report_type=ReportType.EXECUTIVE_SUMMARY,
        output_format=ReportFormat.MARKDOWN,
        title="Executive Summary",
        description="Summary of validation results for executive review"
    )
    
    # Print report paths
    print(f"HTML report generated: {html_report['path']}")
    print(f"Markdown report generated: {markdown_report['path']}")