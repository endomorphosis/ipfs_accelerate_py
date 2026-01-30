"""
Report Manager for Simulation Validation Framework.

This module provides functionality for managing report generation, scheduling,
versioning, archiving, and distribution.
"""

import os
import json
import datetime
import logging
import threading
import shutil
import zipfile
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import schedule

from .report_generator import ReportGenerator, ReportFormat, ReportType
from .executive_summary import ExecutiveSummaryGenerator
from .technical_report import TechnicalReportGenerator
from .comparative_report import ComparativeReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simulation_validation.reporting.manager")

class ReportManager:
    """
    Manages report generation, scheduling, versioning, archiving, and distribution.
    
    This class provides centralized management for all reporting-related activities
    in the Simulation Validation Framework, including:
    - Report generation (all report types)
    - Scheduled report generation
    - Report versioning and archiving
    - Report distribution (email, file system, etc.)
    - Report repositories and organization
    """
    
    def __init__(
        self,
        output_dir: str = "reports",
        archive_dir: str = "report_archives",
        template_dirs: Optional[Dict[str, str]] = None,
        default_format: ReportFormat = ReportFormat.HTML,
        include_visualizations: bool = True,
        include_timestamps: bool = True,
        include_version_info: bool = True,
        custom_css: Optional[str] = None,
        custom_logo: Optional[str] = None,
        company_name: Optional[str] = None,
        email_config: Optional[Dict[str, Any]] = None,
        auto_archive_days: int = 30,
        auto_archive_enabled: bool = True,
        report_index_file: str = "report_index.json",
        report_catalog_file: str = "report_catalog.html"
    ):
        """
        Initialize the report manager.
        
        Args:
            output_dir: Directory to store generated reports
            archive_dir: Directory to store archived reports
            template_dirs: Dict mapping report types to template directories
            default_format: Default report format
            include_visualizations: Whether to include visualizations in reports
            include_timestamps: Whether to include timestamps in reports
            include_version_info: Whether to include version info in reports
            custom_css: Path to custom CSS file for HTML reports
            custom_logo: Path to custom logo image for reports
            company_name: Company name to include in reports
            email_config: Email configuration for report distribution
            auto_archive_days: Number of days after which to archive reports
            auto_archive_enabled: Whether to enable automatic archiving
            report_index_file: Filename for the report index JSON file
            report_catalog_file: Filename for the HTML report catalog
        """
        self.output_dir = output_dir
        self.archive_dir = archive_dir
        self.template_dirs = template_dirs or {}
        self.default_format = default_format
        self.include_visualizations = include_visualizations
        self.include_timestamps = include_timestamps
        self.include_version_info = include_version_info
        self.custom_css = custom_css
        self.custom_logo = custom_logo
        self.company_name = company_name
        self.email_config = email_config
        self.auto_archive_days = auto_archive_days
        self.auto_archive_enabled = auto_archive_enabled
        self.report_index_file = report_index_file
        self.report_catalog_file = report_catalog_file
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
        
        # Initialize report generators
        self._init_report_generators()
        
        # Initialize report index
        self.report_index = self._load_report_index()
        
        # Initialize scheduler
        self.scheduler = schedule.Scheduler()
        self.schedule_thread = None
        self.schedule_stop_event = None
        self.scheduled_jobs = {}
        
        # Generate initial catalog if it doesn't exist
        catalog_path = os.path.join(self.output_dir, self.report_catalog_file)
        if not os.path.exists(catalog_path):
            self._generate_report_catalog()
        
        # Start auto-archiving if enabled
        if self.auto_archive_enabled:
            self._schedule_auto_archive()
        
        logger.info(f"Report manager initialized with output directory: {output_dir}")
    
    def _init_report_generators(self) -> None:
        """Initialize report generators for different report types."""
        # Create a standard report generator
        self.report_generator = ReportGenerator(
            output_dir=self.output_dir,
            templates_dir=self.template_dirs.get(ReportType.COMPREHENSIVE_REPORT.value),
            default_format=self.default_format,
            include_visualizations=self.include_visualizations,
            include_timestamps=self.include_timestamps,
            include_version_info=self.include_version_info,
            custom_css=self.custom_css,
            custom_logo=self.custom_logo,
            company_name=self.company_name
        )
        
        # Create an executive summary generator
        self.executive_summary_generator = ExecutiveSummaryGenerator(
            output_dir=self.output_dir,
            templates_dir=self.template_dirs.get(ReportType.EXECUTIVE_SUMMARY.value),
            default_format=self.default_format,
            include_visualizations=self.include_visualizations,
            include_timestamps=self.include_timestamps,
            include_version_info=self.include_version_info,
            custom_css=self.custom_css,
            custom_logo=self.custom_logo,
            company_name=self.company_name
        )
        
        # Create a technical report generator
        self.technical_report_generator = TechnicalReportGenerator(
            output_dir=self.output_dir,
            templates_dir=self.template_dirs.get(ReportType.TECHNICAL_REPORT.value),
            default_format=self.default_format,
            include_visualizations=self.include_visualizations,
            include_timestamps=self.include_timestamps,
            include_version_info=self.include_version_info,
            custom_css=self.custom_css,
            custom_logo=self.custom_logo,
            company_name=self.company_name
        )
        
        # Create a comparative report generator
        self.comparative_report_generator = ComparativeReportGenerator(
            output_dir=self.output_dir,
            templates_dir=self.template_dirs.get(ReportType.COMPARATIVE_REPORT.value),
            default_format=self.default_format,
            include_visualizations=self.include_visualizations,
            include_timestamps=self.include_timestamps,
            include_version_info=self.include_version_info,
            custom_css=self.custom_css,
            custom_logo=self.custom_logo,
            company_name=self.company_name
        )
    
    def _load_report_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load the report index from the index file.
        
        Returns:
            Dictionary containing report index data
        """
        index_path = os.path.join(self.output_dir, self.report_index_file)
        
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode report index file: {index_path}")
                return {
                    "reports": [],
                    "archives": [],
                    "last_updated": datetime.datetime.now().isoformat()
                }
        else:
            return {
                "reports": [],
                "archives": [],
                "last_updated": datetime.datetime.now().isoformat()
            }
    
    def _save_report_index(self) -> None:
        """Save the report index to the index file."""
        # Update last updated timestamp
        self.report_index["last_updated"] = datetime.datetime.now().isoformat()
        
        index_path = os.path.join(self.output_dir, self.report_index_file)
        
        try:
            with open(index_path, 'w') as f:
                json.dump(self.report_index, f, indent=2)
            logger.info(f"Saved report index to {index_path}")
        except Exception as e:
            logger.error(f"Failed to save report index: {e}")
    
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
        distribution_list: Optional[List[str]] = None,
        auto_open: bool = False
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
            distribution_list: List of email addresses to send the report to
            auto_open: Whether to automatically open the report after generation
        
        Returns:
            Dictionary with information about the generated report
        """
        # Choose the appropriate generator based on report type
        if report_type == ReportType.EXECUTIVE_SUMMARY:
            report_entry = self.executive_summary_generator.generate_executive_summary(
                validation_results=validation_results,
                output_format=output_format,
                output_filename=output_filename,
                title=title,
                description=description,
                metadata=metadata,
                visualization_paths=visualization_paths,
                comparative_data=comparative_data
            )
        elif report_type == ReportType.TECHNICAL_REPORT:
            report_entry = self.technical_report_generator.generate_technical_report(
                validation_results=validation_results,
                output_format=output_format,
                output_filename=output_filename,
                title=title,
                description=description,
                metadata=metadata,
                visualization_paths=visualization_paths,
                comparative_data=comparative_data
            )
        elif report_type == ReportType.COMPARATIVE_REPORT and comparative_data:
            # For comparative reports, we need to extract the "before" results
            # Note: This is a simplified approach; in a real implementation,
            # the ComparativeReportGenerator would have a different API
            if "validation_results_before" in comparative_data:
                report_entry = self.comparative_report_generator.generate_comparative_report(
                    validation_results_before=comparative_data["validation_results_before"],
                    validation_results_after=validation_results,
                    output_format=output_format,
                    output_filename=output_filename,
                    title=title,
                    description=description,
                    metadata=metadata,
                    visualization_paths=visualization_paths
                )
            else:
                logger.warning("Comparative report requested but no 'validation_results_before' provided")
                report_entry = self.report_generator.generate_report(
                    validation_results=validation_results,
                    report_type=report_type,
                    output_format=output_format,
                    output_filename=output_filename,
                    title=title,
                    description=description,
                    metadata=metadata,
                    template_vars=template_vars,
                    version=version,
                    sections=sections,
                    visualization_paths=visualization_paths,
                    comparative_data=comparative_data,
                    show_improvement=show_improvement
                )
        else:
            # Standard report
            report_entry = self.report_generator.generate_report(
                validation_results=validation_results,
                report_type=report_type,
                output_format=output_format,
                output_filename=output_filename,
                title=title,
                description=description,
                metadata=metadata,
                template_vars=template_vars,
                version=version,
                sections=sections,
                visualization_paths=visualization_paths,
                comparative_data=comparative_data,
                show_improvement=show_improvement
            )
        
        # Add the report to the index
        self._add_report_to_index(report_entry)
        
        # Update the report catalog
        self._generate_report_catalog()
        
        # Distribute the report if a distribution list is provided
        if distribution_list:
            self.distribute_report(report_entry, distribution_list)
        
        # Open the report if auto_open is True
        if auto_open:
            self._open_report(report_entry["path"])
        
        return report_entry
    
    def _add_report_to_index(self, report_entry: Dict[str, Any]) -> None:
        """
        Add a report to the index.
        
        Args:
            report_entry: Report entry to add to the index
        """
        # Create a copy of the report entry
        entry = dict(report_entry)
        
        # Add additional metadata
        entry["added_to_index"] = datetime.datetime.now().isoformat()
        entry["filename"] = os.path.basename(entry["path"])
        
        # Add the report to the index
        self.report_index["reports"].append(entry)
        
        # Save the index
        self._save_report_index()
        
        logger.info(f"Added report {entry['id']} to index")
    
    def _generate_report_catalog(self) -> str:
        """
        Generate an HTML catalog of all reports.
        
        Returns:
            Path to the generated catalog file
        """
        catalog_path = os.path.join(self.output_dir, self.report_catalog_file)
        
        # Create HTML content
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simulation Validation Report Catalog</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .header {{ border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 20px; }}
        .footer {{ border-top: 1px solid #eee; margin-top: 30px; padding-top: 10px; font-size: 0.8em; color: #777; }}
        .search-container {{ margin-bottom: 20px; }}
        .search-container input {{ padding: 8px; width: 300px; }}
        .filter-container {{ margin-bottom: 20px; }}
        .filter-container select {{ padding: 8px; width: 200px; }}
        .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
        .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }}
        .tab button:hover {{ background-color: #ddd; }}
        .tab button.active {{ background-color: #ccc; }}
        .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }}
        #Reports {{ display: block; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Simulation Validation Report Catalog</h1>
        <p>Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="tab">
        <button class="tablinks active" onclick="openTab(event, 'Reports')">Reports</button>
        <button class="tablinks" onclick="openTab(event, 'Archives')">Archives</button>
    </div>
    
    <div id="Reports" class="tabcontent">
        <div class="search-container">
            <input type="text" id="reportSearch" onkeyup="filterReports()" placeholder="Search reports...">
        </div>
        
        <div class="filter-container">
            <select id="reportTypeFilter" onchange="filterReports()">
                <option value="">All Report Types</option>
                <option value="comprehensive_report">Comprehensive</option>
                <option value="executive_summary">Executive Summary</option>
                <option value="technical_report">Technical</option>
                <option value="comparative_report">Comparative</option>
            </select>
            
            <select id="reportFormatFilter" onchange="filterReports()">
                <option value="">All Formats</option>
                <option value="html">HTML</option>
                <option value="markdown">Markdown</option>
                <option value="pdf">PDF</option>
                <option value="json">JSON</option>
            </select>
        </div>
        
        <table id="reportTable">
            <tr>
                <th>Title</th>
                <th>Type</th>
                <th>Format</th>
                <th>Generated</th>
                <th>Actions</th>
            </tr>
"""
        
        # Add reports to the table
        for report in self.report_index["reports"]:
            html += f"""
            <tr>
                <td>{report.get('metadata', {}).get('title', 'Untitled')}</td>
                <td>{report.get('type', 'Unknown')}</td>
                <td>{report.get('format', 'Unknown')}</td>
                <td>{report.get('timestamp', 'Unknown')}</td>
                <td>
                    <a href="{os.path.basename(report.get('path', '#'))}" target="_blank">View</a> | 
                    <a href="javascript:void(0)" onclick="archiveReport('{report.get('id', '')}')">Archive</a>
                </td>
            </tr>"""
        
        html += """
        </table>
    </div>
    
    <div id="Archives" class="tabcontent">
        <div class="search-container">
            <input type="text" id="archiveSearch" onkeyup="filterArchives()" placeholder="Search archives...">
        </div>
        
        <table id="archiveTable">
            <tr>
                <th>Title</th>
                <th>Type</th>
                <th>Archived</th>
                <th>Actions</th>
            </tr>
"""
        
        # Add archives to the table
        for archive in self.report_index["archives"]:
            html += f"""
            <tr>
                <td>{archive.get('title', 'Untitled')}</td>
                <td>{archive.get('type', 'Unknown')}</td>
                <td>{archive.get('archived_date', 'Unknown')}</td>
                <td>
                    <a href="{os.path.basename(archive.get('archive_path', '#'))}" download>Download</a> | 
                    <a href="javascript:void(0)" onclick="restoreArchive('{archive.get('id', '')}')">Restore</a>
                </td>
            </tr>"""
        
        html += """
        </table>
    </div>
    
    <div class="footer">
        <p>Generated by Simulation Validation Reporting Framework</p>
    </div>
    
    <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
        
        function filterReports() {
            var input, filter, table, tr, td, i, txtValue;
            input = document.getElementById("reportSearch");
            filter = input.value.toUpperCase();
            table = document.getElementById("reportTable");
            tr = table.getElementsByTagName("tr");
            
            var typeFilter = document.getElementById("reportTypeFilter").value;
            var formatFilter = document.getElementById("reportFormatFilter").value;
            
            for (i = 1; i < tr.length; i++) {
                var titleCell = tr[i].getElementsByTagName("td")[0];
                var typeCell = tr[i].getElementsByTagName("td")[1];
                var formatCell = tr[i].getElementsByTagName("td")[2];
                
                if (titleCell && typeCell && formatCell) {
                    txtValue = titleCell.textContent || titleCell.innerText;
                    var typeValue = typeCell.textContent || typeCell.innerText;
                    var formatValue = formatCell.textContent || formatCell.innerText;
                    
                    var matchesSearch = txtValue.toUpperCase().indexOf(filter) > -1;
                    var matchesType = typeFilter === "" || typeValue.toUpperCase().indexOf(typeFilter.toUpperCase()) > -1;
                    var matchesFormat = formatFilter === "" || formatValue.toUpperCase().indexOf(formatFilter.toUpperCase()) > -1;
                    
                    if (matchesSearch && matchesType && matchesFormat) {
                        tr[i].style.display = "";
                    } else {
                        tr[i].style.display = "none";
                    }
                }
            }
        }
        
        function filterArchives() {
            var input, filter, table, tr, td, i, txtValue;
            input = document.getElementById("archiveSearch");
            filter = input.value.toUpperCase();
            table = document.getElementById("archiveTable");
            tr = table.getElementsByTagName("tr");
            
            for (i = 1; i < tr.length; i++) {
                td = tr[i].getElementsByTagName("td")[0];
                if (td) {
                    txtValue = td.textContent || td.innerText;
                    if (txtValue.toUpperCase().indexOf(filter) > -1) {
                        tr[i].style.display = "";
                    } else {
                        tr[i].style.display = "none";
                    }
                }
            }
        }
        
        function archiveReport(reportId) {
            if (confirm("Are you sure you want to archive this report?")) {
                window.location.href = "archive_report.php?id=" + reportId;
            }
        }
        
        function restoreArchive(archiveId) {
            if (confirm("Are you sure you want to restore this archive?")) {
                window.location.href = "restore_archive.php?id=" + archiveId;
            }
        }
    </script>
</body>
</html>
"""
        
        # Write the HTML catalog
        with open(catalog_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Generated report catalog: {catalog_path}")
        
        return catalog_path
    
    def archive_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Archive a report by ID.
        
        Args:
            report_id: ID of the report to archive
            
        Returns:
            Dictionary with information about the archived report, or None if not found
        """
        # Find the report in the index
        report_entry = None
        report_index = None
        
        for i, report in enumerate(self.report_index["reports"]):
            if report.get("id") == report_id:
                report_entry = report
                report_index = i
                break
        
        if not report_entry:
            logger.warning(f"Report with ID {report_id} not found in index")
            return None
        
        # Create archive filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_type = report_entry.get("type", "unknown")
        report_format = report_entry.get("format", "unknown")
        archive_filename = f"{report_type}_{timestamp}.zip"
        archive_path = os.path.join(self.archive_dir, archive_filename)
        
        try:
            # Create a zip archive
            with zipfile.ZipFile(archive_path, 'w') as archive:
                # Add the report file
                report_file = report_entry.get("path")
                if os.path.exists(report_file):
                    archive.write(report_file, os.path.basename(report_file))
                else:
                    logger.warning(f"Report file not found: {report_file}")
                
                # Add any related files (visualizations, etc.)
                # In a real implementation, this would look for related files
                
                # Add metadata
                metadata = report_entry.get("metadata", {})
                metadata_str = json.dumps(metadata, indent=2)
                archive.writestr("metadata.json", metadata_str)
            
            # Create archive entry
            archive_entry = {
                "id": str(uuid.uuid4()),
                "original_id": report_id,
                "title": report_entry.get("metadata", {}).get("title", "Untitled"),
                "type": report_type,
                "format": report_format,
                "archive_path": archive_path,
                "archived_date": datetime.datetime.now().isoformat(),
                "metadata": metadata
            }
            
            # Add to archives list
            self.report_index["archives"].append(archive_entry)
            
            # Remove from reports list
            self.report_index["reports"].pop(report_index)
            
            # Save the index
            self._save_report_index()
            
            # Update the report catalog
            self._generate_report_catalog()
            
            logger.info(f"Archived report {report_id} to {archive_path}")
            
            return archive_entry
            
        except Exception as e:
            logger.error(f"Failed to archive report {report_id}: {e}")
            return None
    
    def restore_archive(self, archive_id: str) -> Optional[Dict[str, Any]]:
        """
        Restore an archived report.
        
        Args:
            archive_id: ID of the archive to restore
            
        Returns:
            Dictionary with information about the restored report, or None if not found
        """
        # Find the archive in the index
        archive_entry = None
        archive_index = None
        
        for i, archive in enumerate(self.report_index["archives"]):
            if archive.get("id") == archive_id:
                archive_entry = archive
                archive_index = i
                break
        
        if not archive_entry:
            logger.warning(f"Archive with ID {archive_id} not found in index")
            return None
        
        archive_path = archive_entry.get("archive_path")
        if not os.path.exists(archive_path):
            logger.warning(f"Archive file not found: {archive_path}")
            return None
        
        try:
            # Extract the archive
            with zipfile.ZipFile(archive_path, 'r') as archive:
                # Get the list of files in the archive
                file_list = archive.namelist()
                
                # Find the report file (first file that's not metadata.json)
                report_file = None
                for file in file_list:
                    if file != "metadata.json":
                        report_file = file
                        break
                
                if not report_file:
                    logger.warning(f"No report file found in archive: {archive_path}")
                    return None
                
                # Extract the report file
                report_path = os.path.join(self.output_dir, report_file)
                archive.extract(report_file, self.output_dir)
                
                # Extract metadata
                if "metadata.json" in file_list:
                    metadata_str = archive.read("metadata.json").decode('utf-8')
                    metadata = json.loads(metadata_str)
                else:
                    metadata = archive_entry.get("metadata", {})
            
            # Create report entry
            report_entry = {
                "id": str(uuid.uuid4()),
                "path": report_path,
                "type": archive_entry.get("type", "unknown"),
                "format": archive_entry.get("format", "unknown"),
                "timestamp": datetime.datetime.now().isoformat(),
                "metadata": metadata,
                "restored_from": archive_id
            }
            
            # Add to reports list
            self.report_index["reports"].append(report_entry)
            
            # Remove from archives list
            self.report_index["archives"].pop(archive_index)
            
            # Save the index
            self._save_report_index()
            
            # Update the report catalog
            self._generate_report_catalog()
            
            logger.info(f"Restored archive {archive_id} to {report_path}")
            
            return report_entry
            
        except Exception as e:
            logger.error(f"Failed to restore archive {archive_id}: {e}")
            return None
    
    def distribute_report(
        self,
        report_entry: Dict[str, Any],
        recipients: List[str],
        subject: Optional[str] = None,
        message: Optional[str] = None
    ) -> bool:
        """
        Distribute a report to recipients via email.
        
        Args:
            report_entry: Report entry to distribute
            recipients: List of email addresses to send the report to
            subject: Email subject
            message: Email message body
            
        Returns:
            True if successful, False otherwise
        """
        if not self.email_config:
            logger.warning("Email distribution requires email_config to be provided")
            return False
        
        report_path = report_entry.get("path")
        if not os.path.exists(report_path):
            logger.warning(f"Report file not found: {report_path}")
            return False
        
        # Get report metadata
        metadata = report_entry.get("metadata", {})
        report_title = metadata.get("title", "Simulation Validation Report")
        report_type = report_entry.get("type", "report")
        
        # Create email subject
        if not subject:
            subject = f"{report_title} - {report_type.replace('_', ' ').title()}"
        
        # Create email message
        msg = MIMEMultipart()
        msg["From"] = self.email_config.get("from", "noreply@example.com")
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        
        # Add message body
        if not message:
            message = f"""
Hello,

Please find attached the {report_type.replace('_', ' ').title()} "{report_title}".

This report was generated automatically by the Simulation Validation Reporting Framework.

Regards,
Simulation Validation Team
"""
        
        msg.attach(MIMEText(message, 'plain'))
        
        # Add the report as an attachment
        with open(report_path, 'rb') as f:
            attachment = MIMEApplication(f.read(), _subtype="octet-stream")
            attachment.add_header(
                'Content-Disposition',
                'attachment',
                filename=os.path.basename(report_path)
            )
            msg.attach(attachment)
        
        try:
            # Connect to SMTP server
            smtp_server = self.email_config.get("smtp_server", "localhost")
            smtp_port = self.email_config.get("smtp_port", 25)
            smtp_username = self.email_config.get("username")
            smtp_password = self.email_config.get("password")
            use_tls = self.email_config.get("use_tls", False)
            
            if use_tls:
                server = smtplib.SMTP(smtp_server, smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP(smtp_server, smtp_port)
            
            # Login if credentials provided
            if smtp_username and smtp_password:
                server.login(smtp_username, smtp_password)
            
            # Send email
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Distributed report {report_entry['id']} to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to distribute report {report_entry['id']}: {e}")
            return False
    
    def schedule_report_generation(
        self,
        validation_results_provider: Callable[[], Dict[str, Any]],
        schedule_type: str,
        schedule_value: Union[str, int],
        report_type: ReportType = ReportType.COMPREHENSIVE_REPORT,
        output_format: Optional[ReportFormat] = None,
        title_template: Optional[str] = None,
        description_template: Optional[str] = None,
        distribution_list: Optional[List[str]] = None,
        job_id: Optional[str] = None
    ) -> str:
        """
        Schedule periodic report generation.
        
        Args:
            validation_results_provider: Function that returns validation results
            schedule_type: Type of schedule (daily, weekly, monthly, interval)
            schedule_value: Value for the schedule type (weekday, day, hour, minutes)
            report_type: Type of report to generate
            output_format: Output format for the report
            title_template: Template for the report title
            description_template: Template for the report description
            distribution_list: List of email addresses to send the report to
            job_id: Optional job ID, will be generated if not provided
            
        Returns:
            Job ID for the scheduled job
        """
        # Generate job ID if not provided
        if not job_id:
            job_id = str(uuid.uuid4())
        
        # Create a job function that generates and optionally distributes the report
        def job_function():
            try:
                # Get validation results
                validation_results = validation_results_provider()
                
                # Generate title and description with current timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                title = title_template.format(timestamp=timestamp) if title_template else f"Scheduled Report - {timestamp}"
                description = description_template.format(timestamp=timestamp) if description_template else f"Automatically generated report at {timestamp}"
                
                # Generate the report
                report_entry = self.generate_report(
                    validation_results=validation_results,
                    report_type=report_type,
                    output_format=output_format,
                    title=title,
                    description=description
                )
                
                # Distribute the report if distribution list is provided
                if distribution_list:
                    self.distribute_report(report_entry, distribution_list)
                
                logger.info(f"Successfully executed scheduled job {job_id}")
                
            except Exception as e:
                logger.error(f"Error executing scheduled job {job_id}: {e}")
        
        # Create the scheduled job based on schedule type
        if schedule_type == "daily":
            hour, minute = schedule_value.split(":")
            job = self.scheduler.every().day.at(schedule_value).do(job_function)
        elif schedule_type == "weekly":
            weekday, time = schedule_value.split("@")
            weekday = weekday.lower()
            hour, minute = time.split(":")
            
            if weekday == "monday":
                job = self.scheduler.every().monday.at(time).do(job_function)
            elif weekday == "tuesday":
                job = self.scheduler.every().tuesday.at(time).do(job_function)
            elif weekday == "wednesday":
                job = self.scheduler.every().wednesday.at(time).do(job_function)
            elif weekday == "thursday":
                job = self.scheduler.every().thursday.at(time).do(job_function)
            elif weekday == "friday":
                job = self.scheduler.every().friday.at(time).do(job_function)
            elif weekday == "saturday":
                job = self.scheduler.every().saturday.at(time).do(job_function)
            elif weekday == "sunday":
                job = self.scheduler.every().sunday.at(time).do(job_function)
            else:
                raise ValueError(f"Invalid weekday: {weekday}")
        elif schedule_type == "monthly":
            day, time = schedule_value.split("@")
            # Note: there's no built-in support for monthly schedules in the schedule library
            # In a real implementation, we would need to handle this differently
            raise NotImplementedError("Monthly schedules are not implemented")
        elif schedule_type == "interval":
            minutes = int(schedule_value)
            job = self.scheduler.every(minutes).minutes.do(job_function)
        else:
            raise ValueError(f"Invalid schedule type: {schedule_type}")
        
        # Store the job in the scheduled jobs dictionary
        self.scheduled_jobs[job_id] = {
            "job": job,
            "type": schedule_type,
            "value": schedule_value,
            "report_type": report_type.value,
            "output_format": output_format.value if output_format else None,
            "title_template": title_template,
            "description_template": description_template,
            "distribution_list": distribution_list
        }
        
        # Start the scheduler if not already running
        self._start_scheduler()
        
        logger.info(f"Scheduled report generation job {job_id} with {schedule_type} schedule: {schedule_value}")
        
        return job_id
    
    def _start_scheduler(self) -> None:
        """Start the scheduler thread if not already running."""
        if self.schedule_thread is None or not self.schedule_thread.is_alive():
            self.schedule_stop_event = threading.Event()
            self.schedule_thread = threading.Thread(target=self._run_scheduler)
            self.schedule_thread.daemon = True
            self.schedule_thread.start()
            logger.info("Started scheduler thread")
    
    def _run_scheduler(self) -> None:
        """Run the scheduler in a loop until stopped."""
        while not self.schedule_stop_event.is_set():
            self.scheduler.run_pending()
            time.sleep(1)
    
    def stop_scheduler(self) -> None:
        """Stop the scheduler thread."""
        if self.schedule_thread and self.schedule_thread.is_alive():
            self.schedule_stop_event.set()
            self.schedule_thread.join()
            self.schedule_thread = None
            logger.info("Stopped scheduler thread")
    
    def cancel_scheduled_job(self, job_id: str) -> bool:
        """
        Cancel a scheduled job.
        
        Args:
            job_id: ID of the job to cancel
            
        Returns:
            True if successful, False otherwise
        """
        if job_id not in self.scheduled_jobs:
            logger.warning(f"Job with ID {job_id} not found")
            return False
        
        try:
            job = self.scheduled_jobs[job_id]["job"]
            self.scheduler.cancel_job(job)
            del self.scheduled_jobs[job_id]
            logger.info(f"Cancelled scheduled job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def get_scheduled_jobs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a dictionary of scheduled jobs.
        
        Returns:
            Dictionary mapping job IDs to job details
        """
        # Return a copy of the scheduled jobs dictionary without the actual job object
        result = {}
        for job_id, job_details in self.scheduled_jobs.items():
            result[job_id] = {k: v for k, v in job_details.items() if k != "job"}
        return result
    
    def _schedule_auto_archive(self) -> None:
        """Schedule automatic archiving of old reports."""
        def auto_archive_job():
            try:
                # Get current time
                now = datetime.datetime.now()
                
                # Find reports older than auto_archive_days
                reports_to_archive = []
                for report in self.report_index["reports"]:
                    report_time = None
                    try:
                        report_time = datetime.datetime.fromisoformat(report.get("timestamp", ""))
                    except ValueError:
                        # If timestamp is invalid, skip this report
                        continue
                    
                    if (now - report_time).days >= self.auto_archive_days:
                        reports_to_archive.append(report)
                
                # Archive old reports
                for report in reports_to_archive:
                    logger.info(f"Auto-archiving report {report.get('id')} (age: {(now - datetime.datetime.fromisoformat(report.get('timestamp', ''))).days} days)")
                    self.archive_report(report.get("id"))
                
                logger.info(f"Auto-archived {len(reports_to_archive)} reports")
                
            except Exception as e:
                logger.error(f"Error in auto-archive job: {e}")
        
        # Schedule daily auto-archive
        job = self.scheduler.every().day.at("00:00").do(auto_archive_job)
        
        # Store the job
        self.scheduled_jobs["auto_archive"] = {
            "job": job,
            "type": "daily",
            "value": "00:00",
            "description": "Automatic archiving of old reports"
        }
        
        # Start the scheduler
        self._start_scheduler()
    
    def _open_report(self, report_path: str) -> bool:
        """
        Open a report in the default application.
        
        Args:
            report_path: Path to the report to open
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(report_path):
            logger.warning(f"Report file not found: {report_path}")
            return False
        
        try:
            import webbrowser
            webbrowser.open(report_path)
            return True
        except Exception as e:
            logger.error(f"Failed to open report: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Create a report manager
    manager = ReportManager(
        output_dir="reports",
        archive_dir="archives",
        default_format=ReportFormat.HTML,
        include_visualizations=True,
        company_name="Example Corp"
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
    
    # Generate reports using the manager
    standard_report = manager.generate_report(
        validation_results=validation_results,
        report_type=ReportType.COMPREHENSIVE_REPORT,
        title="Standard Validation Report",
        description="Comprehensive validation report with all results"
    )
    
    executive_report = manager.generate_report(
        validation_results=validation_results,
        report_type=ReportType.EXECUTIVE_SUMMARY,
        title="Executive Summary",
        description="High-level summary for executive review"
    )
    
    technical_report = manager.generate_report(
        validation_results=validation_results,
        report_type=ReportType.TECHNICAL_REPORT,
        title="Technical Report",
        description="Detailed technical analysis of validation results"
    )
    
    print(f"Generated standard report: {standard_report['path']}")
    print(f"Generated executive report: {executive_report['path']}")
    print(f"Generated technical report: {technical_report['path']}")
    
    # Print catalog path
    catalog_path = os.path.join(manager.output_dir, manager.report_catalog_file)
    print(f"Report catalog: {catalog_path}")