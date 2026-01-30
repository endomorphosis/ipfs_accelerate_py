"""
Reporting module for the Simulation Accuracy and Validation Framework.

This module provides comprehensive reporting capabilities for simulation validation results,
including multi-format report generation, executive summaries, technical reports,
and comparative reporting.

Key features:
- Multi-format report generation (HTML, PDF, Markdown)
- Executive summary generation
- Detailed technical reports with statistical analysis
- Comparative reporting for tracking improvements
- Report customization options
- Report versioning and archiving
"""

from .report_generator import ReportGenerator
from .executive_summary import ExecutiveSummaryGenerator
from .technical_report import TechnicalReportGenerator
from .comparative_report import ComparativeReportGenerator
from .report_manager import ReportManager

__all__ = [
    'ReportGenerator',
    'ExecutiveSummaryGenerator',
    'TechnicalReportGenerator',
    'ComparativeReportGenerator',
    'ReportManager',
]