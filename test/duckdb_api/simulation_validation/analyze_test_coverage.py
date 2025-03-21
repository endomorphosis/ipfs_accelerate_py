#!/usr/bin/env python3
"""
Test Coverage Analyzer for Simulation Validation Framework

This module analyzes test coverage reports from pytest-cov and generates
summary reports in different formats (text, markdown, HTML, JSON).
It provides insights into test coverage across different components
of the Simulation Validation Framework.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import xml.etree.ElementTree as ET
import re

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("analyze_test_coverage")

class TestCoverageAnalyzer:
    """
    Analyzes test coverage reports from pytest-cov.
    """
    
    def __init__(self):
        """Initialize the test coverage analyzer."""
        pass
    
    def load_coverage_xml(self, coverage_file: str) -> Dict[str, Any]:
        """
        Load and parse a coverage.xml file generated by pytest-cov.
        
        Args:
            coverage_file: Path to the coverage XML file
            
        Returns:
            Dictionary with parsed coverage data
        """
        try:
            if not os.path.exists(coverage_file):
                logger.error(f"Coverage file not found: {coverage_file}")
                return {}
            
            # Parse XML file
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            
            # Extract coverage information
            coverage_data = {
                "summary": {
                    "line_rate": float(root.attrib.get("line-rate", 0)),
                    "branch_rate": float(root.attrib.get("branch-rate", 0)),
                    "lines_covered": int(root.attrib.get("lines-covered", 0)),
                    "lines_valid": int(root.attrib.get("lines-valid", 0)),
                    "timestamp": root.attrib.get("timestamp", ""),
                    "version": root.attrib.get("version", "")
                },
                "packages": [],
                "modules": []
            }
            
            # Process packages
            for package in root.findall(".//package"):
                pkg_data = {
                    "name": package.attrib.get("name", ""),
                    "line_rate": float(package.attrib.get("line-rate", 0)),
                    "branch_rate": float(package.attrib.get("branch-rate", 0)),
                    "complexity": float(package.attrib.get("complexity", 0))
                }
                coverage_data["packages"].append(pkg_data)
                
                # Process classes (modules)
                for cls in package.findall(".//class"):
                    module_name = cls.attrib.get("name", "")
                    file_path = cls.attrib.get("filename", "")
                    
                    # Get lines with coverage data
                    lines_data = []
                    for line in cls.findall(".//line"):
                        lines_data.append({
                            "number": int(line.attrib.get("number", 0)),
                            "hits": int(line.attrib.get("hits", 0)),
                            "branch": line.attrib.get("branch", "false") == "true"
                        })
                    
                    # Count covered and total lines
                    covered_lines = sum(1 for line in lines_data if line["hits"] > 0)
                    total_lines = len(lines_data)
                    line_rate = covered_lines / total_lines if total_lines > 0 else 0
                    
                    module_data = {
                        "name": module_name,
                        "file_path": file_path,
                        "line_rate": line_rate,
                        "lines_covered": covered_lines,
                        "lines_valid": total_lines,
                        "lines": lines_data
                    }
                    coverage_data["modules"].append(module_data)
            
            return coverage_data
            
        except Exception as e:
            logger.error(f"Error loading coverage XML: {e}")
            return {}
    
    def analyze_coverage(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze coverage data to extract insights.
        
        Args:
            coverage_data: Parsed coverage data
            
        Returns:
            Dictionary with coverage analysis
        """
        if not coverage_data:
            return {
                "status": "error",
                "message": "No coverage data available for analysis"
            }
        
        # Extract summary information
        summary = coverage_data.get("summary", {})
        
        # Calculate overall coverage percentage
        line_rate = summary.get("line_rate", 0)
        overall_coverage = round(line_rate * 100, 2)
        
        # Count modules by coverage level
        modules = coverage_data.get("modules", [])
        coverage_levels = {
            "excellent": 0,  # 90-100%
            "good": 0,       # 75-90%
            "moderate": 0,   # 50-75%
            "low": 0,        # 25-50%
            "poor": 0        # 0-25%
        }
        
        for module in modules:
            module_coverage = module.get("line_rate", 0) * 100
            if module_coverage >= 90:
                coverage_levels["excellent"] += 1
            elif module_coverage >= 75:
                coverage_levels["good"] += 1
            elif module_coverage >= 50:
                coverage_levels["moderate"] += 1
            elif module_coverage >= 25:
                coverage_levels["low"] += 1
            else:
                coverage_levels["poor"] += 1
        
        # Identify modules with lowest coverage
        modules_by_coverage = sorted(modules, key=lambda m: m.get("line_rate", 0))
        lowest_coverage = modules_by_coverage[:5] if len(modules_by_coverage) > 5 else modules_by_coverage
        
        # Group modules by component
        components = {}
        component_patterns = {
            "core": r"core/",
            "statistical": r"statistical/",
            "comparison": r"comparison/",
            "calibration": r"calibration/",
            "drift_detection": r"drift_detection/",
            "visualization": r"visualization/"
        }
        
        for module in modules:
            file_path = module.get("file_path", "")
            component_found = False
            
            for component, pattern in component_patterns.items():
                if re.search(pattern, file_path):
                    if component not in components:
                        components[component] = {
                            "modules": [],
                            "total_lines": 0,
                            "covered_lines": 0
                        }
                    
                    components[component]["modules"].append(module)
                    components[component]["total_lines"] += module.get("lines_valid", 0)
                    components[component]["covered_lines"] += module.get("lines_covered", 0)
                    component_found = True
                    break
            
            if not component_found:
                # Use a generic component for files outside specific directories
                component = "other"
                if component not in components:
                    components[component] = {
                        "modules": [],
                        "total_lines": 0,
                        "covered_lines": 0
                    }
                
                components[component]["modules"].append(module)
                components[component]["total_lines"] += module.get("lines_valid", 0)
                components[component]["covered_lines"] += module.get("lines_covered", 0)
        
        # Calculate coverage by component
        component_coverage = {}
        for component, data in components.items():
            total_lines = data["total_lines"]
            covered_lines = data["covered_lines"]
            coverage_pct = (covered_lines / total_lines) * 100 if total_lines > 0 else 0
            
            component_coverage[component] = {
                "name": component,
                "modules_count": len(data["modules"]),
                "lines_total": total_lines,
                "lines_covered": covered_lines,
                "coverage_percent": round(coverage_pct, 2)
            }
        
        # Create analysis results
        analysis = {
            "status": "success",
            "overall_coverage": overall_coverage,
            "lines_covered": summary.get("lines_covered", 0),
            "lines_total": summary.get("lines_valid", 0),
            "modules_count": len(modules),
            "coverage_levels": coverage_levels,
            "lowest_coverage_modules": [{
                "name": m.get("name", ""),
                "file_path": m.get("file_path", ""),
                "coverage_percent": round(m.get("line_rate", 0) * 100, 2),
                "lines_covered": m.get("lines_covered", 0),
                "lines_total": m.get("lines_valid", 0)
            } for m in lowest_coverage],
            "component_coverage": component_coverage
        }
        
        return analysis
    
    def generate_report(
        self, 
        analysis: Dict[str, Any], 
        format: str = "text", 
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate a test coverage report.
        
        Args:
            analysis: Coverage analysis data
            format: Report format (text, markdown, html, json)
            output_file: File to write report to (optional)
            
        Returns:
            Report content
        """
        if analysis.get("status") != "success":
            return f"Error: {analysis.get('message', 'Unknown error')}"
        
        # Generate report in requested format
        if format == "json":
            result = json.dumps(analysis, indent=2)
            
        elif format == "markdown":
            lines = ["# Test Coverage Report\n"]
            
            # Summary section
            lines.append("## Summary\n")
            lines.append(f"- **Overall Coverage**: {analysis['overall_coverage']}%")
            lines.append(f"- **Lines Covered**: {analysis['lines_covered']} of {analysis['lines_total']}")
            lines.append(f"- **Modules Analyzed**: {analysis['modules_count']}\n")
            
            # Coverage distribution
            levels = analysis["coverage_levels"]
            lines.append("## Coverage Distribution\n")
            lines.append("| Coverage Level | Module Count | Percentage |")
            lines.append("|---------------|-------------|------------|")
            
            total_modules = analysis["modules_count"]
            if total_modules > 0:
                lines.append(f"| Excellent (90-100%) | {levels['excellent']} | {round(levels['excellent']/total_modules*100, 1)}% |")
                lines.append(f"| Good (75-90%) | {levels['good']} | {round(levels['good']/total_modules*100, 1)}% |")
                lines.append(f"| Moderate (50-75%) | {levels['moderate']} | {round(levels['moderate']/total_modules*100, 1)}% |")
                lines.append(f"| Low (25-50%) | {levels['low']} | {round(levels['low']/total_modules*100, 1)}% |")
                lines.append(f"| Poor (0-25%) | {levels['poor']} | {round(levels['poor']/total_modules*100, 1)}% |")
            else:
                lines.append("| No modules analyzed | 0 | 0% |")
            
            lines.append("\n## Component Coverage\n")
            lines.append("| Component | Coverage | Lines Covered | Total Lines |")
            lines.append("|-----------|----------|---------------|-------------|")
            
            # Sort components by coverage percentage (descending)
            sorted_components = sorted(
                analysis["component_coverage"].values(), 
                key=lambda c: c["coverage_percent"], 
                reverse=True
            )
            
            for component in sorted_components:
                lines.append(f"| {component['name']} | {component['coverage_percent']}% | {component['lines_covered']} | {component['lines_total']} |")
            
            # Low coverage modules
            if analysis["lowest_coverage_modules"]:
                lines.append("\n## Modules with Lowest Coverage\n")
                lines.append("| Module | Coverage | Lines Covered | Total Lines |")
                lines.append("|--------|----------|---------------|-------------|")
                
                for module in analysis["lowest_coverage_modules"]:
                    name = module["name"].split(".")[-1]  # Use just the filename without path
                    lines.append(f"| {name} | {module['coverage_percent']}% | {module['lines_covered']} | {module['lines_total']} |")
            
            result = "\n".join(lines)
            
        elif format == "html":
            html_lines = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "    <title>Test Coverage Report</title>",
                "    <style>",
                "        body { font-family: Arial, sans-serif; margin: 20px; }",
                "        .summary { background-color: #f5f5f5; padding: 15px; margin-bottom: 20px; }",
                "        .good { color: #2e7d32; }",
                "        .moderate { color: #ff9800; }",
                "        .poor { color: #d32f2f; }",
                "        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
                "        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }",
                "        th { background-color: #f5f5f5; }",
                "        .progress-container { width: 100px; background-color: #f1f1f1; border-radius: 3px; }",
                "        .progress-bar { height: 15px; border-radius: 3px; }",
                "        .excellent { background-color: #4caf50; }",
                "        .good { background-color: #8bc34a; }",
                "        .moderate { background-color: #ffeb3b; }",
                "        .low { background-color: #ff9800; }",
                "        .poor { background-color: #f44336; }",
                "    </style>",
                "</head>",
                "<body>",
                "    <h1>Test Coverage Report</h1>",
                ""
            ]
            
            # Summary section
            overall_class = "good" if analysis["overall_coverage"] >= 75 else "moderate" if analysis["overall_coverage"] >= 50 else "poor"
            
            html_lines.extend([
                "    <div class='summary'>",
                "        <h2>Summary</h2>",
                f"        <p><strong>Overall Coverage</strong>: <span class='{overall_class}'>{analysis['overall_coverage']}%</span></p>",
                f"        <p><strong>Lines Covered</strong>: {analysis['lines_covered']} of {analysis['lines_total']}</p>",
                f"        <p><strong>Modules Analyzed</strong>: {analysis['modules_count']}</p>",
                "    </div>",
                ""
            ])
            
            # Coverage distribution
            levels = analysis["coverage_levels"]
            total_modules = analysis["modules_count"]
            
            html_lines.extend([
                "    <h2>Coverage Distribution</h2>",
                "    <table>",
                "        <tr>",
                "            <th>Coverage Level</th>",
                "            <th>Module Count</th>",
                "            <th>Percentage</th>",
                "            <th>Distribution</th>",
                "        </tr>"
            ])
            
            if total_modules > 0:
                excellent_pct = round(levels['excellent']/total_modules*100, 1)
                good_pct = round(levels['good']/total_modules*100, 1)
                moderate_pct = round(levels['moderate']/total_modules*100, 1)
                low_pct = round(levels['low']/total_modules*100, 1)
                poor_pct = round(levels['poor']/total_modules*100, 1)
                
                html_lines.extend([
                    "        <tr>",
                    f"            <td>Excellent (90-100%)</td>",
                    f"            <td>{levels['excellent']}</td>",
                    f"            <td>{excellent_pct}%</td>",
                    f"            <td><div class='progress-container'><div class='progress-bar excellent' style='width:{excellent_pct}%'></div></div></td>",
                    "        </tr>",
                    "        <tr>",
                    f"            <td>Good (75-90%)</td>",
                    f"            <td>{levels['good']}</td>",
                    f"            <td>{good_pct}%</td>",
                    f"            <td><div class='progress-container'><div class='progress-bar good' style='width:{good_pct}%'></div></div></td>",
                    "        </tr>",
                    "        <tr>",
                    f"            <td>Moderate (50-75%)</td>",
                    f"            <td>{levels['moderate']}</td>",
                    f"            <td>{moderate_pct}%</td>",
                    f"            <td><div class='progress-container'><div class='progress-bar moderate' style='width:{moderate_pct}%'></div></div></td>",
                    "        </tr>",
                    "        <tr>",
                    f"            <td>Low (25-50%)</td>",
                    f"            <td>{levels['low']}</td>",
                    f"            <td>{low_pct}%</td>",
                    f"            <td><div class='progress-container'><div class='progress-bar low' style='width:{low_pct}%'></div></div></td>",
                    "        </tr>",
                    "        <tr>",
                    f"            <td>Poor (0-25%)</td>",
                    f"            <td>{levels['poor']}</td>",
                    f"            <td>{poor_pct}%</td>",
                    f"            <td><div class='progress-container'><div class='progress-bar poor' style='width:{poor_pct}%'></div></div></td>",
                    "        </tr>"
                ])
            else:
                html_lines.append("        <tr><td colspan='4'>No modules analyzed</td></tr>")
                
            html_lines.append("    </table>")
            
            # Component coverage
            html_lines.extend([
                "    <h2>Component Coverage</h2>",
                "    <table>",
                "        <tr>",
                "            <th>Component</th>",
                "            <th>Coverage</th>",
                "            <th>Lines Covered</th>",
                "            <th>Total Lines</th>",
                "            <th>Coverage Bar</th>",
                "        </tr>"
            ])
            
            # Sort components by coverage percentage (descending)
            sorted_components = sorted(
                analysis["component_coverage"].values(), 
                key=lambda c: c["coverage_percent"], 
                reverse=True
            )
            
            for component in sorted_components:
                cov_pct = component["coverage_percent"]
                bar_class = "excellent" if cov_pct >= 90 else "good" if cov_pct >= 75 else "moderate" if cov_pct >= 50 else "low" if cov_pct >= 25 else "poor"
                
                html_lines.extend([
                    "        <tr>",
                    f"            <td>{component['name']}</td>",
                    f"            <td>{cov_pct}%</td>",
                    f"            <td>{component['lines_covered']}</td>",
                    f"            <td>{component['lines_total']}</td>",
                    f"            <td><div class='progress-container'><div class='progress-bar {bar_class}' style='width:{cov_pct}%'></div></div></td>",
                    "        </tr>"
                ])
                
            html_lines.append("    </table>")
            
            # Low coverage modules
            if analysis["lowest_coverage_modules"]:
                html_lines.extend([
                    "    <h2>Modules with Lowest Coverage</h2>",
                    "    <table>",
                    "        <tr>",
                    "            <th>Module</th>",
                    "            <th>Coverage</th>",
                    "            <th>Lines Covered</th>",
                    "            <th>Total Lines</th>",
                    "            <th>Coverage Bar</th>",
                    "        </tr>"
                ])
                
                for module in analysis["lowest_coverage_modules"]:
                    name = module["name"].split(".")[-1]  # Use just the filename without path
                    cov_pct = module["coverage_percent"]
                    bar_class = "excellent" if cov_pct >= 90 else "good" if cov_pct >= 75 else "moderate" if cov_pct >= 50 else "low" if cov_pct >= 25 else "poor"
                    
                    html_lines.extend([
                        "        <tr>",
                        f"            <td>{name}</td>",
                        f"            <td>{cov_pct}%</td>",
                        f"            <td>{module['lines_covered']}</td>",
                        f"            <td>{module['lines_total']}</td>",
                        f"            <td><div class='progress-container'><div class='progress-bar {bar_class}' style='width:{cov_pct}%'></div></div></td>",
                        "        </tr>"
                    ])
                    
                html_lines.append("    </table>")
            
            html_lines.extend([
                "</body>",
                "</html>"
            ])
            
            result = "\n".join(html_lines)
            
        else:
            # Plain text output
            lines = ["Test Coverage Report", "=" * 20, ""]
            lines.append(f"Overall Coverage: {analysis['overall_coverage']}%")
            lines.append(f"Lines Covered: {analysis['lines_covered']} of {analysis['lines_total']}")
            lines.append(f"Modules Analyzed: {analysis['modules_count']}")
            lines.append("")
            
            # Coverage distribution
            levels = analysis["coverage_levels"]
            lines.append("Coverage Distribution:")
            total_modules = analysis["modules_count"]
            if total_modules > 0:
                lines.append(f"  Excellent (90-100%): {levels['excellent']} ({round(levels['excellent']/total_modules*100, 1)}%)")
                lines.append(f"  Good (75-90%): {levels['good']} ({round(levels['good']/total_modules*100, 1)}%)")
                lines.append(f"  Moderate (50-75%): {levels['moderate']} ({round(levels['moderate']/total_modules*100, 1)}%)")
                lines.append(f"  Low (25-50%): {levels['low']} ({round(levels['low']/total_modules*100, 1)}%)")
                lines.append(f"  Poor (0-25%): {levels['poor']} ({round(levels['poor']/total_modules*100, 1)}%)")
            else:
                lines.append("  No modules analyzed")
            lines.append("")
            
            # Component coverage
            lines.append("Component Coverage:")
            sorted_components = sorted(
                analysis["component_coverage"].values(), 
                key=lambda c: c["coverage_percent"], 
                reverse=True
            )
            
            for component in sorted_components:
                lines.append(f"  {component['name']}: {component['coverage_percent']}% ({component['lines_covered']}/{component['lines_total']} lines)")
            lines.append("")
            
            # Low coverage modules
            if analysis["lowest_coverage_modules"]:
                lines.append("Modules with Lowest Coverage:")
                for module in analysis["lowest_coverage_modules"]:
                    name = module["name"].split(".")[-1]  # Use just the filename without path
                    lines.append(f"  {name}: {module['coverage_percent']}% ({module['lines_covered']}/{module['lines_total']} lines)")
            
            result = "\n".join(lines)
        
        # Write to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(result)
                
        return result


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Analyze test coverage for Simulation Validation Framework")
    parser.add_argument("--coverage-file", type=str, required=True, help="Path to coverage.xml file")
    parser.add_argument("--output-format", type=str, default="text", choices=["text", "json", "markdown", "html"], help="Output format")
    parser.add_argument("--output-file", type=str, help="Output file")
    
    args = parser.parse_args()
    
    # Validate coverage file
    if not os.path.exists(args.coverage_file):
        logger.error(f"Coverage file does not exist: {args.coverage_file}")
        sys.exit(1)
    
    # Create analyzer
    analyzer = TestCoverageAnalyzer()
    
    # Load and analyze coverage data
    coverage_data = analyzer.load_coverage_xml(args.coverage_file)
    analysis = analyzer.analyze_coverage(coverage_data)
    
    # Generate report
    report = analyzer.generate_report(
        analysis, 
        format=args.output_format, 
        output_file=args.output_file
    )
    
    # Output report if not writing to file
    if not args.output_file:
        print(report)
    else:
        print(f"Coverage report written to {args.output_file}")
    
    # Return success if coverage data was loaded and analyzed
    return 0 if analysis.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())