#!/usr/bin/env python3
"""
Run validation on all generated model tests.

This script runs validation checks on the generated test files to ensure they
comply with the ModelTest pattern and have proper architecture support.
"""

import os
import sys
import argparse
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"validation_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
try:
    from refactored_test_suite.validation.test_validator import validate_test_files, generate_validation_report
    from refactored_test_suite.generators.architecture_detector import ARCHITECTURE_TYPES
except ImportError:
    logger.error("Failed to import required modules")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run validation on generated model tests")
    
    parser.add_argument(
        "--test-dir",
        default="./generated_tests",
        help="Directory containing generated test files"
    )
    
    parser.add_argument(
        "--report-dir",
        default="./reports",
        help="Directory to save validation reports"
    )
    
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix common issues in invalid tests"
    )
    
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=list(ARCHITECTURE_TYPES.keys()) + ["all"],
        default=["all"],
        help="Specific architectures to validate"
    )
    
    return parser.parse_args()

def summarize_validation_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarize validation results with additional metrics.
    
    Args:
        results: Validation results from validate_test_files
        
    Returns:
        Dict with summary metrics
    """
    summary = {
        "total_files": results["total"],
        "valid_files": results["valid"],
        "invalid_files": results["invalid"],
        "validity_percentage": (results["valid"] / results["total"]) * 100 if results["total"] > 0 else 0,
        "architecture_breakdown": {},
        "common_issues": {}
    }
    
    # Count files by architecture
    arch_counts = {}
    for file_result in results["files"]:
        file_path = file_result["file_path"]
        file_name = os.path.basename(file_path)
        
        # Try to determine architecture from parent class
        arch_type = None
        pattern_details = file_result.get("pattern_details", {})
        if "ModelTest" in pattern_details.get("model_test_import", ""):
            import_path = pattern_details["model_test_import"]
            for arch_name in ["EncoderOnly", "DecoderOnly", "EncoderDecoder", "Vision", "Speech", "VisionText", "Multimodal"]:
                if arch_name in import_path:
                    arch_type = arch_name.lower().replace("text", "-text").replace("only", "-only")
                    if arch_type == "encoderdecoder":
                        arch_type = "encoder-decoder"
                    break
        
        # If architecture can't be determined from import, try from file name
        if not arch_type:
            for arch_type_name, models in ARCHITECTURE_TYPES.items():
                for model in models:
                    if model in file_name.lower():
                        arch_type = arch_type_name
                        break
                if arch_type:
                    break
        
        if not arch_type:
            arch_type = "unknown"
        
        arch_counts[arch_type] = arch_counts.get(arch_type, 0) + 1
    
    summary["architecture_breakdown"] = arch_counts
    
    # Count common issues
    issues = {}
    for file_result in results["files"]:
        if not file_result["valid"]:
            if not file_result["syntax_valid"]:
                issues["syntax_errors"] = issues.get("syntax_errors", 0) + 1
            else:
                pattern_details = file_result.get("pattern_details", {})
                
                if not pattern_details.get("imports_model_test", False):
                    issues["missing_model_test_import"] = issues.get("missing_model_test_import", 0) + 1
                
                if not pattern_details.get("inherits_model_test", False):
                    issues["not_inheriting_model_test"] = issues.get("not_inheriting_model_test", 0) + 1
                
                missing_methods = pattern_details.get("missing_methods", [])
                for method in missing_methods:
                    key = f"missing_method_{method}"
                    issues[key] = issues.get(key, 0) + 1
    
    summary["common_issues"] = issues
    
    return summary

def generate_summary_report(summary: Dict[str, Any], output_file: str) -> None:
    """
    Generate a summary report.
    
    Args:
        summary: Summary metrics
        output_file: Path to save report
    """
    with open(output_file, "w") as f:
        f.write("# Model Test Validation Summary\n\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall stats
        f.write("## Overall Statistics\n\n")
        f.write(f"- **Total files**: {summary['total_files']}\n")
        f.write(f"- **Valid files**: {summary['valid_files']} ({summary['validity_percentage']:.1f}%)\n")
        f.write(f"- **Invalid files**: {summary['invalid_files']}\n\n")
        
        # Architecture breakdown
        f.write("## Architecture Breakdown\n\n")
        f.write("| Architecture | Count | Percentage |\n")
        f.write("|--------------|-------|------------|\n")
        
        for arch, count in sorted(summary["architecture_breakdown"].items(), key=lambda x: x[1], reverse=True):
            pct = (count / summary["total_files"]) * 100 if summary["total_files"] > 0 else 0
            f.write(f"| {arch} | {count} | {pct:.1f}% |\n")
        
        # Common issues
        if summary["common_issues"]:
            f.write("\n## Common Issues\n\n")
            f.write("| Issue | Count | Percentage |\n")
            f.write("|-------|-------|------------|\n")
            
            for issue, count in sorted(summary["common_issues"].items(), key=lambda x: x[1], reverse=True):
                pct = (count / summary["invalid_files"]) * 100 if summary["invalid_files"] > 0 else 0
                # Format issue name for readability
                issue_name = issue.replace("_", " ").title().replace("Model Test", "ModelTest")
                f.write(f"| {issue_name} | {count} | {pct:.1f}% |\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. Fix invalid test files\n")
        f.write("2. Increase test coverage across all architectures\n")
        f.write("3. Validate all tests against actual models\n")
    
    logger.info(f"Summary report written to {output_file}")

def run_validation(args):
    """
    Run validation and generate reports.
    
    Args:
        args: Command-line arguments
    """
    # Create report directory if it doesn't exist
    os.makedirs(args.report_dir, exist_ok=True)
    
    # Validate test files
    logger.info(f"Validating test files in {args.test_dir}")
    results = validate_test_files(args.test_dir)
    
    # Generate detailed validation report
    detailed_report_path = os.path.join(args.report_dir, f"validation_details_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    generate_validation_report(results, detailed_report_path)
    
    # Generate summary report
    summary = summarize_validation_results(results)
    summary_report_path = os.path.join(args.report_dir, f"validation_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    generate_summary_report(summary, summary_report_path)
    
    # Print summary to console
    logger.info("\nValidation Summary:")
    logger.info(f"- Total files: {summary['total_files']}")
    logger.info(f"- Valid files: {summary['valid_files']} ({summary['validity_percentage']:.1f}%)")
    logger.info(f"- Invalid files: {summary['invalid_files']}")
    
    if summary["invalid_files"] > 0:
        logger.info("\nTop Issues:")
        for issue, count in sorted(summary["common_issues"].items(), key=lambda x: x[1], reverse=True)[:3]:
            issue_name = issue.replace("_", " ").title().replace("Model Test", "ModelTest")
            logger.info(f"- {issue_name}: {count} files")
    
    logger.info(f"\nDetailed report: {detailed_report_path}")
    logger.info(f"Summary report: {summary_report_path}")
    
    return summary["invalid_files"] == 0

def main():
    """Main entry point."""
    args = parse_args()
    success = run_validation(args)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())