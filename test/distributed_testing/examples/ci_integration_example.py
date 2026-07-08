#!/usr/bin/env python3
"""
CI Integration Example

This script demonstrates how to use the CI integration features
of the hardware monitoring system, including:
1. Running tests with CI integration
2. Generating status badges
3. Sending notifications based on test results

Usage:
    python ci_integration_example.py [options]

Options:
    --test-mode MODE       Test mode (standard, basic, full, long)
    --notification         Enable test notifications
    --generate-badge       Generate status badge
    --output-dir DIR       Output directory for reports and badges
    --db-path PATH         Path to test database
"""

import os
import sys
import argparse
import subprocess
import logging
import json
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("ci_integration_example")

# Add parent directory to path to ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Try to import CI modules
try:
    from ci_notification import build_notification_context, send_notifications, load_config
    from generate_status_badge import generate_badge_svg, get_test_status
    CI_MODULES_AVAILABLE = True
except ImportError:
    logger.warning("CI modules not found. Some features may not be available.")
    CI_MODULES_AVAILABLE = False


def run_tests(args):
    """
    Run hardware monitoring tests with CI integration.
    
    Args:
        args: Command-line arguments
    
    Returns:
        bool: Success status
    """
    logger.info("Running hardware monitoring tests with CI integration...")
    
    # Build command
    cmd = [
        os.path.join(parent_dir, "run_hardware_monitoring_ci_tests.sh"),
        f"--mode", args.test_mode
    ]
    
    # Add options
    if args.generate_badge:
        cmd.append("--generate-badge")
    
    if args.notification:
        cmd.append("--send-notifications")
    
    if args.ci_integration:
        cmd.append("--ci-integration")
    
    # Set environment variables
    env = os.environ.copy()
    if args.db_path:
        env["BENCHMARK_DB_PATH"] = args.db_path
    
    # Run command
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=parent_dir,
            check=False,
            capture_output=True,
            text=True
        )
        
        # Log output
        logger.info(f"Command output:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Command errors:\n{result.stderr}")
        
        # Return success status
        return result.returncode == 0
    
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return False


def generate_badge(args):
    """
    Generate status badge.
    
    Args:
        args: Command-line arguments
    
    Returns:
        bool: Success status
    """
    if not CI_MODULES_AVAILABLE:
        logger.error("Badge generation requires CI modules that aren't available.")
        return False
    
    logger.info("Generating status badge...")
    
    try:
        # Define paths
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        badge_path = output_dir / "hardware_monitoring_status.svg"
        json_path = output_dir / "hardware_monitoring_status.json"
        
        # Get test status from database
        db_path = args.db_path or os.path.join(parent_dir, "hardware_metrics.duckdb")
        status, passing_runs, total_runs = get_test_status(db_path, days=1, min_runs=1)
        
        logger.info(f"Test status: {status} ({passing_runs}/{total_runs} passing)")
        
        # Generate badge
        badge_svg = generate_badge_svg("tests", status)
        
        # Write badge to file
        badge_path.write_text(badge_svg)
        logger.info(f"Badge generated at {badge_path}")
        
        # Generate JSON status file
        status_json = {
            "schemaVersion": 1,
            "label": "tests",
            "message": status,
            "color": "#4c1" if status == "passing" else "#e05d44",
            "isError": status == "failing",
            "timestamp": datetime.now().isoformat(),
            "runs": {
                "passing": passing_runs,
                "total": total_runs
            }
        }
        
        with open(json_path, "w") as f:
            json.dump(status_json, f, indent=2)
        
        logger.info(f"Status JSON generated at {json_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating badge: {str(e)}")
        return False


def send_notification(args, test_status):
    """
    Send test notifications.
    
    Args:
        args: Command-line arguments
        test_status: Test status (success or failure)
    
    Returns:
        bool: Success status
    """
    if not CI_MODULES_AVAILABLE:
        logger.error("Notification requires CI modules that aren't available.")
        return False
    
    logger.info(f"Sending test notifications for status: {test_status}...")
    
    try:
        # Define paths
        config_path = os.path.join(parent_dir, "notification_config.json")
        report_path = os.path.join(args.output_dir, "test_report.html")
        
        # Load configuration
        config = load_config(config_path)
        
        # Build notification context
        context = {
            "status": test_status,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "workflow": "Example CI Integration",
            "run_id": f"example-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "commit": "example",
            "summary": f"Example test run with status: {test_status}",
            "report_url": os.path.abspath(report_path) if os.path.exists(report_path) else "",
            "dry_run": True  # Set to True for example purposes
        }
        
        # Send notifications
        channels = ["email", "slack", "github"] if args.all_channels else ["github"]
        for channel in channels:
            logger.info(f"Sending notification to {channel}...")
            
        return True
    
    except Exception as e:
        logger.error(f"Error sending notifications: {str(e)}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CI Integration Example")
    parser.add_argument("--test-mode", choices=["standard", "basic", "full", "long"], default="standard",
                      help="Test mode")
    parser.add_argument("--notification", action="store_true",
                      help="Enable test notifications")
    parser.add_argument("--generate-badge", action="store_true",
                      help="Generate status badge")
    parser.add_argument("--ci-integration", action="store_true",
                      help="Run CI integration tests")
    parser.add_argument("--output-dir", default="./example_output",
                      help="Output directory for reports and badges")
    parser.add_argument("--db-path",
                      help="Path to test database")
    parser.add_argument("--all-channels", action="store_true",
                      help="Use all notification channels")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run tests
    success = run_tests(args)
    
    # Generate badge if requested
    if args.generate_badge:
        badge_success = generate_badge(args)
        success = success and badge_success
    
    # Send notifications if requested
    if args.notification:
        notification_success = send_notification(args, "success" if success else "failure")
        success = success and notification_success
    
    # Return exit code based on success
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())