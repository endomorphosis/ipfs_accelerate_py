#!/usr/bin/env python3
"""
Test script for the enhanced error notification system with the new system-critical sound.

This script simulates different severities of errors and tests the sound notification
system's response to each, with a focus on the newly added system-critical level.
"""

import os
import sys
import time
import json
import logging
import argparse
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("error_notification_test")

def generate_error_data(severity, error_category, message, worker_id="test-worker"):
    """
    Generate error data with the specified severity and message.
    
    Args:
        severity: Error severity (system_critical, critical, warning, info)
        error_category: Category of error (e.g., DATABASE_CORRUPTION, HARDWARE_AVAILABILITY_ERROR)
        message: Error message
        worker_id: ID of the worker node (default: test-worker)
        
    Returns:
        dict: Error data dictionary
    """
    # Create base error data
    error_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "worker_id": worker_id,
        "type": "ErrorNotificationTest",
        "error_category": error_category,
        "message": message,
        "traceback": f"Simulated {severity} error for testing",
        "system_context": {
            "hostname": "test-node-1",
            "platform": "linux",
            "metrics": {
                "cpu": {"percent": 50},
                "memory": {"used_percent": 60}
            }
        },
        "hardware_context": {
            "hardware_type": "test",
            "hardware_status": {
                "overheating": False,
                "memory_pressure": False
            }
        }
    }
    
    # Add severity-specific flags
    if severity == "system_critical":
        error_data["is_system_critical"] = True
        error_data["is_critical"] = True
    elif severity == "critical":
        error_data["is_critical"] = True
    
    return error_data

def report_error(dashboard_url, error_data):
    """
    Report an error to the dashboard via API.
    
    Args:
        dashboard_url: URL of the dashboard API
        error_data: Error data dictionary
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.post(
            f"{dashboard_url}/api/report-error",
            json=error_data,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully reported {error_data.get('error_category')} error")
            return True
        else:
            logger.error(f"Failed to report error: {response.status_code} - {response.text}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return False

def test_system_critical_error(dashboard_url):
    """
    Test system-critical error notification.
    
    Args:
        dashboard_url: URL of the dashboard API
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing SYSTEM-CRITICAL error notification")
    
    error_data = generate_error_data(
        severity="system_critical",
        error_category="COORDINATOR_FAILURE",
        message="Coordinator process has crashed unexpectedly"
    )
    
    return report_error(dashboard_url, error_data)

def test_database_corruption_error(dashboard_url):
    """
    Test database corruption error notification.
    
    Args:
        dashboard_url: URL of the dashboard API
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing DATABASE CORRUPTION error notification")
    
    error_data = generate_error_data(
        severity="system_critical",
        error_category="DATABASE_CORRUPTION",
        message="Database corruption detected in main results table"
    )
    
    return report_error(dashboard_url, error_data)

def test_security_breach_error(dashboard_url):
    """
    Test security breach error notification.
    
    Args:
        dashboard_url: URL of the dashboard API
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing SECURITY BREACH error notification")
    
    error_data = generate_error_data(
        severity="system_critical",
        error_category="SECURITY_BREACH",
        message="Unauthorized access attempt detected"
    )
    
    return report_error(dashboard_url, error_data)

def test_critical_error(dashboard_url):
    """
    Test critical error notification.
    
    Args:
        dashboard_url: URL of the dashboard API
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing CRITICAL error notification")
    
    error_data = generate_error_data(
        severity="critical",
        error_category="HARDWARE_AVAILABILITY_ERROR",
        message="GPU device unavailable: CUDA error (2): out of memory"
    )
    
    return report_error(dashboard_url, error_data)

def test_warning_error(dashboard_url):
    """
    Test warning error notification.
    
    Args:
        dashboard_url: URL of the dashboard API
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing WARNING error notification")
    
    error_data = generate_error_data(
        severity="warning",
        error_category="NETWORK_CONNECTION_ERROR",
        message="Network connection to worker-05 unstable"
    )
    
    return report_error(dashboard_url, error_data)

def test_info_error(dashboard_url):
    """
    Test info error notification.
    
    Args:
        dashboard_url: URL of the dashboard API
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Testing INFO error notification")
    
    error_data = generate_error_data(
        severity="info",
        error_category="TEST_EXECUTION_ERROR",
        message="Test batch_size=32 failed with non-zero exit code"
    )
    
    return report_error(dashboard_url, error_data)

def main():
    """Run the error notification test script."""
    parser = argparse.ArgumentParser(description="Test error notification system")
    parser.add_argument("--url", default="http://localhost:8080", 
                        help="URL of the dashboard server (default: http://localhost:8080)")
    parser.add_argument("--test", choices=["all", "system_critical", "critical", "warning", "info"],
                        default="all", help="Type of error to test (default: all)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Interval between error reports in seconds (default: 2.0)")
    parser.add_argument("--system-critical-only", action="store_true",
                        help="Test only system-critical errors with different categories")
    
    args = parser.parse_args()
    
    logger.info(f"Starting error notification test with dashboard URL: {args.url}")
    
    if args.system_critical_only:
        # Test all system-critical error categories
        logger.info("Testing all system-critical error categories")
        test_system_critical_error(args.url)
        time.sleep(args.interval)
        test_database_corruption_error(args.url)
        time.sleep(args.interval)
        test_security_breach_error(args.url)
        return
    
    if args.test == "all" or args.test == "system_critical":
        test_system_critical_error(args.url)
        time.sleep(args.interval)
    
    if args.test == "all" or args.test == "critical":
        test_critical_error(args.url)
        time.sleep(args.interval)
    
    if args.test == "all" or args.test == "warning":
        test_warning_error(args.url)
        time.sleep(args.interval)
    
    if args.test == "all" or args.test == "info":
        test_info_error(args.url)
    
    logger.info("Error notification test completed")

if __name__ == "__main__":
    main()