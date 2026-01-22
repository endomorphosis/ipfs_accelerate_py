#!/usr/bin/env python3
"""
Test script for generating system-critical error events.

This script sends sample system-critical error events to the dashboard
for testing the error notification system.
"""

import json
import time
import logging
import argparse
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system_critical_test")

def generate_system_critical_error(error_category, message, worker_id="test-worker"):
    """
    Generate a system-critical error with the specified category and message.
    
    Args:
        error_category: Category of error (e.g., COORDINATOR_FAILURE)
        message: Error message
        worker_id: ID of the worker node (default: test-worker)
        
    Returns:
        dict: Error data dictionary with system-critical flag
    """
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "worker_id": worker_id,
        "type": "SystemCriticalErrorTest",
        "error_category": error_category,
        "message": message,
        "traceback": f"Simulated system-critical error for testing: {error_category}",
        "is_system_critical": True,  # Explicitly mark as system-critical
        "is_critical": True,  # Also mark as critical for backward compatibility
        "system_context": {
            "hostname": "test-node-1",
            "platform": "linux",
            "metrics": {
                "cpu": {"percent": 95, "temperature": 82},
                "memory": {"used_percent": 98}
            }
        },
        "hardware_context": {
            "hardware_type": "test",
            "hardware_status": {
                "overheating": True,
                "memory_pressure": True
            }
        }
    }

def send_error_event(dashboard_url, error_data):
    """
    Send an error event to the dashboard.
    
    Args:
        dashboard_url: URL of the dashboard
        error_data: Error data to send
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Format endpoint URL
        endpoint = f"{dashboard_url}/api/report-error"
        logger.info(f"Sending error event to {endpoint}")
        
        # Send the request
        response = requests.post(
            endpoint,
            json=error_data,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully sent {error_data.get('error_category')} event")
            logger.info(f"Response: {response.text}")
            return True
        else:
            logger.error(f"Failed to send event: {response.status_code} - {response.text}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return False

def run_error_demo(dashboard_url, interval=5, count=3):
    """
    Run the system-critical error demo by sending multiple error events.
    
    Args:
        dashboard_url: URL of the dashboard
        interval: Interval between events in seconds
        count: Number of events to send
    """
    # System-critical error scenarios
    scenarios = [
        ("COORDINATOR_FAILURE", "Coordinator process has crashed unexpectedly"),
        ("DATABASE_CORRUPTION", "Database corruption detected in main results table"),
        ("SECURITY_BREACH", "Unauthorized access attempt detected from IP 192.168.1.100"),
        ("SYSTEM_CRASH", "Catastrophic system failure in primary monitoring node"),
        ("RESOURCE_EXHAUSTION_CRITICAL", "Critical resource exhaustion: disk space < 1%")
    ]
    
    logger.info(f"Starting system-critical error demo: {count} events with {interval}s interval")
    
    for i in range(count):
        # Get a scenario (cycle through them)
        scenario = scenarios[i % len(scenarios)]
        error_category, message = scenario
        
        # Generate error data
        error_data = generate_system_critical_error(
            error_category=error_category,
            message=message,
            worker_id=f"test-worker-{i+1}"
        )
        
        # Send the error event
        success = send_error_event(dashboard_url, error_data)
        
        if success:
            logger.info(f"Sent system-critical error event {i+1}/{count}: {error_category}")
        else:
            logger.error(f"Failed to send error event {i+1}/{count}")
        
        # Wait for the next interval (except for the last one)
        if i < count - 1:
            logger.info(f"Waiting {interval} seconds before next event...")
            time.sleep(interval)
    
    logger.info("System-critical error demo completed")

def main():
    """Run the system-critical error demo."""
    parser = argparse.ArgumentParser(description="System-Critical Error Demo")
    parser.add_argument("--url", default="http://localhost:8080", 
                      help="URL of the dashboard server (default: http://localhost:8080)")
    parser.add_argument("--interval", type=float, default=5.0,
                      help="Interval between error events in seconds (default: 5.0)")
    parser.add_argument("--count", type=int, default=3,
                      help="Number of error events to send (default: 3)")
    
    args = parser.parse_args()
    
    run_error_demo(args.url, args.interval, args.count)

if __name__ == "__main__":
    main()