#!/usr/bin/env python3
"""
Test script for real-time error visualization.

This script demonstrates how to use the error reporting API to send
real-time error updates to the Error Visualization Dashboard.
"""

import argparse
import anyio
import json
import logging
import random
import sys
import time
from datetime import datetime
import traceback
from typing import Dict, Any, List, Optional

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Error categories
ERROR_CATEGORIES = [
    'RESOURCE_ALLOCATION_ERROR',
    'RESOURCE_CLEANUP_ERROR',
    'NETWORK_CONNECTION_ERROR',
    'NETWORK_TIMEOUT_ERROR',
    'HARDWARE_AVAILABILITY_ERROR',
    'HARDWARE_CAPABILITY_ERROR',
    'HARDWARE_PERFORMANCE_ERROR',
    'WORKER_CRASH_ERROR',
    'WORKER_TIMEOUT_ERROR',
    'TEST_EXECUTION_ERROR',
    'TEST_VALIDATION_ERROR',
    'UNKNOWN_ERROR'
]

# Error types
ERROR_TYPES = [
    'ResourceError',
    'NetworkError',
    'HardwareError',
    'WorkerError',
    'TestError',
    'UnknownError'
]

# Error messages
ERROR_MESSAGES = [
    'Failed to allocate GPU memory',
    'GPU memory allocation exceeded available resources',
    'Network connection to worker timed out',
    'Failed to connect to worker node',
    'CUDA device not available',
    'Hardware temperature exceeded safe limits',
    'Hardware performance degraded significantly',
    'Worker crashed unexpectedly',
    'Worker failed to respond within timeout period',
    'Test execution failed with exception',
    'Test result validation failed',
    'Unknown error occurred during test execution'
]

# Hardware types
HARDWARE_TYPES = [
    'cpu',
    'cuda',
    'rocm',
    'mps',
    'openvino',
    'qualcomm',
    'webnn',
    'webgpu'
]

# Worker IDs
WORKER_IDS = [
    'worker-1',
    'worker-2',
    'worker-3',
    'worker-4'
]

def generate_system_context() -> Dict[str, Any]:
    """Generate mock system context information."""
    return {
        'hostname': f'test-node-{random.randint(1, 5)}',
        'platform': random.choice(['linux', 'darwin', 'win32']),
        'architecture': random.choice(['x86_64', 'arm64']),
        'python_version': f'3.{random.randint(8, 11)}.{random.randint(0, 10)}',
        'metrics': {
            'cpu': {
                'percent': random.randint(10, 95),
                'count': random.randint(4, 32),
                'physical_count': random.randint(2, 16),
                'frequency_mhz': random.randint(1800, 3600)
            },
            'memory': {
                'used_percent': random.randint(20, 95),
                'total_gb': random.randint(8, 128),
                'available_gb': random.randint(1, 64)
            },
            'disk': {
                'used_percent': random.randint(10, 90),
                'total_gb': random.randint(256, 2048),
                'free_gb': random.randint(32, 512)
            }
        },
        'gpu_metrics': {
            'count': random.randint(1, 4),
            'devices': [
                {
                    'index': i,
                    'name': f'Test GPU {i}',
                    'memory_utilization': random.randint(10, 95),
                    'temperature': random.randint(40, 90)
                }
                for i in range(random.randint(1, 4))
            ]
        }
    }

def generate_hardware_context() -> Dict[str, Any]:
    """Generate mock hardware context information."""
    hardware_type = random.choice(HARDWARE_TYPES)
    return {
        'hardware_type': hardware_type,
        'hardware_types': [hardware_type] + random.sample(HARDWARE_TYPES, k=random.randint(0, 3)),
        'hardware_status': {
            'overheating': random.random() < 0.1,  # 10% chance of overheating
            'memory_pressure': random.random() < 0.15,  # 15% chance of memory pressure
            'throttling': random.random() < 0.2   # 20% chance of throttling
        }
    }

def generate_error_frequency() -> Dict[str, Any]:
    """Generate mock error frequency information."""
    is_recurring = random.random() < 0.25  # 25% chance of being a recurring error
    
    return {
        'recurring': is_recurring,
        'same_type': {
            'last_1h': random.randint(1, 5),
            'last_6h': random.randint(5, 15),
            'last_24h': random.randint(10, 30)
        },
        'similar_message': {
            'last_1h': random.randint(0, 3),
            'last_6h': random.randint(0, 8),
            'last_24h': random.randint(0, 20)
        }
    }

def generate_mock_error() -> Dict[str, Any]:
    """Generate a mock error for testing."""
    # Choose error category
    error_category = random.choice(ERROR_CATEGORIES)
    
    # Derive error type based on category
    category_prefix = error_category.split('_')[0]
    error_type = next((t for t in ERROR_TYPES if category_prefix.lower() in t.lower()), 'UnknownError')
    
    # Choose appropriate message based on category
    if 'RESOURCE' in error_category:
        message = ERROR_MESSAGES[random.randint(0, 1)]
    elif 'NETWORK' in error_category:
        message = ERROR_MESSAGES[random.randint(2, 3)]
    elif 'HARDWARE' in error_category:
        message = ERROR_MESSAGES[random.randint(4, 6)]
    elif 'WORKER' in error_category:
        message = ERROR_MESSAGES[random.randint(7, 8)]
    elif 'TEST' in error_category:
        message = ERROR_MESSAGES[random.randint(9, 10)]
    else:
        message = ERROR_MESSAGES[11]  # Unknown error
    
    # Generate traceback for some errors
    tb = None
    if random.random() < 0.5:  # 50% chance of having a traceback
        try:
            # Deliberately cause an exception
            raise Exception(message)
        except Exception:
            tb = traceback.format_exc()
    
    # Generate the error data
    return {
        'timestamp': datetime.now().isoformat(),
        'worker_id': random.choice(WORKER_IDS),
        'type': error_type,
        'error_category': error_category,
        'message': message,
        'traceback': tb,
        'system_context': generate_system_context(),
        'hardware_context': generate_hardware_context(),
        'error_frequency': generate_error_frequency()
    }

async def report_error(dashboard_url: str, error_data: Dict[str, Any]) -> bool:
    """Report an error to the Error Visualization Dashboard API.
    
    Args:
        dashboard_url: URL of the dashboard API
        error_data: Error data to report
        
    Returns:
        True if error was successfully reported, False otherwise
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{dashboard_url}/api/report-error",
                json=error_data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                result = await response.json()
                if response.status == 200 and result.get('status') == 'success':
                    logger.info(f"Error reported successfully: {error_data.get('type')} - {error_data.get('message')}")
                    return True
                else:
                    logger.error(f"Failed to report error: {response.status} - {result}")
                    return False
    except Exception as e:
        logger.error(f"Exception while reporting error: {e}")
        return False

async def run_error_generator(dashboard_url: str, count: int, interval: float, critical_percent: float) -> None:
    """Run the error generator to send errors to the dashboard.
    
    Args:
        dashboard_url: URL of the dashboard API
        count: Number of errors to generate
        interval: Interval between error reports in seconds
        critical_percent: Percentage of errors that should be critical
    """
    logger.info(f"Starting error generator with {count} errors at {interval}s intervals")
    logger.info(f"Dashboard URL: {dashboard_url}")
    
    for i in range(count):
        # Generate a mock error
        error = generate_mock_error()
        
        # Make some errors critical based on critical_percent
        is_critical = random.random() < (critical_percent / 100.0)
        if is_critical:
            # Force critical error properties
            if random.random() < 0.33:
                # Resource error
                error['error_category'] = 'RESOURCE_ALLOCATION_ERROR'
                error['type'] = 'ResourceError'
                error['message'] = 'Critical: Failed to allocate GPU memory'
            elif random.random() < 0.5:
                # Hardware error
                error['error_category'] = 'HARDWARE_AVAILABILITY_ERROR'
                error['type'] = 'HardwareError'
                error['message'] = 'Critical: CUDA device not available'
                error['hardware_context']['hardware_status']['overheating'] = True
            else:
                # Worker crash
                error['error_category'] = 'WORKER_CRASH_ERROR'
                error['type'] = 'WorkerError'
                error['message'] = 'Critical: Worker crashed unexpectedly'
            
            # Update system context for critical errors
            if random.random() < 0.5:
                # High CPU usage
                error['system_context']['metrics']['cpu']['percent'] = random.randint(90, 100)
            else:
                # High memory usage
                error['system_context']['metrics']['memory']['used_percent'] = random.randint(95, 100)
        
        # Report the error
        success = await report_error(dashboard_url, error)
        
        # Log the result
        if success:
            logger.info(f"Reported error {i+1}/{count}: {error['type']} - {error['message']}")
        else:
            logger.error(f"Failed to report error {i+1}/{count}")
        
        # Wait for the specified interval
        if i < count - 1:  # Don't wait after the last error
            await anyio.sleep(interval)
    
    logger.info(f"Error generator completed ({count} errors reported)")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Error Visualization Real-Time Test')
    parser.add_argument('--url', default='http://localhost:8080', help='URL of the dashboard server')
    parser.add_argument('--count', type=int, default=10, help='Number of errors to generate')
    parser.add_argument('--interval', type=float, default=2.0, help='Interval between error reports in seconds')
    parser.add_argument('--critical-percent', type=float, default=20.0, help='Percentage of errors that should be critical')
    args = parser.parse_args()
    
    # Run the error generator
    anyio.run(run_error_generator(args.url, args.count, args.interval, args.critical_percent))

if __name__ == '__main__':
    main()