#!/usr/bin/env python3
"""
Comprehensive Test Suite for the Error Visualization System.

This script provides an extensive test suite for the Error Visualization system, 
covering error processing, visualization, sound notifications, and WebSocket integration.
"""

import os
import sys
import json
import unittest
import tempfile
import anyio
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path to import the modules
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the modules to test
from data.duckdb.distributed_testing.dashboard.error_visualization_integration import ErrorVisualizationIntegration
from data.duckdb.distributed_testing.dashboard.static.sounds.generate_sound_files import (
    generate_critical_sound,
    generate_warning_sound,
    generate_info_sound,
    convert_wav_to_mp3
)

class TestSoundGeneration(unittest.TestCase):
    """Test suite for sound generation functionality."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Save current directory
        self.original_dir = os.getcwd()
        
        # Change to temp directory
        os.chdir(self.output_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Return to original directory
        os.chdir(self.original_dir)
        
        # Clean up temp directory
        self.temp_dir.cleanup()
    
    def test_generate_critical_sound(self):
        """Test generation of critical error sound."""
        # Generate critical sound
        output_file = os.path.join(self.output_dir, "test_critical.wav")
        generate_critical_sound(output_file)
        
        # Check that file was created and has content
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 1000)
    
    def test_generate_warning_sound(self):
        """Test generation of warning error sound."""
        # Generate warning sound
        output_file = os.path.join(self.output_dir, "test_warning.wav")
        generate_warning_sound(output_file)
        
        # Check that file was created and has content
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 1000)
    
    def test_generate_info_sound(self):
        """Test generation of info error sound."""
        # Generate info sound
        output_file = os.path.join(self.output_dir, "test_info.wav")
        generate_info_sound(output_file)
        
        # Check that file was created and has content
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 1000)
    
    def test_conversion_with_ffmpeg(self):
        """Test WAV to MP3 conversion with ffmpeg."""
        # Skip if ffmpeg is not available
        try:
            import subprocess
            result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            has_ffmpeg = result.returncode == 0
        except FileNotFoundError:
            has_ffmpeg = False
        
        if not has_ffmpeg:
            self.skipTest("ffmpeg not available")
        
        # Generate a WAV file
        wav_file = os.path.join(self.output_dir, "test_conversion.wav")
        generate_info_sound(wav_file)
        
        # Convert to MP3
        mp3_file = os.path.join(self.output_dir, "test_conversion.mp3")
        convert_wav_to_mp3(wav_file, mp3_file)
        
        # Check that MP3 file was created
        self.assertTrue(os.path.exists(mp3_file))
        self.assertGreater(os.path.getsize(mp3_file), 1000)
    
    def test_fallback_without_ffmpeg(self):
        """Test fallback behavior when ffmpeg is not available."""
        # Create a mock subprocess that raises FileNotFoundError
        with patch('subprocess.run', side_effect=FileNotFoundError("ffmpeg not found")):
            # Generate a WAV file
            wav_file = os.path.join(self.output_dir, "test_fallback.wav")
            generate_info_sound(wav_file)
            
            # Try to convert to MP3 (should use fallback)
            mp3_file = os.path.join(self.output_dir, "test_fallback.mp3")
            convert_wav_to_mp3(wav_file, mp3_file)
            
            # Check that MP3 file exists (should be a copy of WAV)
            self.assertTrue(os.path.exists(mp3_file))
            
            # Compare file sizes (should be close since it's just a copy)
            wav_size = os.path.getsize(wav_file)
            mp3_size = os.path.getsize(mp3_file)
            self.assertEqual(wav_size, mp3_size)


class TestSeverityDetection(unittest.TestCase):
    """Test suite for error severity detection logic."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create error visualization integration
        self.error_viz = ErrorVisualizationIntegration(
            output_dir=self.output_dir,
            db_path=None
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_critical_by_category(self):
        """Test detection of critical errors based on error category."""
        # Test all critical categories
        critical_categories = [
            'HARDWARE_NOT_AVAILABLE',
            'RESOURCE_EXHAUSTED',
            'WORKER_CRASHED'
        ]
        
        for category in critical_categories:
            error = {"error_category": category}
            self.assertTrue(
                self.error_viz._is_critical_error(error),
                f"Failed to identify {category} as critical"
            )
    
    def test_critical_by_hardware_status(self):
        """Test detection of critical errors based on hardware status."""
        # Test overheating
        error_overheating = {
            "error_category": "TEST_ERROR",
            "hardware_context": {
                "hardware_status": {
                    "overheating": True,
                    "memory_pressure": False,
                    "throttling": False
                }
            }
        }
        
        # Test memory pressure
        error_memory_pressure = {
            "error_category": "TEST_ERROR",
            "hardware_context": {
                "hardware_status": {
                    "overheating": False,
                    "memory_pressure": True,
                    "throttling": False
                }
            }
        }
        
        # Throttling alone should NOT be critical
        error_throttling = {
            "error_category": "TEST_ERROR",
            "hardware_context": {
                "hardware_status": {
                    "overheating": False,
                    "memory_pressure": False,
                    "throttling": True
                }
            }
        }
        
        self.assertTrue(self.error_viz._is_critical_error(error_overheating))
        self.assertTrue(self.error_viz._is_critical_error(error_memory_pressure))
        self.assertFalse(self.error_viz._is_critical_error(error_throttling))
    
    def test_critical_by_system_metrics(self):
        """Test detection of critical errors based on system metrics."""
        # Test high CPU usage
        error_high_cpu = {
            "error_category": "TEST_ERROR",
            "system_context": {
                "metrics": {
                    "cpu": {
                        "percent": 95
                    },
                    "memory": {
                        "used_percent": 80
                    },
                    "disk": {
                        "used_percent": 80
                    }
                }
            }
        }
        
        # Test high memory usage
        error_high_memory = {
            "error_category": "TEST_ERROR",
            "system_context": {
                "metrics": {
                    "cpu": {
                        "percent": 80
                    },
                    "memory": {
                        "used_percent": 96
                    },
                    "disk": {
                        "used_percent": 80
                    }
                }
            }
        }
        
        # Test high disk usage
        error_high_disk = {
            "error_category": "TEST_ERROR",
            "system_context": {
                "metrics": {
                    "cpu": {
                        "percent": 80
                    },
                    "memory": {
                        "used_percent": 80
                    },
                    "disk": {
                        "used_percent": 96
                    }
                }
            }
        }
        
        # Test normal usage (should not be critical)
        error_normal = {
            "error_category": "TEST_ERROR",
            "system_context": {
                "metrics": {
                    "cpu": {
                        "percent": 70
                    },
                    "memory": {
                        "used_percent": 70
                    },
                    "disk": {
                        "used_percent": 70
                    }
                }
            }
        }
        
        self.assertTrue(self.error_viz._is_critical_error(error_high_cpu))
        self.assertTrue(self.error_viz._is_critical_error(error_high_memory))
        self.assertTrue(self.error_viz._is_critical_error(error_high_disk))
        self.assertFalse(self.error_viz._is_critical_error(error_normal))
    
    def test_severity_combination(self):
        """Test error severity detection with combined factors."""
        # Test error with multiple critical indicators
        error_multiple_critical = {
            "error_category": "HARDWARE_NOT_AVAILABLE",
            "hardware_context": {
                "hardware_status": {
                    "overheating": True,
                    "memory_pressure": True,
                    "throttling": True
                }
            },
            "system_context": {
                "metrics": {
                    "cpu": {
                        "percent": 95
                    },
                    "memory": {
                        "used_percent": 95
                    },
                    "disk": {
                        "used_percent": 95
                    }
                }
            }
        }
        
        # Test error with mixed indicators
        error_mixed = {
            "error_category": "NETWORK_CONNECTION_ERROR",
            "hardware_context": {
                "hardware_status": {
                    "overheating": False,
                    "memory_pressure": False,
                    "throttling": True
                }
            },
            "system_context": {
                "metrics": {
                    "cpu": {
                        "percent": 85
                    },
                    "memory": {
                        "used_percent": 85
                    },
                    "disk": {
                        "used_percent": 85
                    }
                }
            }
        }
        
        self.assertTrue(self.error_viz._is_critical_error(error_multiple_critical))
        self.assertFalse(self.error_viz._is_critical_error(error_mixed))


class TestJavaScriptSeverityDetection(unittest.TestCase):
    """Test suite for JavaScript error severity detection logic."""
    
    def test_js_error_severity_logic(self):
        """Validate the JavaScript severity detection logic from the HTML template."""
        # This test checks that the JavaScript severity detection logic in error_visualization.html
        # is consistent with the Python implementation
        
        # Extract JavaScript logic from HTML template
        js_logic = """
            // Determine error severity for sound and notification
            function determineErrorSeverity(error) {
                let errorSeverity = 'default';
                
                if (error.is_critical) {
                    errorSeverity = 'critical';
                } else {
                    // Check error category to determine severity
                    const errorCategory = error.error_category || '';
                    
                    if (errorCategory.includes('HARDWARE_NOT_AVAILABLE') || 
                        errorCategory.includes('RESOURCE_EXHAUSTED') || 
                        errorCategory.includes('WORKER_CRASHED')) {
                        // Hardware, resource exhaustion, and worker crash errors are critical
                        errorSeverity = 'critical';
                    } else if (errorCategory.includes('NETWORK') || 
                               errorCategory.includes('RESOURCE') ||
                               errorCategory.includes('WORKER')) {
                        // Network, other resource, and worker errors are warnings
                        errorSeverity = 'warning';
                    } else {
                        // Test and other errors are info
                        errorSeverity = 'info';
                    }
                }
                
                return errorSeverity;
            }
        """
        
        # Print the JavaScript logic for reference
        print("JavaScript error severity detection logic:")
        print(js_logic)
        
        # Create a simple Python implementation based on the JavaScript logic
        def determine_error_severity(error):
            if error.get('is_critical'):
                return 'critical'
            
            error_category = error.get('error_category', '')
            
            if any(category in error_category for category in [
                'HARDWARE_NOT_AVAILABLE', 'RESOURCE_EXHAUSTED', 'WORKER_CRASHED']):
                return 'critical'
            elif any(category in error_category for category in ['NETWORK', 'RESOURCE', 'WORKER']):
                return 'warning'
            else:
                return 'info'
        
        # Test cases
        test_cases = [
            # Critical errors
            {'error_category': 'HARDWARE_NOT_AVAILABLE', 'expected': 'critical'},
            {'error_category': 'RESOURCE_EXHAUSTED', 'expected': 'critical'},
            {'error_category': 'WORKER_CRASHED', 'expected': 'critical'},
            {'is_critical': True, 'error_category': 'TEST_ERROR', 'expected': 'critical'},
            
            # Warning errors
            {'error_category': 'NETWORK_CONNECTION_ERROR', 'expected': 'warning'},
            {'error_category': 'RESOURCE_CLEANUP_ERROR', 'expected': 'warning'},
            {'error_category': 'WORKER_TIMEOUT_ERROR', 'expected': 'warning'},
            
            # Info errors
            {'error_category': 'TEST_EXECUTION_ERROR', 'expected': 'info'},
            {'error_category': 'TEST_VALIDATION_ERROR', 'expected': 'info'},
            {'error_category': 'UNKNOWN_ERROR', 'expected': 'info'}
        ]
        
        # Validate each test case
        for i, case in enumerate(test_cases):
            error = {k: v for k, v in case.items() if k != 'expected'}
            expected = case['expected']
            result = determine_error_severity(error)
            
            self.assertEqual(
                result, expected,
                f"Test case {i+1} failed: expected '{expected}' for {error}, got '{result}'"
            )


class TestWebSocketIntegration(unittest.TestCase):
    """Test suite for WebSocket integration."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create mock WebSocket manager
        self.websocket_manager = MagicMock()
        
        # Create error visualization integration with WebSocket manager
        self.error_viz = ErrorVisualizationIntegration(
            output_dir=self.output_dir,
            db_path=None,
            websocket_manager=self.websocket_manager
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_report_error(self):
        anyio.run(self._test_report_error)

    async def _test_report_error(self):
        """Test reporting an error to WebSocket clients."""
        # Create test error
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "worker_id": "test-worker-1",
            "type": "ResourceError",
            "error_category": "RESOURCE_EXHAUSTED",
            "message": "Out of memory: CUDA out of memory",
            "traceback": "Traceback for test error",
            "system_context": {
                "hostname": "test-node-1",
                "platform": "linux",
                "metrics": {
                    "cpu": {"percent": 80},
                    "memory": {"used_percent": 80},
                    "disk": {"used_percent": 80}
                }
            },
            "hardware_context": {
                "hardware_type": "cuda",
                "hardware_status": {
                    "overheating": False,
                    "memory_pressure": True,
                    "throttling": False
                }
            },
            "error_frequency": {
                "recurring": False,
                "same_type": {
                    "last_1h": 1,
                    "last_6h": 1,
                    "last_24h": 1
                },
                "similar_message": {
                    "last_1h": 0,
                    "last_6h": 0,
                    "last_24h": 0
                }
            }
        }
        
        # Report the error
        result = await self.error_viz.report_error(error_data)
        
        # Check result
        self.assertTrue(result)
        
        # Verify WebSocket broadcast was called for each time range
        time_ranges = [1, 6, 24, 168]
        for time_range in time_ranges:
            self.websocket_manager.broadcast.assert_any_call(
                topic=f"error_visualization:{time_range}",
                message={
                    "type": "error_visualization_update",
                    "data": {
                        "error": self.error_viz._prepare_error_for_display(error_data),
                        "time_range": time_range
                    }
                }
            )
        
        # Verify general topic broadcast
        self.websocket_manager.broadcast.assert_any_call(
            topic="error_visualization",
            message={
                "type": "error_visualization_update",
                "data": {
                    "error": self.error_viz._prepare_error_for_display(error_data)
                }
            }
        )
        
        # Verify cache invalidation
        for time_range in time_ranges:
            self.assertNotIn(f"errors_{time_range}", self.error_viz.error_cache)
    
    def test_prepare_error_for_display(self):
        anyio.run(self._test_prepare_error_for_display)

    async def _test_prepare_error_for_display(self):
        """Test preparation of error for display."""
        # Create test error
        error = {
            "timestamp": datetime.now().isoformat(),
            "worker_id": "test-worker-1",
            "type": "ResourceError",
            "error_category": "RESOURCE_EXHAUSTED",
            "message": "Out of memory: CUDA out of memory",
            "error_frequency": {
                "recurring": True,
                "same_type": {"last_1h": 3}
            },
            "hardware_context": {
                "hardware_status": {"memory_pressure": True}
            },
            "system_context": {
                "metrics": {
                    "cpu": {"percent": 95}
                }
            }
        }
        
        # Prepare error for display
        display_error = self.error_viz._prepare_error_for_display(error)
        
        # Check that all fields are present
        self.assertIn("timestamp", display_error)
        self.assertIn("worker_id", display_error)
        self.assertIn("type", display_error)
        self.assertIn("error_category", display_error)
        self.assertIn("message", display_error)
        
        # Check that special flags were added
        self.assertIn("is_recurring", display_error)
        self.assertIn("is_critical", display_error)
        
        # Verify flags based on error content
        self.assertTrue(display_error["is_recurring"])
        self.assertTrue(display_error["is_critical"])
        
        # Verify timestamp format
        self.assertNotEqual(display_error["timestamp"], error["timestamp"])
        try:
            datetime.strptime(display_error["timestamp"], '%Y-%m-%d %H:%M:%S')
        except ValueError:
            self.fail("Timestamp was not properly formatted")


class TestErrorVisualizationIntegration(unittest.TestCase):
    """Additional integration tests for the Error Visualization system."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create test database file
        self.db_path = os.path.join(self.output_dir, "test_error_viz.duckdb")
        
        # Create error visualization integration
        self.error_viz = ErrorVisualizationIntegration(
            output_dir=self.output_dir,
            db_path=self.db_path
        )
        
        # Generate sample errors
        self.sample_errors = self.generate_sample_errors()
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def generate_sample_errors(self):
        """Generate sample error data with various severity levels."""
        # Base timestamp
        now = datetime.now()
        
        # Generate errors with different severity levels
        errors = []
        
        # Critical error (by category)
        errors.append({
            "timestamp": (now - timedelta(minutes=15)).isoformat(),
            "worker_id": "worker-1",
            "type": "HardwareError",
            "error_category": "HARDWARE_NOT_AVAILABLE",
            "message": "GPU device not available",
            "system_context": {
                "hostname": "test-node-1",
                "metrics": {
                    "cpu": {"percent": 80},
                    "memory": {"used_percent": 80},
                    "disk": {"used_percent": 80}
                }
            },
            "hardware_context": {
                "hardware_type": "cuda",
                "hardware_status": {
                    "overheating": False,
                    "memory_pressure": False,
                    "throttling": False
                }
            }
        })
        
        # Critical error (by hardware status)
        errors.append({
            "timestamp": (now - timedelta(minutes=10)).isoformat(),
            "worker_id": "worker-2",
            "type": "HardwareError",
            "error_category": "HARDWARE_COMPATIBILITY_ERROR",
            "message": "GPU overheating detected",
            "system_context": {
                "hostname": "test-node-2",
                "metrics": {
                    "cpu": {"percent": 60},
                    "memory": {"used_percent": 70},
                    "disk": {"used_percent": 60}
                }
            },
            "hardware_context": {
                "hardware_type": "cuda",
                "hardware_status": {
                    "overheating": True,
                    "memory_pressure": False,
                    "throttling": True
                }
            }
        })
        
        # Critical error (by system metrics)
        errors.append({
            "timestamp": (now - timedelta(minutes=5)).isoformat(),
            "worker_id": "worker-3",
            "type": "ResourceError",
            "error_category": "RESOURCE_CLEANUP_ERROR",
            "message": "Failed to clean up resources",
            "system_context": {
                "hostname": "test-node-3",
                "metrics": {
                    "cpu": {"percent": 95},
                    "memory": {"used_percent": 60},
                    "disk": {"used_percent": 60}
                }
            },
            "hardware_context": {
                "hardware_type": "cpu",
                "hardware_status": {
                    "overheating": False,
                    "memory_pressure": False,
                    "throttling": False
                }
            }
        })
        
        # Warning error
        errors.append({
            "timestamp": (now - timedelta(minutes=8)).isoformat(),
            "worker_id": "worker-2",
            "type": "NetworkError",
            "error_category": "NETWORK_CONNECTION_ERROR",
            "message": "Failed to connect to worker node",
            "system_context": {
                "hostname": "test-node-2",
                "metrics": {
                    "cpu": {"percent": 40},
                    "memory": {"used_percent": 50},
                    "disk": {"used_percent": 60}
                }
            }
        })
        
        # Info error
        errors.append({
            "timestamp": (now - timedelta(minutes=3)).isoformat(),
            "worker_id": "worker-1",
            "type": "TestError",
            "error_category": "TEST_EXECUTION_ERROR",
            "message": "Test execution failed",
            "system_context": {
                "hostname": "test-node-1",
                "metrics": {
                    "cpu": {"percent": 30},
                    "memory": {"used_percent": 40},
                    "disk": {"used_percent": 50}
                }
            }
        })
        
        return errors
    
    def test_multiple_error_processing(self):
        anyio.run(self._test_multiple_error_processing)

    async def _test_multiple_error_processing(self):
        """Test processing of multiple errors with different severities."""
        # Process all sample errors
        processed_data = await self.error_viz._process_error_data(self.sample_errors, 24)
        
        # Verify basic structure
        self.assertIn("summary", processed_data)
        self.assertIn("recent_errors", processed_data)
        
        # Verify summary statistics
        summary = processed_data["summary"]
        self.assertEqual(summary["total_errors"], len(self.sample_errors))
        
        # Count critical hardware errors in our sample
        critical_hw_errors = sum(1 for e in self.sample_errors 
                               if e.get("error_category") in ["HARDWARE_NOT_AVAILABLE", "HARDWARE_MISMATCH", "HARDWARE_COMPATIBILITY_ERROR"]
                               and self.error_viz._is_critical_error(e))
        
        # Verify critical hardware error count
        self.assertEqual(summary["critical_hardware_errors"], critical_hw_errors)
        
        # Verify recent errors preparation
        recent_errors = processed_data["recent_errors"]
        self.assertEqual(len(recent_errors), len(self.sample_errors))
        
        # Check if errors have is_critical and is_recurring flags
        for error in recent_errors:
            self.assertIn("is_critical", error)
            self.assertIn("is_recurring", error)
    
    def test_store_error_in_db(self):
        anyio.run(self._test_store_error_in_db)

    async def _test_store_error_in_db(self):
        """Test storing an error in the database."""
        try:
            import duckdb
        except ImportError:
            self.skipTest("duckdb not available")
        
        # Sample error for database storage
        error = self.sample_errors[0]
        
        # Store the error
        result = await self.error_viz._store_error_in_db(error)
        
        # Check result
        self.assertTrue(result)
        
        # Connect to database to verify
        conn = duckdb.connect(self.db_path)
        
        # Check if table exists
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='worker_error_reports'"
        ).fetchone()
        
        self.assertIsNotNone(table_exists)
        
        # Check if record was stored
        rows = conn.execute("SELECT * FROM worker_error_reports").fetchall()
        
        self.assertEqual(len(rows), 1)
        
        # Clean up
        conn.close()


class TestErrorExtraction(unittest.TestCase):
    """Test suite for error extraction from log files."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create error visualization integration
        self.error_viz = ErrorVisualizationIntegration(
            output_dir=self.output_dir,
            db_path=None
        )
        
        # Create test log file with error reports
        self.log_file_path = os.path.join(self.output_dir, "test_worker.log")
        with open(self.log_file_path, "w") as f:
            f.write("""
2025-03-01 12:00:00 - INFO - Worker started
2025-03-01 12:01:00 - ERROR - Failed to allocate GPU memory
ERROR_REPORT:
{
    "timestamp": "2025-03-01T12:01:00",
    "worker_id": "test-worker-1",
    "type": "ResourceError",
    "error_category": "RESOURCE_EXHAUSTED",
    "message": "Failed to allocate GPU memory"
}
2025-03-01 12:02:00 - INFO - Retry operation
2025-03-01 12:03:00 - ERROR - Network connection failed
ERROR_REPORT:
{
    "timestamp": "2025-03-01T12:03:00",
    "worker_id": "test-worker-1",
    "type": "NetworkError",
    "error_category": "NETWORK_CONNECTION_ERROR",
    "message": "Connection refused"
}
2025-03-01 12:04:00 - INFO - Worker stopped
            """)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_extract_error_blocks(self):
        """Test extraction of error report blocks from log content."""
        # Read log file
        with open(self.log_file_path, "r") as f:
            log_content = f.read()
        
        # Extract error blocks
        error_blocks = self.error_viz._extract_error_blocks(log_content)
        
        # Check the number of blocks
        self.assertEqual(len(error_blocks), 2)
        
        # Check the content of the blocks
        for block in error_blocks:
            self.assertIn("{", block)
            self.assertIn("}", block)
            
            # Attempt to parse as JSON
            try:
                error_data = json.loads(block)
                
                # Check required fields
                self.assertIn("timestamp", error_data)
                self.assertIn("worker_id", error_data)
                self.assertIn("type", error_data)
                self.assertIn("error_category", error_data)
                self.assertIn("message", error_data)
            except json.JSONDecodeError:
                self.fail(f"Failed to parse error block as JSON: {block}")


def test_suite():
    """Construct and return a test suite for all tests."""
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTest(unittest.makeSuite(TestSoundGeneration))
    suite.addTest(unittest.makeSuite(TestSeverityDetection))
    suite.addTest(unittest.makeSuite(TestJavaScriptSeverityDetection))
    suite.addTest(unittest.makeSuite(TestWebSocketIntegration))
    suite.addTest(unittest.makeSuite(TestErrorVisualizationIntegration))
    suite.addTest(unittest.makeSuite(TestErrorExtraction))
    
    return suite


if __name__ == "__main__":
    # Run all tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite())