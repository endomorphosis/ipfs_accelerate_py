#!/usr/bin/env python3
"""
Test the Error Visualization System.

This script tests the Error Visualization integration for the Distributed Testing Framework,
ensuring that it properly analyzes and visualizes error data.
"""

import os
import sys
import json
import unittest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import anyio

# Add parent directory to path to import the modules
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data.duckdb.distributed_testing.dashboard.error_visualization_integration import ErrorVisualizationIntegration

class TestErrorVisualization(unittest.TestCase):
    """Test cases for the Error Visualization System."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test output
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create test database file
        self.db_path = os.path.join(self.output_dir, "test_error_visualization.duckdb")
        
        # Create error visualization integration
        self.error_viz = ErrorVisualizationIntegration(
            output_dir=self.output_dir,
            db_path=None  # We'll use test data instead of a real database
        )
        
        # Generate sample error data
        self.sample_errors = self.generate_sample_errors()
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def generate_sample_errors(self):
        """Generate sample error data for testing."""
        errors = []
        
        # Generate errors of different categories
        error_types = [
            ("ResourceError", "Out of memory: CUDA out of memory", "RESOURCE_EXHAUSTED"),
            ("NetworkError", "Connection reset by peer", "NETWORK_CONNECTION_ERROR"),
            ("HardwareError", "Device unavailable", "HARDWARE_NOT_AVAILABLE"),
            ("WorkerError", "Worker crashed unexpectedly", "WORKER_CRASHED"),
            ("TestError", "Test execution failed", "TEST_ASSERTION_ERROR"),
            ("ResourceError", "Unable to free CUDA memory", "RESOURCE_UNAVAILABLE"),
            ("NetworkError", "Request timed out", "NETWORK_TIMEOUT"),
            ("HardwareError", "GPU throttling detected", "HARDWARE_COMPATIBILITY_ERROR"),
            ("WorkerError", "Worker overloaded", "WORKER_TIMEOUT"),
            ("TestError", "Missing test dependency", "TEST_DEPENDENCY_ERROR")
        ]
        
        # Generate 10 errors for each type with different timestamps
        for i in range(10):
            for error_type, message, category in error_types:
                # Create timestamp with varying offsets
                hours_ago = i * 2  # Spread errors over time
                timestamp = datetime.now() - timedelta(hours=hours_ago)
                
                # Create error report manually instead of using EnhancedErrorReporter
                error_report = {
                    "timestamp": timestamp.isoformat(),
                    "worker_id": f"test-worker-{i % 4 + 1}",
                    "type": error_type,
                    "error_category": category,
                    "message": f"{message} (instance {i+1})",
                    "traceback": f"Traceback for {message} (instance {i+1})",
                    "task_id": f"task-{i+1}",
                    "system_context": {
                        "hostname": f"test-node-{i % 5 + 1}",
                        "platform": "linux",
                        "architecture": "x86_64",
                        "python_version": "3.10.0",
                        "metrics": {
                            "cpu": {
                                "percent": 50 + (i * 5) % 50,
                                "count": 8,
                                "physical_count": 4,
                                "frequency_mhz": 3200
                            },
                            "memory": {
                                "used_percent": 60 + (i * 5) % 40,
                                "total_gb": 32,
                                "available_gb": 12
                            },
                            "disk": {
                                "used_percent": 40 + (i * 5) % 50,
                                "total_gb": 512,
                                "free_gb": 256
                            }
                        },
                        "gpu_metrics": {
                            "count": 2,
                            "devices": [
                                {
                                    "index": 0,
                                    "name": "Test GPU 0",
                                    "memory_utilization": 70 + (i * 5) % 30,
                                    "temperature": 60 + (i * 2) % 20
                                },
                                {
                                    "index": 1,
                                    "name": "Test GPU 1",
                                    "memory_utilization": 65 + (i * 4) % 35,
                                    "temperature": 55 + (i * 3) % 25
                                }
                            ]
                        }
                    },
                    "hardware_context": {
                        "hardware_type": "cuda",
                        "hardware_types": ["cuda", "cpu"],
                        "hardware_status": {
                            "overheating": i % 10 == 0,
                            "memory_pressure": i % 8 == 0,
                            "throttling": i % 5 == 0
                        }
                    },
                    "error_frequency": {
                        "recurring": i % 3 == 0,
                        "same_type": {
                            "last_1h": i % 5 + 1,
                            "last_6h": (i % 5 + 1) * 3,
                            "last_24h": (i % 5 + 1) * 6
                        },
                        "similar_message": {
                            "last_1h": max(0, i % 4 - 1),
                            "last_6h": max(0, i % 4 - 1) * 2,
                            "last_24h": max(0, i % 4 - 1) * 4
                        }
                    }
                }
                
                # Add to sample errors
                errors.append(error_report)
        
        # Generate a few recurring errors
        recurring_message = "Connection reset by peer at endpoint XYZ"
        for i in range(5):
            hours_ago = i * 0.5  # Multiple occurrences within a few hours
            timestamp = datetime.now() - timedelta(hours=hours_ago)
            
            # Create recurring error manually
            error_report = {
                "timestamp": timestamp.isoformat(),
                "worker_id": "test-worker-1",
                "type": "NetworkError",
                "error_category": "NETWORK_CONNECTION_ERROR",
                "message": recurring_message,
                "traceback": f"Traceback for {recurring_message}",
                "task_id": f"task-recurring-{i+1}",
                "system_context": {
                    "hostname": "test-node-1",
                    "platform": "linux",
                    "architecture": "x86_64",
                    "python_version": "3.10.0",
                    "metrics": {
                        "cpu": {
                            "percent": 60,
                            "count": 8,
                            "physical_count": 4,
                            "frequency_mhz": 3200
                        },
                        "memory": {
                            "used_percent": 70,
                            "total_gb": 32,
                            "available_gb": 9.6
                        },
                        "disk": {
                            "used_percent": 50,
                            "total_gb": 512,
                            "free_gb": 256
                        }
                    }
                },
                "hardware_context": {
                    "hardware_type": "cuda",
                    "hardware_types": ["cuda", "cpu"],
                    "hardware_status": {
                        "overheating": False,
                        "memory_pressure": False,
                        "throttling": False
                    }
                },
                "error_frequency": {
                    "recurring": True,
                    "same_type": {
                        "last_1h": 5 - i,
                        "last_6h": (5 - i) * 2,
                        "last_24h": (5 - i) * 4
                    },
                    "similar_message": {
                        "last_1h": 5 - i,
                        "last_6h": (5 - i) * 2,
                        "last_24h": (5 - i) * 4
                    }
                }
            }
            
            # Add to sample errors
            errors.append(error_report)
        
        return errors
    
    def test_error_data_processing(self):
        anyio.run(self._test_error_data_processing)

    async def _test_error_data_processing(self):
        """Test processing of error data."""
        # Process sample error data
        processed_data = await self.error_viz._process_error_data(self.sample_errors, 24)
        
        # Verify basic structure
        self.assertIn("summary", processed_data)
        self.assertIn("timestamp", processed_data)
        self.assertIn("recent_errors", processed_data)
        self.assertIn("error_distribution", processed_data)
        self.assertIn("error_patterns", processed_data)
        self.assertIn("worker_errors", processed_data)
        self.assertIn("hardware_errors", processed_data)
        
        # Verify summary data
        summary = processed_data["summary"]
        self.assertIn("total_errors", summary)
        self.assertIn("recurring_errors", summary)
        self.assertIn("resource_errors", summary)
        self.assertIn("network_errors", summary)
        self.assertIn("hardware_errors", summary)
        self.assertIn("critical_hardware_errors", summary)
        
        # Check error counts
        self.assertEqual(summary["total_errors"], len(self.sample_errors))
        self.assertGreater(summary["resource_errors"], 0)
        self.assertGreater(summary["network_errors"], 0)
        self.assertGreater(summary["hardware_errors"], 0)
    
    def test_error_distribution(self):
        anyio.run(self._test_error_distribution)

    async def _test_error_distribution(self):
        """Test error distribution generation."""
        # Generate error distribution
        error_distribution = await self.error_viz._generate_error_distribution(self.sample_errors)
        
        # Verify structure
        self.assertIn("chart_data", error_distribution)
        self.assertIn("categories", error_distribution)
        
        # Check chart data
        chart_data = error_distribution["chart_data"]
        self.assertIn("data", chart_data)
        self.assertIn("layout", chart_data)
        
        # Check categories data
        categories = error_distribution["categories"]
        self.assertIsInstance(categories, list)
        self.assertGreater(len(categories), 0)
        
        # Verify each category has expected fields
        for category in categories:
            self.assertIn("Category", category)
            self.assertIn("Count", category)
    
    def test_error_patterns(self):
        anyio.run(self._test_error_patterns)

    async def _test_error_patterns(self):
        """Test error pattern detection."""
        # Generate error patterns
        error_patterns = await self.error_viz._generate_error_patterns(self.sample_errors)
        
        # Verify structure
        self.assertIn("chart_data", error_patterns)
        self.assertIn("top_patterns", error_patterns)
        
        # Check chart data
        chart_data = error_patterns["chart_data"]
        self.assertIn("data", chart_data)
        self.assertIn("layout", chart_data)
        
        # Check top patterns
        top_patterns = error_patterns["top_patterns"]
        self.assertIsInstance(top_patterns, list)
        self.assertGreater(len(top_patterns), 0)
        
        # Verify recurring patterns are detected
        recurring_pattern_found = False
        for pattern in top_patterns:
            self.assertIn("pattern", pattern)
            self.assertIn("occurrences", pattern)
            self.assertIn("category", pattern)
            self.assertIn("first_seen", pattern)
            self.assertIn("last_seen", pattern)
            
            # Check if our recurring pattern was detected
            if "Connection reset by peer" in pattern["pattern"] and pattern["occurrences"] > 1:
                recurring_pattern_found = True
        
        self.assertTrue(recurring_pattern_found, "Failed to detect recurring error pattern")
    
    def test_worker_error_analysis(self):
        anyio.run(self._test_worker_error_analysis)

    async def _test_worker_error_analysis(self):
        """Test worker error analysis."""
        # Generate worker error analysis
        worker_errors = await self.error_viz._generate_worker_error_analysis(self.sample_errors)
        
        # Verify structure
        self.assertIn("chart_data", worker_errors)
        self.assertIn("worker_stats", worker_errors)
        
        # Check chart data
        chart_data = worker_errors["chart_data"]
        self.assertIn("data", chart_data)
        self.assertIn("layout", chart_data)
        
        # Check worker stats
        worker_stats = worker_errors["worker_stats"]
        self.assertIsInstance(worker_stats, list)
        self.assertGreater(len(worker_stats), 0)
        
        # Verify worker stats fields
        worker = worker_stats[0]
        self.assertIn("worker_id", worker)
        self.assertIn("error_count", worker)
        self.assertIn("most_common_error", worker)
        self.assertIn("last_error_time", worker)
        self.assertIn("status", worker)
        self.assertIn("critical_errors", worker)
    
    def test_hardware_error_analysis(self):
        anyio.run(self._test_hardware_error_analysis)

    async def _test_hardware_error_analysis(self):
        """Test hardware error analysis."""
        # Generate hardware error analysis
        hardware_errors = await self.error_viz._generate_hardware_error_analysis(self.sample_errors)
        
        # Verify structure
        self.assertIn("chart_data", hardware_errors)
        self.assertIn("hardware_status", hardware_errors)
        self.assertIn("recent_errors", hardware_errors)
        
        # Check chart data
        chart_data = hardware_errors["chart_data"]
        self.assertIn("data", chart_data)
        self.assertIn("layout", chart_data)
        
        # Check hardware status
        hardware_status = hardware_errors["hardware_status"]
        self.assertIsInstance(hardware_status, dict)
        self.assertGreater(len(hardware_status), 0)
        
        # Check recent errors
        recent_errors = hardware_errors["recent_errors"]
        self.assertIsInstance(recent_errors, list)
        
        # Verify hardware error counts
        hardware_error_count = 0
        for error in self.sample_errors:
            if error.get("error_category") in [
                "HARDWARE_NOT_AVAILABLE",
                "HARDWARE_MISMATCH",
                "HARDWARE_COMPATIBILITY_ERROR"
            ]:
                hardware_error_count += 1
        
        # Recent errors list might be limited to a maximum number (e.g., 50)
        self.assertLessEqual(len(recent_errors), hardware_error_count)
    
    def test_get_error_data(self):
        anyio.run(self._test_get_error_data)

    async def _test_get_error_data(self):
        """Test the get_error_data method."""
        # Mock the _get_error_data_from_files method to return our sample data
        original_method = self.error_viz._get_error_data_from_files
        
        async def mock_get_error_data(*args, **kwargs):
            return self.sample_errors
        
        self.error_viz._get_error_data_from_files = mock_get_error_data
        
        try:
            # Get error data
            error_data = await self.error_viz.get_error_data(time_range_hours=24)
            
            # Verify structure
            self.assertIn("summary", error_data)
            self.assertIn("timestamp", error_data)
            self.assertIn("recent_errors", error_data)
            self.assertIn("error_distribution", error_data)
            self.assertIn("error_patterns", error_data)
            self.assertIn("worker_errors", error_data)
            self.assertIn("hardware_errors", error_data)
            
            # Verify caching
            self.assertIn(f"errors_24", self.error_viz.error_cache)
            
            # Get data again and verify it's from cache
            cache_time = self.error_viz.error_cache[f"errors_24"][0]
            cached_data = await self.error_viz.get_error_data(time_range_hours=24)
            
            # The timestamp should be the same if it's from cache
            self.assertEqual(error_data["timestamp"], cached_data["timestamp"])
        finally:
            # Restore original method
            self.error_viz._get_error_data_from_files = original_method
    
    def test_group_similar_messages(self):
        """Test grouping of similar error messages."""
        # Simple message grouping test
        messages = [
            "Error code 123: Resource not found",
            "Error code 456: Resource not found",
            "Connection failed with status 500",
            "Connection failed with status 404",
            "Unknown error occurred"
        ]
        
        # Group messages
        groups = self.error_viz._group_similar_messages(messages)
        
        # Print groups for debugging
        print("Message pattern groups:")
        for pattern, indices in groups.items():
            print(f"Pattern: '{pattern}', Indices: {indices}")
            print(f"Messages: {[messages[i] for i in indices]}")
            print()
        
        # Check if some grouping is happening
        has_multiple_items = False
        for indices in groups.values():
            if len(indices) > 1:
                has_multiple_items = True
                break
        
        # Verify that at least one group has multiple items
        self.assertTrue(has_multiple_items, "No pattern groups with multiple messages found")
        
        # Count total messages accounted for
        total_indices = sum(len(indices) for indices in groups.values())
        self.assertEqual(total_indices, len(messages), "Not all messages were grouped")
    
    def test_is_recurring_error(self):
        """Test detection of recurring errors."""
        # Create error with recurring flag set
        error_with_flag = {
            "error_frequency": {
                "recurring": True,
                "same_type": {
                    "last_1h": 1,
                    "last_6h": 1,
                    "last_24h": 1
                }
            }
        }
        
        # Create error with high occurrence count
        error_with_high_count = {
            "error_frequency": {
                "recurring": False,
                "same_type": {
                    "last_1h": 1,
                    "last_6h": 1,
                    "last_24h": 1
                },
                "similar_message": {
                    "last_1h": 3,  # > 2 in last hour
                    "last_6h": 5,
                    "last_24h": 10
                }
            }
        }
        
        # Create error with high 6-hour count
        error_with_high_6h_count = {
            "error_frequency": {
                "recurring": False,
                "same_type": {
                    "last_1h": 1,
                    "last_6h": 1,
                    "last_24h": 1
                },
                "similar_message": {
                    "last_1h": 1,
                    "last_6h": 6,  # > 5 in last 6 hours
                    "last_24h": 10
                }
            }
        }
        
        # Create non-recurring error
        non_recurring_error = {
            "error_frequency": {
                "recurring": False,
                "same_type": {
                    "last_1h": 1,
                    "last_6h": 1,
                    "last_24h": 1
                },
                "similar_message": {
                    "last_1h": 1,
                    "last_6h": 2,
                    "last_24h": 3
                }
            }
        }
        
        # Test cases
        self.assertTrue(self.error_viz._is_recurring_error(error_with_flag))
        self.assertTrue(self.error_viz._is_recurring_error(error_with_high_count))
        self.assertTrue(self.error_viz._is_recurring_error(error_with_high_6h_count))
        self.assertFalse(self.error_viz._is_recurring_error(non_recurring_error))
    
    def test_is_critical_error(self):
        """Test detection of critical errors."""
        # Critical by category
        critical_by_category = {
            "error_category": "HARDWARE_NOT_AVAILABLE"
        }
        
        # Critical by hardware status
        critical_by_hw_status = {
            "error_category": "TEST_ASSERTION_ERROR",
            "hardware_context": {
                "hardware_status": {
                    "overheating": True,
                    "memory_pressure": False,
                    "throttling": False
                }
            }
        }
        
        # Critical by system metrics
        critical_by_metrics = {
            "error_category": "TEST_ASSERTION_ERROR",
            "hardware_context": {
                "hardware_status": {
                    "overheating": False,
                    "memory_pressure": False,
                    "throttling": False
                }
            },
            "system_context": {
                "metrics": {
                    "cpu": {
                        "percent": 95  # > 90%
                    },
                    "memory": {
                        "used_percent": 80
                    },
                    "disk": {
                        "used_percent": 70
                    }
                }
            }
        }
        
        # Non-critical error
        non_critical_error = {
            "error_category": "TEST_EXECUTION_ERROR",
            "hardware_context": {
                "hardware_status": {
                    "overheating": False,
                    "memory_pressure": False,
                    "throttling": False
                }
            },
            "system_context": {
                "metrics": {
                    "cpu": {
                        "percent": 50
                    },
                    "memory": {
                        "used_percent": 60
                    },
                    "disk": {
                        "used_percent": 70
                    }
                }
            }
        }
        
        # Test cases
        self.assertTrue(self.error_viz._is_critical_error(critical_by_category))
        self.assertTrue(self.error_viz._is_critical_error(critical_by_hw_status))
        self.assertTrue(self.error_viz._is_critical_error(critical_by_metrics))
        self.assertFalse(self.error_viz._is_critical_error(non_critical_error))

if __name__ == "__main__":
    unittest.main()