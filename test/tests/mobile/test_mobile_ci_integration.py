#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the Mobile CI Integration Tools.

This script tests the Mobile CI Integration components:
1. merge_benchmark_databases.py
2. check_mobile_regressions.py
3. generate_mobile_dashboard.py

It creates test data and verifies that each component works correctly.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import the components
from test.merge_benchmark_databases import BenchmarkDatabaseMerger
from test.check_mobile_regressions import MobileRegressionDetector
from test.generate_mobile_dashboard import MobileDashboardGenerator


class TestMobileCIIntegration(unittest.TestCase):
    """Test class for Mobile CI Integration components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name
        
        # Create test data paths
        self.android_db_path = os.path.join(self.temp_path, "android_results.duckdb")
        self.ios_db_path = os.path.join(self.temp_path, "ios_results.duckdb")
        self.merged_db_path = os.path.join(self.temp_path, "merged_results.duckdb")
        self.analysis_json_path = os.path.join(self.temp_path, "analysis_results.json")
        self.regression_report_path = os.path.join(self.temp_path, "regression_report.md")
        self.dashboard_path = os.path.join(self.temp_path, "mobile_dashboard.html")
        
        # Create dummy database files
        with open(self.android_db_path, 'w') as f:
            f.write("mock duckdb file")
        
        with open(self.ios_db_path, 'w') as f:
            f.write("mock duckdb file")
        
        # Create test analysis data
        self.create_test_analysis_data()
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def create_test_analysis_data(self):
        """Create test analysis data in JSON format."""
        analysis_data = {
            "timestamp": "2025-04-01T12:00:00Z",
            "platforms": {
                "android": {
                    "devices": {
                        "Pixel 4": {
                            "device_id": "emulator-5554",
                            "android_version": "11",
                            "chipset": "Snapdragon 855"
                        }
                    },
                    "models": {
                        "bert-base-uncased": {
                            "batch_sizes": {
                                "1": {
                                    "throughput": 10.5,
                                    "latency": 95.2,
                                    "memory_mb": 420.5,
                                    "battery_impact": 12.3
                                },
                                "4": {
                                    "throughput": 32.1,
                                    "latency": 124.6,
                                    "memory_mb": 450.2,
                                    "battery_impact": 15.7
                                }
                            }
                        },
                        "mobilenet-v2": {
                            "batch_sizes": {
                                "1": {
                                    "throughput": 45.3,
                                    "latency": 22.1,
                                    "memory_mb": 110.5,
                                    "battery_impact": 5.2
                                },
                                "4": {
                                    "throughput": 120.8,
                                    "latency": 33.2,
                                    "memory_mb": 115.8,
                                    "battery_impact": 7.5
                                }
                            }
                        }
                    }
                },
                "ios": {
                    "devices": {
                        "iPhone 12": {
                            "device_id": "00008101-001D38810168001E",
                            "ios_version": "15.5",
                            "neural_engine": True
                        }
                    },
                    "models": {
                        "bert-base-uncased": {
                            "batch_sizes": {
                                "1": {
                                    "throughput": 12.3,
                                    "latency": 81.5,
                                    "memory_mb": 380.2,
                                    "battery_impact": 10.1
                                },
                                "4": {
                                    "throughput": 38.5,
                                    "latency": 103.8,
                                    "memory_mb": 402.5,
                                    "battery_impact": 13.2
                                }
                            }
                        },
                        "mobilenet-v2": {
                            "batch_sizes": {
                                "1": {
                                    "throughput": 58.9,
                                    "latency": 17.0,
                                    "memory_mb": 95.2,
                                    "battery_impact": 4.3
                                },
                                "4": {
                                    "throughput": 150.2,
                                    "latency": 26.7,
                                    "memory_mb": 105.1,
                                    "battery_impact": 6.2
                                }
                            }
                        }
                    }
                }
            },
            "models": {
                "bert-base-uncased": {
                    "platforms": {
                        "android": {
                            "batch_sizes": {
                                "1": {
                                    "throughput": 10.5,
                                    "latency": 95.2
                                },
                                "4": {
                                    "throughput": 32.1,
                                    "latency": 124.6
                                }
                            }
                        },
                        "ios": {
                            "batch_sizes": {
                                "1": {
                                    "throughput": 12.3,
                                    "latency": 81.5
                                },
                                "4": {
                                    "throughput": 38.5,
                                    "latency": 103.8
                                }
                            }
                        }
                    }
                },
                "mobilenet-v2": {
                    "platforms": {
                        "android": {
                            "batch_sizes": {
                                "1": {
                                    "throughput": 45.3,
                                    "latency": 22.1
                                },
                                "4": {
                                    "throughput": 120.8,
                                    "latency": 33.2
                                }
                            }
                        },
                        "ios": {
                            "batch_sizes": {
                                "1": {
                                    "throughput": 58.9,
                                    "latency": 17.0
                                },
                                "4": {
                                    "throughput": 150.2,
                                    "latency": 26.7
                                }
                            }
                        }
                    }
                }
            }
        }
        
        with open(self.analysis_json_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
    
    def test_1_database_merger_initialization(self):
        """Test initialization of BenchmarkDatabaseMerger."""
        try:
            merger = BenchmarkDatabaseMerger(
                output_db=self.merged_db_path,
                input_dbs=[self.android_db_path, self.ios_db_path]
            )
            self.assertEqual(merger.output_db, self.merged_db_path)
            self.assertEqual(len(merger.input_dbs), 2)
            self.assertEqual(merger.input_dbs[0], self.android_db_path)
            self.assertEqual(merger.input_dbs[1], self.ios_db_path)
        except Exception as e:
            self.fail(f"BenchmarkDatabaseMerger initialization failed: {e}")
    
    def test_2_find_input_files(self):
        """Test finding input files."""
        merger = BenchmarkDatabaseMerger(
            output_db=self.merged_db_path,
            input_dbs=[self.android_db_path, self.ios_db_path]
        )
        
        input_files = merger.find_input_files()
        self.assertEqual(len(input_files), 2)
        self.assertTrue(self.android_db_path in input_files)
        self.assertTrue(self.ios_db_path in input_files)
    
    def test_3_regression_detector_initialization(self):
        """Test initialization of MobileRegressionDetector."""
        try:
            detector = MobileRegressionDetector(
                data_file=self.analysis_json_path,
                threshold=15.0
            )
            self.assertEqual(detector.data_file, self.analysis_json_path)
            self.assertEqual(detector.threshold, 15.0)
        except Exception as e:
            self.fail(f"MobileRegressionDetector initialization failed: {e}")
    
    def test_4_load_analysis_data(self):
        """Test loading analysis data."""
        detector = MobileRegressionDetector(
            data_file=self.analysis_json_path
        )
        
        result = detector.load_current_data()
        self.assertTrue(result)
        self.assertTrue(isinstance(detector.current_data, dict))
        self.assertTrue("platforms" in detector.current_data)
        self.assertTrue("models" in detector.current_data)
    
    def test_5_dashboard_generator_initialization(self):
        """Test initialization of MobileDashboardGenerator."""
        try:
            generator = MobileDashboardGenerator(
                data_file=self.analysis_json_path,
                output_path=self.dashboard_path,
                theme="dark"
            )
            self.assertEqual(generator.data_file, self.analysis_json_path)
            self.assertEqual(generator.output_path, self.dashboard_path)
            self.assertEqual(generator.theme, "dark")
        except Exception as e:
            self.fail(f"MobileDashboardGenerator initialization failed: {e}")
    
    def test_6_load_dashboard_data(self):
        """Test loading dashboard data."""
        generator = MobileDashboardGenerator(
            data_file=self.analysis_json_path,
            output_path=self.dashboard_path
        )
        
        result = generator.load_data()
        self.assertTrue(result)
        self.assertTrue(isinstance(generator.data, dict))
        self.assertTrue("platforms" in generator.data)
        self.assertTrue("models" in generator.data)


if __name__ == "__main__":
    unittest.main()