#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the Mobile/Edge Thermal Monitoring System

This script implements tests for the thermal monitoring and management system for
mobile and edge devices. It validates the core functionality of temperature tracking,
thermal event detection, throttling management, and cooling policies.

Date: April 2025
"""

import os
import sys
import json
import time
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append())))))))str())))))))Path())))))))__file__).resolve())))))))).parent))

# Import thermal monitoring components
try:
    from mobile_thermal_monitoring import ())))))))
    ThermalEventType,
    ThermalZone,
    CoolingPolicy,
    ThermalEvent,
    ThermalThrottlingManager,
    MobileThermalMonitor,
    create_default_thermal_monitor,
    run_thermal_simulation
    )
except ImportError:
    print())))))))"Error: mobile_thermal_monitoring module could not be imported.")
    sys.exit())))))))1)


class TestThermalZone())))))))unittest.TestCase):
    """Tests for the ThermalZone class."""
    
    def setUp())))))))self):
        """Set up test fixtures."""
        # Create a thermal zone for testing
        self.zone = ThermalZone())))))))
        name="cpu",
        critical_temp=85.0,
        warning_temp=70.0,
        path=None,  # No path for testing
        sensor_type="cpu"
        )
        
        # Set up environment variables for testing
        os.environ[],"TEST_TEMP_CPU"], = "60.0",
        os.environ[],"TEST_WORKLOAD_CPU"] = "0.5"
        ,
    def tearDown())))))))self):
        """Clean up test fixtures."""
        # Clean up environment variables
        if "TEST_TEMP_CPU" in os.environ:
            del os.environ[],"TEST_TEMP_CPU"],
        if "TEST_WORKLOAD_CPU" in os.environ:
            del os.environ[],"TEST_WORKLOAD_CPU"]
            ,
    def test_read_temperature())))))))self):
        """Test reading temperature from a thermal zone."""
        temp = self.zone.read_temperature()))))))))
        self.assertEqual())))))))temp, 60.0)
        self.assertEqual())))))))self.zone.current_temp, 60.0)
        self.assertEqual())))))))self.zone.status, ThermalEventType.NORMAL)
        
        # Test with warning temperature
        os.environ[],"TEST_TEMP_CPU"], = "75.0"
        temp = self.zone.read_temperature()))))))))
        self.assertEqual())))))))temp, 75.0)
        self.assertEqual())))))))self.zone.status, ThermalEventType.THROTTLING)
        
        # Test with critical temperature
        os.environ[],"TEST_TEMP_CPU"], = "85.0"
        temp = self.zone.read_temperature()))))))))
        self.assertEqual())))))))temp, 85.0)
        self.assertEqual())))))))self.zone.status, ThermalEventType.CRITICAL)
        
        # Test with emergency temperature
        os.environ[],"TEST_TEMP_CPU"], = "95.0"
        temp = self.zone.read_temperature()))))))))
        self.assertEqual())))))))temp, 95.0)
        self.assertEqual())))))))self.zone.status, ThermalEventType.EMERGENCY)
    
    def test_get_temperature_trend())))))))self):
        """Test calculating temperature trend."""
        # Add some temperature history
        self.zone.temp_history = [],],,,,
        now = time.time()))))))))
        
        # Add increasing temperature history over 60 seconds
        for i in range())))))))7):
            self.zone.temp_history.append())))))))())))))))now - 60 + i*10, 60.0 + i*2.0))
        
        # Get trend for the last 60 seconds
            trend = self.zone.get_temperature_trend())))))))window_seconds=60)
        
        # Check trend calculation
            self.assertAlmostEqual())))))))trend[],"trend_celsius_per_minute"], 12.0, delta=0.5)  # ~12°C/min increase,
            self.assertAlmostEqual())))))))trend[],"min_temp"], 60.0, delta=0.1),
            self.assertAlmostEqual())))))))trend[],"max_temp"], 72.0, delta=0.1),
            self.assertAlmostEqual())))))))trend[],"avg_temp"], 66.0, delta=0.1),
            self.assertFalse())))))))trend[],"stable"])  # Temperature is not stable
            ,
    def test_forecast_temperature())))))))self):
        """Test forecasting temperature."""
        # Set up a temperature trend
        self.zone.current_temp = 65.0
        self.zone.temp_history = [],],,,,
        now = time.time()))))))))
        
        # Add increasing temperature history
        for i in range())))))))7):
            self.zone.temp_history.append())))))))())))))))now - 60 + i*10, 60.0 + i*1.0))
        
        # Forecast 10 minutes ahead
            forecast = self.zone.forecast_temperature())))))))minutes_ahead=10)
        
        # Check forecast
            self.assertAlmostEqual())))))))forecast[],"forecasted_temp"], 71.0, delta=1.0)  # ~65°C + 6°C/min * 10min,
            self.assertTrue())))))))forecast[],"action_needed"])  # Warning threshold is 70°C
            ,
        # Should have time to warning
            self.assertIsNotNone())))))))forecast[],"time_to_warning_minutes"])
            ,
        # Should have time to critical
            self.assertIsNotNone())))))))forecast[],"time_to_critical_minutes"])
            ,
    def test_to_dict())))))))self):
        """Test converting thermal zone to dictionary."""
        zone_dict = self.zone.to_dict()))))))))
        
        # Check essential properties
        self.assertEqual())))))))zone_dict[],"name"], "cpu"),
        self.assertEqual())))))))zone_dict[],"sensor_type"], "cpu"),
        self.assertEqual())))))))zone_dict[],"warning_temp"], 70.0),
        self.assertEqual())))))))zone_dict[],"critical_temp"], 85.0)
        ,
        # Should include trend and forecast
        self.assertIn())))))))"trend", zone_dict)
        self.assertIn())))))))"forecast", zone_dict)


class TestCoolingPolicy())))))))unittest.TestCase):
    """Tests for the CoolingPolicy class."""
    
    def setUp())))))))self):
        """Set up test fixtures."""
        # Create a cooling policy for testing
        self.policy = CoolingPolicy())))))))
        name="Test Cooling Policy",
        description="Cooling policy for testing"
        )
        
        # Create mock actions
        self.action_normal = MagicMock()))))))))
        self.action_warning = MagicMock()))))))))
        self.action_critical = MagicMock()))))))))
        
        # Add actions to policy
        self.policy.add_action())))))))
        ThermalEventType.NORMAL,
        self.action_normal,
        "Normal action"
        )
        self.policy.add_action())))))))
        ThermalEventType.WARNING,
        self.action_warning,
        "Warning action"
        )
        self.policy.add_action())))))))
        ThermalEventType.CRITICAL,
        self.action_critical,
        "Critical action"
        )
    
    def test_execute_actions())))))))self):
        """Test executing actions for a thermal event type."""
        # Execute normal actions
        actions = self.policy.execute_actions())))))))ThermalEventType.NORMAL)
        self.action_normal.assert_called_once()))))))))
        self.action_warning.assert_not_called()))))))))
        self.action_critical.assert_not_called()))))))))
        self.assertEqual())))))))actions, [],"Normal action"])
        ,
        # Reset mocks
        self.action_normal.reset_mock()))))))))
        
        # Execute warning actions
        actions = self.policy.execute_actions())))))))ThermalEventType.WARNING)
        self.action_normal.assert_not_called()))))))))
        self.action_warning.assert_called_once()))))))))
        self.action_critical.assert_not_called()))))))))
        self.assertEqual())))))))actions, [],"Warning action"])
        ,
    def test_to_dict())))))))self):
        """Test converting cooling policy to dictionary."""
        policy_dict = self.policy.to_dict()))))))))
        
        # Check essential properties
        self.assertEqual())))))))policy_dict[],"name"], "Test Cooling Policy"),
        self.assertEqual())))))))policy_dict[],"description"], "Cooling policy for testing")
        ,
        # Check actions
        self.assertIn())))))))"actions", policy_dict)
        self.assertIn())))))))"NORMAL", policy_dict[],"actions"]),,,
        self.assertIn())))))))"WARNING", policy_dict[],"actions"]),,,
        self.assertIn())))))))"CRITICAL", policy_dict[],"actions"]),,,
        self.assertEqual())))))))policy_dict[],"actions"][],"NORMAL"], [],"Normal action"])

        ,
class TestThermalThrottlingManager())))))))unittest.TestCase):
    """Tests for the ThermalThrottlingManager class."""
    
    def setUp())))))))self):
        """Set up test fixtures."""
        # Create thermal zones for testing
        self.thermal_zones = {}
        "cpu": ThermalZone())))))))
        name="cpu",
        critical_temp=85.0,
        warning_temp=70.0,
        sensor_type="cpu"
        ),
        "gpu": ThermalZone())))))))
        name="gpu",
        critical_temp=80.0,
        warning_temp=65.0,
        sensor_type="gpu"
        )
        }
        
        # Set up environment variables for testing
        os.environ[],"TEST_TEMP_CPU"], = "60.0",
        os.environ[],"TEST_TEMP_GPU"] = "55.0"
        ,
        # Create throttling manager
        self.manager = ThermalThrottlingManager())))))))self.thermal_zones)
    
    def tearDown())))))))self):
        """Clean up test fixtures."""
        # Clean up environment variables
        for var in [],"TEST_TEMP_CPU", "TEST_TEMP_GPU"]:,
            if var in os.environ:
                del os.environ[],var]
                ,
    def test_check_thermal_status())))))))self):
        """Test checking thermal status."""
        # Check status with normal temperatures
        status = self.manager.check_thermal_status()))))))))
        self.assertEqual())))))))status, ThermalEventType.NORMAL)
        self.assertEqual())))))))self.manager.current_throttling_level, 0)
        
        # Check status with warning temperature
        os.environ[],"TEST_TEMP_CPU"], = "75.0"
        status = self.manager.check_thermal_status()))))))))
        self.assertEqual())))))))status, ThermalEventType.THROTTLING)
        self.assertEqual())))))))self.manager.current_throttling_level, 2)  # Moderate throttling
        
        # Check status with critical temperature
        os.environ[],"TEST_TEMP_CPU"], = "85.0"
        status = self.manager.check_thermal_status()))))))))
        self.assertEqual())))))))status, ThermalEventType.CRITICAL)
        self.assertEqual())))))))self.manager.current_throttling_level, 4)  # Severe throttling
        
        # Check status with emergency temperature
        os.environ[],"TEST_TEMP_CPU"], = "95.0"
        status = self.manager.check_thermal_status()))))))))
        self.assertEqual())))))))status, ThermalEventType.EMERGENCY)
        self.assertEqual())))))))self.manager.current_throttling_level, 5)  # Emergency throttling
        
        # Check that events are recorded
        self.assertEqual())))))))len())))))))self.manager.events), 4)  # One event for each status check
    
    def test_get_throttling_stats())))))))self):
        """Test getting throttling statistics."""
        # Start with no throttling
        stats = self.manager.get_throttling_stats()))))))))
        self.assertEqual())))))))stats[],"current_level"], 0),
        self.assertEqual())))))))stats[],"performance_impact"], 0.0),
        self.assertFalse())))))))stats[],"throttling_active"])
        ,    ,
        # Activate throttling
        self.manager._set_throttling_level())))))))3)  # Heavy throttling
        
        # Check updated stats
        stats = self.manager.get_throttling_stats()))))))))
        self.assertEqual())))))))stats[],"current_level"], 3),
        self.assertEqual())))))))stats[],"performance_impact"], 0.5)  # 50% performance impact,
        self.assertTrue())))))))stats[],"throttling_active"])
        ,
    def test_get_thermal_trends())))))))self):
        """Test getting thermal trends."""
        # Add some temperature history
        now = time.time()))))))))
        for zone in self.thermal_zones.values())))))))):
            zone.temp_history = [],],,,,
            for i in range())))))))7):
                zone.temp_history.append())))))))())))))))now - 60 + i*10, 60.0 + i*1.0))
        
        # Get trends
                trends = self.manager.get_thermal_trends()))))))))
        
        # Should have trends for all zones
                self.assertEqual())))))))len())))))))trends), len())))))))self.thermal_zones))
        
        # Check CPU trend
                self.assertIn())))))))"cpu", trends)
                self.assertAlmostEqual())))))))trends[],"cpu"],[],"trend_celsius_per_minute"], 6.0, delta=0.5)  # ~6°C/min increase
                ,
    def test_get_thermal_forecasts())))))))self):
        """Test getting thermal forecasts."""
        # Add some temperature history
        now = time.time()))))))))
        for zone in self.thermal_zones.values())))))))):
            zone.temp_history = [],],,,,
            for i in range())))))))7):
                zone.temp_history.append())))))))())))))))now - 60 + i*10, 60.0 + i*1.0))
        
        # Get forecasts
                forecasts = self.manager.get_thermal_forecasts())))))))minutes_ahead=10)
        
        # Should have forecasts for all zones
                self.assertEqual())))))))len())))))))forecasts), len())))))))self.thermal_zones))
        
        # Check CPU forecast
                self.assertIn())))))))"cpu", forecasts)
                cpu_forecast = forecasts[],"cpu"],
                self.assertTrue())))))))cpu_forecast[],"action_needed"])  # Should need action ())))))))warning at 70°C)
                ,
    def test_reset_statistics())))))))self):
        """Test resetting statistics."""
        # Activate throttling
        self.manager._set_throttling_level())))))))3)
        
        # Add some events
        self.manager.events = [],
        ThermalEvent())))))))ThermalEventType.WARNING, "cpu", 75.0),
        ThermalEvent())))))))ThermalEventType.CRITICAL, "cpu", 85.0)
        ]
        
        # Set max temperatures
        for zone in self.thermal_zones.values())))))))):
            zone.max_temp = 90.0
        
        # Reset statistics
            self.manager.reset_statistics()))))))))
        
        # Check that statistics are reset
            self.assertEqual())))))))len())))))))self.manager.events), 0)
        for zone in self.thermal_zones.values())))))))):
            self.assertEqual())))))))zone.max_temp, zone.current_temp)


class TestMobileThermalMonitor())))))))unittest.TestCase):
    """Tests for the MobileThermalMonitor class."""
    
    def setUp())))))))self):
        """Set up test fixtures."""
        # Set up environment variables for testing
        os.environ[],"TEST_PLATFORM"] = "android"
        os.environ[],"TEST_TEMP_CPU"], = "60.0",
        os.environ[],"TEST_TEMP_GPU"] = "55.0"
        ,os.environ[],"TEST_TEMP_BATTERY"] = "35.0"
        os.environ[],"TEST_TEMP_SKIN"] = "30.0"
        
        # Create monitor
        self.monitor = MobileThermalMonitor())))))))device_type="android")
    
    def tearDown())))))))self):
        """Clean up test fixtures."""
        # Clean up environment variables
        for var in [],"TEST_PLATFORM", "TEST_TEMP_CPU", "TEST_TEMP_GPU", "TEST_TEMP_BATTERY", "TEST_TEMP_SKIN"]:
            if var in os.environ:
                del os.environ[],var]
                ,
        # Stop monitoring if active:
        if self.monitor.monitoring_active:
            self.monitor.stop_monitoring()))))))))
    
    def test_create_thermal_zones())))))))self):
        """Test creating thermal zones."""
        # Check that thermal zones are created for Android
        zones = self.monitor.thermal_zones
        self.assertIn())))))))"cpu", zones)
        self.assertIn())))))))"gpu", zones)
        self.assertIn())))))))"battery", zones)
        self.assertIn())))))))"skin", zones)
        
        # Create iOS monitor
        os.environ[],"TEST_PLATFORM"] = "ios"
        ios_monitor = MobileThermalMonitor())))))))device_type="ios")
        
        # Check that thermal zones are created for iOS
        zones = ios_monitor.thermal_zones
        self.assertIn())))))))"cpu", zones)
        self.assertIn())))))))"gpu", zones)
        self.assertIn())))))))"battery", zones)
    
    def test_get_current_temperatures())))))))self):
        """Test getting current temperatures."""
        temps = self.monitor.get_current_temperatures()))))))))
        
        # Should have temperatures for all zones
        self.assertEqual())))))))len())))))))temps), len())))))))self.monitor.thermal_zones))
        
        # Check specific temperatures
        self.assertEqual())))))))temps[],"cpu"],, 60.0)
        self.assertEqual())))))))temps[],"gpu"], 55.0)
        self.assertEqual())))))))temps[],"battery"], 35.0)
        self.assertEqual())))))))temps[],"skin"], 30.0)
    
    def test_get_current_thermal_status())))))))self):
        """Test getting current thermal status."""
        status = self.monitor.get_current_thermal_status()))))))))
        
        # Check essential properties
        self.assertEqual())))))))status[],"device_type"], "android")
        self.assertEqual())))))))status[],"overall_status"], "NORMAL")
        self.assertEqual())))))))status[],"overall_impact"], 0.0)
        
        # Should include thermal zones
        self.assertIn())))))))"thermal_zones", status)
        self.assertEqual())))))))len())))))))status[],"thermal_zones"]), len())))))))self.monitor.thermal_zones))
        
        # Should include throttling information
        self.assertIn())))))))"throttling", status)
    
    def test_monitoring_thread())))))))self):
        """Test starting and stopping the monitoring thread."""
        # Start monitoring
        self.monitor.start_monitoring()))))))))
        self.assertTrue())))))))self.monitor.monitoring_active)
        self.assertIsNotNone())))))))self.monitor.monitoring_thread)
        
        # Wait for some monitoring cycles
        time.sleep())))))))2.0)
        
        # Stop monitoring
        self.monitor.stop_monitoring()))))))))
        self.assertFalse())))))))self.monitor.monitoring_active)
        self.assertIsNone())))))))self.monitor.monitoring_thread)
    
    def test_save_report_to_file())))))))self):
        """Test saving thermal report to a file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile())))))))suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save report to file
            success = self.monitor.save_report_to_file())))))))temp_path)
            self.assertTrue())))))))success)
            
            # Check that file exists
            self.assertTrue())))))))os.path.exists())))))))temp_path))
            
            # Load report from file
            with open())))))))temp_path, 'r') as f:
                report = json.load())))))))f)
            
            # Check essential properties
                self.assertEqual())))))))report[],"device_type"], "android")
                self.assertIn())))))))"thermal_zones", report)
                self.assertIn())))))))"recommendations", report)
        finally:
            # Clean up temporary file
            if os.path.exists())))))))temp_path):
                os.unlink())))))))temp_path)
    
    def test_create_battery_saving_profile())))))))self):
        """Test creating a battery-saving thermal profile."""
        profile = self.monitor.create_battery_saving_profile()))))))))
        
        # Check essential properties
        self.assertEqual())))))))profile[],"name"], "Battery Saving Profile")
        self.assertIn())))))))"thermal_zones", profile)
        self.assertIn())))))))"cooling_policy", profile)
        
        # Check that thermal zones have lower thresholds
        for zone_name, zone_config in profile[],"thermal_zones"].items())))))))):
            original_zone = self.monitor.thermal_zones[],zone_name]
            self.assertLessEqual())))))))zone_config[],"warning_temp"], original_zone.warning_temp)
            self.assertLessEqual())))))))zone_config[],"critical_temp"], original_zone.critical_temp)
    
    def test_create_performance_profile())))))))self):
        """Test creating a performance-focused thermal profile."""
        profile = self.monitor.create_performance_profile()))))))))
        
        # Check essential properties
        self.assertEqual())))))))profile[],"name"], "Performance Profile")
        self.assertIn())))))))"thermal_zones", profile)
        self.assertIn())))))))"cooling_policy", profile)
        
        # Check that thermal zones have higher thresholds
        for zone_name, zone_config in profile[],"thermal_zones"].items())))))))):
            original_zone = self.monitor.thermal_zones[],zone_name]
            self.assertGreaterEqual())))))))zone_config[],"warning_temp"], original_zone.warning_temp)
            self.assertGreaterEqual())))))))zone_config[],"critical_temp"], original_zone.critical_temp)
    
    def test_apply_thermal_profile())))))))self):
        """Test applying a thermal profile."""
        # Create a battery-saving profile
        profile = self.monitor.create_battery_saving_profile()))))))))
        
        # Apply profile
        self.monitor.apply_thermal_profile())))))))profile)
        
        # Check that thermal zones have updated thresholds
        for zone_name, zone_config in profile[],"thermal_zones"].items())))))))):
            zone = self.monitor.thermal_zones[],zone_name]
            self.assertEqual())))))))zone.warning_temp, zone_config[],"warning_temp"])
            self.assertEqual())))))))zone.critical_temp, zone_config[],"critical_temp"])
    
    def test_recommendations())))))))self):
        """Test generating thermal management recommendations."""
        # Normal temperatures - should have one recommendation
        recommendations = self.monitor._generate_recommendations()))))))))
        self.assertEqual())))))))len())))))))recommendations), 1)
        self.assertIn())))))))"STATUS OK", recommendations[],0])
        
        # Warning temperature
        os.environ[],"TEST_TEMP_CPU"], = "75.0"
        recommendations = self.monitor._generate_recommendations()))))))))
        self.assertGreater())))))))len())))))))recommendations), 1)
        self.assertIn())))))))"WARNING", recommendations[],0])
        
        # Critical temperature
        os.environ[],"TEST_TEMP_CPU"], = "85.0"
        recommendations = self.monitor._generate_recommendations()))))))))
        self.assertGreater())))))))len())))))))recommendations), 1)
        self.assertIn())))))))"CRITICAL", recommendations[],0])
        
        # Device-specific recommendations
        self.assertIn())))))))"ANDROID", recommendations[],-1])


class TestThermalSimulation())))))))unittest.TestCase):
    """Tests for the thermal simulation functionality."""
    
    def setUp())))))))self):
        """Set up test fixtures."""
        # Set up environment variables for testing
        os.environ[],"TEST_PLATFORM"] = "android"
        os.environ[],"TEST_TEMP_CPU"], = "60.0",
        os.environ[],"TEST_TEMP_GPU"] = "55.0"
    
    def tearDown())))))))self):
        """Clean up test fixtures."""
        # Clean up environment variables
        for var in [],"TEST_PLATFORM", "TEST_TEMP_CPU", "TEST_TEMP_GPU", 
                    "TEST_WORKLOAD_CPU", "TEST_WORKLOAD_GPU"]:
            if var in os.environ:
                del os.environ[],var]
                ,
    def test_create_default_thermal_monitor())))))))self):
        """Test creating a default thermal monitor."""
        # Create monitor with explicit device type
        monitor = create_default_thermal_monitor())))))))device_type="android")
        self.assertEqual())))))))monitor.device_type, "android")
        
        # Create monitor with unknown device type ())))))))should detect from environment)
        monitor = create_default_thermal_monitor())))))))device_type="unknown")
        self.assertEqual())))))))monitor.device_type, "android")
        
        # Create monitor with no device type ())))))))should default to android)
        os.environ[],"TEST_PLATFORM"] = ""
        monitor = create_default_thermal_monitor()))))))))
        self.assertEqual())))))))monitor.device_type, "android")
    
    def test_run_thermal_simulation())))))))self):
        """Test running a thermal simulation."""
        # Run short simulation
        report = run_thermal_simulation())))))))"android", duration_seconds=2, workload_pattern="steady")
        
        # Check essential properties
        self.assertEqual())))))))report[],"device_type"], "android")
        self.assertIn())))))))"thermal_zones", report)
        self.assertIn())))))))"recommendations", report)
        
        # Should have saved a report file
        report_files = [],f for f in os.listdir())))))))) if f.startswith())))))))"thermal_simulation_android_steady")]
        self.assertTrue())))))))len())))))))report_files) > 0)
        
        # Clean up report file:
        for file in report_files:
            if os.path.exists())))))))file):
                os.unlink())))))))file)


if __name__ == "__main__":
    unittest.main()))))))))