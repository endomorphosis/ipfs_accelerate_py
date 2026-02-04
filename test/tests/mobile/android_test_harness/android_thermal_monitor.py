#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Android Thermal Monitoring and Management

This module provides specialized tools for monitoring and managing thermal conditions
on Android devices during model execution, including throttling detection, temperature
trends, and cooling policy enforcement.

Features:
    - Real-time temperature monitoring for Android devices
    - Thermal zone mapping and analysis
    - Throttling detection and measurement
    - Cooling policy implementation
    - Temperature forecasting
    - Battery impact correlation
    - Performance correlation with thermal conditions

Date: April 2025
"""

import os
import sys
import time
import json
import logging
import datetime
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from enum import Enum, auto

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Local imports
from test.tests.mobile.android_test_harness.android_test_harness import AndroidDevice

try:
    from mobile_thermal_monitoring import (
        ThermalEventType,
        ThermalZone,
        CoolingPolicy,
        ThermalEvent
    )
    THERMAL_MONITORING_AVAILABLE = True
except ImportError:
    logger.warning("mobile_thermal_monitoring module not available. Using local implementations.")
    THERMAL_MONITORING_AVAILABLE = False
    
    # Define local implementations if imports fail
    class ThermalEventType(Enum):
        """Types of thermal events that can be detected."""
        NORMAL = auto()
        WARNING = auto()
        THROTTLING = auto()
        CRITICAL = auto()
        EMERGENCY = auto()
    
    class ThermalZone:
        """Represents a thermal monitoring zone in a device."""
        
        def __init__(self, name: str, critical_temp: float, warning_temp: float, 
                     path: Optional[str] = None, sensor_type: str = "unknown"):
            """Initialize a thermal zone."""
            self.name = name
            self.critical_temp = critical_temp
            self.warning_temp = warning_temp
            self.path = path
            self.sensor_type = sensor_type
            self.current_temp = 0.0
            self.baseline_temp = 0.0
            self.max_temp = 0.0
            self.temp_history = []
            self.status = ThermalEventType.NORMAL
    
    class CoolingPolicy:
        """Defines a cooling policy for thermal management."""
        
        def __init__(self, name: str, description: str):
            """Initialize a cooling policy."""
            self.name = name
            self.description = description
            self.actions = {
                ThermalEventType.NORMAL: [],
                ThermalEventType.WARNING: [],
                ThermalEventType.THROTTLING: [],
                ThermalEventType.CRITICAL: [],
                ThermalEventType.EMERGENCY: []
            }
    
    class ThermalEvent:
        """Represents a thermal event that occurred."""
        
        def __init__(self, event_type: ThermalEventType, zone_name: str, 
                     temperature: float, timestamp: Optional[float] = None):
            """Initialize a thermal event."""
            self.event_type = event_type
            self.zone_name = zone_name
            self.temperature = temperature
            self.timestamp = timestamp or time.time()
            self.actions_taken = []
            self.impact_score = 0.0


class AndroidThermalZone(ThermalZone):
    """
    Thermal zone implementation specifically for Android devices.
    
    Extends the base ThermalZone class with Android-specific functionality
    for temperature reading and management.
    """
    
    def __init__(self, device: AndroidDevice, name: str, critical_temp: float, warning_temp: float,
                 path: Optional[str] = None, zone_type: Optional[str] = None):
        """
        Initialize an Android thermal zone.
        
        Args:
            device: Android device
            name: Name of the thermal zone
            critical_temp: Critical temperature threshold in Celsius
            warning_temp: Warning temperature threshold in Celsius
            path: Optional specific path to thermal zone on device
            zone_type: Optional thermal zone type
        """
        super().__init__(name, critical_temp, warning_temp, path, zone_type or name)
        self.device = device
        
        # Determine the thermal zone path if not provided
        if not self.path:
            self._find_thermal_zone_path()
    
    def _find_thermal_zone_path(self) -> None:
        """Find the thermal zone path on the Android device."""
        if not self.device or not self.device.connected:
            logger.warning(f"Cannot find thermal zone path: device not connected")
            return
        
        # Get thermal zone types
        result = self.device._adb_command(["shell", "cat", "/sys/class/thermal/thermal_zone*/type"])
        types = result.strip().split('\n')
        
        # Find matching thermal zone
        for i, zone_type in enumerate(types):
            zone_type = zone_type.strip()
            
            # Check for matching zone type
            if (zone_type.lower() == self.name.lower() or 
                self.name.lower() in zone_type.lower() or 
                zone_type.lower() in self.name.lower()):
                
                self.path = f"/sys/class/thermal/thermal_zone{i}/temp"
                self.sensor_type = zone_type
                logger.debug(f"Found thermal zone path for {self.name}: {self.path}")
                return
        
        logger.warning(f"Could not find thermal zone path for {self.name}")
    
    def read_temperature(self) -> float:
        """
        Read the current temperature from the Android thermal zone.
        
        Returns:
            Current temperature in Celsius
        """
        if not self.device or not self.device.connected:
            logger.warning(f"Cannot read temperature: device not connected")
            return self._simulate_temperature()
        
        if self.path:
            result = self.device._adb_command(["shell", "cat", self.path])
            
            try:
                # Thermal zone files typically contain temperature in millidegrees Celsius
                temp_millicelsius = int(result.strip())
                self.current_temp = temp_millicelsius / 1000.0
            except (ValueError, TypeError) as e:
                logger.warning(f"Error reading temperature from {self.path}: {e}")
                # Fall back to simulation
                self.current_temp = self._simulate_temperature()
        else:
            # Fall back to thermal zone mapping
            thermal_info = self.device.get_thermal_info()
            
            # Try to find matching thermal zone
            for zone_type, temp in thermal_info.items():
                if (zone_type.lower() == self.name.lower() or 
                    self.name.lower() in zone_type.lower() or 
                    zone_type.lower() in self.name.lower()):
                    
                    self.current_temp = temp
                    break
            else:
                # If no match found, simulate temperature
                self.current_temp = self._simulate_temperature()
        
        # Update history and maximum temperature
        self.temp_history.append((time.time(), self.current_temp))
        if len(self.temp_history) > 1000:  # Limit history size
            self.temp_history.pop(0)
        
        self.max_temp = max(self.max_temp, self.current_temp)
        
        # Update status based on temperature
        self._update_status()
        
        return self.current_temp


class AndroidThermalMonitor:
    """
    Thermal monitor implementation for Android devices.
    
    Provides tools for monitoring and managing thermal conditions on
    Android devices during model execution.
    """
    
    def __init__(self, device: AndroidDevice):
        """
        Initialize the Android thermal monitor.
        
        Args:
            device: Android device to monitor
        """
        self.device = device
        
        # Initialize thermal zones
        self.thermal_zones = self._create_thermal_zones()
        
        # Initialize throttling detection
        self.throttling_detected = False
        self.throttling_start_time = None
        self.throttling_duration = 0.0
        
        # Initialize performance impact tracking
        self.performance_impact = 0.0
        
        # Initialize monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 1.0  # seconds
        
        # Initialize thermal events
        self.thermal_events = []
    
    def _create_thermal_zones(self) -> Dict[str, AndroidThermalZone]:
        """
        Create Android-specific thermal zones.
        
        Returns:
            Dictionary of thermal zones
        """
        zones = {}
        
        # Get device thermal info to automatically map thermal zones
        thermal_info = self.device.get_thermal_info()
        
        # Create common zones with sensible default thresholds
        zones["cpu"] = AndroidThermalZone(
            device=self.device,
            name="cpu",
            critical_temp=85.0,
            warning_temp=70.0,
            zone_type="cpu"
        )
        
        zones["gpu"] = AndroidThermalZone(
            device=self.device,
            name="gpu",
            critical_temp=80.0,
            warning_temp=65.0,
            zone_type="gpu"
        )
        
        zones["battery"] = AndroidThermalZone(
            device=self.device,
            name="battery",
            critical_temp=45.0,
            warning_temp=40.0,
            zone_type="battery"
        )
        
        # Add additional zones based on detected thermal sensors
        for zone_type in thermal_info.keys():
            # Skip already created zones
            if any(z.sensor_type.lower() == zone_type.lower() for z in zones.values()):
                continue
            
            # Skip unknown or non-descriptive zones
            if zone_type.lower() in ["unknown", "none", ""]:
                continue
            
            # Determine appropriate thresholds based on zone type
            if "soc" in zone_type.lower():
                critical_temp = 85.0
                warning_temp = 70.0
            elif "skin" in zone_type.lower():
                critical_temp = 45.0
                warning_temp = 40.0
            else:
                critical_temp = 75.0
                warning_temp = 60.0
            
            # Create zone
            zones[zone_type] = AndroidThermalZone(
                device=self.device,
                name=zone_type,
                critical_temp=critical_temp,
                warning_temp=warning_temp,
                zone_type=zone_type
            )
        
        return zones
    
    def start_monitoring(self) -> bool:
        """
        Start thermal monitoring.
        
        Returns:
            Success status
        """
        if self.monitoring_active:
            logger.warning("Thermal monitoring is already active")
            return True
        
        if not self.device or not self.device.connected:
            logger.error("Cannot start monitoring: device not connected")
            return False
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Android thermal monitoring started")
        return True
    
    def stop_monitoring(self) -> None:
        """Stop thermal monitoring."""
        if not self.monitoring_active:
            logger.warning("Thermal monitoring is not active")
            return
        
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
            if self.monitoring_thread.is_alive():
                logger.warning("Could not gracefully stop monitoring thread")
            
            self.monitoring_thread = None
        
        logger.info("Android thermal monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background thread for continuous thermal monitoring."""
        logger.info("Thermal monitoring loop started")
        
        while self.monitoring_active:
            try:
                # Update all thermal zones
                self._update_thermal_zones()
                
                # Check for thermal events
                self._check_thermal_events()
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
            
            except Exception as e:
                logger.error(f"Error in thermal monitoring loop: {e}")
                # Continue monitoring despite errors
        
        logger.info("Thermal monitoring loop ended")
    
    def _update_thermal_zones(self) -> None:
        """Update temperatures in all thermal zones."""
        for zone in self.thermal_zones.values():
            zone.read_temperature()
    
    def _check_thermal_events(self) -> None:
        """Check for thermal events and update status."""
        # Find the most severe status
        most_severe_status = ThermalEventType.NORMAL
        most_severe_zone = None
        
        for name, zone in self.thermal_zones.items():
            if zone.status.value > most_severe_status.value:
                most_severe_status = zone.status
                most_severe_zone = zone
        
        # Check for throttling
        if most_severe_status in [ThermalEventType.THROTTLING, ThermalEventType.CRITICAL, ThermalEventType.EMERGENCY]:
            if not self.throttling_detected:
                # Throttling just started
                self.throttling_detected = True
                self.throttling_start_time = time.time()
                
                # Create throttling event
                self._create_thermal_event(most_severe_status, most_severe_zone)
        elif self.throttling_detected:
            # Throttling just ended
            self.throttling_detected = False
            if self.throttling_start_time is not None:
                self.throttling_duration += time.time() - self.throttling_start_time
                self.throttling_start_time = None
            
            # Create normal event
            self._create_thermal_event(ThermalEventType.NORMAL, most_severe_zone)
    
    def _create_thermal_event(self, event_type: ThermalEventType, zone: Optional[AndroidThermalZone]) -> None:
        """
        Create and record a thermal event.
        
        Args:
            event_type: Type of thermal event
            zone: Thermal zone where the event occurred
        """
        if zone is None:
            # Use the hottest zone if none provided
            zone = max(
                self.thermal_zones.values(),
                key=lambda z: z.current_temp,
                default=None
            )
        
        if zone is None:
            return
        
        # Create event
        event = ThermalEvent(
            event_type=event_type,
            zone_name=zone.name,
            temperature=zone.current_temp,
            timestamp=time.time()
        )
        
        # Log the event
        logger.info(f"Thermal event: {event_type.name} in {zone.name} zone at {zone.current_temp:.1f}°C")
        
        # Add to event history
        self.thermal_events.append(event)
        
        # Limit event history size
        if len(self.thermal_events) > 100:
            self.thermal_events.pop(0)
    
    def get_current_temperatures(self) -> Dict[str, float]:
        """
        Get current temperatures from all thermal zones.
        
        Returns:
            Dictionary mapping zone names to temperatures
        """
        # Update all zones
        self._update_thermal_zones()
        
        # Return current temperatures
        return {
            name: zone.current_temp
            for name, zone in self.thermal_zones.items()
        }
    
    def get_temperature_trends(self) -> Dict[str, Dict[str, Any]]:
        """
        Get temperature trends for all thermal zones.
        
        Returns:
            Dictionary mapping zone names to trend information
        """
        trends = {}
        
        for name, zone in self.thermal_zones.items():
            if hasattr(zone, "get_temperature_trend"):
                trends[name] = zone.get_temperature_trend()
            else:
                # Calculate trend manually if method not available
                window_seconds = 60
                now = time.time()
                window_start = now - window_seconds
                
                # Filter history to the specified window
                window_history = [
                    (t, temp) for t, temp in zone.temp_history 
                    if t >= window_start
                ]
                
                if len(window_history) < 2:
                    trends[name] = {
                        "trend_celsius_per_minute": 0.0,
                        "min_temp": zone.current_temp,
                        "max_temp": zone.current_temp,
                        "avg_temp": zone.current_temp,
                        "stable": True
                    }
                    continue
                
                # Extract times and temperatures
                times, temps = zip(*window_history)
                times = np.array(times)
                temps = np.array(temps)
                
                # Calculate trend (linear regression)
                times_minutes = (times - times[0]) / 60.0
                slope, intercept = np.polyfit(times_minutes, temps, 1)
                
                # Calculate statistics
                min_temp = np.min(temps)
                max_temp = np.max(temps)
                avg_temp = np.mean(temps)
                
                # Determine if temperature is stable
                temp_range = max_temp - min_temp
                stable = temp_range < 3.0 and abs(slope) < 0.5
                
                trends[name] = {
                    "trend_celsius_per_minute": slope,
                    "min_temp": min_temp,
                    "max_temp": max_temp,
                    "avg_temp": avg_temp,
                    "stable": stable
                }
        
        return trends
    
    def get_throttling_stats(self) -> Dict[str, Any]:
        """
        Get statistics about throttling.
        
        Returns:
            Dictionary with throttling statistics
        """
        # Calculate total throttling time
        total_throttling_time = self.throttling_duration
        if self.throttling_detected and self.throttling_start_time is not None:
            total_throttling_time += time.time() - self.throttling_start_time
        
        # Get throttling level and description
        if not self.throttling_detected:
            throttling_level = 0
            level_description = "No throttling"
        else:
            # Determine throttling level based on temperature
            hottest_zone = max(
                self.thermal_zones.values(),
                key=lambda z: (z.current_temp - z.warning_temp) / (z.critical_temp - z.warning_temp)
                if z.critical_temp > z.warning_temp else 0.0,
                default=None
            )
            
            if hottest_zone is None:
                throttling_level = 0
                level_description = "No throttling"
            else:
                # Calculate throttling level (0-5)
                temp_ratio = (hottest_zone.current_temp - hottest_zone.warning_temp) / (
                    hottest_zone.critical_temp - hottest_zone.warning_temp
                ) if hottest_zone.critical_temp > hottest_zone.warning_temp else 0.0
                
                temp_ratio = max(0.0, min(1.0, temp_ratio))
                throttling_level = int(temp_ratio * 5)
                
                level_descriptions = [
                    "No throttling",
                    "Mild throttling",
                    "Moderate throttling",
                    "Heavy throttling",
                    "Severe throttling",
                    "Emergency throttling"
                ]
                
                level_description = level_descriptions[throttling_level]
        
        # Calculate performance impact
        performance_impact = throttling_level * 0.2  # 0-1.0 scale
        
        return {
            "throttling_detected": self.throttling_detected,
            "throttling_level": throttling_level,
            "level_description": level_description,
            "throttling_time_seconds": total_throttling_time,
            "performance_impact": performance_impact
        }
    
    def get_thermal_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive thermal report.
        
        Returns:
            Dictionary with thermal report
        """
        # Update all thermal zones
        current_temps = self.get_current_temperatures()
        
        # Get temperature trends
        trends = self.get_temperature_trends()
        
        # Get throttling statistics
        throttling_stats = self.get_throttling_stats()
        
        # Calculate overall thermal status
        overall_status = max(
            zone.status for zone in self.thermal_zones.values()
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Create report
        report = {
            "timestamp": time.time(),
            "datetime": datetime.datetime.now().isoformat(),
            "device_model": self.device.device_info.get("model", "Unknown"),
            "android_version": self.device.device_info.get("android_version", "Unknown"),
            "overall_status": overall_status.name,
            "throttling": throttling_stats,
            "current_temperatures": current_temps,
            "temperature_trends": trends,
            "max_temperatures": {
                name: zone.max_temp
                for name, zone in self.thermal_zones.items()
            },
            "thermal_zones": {
                name: {
                    "current_temp": zone.current_temp,
                    "warning_temp": zone.warning_temp,
                    "critical_temp": zone.critical_temp,
                    "status": zone.status.name,
                    "type": zone.sensor_type
                }
                for name, zone in self.thermal_zones.items()
            },
            "recent_events": [
                {
                    "event_type": event.event_type.name,
                    "zone_name": event.zone_name,
                    "temperature": event.temperature,
                    "timestamp": event.timestamp,
                    "datetime": datetime.datetime.fromtimestamp(event.timestamp).isoformat(),
                    "impact_score": event.impact_score
                }
                for event in self.thermal_events[-10:]  # Last 10 events
            ],
            "recommendations": recommendations
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate thermal management recommendations.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Get thermal statuses
        statuses = {
            name: zone.status
            for name, zone in self.thermal_zones.items()
        }
        
        # Get temperature trends
        trends = self.get_temperature_trends()
        
        # Check for critical temperatures
        critical_zones = [
            name for name, zone in self.thermal_zones.items()
            if zone.status == ThermalEventType.CRITICAL or zone.status == ThermalEventType.EMERGENCY
        ]
        
        if critical_zones:
            recommendations.append(
                f"CRITICAL: {', '.join(critical_zones)} temperature(s) exceeding critical threshold. "
                "Immediate action required."
            )
        
        # Check for warning temperatures
        warning_zones = [
            name for name, zone in self.thermal_zones.items()
            if zone.status == ThermalEventType.WARNING and name not in critical_zones
        ]
        
        if warning_zones:
            recommendations.append(
                f"WARNING: {', '.join(warning_zones)} temperature(s) exceeding warning threshold. "
                "Consider thermal management."
            )
        
        # Check for increasing trends
        increasing_zones = [
            name for name, trend in trends.items()
            if trend.get("trend_celsius_per_minute", 0) > 0.5  # More than 0.5°C per minute
        ]
        
        if increasing_zones:
            recommendations.append(
                f"TREND: {', '.join(increasing_zones)} temperature(s) increasing rapidly. "
                "Monitor closely."
            )
        
        # Check throttling status
        throttling_stats = self.get_throttling_stats()
        if throttling_stats["throttling_detected"]:
            recommendations.append(
                f"THROTTLING: Performance reduced by {throttling_stats['performance_impact'] * 100:.1f}%. "
                "Consider reducing workload."
            )
            
            if throttling_stats["throttling_time_seconds"] > 300:  # More than 5 minutes
                recommendations.append(
                    f"EXTENDED THROTTLING: Device has been throttling for "
                    f"{throttling_stats['throttling_time_seconds'] / 60:.1f} minutes. "
                    "Device may be unsuitable for current workload."
                )
        
        # Add device-specific recommendations
        chipset = self.device.device_info.get("chipset", "").lower()
        
        if "qualcomm" in chipset or "snapdragon" in chipset:
            if any(zone.status.value >= ThermalEventType.WARNING.value for zone in self.thermal_zones.values()):
                recommendations.append(
                    "QUALCOMM: Consider using QNN-optimized models for reduced thermal impact."
                )
        elif "exynos" in chipset:
            if any(zone.status.value >= ThermalEventType.WARNING.value for zone in self.thermal_zones.values()):
                recommendations.append(
                    "SAMSUNG: Consider using One UI optimizations for reduced thermal impact."
                )
        elif "mediatek" in chipset:
            if any(zone.status.value >= ThermalEventType.WARNING.value for zone in self.thermal_zones.values()):
                recommendations.append(
                    "MEDIATEK: Consider using APU accelerations for reduced thermal impact."
                )
        
        # Add general recommendations if none specific
        if not recommendations:
            recommendations.append(
                "STATUS OK: All thermal zones within normal operating temperatures."
            )
        
        return recommendations
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the thermal monitor to a dictionary.
        
        Returns:
            Dictionary representation of the thermal monitor
        """
        return {
            "device_model": self.device.device_info.get("model", "Unknown"),
            "thermal_zones": {
                name: {
                    "current_temp": zone.current_temp,
                    "max_temp": zone.max_temp,
                    "warning_temp": zone.warning_temp,
                    "critical_temp": zone.critical_temp,
                    "status": zone.status.name
                }
                for name, zone in self.thermal_zones.items()
            },
            "throttling": self.get_throttling_stats(),
            "event_count": len(self.thermal_events)
        }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Android Thermal Monitoring")
    parser.add_argument("--serial", help="Device serial number")
    parser.add_argument("--duration", type=int, default=0, help="Monitoring duration in seconds (0 for indefinite)")
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval in seconds")
    parser.add_argument("--report", help="Path to save thermal report")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Connect to device
        device = AndroidDevice(args.serial)
        
        if not device.connected:
            print("Failed to connect to Android device")
            return 1
        
        print(f"Connected to Android device: {device.device_info.get('model', device.serial)}")
        
        # Create thermal monitor
        monitor = AndroidThermalMonitor(device)
        monitor.monitoring_interval = args.interval
        
        # Start monitoring
        print(f"Starting thermal monitoring with {args.interval:.1f}s interval")
        monitor.start_monitoring()
        
        try:
            # Monitor for specified duration or until interrupted
            if args.duration > 0:
                print(f"Monitoring for {args.duration} seconds")
                time.sleep(args.duration)
            else:
                print("Monitoring indefinitely (press Ctrl+C to stop)")
                while True:
                    time.sleep(1)
                    
                    # Periodically print temperature updates
                    if int(time.time()) % 10 == 0:  # Every 10 seconds
                        temps = monitor.get_current_temperatures()
                        hottest_zone = max(temps.items(), key=lambda x: x[1], default=(None, 0))
                        if hottest_zone[0]:
                            print(f"Hottest zone: {hottest_zone[0]} at {hottest_zone[1]:.1f}°C")
                            
                            # Print throttling status
                            throttling = monitor.get_throttling_stats()
                            if throttling["throttling_detected"]:
                                print(f"Throttling: {throttling['level_description']} "
                                      f"(Impact: {throttling['performance_impact']*100:.1f}%)")
        
        except KeyboardInterrupt:
            print("\nMonitoring interrupted")
        
        finally:
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Generate report
            report = monitor.get_thermal_report()
            
            if args.report:
                # Save report to file
                with open(args.report, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"Thermal report saved to: {args.report}")
            else:
                # Print report summary
                print("\nThermal Monitoring Report Summary:")
                print(f"Device: {report['device_model']}")
                print(f"Overall status: {report['overall_status']}")
                print("\nTemperatures:")
                for zone, temp in report['current_temperatures'].items():
                    print(f"  {zone}: {temp:.1f}°C")
                
                if report['throttling']['throttling_detected']:
                    print("\nThrottling:")
                    print(f"  Level: {report['throttling']['throttling_level']} ({report['throttling']['level_description']})")
                    print(f"  Duration: {report['throttling']['throttling_time_seconds']:.1f}s")
                    print(f"  Performance impact: {report['throttling']['performance_impact']*100:.1f}%")
                
                print("\nRecommendations:")
                for rec in report['recommendations']:
                    print(f"  - {rec}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())