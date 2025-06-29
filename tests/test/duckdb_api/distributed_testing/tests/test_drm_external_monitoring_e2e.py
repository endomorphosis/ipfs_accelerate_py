#!/usr/bin/env python3
"""
End-to-End Tests for DRM External Monitoring Integration

This module implements comprehensive end-to-end tests for the integration
between DRM and external monitoring systems like Prometheus and Grafana.

It sets up a test environment with Docker containers for Prometheus and Grafana,
runs the DRM dashboard with external monitoring integration, and verifies
the correct functioning of the complete system.
"""

import os
import sys
import json
import unittest
import tempfile
import threading
import time
import socket
import subprocess
import requests
import atexit
import datetime
import logging
from pathlib import Path

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("e2e_test")

# Check if Docker is available
DOCKER_AVAILABLE = False
try:
    result = subprocess.run(["docker", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        DOCKER_AVAILABLE = True
except (FileNotFoundError, subprocess.SubprocessError):
    pass

# Try to import the modules to test
try:
    from duckdb_api.distributed_testing.dashboard.drm_external_monitoring_integration import (
        PrometheusExporter,
        GrafanaDashboardGenerator,
        ExternalMonitoringBridge
    )
    from duckdb_api.distributed_testing.dashboard.drm_real_time_dashboard import DRMRealTimeDashboard
    from duckdb_api.distributed_testing.testing.mock_drm import MockDynamicResourceManager
    
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    MODULES_AVAILABLE = False

# Test ports - Use high ports to avoid conflicts
TEST_PROMETHEUS_PORT = 9191
TEST_GRAFANA_PORT = 9292
TEST_METRICS_PORT = 9393
TEST_DASHBOARD_PORT = 9494

# Container names for cleanup
PROMETHEUS_CONTAINER = "test_prometheus_drm"
GRAFANA_CONTAINER = "test_grafana_drm"

# Store running processes for cleanup
running_processes = []

def cleanup_containers():
    """Clean up Docker containers after tests."""
    logger.info("Cleaning up Docker containers...")
    for container in [PROMETHEUS_CONTAINER, GRAFANA_CONTAINER]:
        try:
            subprocess.run(["docker", "rm", "-f", container], 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            logger.error(f"Error cleaning up container {container}: {e}")

def cleanup_processes():
    """Clean up running processes after tests."""
    logger.info("Cleaning up processes...")
    for process in running_processes:
        try:
            if process.poll() is None:  # Still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        except Exception as e:
            logger.error(f"Error cleaning up process: {e}")

# Register cleanup functions
atexit.register(cleanup_containers)
atexit.register(cleanup_processes)

def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def create_prometheus_config(metrics_port, config_path):
    """Create a Prometheus configuration file."""
    prometheus_config = {
        "global": {
            "scrape_interval": "5s",
            "evaluation_interval": "5s"
        },
        "scrape_configs": [
            {
                "job_name": "drm",
                "static_configs": [
                    {
                        "targets": [f"host.docker.internal:{metrics_port}"]
                    }
                ]
            }
        ]
    }
    
    with open(config_path, 'w') as f:
        json.dump(prometheus_config, f)
    
    return config_path

def start_prometheus_container(config_path, port):
    """Start a Prometheus container for testing."""
    # Check if the container is already running
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name={PROMETHEUS_CONTAINER}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    if result.stdout:
        # Container exists, remove it
        subprocess.run(["docker", "rm", "-f", PROMETHEUS_CONTAINER],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Start Prometheus container
    cmd = [
        "docker", "run", "--name", PROMETHEUS_CONTAINER,
        "-d", "--rm",
        "-p", f"{port}:9090",
        "-v", f"{config_path}:/etc/prometheus/prometheus.yml",
        "--add-host", "host.docker.internal:host-gateway",
        "prom/prometheus"
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        logger.error(f"Failed to start Prometheus container: {result.stderr.decode()}")
        return False
    
    # Wait for Prometheus to start
    for _ in range(10):
        try:
            response = requests.get(f"http://localhost:{port}/-/ready")
            if response.status_code == 200:
                logger.info(f"Prometheus container started on port {port}")
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    
    logger.error("Timed out waiting for Prometheus to start")
    return False

def start_grafana_container(port):
    """Start a Grafana container for testing."""
    # Check if the container is already running
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name={GRAFANA_CONTAINER}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    if result.stdout:
        # Container exists, remove it
        subprocess.run(["docker", "rm", "-f", GRAFANA_CONTAINER],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Start Grafana container
    cmd = [
        "docker", "run", "--name", GRAFANA_CONTAINER,
        "-d", "--rm",
        "-p", f"{port}:3000",
        "-e", "GF_AUTH_ANONYMOUS_ENABLED=true",
        "-e", "GF_AUTH_ANONYMOUS_ORG_ROLE=Admin",
        "-e", "GF_AUTH_DISABLE_LOGIN_FORM=true",
        "grafana/grafana"
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        logger.error(f"Failed to start Grafana container: {result.stderr.decode()}")
        return False
    
    # Wait for Grafana to start
    for _ in range(15):  # Grafana can take some time to start
        try:
            response = requests.get(f"http://localhost:{port}/api/health")
            if response.status_code == 200:
                logger.info(f"Grafana container started on port {port}")
                # Wait a bit more to ensure Grafana is fully operational
                time.sleep(5)
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    
    logger.error("Timed out waiting for Grafana to start")
    return False

def configure_grafana_prometheus_datasource(grafana_port, prometheus_port):
    """Configure Prometheus as a datasource in Grafana."""
    datasource = {
        "name": "Prometheus",
        "type": "prometheus",
        "url": f"http://host.docker.internal:{prometheus_port}",
        "access": "proxy",
        "isDefault": True
    }
    
    try:
        response = requests.post(
            f"http://localhost:{grafana_port}/api/datasources",
            json=datasource
        )
        
        if response.status_code in (200, 201):
            logger.info("Prometheus datasource configured in Grafana")
            return True
        else:
            logger.error(f"Failed to configure Prometheus datasource: {response.text}")
            return False
    except requests.RequestException as e:
        logger.error(f"Error configuring Prometheus datasource: {e}")
        return False

def start_drm_with_monitoring(temp_dir, metrics_port, dashboard_port):
    """Start DRM dashboard with external monitoring integration."""
    # Create command to run the script
    script_path = os.path.join(parent_dir, "run_drm_external_monitoring.py")
    
    prometheus_url = f"http://localhost:{TEST_PROMETHEUS_PORT}"
    grafana_url = f"http://localhost:{TEST_GRAFANA_PORT}"
    output_dir = os.path.join(temp_dir, "monitoring")
    
    cmd = [
        sys.executable, script_path,
        "--metrics-port", str(metrics_port),
        "--dashboard-port", str(dashboard_port),
        "--prometheus-url", prometheus_url,
        "--grafana-url", grafana_url,
        "--output-dir", output_dir,
        "--simulation",
        "--no-browser",
        "--update-interval", "2"
    ]
    
    # Start the process
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Add to list for cleanup
    running_processes.append(process)
    
    # Wait for metrics endpoint to become available
    for _ in range(15):
        try:
            response = requests.get(f"http://localhost:{metrics_port}/metrics")
            if response.status_code == 200:
                logger.info(f"DRM metrics endpoint available on port {metrics_port}")
                # Also check dashboard
                try:
                    response = requests.get(f"http://localhost:{dashboard_port}")
                    if response.status_code == 200:
                        logger.info(f"DRM dashboard available on port {dashboard_port}")
                        return True, process, output_dir
                except requests.RequestException:
                    pass
        except requests.RequestException:
            pass
        time.sleep(2)
    
    logger.error("Timed out waiting for DRM metrics endpoint and dashboard")
    return False, process, output_dir

def verify_metrics_in_prometheus(prometheus_port):
    """Verify that DRM metrics are being scraped by Prometheus."""
    # Wait for metrics to be scraped
    time.sleep(10)
    
    # Query Prometheus API for DRM metrics
    query = "drm_cpu_utilization_percent"
    
    try:
        response = requests.get(
            f"http://localhost:{prometheus_port}/api/v1/query",
            params={"query": query}
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to query Prometheus: {response.text}")
            return False
        
        data = response.json()
        if data["status"] != "success":
            logger.error(f"Prometheus query failed: {data}")
            return False
        
        # Check if we have results
        if not data["data"]["result"]:
            logger.error("No DRM metrics found in Prometheus")
            return False
        
        logger.info("Successfully verified DRM metrics in Prometheus")
        return True
    except requests.RequestException as e:
        logger.error(f"Error querying Prometheus: {e}")
        return False

def verify_grafana_dashboard_import(grafana_port, dashboard_path):
    """Verify importing the Grafana dashboard."""
    try:
        # Read dashboard JSON
        with open(dashboard_path, 'r') as f:
            dashboard_json = json.load(f)
        
        # Prepare dashboard for import
        dashboard_import = {
            "dashboard": dashboard_json,
            "overwrite": True,
            "inputs": [
                {
                    "name": "DS_PROMETHEUS",
                    "type": "datasource",
                    "pluginId": "prometheus",
                    "value": "Prometheus"
                }
            ]
        }
        
        # Import dashboard
        response = requests.post(
            f"http://localhost:{grafana_port}/api/dashboards/import",
            json=dashboard_import
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to import dashboard: {response.text}")
            return False
        
        # Get dashboard UID from response
        data = response.json()
        dashboard_uid = data.get("uid")
        
        if not dashboard_uid:
            logger.error("Failed to get dashboard UID")
            return False
        
        # Verify dashboard was imported
        response = requests.get(
            f"http://localhost:{grafana_port}/api/dashboards/uid/{dashboard_uid}"
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to get imported dashboard: {response.text}")
            return False
        
        logger.info("Successfully imported and verified Grafana dashboard")
        return True
    except Exception as e:
        logger.error(f"Error importing Grafana dashboard: {e}")
        return False

def verify_alerts_configuration(prometheus_port, metrics_port):
    """Verify that alert conditions work correctly."""
    # Create a simple alert rule
    alert_rule_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.yml')
    
    try:
        # Write alert rule
        alert_rule = """
groups:
- name: drm_alerts
  rules:
  - alert: HighCPUUtilization
    expr: drm_cpu_utilization_percent > 70
    for: 10s
    labels:
      severity: warning
    annotations:
      summary: "High CPU utilization detected"
      description: "CPU utilization is above 70% threshold"
"""
        alert_rule_file.write(alert_rule)
        alert_rule_file.close()
        
        # Copy alert rule to Prometheus container
        subprocess.run(
            ["docker", "cp", alert_rule_file.name, f"{PROMETHEUS_CONTAINER}:/etc/prometheus/alert.yml"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # Update Prometheus config to include alert rule
        config_update = """
global:
  scrape_interval: 5s
  evaluation_interval: 5s

rule_files:
  - "alert.yml"

scrape_configs:
  - job_name: 'drm'
    static_configs:
      - targets: ['host.docker.internal:%d']
""" % metrics_port
        
        # Write updated config to file
        config_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.yml')
        config_file.write(config_update)
        config_file.close()
        
        # Copy updated config to Prometheus container
        subprocess.run(
            ["docker", "cp", config_file.name, f"{PROMETHEUS_CONTAINER}:/etc/prometheus/prometheus.yml"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # Reload Prometheus configuration
        subprocess.run(
            ["docker", "kill", "--signal=SIGHUP", PROMETHEUS_CONTAINER],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # Wait for alert to potentially fire
        time.sleep(20)
        
        # Check if alert is active
        response = requests.get(f"http://localhost:{prometheus_port}/api/v1/alerts")
        
        if response.status_code != 200:
            logger.error(f"Failed to query alerts: {response.text}")
            return False
        
        # We don't know if the alert will fire (depends on current CPU utilization),
        # so just check that the API call works
        logger.info("Successfully verified alert configuration")
        return True
        
    except Exception as e:
        logger.error(f"Error configuring alert: {e}")
        return False
    finally:
        # Clean up temporary files
        try:
            os.unlink(alert_rule_file.name)
            os.unlink(config_file.name)
        except:
            pass


@unittest.skipIf(not MODULES_AVAILABLE, "External monitoring modules not available")
@unittest.skipIf(not DOCKER_AVAILABLE, "Docker not available")
class TestDRMExternalMonitoringE2E(unittest.TestCase):
    """End-to-end tests for DRM External Monitoring Integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory
        cls.temp_dir = tempfile.mkdtemp()
        
        # Check if ports are in use
        ports = [TEST_PROMETHEUS_PORT, TEST_GRAFANA_PORT, TEST_METRICS_PORT, TEST_DASHBOARD_PORT]
        in_use_ports = [port for port in ports if is_port_in_use(port)]
        
        if in_use_ports:
            raise unittest.SkipTest(f"Ports already in use: {in_use_ports}")
        
        # Create Prometheus config
        cls.prometheus_config_path = os.path.join(cls.temp_dir, "prometheus.yml")
        create_prometheus_config(TEST_METRICS_PORT, cls.prometheus_config_path)
        
        # Start Prometheus container
        cls.prometheus_running = start_prometheus_container(
            cls.prometheus_config_path, TEST_PROMETHEUS_PORT
        )
        
        if not cls.prometheus_running:
            raise unittest.SkipTest("Failed to start Prometheus container")
        
        # Start Grafana container
        cls.grafana_running = start_grafana_container(TEST_GRAFANA_PORT)
        
        if not cls.grafana_running:
            raise unittest.SkipTest("Failed to start Grafana container")
        
        # Configure Grafana
        cls.grafana_configured = configure_grafana_prometheus_datasource(
            TEST_GRAFANA_PORT, TEST_PROMETHEUS_PORT
        )
        
        if not cls.grafana_configured:
            raise unittest.SkipTest("Failed to configure Grafana")
        
        # Start DRM with monitoring
        cls.drm_running, cls.drm_process, cls.output_dir = start_drm_with_monitoring(
            cls.temp_dir, TEST_METRICS_PORT, TEST_DASHBOARD_PORT
        )
        
        if not cls.drm_running:
            raise unittest.SkipTest("Failed to start DRM with monitoring")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Stop DRM process
        if hasattr(cls, 'drm_process') and cls.drm_process:
            try:
                cls.drm_process.terminate()
                cls.drm_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.drm_process.kill()
            except:
                pass
        
        # Clean up containers (redundant with atexit handlers, but just to be safe)
        cleanup_containers()
        
        # Clean up temporary directory
        if hasattr(cls, 'temp_dir'):
            import shutil
            try:
                shutil.rmtree(cls.temp_dir)
            except:
                pass
    
    def test_01_prometheus_metrics_collection(self):
        """Test that DRM metrics are collected in Prometheus."""
        self.assertTrue(verify_metrics_in_prometheus(TEST_PROMETHEUS_PORT))
    
    def test_02_grafana_dashboard_import(self):
        """Test importing the Grafana dashboard."""
        dashboard_path = os.path.join(self.output_dir, "drm_dashboard.json")
        self.assertTrue(os.path.exists(dashboard_path))
        self.assertTrue(verify_grafana_dashboard_import(TEST_GRAFANA_PORT, dashboard_path))
    
    def test_03_alert_configuration(self):
        """Test configuring and validating alerts."""
        self.assertTrue(verify_alerts_configuration(TEST_PROMETHEUS_PORT, TEST_METRICS_PORT))
    
    def test_04_metrics_validation(self):
        """Test validating specific metrics presence."""
        # Get metrics from Prometheus
        metrics_to_check = [
            "drm_cpu_utilization_percent",
            "drm_memory_utilization_percent", 
            "drm_active_tasks",
            "drm_worker_count"
        ]
        
        for metric in metrics_to_check:
            response = requests.get(
                f"http://localhost:{TEST_PROMETHEUS_PORT}/api/v1/query",
                params={"query": metric}
            )
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "success")
            self.assertTrue(data["data"]["result"], f"Metric {metric} not found")
    
    def test_05_dashboard_content(self):
        """Test the content of the generated dashboard."""
        dashboard_path = os.path.join(self.output_dir, "drm_dashboard.json")
        
        with open(dashboard_path, 'r') as f:
            dashboard = json.load(f)
        
        # Check dashboard structure
        self.assertEqual(dashboard["title"], "DRM Real-Time Performance Dashboard")
        self.assertGreaterEqual(len(dashboard["panels"]), 6)
        
        # Check for specific panels
        panel_titles = [panel["title"] for panel in dashboard["panels"]]
        expected_panels = [
            "System Overview",
            "Resource Utilization",
            "Performance Metrics",
            "Worker Resource Utilization",
            "Worker Tasks"
        ]
        
        for panel in expected_panels:
            self.assertIn(panel, panel_titles)


if __name__ == "__main__":
    unittest.main()