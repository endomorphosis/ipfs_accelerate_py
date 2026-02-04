#!/usr/bin/env python3
"""
DRM External Monitoring Integration Runner

This script launches the integration between DRM and external monitoring systems
like Prometheus and Grafana, enabling metrics export and integration with 
existing monitoring infrastructure.

Features:
- Prometheus metrics export for all DRM metrics
- Grafana dashboard generation for DRM monitoring
- Automatic integration with the DRM dashboard
- Support for simulated and real DRM instances
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("drm_monitoring_runner")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def check_dependencies():
    """Check if required dependencies are installed."""
    missing_deps = []
    
    # Check Prometheus client
    try:
        import prometheus_client
    except ImportError:
        missing_deps.append("prometheus_client")
    
    # Check requests
    try:
        import requests
    except ImportError:
        missing_deps.append("requests")
    
    # Check DRM dependencies
    try:
        import dash
    except ImportError:
        missing_deps.append("dash")
    
    try:
        import dash_bootstrap_components
    except ImportError:
        missing_deps.append("dash-bootstrap-components")
    
    try:
        import plotly
    except ImportError:
        missing_deps.append("plotly")
    
    return missing_deps

def main():
    """Run the DRM external monitoring integration."""
    parser = argparse.ArgumentParser(description="DRM External Monitoring Integration")
    
    # Monitoring configuration
    parser.add_argument("--metrics-port", type=int, default=9100, help="Port to expose Prometheus metrics (default: 9100)")
    parser.add_argument("--prometheus-url", default="http://localhost:9090", help="Prometheus server URL")
    parser.add_argument("--grafana-url", default="http://localhost:3000", help="Grafana server URL")
    parser.add_argument("--grafana-api-key", help="Grafana API key for dashboard upload")
    
    # Dashboard configuration
    parser.add_argument("--dashboard-port", type=int, default=8085, help="DRM dashboard port (default: 8085)")
    parser.add_argument("--update-interval", type=int, default=5, help="Update interval in seconds (default: 5)")
    parser.add_argument("--theme", choices=["light", "dark"], default="dark", help="Dashboard theme")
    
    # Output options
    parser.add_argument("--output-dir", help="Directory to save dashboard files")
    parser.add_argument("--export-only", action="store_true", help="Only export Grafana dashboard without starting services")
    parser.add_argument("--save-guide", action="store_true", help="Save integration guide to file")
    
    # Integration options
    parser.add_argument("--drm-url", help="URL of DRM coordinator (optional, for connecting to live DRM)")
    parser.add_argument("--api-key", help="API key for DRM coordinator authentication")
    parser.add_argument("--simulation", action="store_true", help="Use simulated DRM data (default if no DRM URL provided)")
    parser.add_argument("--no-browser", action="store_true", help="Don't automatically open browser")
    
    args = parser.parse_args()
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        critical_deps = [dep for dep in missing_deps if dep in ["prometheus_client", "dash"]]
        if critical_deps:
            logger.error("Missing critical dependencies:")
            for dep in critical_deps:
                logger.error(f"  - {dep}")
            logger.error("Please install required dependencies:")
            logger.error("pip install -r requirements_dashboard.txt")
            sys.exit(1)
        else:
            logger.warning("Missing some optional dependencies:")
            for dep in missing_deps:
                logger.warning(f"  - {dep}")
            logger.warning("Some features may be limited. Install all dependencies with:")
            logger.warning("pip install -r requirements_dashboard.txt")
    
    # Determine output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitoring_output")
        logger.info(f"No output directory specified, using: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import modules
        from data.duckdb.distributed_testing.dashboard.drm_real_time_dashboard import DRMRealTimeDashboard
        from data.duckdb.distributed_testing.dashboard.drm_external_monitoring_integration import ExternalMonitoringBridge
        
        # Determine DRM instance
        drm_instance = None
        
        if args.drm_url and args.api_key:
            # Try to connect to real DRM
            try:
                from data.duckdb.distributed_testing.drm_api_client import DRMAPIClient
                logger.info(f"Connecting to DRM coordinator at {args.drm_url}")
                drm_instance = DRMAPIClient(args.drm_url, args.api_key)
                logger.info("Connected to DRM coordinator")
            except ImportError:
                logger.warning("Could not import DRMAPIClient. Running in simulation mode.")
        
        if not drm_instance or args.simulation:
            # Use mock DRM for testing
            from data.duckdb.distributed_testing.testing.mock_drm import MockDynamicResourceManager
            logger.info("Using simulated DRM data")
            drm_instance = MockDynamicResourceManager()
        
        # Set up DRM dashboard
        logger.info("Setting up DRM dashboard")
        dashboard = DRMRealTimeDashboard(
            dynamic_resource_manager=drm_instance,
            port=args.dashboard_port,
            update_interval=args.update_interval,
            theme=args.theme,
            debug=False
        )
        
        # Set up external monitoring bridge
        logger.info("Setting up external monitoring bridge")
        bridge = ExternalMonitoringBridge(
            drm_dashboard=dashboard,
            metrics_port=args.metrics_port,
            prometheus_url=args.prometheus_url,
            grafana_url=args.grafana_url,
            grafana_api_key=args.grafana_api_key,
            export_grafana_dashboard=True,
            output_dir=output_dir
        )
        
        # Save integration guide if requested
        if args.save_guide:
            guide = bridge.get_metrics_integration_guide()
            guide_path = os.path.join(output_dir, "monitoring_integration_guide.md")
            with open(guide_path, 'w') as f:
                f.write(guide)
            logger.info(f"Integration guide saved to: {guide_path}")
        
        # Export only mode
        if args.export_only:
            logger.info("Export-only mode: Exporting Grafana dashboard")
            dashboard._start_data_collection()
            # Wait for some data to be collected
            import time
            time.sleep(5)
            
            # Export dashboard
            bridge._export_grafana_dashboard()
            
            # Stop data collection
            dashboard._stop_data_collection()
            
            # Print summary
            grafana_dashboard_path = bridge.get_grafana_dashboard_path()
            logger.info(f"Grafana dashboard exported to: {grafana_dashboard_path}")
            
            return
        
        # Start bridge and dashboard
        logger.info("Starting external monitoring bridge")
        bridge.start(update_interval=args.update_interval)
        
        logger.info("Starting DRM dashboard")
        if not args.no_browser:
            dashboard.start()  # This will block until the dashboard is closed
        else:
            dashboard.start_in_background()
            
            # Keep running until interrupted
            import time
            try:
                logger.info("Press Ctrl+C to stop...")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Stopping...")
        
        # Stop bridge when dashboard is closed
        bridge.stop()
        logger.info("External monitoring bridge stopped")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure the DRM dashboard and its dependencies are installed:")
        logger.error("pip install -r requirements_dashboard.txt")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()