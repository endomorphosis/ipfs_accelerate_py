#!/usr/bin/env python3
"""
Test script for Dashboard-Enhanced Visualization System.

This script tests the DashboardEnhancedVisualizationSystem with Monitoring Dashboard integration.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_dashboard_enhanced_visualization")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

# Import dependencies
try:
    from duckdb_api.visualization.dashboard_enhanced_visualization import DashboardEnhancedVisualizationSystem
    HAS_DASHBOARD_VISUALIZATION = True
except ImportError as e:
    logger.error(f"Error importing DashboardEnhancedVisualizationSystem: {e}")
    HAS_DASHBOARD_VISUALIZATION = False

try:
    from duckdb_api.core.benchmark_db_api import BenchmarkDBAPI
    HAS_DB_API = True
except ImportError as e:
    logger.error(f"Error importing BenchmarkDBAPI: {e}")
    HAS_DB_API = False


def create_test_visualizations(vis_system, num_visualizations=3):
    """Create test visualizations using the provided visualization system."""
    # Create visualizations
    viz_paths = []
    
    # Create 3D visualization
    if num_visualizations > 0:
        viz_path = vis_system.create_3d_performance_visualization(
            metrics=["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
            dimensions=["model_family", "hardware_type"],
            title="Test 3D Performance Visualization"
        )
        if viz_path:
            viz_paths.append(viz_path)
    
    # Create heatmap visualization
    if num_visualizations > 1:
        viz_path = vis_system.create_hardware_comparison_heatmap(
            metric="throughput",
            batch_size=1,
            title="Test Hardware Comparison Heatmap"
        )
        if viz_path:
            viz_paths.append(viz_path)
    
    # Create time-series visualization
    if num_visualizations > 2:
        viz_path = vis_system.create_animated_time_series_visualization(
            metric="throughput_items_per_second",
            dimensions=["model_family", "hardware_type"],
            include_trend=True,
            window_size=3,
            title="Test Time Series Visualization"
        )
        if viz_path:
            viz_paths.append(viz_path)
    
    return viz_paths


def test_dashboard_enhanced_visualization(db_path, output_dir, dashboard_url=None, api_key=None,
                                         create_visualizations=True, synchronize=True, create_panel=True):
    """Test the Dashboard-Enhanced Visualization System."""
    if not HAS_DASHBOARD_VISUALIZATION:
        logger.error("Dashboard-enhanced visualization not available.")
        return False
    
    if not HAS_DB_API:
        logger.error("DuckDB API not available.")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DuckDB API
    db_api = BenchmarkDBAPI(db_path=db_path)
    
    # Initialize visualization system with dashboard integration
    vis_system = DashboardEnhancedVisualizationSystem(
        db_api=db_api,
        output_dir=output_dir,
        dashboard_url=dashboard_url,
        api_key=api_key
    )
    
    logger.info("Dashboard-enhanced visualization system initialized.")
    
    # Create test visualizations
    if create_visualizations:
        viz_paths = create_test_visualizations(vis_system, num_visualizations=3)
        if not viz_paths:
            logger.warning("No visualizations created.")
        else:
            logger.info(f"Created {len(viz_paths)} visualizations.")
    
    # Synchronize with dashboard
    if synchronize:
        num_synced = vis_system.synchronize_with_dashboard(
            dashboard_url=dashboard_url,
            api_key=api_key,
            create_panel=create_panel,
            panel_title="Test Visualization Panel"
        )
        
        if num_synced > 0:
            logger.info(f"Synchronized {num_synced} visualizations with dashboard.")
            return True
        else:
            logger.warning("Failed to synchronize visualizations with dashboard.")
            return False
    
    return True


def test_combined_creation_and_sync(db_path, output_dir, dashboard_url=None, api_key=None):
    """Test creating a visualization and synchronizing it in one step."""
    if not HAS_DASHBOARD_VISUALIZATION or not HAS_DB_API:
        logger.error("Required components not available.")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DuckDB API
    db_api = BenchmarkDBAPI(db_path=db_path)
    
    # Initialize visualization system with dashboard integration
    vis_system = DashboardEnhancedVisualizationSystem(
        db_api=db_api,
        output_dir=output_dir,
        dashboard_url=dashboard_url,
        api_key=api_key
    )
    
    # Create visualization and sync in one step
    viz_path = vis_system.create_visualization_and_sync(
        visualization_type="3d",
        visualization_params={
            "metrics": ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
            "dimensions": ["model_family", "hardware_type"],
            "title": "3D Performance Visualization (Combined Creation and Sync)"
        },
        sync_with_dashboard=True,
        dashboard_url=dashboard_url,
        api_key=api_key
    )
    
    if viz_path:
        logger.info(f"Successfully created and synchronized visualization: {viz_path}")
        return True
    else:
        logger.warning("Failed to create and synchronize visualization.")
        return False


def test_snapshot_export_import(db_path, output_dir, dashboard_url=None, api_key=None):
    """Test exporting and importing a dashboard snapshot."""
    if not HAS_DASHBOARD_VISUALIZATION or not HAS_DB_API:
        logger.error("Required components not available.")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DuckDB API
    db_api = BenchmarkDBAPI(db_path=db_path)
    
    # Initialize visualization system with dashboard integration
    vis_system = DashboardEnhancedVisualizationSystem(
        db_api=db_api,
        output_dir=output_dir,
        dashboard_url=dashboard_url,
        api_key=api_key
    )
    
    # Export snapshot
    snapshot_path = vis_system.export_dashboard_snapshot(
        output_path=os.path.join(output_dir, "dashboard_snapshot_test.json"),
        include_visualizations=True
    )
    
    if not snapshot_path:
        logger.warning("Failed to export dashboard snapshot.")
        return False
    
    logger.info(f"Exported dashboard snapshot to {snapshot_path}")
    
    # Import snapshot to a different dashboard ID
    import_success = vis_system.import_dashboard_snapshot(
        snapshot_path=snapshot_path,
        target_dashboard_id="test_import",
        merge_strategy="replace"
    )
    
    if import_success:
        logger.info("Successfully imported dashboard snapshot.")
        return True
    else:
        logger.warning("Failed to import dashboard snapshot.")
        return False


def test_regression_detection_integration(db_path, output_dir, dashboard_url=None, api_key=None):
    """Test the integration of regression detection with the dashboard."""
    if not HAS_DASHBOARD_VISUALIZATION or not HAS_DB_API:
        logger.error("Required components not available.")
        return False
    
    # Attempt to import RegressionDetector
    try:
        from duckdb_api.distributed_testing.dashboard.regression_detection import RegressionDetector
        HAS_REGRESSION_DETECTION = True
    except ImportError as e:
        logger.error(f"Error importing RegressionDetector: {e}")
        logger.error("Regression detection test will be skipped.")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize DuckDB API
    db_api = BenchmarkDBAPI(db_path=db_path)
    
    # Initialize visualization system with dashboard integration and regression detection
    try:
        vis_system = DashboardEnhancedVisualizationSystem(
            db_api=db_api,
            output_dir=output_dir,
            dashboard_url=dashboard_url,
            api_key=api_key,
            enable_regression_detection=True
        )
        logger.info("Dashboard-enhanced visualization system with regression detection initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize visualization system with regression detection: {e}")
        return False
    
    # Create test visualization with regression detection
    try:
        # Create performance visualization with regression detection
        viz_path = vis_system.create_visualization_with_regression_detection(
            visualization_type="time_series",
            metric="latency_ms",
            filters={"model_name": "bert-base", "hardware_type": "cuda"},
            title="Latency Regression Analysis",
            include_statistical_significance=True,
            confidence_level=0.95
        )
        
        if viz_path:
            logger.info(f"Successfully created regression detection visualization: {viz_path}")
            
            # Synchronize with dashboard
            if dashboard_url:
                sync_success = vis_system.synchronize_with_dashboard(
                    visualization_paths=[viz_path],
                    dashboard_url=dashboard_url,
                    api_key=api_key,
                    create_panel=True,
                    panel_title="Regression Analysis Panel"
                )
                
                if sync_success > 0:
                    logger.info(f"Synchronized regression visualization with dashboard.")
                    return True
                else:
                    logger.warning("Failed to synchronize regression visualization with dashboard.")
                    return False
            return True
        else:
            logger.warning("Failed to create regression detection visualization.")
            return False
            
    except Exception as e:
        logger.error(f"Error in regression detection test: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Dashboard-Enhanced Visualization System")
    parser.add_argument("--db-path", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    parser.add_argument("--output-dir", default="./dashboard_visualization_tests",
                       help="Directory to save visualizations")
    parser.add_argument("--dashboard-url", default="http://localhost:8082",
                       help="URL of the monitoring dashboard")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--no-create", action="store_true",
                       help="Skip creating test visualizations")
    parser.add_argument("--no-sync", action="store_true",
                       help="Skip synchronizing with dashboard")
    parser.add_argument("--no-panel", action="store_true",
                       help="Skip creating dashboard panel")
    parser.add_argument("--test", choices=["basic", "combined", "snapshot", "regression", "all"],
                       default="all", help="Test to run")
    
    args = parser.parse_args()
    
    # Check required components
    if not HAS_DASHBOARD_VISUALIZATION:
        logger.error("Dashboard-enhanced visualization not available.")
        logger.error("Please check if you have the required modules installed.")
        return 1
    
    if not HAS_DB_API:
        logger.error("DuckDB API not available.")
        logger.error("Please check if you have the required modules installed.")
        return 1
    
    # Run basic test
    if args.test == "basic" or args.test == "all":
        logger.info("Running basic test...")
        success = test_dashboard_enhanced_visualization(
            db_path=args.db_path,
            output_dir=args.output_dir,
            dashboard_url=args.dashboard_url,
            api_key=args.api_key,
            create_visualizations=not args.no_create,
            synchronize=not args.no_sync,
            create_panel=not args.no_panel
        )
        
        if success:
            logger.info("✅ Basic test passed.")
        else:
            logger.warning("❌ Basic test failed.")
    
    # Run combined creation and sync test
    if args.test == "combined" or args.test == "all":
        logger.info("Running combined creation and sync test...")
        success = test_combined_creation_and_sync(
            db_path=args.db_path,
            output_dir=args.output_dir,
            dashboard_url=args.dashboard_url,
            api_key=args.api_key
        )
        
        if success:
            logger.info("✅ Combined creation and sync test passed.")
        else:
            logger.warning("❌ Combined creation and sync test failed.")
    
    # Run snapshot export/import test
    if args.test == "snapshot" or args.test == "all":
        logger.info("Running snapshot export/import test...")
        success = test_snapshot_export_import(
            db_path=args.db_path,
            output_dir=args.output_dir,
            dashboard_url=args.dashboard_url,
            api_key=args.api_key
        )
        
        if success:
            logger.info("✅ Snapshot export/import test passed.")
        else:
            logger.warning("❌ Snapshot export/import test failed.")
    
    # Run regression detection integration test
    if args.test == "regression" or args.test == "all":
        logger.info("Running regression detection integration test...")
        success = test_regression_detection_integration(
            db_path=args.db_path,
            output_dir=args.output_dir,
            dashboard_url=args.dashboard_url,
            api_key=args.api_key
        )
        
        if success:
            logger.info("✅ Regression detection integration test passed.")
        else:
            logger.warning("❌ Regression detection integration test failed.")
    
    logger.info("Testing complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())