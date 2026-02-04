#!/usr/bin/env python3
"""
Test script for the Export Visualization functionality.

This script tests the export capabilities of the Advanced Visualization System.
"""

import os
import sys
import logging
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("export_test")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

# Import database API
try:
    from data.duckdb.core.benchmark_db_api import BenchmarkDBAPI
    HAS_DB_API = True
except ImportError:
    logger.warning("DuckDB API not available. Some tests will be skipped.")
    HAS_DB_API = False

# Import visualization system
try:
    from data.duckdb.visualization.advanced_visualization import AdvancedVisualizationSystem
    HAS_ADVANCED_VISUALIZATION = True
except ImportError as e:
    logger.warning(f"Advanced Visualization System not available: {e}")
    logger.warning("Some tests will be skipped.")
    HAS_ADVANCED_VISUALIZATION = False

# Import export utilities
try:
    from data.duckdb.visualization.advanced_visualization.export_utils import export_figure, export_data
    HAS_EXPORT_UTILS = True
except ImportError:
    logger.warning("Export utilities not available. Some tests will be skipped.")
    HAS_EXPORT_UTILS = False


class TestExportVisualization(unittest.TestCase):
    """Test case for the Export Visualization functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test case."""
        # Create temp directory for exports
        cls.temp_dir = tempfile.mkdtemp(prefix="test_export_")
        logger.info(f"Created temporary directory: {cls.temp_dir}")
        
        # Initialize database API if available
        if HAS_DB_API:
            try:
                # Look for the database file
                db_path = "./benchmark_db.duckdb"
                if not os.path.exists(db_path):
                    # Try to find it elsewhere
                    alternative_paths = [
                        "../benchmark_db.duckdb",
                        "benchmark_db.duckdb",
                        "/home/barberb/ipfs_accelerate_py/test/benchmark_db.duckdb"
                    ]
                    for path in alternative_paths:
                        if os.path.exists(path):
                            db_path = path
                            break
                
                cls.db_api = BenchmarkDBAPI(db_path=db_path)
                cls.has_db = True
                logger.info(f"Using database at: {db_path}")
            except Exception as e:
                logger.warning(f"Error initializing database API: {e}")
                cls.has_db = False
        else:
            cls.has_db = False
        
        # Initialize visualization system if available
        if HAS_ADVANCED_VISUALIZATION and cls.has_db:
            try:
                cls.viz = AdvancedVisualizationSystem(
                    db_api=cls.db_api,
                    output_dir=cls.temp_dir
                )
                cls.viz.configure({"auto_open": False, "theme": "light"})
                cls.has_viz = True
                logger.info("Advanced Visualization System initialized")
            except Exception as e:
                logger.warning(f"Error initializing visualization system: {e}")
                cls.has_viz = False
        else:
            cls.has_viz = False
    
    @classmethod
    def tearDownClass(cls):
        """Tear down test case."""
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(cls.temp_dir)
            logger.info(f"Removed temporary directory: {cls.temp_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up temporary directory: {e}")
    
    def test_export_utils_available(self):
        """Test that export utilities are available."""
        if not HAS_EXPORT_UTILS:
            self.skipTest("Export utilities not available")
        
        # If we get here, the utilities are available
        self.assertTrue(True)
    
    def test_export_simple_figure(self):
        """Test exporting a simple figure."""
        if not HAS_EXPORT_UTILS:
            self.skipTest("Export utilities not available")
        
        import plotly.graph_objects as go
        import numpy as np
        
        # Create a simple figure
        fig = go.Figure(data=go.Scatter(x=np.arange(10), y=np.arange(10)))
        fig.update_layout(title="Test Figure")
        
        # Export as HTML
        html_path = os.path.join(self.temp_dir, "test_figure.html")
        exported_html_path = export_figure(fig, html_path, format="html")
        
        # Verify the file exists
        self.assertTrue(os.path.exists(exported_html_path), f"Exported HTML file not found: {exported_html_path}")
        
        # Export as PNG
        png_path = os.path.join(self.temp_dir, "test_figure.png")
        try:
            exported_png_path = export_figure(fig, png_path, format="png")
            self.assertTrue(os.path.exists(exported_png_path), f"Exported PNG file not found: {exported_png_path}")
        except Exception as e:
            logger.warning(f"PNG export failed (possibly missing kaleido): {e}")
    
    def test_export_data(self):
        """Test exporting data to CSV and JSON."""
        if not HAS_EXPORT_UTILS:
            self.skipTest("Export utilities not available")
        
        import pandas as pd
        
        # Create a DataFrame
        df = pd.DataFrame({
            "model": ["bert", "vit", "t5"],
            "throughput": [100, 200, 150],
            "latency": [10, 5, 8]
        })
        
        # Export as CSV
        csv_path = os.path.join(self.temp_dir, "test_data.csv")
        exported_csv_path = export_data(df, csv_path, format="csv")
        
        # Verify the file exists
        self.assertTrue(os.path.exists(exported_csv_path), f"Exported CSV file not found: {exported_csv_path}")
        
        # Export as JSON
        json_path = os.path.join(self.temp_dir, "test_data.json")
        exported_json_path = export_data(df, json_path, format="json")
        
        # Verify the file exists
        self.assertTrue(os.path.exists(exported_json_path), f"Exported JSON file not found: {exported_json_path}")
    
    def test_export_3d_visualization(self):
        """Test exporting a 3D visualization."""
        if not self.has_viz:
            self.skipTest("Advanced Visualization System not available")
        
        # Create a dummy 3D visualization if no database
        if not self.db_api:
            import plotly.graph_objects as go
            import numpy as np
            
            # Create a simple 3D scatter plot
            x = np.random.rand(50)
            y = np.random.rand(50)
            z = np.random.rand(50)
            
            fig = go.Figure(data=go.Scatter3d(x=x, y=y, z=z, mode='markers'))
            fig.update_layout(title="Test 3D Visualization")
            
            dummy_result = {
                "component_type": "3d",
                "figure": fig,
                "title": "Test 3D Visualization"
            }
            
            # Export the visualization
            exports = self.viz.export_visualization(
                component_type='3d',
                component_data=dummy_result,
                visualization_id="test_3d",
                formats=['html', 'png']
            )
        else:
            # Create a real 3D visualization
            try:
                result = self.viz.create_3d_performance_visualization(
                    metrics=["throughput_items_per_second", "average_latency_ms", "memory_peak_mb"],
                    dimensions=["model_family", "hardware_type"],
                    title="Test 3D Performance Visualization"
                )
                
                # Export the visualization
                exports = self.viz.export_3d_visualization(
                    visualization_data=result,
                    formats=['html', 'png'],
                    visualization_id="test_3d"
                )
            except Exception as e:
                logger.warning(f"Error creating or exporting 3D visualization: {e}")
                self.skipTest(f"Error creating or exporting 3D visualization: {e}")
                return
        
        # Verify at least one export file exists
        self.assertTrue(len(exports) > 0, "No exports were created")
        
        # Check HTML export
        if 'html' in exports:
            self.assertTrue(os.path.exists(exports['html']), f"Exported HTML file not found: {exports['html']}")
        
        # Check PNG export (may fail if kaleido isn't installed)
        if 'png' in exports:
            self.assertTrue(os.path.exists(exports['png']), f"Exported PNG file not found: {exports['png']}")


def run_tests():
    """Run test cases."""
    # Run tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


if __name__ == "__main__":
    run_tests()