"""
Monitoring Dashboard E2E Test Results Integration

This module provides integration between the Monitoring Dashboard and the End-to-End
Testing Framework, displaying test results in the dashboard.

Usage:
    from duckdb_api.distributed_testing.dashboard.monitoring_dashboard_e2e_integration import E2ETestResultsIntegration
    
    # Create integration
    e2e_integration = E2ETestResultsIntegration()
    
    # Add to monitoring dashboard
    dashboard = MonitoringDashboard(e2e_test_integration=e2e_integration)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class E2ETestResultsIntegration:
    """Integration between Monitoring Dashboard and End-to-End Testing Framework."""
    
    def __init__(self, report_dir: str = './e2e_test_reports', visualization_dir: str = './e2e_visualizations'):
        """Initialize the E2E test results integration.
        
        Args:
            report_dir: Directory containing E2E test reports
            visualization_dir: Directory containing E2E test visualizations
        """
        self.report_dir = Path(report_dir)
        self.visualization_dir = Path(visualization_dir)
        self.test_results = {}
        self.visualizations = {}
        
        # Create directories if they don't exist
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Load existing test results
        self._load_test_results()
    
    def _load_test_results(self):
        """Load existing test results from the report directory."""
        try:
            # Find all report files in the report directory
            report_files = list(self.report_dir.glob("*_results.json"))
            
            for report_file in report_files:
                try:
                    # Extract test ID from filename
                    test_id = report_file.name.replace("_results.json", "")
                    
                    # Load report data
                    with open(report_file, 'r') as f:
                        report_data = json.load(f)
                    
                    # Store in test results dictionary
                    self.test_results[test_id] = report_data
                    
                    # Check if visualizations exist for this test
                    visualization_files = {
                        'summary': self.visualization_dir / f"{test_id}_summary.html",
                        'component': self.visualization_dir / f"{test_id}_component_status.html",
                        'timing': self.visualization_dir / f"{test_id}_timing.html",
                        'failures': self.visualization_dir / f"{test_id}_failures.html"
                    }
                    
                    # Store visualizations if they exist
                    self.visualizations[test_id] = {}
                    for viz_type, viz_file in visualization_files.items():
                        if viz_file.exists():
                            with open(viz_file, 'r') as f:
                                self.visualizations[test_id][viz_type] = f.read()
                
                except Exception as e:
                    logger.error(f"Error loading test result from {report_file}: {e}")
        
        except Exception as e:
            logger.error(f"Error loading test results: {e}")
    
    def get_test_list(self) -> List[Dict[str, Any]]:
        """Get a list of available tests.
        
        Returns:
            List of test metadata including ID and timestamp
        """
        tests = []
        for test_id, test_data in self.test_results.items():
            try:
                timestamp = test_data.get('timestamp', '')
                tests.append({
                    'id': test_id,
                    'timestamp': timestamp
                })
            except Exception as e:
                logger.error(f"Error processing test {test_id}: {e}")
        
        # Sort by timestamp, most recent first
        tests.sort(key=lambda x: x['timestamp'], reverse=True)
        return tests
    
    def get_test_details(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific test.
        
        Args:
            test_id: ID of the test to get details for
            
        Returns:
            Test details or None if not found
        """
        return self.test_results.get(test_id)
    
    def get_test_visualizations(self, test_id: str) -> Dict[str, str]:
        """Get visualizations for a specific test.
        
        Args:
            test_id: ID of the test to get visualizations for
            
        Returns:
            Dictionary of visualization HTML content by type
        """
        return self.visualizations.get(test_id, {})
    
    def add_test_result(self, test_id: str, result_data: Dict[str, Any], visualizations: Dict[str, str]) -> bool:
        """Add a new test result with visualizations.
        
        Args:
            test_id: ID of the test
            result_data: Test result data
            visualizations: Dictionary of visualization HTML content by type
            
        Returns:
            Success status
        """
        try:
            # Store the test result
            self.test_results[test_id] = result_data
            
            # Store the visualizations
            self.visualizations[test_id] = visualizations
            
            # Write test result to file
            report_file = self.report_dir / f"{test_id}_results.json"
            with open(report_file, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            # Write visualizations to files
            for viz_type, viz_content in visualizations.items():
                viz_file = self.visualization_dir / f"{test_id}_{viz_type}.html"
                with open(viz_file, 'w') as f:
                    f.write(viz_content)
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding test result {test_id}: {e}")
            return False