#!/usr/bin/env python3
"""
Error Recovery Visualization Integration for Distributed Testing Framework

This module integrates the error recovery visualization system with the performance-based
error recovery tracking system. It provides helper functions to generate visualizations
and dashboards from error recovery performance data.
"""

import os
import sys
import logging
import anyio
from typing import Dict, List, Any, Optional
import argparse

# Import visualization module
from error_recovery_visualization import ErrorRecoveryVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("error_recovery_visualization_integration")


class ErrorRecoveryVisualizationIntegration:
    """
    Integration between the error recovery system and visualization tools.
    
    This class provides methods to generate visualizations and dashboards
    from error recovery performance data collected by the PerformanceBasedErrorRecovery system.
    """
    
    def __init__(self, recovery_system=None, db_connection=None, output_dir: str = None):
        """
        Initialize the integration.
        
        Args:
            recovery_system: Optional reference to the PerformanceBasedErrorRecovery instance
            db_connection: Optional database connection for direct data retrieval
            output_dir: Directory to save visualizations
        """
        self.recovery_system = recovery_system
        self.db_connection = db_connection
        self.output_dir = output_dir or os.path.join(os.getcwd(), "images")
        
        # Create visualizer
        self.visualizer = ErrorRecoveryVisualizer(output_dir=self.output_dir)
        
    async def generate_dashboard(self, days: int = 30, file_format: str = "png") -> str:
        """
        Generate a comprehensive dashboard of error recovery performance.
        
        Args:
            days: Number of days of history to include
            file_format: File format for visualizations (png, pdf, svg)
            
        Returns:
            Path to the generated dashboard
        """
        if not self.recovery_system:
            raise ValueError("Recovery system not provided")
        
        # Get performance data
        performance_data = self.recovery_system.get_performance_metrics()
        
        # Get progressive recovery data
        progressive_data = self.recovery_system.get_progressive_recovery_history()
        
        # Get time series data
        analysis_data = await self.recovery_system.analyze_recovery_performance(days=days)
        time_series_data = analysis_data.get("time_series", [])
        
        # Create visualizer with format
        self.visualizer = ErrorRecoveryVisualizer(output_dir=self.output_dir, file_format=file_format)
        
        # Generate dashboard
        dashboard_path = self.visualizer.generate_recovery_dashboard(
            performance_data,
            time_series_data,
            progressive_data
        )
        
        return dashboard_path
    
    async def generate_individual_visualizations(self, days: int = 30, file_format: str = "png") -> Dict[str, str]:
        """
        Generate individual visualizations for different aspects of error recovery performance.
        
        Args:
            days: Number of days of history to include
            file_format: File format for visualizations (png, pdf, svg)
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        if not self.recovery_system:
            raise ValueError("Recovery system not provided")
        
        # Get performance data
        performance_data = self.recovery_system.get_performance_metrics()
        
        # Get progressive recovery data
        progressive_data = self.recovery_system.get_progressive_recovery_history()
        
        # Get time series data
        analysis_data = await self.recovery_system.analyze_recovery_performance(days=days)
        time_series_data = analysis_data.get("time_series", [])
        
        # Create visualizer with format
        self.visualizer = ErrorRecoveryVisualizer(output_dir=self.output_dir, file_format=file_format)
        
        # Generate visualizations
        results = {}
        
        # Strategy performance
        results["strategy_performance"] = self.visualizer.visualize_strategy_performance(
            performance_data,
            filename="strategy_performance_dashboard"
        )
        
        # Error recovery heatmap
        results["error_recovery_heatmap"] = self.visualizer.visualize_error_recovery_heatmap(
            performance_data,
            filename="error_recovery_heatmap"
        )
        
        # Performance trends
        results["performance_trends"] = self.visualizer.visualize_performance_trends(
            time_series_data,
            filename="performance_trend_graphs"
        )
        
        # Progressive recovery
        results["progressive_recovery"] = self.visualizer.visualize_progressive_recovery(
            progressive_data,
            filename="progressive_recovery_analysis"
        )
        
        return results
    
    async def generate_visualization_for_error_type(self, error_type: str, file_format: str = "png") -> str:
        """
        Generate visualization focused on a specific error type.
        
        Args:
            error_type: The error type to focus on
            file_format: File format for visualizations (png, pdf, svg)
            
        Returns:
            Path to the generated visualization
        """
        if not self.recovery_system:
            raise ValueError("Recovery system not provided")
        
        # Get recommendations for this error type
        recommendations = self.recovery_system.get_strategy_recommendations(error_type)
        
        if not recommendations:
            logger.warning(f"No recommendations available for error type {error_type}")
            return f"No data available for error type {error_type}"
        
        # Create visualizer with format
        self.visualizer = ErrorRecoveryVisualizer(output_dir=self.output_dir, file_format=file_format)
        
        # Prepare data for visualization
        import matplotlib.pyplot as plt
        import numpy as np
        
        try:
            # Extract data
            strategy_ids = [rec["strategy_name"] for rec in recommendations]
            scores = [rec["score"] for rec in recommendations]
            success_rates = [rec["success_rate"] for rec in recommendations]
            recovery_times = [rec["avg_recovery_time"] for rec in recommendations]
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Left plot: Overall scores
            colors = ['red' if s < 0.4 else 'orange' if s < 0.7 else 'green' for s in scores]
            bars = ax1.bar(strategy_ids, scores, color=colors, alpha=0.7)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom')
            
            ax1.set_title(f'Strategy Scores for {error_type}')
            ax1.set_ylabel('Overall Score')
            ax1.set_ylim(0, 1.1)
            plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
            
            # Right plot: Recovery Time vs Success Rate (scatter)
            ax2.scatter(recovery_times, success_rates, s=80, alpha=0.7)
            
            # Add labels to points
            for i, name in enumerate(strategy_ids):
                ax2.annotate(name, (recovery_times[i], success_rates[i]),
                           xytext=(5, 0), textcoords='offset points')
            
            ax2.set_title(f'Recovery Time vs Success Rate for {error_type}')
            ax2.set_xlabel('Recovery Time (seconds)')
            ax2.set_ylabel('Success Rate')
            ax2.set_ylim(0, 1.1)
            ax2.grid(True, alpha=0.3)
            
            # Add overall title
            fig.suptitle(f'Error Recovery Analysis for {error_type}', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
            
            # Save visualization
            output_path = os.path.join(self.output_dir, f"error_type_{error_type.lower()}.{file_format}")
            plt.savefig(output_path)
            plt.close()
            
            return output_path
        except Exception as e:
            logger.error(f"Error generating visualization for error type {error_type}: {e}")
            return f"Error generating visualization: {e}"


def create_visualization_integration(recovery_system=None, db_connection=None, output_dir: str = None) -> ErrorRecoveryVisualizationIntegration:
    """
    Create an instance of the error recovery visualization integration.
    
    Args:
        recovery_system: Optional reference to the PerformanceBasedErrorRecovery instance
        db_connection: Optional database connection for direct data retrieval
        output_dir: Directory to save visualizations
        
    Returns:
        ErrorRecoveryVisualizationIntegration instance
    """
    return ErrorRecoveryVisualizationIntegration(
        recovery_system=recovery_system,
        db_connection=db_connection,
        output_dir=output_dir
    )


# Main function to run as a standalone tool
async def async_main(args):
    """Async main function."""
    # Import necessary modules
    import duckdb
    import sys
    import os
    
    # Import the mock implementation from run_test_error_recovery_visualization.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_test_error_recovery_visualization import PerformanceBasedErrorRecovery
    
    # Connect to database
    if args.db_path:
        db_connection = duckdb.connect(args.db_path)
    else:
        # Try to connect to default path or create in-memory database
        try:
            db_connection = duckdb.connect("./error_recovery_with_performance_test_db/recovery_data.duckdb")
        except:
            logger.warning("No database file specified, using in-memory database")
            db_connection = duckdb.connect(":memory:")
    
    # Create dummy coordinator
    class DummyCoordinator:
        def __init__(self):
            self.db = db_connection
    
    # Create a mock recovery manager
    class MockRecoveryManager:
        def __init__(self):
            self.strategies = {
                "retry": MockStrategy("Retry Strategy", "low"),
                "worker_recovery": MockStrategy("Worker Recovery Strategy", "medium"),
                "database_recovery": MockStrategy("Database Recovery Strategy", "medium"),
                "coordinator_recovery": MockStrategy("Coordinator Recovery Strategy", "high"),
                "system_recovery": MockStrategy("System Recovery Strategy", "critical")
            }
            
    class MockStrategy:
        def __init__(self, name, level):
            self.name = name
            self.level = MockLevel(level)
            
        async def execute(self, error_info):
            return True
            
    class MockLevel:
        def __init__(self, value):
            self.value = value
            
    class MockErrorHandler:
        def __init__(self):
            pass
            
        def register_error_hook(self, error_type, hook):
            pass
    
    # Create recovery system
    recovery_system = PerformanceBasedErrorRecovery(
        error_handler=MockErrorHandler(),
        recovery_manager=MockRecoveryManager(),
        coordinator=DummyCoordinator(),
        db_connection=db_connection
    )
    
    # Create visualization integration
    integration = create_visualization_integration(
        recovery_system=recovery_system,
        db_connection=db_connection,
        output_dir=args.output_dir
    )
    
    if args.command == 'dashboard':
        # Generate dashboard
        dashboard_path = await integration.generate_dashboard(days=args.days, file_format=args.format)
        logger.info(f"Dashboard generated at: {dashboard_path}")
        print(f"Dashboard generated at: {dashboard_path}")
        
    elif args.command == 'individual':
        # Generate individual visualizations
        vis_paths = await integration.generate_individual_visualizations(days=args.days, file_format=args.format)
        logger.info(f"Individual visualizations generated:")
        for name, path in vis_paths.items():
            logger.info(f"  - {name}: {path}")
            print(f"{name}: {path}")
            
    elif args.command == 'error-type':
        # Generate visualization for specific error type
        if not args.error_type:
            logger.error("Error type must be specified with --error-type")
            return
            
        vis_path = await integration.generate_visualization_for_error_type(args.error_type, file_format=args.format)
        logger.info(f"Error type visualization generated at: {vis_path}")
        print(f"Error type visualization generated at: {vis_path}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Error Recovery Visualization Integration Tool")
    parser.add_argument("--db-path", type=str, help="Path to the database file")
    parser.add_argument("--output-dir", type=str, help="Directory to save visualizations")
    parser.add_argument("--days", type=int, default=30, help="Number of days of history to analyze")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Output file format")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Generate comprehensive dashboard')
    
    # Individual visualizations command
    individual_parser = subparsers.add_parser('individual', help='Generate individual visualizations')
    
    # Error type visualization command
    error_type_parser = subparsers.add_parser('error-type', help='Generate visualization for specific error type')
    error_type_parser.add_argument("--error-type", type=str, help="Error type to visualize")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run async main
    anyio.run(async_main(args))


if __name__ == "__main__":
    main()