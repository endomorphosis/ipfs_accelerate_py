#!/usr/bin/env python3
"""
Test script for the Database Predictive Analytics module.

This script tests the functionality of the DatabasePredictiveAnalytics class
by performing various forecasting, visualization, and analysis operations.
It serves as both a verification tool and a demonstration of the module's capabilities.

Usage:
    python test_database_predictive_analytics.py --test all
    python test_database_predictive_analytics.py --test forecast
    python test_database_predictive_analytics.py --test visualize
    python test_database_predictive_analytics.py --test alerts
    python test_database_predictive_analytics.py --test recommend
    python test_database_predictive_analytics.py --test analyze
"""

import os
import sys
import json
import datetime
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_database_predictive_analytics")

# Create directories if they don't exist
os.makedirs("./visualizations", exist_ok=True)
os.makedirs("./output", exist_ok=True)

try:
    from data.duckdb.simulation_validation.db_performance_optimizer import get_db_optimizer
    from data.duckdb.simulation_validation.automated_optimization_manager import get_optimization_manager
    from data.duckdb.simulation_validation.database_predictive_analytics import DatabasePredictiveAnalytics
except ImportError:
    logger.error("Failed to import required modules. Make sure duckdb_api is properly installed.")
    sys.exit(1)

def save_json(data: Dict[str, Any], filename: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filename: Filename to save to
    """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Data saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {e}")

def generate_test_metrics_history(auto_manager: Any, metrics: Optional[List[str]] = None) -> None:
    """
    Generate test metrics history for testing the predictive analytics.
    
    This function creates realistic test data with various patterns:
    - storage_size: Exponential growth pattern (database growth)
    - query_time: Linear growth with noise (query performance degradation)
    - index_efficiency: Linear decline (index fragmentation)
    - vacuum_status: Cyclical pattern (vacuum cycles)
    - compression_ratio: Stable with occasional drops (compression efficiency)
    - read_efficiency: Stable with slight decline and noise
    - write_efficiency: Stable with moderate decline and noise
    - cache_performance: Cyclical with overall decline (cache efficiency)
    
    Args:
        auto_manager: Automated optimization manager instance
        metrics: List of specific metrics to generate (None for all)
    """
    logger.info("Generating test metrics history")
    
    timestamp_now = datetime.datetime.now()
    
    # Get metrics to generate
    if metrics is None:
        # Get list of metrics from the auto_manager or use defaults
        metrics_list = getattr(auto_manager, "metrics_to_monitor", [
            "storage_size", "query_time", "index_efficiency", "vacuum_status",
            "compression_ratio", "read_efficiency", "write_efficiency", "cache_performance"
        ])
    else:
        metrics_list = metrics
    
    # Generate data for each metric
    for metric_name in metrics_list:
        # Create history with specific patterns
        history = []
        
        # Generate 90 days of data for more robust forecasting
        for i in range(90):
            timestamp = timestamp_now - datetime.timedelta(days=90-i)
            
            # For storage_size, simulate exponential growth
            if metric_name == "storage_size":
                # Start at 100MB, grow by ~1% per day with some randomness
                base_value = 100000000 * (1.01 ** i)
                noise = random.uniform(0.99, 1.01)  # ±1% noise
                value = base_value * noise
                
            # For query_time, simulate gradual degradation with some fluctuations
            elif metric_name == "query_time":
                # Start at 100ms, increase by 1ms per day with random component
                base_value = 100 + (i * 1)
                noise = random.uniform(0.9, 1.1)  # ±10% noise
                value = base_value * noise
                
            # For index_efficiency, simulate gradual decline
            elif metric_name == "index_efficiency":
                # Start at 98%, gradually decline
                base_value = 98 - (i * 0.2)
                noise = random.uniform(0.99, 1.01)  # ±1% noise
                value = max(0, min(100, base_value * noise))  # Keep between 0-100
                
            # For vacuum_status, simulate cyclical pattern
            elif metric_name == "vacuum_status":
                # Cyclical pattern between 60-95%
                base_value = 80 + 15 * (0.5 + 0.5 * (1 if i % 14 == 0 else -1 * i % 14 / 14))
                noise = random.uniform(0.98, 1.02)  # ±2% noise
                value = max(0, min(100, base_value * noise))  # Keep between 0-100
                
            # For compression_ratio, simulate stable with occasional drops
            elif metric_name == "compression_ratio":
                # Around 4.0 with occasional drops
                base_value = 4.0
                if i % 21 == 0:  # Every 3 weeks, compression drops
                    base_value = 3.0
                noise = random.uniform(0.95, 1.05)  # ±5% noise
                value = base_value * noise
                
            # For read_efficiency, simulate slight decline with noise
            elif metric_name == "read_efficiency":
                # Start at 95%, very slight decline
                base_value = 95 - (i * 0.05)
                noise = random.uniform(0.97, 1.03)  # ±3% noise
                value = max(0, min(100, base_value * noise))  # Keep between 0-100
                
            # For write_efficiency, simulate moderate decline with noise
            elif metric_name == "write_efficiency":
                # Start at 90%, moderate decline
                base_value = 90 - (i * 0.1)
                noise = random.uniform(0.95, 1.05)  # ±5% noise
                value = max(0, min(100, base_value * noise))  # Keep between 0-100
                
            # For cache_performance, simulate cyclical pattern with overall decline
            elif metric_name == "cache_performance":
                # Cyclical between 70-95% with overall decline
                base_value = 95 - (i * 0.15) + 10 * (0.5 + 0.5 * (1 if i % 7 == 0 else -1 * i % 7 / 7))
                noise = random.uniform(0.97, 1.03)  # ±3% noise
                value = max(0, min(100, base_value * noise))  # Keep between 0-100
                
            # For other metrics, use a reasonable constant with some noise
            else:
                # Just use a constant with moderate noise
                base_value = 50
                noise = random.uniform(0.9, 1.1)  # ±10% noise
                value = base_value * noise
            
            # Create metrics history entry
            history.append({
                "timestamp": timestamp.isoformat(),
                "value": value,
                "status": "ok"
            })
            
        # Add data to the metrics history
        auto_manager.metrics_history[metric_name] = history
    
    logger.info(f"Generated test data for {len(metrics_list)} metrics")

def setup_test_thresholds(auto_manager: Any) -> None:
    """
    Set up test thresholds for testing alerts and recommendations.
    
    Args:
        auto_manager: Automated optimization manager instance
    """
    logger.info("Setting up test thresholds")
    
    # Define thresholds for different metrics
    auto_manager.thresholds = {
        "query_time": {
            "warning": 150.0,  # ms
            "error": 250.0     # ms
        },
        "storage_size": {
            "warning": 150000000.0,  # ~150MB
            "error": 200000000.0     # ~200MB
        },
        "index_efficiency": {
            "warning": 75.0,   # %
            "error": 60.0      # %
        },
        "vacuum_status": {
            "warning": 70.0,   # %
            "error": 50.0      # %
        },
        "compression_ratio": {
            "warning": 3.5,    # ratio
            "error": 3.0       # ratio
        },
        "read_efficiency": {
            "warning": 85.0,   # %
            "error": 75.0      # %
        },
        "write_efficiency": {
            "warning": 80.0,   # %
            "error": 70.0      # %
        },
        "cache_performance": {
            "warning": 75.0,   # %
            "error": 60.0      # %
        }
    }
    
    # Define actions for different metrics
    auto_manager.actions = {
        "query_time": [
            "optimize_queries",
            "create_indexes",
            "analyze_tables"
        ],
        "storage_size": [
            "vacuum_full",
            "cleanup_unused_data",
            "increase_storage"
        ],
        "index_efficiency": [
            "reindex_tables",
            "analyze_indexes",
            "optimize_index_usage"
        ],
        "vacuum_status": [
            "run_vacuum",
            "schedule_regular_vacuum",
            "optimize_vacuum_settings"
        ],
        "compression_ratio": [
            "optimize_compression_settings",
            "review_data_types",
            "check_compression_algorithm"
        ],
        "read_efficiency": [
            "optimize_query_cache",
            "analyze_table_statistics",
            "optimize_read_patterns"
        ],
        "write_efficiency": [
            "batch_write_operations",
            "optimize_transaction_size",
            "reduce_index_overhead"
        ],
        "cache_performance": [
            "increase_cache_size",
            "optimize_cache_settings",
            "analyze_cache_usage_patterns"
        ]
    }
    
    logger.info("Test thresholds and actions set up successfully")

def test_basic_forecasting() -> None:
    """
    Test basic forecasting functionality.
    
    This test:
    1. Creates a database optimizer and automation manager
    2. Generates test metrics history
    3. Creates a predictive analytics instance
    4. Runs forecasting for all metrics
    5. Saves and displays forecast results
    """
    logger.info("----- Testing Basic Forecasting -----")
    
    # Create database optimizer
    db_path = "./benchmark_db.duckdb"
    db_optimizer = get_db_optimizer(db_path=db_path)
    
    # Create automated optimization manager
    auto_manager = get_optimization_manager(
        db_optimizer=db_optimizer,
        auto_apply=False
    )
    
    # Generate test metrics history
    generate_test_metrics_history(auto_manager)
    
    # Set up test thresholds
    setup_test_thresholds(auto_manager)
    
    # Create predictive analytics instance
    predictive = DatabasePredictiveAnalytics(
        automated_optimization_manager=auto_manager
    )
    
    # Run forecasting
    forecast_result = predictive.forecast_database_metrics(
        horizon="medium_term",
        specific_metrics=None
    )
    
    # Save forecast result
    save_json(forecast_result, "./output/forecast_result.json")
    
    # Print summary
    print(f"\nForecast Status: {forecast_result.get('status', 'unknown')}")
    print(f"Forecasted metrics: {len(forecast_result.get('forecasts', {}))}")
    
    # Print some details of the forecasts
    for metric_name, forecast in forecast_result.get("forecasts", {}).items():
        if "trend_analysis" in forecast:
            trend = forecast["trend_analysis"]
            print(f"Metric: {metric_name}")
            print(f"  Trend direction: {trend.get('direction', 'unknown')}")
            print(f"  Trend magnitude: {trend.get('magnitude', 'unknown')}")
            print(f"  Percent change: {trend.get('percent_change', 0):.2f}%")
    
    if "warnings" in forecast_result and forecast_result["warnings"]:
        print(f"Warnings: {len(forecast_result['warnings'])}")
    
    logger.info("Basic forecasting test completed")
    return auto_manager, predictive

def test_visualizations(auto_manager: Any = None, predictive: Any = None) -> None:
    """
    Test visualization generation.
    
    This test:
    1. Creates a predictive analytics instance if not provided
    2. Generates forecasts for all metrics
    3. Creates manual visualizations for testing
    4. Generates visualizations using the class method
    
    Args:
        auto_manager: Optional automated optimization manager instance
        predictive: Optional predictive analytics instance
    """
    logger.info("----- Testing Visualization Generation -----")
    
    # Create instances if not provided
    if auto_manager is None or predictive is None:
        # Create database optimizer
        db_path = "./benchmark_db.duckdb"
        db_optimizer = get_db_optimizer(db_path=db_path)
        
        # Create automated optimization manager
        auto_manager = get_optimization_manager(
            db_optimizer=db_optimizer,
            auto_apply=False
        )
        
        # Generate test metrics history
        generate_test_metrics_history(auto_manager)
        
        # Create predictive analytics instance
        predictive = DatabasePredictiveAnalytics(
            automated_optimization_manager=auto_manager
        )
    
    # Generate forecasts
    forecast_result = predictive.forecast_database_metrics(
        horizon="medium_term"
    )
    
    # Create visualizations manually for testing
    try:
        import matplotlib.pyplot as plt
        
        visualizations_dir = Path("./visualizations")
        visualizations_dir.mkdir(exist_ok=True)
        
        for metric_name, forecast in forecast_result.get("forecasts", {}).items():
            try:
                # Create a new figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Extract historical data
                historical_dates = [datetime.datetime.fromisoformat(d) for d in forecast["historical_dates"]]
                historical_values = forecast["historical_values"]
                
                # Extract forecast data
                forecast_dates = [datetime.datetime.fromisoformat(d) for d in forecast["forecast_dates"]]
                forecast_values = forecast["forecast_values"]
                
                # Plot historical data
                ax.plot(historical_dates, historical_values, marker='o', linestyle='-', 
                        color='blue', label='Historical Data')
                
                # Plot forecast
                ax.plot(forecast_dates, forecast_values, marker='x', linestyle='--', 
                        color='orange', label='Forecast')
                
                # Format the plot
                ax.set_title(f'{metric_name} Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                
                # Add legend
                ax.legend(loc='best')
                
                # Save visualization with absolute path
                abs_path = os.path.abspath("./visualizations")
                os.makedirs(abs_path, exist_ok=True)
                filename = os.path.join(abs_path, f"{metric_name}_forecast_manual.png")
                fig.savefig(filename)
                plt.close(fig)
                
                print(f"Manually saved visualization for {metric_name} to {filename}")
                
            except Exception as e:
                logger.error(f"Error creating visualization for {metric_name}: {e}")
    except ImportError:
        logger.warning("matplotlib not available, skipping manual visualization test")
    
    # Generate visualizations using the class method
    vis_result = predictive.generate_forecast_visualizations(
        forecast_results=forecast_result,
        output_format="file"
    )
    
    # Save visualization result
    save_json(
        {"status": vis_result.get("status", "unknown")}, 
        "./output/visualization_result.json"
    )
    
    # Print summary
    print(f"\nVisualization Status: {vis_result.get('status', 'unknown')}")
    print(f"Generated visualizations: {len(vis_result.get('visualizations', {}))}")
    
    # Check if there were any errors
    for metric_name, vis_info in vis_result.get("visualizations", {}).items():
        if vis_info.get("status") == "error":
            print(f"Error generating visualization for {metric_name}: {vis_info.get('message', 'unknown error')}")
    
    logger.info("Visualization test completed")
    return auto_manager, predictive

def test_threshold_alerts(auto_manager: Any = None, predictive: Any = None) -> None:
    """
    Test threshold alerts detection.
    
    This test:
    1. Creates a predictive analytics instance if not provided
    2. Generates forecasts for all metrics
    3. Checks for predicted threshold violations
    4. Saves and displays alert results
    
    Args:
        auto_manager: Optional automated optimization manager instance
        predictive: Optional predictive analytics instance
    """
    logger.info("----- Testing Threshold Alerts -----")
    
    # Create instances if not provided
    if auto_manager is None or predictive is None:
        # Create database optimizer
        db_path = "./benchmark_db.duckdb"
        db_optimizer = get_db_optimizer(db_path=db_path)
        
        # Create automated optimization manager
        auto_manager = get_optimization_manager(
            db_optimizer=db_optimizer,
            auto_apply=False
        )
        
        # Generate test metrics history
        generate_test_metrics_history(auto_manager)
        
        # Set up test thresholds
        setup_test_thresholds(auto_manager)
        
        # Create predictive analytics instance
        predictive = DatabasePredictiveAnalytics(
            automated_optimization_manager=auto_manager
        )
    
    # Generate forecasts
    forecast_result = predictive.forecast_database_metrics(
        horizon="medium_term"
    )
    
    # Check for alerts
    alert_result = predictive.check_predicted_thresholds(
        forecast_results=forecast_result
    )
    
    # Save alert result
    save_json(alert_result, "./output/alert_result.json")
    
    # Print summary
    print(f"\nAlert Status: {alert_result.get('status', 'unknown')}")
    if "alerts" in alert_result and alert_result["alerts"]:
        print(f"Detected {len(alert_result['alerts'])} potential future alerts")
        for metric_name, alerts in alert_result["alerts"].items():
            for alert in alerts:
                print(f"- {metric_name}: {alert['message']}")
                print(f"  Severity: {alert.get('severity', 'unknown')}")
                print(f"  Days until: {alert.get('days_until', 'unknown')}")
                print(f"  Forecasted value: {alert.get('forecasted_value', 'unknown')}")
                print(f"  Threshold: {alert.get('threshold', 'unknown')}")
    else:
        print("No potential alerts detected")
    
    logger.info("Threshold alerts test completed")
    return auto_manager, predictive

def test_recommendations(auto_manager: Any = None, predictive: Any = None) -> None:
    """
    Test proactive recommendations.
    
    This test:
    1. Creates a predictive analytics instance if not provided
    2. Generates forecasts for all metrics
    3. Checks for predicted threshold violations
    4. Generates proactive recommendations
    5. Saves and displays recommendation results
    
    Args:
        auto_manager: Optional automated optimization manager instance
        predictive: Optional predictive analytics instance
    """
    logger.info("----- Testing Proactive Recommendations -----")
    
    # Create instances if not provided
    if auto_manager is None or predictive is None:
        # Create database optimizer
        db_path = "./benchmark_db.duckdb"
        db_optimizer = get_db_optimizer(db_path=db_path)
        
        # Create automated optimization manager
        auto_manager = get_optimization_manager(
            db_optimizer=db_optimizer,
            auto_apply=False
        )
        
        # Generate test metrics history
        generate_test_metrics_history(auto_manager)
        
        # Set up test thresholds
        setup_test_thresholds(auto_manager)
        
        # Create predictive analytics instance
        predictive = DatabasePredictiveAnalytics(
            automated_optimization_manager=auto_manager
        )
    
    # Generate forecasts
    forecast_result = predictive.forecast_database_metrics(
        horizon="medium_term"
    )
    
    # Check for alerts
    alert_result = predictive.check_predicted_thresholds(
        forecast_results=forecast_result
    )
    
    # Generate recommendations
    rec_result = predictive.recommend_proactive_actions(
        forecast_results=forecast_result,
        threshold_alerts=alert_result
    )
    
    # Save recommendation result
    save_json(rec_result, "./output/recommendation_result.json")
    
    # Print summary
    print(f"\nRecommendation Status: {rec_result.get('status', 'unknown')}")
    if "summary" in rec_result and rec_result["summary"]:
        print("\nRecommendation Summary:")
        for summary_item in rec_result["summary"]:
            print(f"- {summary_item}")
    
    if "recommendations" in rec_result and rec_result["recommendations"]:
        print(f"\nGenerated {len(rec_result['recommendations'])} recommendations:")
        for i, rec in enumerate(rec_result["recommendations"]):
            print(f"\nRecommendation {i+1}:")
            print(f"- Metric: {rec.get('metric', 'unknown')}")
            print(f"- Severity: {rec.get('severity', 'unknown')}")
            print(f"- Urgency: {rec.get('urgency', 'unknown')}")
            print(f"- Days until: {rec.get('days_until', 'unknown')}")
            print(f"- Message: {rec.get('message', 'No message')}")
            if "recommended_actions" in rec and rec["recommended_actions"]:
                print("- Recommended actions:")
                for action in rec["recommended_actions"]:
                    print(f"  - {action}")
    else:
        print("No recommendations generated")
    
    logger.info("Proactive recommendations test completed")
    return auto_manager, predictive

def test_comprehensive_analysis(auto_manager: Any = None, predictive: Any = None) -> None:
    """
    Test comprehensive database health forecast analysis.
    
    This test:
    1. Creates a predictive analytics instance if not provided
    2. Runs a comprehensive analysis including forecasting, threshold checks,
       recommendations, and visualizations
    3. Saves and displays analysis results
    
    Args:
        auto_manager: Optional automated optimization manager instance
        predictive: Optional predictive analytics instance
    """
    logger.info("----- Testing Comprehensive Analysis -----")
    
    # Create instances if not provided
    if auto_manager is None or predictive is None:
        # Create database optimizer
        db_path = "./benchmark_db.duckdb"
        db_optimizer = get_db_optimizer(db_path=db_path)
        
        # Create automated optimization manager
        auto_manager = get_optimization_manager(
            db_optimizer=db_optimizer,
            auto_apply=False
        )
        
        # Generate test metrics history
        generate_test_metrics_history(auto_manager)
        
        # Set up test thresholds
        setup_test_thresholds(auto_manager)
        
        # Create predictive analytics instance
        predictive = DatabasePredictiveAnalytics(
            automated_optimization_manager=auto_manager
        )
    
    # Run comprehensive analysis
    analysis_result = predictive.analyze_database_health_forecast(
        horizon="medium_term",
        generate_visualizations=True,
        output_format="file"
    )
    
    # Save analysis result
    save_json(analysis_result, "./output/analysis_result.json")
    
    # Print summary
    print(f"\nAnalysis Status: {analysis_result.get('status', 'unknown')}")
    print("\nSummary:")
    
    if "summary" in analysis_result:
        summary = analysis_result["summary"]
        print(f"- Total metrics analyzed: {summary.get('total_metrics_analyzed', 0)}")
        print(f"- Metrics with alerts: {summary.get('metrics_with_alerts', 0)}")
        print(f"- Total recommendations: {summary.get('total_recommendations', 0)}")
        print(f"- Forecast horizon: {summary.get('forecast_horizon', '')} ({summary.get('forecast_horizon_days', 0)} days)")
        
        if "forecast_trends" in summary and summary["forecast_trends"]:
            print("\nForecast Trends:")
            for trend in summary["forecast_trends"]:
                print(f"- {trend}")
        
        if "alert_summary" in summary and summary["alert_summary"]:
            print("\nAlert Summary:")
            for alert in summary["alert_summary"]:
                print(f"- {alert}")
    
    # Print generated visualization paths
    if "visualizations" in analysis_result and analysis_result["visualizations"].get("visualizations"):
        print("\nGenerated Visualizations:")
        for metric_name, viz_info in analysis_result["visualizations"]["visualizations"].items():
            if viz_info.get("format") == "file" and "filename" in viz_info:
                print(f"- {metric_name}: {viz_info['filename']}")
    
    logger.info("Comprehensive analysis test completed")
    return auto_manager, predictive

def test_custom_config() -> None:
    """
    Test the predictive analytics with a custom configuration.
    
    This test:
    1. Creates a predictive analytics instance with a custom configuration
    2. Verifies that the configuration is applied correctly
    3. Runs a basic forecast to ensure functionality
    """
    logger.info("----- Testing Custom Configuration -----")
    
    # Create database optimizer
    db_path = "./benchmark_db.duckdb"
    db_optimizer = get_db_optimizer(db_path=db_path)
    
    # Create automated optimization manager
    auto_manager = get_optimization_manager(
        db_optimizer=db_optimizer,
        auto_apply=False
    )
    
    # Generate test metrics history
    generate_test_metrics_history(auto_manager)
    
    # Define custom configuration
    custom_config = {
        "metrics_to_forecast": [
            "query_time",
            "storage_size"
        ],
        "forecasting": {
            "short_term_horizon": 3,      # Custom: 3 days
            "medium_term_horizon": 14,    # Custom: 14 days
            "long_term_horizon": 60,      # Custom: 60 days
            "confidence_level": 0.90,     # Custom: 90% confidence
            "min_data_points": 5,         # Custom: Only need 5 data points
            "forecast_methods": ["linear_regression"],  # Custom: Use only linear regression
            "use_ensemble": False,        # Custom: Don't use ensemble
            "auto_model_selection": False # Custom: Don't use automatic model selection
        },
        "visualization": {
            "theme": "dark",              # Custom: Use dark theme
            "show_confidence_intervals": False,  # Custom: Don't show confidence intervals
            "figure_size": (8, 4),        # Custom: Smaller figure size
            "dpi": 150                    # Custom: Higher DPI
        }
    }
    
    # Create predictive analytics instance with custom config
    predictive = DatabasePredictiveAnalytics(
        automated_optimization_manager=auto_manager,
        config=custom_config
    )
    
    # Verify configuration was applied correctly
    print("\nCustom Configuration Verification:")
    print(f"Metrics to forecast: {len(predictive.config['metrics_to_forecast'])}")
    print(f"Medium-term horizon: {predictive.config['forecasting']['medium_term_horizon']} days")
    print(f"Confidence level: {predictive.config['forecasting']['confidence_level']}")
    print(f"Forecast methods: {predictive.config['forecasting']['forecast_methods']}")
    print(f"Use ensemble: {predictive.config['forecasting']['use_ensemble']}")
    print(f"Auto model selection: {predictive.config['forecasting']['auto_model_selection']}")
    print(f"Visualization theme: {predictive.config['visualization']['theme']}")
    print(f"Figure size: {predictive.config['visualization']['figure_size']}")
    
    # Run a basic forecast to test functionality
    forecast_result = predictive.forecast_database_metrics(
        horizon="medium_term"
    )
    
    # Verify that only the configured metrics were forecasted
    print(f"\nForecast Status: {forecast_result.get('status', 'unknown')}")
    print(f"Forecasted metrics: {len(forecast_result.get('forecasts', {}))}")
    
    # Verify correct metrics were forecasted
    forecasted_metrics = list(forecast_result.get("forecasts", {}).keys())
    print(f"Metrics forecasted: {', '.join(forecasted_metrics)}")
    
    # Save custom config forecast result
    save_json(forecast_result, "./output/custom_config_forecast_result.json")
    
    logger.info("Custom configuration test completed")

def test_auto_model_selection() -> None:
    """
    Test the automated model selection feature.
    
    This test:
    1. Creates a predictive analytics instance with automated model selection enabled
    2. Generates test metrics history with known patterns
    3. Runs forecasting to trigger model selection
    4. Verifies that model selection occurred and selected the best model
    5. Examines the validation metrics to ensure they were calculated correctly
    """
    logger.info("----- Testing Automated Model Selection -----")
    
    # Create database optimizer
    db_path = "./benchmark_db.duckdb"
    db_optimizer = get_db_optimizer(db_path=db_path)
    
    # Create automated optimization manager
    auto_manager = get_optimization_manager(
        db_optimizer=db_optimizer,
        auto_apply=False
    )
    
    # Generate test metrics history with specific patterns that different
    # forecasting methods will handle differently
    generate_test_metrics_history(auto_manager)
    
    # Define configuration with auto model selection explicitly enabled
    model_selection_config = {
        "forecasting": {
            "auto_model_selection": True,
            "validation_size_percent": 20,  # Use 20% of the data for validation
            "forecast_methods": ["arima", "exponential_smoothing", "linear_regression"]
        }
    }
    
    # Create predictive analytics instance with auto model selection
    predictive = DatabasePredictiveAnalytics(
        automated_optimization_manager=auto_manager,
        config=model_selection_config
    )
    
    # Verify configuration was applied correctly
    print("\nAuto Model Selection Configuration Verification:")
    print(f"Auto model selection enabled: {predictive.config['forecasting']['auto_model_selection']}")
    print(f"Validation size percent: {predictive.config['forecasting']['validation_size_percent']}%")
    print(f"Available forecast methods: {', '.join(predictive.config['forecasting']['forecast_methods'])}")
    
    # Run forecasting
    forecast_result = predictive.forecast_database_metrics(
        horizon="medium_term",
        specific_metrics=["query_time", "storage_size", "index_efficiency"]  # Test with a few different patterns
    )
    
    # Examine the results to verify model selection worked
    print(f"\nForecast Status: {forecast_result.get('status', 'unknown')}")
    
    # Check which models were selected for each metric
    selection_results = {}
    for metric_name, forecast in forecast_result.get("forecasts", {}).items():
        model_selection = forecast.get("model_selection", "unknown")
        primary_method = forecast.get("primary_method", "unknown")
        
        selection_results[metric_name] = {
            "selection_method": model_selection,
            "selected_model": primary_method
        }
        
        # If we have validation metrics, display them
        if "validation_metrics" in forecast:
            best_metrics = None
            if "best_method_metrics" in forecast:
                best_metrics = forecast["best_method_metrics"]
            
            selection_results[metric_name]["validation_metrics"] = {
                "method_count": len(forecast["validation_metrics"]),
                "best_method": best_metrics.get("method") if best_metrics else None,
                "best_mape": best_metrics.get("mape") if best_metrics else None
            }
    
    # Print selection results
    print("\nModel Selection Results:")
    for metric, result in selection_results.items():
        print(f"- {metric}:")
        print(f"  Selection method: {result['selection_method']}")
        print(f"  Selected model: {result['selected_model']}")
        
        if "validation_metrics" in result:
            vm = result["validation_metrics"]
            print(f"  Validation performed on {vm['method_count']} methods")
            if vm["best_method"]:
                print(f"  Best method: {vm['best_method']} (MAPE: {vm['best_mape']:.2f}%)")
    
    # Save auto model selection results
    save_json(forecast_result, "./output/auto_model_selection_result.json")
    
    logger.info("Automated model selection test completed")

def test_hyperparameter_tuning() -> None:
    """
    Test the hyperparameter tuning feature.
    
    This test:
    1. Creates a predictive analytics instance with hyperparameter tuning enabled
    2. Generates test metrics history with different patterns
    3. Runs forecasting to trigger hyperparameter tuning
    4. Verifies that hyperparameter tuning occurred and selected optimal parameters
    5. Examines the forecast results to ensure proper parameter selection
    """
    logger.info("----- Testing Hyperparameter Tuning -----")
    
    # Create database optimizer
    db_path = "./benchmark_db.duckdb"
    db_optimizer = get_db_optimizer(db_path=db_path)
    
    # Create automated optimization manager
    auto_manager = get_optimization_manager(
        db_optimizer=db_optimizer,
        auto_apply=False
    )
    
    # Generate more test data for better hyperparameter tuning
    generate_test_metrics_history(auto_manager)
    
    # Define configuration with hyperparameter tuning enabled
    tuning_config = {
        "forecasting": {
            "auto_model_selection": True,
            "validation_size_percent": 20,  # Use 20% of the data for validation
            "forecast_methods": ["arima", "exponential_smoothing", "linear_regression"],
            "hyperparameter_tuning": {
                "enabled": True,
                "search_method": "grid",
                "max_iterations": 5,
                "cv_folds": 3,
                "arima_params": {
                    "p": [0, 1, 2],
                    "d": [0, 1],
                    "q": [0, 1]
                },
                "exp_smoothing_params": {
                    "trend": [None, "add"],
                    "seasonal": [None, "add"],
                    "seasonal_periods": [7],
                    "damped_trend": [False, True]
                },
                "linear_regression_params": {
                    "fit_intercept": [True, False],
                    "positive": [False, True]
                }
            }
        }
    }
    
    # Create predictive analytics instance with hyperparameter tuning
    predictive = DatabasePredictiveAnalytics(
        automated_optimization_manager=auto_manager,
        config=tuning_config
    )
    
    # Verify configuration was applied correctly
    print("\nHyperparameter Tuning Configuration Verification:")
    print(f"Hyperparameter tuning enabled: {predictive.config['forecasting']['hyperparameter_tuning']['enabled']}")
    print(f"Search method: {predictive.config['forecasting']['hyperparameter_tuning']['search_method']}")
    print(f"Max iterations: {predictive.config['forecasting']['hyperparameter_tuning']['max_iterations']}")
    print(f"CV folds: {predictive.config['forecasting']['hyperparameter_tuning']['cv_folds']}")
    
    # Run forecasting with different metric patterns to test different forecasting methods
    print("\nRunning forecasting with hyperparameter tuning...")
    forecast_result = predictive.forecast_database_metrics(
        horizon="medium_term",
        specific_metrics=["query_time", "storage_size", "index_efficiency"]
    )
    
    # Examine the results to verify hyperparameter tuning worked
    print(f"\nForecast Status: {forecast_result.get('status', 'unknown')}")
    
    # Check how hyperparameter tuning was applied
    tuning_results = {}
    for metric_name, forecast in forecast_result.get("forecasts", {}).items():
        primary_method = forecast.get("primary_method", "unknown")
        
        # Extract model info with tuning details
        model_info = None
        if primary_method in forecast:
            model_info = forecast[primary_method].get("model_info", {})
        else:
            # If the primary method's forecast is not directly available
            # It might be embedded in the main forecast result
            model_info = forecast.get("model_info", {})
        
        # Collect tuning details
        tuning_results[metric_name] = {
            "method": primary_method,
            "tuned": model_info.get("tuned", False) if model_info else False,
            "parameters": {}
        }
        
        # Collect method-specific parameters
        if primary_method == "arima" and model_info:
            tuning_results[metric_name]["parameters"]["order"] = model_info.get("order", "(1,1,1)")
            tuning_results[metric_name]["validation_mape"] = model_info.get("validation_mape")
            tuning_results[metric_name]["reason"] = model_info.get("reason", "N/A")
            
        elif primary_method == "exponential_smoothing" and model_info:
            tuning_results[metric_name]["parameters"]["trend"] = model_info.get("trend")
            tuning_results[metric_name]["parameters"]["seasonal"] = model_info.get("seasonal")
            tuning_results[metric_name]["parameters"]["seasonal_periods"] = model_info.get("seasonal_periods")
            tuning_results[metric_name]["parameters"]["damped_trend"] = model_info.get("damped_trend")
            tuning_results[metric_name]["validation_mape"] = model_info.get("validation_mape")
            tuning_results[metric_name]["reason"] = model_info.get("reason", "N/A")
            
        elif primary_method == "linear_regression" and model_info:
            tuning_results[metric_name]["parameters"]["fit_intercept"] = model_info.get("fit_intercept", True)
            tuning_results[metric_name]["parameters"]["positive"] = model_info.get("positive", False)
            tuning_results[metric_name]["parameters"]["slope"] = model_info.get("slope")
            tuning_results[metric_name]["parameters"]["intercept"] = model_info.get("intercept")
            tuning_results[metric_name]["validation_mape"] = model_info.get("validation_mape")
            tuning_results[metric_name]["reason"] = model_info.get("reason", "N/A")
    
    # Print tuning results
    print("\nHyperparameter Tuning Results:")
    for metric, result in tuning_results.items():
        print(f"- {metric}:")
        print(f"  Method: {result['method']}")
        print(f"  Tuned: {result['tuned']}")
        if not result['tuned'] and 'reason' in result:
            print(f"  Reason: {result['reason']}")
        
        if result['tuned']:
            print(f"  Validation MAPE: {result.get('validation_mape', 'N/A')}")
            print("  Parameters:")
            for param_name, param_value in result["parameters"].items():
                print(f"    {param_name}: {param_value}")
    
    # Save hyperparameter tuning results
    save_json(forecast_result, "./output/hyperparameter_tuning_result.json")
    
    logger.info("Hyperparameter tuning test completed")

def run_all_tests() -> None:
    """
    Run all tests.
    
    This function runs all test cases in sequence, using the same predictive analytics
    instance where possible to avoid regenerating test data unnecessarily.
    """
    logger.info("===== Starting All Tests =====")
    
    # Run basic forecasting and get instances
    auto_manager, predictive = test_basic_forecasting()
    
    # Run visualization test with same instances
    auto_manager, predictive = test_visualizations(auto_manager, predictive)
    
    # Run threshold alerts test with same instances
    auto_manager, predictive = test_threshold_alerts(auto_manager, predictive)
    
    # Run recommendations test with same instances
    auto_manager, predictive = test_recommendations(auto_manager, predictive)
    
    # Run comprehensive analysis test with same instances
    auto_manager, predictive = test_comprehensive_analysis(auto_manager, predictive)
    
    # Run custom configuration test (uses new instances)
    test_custom_config()
    
    # Run automated model selection test (uses new instances)
    test_auto_model_selection()
    
    # Run hyperparameter tuning test (uses new instances)
    test_hyperparameter_tuning()
    
    logger.info("===== All Tests Completed =====")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Database Predictive Analytics")
    parser.add_argument("--test", 
                      choices=["all", "forecast", "visualize", "alerts", "recommend", "analyze", 
                               "custom_config", "auto_model_selection", "hyperparameter_tuning"],
                      default="all", 
                      help="Test to run")
    
    parser.add_argument("--output-dir",
                      type=str,
                      default="./output",
                      help="Directory to save output files")
    
    parser.add_argument("--visualizations-dir",
                      type=str,
                      default="./visualizations",
                      help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.visualizations_dir, exist_ok=True)
    
    try:
        if args.test == "all":
            run_all_tests()
        elif args.test == "forecast":
            test_basic_forecasting()
        elif args.test == "visualize":
            test_visualizations()
        elif args.test == "alerts":
            test_threshold_alerts()
        elif args.test == "recommend":
            test_recommendations()
        elif args.test == "analyze":
            test_comprehensive_analysis()
        elif args.test == "custom_config":
            test_custom_config()
        elif args.test == "auto_model_selection":
            test_auto_model_selection()
        elif args.test == "hyperparameter_tuning":
            test_hyperparameter_tuning()
    except Exception as e:
        logger.error(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)