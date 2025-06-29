#!/usr/bin/env python3
"""
Launcher for the API Predictive Analytics Management UI

This script launches the API Management UI, optionally connecting it
to an existing API monitoring dashboard and loading sample data if needed.
It also supports integration with FastAPI for programmatic access.
"""

import os
import sys
import json
import logging
import argparse
import datetime
import subprocess
from typing import Dict, Any, Optional, List, Tuple, Union
import threading

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import UI and supporting modules
try:
    from api_management_ui import PredictiveAnalyticsUI
    from api_monitoring_dashboard import APIMonitoringDashboard
except ImportError:
    from test.api_management_ui import PredictiveAnalyticsUI
    from test.api_monitoring_dashboard import APIMonitoringDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_data() -> Dict[str, Any]:
    """
    Generate sample data for demo purposes when no real data is available.
    
    Returns:
        Dict containing historical_data, predictions, anomalies, recommendations,
        and comparative_data for API providers.
    """
    # Base timestamp for sample data
    now = datetime.datetime.now()
    start_date = now - datetime.timedelta(days=30)
    
    # APIs to include
    apis = ["OpenAI", "Anthropic", "Cohere", "Groq", "Mistral"]
    
    # Metrics to include
    metrics = ["latency", "cost", "throughput", "success_rate", "tokens_per_second"]
    
    # Generate sample data
    data = {
        "historical_data": {},
        "predictions": {},
        "anomalies": {},
        "recommendations": {},
        "comparative_data": {}
    }
    
    # Generate historical data for each metric and API
    import numpy as np
    
    for metric in metrics:
        data["historical_data"][metric] = {}
        data["predictions"][metric] = {}
        data["anomalies"][metric] = {}
        data["comparative_data"][metric] = []
        
        for api in apis:
            # Historical data with different patterns per API and metric
            timestamps = [start_date + datetime.timedelta(hours=i) for i in range(24*30)]
            
            # Base value depends on metric
            if metric == "latency":
                base = 200  # ms
                noise_scale = 50
                trend = 0.1  # Slight upward trend
            elif metric == "cost":
                base = 0.02  # $ per request
                noise_scale = 0.005
                trend = 0.05  # Slight upward trend
            elif metric == "throughput":
                base = 100  # requests per minute
                noise_scale = 20
                trend = -0.05  # Slight downward trend
            elif metric == "success_rate":
                base = 98  # percentage
                noise_scale = 1
                trend = -0.02  # Slight downward trend
            else:  # tokens_per_second
                base = 1000
                noise_scale = 200
                trend = 0
            
            # Add API-specific modifier
            if api == "OpenAI":
                api_mod = 1.0
                weekly_pattern = True
                spikes = True
            elif api == "Anthropic":
                api_mod = 1.2
                weekly_pattern = True
                spikes = False
            elif api == "Cohere":
                api_mod = 0.8
                weekly_pattern = False
                spikes = True
            elif api == "Groq":
                api_mod = 0.9
                weekly_pattern = True
                spikes = True
            else:  # Mistral
                api_mod = 1.1
                weekly_pattern = False
                spikes = False
            
            # Generate values with trends, weekly patterns, and random noise
            values = []
            for i, ts in enumerate(timestamps):
                # Base value with trend
                value = base * api_mod * (1 + trend * i/len(timestamps))
                
                # Add weekly pattern if applicable
                if weekly_pattern:
                    day_of_week = ts.weekday()
                    if day_of_week >= 5:  # Weekend
                        value *= 0.7
                    elif day_of_week == 0:  # Monday
                        value *= 1.1
                
                # Add time of day pattern
                hour = ts.hour
                if 9 <= hour <= 17:  # Business hours
                    value *= 1.2
                elif 0 <= hour <= 5:  # Early morning
                    value *= 0.6
                
                # Add random noise
                value += np.random.normal(0, noise_scale * 0.1)
                
                # Add occasional spikes
                if spikes and np.random.random() < 0.01:  # 1% chance of spike
                    value *= 1.5 + np.random.random()
                
                # Ensure reasonable values (no negatives for most metrics)
                if metric != "success_rate":
                    value = max(value, 0)
                else:
                    value = min(max(value, 0), 100)  # Success rate between 0-100%
                
                values.append(value)
            
            # Store historical data
            data["historical_data"][metric][api] = [
                {"timestamp": ts.isoformat(), "value": val}
                for ts, val in zip(timestamps, values)
            ]
            
            # Generate predictions (next 14 days)
            future_timestamps = [now + datetime.timedelta(hours=i) for i in range(24*14)]
            future_values = []
            
            for i, ts in enumerate(future_timestamps):
                # Base prediction with trend
                value = values[-1] * (1 + trend * i/len(future_timestamps))
                
                # Add patterns
                if weekly_pattern:
                    day_of_week = ts.weekday()
                    if day_of_week >= 5:  # Weekend
                        value *= 0.7
                    elif day_of_week == 0:  # Monday
                        value *= 1.1
                
                # Add time of day pattern
                hour = ts.hour
                if 9 <= hour <= 17:  # Business hours
                    value *= 1.2
                elif 0 <= hour <= 5:  # Early morning
                    value *= 0.6
                
                # Add random noise (less than historical for predictions)
                value += np.random.normal(0, noise_scale * 0.05)
                
                # Calculate confidence interval
                std_dev = noise_scale * 0.15
                lower_bound = value - 1.96 * std_dev
                upper_bound = value + 1.96 * std_dev
                
                future_values.append({
                    "timestamp": ts.isoformat(),
                    "value": value,
                    "lower_bound": max(0, lower_bound) if metric != "success_rate" else min(max(lower_bound, 0), 100),
                    "upper_bound": upper_bound if metric != "success_rate" else min(upper_bound, 100)
                })
            
            # Store predictions
            data["predictions"][metric][api] = future_values
            
            # Generate anomalies (a few per API/metric)
            anomalies = []
            
            # Sample timestamps for anomalies
            anomaly_indices = np.random.choice(len(timestamps), size=min(5, len(timestamps)//10), replace=False)
            
            for idx in anomaly_indices:
                ts = timestamps[idx]
                val = values[idx]
                
                # Determine anomaly type
                anomaly_types = ["spike", "trend_break", "oscillation", "seasonal"]
                anomaly_type = np.random.choice(anomaly_types)
                
                # Generate confidence between 0.7 and 0.99
                confidence = 0.7 + np.random.random() * 0.29
                
                # Determine severity
                severity_levels = ["low", "medium", "high", "critical"]
                severity_weights = [0.4, 0.3, 0.2, 0.1]  # Probability distribution
                severity = np.random.choice(severity_levels, p=severity_weights)
                
                # Description based on type
                if anomaly_type == "spike":
                    description = f"Unexpected {metric} spike detected"
                elif anomaly_type == "trend_break":
                    description = f"Breaking trend in {metric} values"
                elif anomaly_type == "oscillation":
                    description = f"Unusual oscillation pattern in {metric}"
                else:  # seasonal
                    description = f"Deviation from expected seasonal {metric} pattern"
                
                anomalies.append({
                    "timestamp": ts.isoformat(),
                    "value": val,
                    "type": anomaly_type,
                    "confidence": confidence,
                    "description": description,
                    "severity": severity
                })
            
            # Store anomalies
            data["anomalies"][metric][api] = anomalies
            
            # Generate recommendations
            recommendations = []
            
            # Generic recommendations per API
            if api == "OpenAI":
                recommendations = [
                    {
                        "title": "Optimize Batch Size",
                        "description": "Increase batch size to reduce the number of API calls.",
                        "impact": 0.15,
                        "effort": "Low",
                        "implementation_time": "Days",
                        "roi_period": "Weeks",
                        "status": "New"
                    },
                    {
                        "title": "Implement Response Caching",
                        "description": "Cache frequent identical requests to reduce costs.",
                        "impact": 0.22,
                        "effort": "Medium",
                        "implementation_time": "Weeks",
                        "roi_period": "Months",
                        "status": "In Progress"
                    },
                    {
                        "title": "Use Cheaper Models When Possible",
                        "description": "Use less expensive models for simple tasks.",
                        "impact": 0.30,
                        "effort": "Medium",
                        "implementation_time": "Days",
                        "roi_period": "Weeks",
                        "status": "New"
                    }
                ]
            elif api == "Anthropic":
                recommendations = [
                    {
                        "title": "Prompt Optimization",
                        "description": "Optimize prompts to reduce token usage.",
                        "impact": 0.18,
                        "effort": "Medium",
                        "implementation_time": "Days",
                        "roi_period": "Weeks",
                        "status": "New"
                    },
                    {
                        "title": "Implement Retry Strategy",
                        "description": "Add exponential backoff to reduce failed requests.",
                        "impact": 0.12,
                        "effort": "Low",
                        "implementation_time": "Hours",
                        "roi_period": "Days",
                        "status": "Implemented"
                    }
                ]
            elif api == "Cohere":
                recommendations = [
                    {
                        "title": "Switch to Embedding Caching",
                        "description": "Cache embeddings for frequently accessed content.",
                        "impact": 0.25,
                        "effort": "Medium",
                        "implementation_time": "Weeks",
                        "roi_period": "Months",
                        "status": "New"
                    },
                    {
                        "title": "Adjust Rate Limiting",
                        "description": "Adjust client-side rate limits to match API quotas.",
                        "impact": 0.10,
                        "effort": "Low",
                        "implementation_time": "Hours",
                        "roi_period": "Days",
                        "status": "Verified"
                    }
                ]
            else:  # Generic for others
                recommendations = [
                    {
                        "title": "Optimize Request Patterns",
                        "description": "Analyze and optimize API request patterns.",
                        "impact": 0.20,
                        "effort": "Medium",
                        "implementation_time": "Weeks",
                        "roi_period": "Months",
                        "status": "New"
                    },
                    {
                        "title": "Implement Error Rate Monitoring",
                        "description": "Add monitoring for error rates to improve reliability.",
                        "impact": 0.15,
                        "effort": "Low",
                        "implementation_time": "Days",
                        "roi_period": "Weeks",
                        "status": "In Progress"
                    }
                ]
            
            # Store recommendations
            data["recommendations"][api] = recommendations
        
        # Generate comparative data for this metric
        for i, ts in enumerate(timestamps[::24]):  # Daily samples for comparative data
            comparative_values = {}
            for api in apis:
                # Get the corresponding daily value
                if i < len(data["historical_data"][metric][api]):
                    comparative_values[api] = data["historical_data"][metric][api][i*24]["value"]
            
            data["comparative_data"][metric].append({
                "timestamp": ts.isoformat(),
                "values": comparative_values
            })
    
    return data


def save_sample_data(file_path: str) -> None:
    """
    Generate and save sample data to a file.
    
    Args:
        file_path: Path to save the sample data
    """
    data = generate_sample_data()
    
    try:
        os.makedirs(os.path.dirname(os.path.path), exist_ok=True) if os.path.dirname(file_path) else None
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Sample data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving sample data: {e}")


def start_fastapi_server(port: int, data_path: Optional[str] = None, 
                        connect_dashboard: bool = False,
                        generate_sample: bool = False,
                        sample_path: str = "./sample_api_data.json",
                        debug: bool = False) -> subprocess.Popen:
    """
    Start the FastAPI server for API Management UI.
    
    Args:
        port: Port for FastAPI server
        data_path: Path to data file
        connect_dashboard: Whether to connect to a monitoring dashboard
        generate_sample: Whether to generate sample data
        sample_path: Path for sample data
        debug: Whether to enable debug mode
        
    Returns:
        Subprocess object for the FastAPI server
    """
    # Build command
    cmd = [sys.executable, "api_management_ui_server.py", "--port", str(port)]
    
    if data_path:
        cmd.extend(["--data", data_path])
    
    if connect_dashboard:
        cmd.append("--connect-dashboard")
    
    if generate_sample:
        cmd.append("--generate-sample")
        cmd.extend(["--sample-path", sample_path])
    
    if debug:
        cmd.append("--debug")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Start the server as a subprocess
    logger.info(f"Starting FastAPI server on port {port} with command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, cwd=current_dir)
    
    return process


def main():
    """Main function to parse arguments and start the UI."""
    parser = argparse.ArgumentParser(description='API Predictive Analytics Management UI')
    parser.add_argument('-p', '--port', type=int, default=8050, 
                       help='Port to run the Dash UI server on (default: 8050)')
    parser.add_argument('-d', '--data', type=str, help='Path to JSON data file')
    parser.add_argument('--generate-sample', action='store_true', help='Generate sample data')
    parser.add_argument('--sample-path', type=str, default='./sample_api_data.json', 
                       help='Path to save/load sample data')
    parser.add_argument('--connect-dashboard', action='store_true', help='Connect to live monitoring dashboard')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--theme', type=str, default='cosmo', 
                       help='UI theme (cosmo, darkly, flatly, etc.)')
    parser.add_argument('--db-path', type=str, help='Path to DuckDB database')
    parser.add_argument('--db-generate-sample', action='store_true',
                       help='Generate sample data in the DuckDB database')
    parser.add_argument('--fastapi-integration', action='store_true', 
                       help='Enable FastAPI integration for external access')
    parser.add_argument('--fastapi-port', type=int, default=8000, 
                       help='Port for FastAPI server if enabled (default: 8000)')
    parser.add_argument('--export-format', type=str, choices=['html', 'png', 'svg', 'pdf'], 
                       default='html', help='Default export format')
    parser.add_argument('--enable-caching', action='store_true', 
                       help='Enable data caching for improved performance')
    args = parser.parse_args()
    
    # Handle sample data generation
    if args.generate_sample:
        save_sample_data(args.sample_path)
        logger.info(f"Sample data generated and saved to {args.sample_path}")
        if not args.data:
            args.data = args.sample_path
    
    # Connect to dashboard if requested
    dashboard = None
    if args.connect_dashboard:
        try:
            logger.info("Connecting to API monitoring dashboard...")
            dashboard = APIMonitoringDashboard(enable_predictive_analytics=True)
            logger.info("Connected to dashboard successfully")
        except Exception as e:
            logger.error(f"Error connecting to dashboard: {e}")
            logger.info("Falling back to file-based data")
    
    # Start FastAPI server if requested
    fastapi_process = None
    if args.fastapi_integration:
        fastapi_process = start_fastapi_server(
            port=args.fastapi_port,
            data_path=args.data,
            connect_dashboard=args.connect_dashboard,
            generate_sample=args.generate_sample,
            sample_path=args.sample_path,
            debug=args.debug
        )
        logger.info(f"FastAPI server started on port {args.fastapi_port}")
        logger.info(f"API documentation available at http://localhost:{args.fastapi_port}/docs")
    
    # Handle DuckDB integration
    db_repository = None
    if args.db_path:
        try:
            # Import DuckDBAPIMetricsRepository
            try:
                from duckdb_api.api_management import DuckDBAPIMetricsRepository
            except ImportError:
                from test.duckdb_api.api_management import DuckDBAPIMetricsRepository
            
            # Create repository
            db_repository = DuckDBAPIMetricsRepository(
                db_path=args.db_path,
                create_if_missing=True
            )
            
            # Generate sample data if requested
            if args.db_generate_sample:
                logger.info(f"Generating sample data in DuckDB database at {args.db_path}...")
                
                # Import necessary modules for sample generation
                import numpy as np
                from datetime import datetime, timedelta
                
                # Call the sample data generation method
                import subprocess
                cmd = [sys.executable, "duckdb_api/api_management/duckdb_api_metrics.py", 
                       "--db-path", args.db_path, "--generate-sample"]
                
                try:
                    subprocess.run(cmd, check=True)
                    logger.info("Sample data generation completed successfully")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error generating sample data: {e}")
                    
                    # Fallback to direct sample generation if available
                    if hasattr(db_repository, 'generate_sample_data'):
                        logger.info("Attempting direct sample data generation...")
                        db_repository.generate_sample_data()
                        logger.info("Direct sample data generation completed")
            
            logger.info(f"Connected to DuckDB database at {args.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to DuckDB database: {e}")
            if args.db_generate_sample:
                logger.warning("Sample data generation skipped due to database connection error")
    
    # Initialize UI
    ui = PredictiveAnalyticsUI(
        monitoring_dashboard=dashboard,
        data_path=args.data,
        theme=args.theme,
        debug=args.debug,
        enable_caching=args.enable_caching,
        db_path=args.db_path,
        db_repository=db_repository
    )
    
    # Run server
    logger.info(f"Starting API Predictive Analytics UI on port {args.port}...")
    print(f"Starting API Predictive Analytics UI on port {args.port}...")
    print(f"Open your browser and navigate to http://localhost:{args.port} to view the dashboard.")
    
    if args.fastapi_integration:
        print(f"FastAPI integration enabled on port {args.fastapi_port}")
        print(f"API documentation available at http://localhost:{args.fastapi_port}/docs")
    
    # Run the UI server
    try:
        ui.run_server(port=args.port, debug=args.debug)
    finally:
        # Clean up FastAPI process if it was started
        if fastapi_process:
            logger.info("Stopping FastAPI server...")
            fastapi_process.terminate()
            fastapi_process.wait()
            logger.info("FastAPI server stopped")


if __name__ == "__main__":
    main()