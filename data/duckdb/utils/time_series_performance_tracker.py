#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Series Performance Tracker for IPFS Accelerate Python Framework

This module implements the time-series performance tracking system mentioned in NEXT_STEPS.md.
It provides components for versioned test results, regression detection, trend visualization,
and notification systems.

Date: March 2025
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

# Add parent directory to path
sys.path.append()))))))str()))))))Path()))))))__file__).resolve()))))))).parent.parent))

# Local imports
try:
    from data.duckdb.core.benchmark_db_api import get_db_connection
    from data.duckdb.core.benchmark_db_query import query_database
    from benchmark_regression_analyzer import analyze_regressions
except ImportError:
    print()))))))"Warning: Some local modules could not be imported.")


class TimeSeriesSchema:
    """Schema extension for versioned test results in DuckDB database."""
    
    def __init__()))))))self, db_path: Optional[str] = None):,,,
    """Initialize with optional database path."""
    self.db_path = db_path or os.environ.get()))))))'BENCHMARK_DB_PATH', './benchmark_db.duckdb')
    
    def create_schema_extensions()))))))self) -> None:
        """Create schema extensions for versioned test results."""
        conn = get_db_connection()))))))self.db_path)
        
        # Create version history table if it doesn't exist
        conn.execute()))))))"""
        CREATE TABLE IF NOT EXISTS version_history ()))))))
        id INTEGER PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        version_tag VARCHAR,
        description VARCHAR,
        user_id VARCHAR,
        commit_hash VARCHAR,
        git_branch VARCHAR
        )
        """)
        
        # Create performance time series table if it doesn't exist
        conn.execute()))))))"""
        CREATE TABLE IF NOT EXISTS performance_time_series ()))))))
        id INTEGER PRIMARY KEY,
        version_id INTEGER,
        model_id INTEGER,
        hardware_id INTEGER,
        batch_size INTEGER,
        test_type VARCHAR,
        throughput FLOAT,
        latency FLOAT,
        memory_usage FLOAT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY ()))))))version_id) REFERENCES version_history()))))))id),
        FOREIGN KEY ()))))))model_id) REFERENCES models()))))))id),
        FOREIGN KEY ()))))))hardware_id) REFERENCES hardware_platforms()))))))id)
        )
        """)
        
        # Create regression alerts table if it doesn't exist
        conn.execute()))))))"""
        CREATE TABLE IF NOT EXISTS regression_alerts ()))))))
        id INTEGER PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        model_id INTEGER,
        hardware_id INTEGER,
        metric VARCHAR,
        previous_value FLOAT,
        current_value FLOAT,
        percent_change FLOAT,
        severity VARCHAR,
        status VARCHAR DEFAULT 'active',
        notification_sent BOOLEAN DEFAULT FALSE,
        resolved_at TIMESTAMP,
        FOREIGN KEY ()))))))model_id) REFERENCES models()))))))id),
        FOREIGN KEY ()))))))hardware_id) REFERENCES hardware_platforms()))))))id)
        )
        """)
        
        conn.close())))))))
        print()))))))"Schema extensions created successfully!")

:
class RegressionDetector:
    """Detects performance regressions in time series data."""
    
    def __init__()))))))self, db_path: Optional[str] = None, threshold: float = 0.10):,
    """Initialize with database path and regression threshold."""
    self.db_path = db_path or os.environ.get()))))))'BENCHMARK_DB_PATH', './benchmark_db.duckdb')
    self.threshold = threshold
        
    def detect_regressions()))))))self, model_id: Optional[int] = None, 
    hardware_id: Optional[int] = None,
    metric: str = 'throughput',
    window_size: int = 5) -> List[Dict[str, Any]]:,,
    """
    Detect performance regressions for specified model/hardware.
        
        Args:
            model_id: Filter by model ID
            hardware_id: Filter by hardware ID
            metric: Performance metric to analyze ()))))))throughput, latency, memory_usage)
            window_size: Number of previous data points to use for baseline
            
        Returns:
            List of regression events
            """
            conn = get_db_connection()))))))self.db_path)
        
        # Build query filters
            filters = [],,
            params = {}}}}}
        
        if model_id is not None:
            filters.append()))))))"model_id = :model_id")
            params['model_id'] = model_id
            ,
        if hardware_id is not None:
            filters.append()))))))"hardware_id = :hardware_id")
            params['hardware_id'] = hardware_id
            ,
            where_clause = f"WHERE {}}}}' AND '.join()))))))filters)}" if filters else ""
        
        # Query time series data
            query = f"""
            SELECT
            ph.id,
            m.name as model_name,
            hp.type as hardware_type,
            ph.batch_size,
            ph.{}}}}metric},
            ph.timestamp
            FROM
            performance_time_series ph
            JOIN
            models m ON ph.model_id = m.id
            JOIN
            hardware_platforms hp ON ph.hardware_id = hp.id
            {}}}}where_clause}
            ORDER BY
            m.name, hp.type, ph.batch_size, ph.timestamp
            """
        
            result = conn.execute()))))))query, params).fetchall())))))))
            conn.close())))))))
        :
        if not result:
            return [],,
        
        # Convert to pandas DataFrame for analysis
            df = pd.DataFrame()))))))result, columns=['id', 'model_name', 'hardware_type',
            'batch_size', metric, 'timestamp'])
        
        # Group by model, hardware, and batch size
            groups = df.groupby()))))))['model_name', 'hardware_type', 'batch_size'])
            ,
            regressions = [],,
        
        for name, group in groups:
            model_name, hardware_type, batch_size = name
            
            # Sort by timestamp
            group = group.sort_values()))))))'timestamp')
            
            # Need at least window_size+1 points to detect regression
            if len()))))))group) <= window_size:
            continue
                
            # Calculate rolling window statistics
            values = group[metric].values,
            timestamps = group['timestamp'],.values
            ,
            for i in range()))))))window_size, len()))))))values)):
                window = values[i-window_size:i],
                current = values[i]
                ,
                # Calculate baseline ()))))))mean of window)
                baseline = np.mean()))))))window)
                
                # For latency, lower is better, so check for increase
                if metric == 'latency':
                    percent_change = ()))))))current - baseline) / baseline
                    is_regression = percent_change > self.threshold
                # For other metrics, higher is better, so check for decrease
                else:
                    percent_change = ()))))))baseline - current) / baseline
                    is_regression = percent_change > self.threshold
                
                if is_regression:
                    # Determine severity
                    if percent_change > 0.25:
                        severity = 'critical'
                    elif percent_change > 0.15:
                        severity = 'high'
                    else:
                        severity = 'medium'
                        
                    # Add to regressions list
                        regressions.append())))))){}}}}
                        'model_name': model_name,
                        'hardware_type': hardware_type,
                        'batch_size': batch_size,
                        'metric': metric,
                        'previous_value': baseline,
                        'current_value': current,
                        'percent_change': percent_change * 100,
                        'timestamp': timestamps[i],
                        'severity': severity
                        })
        
                        return regressions
    
                        def record_regressions()))))))self, regressions: List[Dict[str, Any]]) -> None:,
                        """
                        Record detected regressions in the database.
        
        Args:
            regressions: List of regression events
            """
        if not regressions:
            return
            
            conn = get_db_connection()))))))self.db_path)
        
        for reg in regressions:
            # Get model_id and hardware_id
            model_query = "SELECT id FROM models WHERE name = ?"
            model_id = conn.execute()))))))model_query, ()))))))reg['model_name'],)).fetchone())))))))[0]
            ,
            hardware_query = "SELECT id FROM hardware_platforms WHERE type = ?"
            hardware_id = conn.execute()))))))hardware_query, ()))))))reg['hardware_type'],)).fetchone())))))))[0]
            ,
            # Insert regression alert
            insert_query = """
            INSERT INTO regression_alerts 
            ()))))))model_id, hardware_id, metric, previous_value, current_value,
            percent_change, severity, timestamp)
            VALUES ()))))))?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            conn.execute()))))))insert_query, ()))))))
            model_id,
            hardware_id,
            reg['metric'],,
            reg['previous_value'],
            reg['current_value'],
            reg['percent_change'],
            reg['severity'],
            reg['timestamp'],
            ))
        
            conn.commit())))))))
            conn.close())))))))
        
            print()))))))f"Recorded {}}}}len()))))))regressions)} regression alerts in the database.")


class TrendVisualizer:
    """Creates visualizations for performance trends over time."""
    
    def __init__()))))))self, db_path: Optional[str] = None):,,,
    """Initialize with database path."""
    self.db_path = db_path or os.environ.get()))))))'BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        
    def visualize_metric_trend()))))))self, 
    model_name: str,
    hardware_type: str,
    metric: str = 'throughput',
    batch_size: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_file: Optional[str] = None) -> str:,
    """
    Create visualization for a metric trend over time.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware
            metric: Performance metric to visualize
            batch_size: Optional filter for batch size
            start_date: Optional start date for filtering ()))))))format: YYYY-MM-DD)
            end_date: Optional end date for filtering ()))))))format: YYYY-MM-DD)
            output_file: Path to save the visualization
            
        Returns:
            Path to the saved visualization
            """
            conn = get_db_connection()))))))self.db_path)
        
        # Build query filters
            filters = ["m.name = :model_name", "hp.type = :hardware_type"],
            params = {}}}}
            'model_name': model_name,
            'hardware_type': hardware_type
            }
        
        if batch_size is not None:
            filters.append()))))))"pts.batch_size = :batch_size")
            params['batch_size'] = batch_size
            ,
        if start_date:
            filters.append()))))))"pts.timestamp >= :start_date")
            params['start_date'] = start_date
            ,
        if end_date:
            filters.append()))))))"pts.timestamp <= :end_date")
            params['end_date'] = end_date
            ,
            where_clause = f"WHERE {}}}}' AND '.join()))))))filters)}"
        
        # Query time series data
            query = f"""
            SELECT
            pts.batch_size,
            pts.{}}}}metric},
            pts.timestamp,
            vh.version_tag
            FROM
            performance_time_series pts
            JOIN
            models m ON pts.model_id = m.id
            JOIN
            hardware_platforms hp ON pts.hardware_id = hp.id
            LEFT JOIN
            version_history vh ON pts.version_id = vh.id
            {}}}}where_clause}
            ORDER BY
            pts.timestamp
            """
        
            result = conn.execute()))))))query, params).fetchall())))))))
            conn.close())))))))
        
        if not result:
            print()))))))f"No data found for {}}}}model_name} on {}}}}hardware_type}")
            return ""
        
        # Convert to pandas DataFrame
            df = pd.DataFrame()))))))result, columns=['batch_size', metric, 'timestamp', 'version_tag'])
            ,
        # Convert timestamp to datetime
            df['timestamp'], = pd.to_datetime()))))))df['timestamp'],)
        
        # Set up plot
            plt.figure()))))))figsize=()))))))12, 6))
            sns.set_style()))))))'whitegrid')
        
        # If batch_size is not specified, group by batch_size
        if batch_size is None:
            for bs, group in df.groupby()))))))'batch_size'):
                plt.plot()))))))group['timestamp'],, group[metric], marker='o', linestyle='-', 
                label=f'Batch Size {}}}}bs}')
        else:
            plt.plot()))))))df['timestamp'],, df[metric], marker='o', linestyle='-')
            
            # Add version tags as annotations
            for i, row in df.iterrows()))))))):
                if row['version_tag']:,
                plt.annotate()))))))row['version_tag'],
                ()))))))row['timestamp'],, row[metric]),
                textcoords="offset points",
                xytext=()))))))0, 10),
                ha='center')
        
        # Set labels and title
                plt.xlabel()))))))'Date')
                plt.ylabel()))))))f'{}}}}metric.capitalize())))))))}')
                metric_label = metric.replace()))))))'_', ' ').capitalize())))))))
                plt.title()))))))f'{}}}}metric_label} Trend for {}}}}model_name} on {}}}}hardware_type}')
        
        if batch_size is None:
            plt.legend()))))))title='Batch Size')
            
            plt.xticks()))))))rotation=45)
            plt.tight_layout())))))))
        
        # Save the plot if output file is specified:
        if output_file:
            plt.savefig()))))))output_file)
            print()))))))f"Saved visualization to {}}}}output_file}")
            return output_file
        
        # Generate a default filename
            timestamp = datetime.datetime.now()))))))).strftime()))))))"%Y%m%d_%H%M%S")
            output_dir = "benchmark_visualizations"
            os.makedirs()))))))output_dir, exist_ok=True)
        
            filename = f"{}}}}output_dir}/{}}}}model_name}_{}}}}hardware_type}_{}}}}metric}_{}}}}timestamp}.png"
            plt.savefig()))))))filename)
            plt.close())))))))
        
            print()))))))f"Saved visualization to {}}}}filename}")
                return filename
    
                def create_regression_dashboard()))))))self,
                output_file: Optional[str] = None,
                days: int = 30,
                                  limit: int = 10) -> str:
                                      """
                                      Create a dashboard of recent regressions.
        
        Args:
            output_file: Path to save the dashboard
            days: Number of days to include
            limit: Maximum number of regressions to show
            
        Returns:
            Path to the saved dashboard
            """
            conn = get_db_connection()))))))self.db_path)
        
        # Query recent regression alerts
            query = """
            SELECT
            ra.id,
            m.name as model_name,
            hp.type as hardware_type,
            ra.metric,
            ra.previous_value,
            ra.current_value,
            ra.percent_change,
            ra.severity,
            ra.timestamp
            FROM
            regression_alerts ra
            JOIN
            models m ON ra.model_id = m.id
            JOIN
            hardware_platforms hp ON ra.hardware_id = hp.id
            WHERE
            ra.timestamp >= :start_date
            ORDER BY
            ra.timestamp DESC
            LIMIT :limit
            """
        
            start_date = ()))))))datetime.datetime.now()))))))) - datetime.timedelta()))))))days=days)).strftime()))))))"%Y-%m-%d")
            params = {}}}}'start_date': start_date, 'limit': limit}
        
            result = conn.execute()))))))query, params).fetchall())))))))
            conn.close())))))))
        
        if not result:
            print()))))))f"No regression alerts found in the last {}}}}days} days")
            return ""
        
        # Convert to pandas DataFrame
            df = pd.DataFrame()))))))result, columns=['id', 'model_name', 'hardware_type', 'metric',
            'previous_value', 'current_value', 'percent_change',
            'severity', 'timestamp'])
        
        # Create dashboard
            plt.figure()))))))figsize=()))))))14, 10))
        
        # Set up grid for subplots
            grid_size = min()))))))len()))))))df), 9)  # Show at most 9 plots
            rows = ()))))))grid_size + 2) // 3  # Calculate rows needed
        
        # Summary bar chart for all regressions by severity
            plt.subplot()))))))rows + 1, 1, 1)  # Top subplot spans the width
            severity_counts = df['severity'].value_counts()))))))),
            colors = {}}}}'critical': 'red', 'high': 'orange', 'medium': 'yellow'}
            severity_counts.plot()))))))kind='bar', color=[colors.get()))))))s, 'blue') for s in severity_counts.index]):,
            plt.title()))))))'Regression Alerts by Severity')
            plt.ylabel()))))))'Count')
            plt.tight_layout())))))))
        
        # Individual regression plots
            for i, ()))))))_, row) in enumerate()))))))df.iterrows())))))))[:grid_size]):,
            plt.subplot()))))))rows + 1, 3, i + 4)  # +4 to start after the summary
            
            # Get historical data for this model/hardware
            trend_data = self.get_historical_data()))))))
            row['model_name'],
            row['hardware_type'],
            row['metric'],
            )
            
            # Plot trend
            if trend_data:
                x = range()))))))len()))))))trend_data))
                plt.plot()))))))x, trend_data, marker='o', linestyle='-')
                
                # Mark the regression point
                regression_index = len()))))))trend_data) - 1
                plt.scatter()))))))[regression_index], [trend_data[-1]], color='red', s=100, zorder=5)
                ,
                # Add a horizontal line for the previous baseline
                plt.axhline()))))))y=row['previous_value'], color='green', linestyle='--', alpha=0.7)
                ,
                plt.title()))))))f"{}}}}row['model_name']},\n{}}}}row['hardware_type']},", fontsize=9),
                plt.ylabel()))))))row['metric'],)
            
                plt.tight_layout())))))))
        
        # Save the dashboard
        if output_file:
            plt.savefig()))))))output_file)
                return output_file
            
        # Generate a default filename
                timestamp = datetime.datetime.now()))))))).strftime()))))))"%Y%m%d_%H%M%S")
                output_dir = "benchmark_visualizations"
                os.makedirs()))))))output_dir, exist_ok=True)
        
                filename = f"{}}}}output_dir}/regression_dashboard_{}}}}timestamp}.png"
                plt.savefig()))))))filename)
                plt.close())))))))
        
            return filename
    
            def get_historical_data()))))))self, model_name: str, hardware_type: str, metric: str) -> List[float]:,
            """Get historical data for a model/hardware combination."""
            conn = get_db_connection()))))))self.db_path)
        
            query = """
            SELECT
            pts.{}}}}metric}
            FROM
            performance_time_series pts
            JOIN
            models m ON pts.model_id = m.id
            JOIN
            hardware_platforms hp ON pts.hardware_id = hp.id
            WHERE
            m.name = :model_name
            AND hp.type = :hardware_type
            ORDER BY
            pts.timestamp
            LIMIT 20
            """
        
            query = query.format()))))))metric=metric)
            params = {}}}}'model_name': model_name, 'hardware_type': hardware_type}
        
            result = conn.execute()))))))query, params).fetchall())))))))
            conn.close())))))))
        
        if not result:
            return [],,
            
            return [row[0] for row in result]:,
class NotificationSystem:
    """Notification system for performance regressions."""
    
    def __init__()))))))self, db_path: Optional[str] = None):,,,
    """Initialize with database path."""
    self.db_path = db_path or os.environ.get()))))))'BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        
    def get_pending_notifications()))))))self) -> List[Dict[str, Any]]:,,
    """Get regression alerts that need notifications."""
    conn = get_db_connection()))))))self.db_path)
        
    query = """
    SELECT
    ra.id,
    m.name as model_name,
    hp.type as hardware_type,
    ra.metric,
    ra.previous_value,
    ra.current_value,
    ra.percent_change,
    ra.severity,
    ra.timestamp
    FROM
    regression_alerts ra
    JOIN
    models m ON ra.model_id = m.id
    JOIN
    hardware_platforms hp ON ra.hardware_id = hp.id
    WHERE
    ra.notification_sent = FALSE
    AND ra.status = 'active'
    ORDER BY
    ra.severity DESC,
    ra.timestamp DESC
    """
        
    result = conn.execute()))))))query).fetchall())))))))
    conn.close())))))))
        
        if not result:
    return [],,
            
        # Convert to list of dictionaries
    notifications = [],,
        for row in result:
            notifications.append())))))){}}}}
            'id': row[0],
            'model_name': row[1],
            'hardware_type': row[2],
            'metric': row[3],
            'previous_value': row[4],
            'current_value': row[5],
            'percent_change': row[6],
            'severity': row[7],
            'timestamp': row[8],
            })
            
    return notifications
    
    def mark_notification_sent()))))))self, alert_id: int) -> None:
        """Mark a regression alert as notified."""
        conn = get_db_connection()))))))self.db_path)
        
        query = """
        UPDATE regression_alerts
        SET notification_sent = TRUE
        WHERE id = ?
        """
        
        conn.execute()))))))query, ()))))))alert_id,))
        conn.commit())))))))
        conn.close())))))))
    
        def send_github_issue()))))))self, regression: Dict[str, Any]) -> str:,
        """
        Create a GitHub issue for a regression alert.
        
        Args:
            regression: Regression alert information
            
        Returns:
            Issue URL if successful, empty string otherwise
            """
        # This is a placeholder for GitHub API integration
        # In a real implementation, you would use a library like PyGithub:
            print()))))))f"Would create GitHub issue for regression: {}}}}regression['model_name']}, on {}}}}regression['hardware_type']},")
            ,
        # Generate issue title
            severity_emoji = {}}}}
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡'
            }
            emoji = severity_emoji.get()))))))regression['severity'], '⚪')
            ,
            title = f"{}}}}emoji} {}}}}regression['severity'].upper())))))))} regression: {}}}}regression['model_name']}, on {}}}}regression['hardware_type']},"
            ,
        # Generate issue body
            body = f"""
## Performance Regression Detected

            - **Model**: {}}}}regression['model_name']},
            - **Hardware**: {}}}}regression['hardware_type']},
            - **Metric**: {}}}}regression['metric'],}
- **Change**: {}}}}regression['percent_change']:.2f}% {}}}}'increase' if regression['metric'], == 'latency' else 'decrease'}:
    - **Previous**: {}}}}regression['previous_value']:.4f},
    - **Current**: {}}}}regression['current_value']:.4f},
    - **Detected**: {}}}}regression['timestamp'],}
    - **Severity**: {}}}}regression['severity']}
    ,
## Recommended Actions

    1. Verify the regression with additional tests
    2. Check recent code changes that might affect performance
    3. Investigate potential hardware or environment issues
    4. Update benchmark baselines if the change is expected

## Automatically Generated

    This issue was automatically generated by the performance regression detection system.
    """
        
        # Simulate issue creation:
            return f"https://github.com/org/repo/issues/{}}}}int()))))))time.time()))))))))}"
    
            def send_email_notification()))))))self, regression: Dict[str, Any]) -> bool:,
            """
            Send email notification about a regression.
        
        Args:
            regression: Regression alert information
            
        Returns:
            True if sent successfully, False otherwise
            """
        # This is a placeholder for email notification:
            print()))))))f"Would send email notification for regression: {}}}}regression['model_name']}, on {}}}}regression['hardware_type']},")
            ,
            return True
    
    def process_notifications()))))))self) -> None:
        """Process all pending notifications."""
        notifications = self.get_pending_notifications())))))))
        
        if not notifications:
            print()))))))"No pending notifications.")
        return
            
        print()))))))f"Processing {}}}}len()))))))notifications)} notification()))))))s)...")
        
        for notification in notifications:
            # Determine notification method based on severity
            if notification['severity'] == 'critical':,
                # For critical regressions, create GitHub issue and send email
            issue_url = self.send_github_issue()))))))notification)
            email_sent = self.send_email_notification()))))))notification)
                
                if issue_url or email_sent:
                    self.mark_notification_sent()))))))notification['id'])
                    ,    ,,
            elif notification['severity'] == 'high':,
                # For high severity, create GitHub issue
            issue_url = self.send_github_issue()))))))notification)
                
                if issue_url:
                    self.mark_notification_sent()))))))notification['id'])
                    ,    ,,
            else:
                # For medium severity, just mark as sent ()))))))will appear in dashboard)
                self.mark_notification_sent()))))))notification['id'])
                ,
                print()))))))"Notification processing complete.")


def main()))))))):
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser()))))))description='Time Series Performance Tracker')
    subparsers = parser.add_subparsers()))))))dest='command', help='Command to execute')
    
    # Schema creation command
    schema_parser = subparsers.add_parser()))))))'create-schema', help='Create schema extensions')
    schema_parser.add_argument()))))))'--db-path', help='Database path')
    
    # Regression detection command
    detect_parser = subparsers.add_parser()))))))'detect', help='Detect performance regressions')
    detect_parser.add_argument()))))))'--db-path', help='Database path')
    detect_parser.add_argument()))))))'--model', help='Model name filter')
    detect_parser.add_argument()))))))'--hardware', help='Hardware type filter')
    detect_parser.add_argument()))))))'--metric', default='throughput', 
    choices=['throughput', 'latency', 'memory_usage'],
    help='Metric to analyze')
    detect_parser.add_argument()))))))'--threshold', type=float, default=0.1,
    help='Regression threshold ()))))))0.1 = 10%%)')
    
    # Visualization command
    visualize_parser = subparsers.add_parser()))))))'visualize', help='Create trend visualizations')
    visualize_parser.add_argument()))))))'--db-path', help='Database path')
    visualize_parser.add_argument()))))))'--model', required=True, help='Model name')
    visualize_parser.add_argument()))))))'--hardware', required=True, help='Hardware type')
    visualize_parser.add_argument()))))))'--metric', default='throughput',
    choices=['throughput', 'latency', 'memory_usage'],
    help='Metric to visualize')
    visualize_parser.add_argument()))))))'--batch-size', type=int, help='Batch size filter')
    visualize_parser.add_argument()))))))'--output', help='Output file path')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser()))))))'dashboard', help='Create regression dashboard')
    dashboard_parser.add_argument()))))))'--db-path', help='Database path')
    dashboard_parser.add_argument()))))))'--days', type=int, default=30, help='Days to include')
    dashboard_parser.add_argument()))))))'--limit', type=int, default=10, help='Maximum regressions')
    dashboard_parser.add_argument()))))))'--output', help='Output file path')
    
    # Notification command
    notify_parser = subparsers.add_parser()))))))'notify', help='Process notifications')
    notify_parser.add_argument()))))))'--db-path', help='Database path')
    
    args = parser.parse_args())))))))
    
    if args.command == 'create-schema':
        schema = TimeSeriesSchema()))))))args.db_path)
        schema.create_schema_extensions())))))))
        
    elif args.command == 'detect':
        detector = RegressionDetector()))))))args.db_path, args.threshold)
        
        # Get model_id if model name provided
        model_id = None:
        if args.model:
            conn = get_db_connection()))))))args.db_path)
            query = "SELECT id FROM models WHERE name = ?"
            result = conn.execute()))))))query, ()))))))args.model,)).fetchone())))))))
            conn.close())))))))
            
            if result:
                model_id = result[0]
                ,,
        # Get hardware_id if hardware type provided
        hardware_id = None:
        if args.hardware:
            conn = get_db_connection()))))))args.db_path)
            query = "SELECT id FROM hardware_platforms WHERE type = ?"
            result = conn.execute()))))))query, ()))))))args.hardware,)).fetchone())))))))
            conn.close())))))))
            
            if result:
                hardware_id = result[0]
                ,,
                regressions = detector.detect_regressions()))))))model_id, hardware_id, args.metric)
        
        if regressions:
            print()))))))f"Detected {}}}}len()))))))regressions)} regression()))))))s):")
            for reg in regressions:
                print()))))))f"- {}}}}reg['model_name']}, on {}}}}reg['hardware_type']},: {}}}}reg['percent_change']:.2f}% {}}}}args.metric} {}}}}'increase' if args.metric == 'latency' else 'decrease'}")
                
            # Record regressions
            detector.record_regressions()))))))regressions):
        else:
            print()))))))"No regressions detected.")
            
    elif args.command == 'visualize':
        visualizer = TrendVisualizer()))))))args.db_path)
        output_file = visualizer.visualize_metric_trend()))))))
        args.model,
        args.hardware,
        args.metric,
        args.batch_size,
        output_file=args.output
        )
        
        if output_file:
            print()))))))f"Visualization saved to: {}}}}output_file}")
            
    elif args.command == 'dashboard':
        visualizer = TrendVisualizer()))))))args.db_path)
        output_file = visualizer.create_regression_dashboard()))))))
        args.output,
        args.days,
        args.limit
        )
        
        if output_file:
            print()))))))f"Dashboard saved to: {}}}}output_file}")
            
    elif args.command == 'notify':
        notifier = NotificationSystem()))))))args.db_path)
        notifier.process_notifications())))))))
        
    else:
        parser.print_help())))))))


if __name__ == "__main__":
    main())))))))