#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-Series Performance Tracking Module

This module provides comprehensive tracking of performance metrics over time,
identifies regressions, analyzes trends, and sends notifications.

Author: IPFS Accelerate Python Framework Team
Date: March 15, 2025
Version: 1.0
"""

import os
import json
import hashlib
import logging
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
import requests
import git

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeSeriesPerformance:
    """
    Time-series performance tracking for IPFS Accelerate Python Framework
    """
    
    def __init__(self, 
                 db_path: str = None, 
                 regression_thresholds: Dict[str, float] = None,
                 notification_config: Dict[str, Any] = None):
        """
        Initialize the time-series performance tracking system
        
        Args:
            db_path: Path to the DuckDB database file
            regression_thresholds: Thresholds for regression detection
                keys: 'throughput', 'latency', 'memory', 'power'
                values: percentage thresholds (-5.0 means 5% worse for throughput)
            notification_config: Configuration for notifications
                must contain keys: 'enabled', 'methods', 'targets'
        """
        # Set database path
        self.db_path = db_path or os.environ.get('BENCHMARK_DB_PATH', './benchmark_db.duckdb')
        
        # Connect to database
        self._connect_to_db()
        
        # Set regression thresholds with defaults
        self.regression_thresholds = regression_thresholds or {
            'throughput': -5.0,  # 5% worse (negative because lower is worse)
            'latency': 5.0,      # 5% worse (positive because higher is worse)
            'memory': 5.0,       # 5% worse (positive because higher is worse)
            'power': 5.0         # 5% worse (positive because higher is worse)
        }
        
        # Set notification configuration with defaults
        if notification_config is None:
            notification_config = {
                'enabled': False,
                'methods': ['log'],  # 'log', 'email', 'slack', 'github_issue', 'webhook'
                'targets': {
                    'email': [],
                    'slack': '',
                    'github': {
                        'repo': '',
                        'token': '',
                        'labels': ['regression', 'performance']
                    },
                    'webhook': ''
                }
            }
        self.notification_config = notification_config
        
        # Initialize git repository for commit info
        self._init_git_repo()
        
        # Create schema if needed
        self._ensure_schema()
    
    def _connect_to_db(self):
        """Connect to the DuckDB database"""
        try:
            self.conn = duckdb.connect(self.db_path)
            logger.info(f"Connected to database at {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def _init_git_repo(self):
        """Initialize git repository information"""
        try:
            # Find the git repository root
            current_dir = os.path.abspath(os.path.dirname(__file__))
            while current_dir != os.path.dirname(current_dir):  # Not at root
                if os.path.exists(os.path.join(current_dir, '.git')):
                    break
                current_dir = os.path.dirname(current_dir)
            
            # If we found a git repository
            if os.path.exists(os.path.join(current_dir, '.git')):
                self.repo = git.Repo(current_dir)
                logger.info(f"Git repository found at {current_dir}")
                self.git_available = True
            else:
                logger.warning("No git repository found")
                self.git_available = False
        except Exception as e:
            logger.warning(f"Error initializing git repository: {e}")
            self.git_available = False
    
    def _ensure_schema(self):
        """Ensure the time-series schema exists in the database"""
        try:
            # Check if the schema file exists
            schema_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'db_schema',
                'time_series_schema.sql'
            )
            
            if not os.path.exists(schema_path):
                logger.error(f"Schema file not found at {schema_path}")
                raise FileNotFoundError(f"Schema file not found at {schema_path}")
            
            # Execute the schema SQL
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
                self.conn.execute(schema_sql)
            
            logger.info("Time-series schema applied successfully")
        except Exception as e:
            logger.error(f"Error ensuring schema: {e}")
            raise
    
    def get_environment_hash(self) -> str:
        """
        Generate a hash representing the current environment
        
        Returns:
            A string hash representing the environment
        """
        env_info = {}
        
        # System information
        env_info['os'] = os.name
        env_info['platform'] = os.sys.platform
        env_info['python_version'] = os.sys.version
        
        # Environment variables (only relevant ones)
        relevant_vars = [
            'CUDA_VISIBLE_DEVICES',
            'BENCHMARK_DB_PATH',
            'DEPRECATE_JSON_OUTPUT',
            'WEBGPU_COMPUTE_SHADERS_ENABLED',
            'WEB_PARALLEL_LOADING_ENABLED',
            'WEBGPU_SHADER_PRECOMPILE_ENABLED'
        ]
        env_info['env_vars'] = {k: os.environ.get(k) for k in relevant_vars if k in os.environ}
        
        # Git commit if available
        if self.git_available:
            env_info['git_commit'] = self.repo.head.commit.hexsha
            env_info['git_branch'] = self.repo.active_branch.name
        
        # Create hash
        env_json = json.dumps(env_info, sort_keys=True)
        env_hash = hashlib.sha256(env_json.encode()).hexdigest()
        
        return env_hash
    
    def record_performance_result(self, 
                                  model_id: int, 
                                  hardware_id: int,
                                  batch_size: int,
                                  sequence_length: Optional[int],
                                  precision: str,
                                  throughput: float,
                                  latency: float,
                                  memory: float,
                                  power: float,
                                  version_tag: Optional[str] = None,
                                  run_group_id: Optional[str] = None) -> int:
        """
        Record a performance result with versioning information
        
        Args:
            model_id: Model ID
            hardware_id: Hardware platform ID
            batch_size: Batch size used
            sequence_length: Sequence length or None
            precision: Precision format (e.g., 'fp32', 'fp16', 'int8')
            throughput: Throughput in items per second
            latency: Latency in milliseconds
            memory: Memory usage in MB
            power: Power consumption in watts
            version_tag: Optional version tag (e.g., 'v1.0.0')
            run_group_id: Optional ID to group related runs
        
        Returns:
            The ID of the inserted performance result
        """
        try:
            # Get git commit hash if available
            git_commit_hash = None
            if self.git_available:
                git_commit_hash = self.repo.head.commit.hexsha
            
            # Get environment hash
            environment_hash = self.get_environment_hash()
            
            # Generate a run group ID if not provided
            if run_group_id is None:
                run_group_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            
            # Insert performance result
            result = self.conn.execute("""
                INSERT INTO performance_results (
                    model_id, hardware_id, batch_size, sequence_length, precision,
                    throughput_items_per_second, latency_ms, memory_mb, power_watts,
                    timestamp, version_tag, git_commit_hash, environment_hash, run_group_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
            """, (
                model_id, hardware_id, batch_size, sequence_length, precision,
                throughput, latency, memory, power,
                datetime.now(), version_tag, git_commit_hash, environment_hash, run_group_id
            )).fetchone()
            
            # Get the inserted ID
            result_id = result[0]
            logger.info(f"Recorded performance result with ID {result_id}")
            
            return result_id
        except Exception as e:
            logger.error(f"Error recording performance result: {e}")
            raise
    
    def set_baseline(self,
                     model_id: int,
                     hardware_id: int,
                     batch_size: int,
                     sequence_length: Optional[int],
                     precision: str,
                     days_lookback: int = 7,
                     min_samples: int = 3) -> int:
        """
        Set a performance baseline from recent results
        
        Args:
            model_id: Model ID
            hardware_id: Hardware platform ID
            batch_size: Batch size
            sequence_length: Sequence length or None
            precision: Precision format
            days_lookback: Number of days to look back for samples
            min_samples: Minimum number of samples required
        
        Returns:
            ID of the created or updated baseline
        """
        try:
            # Call the database function to set baseline
            result = self.conn.execute("""
                SELECT set_performance_baseline(?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, hardware_id, batch_size, sequence_length, precision,
                days_lookback, min_samples
            )).fetchone()
            
            baseline_id = result[0]
            logger.info(f"Set baseline with ID {baseline_id} for model_id={model_id}, hardware_id={hardware_id}")
            
            return baseline_id
        except Exception as e:
            logger.error(f"Error setting baseline: {e}")
            raise
    
    def set_all_baselines(self, 
                          days_lookback: int = 7,
                          min_samples: int = 3) -> List[Dict[str, Any]]:
        """
        Set baselines for all model-hardware-config combinations with enough samples
        
        Args:
            days_lookback: Number of days to look back for samples
            min_samples: Minimum number of samples required
        
        Returns:
            List of dictionaries with information about set baselines
        """
        try:
            # Find all unique model-hardware-config combinations with enough samples
            combinations = self.conn.execute(f"""
                SELECT 
                    model_id,
                    hardware_id,
                    batch_size,
                    sequence_length,
                    precision,
                    COUNT(*) as sample_count
                FROM 
                    performance_results
                WHERE 
                    timestamp >= CURRENT_TIMESTAMP - INTERVAL '{days_lookback} days'
                GROUP BY 
                    model_id, hardware_id, batch_size, sequence_length, precision
                HAVING 
                    COUNT(*) >= {min_samples}
            """).fetchall()
            
            baseline_results = []
            
            # Set baseline for each combination
            for combo in combinations:
                model_id, hardware_id, batch_size, sequence_length, precision, sample_count = combo
                
                try:
                    baseline_id = self.set_baseline(
                        model_id=model_id,
                        hardware_id=hardware_id,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        precision=precision,
                        days_lookback=days_lookback,
                        min_samples=min_samples
                    )
                    
                    # Get model and hardware names for logging
                    names = self.conn.execute("""
                        SELECT m.model_name, h.hardware_type
                        FROM models m, hardware_platforms h
                        WHERE m.model_id = ? AND h.hardware_id = ?
                    """, (model_id, hardware_id)).fetchone()
                    
                    model_name, hardware_type = names
                    
                    baseline_results.append({
                        'baseline_id': baseline_id,
                        'model_id': model_id,
                        'model_name': model_name,
                        'hardware_id': hardware_id,
                        'hardware_type': hardware_type,
                        'batch_size': batch_size,
                        'sequence_length': sequence_length,
                        'precision': precision,
                        'sample_count': sample_count,
                        'status': 'success'
                    })
                except Exception as e:
                    logger.error(f"Error setting baseline for model_id={model_id}, hardware_id={hardware_id}: {e}")
                    baseline_results.append({
                        'model_id': model_id,
                        'hardware_id': hardware_id,
                        'batch_size': batch_size,
                        'sequence_length': sequence_length,
                        'precision': precision,
                        'sample_count': sample_count,
                        'status': 'error',
                        'error': str(e)
                    })
            
            logger.info(f"Set {len([r for r in baseline_results if r['status'] == 'success'])} baselines")
            return baseline_results
        except Exception as e:
            logger.error(f"Error setting all baselines: {e}")
            raise
    
    def detect_regressions(self,
                           model_id: Optional[int] = None,
                           hardware_id: Optional[int] = None,
                           days_lookback: int = 1) -> List[Dict[str, Any]]:
        """
        Detect performance regressions based on configured thresholds
        
        Args:
            model_id: Optional model ID filter
            hardware_id: Optional hardware ID filter
            days_lookback: Number of days to look back for results to compare
        
        Returns:
            List of detected regressions
        """
        try:
            # Call the database function to detect regressions
            regressions = self.conn.execute("""
                SELECT * FROM detect_performance_regressions(?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, hardware_id, days_lookback,
                self.regression_thresholds['throughput'],
                self.regression_thresholds['latency'],
                self.regression_thresholds['memory'],
                self.regression_thresholds['power']
            )).fetchall()
            
            # Convert to list of dictionaries
            regression_list = []
            for reg in regressions:
                regression_id, model_name, hardware_type, batch_size, precision, \
                regression_type, severity, test_date, baseline_date = reg
                
                regression_list.append({
                    'regression_id': regression_id,
                    'model_name': model_name,
                    'hardware_type': hardware_type,
                    'batch_size': batch_size,
                    'precision': precision,
                    'regression_type': regression_type,
                    'severity': severity,
                    'test_date': test_date,
                    'baseline_date': baseline_date
                })
            
            logger.info(f"Detected {len(regression_list)} regressions")
            
            # Record detected regressions
            self._record_detected_regressions(regression_list)
            
            # Send notifications if enabled
            if self.notification_config['enabled'] and regression_list:
                self._send_regression_notifications(regression_list)
            
            return regression_list
        except Exception as e:
            logger.error(f"Error detecting regressions: {e}")
            raise
    
    def _record_detected_regressions(self, regressions: List[Dict[str, Any]]):
        """
        Record detected regressions in the database
        
        Args:
            regressions: List of regression dictionaries
        """
        try:
            for reg in regressions:
                # Get performance_id and baseline_id
                result = self.conn.execute("""
                    SELECT pr.id as performance_id, pb.baseline_id
                    FROM performance_results pr
                    JOIN models m ON pr.model_id = m.model_id
                    JOIN hardware_platforms h ON pr.hardware_id = h.hardware_id
                    JOIN performance_baselines pb ON 
                        m.model_id = pb.model_id AND
                        h.hardware_id = pb.hardware_id AND
                        pr.batch_size = pb.batch_size AND
                        (pr.sequence_length = pb.sequence_length OR (pr.sequence_length IS NULL AND pb.sequence_length IS NULL)) AND
                        pr.precision = pb.precision
                    WHERE m.model_name = ? AND h.hardware_type = ? AND
                          pr.batch_size = ? AND pr.precision = ? AND
                          pr.timestamp = ?
                """, (
                    reg['model_name'], reg['hardware_type'],
                    reg['batch_size'], reg['precision'],
                    reg['test_date']
                )).fetchone()
                
                if result:
                    performance_id, baseline_id = result
                    
                    # Insert into performance_regressions
                    self.conn.execute("""
                        INSERT INTO performance_regressions (
                            regression_id, performance_id, baseline_id,
                            detection_date, regression_type, severity,
                            status, notes
                        )
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, 'detected', ?)
                        ON CONFLICT (regression_id) DO NOTHING
                    """, (
                        reg['regression_id'], performance_id, baseline_id,
                        reg['regression_type'], reg['severity'],
                        f"Regression detected: {reg['severity']:.2f}% worse than baseline"
                    ))
                else:
                    logger.warning(f"Could not find performance_id and baseline_id for regression {reg['regression_id']}")
        except Exception as e:
            logger.error(f"Error recording detected regressions: {e}")
            # Continue processing other regressions
    
    def _send_regression_notifications(self, regressions: List[Dict[str, Any]]):
        """
        Send notifications for detected regressions
        
        Args:
            regressions: List of regression dictionaries
        """
        if not self.notification_config['enabled']:
            logger.info("Notifications disabled, skipping")
            return
        
        for method in self.notification_config['methods']:
            try:
                if method == 'log':
                    self._send_log_notification(regressions)
                elif method == 'email':
                    self._send_email_notification(regressions)
                elif method == 'slack':
                    self._send_slack_notification(regressions)
                elif method == 'github_issue':
                    self._send_github_issue(regressions)
                elif method == 'webhook':
                    self._send_webhook_notification(regressions)
                else:
                    logger.warning(f"Unknown notification method: {method}")
            except Exception as e:
                logger.error(f"Error sending {method} notification: {e}")
    
    def _send_log_notification(self, regressions: List[Dict[str, Any]]):
        """Send notifications to log"""
        logger.warning(f"PERFORMANCE REGRESSION ALERT: {len(regressions)} regressions detected")
        for reg in regressions:
            logger.warning(
                f"Regression: {reg['model_name']} on {reg['hardware_type']} - "
                f"{reg['regression_type']} degraded by {reg['severity']:.2f}% "
                f"(batch_size={reg['batch_size']}, precision={reg['precision']})"
            )
    
    def _send_email_notification(self, regressions: List[Dict[str, Any]]):
        """Send email notifications"""
        if not self.notification_config['targets']['email']:
            logger.warning("No email targets configured, skipping email notification")
            return
        
        # In a real implementation, this would send actual emails
        # For now, just log that we would send emails
        recipients = self.notification_config['targets']['email']
        logger.info(f"Would send email to {recipients} about {len(regressions)} regressions")
        
        # Record notification in database
        for reg in regressions:
            self.conn.execute("""
                INSERT INTO regression_notifications (
                    regression_id, notification_type, notification_target,
                    notification_status, notification_date
                )
                VALUES (?, 'email', ?, 'sent', CURRENT_TIMESTAMP)
            """, (reg['regression_id'], ','.join(recipients)))
    
    def _send_slack_notification(self, regressions: List[Dict[str, Any]]):
        """Send Slack notifications"""
        if not self.notification_config['targets']['slack']:
            logger.warning("No Slack webhook configured, skipping Slack notification")
            return
        
        # In a real implementation, this would send actual Slack messages
        # For now, just log that we would send Slack messages
        webhook_url = self.notification_config['targets']['slack']
        logger.info(f"Would send Slack message to {webhook_url} about {len(regressions)} regressions")
        
        # Record notification in database
        for reg in regressions:
            self.conn.execute("""
                INSERT INTO regression_notifications (
                    regression_id, notification_type, notification_target,
                    notification_status, notification_date
                )
                VALUES (?, 'slack', ?, 'sent', CURRENT_TIMESTAMP)
            """, (reg['regression_id'], webhook_url))
    
    def _send_github_issue(self, regressions: List[Dict[str, Any]]):
        """Create GitHub issues for regressions"""
        github_config = self.notification_config['targets']['github']
        if not github_config['repo'] or not github_config['token']:
            logger.warning("No GitHub repo or token configured, skipping GitHub issue creation")
            return
        
        # In a real implementation, this would create actual GitHub issues
        # For now, just log that we would create GitHub issues
        repo = github_config['repo']
        logger.info(f"Would create GitHub issue in {repo} about {len(regressions)} regressions")
        
        # Record notification in database
        for reg in regressions:
            self.conn.execute("""
                INSERT INTO regression_notifications (
                    regression_id, notification_type, notification_target,
                    notification_status, notification_date
                )
                VALUES (?, 'github_issue', ?, 'sent', CURRENT_TIMESTAMP)
            """, (reg['regression_id'], repo))
    
    def _send_webhook_notification(self, regressions: List[Dict[str, Any]]):
        """Send webhook notifications"""
        if not self.notification_config['targets']['webhook']:
            logger.warning("No webhook URL configured, skipping webhook notification")
            return
        
        # In a real implementation, this would send actual webhook requests
        # For now, just log that we would send webhook requests
        webhook_url = self.notification_config['targets']['webhook']
        logger.info(f"Would send webhook request to {webhook_url} about {len(regressions)} regressions")
        
        # Record notification in database
        for reg in regressions:
            self.conn.execute("""
                INSERT INTO regression_notifications (
                    regression_id, notification_type, notification_target,
                    notification_status, notification_date
                )
                VALUES (?, 'webhook', ?, 'sent', CURRENT_TIMESTAMP)
            """, (reg['regression_id'], webhook_url))
    
    def analyze_trends(self,
                       model_id: Optional[int] = None,
                       hardware_id: Optional[int] = None,
                       metric: str = 'throughput',
                       days_lookback: int = 30,
                       min_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Analyze performance trends over time
        
        Args:
            model_id: Optional model ID filter
            hardware_id: Optional hardware ID filter
            metric: Metric to analyze ('throughput', 'latency', 'memory', 'power')
            days_lookback: Number of days to look back for analysis
            min_samples: Minimum number of samples required for analysis
        
        Returns:
            List of trend analysis results
        """
        try:
            # Map metric to database column
            metric_column = {
                'throughput': 'throughput_items_per_second',
                'latency': 'latency_ms',
                'memory': 'memory_mb',
                'power': 'power_watts'
            }.get(metric)
            
            if not metric_column:
                raise ValueError(f"Invalid metric: {metric}")
            
            # Get data for analysis
            query = f"""
                SELECT 
                    pr.model_id,
                    m.model_name,
                    pr.hardware_id,
                    h.hardware_type,
                    pr.batch_size,
                    pr.precision,
                    pr.{metric_column} as metric_value,
                    pr.timestamp
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms h ON pr.hardware_id = h.hardware_id
                WHERE 
                    pr.timestamp >= CURRENT_TIMESTAMP - INTERVAL '{days_lookback} days'
                    {f'AND pr.model_id = {model_id}' if model_id is not None else ''}
                    {f'AND pr.hardware_id = {hardware_id}' if hardware_id is not None else ''}
                ORDER BY 
                    pr.model_id, pr.hardware_id, pr.batch_size, pr.precision, pr.timestamp
            """
            
            df = self.conn.execute(query).fetchdf()
            
            if df.empty:
                logger.warning("No data found for trend analysis")
                return []
            
            # Group by model, hardware, batch_size, precision
            groups = df.groupby(['model_id', 'hardware_id', 'batch_size', 'precision'])
            
            trend_results = []
            
            for (model_id, hardware_id, batch_size, precision), group_df in groups:
                # Skip if not enough samples
                if len(group_df) < min_samples:
                    continue
                
                # Get model and hardware names
                model_name = group_df['model_name'].iloc[0]
                hardware_type = group_df['hardware_type'].iloc[0]
                
                # Perform trend analysis
                result = self._analyze_single_trend(
                    group_df, model_id, hardware_id, model_name, hardware_type,
                    batch_size, precision, metric
                )
                
                if result:
                    trend_results.append(result)
            
            logger.info(f"Analyzed trends for {len(trend_results)} configurations")
            
            # Record trends in database
            self._record_trends(trend_results, metric)
            
            return trend_results
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            raise
    
    def _analyze_single_trend(self,
                              df: pd.DataFrame,
                              model_id: int,
                              hardware_id: int,
                              model_name: str,
                              hardware_type: str,
                              batch_size: int,
                              precision: str,
                              metric: str) -> Optional[Dict[str, Any]]:
        """
        Analyze trend for a single configuration
        
        Args:
            df: DataFrame with performance data for this configuration
            model_id: Model ID
            hardware_id: Hardware ID
            model_name: Model name
            hardware_type: Hardware type
            batch_size: Batch size
            precision: Precision
            metric: Metric being analyzed
        
        Returns:
            Dictionary with trend analysis, or None if analysis fails
        """
        try:
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Extract values and timestamps
            values = df['metric_value'].values
            timestamps = df['timestamp'].values
            
            # Calculate simple statistics
            mean_value = np.mean(values)
            std_value = np.std(values)
            cv = std_value / mean_value if mean_value != 0 else 0
            
            # Check if we have enough variation to detect a trend
            if cv < 0.01:  # Less than 1% coefficient of variation
                trend_direction = 'stable'
                trend_magnitude = 0.0
                trend_confidence = 1.0
            else:
                # Simple linear regression
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                # Calculate percent change over period
                if len(values) > 1 and values[0] != 0:
                    total_change = (values[-1] - values[0]) / values[0] * 100
                else:
                    total_change = 0
                
                # Determine trend direction
                if metric == 'throughput':
                    # For throughput, higher is better
                    if total_change > 1.0:
                        trend_direction = 'improving'
                    elif total_change < -1.0:
                        trend_direction = 'degrading'
                    else:
                        trend_direction = 'stable'
                else:
                    # For latency, memory, power, lower is better
                    if total_change < -1.0:
                        trend_direction = 'improving'
                    elif total_change > 1.0:
                        trend_direction = 'degrading'
                    else:
                        trend_direction = 'stable'
                
                # If R-squared is low, consider the trend volatile
                if r_value ** 2 < 0.5:
                    trend_direction = 'volatile'
                
                trend_magnitude = abs(total_change)
                trend_confidence = min(1.0, r_value ** 2)
            
            # Create trend data points for JSON storage
            trend_data = []
            for i, (ts, val) in enumerate(zip(timestamps, values)):
                trend_data.append({
                    'index': i,
                    'timestamp': ts.isoformat() if isinstance(ts, datetime) else ts,
                    'value': float(val)
                })
            
            # Create trend result
            trend_result = {
                'model_id': int(model_id),
                'model_name': model_name,
                'hardware_id': int(hardware_id),
                'hardware_type': hardware_type,
                'batch_size': int(batch_size),
                'precision': precision,
                'metric': metric,
                'trend_direction': trend_direction,
                'trend_magnitude': float(trend_magnitude),
                'trend_confidence': float(trend_confidence),
                'mean_value': float(mean_value),
                'std_value': float(std_value),
                'coefficient_variation': float(cv),
                'sample_count': len(values),
                'trend_start_date': timestamps[0].isoformat() if isinstance(timestamps[0], datetime) else timestamps[0],
                'trend_end_date': timestamps[-1].isoformat() if isinstance(timestamps[-1], datetime) else timestamps[-1],
                'trend_data': trend_data
            }
            
            return trend_result
        except Exception as e:
            logger.error(f"Error analyzing trend for model_id={model_id}, hardware_id={hardware_id}: {e}")
            return None
    
    def _record_trends(self, trend_results: List[Dict[str, Any]], metric: str):
        """
        Record trend analysis results in the database
        
        Args:
            trend_results: List of trend analysis results
            metric: Metric type being analyzed
        """
        try:
            for trend in trend_results:
                # Insert into performance_trends
                self.conn.execute("""
                    INSERT INTO performance_trends (
                        model_id, hardware_id, metric_type,
                        trend_start_date, trend_end_date,
                        trend_direction, trend_magnitude, trend_confidence,
                        trend_data, detection_date, notes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                """, (
                    trend['model_id'], trend['hardware_id'], metric,
                    trend['trend_start_date'], trend['trend_end_date'],
                    trend['trend_direction'], trend['trend_magnitude'], trend['trend_confidence'],
                    json.dumps(trend['trend_data']),
                    f"{trend['model_name']} on {trend['hardware_type']} {metric} trend: {trend['trend_direction']} by {trend['trend_magnitude']:.2f}%"
                ))
        except Exception as e:
            logger.error(f"Error recording trends: {e}")
            # Continue processing other trends
    
    def generate_trend_visualization(self, 
                                     model_id: Optional[int] = None,
                                     hardware_id: Optional[int] = None,
                                     metric: str = 'throughput',
                                     days_lookback: int = 30,
                                     output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate visualization of performance trends
        
        Args:
            model_id: Optional model ID filter
            hardware_id: Optional hardware ID filter
            metric: Metric to visualize ('throughput', 'latency', 'memory', 'power')
            days_lookback: Number of days to look back for visualization
            output_path: Optional path to save visualization (if None, a temp file is created)
        
        Returns:
            Path to the generated visualization, or None if generation fails
        """
        try:
            # Map metric to database column and display name
            metric_info = {
                'throughput': {
                    'column': 'throughput_items_per_second',
                    'display': 'Throughput (items/sec)',
                    'higher_better': True
                },
                'latency': {
                    'column': 'latency_ms',
                    'display': 'Latency (ms)',
                    'higher_better': False
                },
                'memory': {
                    'column': 'memory_mb',
                    'display': 'Memory Usage (MB)',
                    'higher_better': False
                },
                'power': {
                    'column': 'power_watts',
                    'display': 'Power Consumption (W)',
                    'higher_better': False
                }
            }.get(metric)
            
            if not metric_info:
                raise ValueError(f"Invalid metric: {metric}")
            
            # Get data for visualization
            query = f"""
                SELECT 
                    m.model_name,
                    h.hardware_type,
                    pr.batch_size,
                    pr.precision,
                    pr.{metric_info['column']} as metric_value,
                    pr.timestamp
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms h ON pr.hardware_id = h.hardware_id
                WHERE 
                    pr.timestamp >= CURRENT_TIMESTAMP - INTERVAL '{days_lookback} days'
                    {f'AND pr.model_id = {model_id}' if model_id is not None else ''}
                    {f'AND pr.hardware_id = {hardware_id}' if hardware_id is not None else ''}
                ORDER BY 
                    m.model_name, h.hardware_type, pr.batch_size, pr.precision, pr.timestamp
            """
            
            df = self.conn.execute(query).fetchdf()
            
            if df.empty:
                logger.warning("No data found for trend visualization")
                return None
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Group by model, hardware, batch_size, precision
            groups = df.groupby(['model_name', 'hardware_type', 'batch_size', 'precision'])
            
            for (model_name, hardware_type, batch_size, precision), group_df in groups:
                # Sort by timestamp
                group_df = group_df.sort_values('timestamp')
                
                # Plot data
                label = f"{model_name} on {hardware_type} (batch={batch_size}, {precision})"
                plt.plot(group_df['timestamp'], group_df['metric_value'], 'o-', label=label)
            
            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel(metric_info['display'])
            plt.title(f"{metric_info['display']} Trend over {days_lookback} Days")
            
            # Add legend
            plt.legend(loc='best')
            
            # Add grid
            plt.grid(alpha=0.3)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Save visualization
            if output_path:
                output_file = output_path
            else:
                # Create temp file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    output_file = tmp.name
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            logger.info(f"Generated trend visualization at {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error generating trend visualization: {e}")
            return None
    
    def get_performance_history(self,
                                model_id: Optional[int] = None,
                                hardware_id: Optional[int] = None,
                                batch_size: Optional[int] = None,
                                precision: Optional[str] = None,
                                days_lookback: int = 30) -> pd.DataFrame:
        """
        Get historical performance data as a pandas DataFrame
        
        Args:
            model_id: Optional model ID filter
            hardware_id: Optional hardware ID filter
            batch_size: Optional batch size filter
            precision: Optional precision filter
            days_lookback: Number of days to look back
        
        Returns:
            DataFrame with historical performance data
        """
        try:
            # Build query with optional filters
            query = f"""
                SELECT *
                FROM performance_metrics_history
                WHERE test_date >= CURRENT_TIMESTAMP - INTERVAL '{days_lookback} days'
            """
            
            params = []
            
            if model_id is not None:
                query += f" AND model_id = ?"
                params.append(model_id)
            
            if hardware_id is not None:
                query += f" AND hardware_id = ?"
                params.append(hardware_id)
            
            if batch_size is not None:
                query += f" AND batch_size = ?"
                params.append(batch_size)
            
            if precision is not None:
                query += f" AND precision = ?"
                params.append(precision)
            
            query += " ORDER BY model_name, hardware_type, batch_size, precision, test_date"
            
            # Execute query
            df = self.conn.execute(query, params).fetchdf()
            
            logger.info(f"Retrieved {len(df)} historical performance records")
            return df
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            raise
    
    def export_performance_report(self, 
                                 model_id: Optional[int] = None,
                                 hardware_id: Optional[int] = None,
                                 days_lookback: int = 30,
                                 format: str = 'markdown',
                                 output_path: Optional[str] = None) -> Optional[str]:
        """
        Export a performance report with trends and regressions
        
        Args:
            model_id: Optional model ID filter
            hardware_id: Optional hardware ID filter
            days_lookback: Number of days to look back
            format: Report format ('markdown' or 'html')
            output_path: Optional path to save report
        
        Returns:
            Path to the generated report, or None if generation fails
        """
        try:
            # Check if format is valid
            if format not in ['markdown', 'html']:
                raise ValueError(f"Invalid format: {format}. Use 'markdown' or 'html'")
            
            # Create temporary directory for report assets
            report_dir = tempfile.mkdtemp()
            
            # Generate trend visualizations
            visualization_paths = {}
            for metric in ['throughput', 'latency', 'memory', 'power']:
                viz_path = os.path.join(report_dir, f"{metric}_trend.png")
                visualization_paths[metric] = self.generate_trend_visualization(
                    model_id=model_id,
                    hardware_id=hardware_id,
                    metric=metric,
                    days_lookback=days_lookback,
                    output_path=viz_path
                )
            
            # Detect regressions
            regressions = self.detect_regressions(
                model_id=model_id,
                hardware_id=hardware_id,
                days_lookback=min(days_lookback, 7)  # Only look at recent regressions
            )
            
            # Analyze trends
            trends = []
            for metric in ['throughput', 'latency', 'memory', 'power']:
                metric_trends = self.analyze_trends(
                    model_id=model_id,
                    hardware_id=hardware_id,
                    metric=metric,
                    days_lookback=days_lookback
                )
                trends.extend(metric_trends)
            
            # Get model and hardware names for title
            title_parts = ["Performance Report"]
            if model_id is not None:
                model_name = self.conn.execute(
                    "SELECT model_name FROM models WHERE model_id = ?",
                    (model_id,)
                ).fetchone()[0]
                title_parts.append(f"for {model_name}")
            
            if hardware_id is not None:
                hardware_type = self.conn.execute(
                    "SELECT hardware_type FROM hardware_platforms WHERE hardware_id = ?",
                    (hardware_id,)
                ).fetchone()[0]
                title_parts.append(f"on {hardware_type}")
            
            title_parts.append(f"over {days_lookback} Days")
            report_title = " ".join(title_parts)
            
            # Generate report content based on format
            if format == 'markdown':
                report_content = self._generate_markdown_report(
                    report_title=report_title,
                    visualization_paths=visualization_paths,
                    regressions=regressions,
                    trends=trends,
                    days_lookback=days_lookback
                )
            else:  # HTML
                report_content = self._generate_html_report(
                    report_title=report_title,
                    visualization_paths=visualization_paths,
                    regressions=regressions,
                    trends=trends,
                    days_lookback=days_lookback
                )
            
            # Write report to file
            if output_path:
                report_file = output_path
            else:
                # Create temp file
                suffix = '.md' if format == 'markdown' else '.html'
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    report_file = tmp.name
            
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Generated performance report at {report_file}")
            return report_file
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")
            return None
    
    def _generate_markdown_report(self,
                                 report_title: str,
                                 visualization_paths: Dict[str, Optional[str]],
                                 regressions: List[Dict[str, Any]],
                                 trends: List[Dict[str, Any]],
                                 days_lookback: int) -> str:
        """
        Generate a markdown performance report
        
        Args:
            report_title: Title of the report
            visualization_paths: Dictionary of visualization file paths by metric
            regressions: List of detected regressions
            trends: List of trend analysis results
            days_lookback: Number of days in the report
        
        Returns:
            Markdown report content
        """
        # Start with report header
        now = datetime.now()
        lines = [
            f"# {report_title}",
            "",
            f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Period:** {(now - timedelta(days=days_lookback)).strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}",
            "",
            "## Summary",
            ""
        ]
        
        # Add summary
        regression_count = len(regressions)
        if regression_count > 0:
            lines.append(f"⚠️ **{regression_count} performance regressions detected**")
        else:
            lines.append("✅ No performance regressions detected")
        
        # Count trends by direction
        trend_counts = {
            'improving': 0,
            'degrading': 0,
            'stable': 0,
            'volatile': 0
        }
        for trend in trends:
            trend_counts[trend['trend_direction']] = trend_counts.get(trend['trend_direction'], 0) + 1
        
        lines.append("")
        lines.append("Trends:")
        lines.append(f"- 📈 Improving: {trend_counts['improving']}")
        lines.append(f"- 📉 Degrading: {trend_counts['degrading']}")
        lines.append(f"- ➖ Stable: {trend_counts['stable']}")
        lines.append(f"- 🔄 Volatile: {trend_counts['volatile']}")
        lines.append("")
        
        # Add performance visualizations
        lines.append("## Performance Visualizations")
        lines.append("")
        
        for metric, path in visualization_paths.items():
            if path and os.path.exists(path):
                # For markdown reports, we can only reference images that exist relative to the report
                # In a real implementation, we might copy images to a known location or embed them
                lines.append(f"### {metric.capitalize()} Trend")
                lines.append("")
                lines.append(f"![{metric.capitalize()} Trend]({path})")
                lines.append("")
        
        # Add regression details
        lines.append("## Performance Regressions")
        lines.append("")
        
        if regressions:
            lines.append("| Model | Hardware | Metric | Severity | Date |")
            lines.append("|-------|----------|--------|----------|------|")
            
            for reg in sorted(regressions, key=lambda x: abs(x['severity']), reverse=True):
                severity_str = f"{reg['severity']:.2f}%"
                date_str = reg['test_date'].strftime('%Y-%m-%d') if isinstance(reg['test_date'], datetime) else reg['test_date']
                
                lines.append(f"| {reg['model_name']} | {reg['hardware_type']} | {reg['regression_type']} | {severity_str} | {date_str} |")
        else:
            lines.append("No regressions detected in the specified period.")
        
        lines.append("")
        
        # Add trend details
        lines.append("## Performance Trends")
        lines.append("")
        
        if trends:
            lines.append("| Model | Hardware | Metric | Direction | Magnitude | Confidence |")
            lines.append("|-------|----------|--------|-----------|-----------|------------|")
            
            for trend in sorted(trends, key=lambda x: abs(x['trend_magnitude']), reverse=True):
                magnitude_str = f"{trend['trend_magnitude']:.2f}%"
                confidence_str = f"{trend['trend_confidence']:.2f}"
                
                # Use emoji for direction
                direction_emoji = {
                    'improving': '📈',
                    'degrading': '📉',
                    'stable': '➖',
                    'volatile': '🔄'
                }.get(trend['trend_direction'], '')
                
                lines.append(f"| {trend['model_name']} | {trend['hardware_type']} | {trend['metric']} | {direction_emoji} {trend['trend_direction']} | {magnitude_str} | {confidence_str} |")
        else:
            lines.append("No significant trends detected in the specified period.")
        
        # Join all lines with newlines
        return "\n".join(lines)
    
    def _generate_html_report(self,
                             report_title: str,
                             visualization_paths: Dict[str, Optional[str]],
                             regressions: List[Dict[str, Any]],
                             trends: List[Dict[str, Any]],
                             days_lookback: int) -> str:
        """
        Generate an HTML performance report
        
        Args:
            report_title: Title of the report
            visualization_paths: Dictionary of visualization file paths by metric
            regressions: List of detected regressions
            trends: List of trend analysis results
            days_lookback: Number of days in the report
        
        Returns:
            HTML report content
        """
        # In a real implementation, this would generate a complete HTML report
        # For now, we'll create a simple HTML version
        now = datetime.now()
        
        # Start with basic HTML structure
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{report_title}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }",
            "h1 { color: #333366; }",
            "h2 { color: #333366; margin-top: 20px; }",
            "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
            "th, td { border: 1px solid #ddd; padding: 8px; }",
            "th { background-color: #f2f2f2; text-align: left; }",
            "tr:nth-child(even) { background-color: #f9f9f9; }",
            ".improving { color: green; }",
            ".degrading { color: red; }",
            ".stable { color: gray; }",
            ".volatile { color: orange; }",
            ".visualization { max-width: 100%; height: auto; margin-bottom: 20px; }",
            ".summary { background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{report_title}</h1>",
            f"<p><strong>Generated:</strong> {now.strftime('%Y-%m-%d %H:%M:%S')}<br>",
            f"<strong>Period:</strong> {(now - timedelta(days=days_lookback)).strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}</p>",
            "<h2>Summary</h2>",
            "<div class='summary'>"
        ]
        
        # Add summary
        regression_count = len(regressions)
        if regression_count > 0:
            html.append(f"<p>⚠️ <strong>{regression_count} performance regressions detected</strong></p>")
        else:
            html.append("<p>✅ No performance regressions detected</p>")
        
        # Count trends by direction
        trend_counts = {
            'improving': 0,
            'degrading': 0,
            'stable': 0,
            'volatile': 0
        }
        for trend in trends:
            trend_counts[trend['trend_direction']] = trend_counts.get(trend['trend_direction'], 0) + 1
        
        html.append("<p>Trends:</p>")
        html.append("<ul>")
        html.append(f"<li>📈 Improving: {trend_counts['improving']}</li>")
        html.append(f"<li>📉 Degrading: {trend_counts['degrading']}</li>")
        html.append(f"<li>➖ Stable: {trend_counts['stable']}</li>")
        html.append(f"<li>🔄 Volatile: {trend_counts['volatile']}</li>")
        html.append("</ul>")
        html.append("</div>")
        
        # Add visualizations
        html.append("<h2>Performance Visualizations</h2>")
        
        for metric, path in visualization_paths.items():
            if path and os.path.exists(path):
                # For HTML reports in a real implementation, we might:
                # 1. Convert images to base64 and embed them
                # 2. Copy images to a relative path
                # 3. Serve images from a web server
                # Here we'll just reference them as files
                html.append(f"<h3>{metric.capitalize()} Trend</h3>")
                html.append(f"<img class='visualization' src='{path}' alt='{metric.capitalize()} Trend'>")
        
        # Add regression details
        html.append("<h2>Performance Regressions</h2>")
        
        if regressions:
            html.append("<table>")
            html.append("<tr><th>Model</th><th>Hardware</th><th>Metric</th><th>Severity</th><th>Date</th></tr>")
            
            for reg in sorted(regressions, key=lambda x: abs(x['severity']), reverse=True):
                severity_str = f"{reg['severity']:.2f}%"
                date_str = reg['test_date'].strftime('%Y-%m-%d') if isinstance(reg['test_date'], datetime) else reg['test_date']
                
                html.append("<tr>")
                html.append(f"<td>{reg['model_name']}</td>")
                html.append(f"<td>{reg['hardware_type']}</td>")
                html.append(f"<td>{reg['regression_type']}</td>")
                html.append(f"<td>{severity_str}</td>")
                html.append(f"<td>{date_str}</td>")
                html.append("</tr>")
            
            html.append("</table>")
        else:
            html.append("<p>No regressions detected in the specified period.</p>")
        
        # Add trend details
        html.append("<h2>Performance Trends</h2>")
        
        if trends:
            html.append("<table>")
            html.append("<tr><th>Model</th><th>Hardware</th><th>Metric</th><th>Direction</th><th>Magnitude</th><th>Confidence</th></tr>")
            
            for trend in sorted(trends, key=lambda x: abs(x['trend_magnitude']), reverse=True):
                magnitude_str = f"{trend['trend_magnitude']:.2f}%"
                confidence_str = f"{trend['trend_confidence']:.2f}"
                
                # Use emoji and class for direction
                direction_emoji = {
                    'improving': '📈',
                    'degrading': '📉',
                    'stable': '➖',
                    'volatile': '🔄'
                }.get(trend['trend_direction'], '')
                
                html.append("<tr>")
                html.append(f"<td>{trend['model_name']}</td>")
                html.append(f"<td>{trend['hardware_type']}</td>")
                html.append(f"<td>{trend['metric']}</td>")
                html.append(f"<td class='{trend['trend_direction']}'>{direction_emoji} {trend['trend_direction']}</td>")
                html.append(f"<td>{magnitude_str}</td>")
                html.append(f"<td>{confidence_str}</td>")
                html.append("</tr>")
            
            html.append("</table>")
        else:
            html.append("<p>No significant trends detected in the specified period.</p>")
        
        # Close HTML
        html.append("</body>")
        html.append("</html>")
        
        # Join all lines with newlines
        return "\n".join(html)


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Time-Series Performance Tracking")
    
    # Main command groups
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Record command
    record_parser = subparsers.add_parser("record", help="Record a performance result")
    record_parser.add_argument("--model-id", type=int, required=True, help="Model ID")
    record_parser.add_argument("--hardware-id", type=int, required=True, help="Hardware ID")
    record_parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    record_parser.add_argument("--sequence-length", type=int, help="Sequence length")
    record_parser.add_argument("--precision", type=str, default="fp32", help="Precision format")
    record_parser.add_argument("--throughput", type=float, required=True, help="Throughput in items/sec")
    record_parser.add_argument("--latency", type=float, required=True, help="Latency in ms")
    record_parser.add_argument("--memory", type=float, required=True, help="Memory usage in MB")
    record_parser.add_argument("--power", type=float, required=True, help="Power consumption in watts")
    record_parser.add_argument("--version-tag", type=str, help="Version tag")
    record_parser.add_argument("--run-group", type=str, help="Run group ID")
    
    # Baseline command
    baseline_parser = subparsers.add_parser("baseline", help="Set performance baselines")
    baseline_parser.add_argument("--model-id", type=int, help="Model ID")
    baseline_parser.add_argument("--hardware-id", type=int, help="Hardware ID")
    baseline_parser.add_argument("--batch-size", type=int, help="Batch size")
    baseline_parser.add_argument("--sequence-length", type=int, help="Sequence length")
    baseline_parser.add_argument("--precision", type=str, help="Precision format")
    baseline_parser.add_argument("--days", type=int, default=7, help="Days to look back")
    baseline_parser.add_argument("--min-samples", type=int, default=3, help="Minimum samples required")
    baseline_parser.add_argument("--all", action="store_true", help="Set baselines for all configurations")
    
    # Regression command
    regression_parser = subparsers.add_parser("regression", help="Detect performance regressions")
    regression_parser.add_argument("--model-id", type=int, help="Model ID")
    regression_parser.add_argument("--hardware-id", type=int, help="Hardware ID")
    regression_parser.add_argument("--days", type=int, default=1, help="Days to look back")
    regression_parser.add_argument("--throughput-threshold", type=float, help="Throughput regression threshold")
    regression_parser.add_argument("--latency-threshold", type=float, help="Latency regression threshold")
    regression_parser.add_argument("--memory-threshold", type=float, help="Memory regression threshold")
    regression_parser.add_argument("--power-threshold", type=float, help="Power regression threshold")
    regression_parser.add_argument("--notify", action="store_true", help="Send notifications")
    
    # Trend command
    trend_parser = subparsers.add_parser("trend", help="Analyze performance trends")
    trend_parser.add_argument("--model-id", type=int, help="Model ID")
    trend_parser.add_argument("--hardware-id", type=int, help="Hardware ID")
    trend_parser.add_argument("--metric", type=str, default="throughput", choices=["throughput", "latency", "memory", "power"], help="Metric to analyze")
    trend_parser.add_argument("--days", type=int, default=30, help="Days to look back")
    trend_parser.add_argument("--min-samples", type=int, default=5, help="Minimum samples required")
    trend_parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    trend_parser.add_argument("--output", type=str, help="Output path for visualization")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate performance report")
    report_parser.add_argument("--model-id", type=int, help="Model ID")
    report_parser.add_argument("--hardware-id", type=int, help="Hardware ID")
    report_parser.add_argument("--days", type=int, default=30, help="Days to look back")
    report_parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "html"], help="Report format")
    report_parser.add_argument("--output", type=str, required=True, help="Output path for report")
    
    # Database configuration
    parser.add_argument("--db-path", type=str, help="Path to DuckDB database file")
    
    args = parser.parse_args()
    
    # Initialize time-series performance tracker
    ts_perf = TimeSeriesPerformance(db_path=args.db_path)
    
    # Process commands
    if args.command == "record":
        result_id = ts_perf.record_performance_result(
            model_id=args.model_id,
            hardware_id=args.hardware_id,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            precision=args.precision,
            throughput=args.throughput,
            latency=args.latency,
            memory=args.memory,
            power=args.power,
            version_tag=args.version_tag,
            run_group_id=args.run_group
        )
        print(f"Recorded performance result with ID: {result_id}")
    
    elif args.command == "baseline":
        if args.all:
            results = ts_perf.set_all_baselines(
                days_lookback=args.days,
                min_samples=args.min_samples
            )
            print(f"Set {len([r for r in results if r['status'] == 'success'])} baselines")
            for result in results:
                if result['status'] == 'success':
                    print(f"Baseline ID {result['baseline_id']}: {result['model_name']} on {result['hardware_type']}")
                else:
                    print(f"Error: {result['model_id']}/{result['hardware_id']}: {result['error']}")
        else:
            if not all([args.model_id, args.hardware_id, args.batch_size, args.precision]):
                print("Error: When not using --all, you must specify --model-id, --hardware-id, --batch-size, and --precision")
                exit(1)
                
            baseline_id = ts_perf.set_baseline(
                model_id=args.model_id,
                hardware_id=args.hardware_id,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                precision=args.precision,
                days_lookback=args.days,
                min_samples=args.min_samples
            )
            print(f"Set baseline with ID: {baseline_id}")
    
    elif args.command == "regression":
        # Set custom thresholds if provided
        thresholds = {}
        if args.throughput_threshold is not None:
            thresholds['throughput'] = args.throughput_threshold
        if args.latency_threshold is not None:
            thresholds['latency'] = args.latency_threshold
        if args.memory_threshold is not None:
            thresholds['memory'] = args.memory_threshold
        if args.power_threshold is not None:
            thresholds['power'] = args.power_threshold
        
        if thresholds:
            ts_perf.regression_thresholds.update(thresholds)
        
        # Enable notifications if requested
        if args.notify:
            ts_perf.notification_config['enabled'] = True
        
        # Detect regressions
        regressions = ts_perf.detect_regressions(
            model_id=args.model_id,
            hardware_id=args.hardware_id,
            days_lookback=args.days
        )
        
        # Print results
        if regressions:
            print(f"Detected {len(regressions)} regressions:")
            for reg in regressions:
                print(f"  {reg['model_name']} on {reg['hardware_type']}: {reg['regression_type']} degraded by {reg['severity']:.2f}%")
        else:
            print("No regressions detected")
    
    elif args.command == "trend":
        # Analyze trends
        trends = ts_perf.analyze_trends(
            model_id=args.model_id,
            hardware_id=args.hardware_id,
            metric=args.metric,
            days_lookback=args.days,
            min_samples=args.min_samples
        )
        
        # Print results
        if trends:
            print(f"Analyzed {len(trends)} trends:")
            for trend in trends:
                print(f"  {trend['model_name']} on {trend['hardware_type']}: {trend['metric']} {trend['trend_direction']} by {trend['trend_magnitude']:.2f}% (confidence: {trend['trend_confidence']:.2f})")
        else:
            print("No significant trends detected")
        
        # Generate visualization if requested
        if args.visualize:
            viz_path = ts_perf.generate_trend_visualization(
                model_id=args.model_id,
                hardware_id=args.hardware_id,
                metric=args.metric,
                days_lookback=args.days,
                output_path=args.output
            )
            if viz_path:
                print(f"Generated visualization at: {viz_path}")
    
    elif args.command == "report":
        # Generate report
        report_path = ts_perf.export_performance_report(
            model_id=args.model_id,
            hardware_id=args.hardware_id,
            days_lookback=args.days,
            format=args.format,
            output_path=args.output
        )
        
        if report_path:
            print(f"Generated report at: {report_path}")
        else:
            print("Failed to generate report")
    
    else:
        parser.print_help()