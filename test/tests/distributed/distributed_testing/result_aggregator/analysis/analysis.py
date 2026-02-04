#!/usr/bin/env python3
"""
Advanced Analysis Module for Result Aggregator

This module provides sophisticated statistical analysis capabilities for distributed testing results,
including correlation analysis, pattern recognition, comparative analysis, regression modeling,
statistical significance testing, and detailed benchmark comparisons.

Usage:
    from result_aggregator.analysis.analysis import AdvancedAnalysis
    
    analyzer = AdvancedAnalysis(db_path='path/to/benchmark_db.duckdb')
    
    # Perform correlation analysis
    correlations = analyzer.correlation_analysis('model_name', metrics=['latency', 'throughput'])
    
    # Detect performance patterns
    patterns = analyzer.detect_performance_patterns('hardware_type', time_period_days=30)
    
    # Compare test runs
    comparison = analyzer.compare_test_runs(run_id1, run_id2)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import math
from pathlib import Path

# Conditional imports for optional dependencies
try:
    from scipy import stats
    from scipy.signal import find_peaks
    from scipy.stats import linregress, ttest_ind, mannwhitneyu, pearsonr, spearmanr
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class AdvancedAnalysis:
    """
    Advanced statistical analysis for distributed testing results.
    
    This class provides methods for sophisticated analysis of test results:
    - Correlation analysis between metrics
    - Performance pattern recognition
    - Comparative analysis between test runs
    - Regression modeling for performance prediction
    - Statistical significance testing
    - Benchmark comparisons across hardware platforms
    """
    
    def __init__(self, db_path: str = None, connection = None):
        """
        Initialize the Advanced Analysis module.
        
        Args:
            db_path: Path to the DuckDB database file
            connection: Existing DuckDB connection (optional)
        
        Raises:
            ImportError: If required dependencies are not available
            ValueError: If neither db_path nor connection is provided
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is required for the AdvancedAnalysis module")
        
        if connection is None and db_path is None:
            raise ValueError("Either db_path or connection must be provided")
        
        self.db_path = db_path
        self._conn = connection if connection else duckdb.connect(db_path)
        self._validate_database_schema()
        
        # Log the availability of optional dependencies
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available. Some statistical functions will be limited.")
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available. Visualization will be disabled.")
    
    def _validate_database_schema(self):
        """
        Validate that the database has the required tables and schema.
        
        Raises:
            ValueError: If the database schema is not compatible
        """
        required_tables = [
            'test_results', 
            'performance_metrics',
            'hardware_platforms',
            'models',
            'test_runs'
        ]
        
        tables = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [table[0] for table in tables]
        
        for table in required_tables:
            if table not in table_names:
                raise ValueError(f"Required table '{table}' not found in database")
    
    def correlation_analysis(self, 
                            group_by: str, 
                            metrics: List[str],
                            filter_criteria: Dict[str, Any] = None,
                            method: str = 'pearson',
                            min_samples: int = 5,
                            output_format: str = 'dataframe',
                            visualize: bool = False,
                            output_path: Optional[str] = None) -> Union[pd.DataFrame, Dict]:
        """
        Perform correlation analysis between different performance metrics.
        
        Args:
            group_by: Column to group results by (e.g., 'model_name', 'hardware_type')
            metrics: List of metrics to analyze correlations between
            filter_criteria: Optional dictionary of filtering criteria
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            min_samples: Minimum number of samples required for correlation analysis
            output_format: Return format ('dataframe' or 'dict')
            visualize: Whether to generate visualization
            output_path: Path to save visualization
        
        Returns:
            DataFrame or dictionary with correlation matrices grouped by the specified column
        
        Raises:
            ValueError: If invalid parameters are provided
            RuntimeError: If correlation calculation fails
        """
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available. Using basic correlation method.")
            if method != 'pearson':
                raise ValueError("Only 'pearson' correlation is available without SciPy")
        
        if len(metrics) < 2:
            raise ValueError("At least two metrics are required for correlation analysis")
        
        # Build the query
        query = f"""
        SELECT r.{group_by}, m.metric_name, m.metric_value
        FROM test_results r
        JOIN performance_metrics m ON r.result_id = m.result_id
        WHERE m.metric_name IN ({', '.join([f"'{m}'" for m in metrics])})
        """
        
        # Add filter criteria if provided
        if filter_criteria:
            conditions = []
            for key, value in filter_criteria.items():
                if isinstance(value, list):
                    conditions.append(f"r.{key} IN ({', '.join([f"'{v}'" for v in value])})")
                else:
                    conditions.append(f"r.{key} = '{value}'")
            
            query += f" AND {' AND '.join(conditions)}"
        
        try:
            # Execute the query and convert to DataFrame
            result = self._conn.execute(query).fetchdf()
            
            if result.empty:
                logger.warning("No data found for the specified criteria")
                return pd.DataFrame() if output_format == 'dataframe' else {}
            
            # Pivot the data to get metrics as columns
            pivoted = result.pivot(index=group_by, columns='metric_name', values='metric_value')
            
            # Calculate correlation matrices for each group
            grouped = pivoted.groupby(level=0)
            correlations = {}
            
            for name, group in grouped:
                if len(group) < min_samples:
                    logger.warning(f"Group '{name}' has fewer than {min_samples} samples. Skipping.")
                    continue
                
                if method == 'pearson':
                    corr_matrix = group.corr(method='pearson')
                elif method == 'spearman' and SCIPY_AVAILABLE:
                    corr_matrix = group.corr(method='spearman')
                elif method == 'kendall' and SCIPY_AVAILABLE:
                    corr_matrix = group.corr(method='kendall')
                else:
                    raise ValueError(f"Unsupported correlation method: {method}")
                
                correlations[name] = corr_matrix
            
            # Generate visualization if requested
            if visualize and PLOTTING_AVAILABLE:
                self._visualize_correlation_matrices(correlations, method, output_path)
            
            if output_format == 'dataframe':
                # Convert to multi-index DataFrame
                dfs = []
                for name, corr in correlations.items():
                    corr[group_by] = name
                    dfs.append(corr)
                return pd.concat(dfs, keys=[d[group_by].iloc[0] for d in dfs])
            else:
                # Convert to dictionary format
                return {name: corr.to_dict() for name, corr in correlations.items()}
                
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            raise RuntimeError(f"Failed to perform correlation analysis: {str(e)}")
    
    def _visualize_correlation_matrices(self, 
                                       correlations: Dict[str, pd.DataFrame],
                                       method: str,
                                       output_path: Optional[str] = None):
        """
        Generate heatmap visualizations for correlation matrices.
        
        Args:
            correlations: Dictionary of correlation matrices
            method: Correlation method used
            output_path: Path to save the visualizations
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available. Skipping visualization.")
            return
        
        n_groups = len(correlations)
        if n_groups == 0:
            return
        
        # Determine the layout for subplots
        ncols = min(3, n_groups)
        nrows = (n_groups + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                                figsize=(5*ncols, 4*nrows),
                                squeeze=False)
        fig.suptitle(f"{method.capitalize()} Correlation Analysis", fontsize=16)
        
        # Create heatmaps for each group
        for i, (name, corr) in enumerate(correlations.items()):
            row, col = i // ncols, i % ncols
            ax = axes[row, col]
            
            mask = np.triu(np.ones_like(corr, dtype=bool))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                        annot=True, square=True, linewidths=.5, ax=ax)
            
            ax.set_title(f"Group: {name}")
        
        # Hide unused subplots
        for i in range(len(correlations), nrows * ncols):
            row, col = i // ncols, i % ncols
            axes[row, col].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved correlation visualization to {output_path}")
        else:
            plt.close()
    
    def detect_performance_patterns(self,
                                   group_by: str,
                                   metrics: List[str] = ['latency', 'throughput'],
                                   time_period_days: int = 30,
                                   min_data_points: int = 10,
                                   detection_methods: List[str] = ['trend', 'seasonality', 'outliers'],
                                   output_format: str = 'dict',
                                   visualize: bool = False,
                                   output_path: Optional[str] = None) -> Dict:
        """
        Detect performance patterns in time series data.
        
        Args:
            group_by: Column to group results by (e.g., 'model_name', 'hardware_type')
            metrics: List of metrics to analyze
            time_period_days: Number of days to look back
            min_data_points: Minimum number of data points required for analysis
            detection_methods: List of methods to use for pattern detection
            output_format: Return format ('dataframe' or 'dict')
            visualize: Whether to generate visualization
            output_path: Path to save visualization
        
        Returns:
            Dictionary with detected patterns for each group and metric
        
        Raises:
            ValueError: If invalid parameters are provided
            RuntimeError: If pattern detection fails
        """
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available. Pattern detection will be limited.")
            if 'seasonality' in detection_methods:
                detection_methods.remove('seasonality')
        
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        
        # Build the query
        query = f"""
        SELECT r.{group_by}, r.timestamp, m.metric_name, m.metric_value
        FROM test_results r
        JOIN performance_metrics m ON r.result_id = m.result_id
        WHERE r.timestamp >= '{cutoff_str}'
        AND m.metric_name IN ({', '.join([f"'{m}'" for m in metrics])})
        ORDER BY r.{group_by}, r.timestamp
        """
        
        try:
            # Execute the query and convert to DataFrame
            result = self._conn.execute(query).fetchdf()
            
            if result.empty:
                logger.warning("No data found for the specified time period")
                return {}
            
            # Ensure timestamp is in datetime format
            result['timestamp'] = pd.to_datetime(result['timestamp'])
            
            # Group by the specified column and metric
            groups = result.groupby([group_by, 'metric_name'])
            
            patterns = {}
            
            for (group_name, metric_name), group_data in groups:
                if len(group_data) < min_data_points:
                    logger.warning(
                        f"Group '{group_name}' with metric '{metric_name}' has fewer than "
                        f"{min_data_points} data points. Skipping."
                    )
                    continue
                
                # Sort by timestamp and set as index
                ts_data = group_data.sort_values('timestamp')
                ts_data = ts_data.set_index('timestamp')['metric_value']
                
                # Initialize pattern results
                if group_name not in patterns:
                    patterns[group_name] = {}
                patterns[group_name][metric_name] = {}
                
                # Detect trends
                if 'trend' in detection_methods:
                    trend_results = self._detect_trend(ts_data)
                    patterns[group_name][metric_name]['trend'] = trend_results
                
                # Detect seasonality
                if 'seasonality' in detection_methods and SCIPY_AVAILABLE:
                    try:
                        seasonality_results = self._detect_seasonality(ts_data)
                        patterns[group_name][metric_name]['seasonality'] = seasonality_results
                    except Exception as e:
                        logger.warning(f"Failed to detect seasonality: {str(e)}")
                        patterns[group_name][metric_name]['seasonality'] = {'detected': False, 'error': str(e)}
                
                # Detect outliers
                if 'outliers' in detection_methods:
                    outlier_results = self._detect_outliers(ts_data)
                    patterns[group_name][metric_name]['outliers'] = outlier_results
                
                # Generate forecasts if ARIMA is available
                if 'forecast' in detection_methods and SCIPY_AVAILABLE:
                    try:
                        forecast_results = self._generate_forecast(ts_data)
                        patterns[group_name][metric_name]['forecast'] = forecast_results
                    except Exception as e:
                        logger.warning(f"Failed to generate forecast: {str(e)}")
                        patterns[group_name][metric_name]['forecast'] = {'success': False, 'error': str(e)}
            
            # Generate visualizations if requested
            if visualize and PLOTTING_AVAILABLE:
                self._visualize_patterns(result, patterns, group_by, output_path)
            
            return patterns
                
        except Exception as e:
            logger.error(f"Error in pattern detection: {str(e)}")
            raise RuntimeError(f"Failed to detect performance patterns: {str(e)}")
    
    def _detect_trend(self, time_series: pd.Series) -> Dict:
        """
        Detect trends in time series data using linear regression.
        
        Args:
            time_series: Time series data
        
        Returns:
            Dictionary with trend analysis results
        """
        # Convert timestamps to numeric values for regression
        x = np.arange(len(time_series))
        y = time_series.values
        
        # Perform linear regression
        if SCIPY_AVAILABLE:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            # Determine trend direction and significance
            if p_value < 0.05:
                if slope > 0:
                    trend_direction = 'increasing'
                else:
                    trend_direction = 'decreasing'
                significance = 'significant'
            else:
                if abs(slope) < 0.001:
                    trend_direction = 'stable'
                elif slope > 0:
                    trend_direction = 'slightly increasing'
                else:
                    trend_direction = 'slightly decreasing'
                significance = 'not significant'
            
            return {
                'detected': True,
                'direction': trend_direction,
                'slope': slope,
                'p_value': p_value,
                'r_squared': r_value**2,
                'significance': significance,
                'percent_change': (time_series.iloc[-1] - time_series.iloc[0]) / time_series.iloc[0] * 100 
                                if time_series.iloc[0] != 0 else float('inf')
            }
        else:
            # Basic trend detection without scipy
            start_value = time_series.iloc[0]
            end_value = time_series.iloc[-1]
            
            if start_value == 0:
                percent_change = float('inf') if end_value > 0 else 0
            else:
                percent_change = (end_value - start_value) / start_value * 100
            
            if abs(percent_change) < 5:
                trend_direction = 'stable'
            elif percent_change > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'
            
            return {
                'detected': True,
                'direction': trend_direction,
                'percent_change': percent_change
            }
    
    def _detect_seasonality(self, time_series: pd.Series) -> Dict:
        """
        Detect seasonality in time series data.
        
        Args:
            time_series: Time series data
        
        Returns:
            Dictionary with seasonality analysis results
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for seasonality detection")
        
        # Ensure the time series is evenly spaced
        # For simplicity, we'll resample to daily frequency if timestamps are available
        if isinstance(time_series.index, pd.DatetimeIndex):
            time_series = time_series.resample('D').mean().interpolate()
        
        # Check if we have enough data points for decomposition
        if len(time_series) < 14:  # Need at least 2x the seasonal period
            return {'detected': False, 'reason': 'insufficient_data'}
        
        try:
            # Determine the optimal period
            acf_values = acf(time_series.dropna(), nlags=len(time_series)//3)
            peaks, _ = find_peaks(acf_values)
            
            if len(peaks) > 0:
                # Use the first significant peak as the period
                period = peaks[0]
                if period < 2:  # Ensure reasonable period
                    period = 7  # Default to weekly
            else:
                period = 7  # Default to weekly seasonality
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                time_series.dropna(), 
                model='additive', 
                period=period
            )
            
            # Calculate the strength of seasonality
            seasonal_strength = np.std(decomposition.seasonal) / np.std(time_series.dropna())
            
            # Determine if seasonality is significant
            is_seasonal = seasonal_strength > 0.1
            
            return {
                'detected': is_seasonal,
                'period': period,
                'strength': seasonal_strength,
                'significance': 'significant' if is_seasonal else 'not significant'
            }
            
        except Exception as e:
            logger.warning(f"Error in seasonality detection: {str(e)}")
            return {'detected': False, 'error': str(e)}
    
    def _detect_outliers(self, time_series: pd.Series) -> Dict:
        """
        Detect outliers in time series data using Z-scores.
        
        Args:
            time_series: Time series data
        
        Returns:
            Dictionary with outlier detection results
        """
        # Calculate Z-scores
        mean = time_series.mean()
        std = time_series.std()
        z_scores = (time_series - mean) / std if std > 0 else pd.Series(0, index=time_series.index)
        
        # Identify outliers (Z-score > 3 or < -3)
        outliers = time_series[abs(z_scores) > 3]
        
        if len(outliers) > 0:
            outlier_info = []
            for idx, value in outliers.items():
                z = z_scores[idx]
                outlier_info.append({
                    'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S') if hasattr(idx, 'strftime') else str(idx),
                    'value': float(value),
                    'z_score': float(z),
                    'type': 'high' if z > 0 else 'low'
                })
            
            return {
                'detected': True,
                'count': len(outliers),
                'percentage': len(outliers) / len(time_series) * 100,
                'outliers': outlier_info
            }
        else:
            return {
                'detected': False,
                'count': 0,
                'percentage': 0
            }
    
    def _generate_forecast(self, time_series: pd.Series, 
                          forecast_periods: int = 7) -> Dict:
        """
        Generate forecasts using ARIMA models.
        
        Args:
            time_series: Time series data
            forecast_periods: Number of periods to forecast
        
        Returns:
            Dictionary with forecast results
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy and statsmodels are required for forecasting")
        
        # Ensure we have enough data
        if len(time_series) < 10:
            return {
                'success': False,
                'error': 'Insufficient data for forecasting'
            }
        
        try:
            # Fit ARIMA model
            model = ARIMA(time_series, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(steps=forecast_periods)
            
            return {
                'success': True,
                'forecast': forecast.tolist(),
                'confidence_intervals': {
                    'lower': model_fit.get_forecast(steps=forecast_periods).conf_int().iloc[:, 0].tolist(),
                    'upper': model_fit.get_forecast(steps=forecast_periods).conf_int().iloc[:, 1].tolist()
                },
                'model_info': {
                    'aic': model_fit.aic,
                    'bic': model_fit.bic
                }
            }
            
        except Exception as e:
            logger.warning(f"Error generating forecast: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _visualize_patterns(self, 
                           data: pd.DataFrame, 
                           patterns: Dict, 
                           group_by: str,
                           output_path: Optional[str] = None):
        """
        Visualize detected patterns.
        
        Args:
            data: Raw data DataFrame
            patterns: Dictionary with detected patterns
            group_by: Column used for grouping
            output_path: Path to save visualizations
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping visualization.")
            return
        
        for group_name, metrics in patterns.items():
            for metric_name, pattern_results in metrics.items():
                # Create a figure with subplots for each analysis type
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f"Performance Patterns: {group_name} - {metric_name}", fontsize=16)
                
                # Filter data for this group and metric
                group_data = data[(data[group_by] == group_name) & 
                                (data['metric_name'] == metric_name)]
                
                # Sort by timestamp
                group_data = group_data.sort_values('timestamp')
                
                # Time series plot with trend (top left)
                ax_ts = axes[0, 0]
                ax_ts.plot(group_data['timestamp'], group_data['metric_value'], 'b-', label='Actual')
                ax_ts.set_title('Time Series with Trend')
                ax_ts.set_xlabel('Time')
                ax_ts.set_ylabel(metric_name)
                
                # Add trend line if available
                if 'trend' in pattern_results and pattern_results['trend']['detected']:
                    x = np.arange(len(group_data))
                    if 'slope' in pattern_results['trend'] and 'intercept' in pattern_results['trend']:
                        slope = pattern_results['trend']['slope']
                        intercept = pattern_results['trend']['intercept']
                        trend_line = intercept + slope * x
                        ax_ts.plot(group_data['timestamp'], trend_line, 'r--', 
                                 label=f"Trend ({pattern_results['trend']['direction']})")
                    
                ax_ts.legend()
                
                # Outliers plot (top right)
                ax_out = axes[0, 1]
                ax_out.plot(group_data['timestamp'], group_data['metric_value'], 'b-')
                ax_out.set_title('Outlier Detection')
                ax_out.set_xlabel('Time')
                ax_out.set_ylabel(metric_name)
                
                # Highlight outliers if detected
                if 'outliers' in pattern_results and pattern_results['outliers']['detected']:
                    outliers = pattern_results['outliers'].get('outliers', [])
                    outlier_timestamps = [datetime.strptime(o['timestamp'], '%Y-%m-%d %H:%M:%S') 
                                        for o in outliers]
                    outlier_values = [o['value'] for o in outliers]
                    
                    if outlier_timestamps and outlier_values:
                        ax_out.scatter(outlier_timestamps, outlier_values, color='red', 
                                     marker='o', s=80, label='Outliers')
                        ax_out.legend()
                
                # Seasonality plot (bottom left)
                ax_seas = axes[1, 0]
                if 'seasonality' in pattern_results and pattern_results['seasonality'].get('detected', False):
                    # We'd need to re-run the decomposition to plot it
                    ax_seas.text(0.5, 0.5, 
                               f"Seasonality detected\nPeriod: {pattern_results['seasonality']['period']}\n"
                               f"Strength: {pattern_results['seasonality']['strength']:.2f}",
                               horizontalalignment='center', verticalalignment='center',
                               transform=ax_seas.transAxes, fontsize=12)
                else:
                    ax_seas.text(0.5, 0.5, "No significant seasonality detected",
                               horizontalalignment='center', verticalalignment='center',
                               transform=ax_seas.transAxes, fontsize=12)
                ax_seas.set_title('Seasonality Analysis')
                ax_seas.set_xticks([])
                ax_seas.set_yticks([])
                
                # Forecast plot (bottom right)
                ax_forecast = axes[1, 1]
                if 'forecast' in pattern_results and pattern_results['forecast'].get('success', False):
                    # Get the last timestamp
                    last_timestamp = group_data['timestamp'].iloc[-1]
                    
                    # Create forecast timestamps
                    if isinstance(last_timestamp, pd.Timestamp):
                        # If timestamps are datetime, extend with appropriate frequency
                        forecast_timestamps = pd.date_range(
                            start=last_timestamp, 
                            periods=len(pattern_results['forecast']['forecast'])+1,
                            freq='D'
                        )[1:]
                    else:
                        # Otherwise, just use indices
                        forecast_timestamps = range(
                            len(group_data),
                            len(group_data) + len(pattern_results['forecast']['forecast'])
                        )
                    
                    # Plot actual data
                    ax_forecast.plot(group_data['timestamp'], group_data['metric_value'], 
                                   'b-', label='Actual')
                    
                    # Plot forecast
                    ax_forecast.plot(forecast_timestamps, 
                                   pattern_results['forecast']['forecast'],
                                   'g--', label='Forecast')
                    
                    # Plot confidence intervals if available
                    if 'confidence_intervals' in pattern_results['forecast']:
                        lower = pattern_results['forecast']['confidence_intervals']['lower']
                        upper = pattern_results['forecast']['confidence_intervals']['upper']
                        ax_forecast.fill_between(forecast_timestamps, lower, upper,
                                              color='g', alpha=0.2, label='95% Confidence')
                    
                    ax_forecast.legend()
                else:
                    ax_forecast.text(0.5, 0.5, "Forecast not available",
                                  horizontalalignment='center', verticalalignment='center',
                                  transform=ax_forecast.transAxes, fontsize=12)
                
                ax_forecast.set_title('Performance Forecast')
                ax_forecast.set_xlabel('Time')
                ax_forecast.set_ylabel(metric_name)
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                
                # Save or display the plot
                if output_path:
                    # Create a specific filename for each group and metric
                    file_name = f"patterns_{group_name}_{metric_name}.png"
                    file_path = os.path.join(output_path, file_name) if os.path.isdir(output_path) else output_path
                    
                    output_dir = os.path.dirname(file_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                        
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Saved pattern visualization to {file_path}")
                
                plt.close()
    
    def compare_test_runs(self, 
                         run_id1: Union[str, int], 
                         run_id2: Union[str, int],
                         metrics: List[str] = None,
                         significance_level: float = 0.05,
                         output_format: str = 'dict',
                         visualize: bool = False,
                         output_path: Optional[str] = None) -> Union[Dict, pd.DataFrame]:
        """
        Compare performance between two test runs.
        
        Args:
            run_id1: ID of the first test run
            run_id2: ID of the second test run
            metrics: List of metrics to compare (if None, all metrics are compared)
            significance_level: Statistical significance level
            output_format: Return format ('dataframe' or 'dict')
            visualize: Whether to generate visualization
            output_path: Path to save visualization
        
        Returns:
            Dictionary or DataFrame with comparison results
        
        Raises:
            ValueError: If invalid parameters are provided
            RuntimeError: If comparison fails
        """
        # Validate inputs
        if not isinstance(run_id1, (str, int)) or not isinstance(run_id2, (str, int)):
            raise ValueError("Run IDs must be strings or integers")
        
        # Build the query to get run information
        runs_query = f"""
        SELECT run_id, run_name, start_time, end_time, status, total_tests
        FROM test_runs
        WHERE run_id IN ('{run_id1}', '{run_id2}')
        """
        
        try:
            run_info = self._conn.execute(runs_query).fetchdf()
            
            if len(run_info) != 2:
                missing_runs = []
                if run_id1 not in run_info['run_id'].values:
                    missing_runs.append(str(run_id1))
                if run_id2 not in run_info['run_id'].values:
                    missing_runs.append(str(run_id2))
                raise ValueError(f"Run IDs not found: {', '.join(missing_runs)}")
            
            # Build the metrics query
            metrics_clause = ""
            if metrics:
                metrics_str = ', '.join([f"'{m}'" for m in metrics])
                metrics_clause = f"AND m.metric_name IN ({metrics_str})"
            
            metrics_query = f"""
            SELECT r.run_id, r.result_id, r.model_name, r.hardware_type, r.test_name, 
                  m.metric_name, m.metric_value
            FROM test_results r
            JOIN performance_metrics m ON r.result_id = m.result_id
            WHERE r.run_id IN ('{run_id1}', '{run_id2}')
            {metrics_clause}
            """
            
            # Execute the metrics query
            results = self._conn.execute(metrics_query).fetchdf()
            
            if results.empty:
                logger.warning("No metrics found for the specified run IDs")
                return pd.DataFrame() if output_format == 'dataframe' else {}
            
            # Prepare run metadata
            run1_info = run_info[run_info['run_id'] == run_id1].iloc[0]
            run2_info = run_info[run_info['run_id'] == run_id2].iloc[0]
            
            run_metadata = {
                'run1': {
                    'run_id': run_id1,
                    'run_name': run1_info['run_name'],
                    'start_time': run1_info['start_time'],
                    'end_time': run1_info['end_time'],
                    'status': run1_info['status'],
                    'total_tests': run1_info['total_tests']
                },
                'run2': {
                    'run_id': run_id2,
                    'run_name': run2_info['run_name'],
                    'start_time': run2_info['start_time'],
                    'end_time': run2_info['end_time'],
                    'status': run2_info['status'],
                    'total_tests': run2_info['total_tests']
                }
            }
            
            # Group metrics by model, hardware, and test
            grouped = results.groupby(['model_name', 'hardware_type', 'test_name', 'metric_name'])
            
            comparison_results = {
                'metadata': run_metadata,
                'metrics': {}
            }
            
            # Process each group
            for (model, hardware, test, metric), group in grouped:
                if len(group) != 2:
                    # Skip if we don't have data for both runs
                    continue
                
                # Get values for each run
                run1_values = group[group['run_id'] == run_id1]['metric_value']
                run2_values = group[group['run_id'] == run_id2]['metric_value']
                
                # For single values, just calculate percent change
                if len(run1_values) == 1 and len(run2_values) == 1:
                    val1 = run1_values.iloc[0]
                    val2 = run2_values.iloc[0]
                    
                    if val1 == 0:
                        percent_change = float('inf') if val2 > 0 else 0.0
                    else:
                        percent_change = (val2 - val1) / val1 * 100
                    
                    # Determine improvement or regression
                    # For latency, lower is better; for throughput, higher is better
                    if metric.lower() in ['latency', 'response_time', 'initialization_time']:
                        is_improvement = percent_change < 0
                    else:  # Assume higher is better for other metrics
                        is_improvement = percent_change > 0
                    
                    change_type = 'improvement' if is_improvement else 'regression'
                    
                    # Calculate significance if we have statistical tests available
                    p_value = None
                    significance = None
                    
                    # Create the result entry
                    metric_key = f"{model}_{hardware}_{test}_{metric}"
                    comparison_results['metrics'][metric_key] = {
                        'model': model,
                        'hardware': hardware,
                        'test': test,
                        'metric': metric,
                        'run1_value': float(val1),
                        'run2_value': float(val2),
                        'absolute_change': float(val2 - val1),
                        'percent_change': float(percent_change),
                        'change_type': change_type,
                        'p_value': p_value,
                        'significance': significance
                    }
                    
                # For multiple values, perform statistical tests if available
                elif len(run1_values) > 1 and len(run2_values) > 1 and SCIPY_AVAILABLE:
                    # Convert to numpy arrays
                    vals1 = run1_values.values
                    vals2 = run2_values.values
                    
                    # Calculate basic statistics
                    mean1 = np.mean(vals1)
                    mean2 = np.mean(vals2)
                    std1 = np.std(vals1)
                    std2 = np.std(vals2)
                    
                    # Calculate percent change
                    if mean1 == 0:
                        percent_change = float('inf') if mean2 > 0 else 0.0
                    else:
                        percent_change = (mean2 - mean1) / mean1 * 100
                    
                    # Determine improvement or regression
                    if metric.lower() in ['latency', 'response_time', 'initialization_time']:
                        is_improvement = percent_change < 0
                    else:  # Assume higher is better for other metrics
                        is_improvement = percent_change > 0
                    
                    change_type = 'improvement' if is_improvement else 'regression'
                    
                    # Perform statistical test
                    try:
                        # Use t-test if data is normally distributed
                        _, p_value = ttest_ind(vals1, vals2, equal_var=False)
                        
                        # Determine significance
                        significance = 'significant' if p_value < significance_level else 'not significant'
                    except:
                        # Fallback to non-parametric test
                        try:
                            _, p_value = mannwhitneyu(vals1, vals2)
                            significance = 'significant' if p_value < significance_level else 'not significant'
                        except:
                            p_value = None
                            significance = None
                    
                    # Create the result entry
                    metric_key = f"{model}_{hardware}_{test}_{metric}"
                    comparison_results['metrics'][metric_key] = {
                        'model': model,
                        'hardware': hardware,
                        'test': test,
                        'metric': metric,
                        'run1_mean': float(mean1),
                        'run1_std': float(std1),
                        'run2_mean': float(mean2),
                        'run2_std': float(std2),
                        'absolute_change': float(mean2 - mean1),
                        'percent_change': float(percent_change),
                        'change_type': change_type,
                        'p_value': float(p_value) if p_value is not None else None,
                        'significance': significance,
                        'sample_size_run1': len(vals1),
                        'sample_size_run2': len(vals2)
                    }
            
            # Generate visualization if requested
            if visualize and PLOTTING_AVAILABLE:
                self._visualize_comparison(comparison_results, output_path)
            
            # Return in the requested format
            if output_format == 'dataframe':
                metrics_df = pd.DataFrame.from_dict(comparison_results['metrics'], orient='index')
                return metrics_df
            else:
                return comparison_results
                
        except Exception as e:
            logger.error(f"Error comparing test runs: {str(e)}")
            raise RuntimeError(f"Failed to compare test runs: {str(e)}")
    
    def _visualize_comparison(self, 
                            comparison_results: Dict,
                            output_path: Optional[str] = None):
        """
        Visualize test run comparison results.
        
        Args:
            comparison_results: Comparison results dictionary
            output_path: Path to save visualizations
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping visualization.")
            return
        
        # Extract metadata
        run1_name = comparison_results['metadata']['run1'].get('run_name', f"Run {comparison_results['metadata']['run1']['run_id']}")
        run2_name = comparison_results['metadata']['run2'].get('run_name', f"Run {comparison_results['metadata']['run2']['run_id']}")
        
        # Group metrics by type
        metrics_by_type = {}
        for metric_key, metric_data in comparison_results['metrics'].items():
            metric_name = metric_data['metric']
            if metric_name not in metrics_by_type:
                metrics_by_type[metric_name] = []
            metrics_by_type[metric_name].append(metric_data)
        
        # Create a separate figure for each metric type
        for metric_name, metrics in metrics_by_type.items():
            # Determine the number of comparisons
            n_comparisons = len(metrics)
            
            if n_comparisons == 0:
                continue
            
            # Set up the figure
            fig, ax = plt.subplots(figsize=(12, max(6, n_comparisons * 0.5)))
            
            # Prepare data for the plot
            labels = []
            values1 = []
            values2 = []
            percent_changes = []
            colors = []
            
            for metric in metrics:
                # Create label
                label = f"{metric['model']} - {metric['hardware']} - {metric['test']}"
                labels.append(label)
                
                # Get values
                if 'run1_mean' in metric:
                    values1.append(metric['run1_mean'])
                    values2.append(metric['run2_mean'])
                else:
                    values1.append(metric['run1_value'])
                    values2.append(metric['run2_value'])
                
                # Get percent change
                percent_changes.append(metric['percent_change'])
                
                # Determine color based on change type and significance
                if metric['change_type'] == 'improvement':
                    if metric.get('significance') == 'significant':
                        colors.append('darkgreen')
                    else:
                        colors.append('lightgreen')
                else:  # regression
                    if metric.get('significance') == 'significant':
                        colors.append('darkred')
                    else:
                        colors.append('salmon')
            
            # Plot the percent changes
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, percent_changes, color=colors)
            
            # Add zero line
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            # Add labels and title
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Percent Change (%)')
            ax.set_title(f"Comparison of {metric_name} - {run1_name} vs {run2_name}")
            
            # Add a legend
            import matplotlib.patches as mpatches
            legend_items = [
                mpatches.Patch(color='darkgreen', label='Significant Improvement'),
                mpatches.Patch(color='lightgreen', label='Non-significant Improvement'),
                mpatches.Patch(color='darkred', label='Significant Regression'),
                mpatches.Patch(color='salmon', label='Non-significant Regression')
            ]
            ax.legend(handles=legend_items, loc='best')
            
            # Add value annotations
            for i, (val1, val2, pct) in enumerate(zip(values1, values2, percent_changes)):
                # Format the values
                if abs(val1) < 0.01 or abs(val2) < 0.01:
                    val_str = f"{val1:.2e} → {val2:.2e}"
                else:
                    val_str = f"{val1:.2f} → {val2:.2f}"
                
                # Add the annotation
                if pct >= 0:
                    ax.text(max(pct + 2, 2), i, val_str, va='center')
                else:
                    ax.text(min(pct - 2, -2), i, val_str, va='center', ha='right')
            
            plt.tight_layout()
            
            # Save or display the plot
            if output_path:
                # Create a specific filename for each metric
                file_name = f"comparison_{metric_name}.png"
                file_path = os.path.join(output_path, file_name) if os.path.isdir(output_path) else output_path
                
                output_dir = os.path.dirname(file_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved comparison visualization to {file_path}")
            
            plt.close()
    
    def predict_performance(self,
                          model_name: str,
                          hardware_type: str,
                          batch_size: int = 1,
                          metrics: List[str] = ['latency', 'throughput'],
                          input_params: Dict[str, Any] = None,
                          confidence_level: float = 0.95,
                          output_format: str = 'dict',
                          visualize: bool = False,
                          output_path: Optional[str] = None) -> Union[Dict, pd.DataFrame]:
        """
        Predict performance metrics based on historical data.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware
            batch_size: Batch size for prediction
            metrics: List of metrics to predict
            input_params: Additional parameters for prediction
            confidence_level: Confidence level for prediction intervals
            output_format: Return format ('dataframe' or 'dict')
            visualize: Whether to generate visualization
            output_path: Path to save visualization
        
        Returns:
            Dictionary or DataFrame with predicted performance metrics
        
        Raises:
            ValueError: If invalid parameters are provided
            RuntimeError: If prediction fails
        """
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available. Prediction accuracy will be limited.")
        
        # Validate inputs
        if not isinstance(model_name, str) or not isinstance(hardware_type, str):
            raise ValueError("Model name and hardware type must be strings")
        
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        
        if confidence_level <= 0 or confidence_level >= 1:
            raise ValueError("Confidence level must be between 0 and 1")
        
        # Build the query to get historical data
        where_clauses = [
            f"r.model_name = '{model_name}'",
            f"r.hardware_type = '{hardware_type}'",
            f"r.batch_size = {batch_size}"
        ]
        
        # Add additional filters from input_params
        if input_params:
            for param, value in input_params.items():
                if isinstance(value, str):
                    where_clauses.append(f"r.{param} = '{value}'")
                else:
                    where_clauses.append(f"r.{param} = {value}")
        
        where_clause = " AND ".join(where_clauses)
        metrics_str = ", ".join([f"'{m}'" for m in metrics])
        
        query = f"""
        SELECT r.result_id, r.timestamp, m.metric_name, m.metric_value
        FROM test_results r
        JOIN performance_metrics m ON r.result_id = m.result_id
        WHERE {where_clause}
        AND m.metric_name IN ({metrics_str})
        ORDER BY r.timestamp
        """
        
        try:
            # Execute the query
            results = self._conn.execute(query).fetchdf()
            
            if results.empty:
                logger.warning("No historical data found for the specified criteria")
                return pd.DataFrame() if output_format == 'dataframe' else {}
            
            # Ensure timestamp is datetime
            results['timestamp'] = pd.to_datetime(results['timestamp'])
            
            # Group by metric
            predictions = {}
            
            for metric in metrics:
                metric_data = results[results['metric_name'] == metric]
                
                if metric_data.empty:
                    logger.warning(f"No historical data found for metric '{metric}'")
                    continue
                
                # Convert to time series
                metric_data = metric_data.sort_values('timestamp')
                time_series = metric_data.set_index('timestamp')['metric_value']
                
                # Calculate basic statistics
                mean_value = time_series.mean()
                std_value = time_series.std()
                min_value = time_series.min()
                max_value = time_series.max()
                
                # Calculate confidence intervals
                if SCIPY_AVAILABLE:
                    # Calculate the z-value for the given confidence level
                    z_value = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                    margin_of_error = z_value * (std_value / np.sqrt(len(time_series)))
                    lower_bound = mean_value - margin_of_error
                    upper_bound = mean_value + margin_of_error
                else:
                    # Simple approximation
                    margin_of_error = 2 * (std_value / np.sqrt(len(time_series)))
                    lower_bound = mean_value - margin_of_error
                    upper_bound = mean_value + margin_of_error
                
                # Use linear regression for prediction if enough data points
                if len(time_series) >= 5 and SCIPY_AVAILABLE:
                    # Convert dates to numeric values for regression
                    time_values = np.array([(ts - time_series.index[0]).total_seconds() 
                                          for ts in time_series.index])
                    
                    # Perform linear regression
                    slope, intercept, r_value, p_value, std_err = linregress(
                        time_values, time_series.values
                    )
                    
                    # Calculate the latest value based on the trend
                    latest_time = (time_series.index[-1] - time_series.index[0]).total_seconds()
                    trend_prediction = intercept + slope * latest_time
                    
                    # Calculate trend strength
                    trend_strength = abs(r_value)
                    
                    # Determine prediction confidence
                    if p_value < 0.05 and trend_strength > 0.7:
                        prediction_confidence = 'high'
                    elif p_value < 0.1 and trend_strength > 0.5:
                        prediction_confidence = 'medium'
                    else:
                        prediction_confidence = 'low'
                    
                    # Store prediction
                    predictions[metric] = {
                        'predicted_value': float(trend_prediction),
                        'confidence_interval': {
                            'lower': float(lower_bound),
                            'upper': float(upper_bound),
                            'confidence_level': confidence_level
                        },
                        'historical_data': {
                            'mean': float(mean_value),
                            'std': float(std_value),
                            'min': float(min_value),
                            'max': float(max_value),
                            'samples': len(time_series)
                        },
                        'trend_analysis': {
                            'slope': float(slope),
                            'r_squared': float(r_value**2),
                            'p_value': float(p_value),
                            'trend_strength': float(trend_strength),
                            'prediction_confidence': prediction_confidence
                        }
                    }
                else:
                    # Simple prediction based on mean
                    predictions[metric] = {
                        'predicted_value': float(mean_value),
                        'confidence_interval': {
                            'lower': float(lower_bound),
                            'upper': float(upper_bound),
                            'confidence_level': confidence_level
                        },
                        'historical_data': {
                            'mean': float(mean_value),
                            'std': float(std_value),
                            'min': float(min_value),
                            'max': float(max_value),
                            'samples': len(time_series)
                        }
                    }
            
            # Create the final prediction result
            prediction_result = {
                'model_name': model_name,
                'hardware_type': hardware_type,
                'batch_size': batch_size,
                'predictions': predictions,
                'input_params': input_params,
                'timestamp': datetime.now().isoformat()
            }
            
            # Generate visualization if requested
            if visualize and PLOTTING_AVAILABLE:
                self._visualize_predictions(results, prediction_result, output_path)
            
            # Return in the requested format
            if output_format == 'dataframe':
                # Convert nested dictionary to DataFrame
                pred_df = pd.DataFrame.from_dict(
                    {(k, k2): v2 for k, v in predictions.items() for k2, v2 in v.items()},
                    orient='index'
                )
                return pred_df
            else:
                return prediction_result
                
        except Exception as e:
            logger.error(f"Error predicting performance: {str(e)}")
            raise RuntimeError(f"Failed to predict performance: {str(e)}")
    
    def _visualize_predictions(self, 
                             historical_data: pd.DataFrame,
                             prediction_result: Dict,
                             output_path: Optional[str] = None):
        """
        Visualize performance predictions.
        
        Args:
            historical_data: DataFrame with historical data
            prediction_result: Dictionary with prediction results
            output_path: Path to save visualizations
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping visualization.")
            return
        
        # Create a separate figure for each metric
        for metric, prediction in prediction_result['predictions'].items():
            # Filter data for this metric
            metric_data = historical_data[historical_data['metric_name'] == metric]
            
            if metric_data.empty:
                continue
            
            # Sort by timestamp
            metric_data = metric_data.sort_values('timestamp')
            
            # Set up the figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            ax.plot(metric_data['timestamp'], metric_data['metric_value'], 
                  'bo-', label='Historical Values')
            
            # Plot prediction
            predicted_value = prediction['predicted_value']
            lower_bound = prediction['confidence_interval']['lower']
            upper_bound = prediction['confidence_interval']['upper']
            
            # Add prediction point
            latest_timestamp = metric_data['timestamp'].max()
            prediction_timestamp = latest_timestamp + pd.Timedelta(days=1)
            ax.plot(prediction_timestamp, predicted_value, 'ro', markersize=10,
                  label='Predicted Value')
            
            # Add confidence interval
            ax.axhspan(lower_bound, upper_bound, alpha=0.2, color='red',
                     label=f"{prediction['confidence_interval']['confidence_level']*100}% Confidence Interval")
            
            # Add trend line if available
            if 'trend_analysis' in prediction:
                # Convert dates to numeric values for regression
                time_values = np.array([(ts - metric_data['timestamp'].min()).total_seconds() 
                                      for ts in metric_data['timestamp']])
                
                # Get trend parameters
                slope = prediction['trend_analysis']['slope']
                intercept = prediction['trend_analysis']['r_squared']
                
                # Calculate trend line
                trend_line = intercept + slope * time_values
                
                # Plot trend line
                ax.plot(metric_data['timestamp'], trend_line, 'r--', 
                      label='Trend Line')
                
                # Add trend info to title
                trend_info = (
                    f"Trend Slope: {slope:.6f}, "
                    f"R²: {prediction['trend_analysis']['r_squared']:.2f}, "
                    f"Confidence: {prediction['trend_analysis']['prediction_confidence']}"
                )
                plt.title(f"Performance Prediction for {metric}\n{trend_info}")
            else:
                plt.title(f"Performance Prediction for {metric}")
            
            # Add labels and grid
            ax.set_xlabel('Time')
            ax.set_ylabel(metric)
            ax.grid(True)
            
            # Add annotation with prediction details
            annotation_text = (
                f"Predicted Value: {predicted_value:.4f}\n"
                f"Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]\n"
                f"Historical Mean: {prediction['historical_data']['mean']:.4f}\n"
                f"Standard Deviation: {prediction['historical_data']['std']:.4f}\n"
                f"Sample Size: {prediction['historical_data']['samples']}"
            )
            
            # Position the annotation in the top left
            ax.annotate(annotation_text, xy=(0.02, 0.98), xycoords='axes fraction',
                       va='top', ha='left', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
            
            # Add legend
            ax.legend()
            
            plt.tight_layout()
            
            # Save or display the plot
            if output_path:
                # Create a specific filename for each metric
                file_name = f"prediction_{prediction_result['model_name']}_{prediction_result['hardware_type']}_{metric}.png"
                file_path = os.path.join(output_path, file_name) if os.path.isdir(output_path) else output_path
                
                output_dir = os.path.dirname(file_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved prediction visualization to {file_path}")
            
            plt.close()
    
    def benchmark_comparison(self,
                            model_types: List[str] = None,
                            hardware_platforms: List[str] = None,
                            metrics: List[str] = ['latency', 'throughput', 'memory_usage'],
                            batch_sizes: List[int] = None,
                            time_period_days: int = 30,
                            normalization: str = 'min-max',
                            output_format: str = 'dict',
                            visualize: bool = False,
                            output_path: Optional[str] = None) -> Union[Dict, pd.DataFrame]:
        """
        Perform comprehensive benchmark comparison across hardware platforms.
        
        Args:
            model_types: List of model types to include
            hardware_platforms: List of hardware platforms to compare
            metrics: List of metrics to include
            batch_sizes: List of batch sizes to include
            time_period_days: Number of days to look back
            normalization: Method for normalizing metrics ('min-max', 'z-score', or None)
            output_format: Return format ('dataframe' or 'dict')
            visualize: Whether to generate visualization
            output_path: Path to save visualization
        
        Returns:
            Dictionary or DataFrame with benchmark comparison results
        
        Raises:
            ValueError: If invalid parameters are provided
            RuntimeError: If comparison fails
        """
        # Calculate the cutoff date
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        
        # Build the base query
        query = """
        SELECT r.model_name, r.hardware_type, r.batch_size, r.timestamp,
               m.metric_name, m.metric_value
        FROM test_results r
        JOIN performance_metrics m ON r.result_id = m.result_id
        WHERE r.timestamp >= ?
        """
        
        # Add filters
        params = [cutoff_str]
        
        if model_types:
            model_types_str = ', '.join([f"'{m}'" for m in model_types])
            query += f" AND r.model_type IN ({model_types_str})"
        
        if hardware_platforms:
            hw_platforms_str = ', '.join([f"'{h}'" for h in hardware_platforms])
            query += f" AND r.hardware_type IN ({hw_platforms_str})"
        
        if batch_sizes:
            batch_sizes_str = ', '.join([str(b) for b in batch_sizes])
            query += f" AND r.batch_size IN ({batch_sizes_str})"
        
        if metrics:
            metrics_str = ', '.join([f"'{m}'" for m in metrics])
            query += f" AND m.metric_name IN ({metrics_str})"
        
        query += " ORDER BY r.model_name, r.hardware_type, r.batch_size, r.timestamp"
        
        try:
            # Execute the query
            results = self._conn.execute(query, params).fetchdf()
            
            if results.empty:
                logger.warning("No data found for the specified criteria")
                return pd.DataFrame() if output_format == 'dataframe' else {}
            
            # Ensure timestamp is datetime
            results['timestamp'] = pd.to_datetime(results['timestamp'])
            
            # Group by model, hardware, batch size, and metric
            grouped = results.groupby(['model_name', 'hardware_type', 'batch_size', 'metric_name'])
            
            # Calculate statistics for each group
            stats = grouped.agg({
                'metric_value': ['mean', 'std', 'min', 'max', 'count']
            }).reset_index()
            
            # Flatten the MultiIndex columns
            stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
            
            # Pivot the data to create a matrix of hardware platforms vs models
            pivot_metrics = {}
            
            for metric in metrics:
                metric_data = stats[stats['metric_name'] == metric]
                
                if metric_data.empty:
                    continue
                
                # Create pivot table with models as rows and hardware as columns
                pivot = pd.pivot_table(
                    metric_data,
                    values='metric_value_mean',
                    index=['model_name', 'batch_size'],
                    columns=['hardware_type']
                )
                
                # Apply normalization if requested
                if normalization == 'min-max':
                    # Min-max normalization (0-1 scale)
                    pivot_normalized = pivot.copy()
                    for col in pivot.columns:
                        min_val = pivot[col].min()
                        max_val = pivot[col].max()
                        if max_val > min_val:
                            pivot_normalized[col] = (pivot[col] - min_val) / (max_val - min_val)
                        else:
                            pivot_normalized[col] = 0
                    pivot_metrics[metric] = pivot_normalized
                elif normalization == 'z-score':
                    # Z-score normalization
                    pivot_normalized = pivot.copy()
                    for col in pivot.columns:
                        mean = pivot[col].mean()
                        std = pivot[col].std()
                        if std > 0:
                            pivot_normalized[col] = (pivot[col] - mean) / std
                        else:
                            pivot_normalized[col] = 0
                    pivot_metrics[metric] = pivot_normalized
                else:
                    # No normalization
                    pivot_metrics[metric] = pivot
            
            # Calculate rankings for each metric
            rankings = {}
            
            for metric, pivot in pivot_metrics.items():
                # For latency, lower is better; for throughput, higher is better
                if metric.lower() in ['latency', 'response_time', 'initialization_time']:
                    rankings[metric] = pivot.rank(axis=1, ascending=True)
                else:
                    rankings[metric] = pivot.rank(axis=1, ascending=False)
            
            # Calculate overall score
            # Average the rankings across all metrics
            overall_rankings = None
            
            for metric, ranking in rankings.items():
                if overall_rankings is None:
                    overall_rankings = ranking.copy()
                else:
                    overall_rankings = overall_rankings.add(ranking, fill_value=0)
            
            if overall_rankings is not None:
                overall_rankings = overall_rankings / len(rankings)
            
            # Prepare the final result
            benchmark_result = {
                'metrics': {},
                'rankings': {},
                'overall_ranking': overall_rankings.to_dict() if overall_rankings is not None else None,
                'metadata': {
                    'model_types': model_types,
                    'hardware_platforms': hardware_platforms,
                    'batch_sizes': batch_sizes,
                    'metrics': metrics,
                    'time_period_days': time_period_days,
                    'normalization': normalization,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Convert pivot tables and rankings to dictionaries
            for metric, pivot in pivot_metrics.items():
                benchmark_result['metrics'][metric] = pivot.to_dict()
            
            for metric, ranking in rankings.items():
                benchmark_result['rankings'][metric] = ranking.to_dict()
            
            # Generate visualization if requested
            if visualize and PLOTTING_AVAILABLE:
                self._visualize_benchmark_comparison(pivot_metrics, rankings, overall_rankings, output_path)
            
            # Return in the requested format
            if output_format == 'dataframe':
                # Return a dictionary of DataFrames
                return {
                    'metrics': pivot_metrics,
                    'rankings': rankings,
                    'overall_ranking': overall_rankings
                }
            else:
                return benchmark_result
                
        except Exception as e:
            logger.error(f"Error in benchmark comparison: {str(e)}")
            raise RuntimeError(f"Failed to perform benchmark comparison: {str(e)}")
    
    def _visualize_benchmark_comparison(self,
                                      pivot_metrics: Dict[str, pd.DataFrame],
                                      rankings: Dict[str, pd.DataFrame],
                                      overall_rankings: pd.DataFrame,
                                      output_path: Optional[str] = None):
        """
        Visualize benchmark comparison results.
        
        Args:
            pivot_metrics: Dictionary of pivot tables for each metric
            rankings: Dictionary of rankings for each metric
            overall_rankings: Overall rankings DataFrame
            output_path: Path to save visualizations
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available. Skipping visualization.")
            return
        
        # Create heatmaps for each metric
        for metric, pivot in pivot_metrics.items():
            # Set up the figure
            fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.4)))
            
            # Create heatmap
            sns.heatmap(pivot, annot=True, cmap="YlGnBu", linewidths=.5, ax=ax)
            
            # Set title and labels
            ax.set_title(f"Benchmark Comparison - {metric}")
            
            plt.tight_layout()
            
            # Save or display the plot
            if output_path:
                # Create a specific filename for each metric
                file_name = f"benchmark_comparison_{metric}.png"
                file_path = os.path.join(output_path, file_name) if os.path.isdir(output_path) else output_path
                
                output_dir = os.path.dirname(file_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved benchmark comparison visualization to {file_path}")
            
            plt.close()
        
        # Create ranking heatmaps
        for metric, ranking in rankings.items():
            # Set up the figure
            fig, ax = plt.subplots(figsize=(12, max(8, len(ranking) * 0.4)))
            
            # Create heatmap with a different colormap for rankings
            sns.heatmap(ranking, annot=True, cmap="coolwarm_r", linewidths=.5, ax=ax)
            
            # Set title and labels
            ax.set_title(f"Hardware Ranking - {metric} (lower is better)")
            
            plt.tight_layout()
            
            # Save or display the plot
            if output_path:
                # Create a specific filename for each metric
                file_name = f"hardware_ranking_{metric}.png"
                file_path = os.path.join(output_path, file_name) if os.path.isdir(output_path) else output_path
                
                output_dir = os.path.dirname(file_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved ranking visualization to {file_path}")
            
            plt.close()
        
        # Create overall ranking heatmap
        if overall_rankings is not None:
            # Set up the figure
            fig, ax = plt.subplots(figsize=(12, max(8, len(overall_rankings) * 0.4)))
            
            # Create heatmap
            sns.heatmap(overall_rankings, annot=True, cmap="coolwarm_r", linewidths=.5, ax=ax)
            
            # Set title and labels
            ax.set_title(f"Overall Hardware Ranking (lower is better)")
            
            plt.tight_layout()
            
            # Save or display the plot
            if output_path:
                file_name = "overall_hardware_ranking.png"
                file_path = os.path.join(output_path, file_name) if os.path.isdir(output_path) else output_path
                
                output_dir = os.path.dirname(file_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved overall ranking visualization to {file_path}")
            
            plt.close()
            
    def close(self):
        """Close the database connection."""
        if hasattr(self, '_conn') and self._conn:
            self._conn.close()
    
    def __del__(self):
        """Destructor to ensure the database connection is closed."""
        self.close()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    try:
        analyzer = AdvancedAnalysis(db_path="benchmark_db.duckdb")
        
        # Example correlation analysis
        correlations = analyzer.correlation_analysis(
            group_by='model_name',
            metrics=['latency', 'throughput', 'memory_usage'],
            visualize=True,
            output_path='correlations.png'
        )
        
        # Example pattern detection
        patterns = analyzer.detect_performance_patterns(
            group_by='hardware_type',
            time_period_days=30,
            visualize=True,
            output_path='patterns.png'
        )
        
        # Example test run comparison
        comparison = analyzer.compare_test_runs(
            run_id1='run1',
            run_id2='run2',
            visualize=True,
            output_path='comparison.png'
        )
        
        # Example performance prediction
        prediction = analyzer.predict_performance(
            model_name='bert-base-uncased',
            hardware_type='cuda',
            batch_size=8,
            visualize=True,
            output_path='prediction.png'
        )
        
        # Example benchmark comparison
        benchmarks = analyzer.benchmark_comparison(
            hardware_platforms=['cpu', 'cuda', 'rocm', 'webgpu'],
            visualize=True,
            output_path='benchmark_comparison.png'
        )
        
    except Exception as e:
        logging.error(f"Error in example: {str(e)}")
    finally:
        if 'analyzer' in locals():
            analyzer.close()