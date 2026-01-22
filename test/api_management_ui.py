#!/usr/bin/env python3
"""
Management UI for API Predictive Analytics Visualization

This module provides an interactive web-based UI for visualizing 
predictive analytics data generated from API monitoring.
It allows users to explore forecasts, anomalies, and optimization 
recommendations through interactive dashboards.
"""

import os
import sys
import json
import logging
import argparse
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc

# Internal imports
try:
    from api_predictive_analytics import TimeSeriesPredictor, AnomalyPredictor
    from api_monitoring_dashboard import APIMonitoringDashboard
    from api_anomaly_detection import AnomalyDetector
    from api_notification_manager import NotificationManager
except ImportError:
    # Handle import path issues
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from test.api_predictive_analytics import TimeSeriesPredictor, AnomalyPredictor
    from test.api_monitoring_dashboard import APIMonitoringDashboard
    from test.api_anomaly_detection import AnomalyDetector
    from test.api_notification_manager import NotificationManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictiveAnalyticsUI:
    """
    Interactive UI for visualizing API predictive analytics data.
    
    This class provides a Dash-based web interface for exploring
    time series forecasts, anomaly predictions, pattern analysis,
    and cost optimization recommendations.
    """
    
    def __init__(
        self, 
        monitoring_dashboard: Optional[APIMonitoringDashboard] = None,
        data_path: Optional[str] = None,
        theme: str = 'cosmo',
        debug: bool = False,
        enable_caching: bool = False,
        db_path: Optional[str] = None,
        db_repository = None
    ):
        """
        Initialize the predictive analytics UI.
        
        Args:
            monitoring_dashboard: Optional existing monitoring dashboard to use for data
            data_path: Path to load data from if not using an existing dashboard
            theme: Dash bootstrap theme to use
            debug: Whether to enable debug mode
            enable_caching: Whether to enable data caching for improved performance
            db_path: Path to DuckDB database file
            db_repository: Existing DuckDB repository instance
        """
        self.debug = debug
        self.data_path = data_path
        self.monitoring_dashboard = monitoring_dashboard
        self.enable_caching = enable_caching
        self.db_path = db_path
        self.db_repository = db_repository
        
        # Initialize data structures
        self.historical_data = {}
        self.predictions = {}
        self.anomalies = {}
        self.recommendations = {}
        self.comparative_data = {}
        
        # Create the Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes[theme.upper()]],
            title="API Predictive Analytics",
            suppress_callback_exceptions=True
        )
        
        # Load data from appropriate source
        if self.monitoring_dashboard:
            self._load_data_from_dashboard()
        elif self.db_repository or self.db_path:
            self._load_data_from_database()
        elif self.data_path:
            self._load_data_from_file()
            
        # Setup the UI layout
        self._setup_layout()
        self._setup_callbacks()
    
    def _load_data_from_database(self) -> None:
        """Load data from DuckDB database."""
        try:
            # Import DuckDBAPIMetricsRepository
            try:
                from duckdb_api.api_management import DuckDBAPIMetricsRepository
            except ImportError:
                from test.duckdb_api.api_management import DuckDBAPIMetricsRepository
            
            # Initialize repository if not already provided
            if not self.db_repository:
                if not self.db_path:
                    logger.warning("No database path provided.")
                    return
                
                self.db_repository = DuckDBAPIMetricsRepository(
                    db_path=self.db_path,
                    create_if_missing=True
                )
                logger.info(f"Initialized DuckDB repository with database at {self.db_path}")
            
            # Load all data formatted for UI
            all_data = self.db_repository.export_all_data_for_ui()
            
            # Set data properties
            self.historical_data = all_data.get('historical_data', {})
            self.predictions = all_data.get('predictions', {})
            self.anomalies = all_data.get('anomalies', {})
            self.recommendations = all_data.get('recommendations', {})
            self.comparative_data = all_data.get('comparative_data', {})
            
            # Log some statistics
            metrics_count = len(self.historical_data)
            apis_count = sum(len(apis) for apis in self.historical_data.values())
            
            logger.info(f"Loaded data from database: {metrics_count} metrics, {apis_count} API providers")
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
    
    def _load_data_from_dashboard(self) -> None:
        """Load data from an existing monitoring dashboard."""
        if not self.monitoring_dashboard:
            logger.warning("No monitoring dashboard provided.")
            return
        
        try:
            # Load historical data
            self.historical_data = self.monitoring_dashboard.historical_data
            
            # Load predictions if available
            if hasattr(self.monitoring_dashboard, 'predictions'):
                self.predictions = self.monitoring_dashboard.predictions
            
            # Load detected anomalies if available
            if hasattr(self.monitoring_dashboard, 'anomalies'):
                self.anomalies = self.monitoring_dashboard.anomalies
                
            # Load recommendations if available
            if hasattr(self.monitoring_dashboard, 'recommendations'):
                self.recommendations = self.monitoring_dashboard.recommendations
                
            # Load comparative data if available
            if hasattr(self.monitoring_dashboard, 'comparative_data'):
                self.comparative_data = self.monitoring_dashboard.comparative_data
            else:
                # Generate comparative data from historical data if not available
                self.comparative_data = self._generate_comparative_data_from_historical()
                
            logger.info("Successfully loaded data from monitoring dashboard")
        except Exception as e:
            logger.error(f"Error loading data from dashboard: {e}")
            
    def _generate_comparative_data_from_historical(self) -> Dict[str, List[Dict]]:
        """
        Generate comparative data from historical data if not explicitly provided.
        
        Returns:
            Dict containing comparative data by metric type
        """
        comparative_data = {}
        
        try:
            for metric in self.historical_data:
                comparative_data[metric] = []
                
                # Get all APIs for this metric
                apis = list(self.historical_data[metric].keys())
                
                if not apis:
                    continue
                    
                # Get timestamps from the first API (assuming all have similar timestamps)
                first_api = apis[0]
                
                if not self.historical_data[metric][first_api]:
                    continue
                    
                # Use only daily samples to avoid overwhelming the comparative view
                daily_indices = list(range(0, len(self.historical_data[metric][first_api]), 24))
                
                for idx in daily_indices:
                    if idx >= len(self.historical_data[metric][first_api]):
                        continue
                        
                    timestamp = self.historical_data[metric][first_api][idx]["timestamp"]
                    
                    # Collect values for all APIs at this timestamp
                    values = {}
                    for api in apis:
                        if idx < len(self.historical_data[metric][api]):
                            values[api] = self.historical_data[metric][api][idx]["value"]
                    
                    comparative_data[metric].append({
                        "timestamp": timestamp,
                        "values": values
                    })
            
            logger.info("Generated comparative data from historical data")
            return comparative_data
        except Exception as e:
            logger.error(f"Error generating comparative data: {e}")
            return {}
    
    def _load_data_from_file(self) -> None:
        """Load data from a file."""
        if not self.data_path or not os.path.exists(self.data_path):
            logger.warning(f"Data path {self.data_path} does not exist.")
            return
        
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                
            # Load data from file
            self.historical_data = data.get('historical_data', {})
            self.predictions = data.get('predictions', {})
            self.anomalies = data.get('anomalies', {})
            self.recommendations = data.get('recommendations', {})
            self.comparative_data = data.get('comparative_data', {})
            
            logger.info(f"Successfully loaded data from {self.data_path}")
        except Exception as e:
            logger.error(f"Error loading data from file: {e}")
    
    def _setup_layout(self) -> None:
        """Setup the UI layout with tabs and components."""
        # Define the navbar
        navbar = dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand("API Predictive Analytics Dashboard", className="ms-2"),
            ]),
            color="primary",
            dark=True,
        )
        
        # Create tabs for different visualizations
        tabs = dbc.Tabs([
            dbc.Tab(self._create_forecast_tab(), label="Time Series Forecasts"),
            dbc.Tab(self._create_anomaly_tab(), label="Anomaly Predictions"),
            dbc.Tab(self._create_patterns_tab(), label="Pattern Analysis"),
            dbc.Tab(self._create_recommendations_tab(), label="Optimization Recommendations"),
            dbc.Tab(self._create_comparative_tab(), label="Comparative Analysis"),
        ])
        
        # Setup the main layout
        self.app.layout = html.Div([
            navbar,
            dbc.Container([
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        html.H4("API Performance Predictive Analytics"),
                        html.P("Visualize performance forecasts, anomaly predictions, and optimization recommendations"),
                    ]),
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        self._create_filters(),
                    ]),
                ]),
                html.Br(),
                tabs,
                html.Br(),
                html.Hr(),
                html.Footer([
                    html.P("API Predictive Analytics UI • IPFS Accelerate • " + 
                           datetime.datetime.now().strftime("%Y")),
                ], className="text-center text-muted"),
            ]),
        ])
    
    def _create_filters(self) -> dbc.Card:
        """Create filter controls for the dashboard."""
        api_options = []
        metric_options = []
        
        # Extract available APIs and metrics from historical data
        for metric_type in self.historical_data:
            if metric_type not in metric_options:
                metric_options.append(metric_type)
            for api in self.historical_data[metric_type]:
                if api not in api_options:
                    api_options.append(api)
        
        return dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select API:"),
                        dcc.Dropdown(
                            id="api-dropdown",
                            options=[{"label": api, "value": api} for api in api_options],
                            value=api_options[0] if api_options else None,
                            clearable=False
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Select Metric:"),
                        dcc.Dropdown(
                            id="metric-dropdown",
                            options=[{"label": metric.replace("_", " ").title(), "value": metric} 
                                    for metric in metric_options],
                            value=metric_options[0] if metric_options else None,
                            clearable=False
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Time Range:"),
                        dcc.DatePickerRange(
                            id="date-range",
                            start_date=datetime.datetime.now() - datetime.timedelta(days=7),
                            end_date=datetime.datetime.now() + datetime.timedelta(days=7)
                        ),
                    ], width=4),
                ]),
            ]),
            className="mb-3"
        )
    
    def _create_forecast_tab(self) -> html.Div:
        """Create the time series forecast tab content."""
        return html.Div([
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H5("Time Series Forecasts"),
                    html.P("Historical data with future predictions"),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="forecast-graph"),
                ], width=12),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H6("Forecast Settings"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Forecast Horizon (days):"),
                            dcc.Slider(
                                id="forecast-horizon-slider",
                                min=1,
                                max=30,
                                step=1,
                                value=7,
                                marks={i: str(i) for i in range(0, 31, 5)}
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("Confidence Interval:"),
                            dcc.Slider(
                                id="confidence-interval-slider",
                                min=50,
                                max=99,
                                step=1,
                                value=95,
                                marks={i: f"{i}%" for i in range(50, 100, 10)}
                            ),
                        ], width=6),
                    ]),
                ]),
            ]),
        ])
    
    def _create_anomaly_tab(self) -> html.Div:
        """Create the anomaly prediction tab content."""
        return html.Div([
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H5("Anomaly Predictions"),
                    html.P("Detected and predicted anomalies with confidence scores"),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="anomaly-graph"),
                ], width=12),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H6("Anomaly Settings"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Detection Sensitivity:"),
                            dcc.Slider(
                                id="sensitivity-slider",
                                min=1,
                                max=10,
                                step=1,
                                value=5,
                                marks={i: str(i) for i in range(1, 11)}
                            ),
                        ], width=6),
                        dbc.Col([
                            html.Label("Show Anomaly Types:"),
                            dcc.Checklist(
                                id="anomaly-type-checklist",
                                options=[
                                    {"label": "Spikes", "value": "spike"},
                                    {"label": "Trend Breaks", "value": "trend_break"},
                                    {"label": "Oscillations", "value": "oscillation"},
                                    {"label": "Seasonal Deviations", "value": "seasonal"},
                                ],
                                value=["spike", "trend_break", "oscillation", "seasonal"],
                                inline=True
                            ),
                        ], width=6),
                    ]),
                ]),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H6("Detected Anomalies"),
                    html.Div(id="anomaly-table"),
                ]),
            ]),
        ])
    
    def _create_patterns_tab(self) -> html.Div:
        """Create the pattern analysis tab content."""
        return html.Div([
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H5("Pattern Analysis"),
                    html.P("Trend and seasonality detection with pattern classification"),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="pattern-graph"),
                ], width=8),
                dbc.Col([
                    html.H6("Pattern Classification"),
                    html.Div(id="pattern-classification"),
                    html.Br(),
                    html.H6("Pattern Settings"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Component Display:"),
                            dcc.Checklist(
                                id="component-checklist",
                                options=[
                                    {"label": "Trend", "value": "trend"},
                                    {"label": "Seasonality", "value": "seasonality"},
                                    {"label": "Residuals", "value": "residual"},
                                ],
                                value=["trend", "seasonality"],
                                inline=True
                            ),
                        ]),
                    ]),
                ], width=4),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="seasonal-pattern-graph"),
                ]),
            ]),
        ])
    
    def _create_recommendations_tab(self) -> html.Div:
        """Create the optimization recommendations tab content."""
        return html.Div([
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H5("Optimization Recommendations"),
                    html.P("Cost and performance optimization suggestions based on predictive analytics"),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="cost-optimization-graph"),
                ], width=8),
                dbc.Col([
                    html.H6("Top Recommendations"),
                    html.Div(id="recommendations-list"),
                ], width=4),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H6("Historical Cost Efficiency"),
                    dcc.Graph(id="cost-efficiency-graph"),
                ]),
            ]),
        ])
    
    def _create_comparative_tab(self) -> html.Div:
        """Create the comparative analysis tab content."""
        return html.Div([
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H5("Comparative API Performance Analysis"),
                    html.P("Compare performance metrics across different API providers"),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Select Metric for Comparison:"),
                    dcc.Dropdown(
                        id="comparative-metric-dropdown",
                        options=[
                            {"label": metric.replace("_", " ").title(), "value": metric}
                            for metric in self.comparative_data.keys()
                        ] if hasattr(self, 'comparative_data') and self.comparative_data else [],
                        value=list(self.comparative_data.keys())[0] if hasattr(self, 'comparative_data') and self.comparative_data else None,
                        clearable=False
                    ),
                ], width=6),
                dbc.Col([
                    html.Label("Chart Type:"),
                    dcc.RadioItems(
                        id="comparative-chart-type",
                        options=[
                            {"label": "Line Chart", "value": "line"},
                            {"label": "Bar Chart", "value": "bar"},
                            {"label": "Radar Chart", "value": "radar"},
                            {"label": "Heatmap", "value": "heatmap"}
                        ],
                        value="line",
                        inline=True
                    ),
                ], width=6),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="comparative-graph"),
                ], width=12),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H6("Performance Summary"),
                    html.Div(id="performance-summary-table"),
                ], width=12),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H6("Feature Comparison Matrix"),
                    html.Div(id="feature-comparison-matrix"),
                ], width=12),
            ]),
        ])
    
    def _setup_callbacks(self) -> None:
        """Setup all the dashboard callbacks."""
        # Forecast graph callback
        @self.app.callback(
            Output("forecast-graph", "figure"),
            [
                Input("api-dropdown", "value"),
                Input("metric-dropdown", "value"),
                Input("date-range", "start_date"),
                Input("date-range", "end_date"),
                Input("forecast-horizon-slider", "value"),
                Input("confidence-interval-slider", "value")
            ]
        )
        def update_forecast_graph(api, metric, start_date, end_date, horizon, confidence):
            """Update the forecast graph based on user selections."""
            if not api or not metric:
                return go.Figure().update_layout(title="No data selected")
            
            # Check if we have data for this API and metric
            if metric not in self.historical_data or api not in self.historical_data[metric]:
                return go.Figure().update_layout(title=f"No data available for {api} - {metric}")
            
            # Get historical data
            hist_data = self.historical_data[metric][api]
            
            # Check if we have predictions
            has_predictions = (metric in self.predictions and 
                              api in self.predictions[metric])
            
            # Convert to DataFrame for easier plotting
            df_hist = pd.DataFrame(hist_data)
            if 'timestamp' in df_hist.columns:
                df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
                df_hist = df_hist.sort_values('timestamp')
            
            # Create figure
            fig = go.Figure()
            
            # Add historical data
            if 'timestamp' in df_hist.columns and 'value' in df_hist.columns:
                fig.add_trace(go.Scatter(
                    x=df_hist['timestamp'],
                    y=df_hist['value'],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='blue')
                ))
            
            # Add predictions if available
            if has_predictions:
                pred_data = self.predictions[metric][api]
                df_pred = pd.DataFrame(pred_data)
                if 'timestamp' in df_pred.columns:
                    df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
                    df_pred = df_pred.sort_values('timestamp')
                    
                    # Filter to forecast horizon
                    now = datetime.datetime.now()
                    future_end = now + datetime.timedelta(days=horizon)
                    df_pred = df_pred[df_pred['timestamp'] <= future_end]
                    
                    if 'value' in df_pred.columns:
                        fig.add_trace(go.Scatter(
                            x=df_pred['timestamp'],
                            y=df_pred['value'],
                            mode='lines',
                            name='Prediction',
                            line=dict(color='green', dash='dash')
                        ))
                    
                    # Add confidence intervals if available
                    if 'lower_bound' in df_pred.columns and 'upper_bound' in df_pred.columns:
                        fig.add_trace(go.Scatter(
                            x=df_pred['timestamp'],
                            y=df_pred['upper_bound'],
                            mode='lines',
                            name=f'{confidence}% Upper Bound',
                            line=dict(width=0),
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            x=df_pred['timestamp'],
                            y=df_pred['lower_bound'],
                            mode='lines',
                            name=f'{confidence}% Lower Bound',
                            line=dict(width=0),
                            fill='tonexty',
                            fillcolor='rgba(0, 255, 0, 0.1)',
                            showlegend=False
                        ))
            
            # Format the figure
            metric_name = metric.replace("_", " ").title()
            fig.update_layout(
                title=f"{api} - {metric_name} Forecast",
                xaxis_title="Time",
                yaxis_title=metric_name,
                legend_title="Data Type",
                hovermode="x unified",
                height=500
            )
            
            return fig
        
        # Anomaly graph callback
        @self.app.callback(
            Output("anomaly-graph", "figure"),
            Output("anomaly-table", "children"),
            [
                Input("api-dropdown", "value"),
                Input("metric-dropdown", "value"),
                Input("date-range", "start_date"),
                Input("date-range", "end_date"),
                Input("sensitivity-slider", "value"),
                Input("anomaly-type-checklist", "value")
            ]
        )
        def update_anomaly_graph(api, metric, start_date, end_date, sensitivity, anomaly_types):
            """Update the anomaly graph and table based on user selections."""
            if not api or not metric:
                return go.Figure().update_layout(title="No data selected"), html.P("No data selected")
            
            # Check if we have data for this API and metric
            if metric not in self.historical_data or api not in self.historical_data[metric]:
                return (go.Figure().update_layout(title=f"No data available for {api} - {metric}"), 
                        html.P(f"No data available for {api} - {metric}"))
            
            # Check if we have anomalies
            has_anomalies = (metric in self.anomalies and 
                            api in self.anomalies[metric])
            
            # Get historical data
            hist_data = self.historical_data[metric][api]
            
            # Convert to DataFrame for easier plotting
            df_hist = pd.DataFrame(hist_data)
            if 'timestamp' in df_hist.columns:
                df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
                df_hist = df_hist.sort_values('timestamp')
            
            # Create figure
            fig = go.Figure()
            
            # Add historical data
            if 'timestamp' in df_hist.columns and 'value' in df_hist.columns:
                fig.add_trace(go.Scatter(
                    x=df_hist['timestamp'],
                    y=df_hist['value'],
                    mode='lines',
                    name='Historical Data',
                    line=dict(color='blue')
                ))
            
            # Anomaly table content
            table_content = []
            
            # Add anomalies if available
            if has_anomalies:
                anomaly_data = self.anomalies[metric][api]
                
                # Filter by anomaly type
                filtered_anomalies = [a for a in anomaly_data if a.get('type') in anomaly_types]
                
                # Sort by confidence
                sorted_anomalies = sorted(filtered_anomalies, key=lambda x: x.get('confidence', 0), reverse=True)
                
                # Apply sensitivity
                sensitivity_factor = sensitivity / 10.0  # Normalize to 0.1-1.0
                threshold = 1.0 - sensitivity_factor
                
                # Filter by confidence threshold
                visible_anomalies = [a for a in sorted_anomalies if a.get('confidence', 0) >= threshold]
                
                # Plot anomalies
                for anomaly in visible_anomalies:
                    if 'timestamp' in anomaly and 'value' in anomaly:
                        anomaly_time = pd.to_datetime(anomaly['timestamp'])
                        
                        # Different colors for different anomaly types
                        color_map = {
                            'spike': 'red',
                            'trend_break': 'orange',
                            'oscillation': 'purple',
                            'seasonal': 'brown'
                        }
                        
                        anomaly_type = anomaly.get('type', 'spike')
                        color = color_map.get(anomaly_type, 'red')
                        
                        fig.add_trace(go.Scatter(
                            x=[anomaly_time],
                            y=[anomaly['value']],
                            mode='markers',
                            marker=dict(
                                color=color,
                                size=12,
                                symbol='x'
                            ),
                            name=f"{anomaly_type.title()} Anomaly"
                        ))
                        
                        # Add to table
                        table_content.append(html.Tr([
                            html.Td(anomaly_time.strftime('%Y-%m-%d %H:%M:%S')),
                            html.Td(anomaly_type.replace('_', ' ').title()),
                            html.Td(f"{anomaly.get('confidence', 0)*100:.1f}%"),
                            html.Td(anomaly.get('description', '-'))
                        ]))
            
            # Create the anomaly table
            if table_content:
                table = dbc.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Timestamp"),
                            html.Th("Type"),
                            html.Th("Confidence"),
                            html.Th("Description")
                        ])
                    ),
                    html.Tbody(table_content)
                ], bordered=True, hover=True, striped=True, size="sm")
            else:
                table = html.P("No anomalies detected with current settings")
            
            # Format the figure
            metric_name = metric.replace("_", " ").title()
            fig.update_layout(
                title=f"{api} - {metric_name} Anomalies",
                xaxis_title="Time",
                yaxis_title=metric_name,
                legend_title="Data Type",
                hovermode="x unified",
                height=500
            )
            
            return fig, table
        
        # Pattern graph callback
        @self.app.callback(
            Output("pattern-graph", "figure"),
            Output("pattern-classification", "children"),
            Output("seasonal-pattern-graph", "figure"),
            [
                Input("api-dropdown", "value"),
                Input("metric-dropdown", "value"),
                Input("component-checklist", "value")
            ]
        )
        def update_pattern_graphs(api, metric, components):
            """Update pattern analysis graphs based on user selections."""
            if not api or not metric:
                empty_fig = go.Figure().update_layout(title="No data selected")
                return empty_fig, html.P("No data selected"), empty_fig
            
            # Check if we have data for this API and metric
            if metric not in self.historical_data or api not in self.historical_data[metric]:
                empty_fig = go.Figure().update_layout(title=f"No data available for {api} - {metric}")
                return (empty_fig, 
                        html.P(f"No data available for {api} - {metric}"),
                        empty_fig)
            
            # Get historical data
            hist_data = self.historical_data[metric][api]
            
            # Convert to DataFrame
            df_hist = pd.DataFrame(hist_data)
            if 'timestamp' in df_hist.columns:
                df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
                df_hist = df_hist.sort_values('timestamp')
            
            # Placeholder for pattern classification information
            pattern_info = []
            
            # Placeholder for detected patterns
            pattern_types = ["Increasing Trend", "Weekly Seasonality", "Daily Variation"]
            pattern_confidences = [0.92, 0.88, 0.75]
            
            # Create pattern classification cards
            for pattern_type, confidence in zip(pattern_types, pattern_confidences):
                # Create a color based on confidence
                color = "success" if confidence > 0.8 else "warning" if confidence > 0.6 else "danger"
                
                pattern_info.append(
                    dbc.Card(
                        dbc.CardBody([
                            html.H6(pattern_type, className="card-title"),
                            html.P(f"Confidence: {confidence*100:.1f}%"),
                            dbc.Progress(value=confidence*100, color=color, style={"height": "10px"})
                        ])
                    )
                )
                pattern_info.append(html.Br())
            
            # Create decomposition figure for trend/seasonality/residual
            decomp_fig = make_subplots(rows=len(components), cols=1, 
                                      shared_xaxes=True,
                                      subplot_titles=[c.title() for c in components])
            
            # Get time and value columns
            if 'timestamp' in df_hist.columns and 'value' in df_hist.columns:
                times = df_hist['timestamp']
                values = df_hist['value']
                
                # Simple decomposition for illustration
                # In a real implementation, you'd use statsmodels or similar for proper time series decomposition
                
                # Simulate trend, seasonality, and residual components
                n = len(values)
                
                # Simple trend (moving average)
                window = min(7, n//3) if n > 3 else 1
                trend = values.rolling(window=window, center=True).mean()
                
                # Fill NaN values at the beginning and end
                trend = trend.fillna(method='bfill').fillna(method='ffill')
                
                # Simple seasonality (difference from daily average)
                timestamps = pd.Series(times)
                day_of_week = timestamps.dt.dayofweek
                
                # Create seasonal effect based on day of week
                seasonal_effect = day_of_week.map({
                    0: 0.1,  # Monday
                    1: 0.05, # Tuesday
                    2: 0,    # Wednesday
                    3: -0.05,# Thursday
                    4: -0.1, # Friday
                    5: -0.15,# Saturday
                    6: -0.2  # Sunday
                })
                
                # Scale seasonal effect
                value_range = values.max() - values.min()
                seasonality = seasonal_effect * value_range
                
                # Residual (original - trend - seasonality)
                residual = values - trend - seasonality
                
                # Plot components based on selection
                row = 1
                for component in components:
                    if component == 'trend':
                        decomp_fig.add_trace(
                            go.Scatter(x=times, y=trend, mode='lines', name='Trend', line=dict(color='red')),
                            row=row, col=1
                        )
                    elif component == 'seasonality':
                        decomp_fig.add_trace(
                            go.Scatter(x=times, y=seasonality, mode='lines', name='Seasonality', line=dict(color='green')),
                            row=row, col=1
                        )
                    elif component == 'residual':
                        decomp_fig.add_trace(
                            go.Scatter(x=times, y=residual, mode='lines', name='Residual', line=dict(color='purple')),
                            row=row, col=1
                        )
                    row += 1
            
            # Format decomposition figure
            metric_name = metric.replace("_", " ").title()
            decomp_fig.update_layout(
                title=f"{api} - {metric_name} Decomposition",
                height=400,
                showlegend=True
            )
            
            # Create seasonal pattern figure
            seasonal_fig = go.Figure()
            
            # Group by hour and day of week for seasonal pattern visualization
            if 'timestamp' in df_hist.columns and 'value' in df_hist.columns:
                # Add hour of day information
                df_hist['hour'] = df_hist['timestamp'].dt.hour
                df_hist['day_of_week'] = df_hist['timestamp'].dt.dayofweek
                
                # Create pivot table for hour vs day heatmap
                try:
                    pivot_df = df_hist.pivot_table(
                        index='day_of_week', 
                        columns='hour', 
                        values='value',
                        aggfunc='mean'
                    )
                    
                    # Map day indices to names
                    day_names = {
                        0: 'Monday',
                        1: 'Tuesday',
                        2: 'Wednesday',
                        3: 'Thursday',
                        4: 'Friday',
                        5: 'Saturday',
                        6: 'Sunday'
                    }
                    
                    # Create heatmap
                    seasonal_fig = px.imshow(
                        pivot_df,
                        labels=dict(x="Hour of Day", y="Day of Week", color="Average Value"),
                        x=[str(h) for h in range(24)],
                        y=[day_names[d] for d in pivot_df.index],
                        color_continuous_scale="Viridis"
                    )
                    
                    seasonal_fig.update_layout(
                        title=f"{api} - {metric_name} Weekly Patterns",
                        height=400
                    )
                except Exception as e:
                    if self.debug:
                        logger.error(f"Error creating seasonal pattern: {e}")
                    # Fallback to empty figure on error
                    seasonal_fig = go.Figure()
                    seasonal_fig.update_layout(
                        title=f"Insufficient data for seasonal pattern",
                        height=400
                    )
            
            return decomp_fig, pattern_info, seasonal_fig
        
        # Recommendations callback
        @self.app.callback(
            Output("cost-optimization-graph", "figure"),
            Output("recommendations-list", "children"),
            Output("cost-efficiency-graph", "figure"),
            [
                Input("api-dropdown", "value"),
                Input("metric-dropdown", "value")
            ]
        )
        def update_recommendations(api, metric):
            """Update recommendations visualizations based on user selections."""
            if not api or not metric:
                empty_fig = go.Figure().update_layout(title="No data selected")
                return empty_fig, html.P("No data selected"), empty_fig
            
            # Check if we have recommendations for this API
            has_recommendations = False
            if hasattr(self, 'recommendations') and self.recommendations:
                has_recommendations = api in self.recommendations
            
            # Create optimization graph
            opt_fig = go.Figure()
            
            # Get historical cost data if available
            if metric == 'cost' and metric in self.historical_data and api in self.historical_data[metric]:
                hist_data = self.historical_data[metric][api]
                df_hist = pd.DataFrame(hist_data)
                
                if 'timestamp' in df_hist.columns:
                    df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
                    df_hist = df_hist.sort_values('timestamp')
                
                # Plot historical cost and projected cost
                if 'value' in df_hist.columns:
                    # Historical cost
                    opt_fig.add_trace(go.Scatter(
                        x=df_hist['timestamp'],
                        y=df_hist['value'],
                        mode='lines',
                        name='Historical Cost',
                        line=dict(color='blue')
                    ))
                    
                    # Projected cost (simple projection)
                    # For a real implementation, use proper forecasting models
                    last_date = df_hist['timestamp'].max()
                    dates_future = pd.date_range(
                        start=last_date, 
                        periods=30, 
                        freq='D'
                    )
                    
                    # Simple linear projection
                    last_values = df_hist['value'].tail(7).mean()
                    future_values = [last_values] * len(dates_future)
                    
                    opt_fig.add_trace(go.Scatter(
                        x=dates_future,
                        y=future_values,
                        mode='lines',
                        name='Projected Cost',
                        line=dict(color='gray', dash='dash')
                    ))
                    
                    # Optimized cost projection
                    optimized_values = [v * 0.8 for v in future_values]  # 20% reduction
                    
                    opt_fig.add_trace(go.Scatter(
                        x=dates_future,
                        y=optimized_values,
                        mode='lines',
                        name='Optimized Cost',
                        line=dict(color='green', dash='dash')
                    ))
                    
                    # Add optimization region
                    opt_fig.add_trace(go.Scatter(
                        x=dates_future,
                        y=future_values,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    opt_fig.add_trace(go.Scatter(
                        x=dates_future,
                        y=optimized_values,
                        mode='lines',
                        name='Potential Savings',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(0, 255, 0, 0.2)'
                    ))
            
            # Format optimization figure
            opt_fig.update_layout(
                title=f"{api} - Cost Optimization Potential",
                xaxis_title="Time",
                yaxis_title="Cost ($)",
                legend_title="Cost Type",
                hovermode="x unified",
                height=400
            )
            
            # Create recommendations list
            recommendations_content = []
            
            if has_recommendations and api in self.recommendations:
                for i, rec in enumerate(self.recommendations[api][:5]):  # Show top 5
                    # Create a card for each recommendation
                    rec_card = dbc.Card(
                        dbc.CardBody([
                            html.H5(rec.get('title', f'Recommendation {i+1}'), className="card-title"),
                            html.P(rec.get('description', 'No description available')),
                            html.P([
                                html.Strong("Impact: "),
                                html.Span(f"{rec.get('impact', 0)*100:.1f}% potential savings")
                            ]),
                            html.P([
                                html.Strong("Effort: "),
                                html.Span(rec.get('effort', 'Medium'))
                            ])
                        ]),
                        className="mb-3"
                    )
                    recommendations_content.append(rec_card)
            
            # If no recommendations available, create sample ones
            if not recommendations_content:
                sample_recs = [
                    {
                        'title': 'Optimize Batch Size',
                        'description': 'Increase batch size to reduce the number of API calls.',
                        'impact': 0.15,
                        'effort': 'Low'
                    },
                    {
                        'title': 'Implement Caching',
                        'description': 'Cache frequent requests to reduce duplicate API calls.',
                        'impact': 0.25,
                        'effort': 'Medium'
                    },
                    {
                        'title': 'Rate Limiting Adjustment',
                        'description': 'Adjust rate limiting to better match usage patterns.',
                        'impact': 0.10,
                        'effort': 'Low'
                    }
                ]
                
                for rec in sample_recs:
                    rec_card = dbc.Card(
                        dbc.CardBody([
                            html.H5(rec['title'], className="card-title"),
                            html.P(rec['description']),
                            html.P([
                                html.Strong("Impact: "),
                                html.Span(f"{rec['impact']*100:.1f}% potential savings")
                            ]),
                            html.P([
                                html.Strong("Effort: "),
                                html.Span(rec['effort'])
                            ])
                        ]),
                        className="mb-3"
                    )
                    recommendations_content.append(rec_card)
            
            # Create cost efficiency graph
            eff_fig = go.Figure()
            
            # Get throughput and cost data if available
            if 'throughput' in self.historical_data and api in self.historical_data['throughput'] and \
               'cost' in self.historical_data and api in self.historical_data['cost']:
                
                throughput_data = self.historical_data['throughput'][api]
                cost_data = self.historical_data['cost'][api]
                
                df_throughput = pd.DataFrame(throughput_data)
                df_cost = pd.DataFrame(cost_data)
                
                if 'timestamp' in df_throughput.columns and 'timestamp' in df_cost.columns:
                    df_throughput['timestamp'] = pd.to_datetime(df_throughput['timestamp'])
                    df_cost['timestamp'] = pd.to_datetime(df_cost['timestamp'])
                    
                    # Merge on timestamp for direct comparison
                    df_merged = pd.merge_asof(
                        df_throughput.sort_values('timestamp'),
                        df_cost.sort_values('timestamp'),
                        on='timestamp',
                        suffixes=('_throughput', '_cost')
                    )
                    
                    # Calculate efficiency (throughput per cost unit)
                    if 'value_throughput' in df_merged.columns and 'value_cost' in df_merged.columns:
                        df_merged['efficiency'] = df_merged['value_throughput'] / df_merged['value_cost'].clip(lower=0.001)
                        
                        # Create scatter plot of efficiency over time
                        eff_fig.add_trace(go.Scatter(
                            x=df_merged['timestamp'],
                            y=df_merged['efficiency'],
                            mode='lines+markers',
                            name='Cost Efficiency',
                            marker=dict(size=4, color='blue'),
                            line=dict(color='blue')
                        ))
                        
                        # Add trend line
                        x = np.arange(len(df_merged))
                        y = df_merged['efficiency'].values
                        
                        try:
                            from sklearn.linear_model import LinearRegression
                            model = LinearRegression()
                            model.fit(x.reshape(-1, 1), y)
                            y_pred = model.predict(x.reshape(-1, 1))
                            
                            eff_fig.add_trace(go.Scatter(
                                x=df_merged['timestamp'],
                                y=y_pred,
                                mode='lines',
                                name='Efficiency Trend',
                                line=dict(color='red', dash='dash')
                            ))
                        except Exception as e:
                            if self.debug:
                                logger.error(f"Error creating trend line: {e}")
            
            # Format efficiency figure
            eff_fig.update_layout(
                title=f"{api} - Cost Efficiency Over Time",
                xaxis_title="Time",
                yaxis_title="Efficiency (Throughput/Cost)",
                legend_title="Metric",
                hovermode="x unified",
                height=300
            )
            
            return opt_fig, recommendations_content, eff_fig
        
        # Comparative analysis callbacks
        @self.app.callback(
            Output("comparative-graph", "figure"),
            Output("performance-summary-table", "children"),
            Output("feature-comparison-matrix", "children"),
            [
                Input("comparative-metric-dropdown", "value"),
                Input("comparative-chart-type", "value")
            ]
        )
        def update_comparative_analysis(metric, chart_type):
            """Update the comparative analysis visualizations."""
            if not metric or not hasattr(self, 'comparative_data') or not self.comparative_data or metric not in self.comparative_data:
                empty_fig = go.Figure().update_layout(title="No comparative data available")
                return empty_fig, html.P("No data available"), html.P("No data available")
            
            # Get the comparative data for this metric
            data = self.comparative_data[metric]
            
            if not data:
                empty_fig = go.Figure().update_layout(title=f"No comparative data available for {metric}")
                return empty_fig, html.P("No data available"), html.P("No data available")
            
            # Extract data for plotting
            timestamps = [entry["timestamp"] for entry in data]
            
            # Convert timestamps to datetime objects
            timestamps = [pd.to_datetime(ts) for ts in timestamps]
            
            # Get all APIs from the first data point that has values
            apis = []
            for entry in data:
                if "values" in entry and entry["values"]:
                    apis = list(entry["values"].keys())
                    break
            
            # Create different chart types
            if chart_type == "line":
                # Line chart
                fig = go.Figure()
                
                for api in apis:
                    values = [entry["values"].get(api, None) for entry in data]
                    
                    # Filter out None values
                    valid_data = [(ts, val) for ts, val in zip(timestamps, values) if val is not None]
                    if not valid_data:
                        continue
                        
                    valid_timestamps, valid_values = zip(*valid_data)
                    
                    fig.add_trace(go.Scatter(
                        x=valid_timestamps,
                        y=valid_values,
                        mode='lines+markers',
                        name=api,
                        line=dict(width=2)
                    ))
                
                title = f"Comparative {metric.replace('_', ' ').title()} Performance"
                
            elif chart_type == "bar":
                # Bar chart (using the most recent data point)
                recent_data = data[-1]["values"] if data else {}
                
                fig = go.Figure()
                
                # Sort APIs by value
                sorted_apis = sorted(apis, key=lambda api: recent_data.get(api, 0), reverse=True)
                
                fig.add_trace(go.Bar(
                    x=sorted_apis,
                    y=[recent_data.get(api, 0) for api in sorted_apis],
                    text=[f"{recent_data.get(api, 0):.2f}" for api in sorted_apis],
                    textposition='auto'
                ))
                
                title = f"Latest {metric.replace('_', ' ').title()} by API Provider"
                
            elif chart_type == "radar":
                # Radar chart for comparing across different dimensions
                # We'll use the most recent data point and normalize values
                recent_data = data[-1]["values"] if data else {}
                
                # Normalize values to 0-1 scale for radar chart
                max_val = max(recent_data.values()) if recent_data else 1
                min_val = min(recent_data.values()) if recent_data else 0
                range_val = max_val - min_val if max_val != min_val else 1
                
                normalized_data = {
                    api: (recent_data.get(api, 0) - min_val) / range_val
                    for api in apis
                }
                
                # If metric is latency or cost, invert the values (lower is better)
                if metric in ["latency", "cost"]:
                    normalized_data = {
                        api: 1 - val if val is not None else None
                        for api, val in normalized_data.items()
                    }
                
                fig = go.Figure()
                
                for api in apis:
                    fig.add_trace(go.Scatterpolar(
                        r=[normalized_data.get(api, 0)],
                        theta=[metric.replace("_", " ").title()],
                        fill='toself',
                        name=api
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    )
                )
                
                title = f"Normalized {metric.replace('_', ' ').title()} Performance"
                
            else:  # heatmap
                # Heatmap of all APIs over time
                heatmap_data = []
                
                for api in apis:
                    api_values = []
                    for entry in data:
                        if "values" in entry and api in entry["values"]:
                            api_values.append(entry["values"][api])
                        else:
                            api_values.append(None)
                    heatmap_data.append(api_values)
                
                # Transpose data for heatmap
                heatmap_data = np.array(heatmap_data)
                
                # Format timestamps for y-axis
                timestamp_labels = [ts.strftime("%Y-%m-%d") for ts in timestamps]
                
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    x=timestamp_labels,
                    y=apis,
                    colorscale='Viridis',
                    colorbar=dict(title=metric.replace("_", " ").title())
                ))
                
                title = f"{metric.replace('_', ' ').title()} Heatmap Over Time"
            
            # Format figure
            fig.update_layout(
                title=title,
                xaxis_title="Time" if chart_type in ["line", "heatmap"] else "API Provider",
                yaxis_title=metric.replace("_", " ").title() if chart_type in ["line", "bar"] else "API Provider",
                legend_title="API Provider",
                height=500
            )
            
            # Create performance summary table
            # Calculate recent performance metrics
            recent_data = data[-1]["values"] if data else {}
            
            # Calculate average for each API over time
            avg_data = {}
            for api in apis:
                values = [entry["values"].get(api, None) for entry in data]
                valid_values = [v for v in values if v is not None]
                avg_data[api] = sum(valid_values) / len(valid_values) if valid_values else None
            
            # Calculate trend (change over the past 7 days if available)
            trend_data = {}
            if len(data) >= 7:
                for api in apis:
                    recent_values = [entry["values"].get(api, None) for entry in data[-7:]]
                    valid_recent = [v for v in recent_values if v is not None]
                    
                    if len(valid_recent) >= 2:
                        first_val = valid_recent[0]
                        last_val = valid_recent[-1]
                        trend_data[api] = ((last_val - first_val) / first_val) * 100 if first_val != 0 else 0
                    else:
                        trend_data[api] = None
            
            # Create summary table
            table_header = [
                html.Thead(html.Tr([
                    html.Th("API Provider"),
                    html.Th("Current Value"),
                    html.Th("Average"),
                    html.Th("7-Day Trend"),
                    html.Th("Rank")
                ]))
            ]
            
            # Determine ranking (higher is better, except for latency and cost)
            ranking_values = {}
            for api in apis:
                if api in recent_data:
                    # For latency and cost, lower values are better
                    if metric in ["latency", "cost"]:
                        ranking_values[api] = -recent_data[api]
                    else:
                        ranking_values[api] = recent_data[api]
            
            # Sort APIs by ranking
            sorted_apis = sorted(
                [api for api in apis if api in ranking_values],
                key=lambda api: ranking_values[api],
                reverse=True
            )
            
            # Create table rows
            rows = []
            for i, api in enumerate(sorted_apis):
                # Format trend with up/down arrow
                trend_display = ""
                if api in trend_data and trend_data[api] is not None:
                    if trend_data[api] > 0:
                        trend_color = "success" if metric not in ["latency", "cost"] else "danger"
                        trend_display = html.Span([
                            f"+{trend_data[api]:.1f}% ", 
                            html.I(className="fas fa-arrow-up")
                        ], className=f"text-{trend_color}")
                    elif trend_data[api] < 0:
                        trend_color = "danger" if metric not in ["latency", "cost"] else "success"
                        trend_display = html.Span([
                            f"{trend_data[api]:.1f}% ", 
                            html.I(className="fas fa-arrow-down")
                        ], className=f"text-{trend_color}")
                    else:
                        trend_display = html.Span([
                            "0% ", 
                            html.I(className="fas fa-equals")
                        ], className="text-secondary")
                
                # Create row
                row = html.Tr([
                    html.Td(api),
                    html.Td(f"{recent_data.get(api, 'N/A'):.2f}" if api in recent_data else "N/A"),
                    html.Td(f"{avg_data.get(api, 'N/A'):.2f}" if api in avg_data and avg_data[api] is not None else "N/A"),
                    html.Td(trend_display),
                    html.Td(f"#{i+1}")
                ])
                
                rows.append(row)
            
            table_body = [html.Tbody(rows)]
            
            table = dbc.Table(
                table_header + table_body,
                bordered=True,
                striped=True,
                hover=True
            )
            
            # Create feature comparison matrix
            # This would typically come from a database or configuration
            # For this example, we'll generate a simple feature matrix
            feature_matrix = {
                "OpenAI": {
                    "Model Types": "GPT-3.5, GPT-4, DALL-E, Embeddings",
                    "Latency Tier": "Low",
                    "Cost Tier": "High",
                    "Token Limit": "128K (GPT-4)",
                    "Features": "Function calling, JSON mode, Vision"
                },
                "Anthropic": {
                    "Model Types": "Claude 3 Opus, Sonnet, Haiku",
                    "Latency Tier": "Medium",
                    "Cost Tier": "Medium",
                    "Token Limit": "200K",
                    "Features": "Vision, XML output, tools"
                },
                "Cohere": {
                    "Model Types": "Command, Embed",
                    "Latency Tier": "Medium",
                    "Cost Tier": "Low",
                    "Token Limit": "128K",
                    "Features": "RAG optimized, connectors"
                },
                "Groq": {
                    "Model Types": "LLaMA, Mixtral",
                    "Latency Tier": "Very Low",
                    "Cost Tier": "Low",
                    "Token Limit": "32K",
                    "Features": "High throughput"
                },
                "Mistral": {
                    "Model Types": "Mistral, Mixtral",
                    "Latency Tier": "Medium",
                    "Cost Tier": "Low",
                    "Token Limit": "32K",
                    "Features": "Function calling, JSON mode"
                }
            }
            
            # Filter to only include the APIs in our data
            features = list(next(iter(feature_matrix.values())).keys())
            filtered_matrix = {api: feature_matrix.get(api, {}) for api in apis if api in feature_matrix}
            
            # Create the matrix table
            matrix_header = [
                html.Thead(html.Tr([
                    html.Th("Feature"),
                    *[html.Th(api) for api in filtered_matrix.keys()]
                ]))
            ]
            
            matrix_rows = []
            for feature in features:
                row = html.Tr([
                    html.Td(feature),
                    *[html.Td(matrix[feature]) if feature in matrix else html.Td("N/A") 
                      for api, matrix in filtered_matrix.items()]
                ])
                matrix_rows.append(row)
            
            matrix_body = [html.Tbody(matrix_rows)]
            
            matrix_table = dbc.Table(
                matrix_header + matrix_body,
                bordered=True,
                striped=True,
                hover=True,
                size="sm"
            )
            
            return fig, table, matrix_table
    
    def run_server(self, host: str = '0.0.0.0', port: int = 8050, debug: bool = False) -> None:
        """Run the Dash server."""
        self.app.run_server(host=host, port=port, debug=debug)


def main():
    """Main function to parse arguments and start the UI."""
    parser = argparse.ArgumentParser(description='API Predictive Analytics UI')
    parser.add_argument('-p', '--port', type=int, default=8050, help='Port to run the server on')
    parser.add_argument('-d', '--data', type=str, help='Path to JSON data file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    # Initialize UI
    ui = PredictiveAnalyticsUI(data_path=args.data, debug=args.debug)
    
    # Run server
    print(f"Starting API Predictive Analytics UI on port {args.port}...")
    ui.run_server(port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()