#!/usr/bin/env python3
"""
Circuit Breaker Visualization for Monitoring Dashboard

This module provides visualization components for circuit breaker states and metrics
to be displayed in the monitoring dashboard of the distributed testing framework.

Key features:
1. Circuit breaker status visualization
2. Health metrics display
3. Interactive circuit breaker state monitoring
4. Historical state transitions
5. Integration with the monitoring dashboard
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

import jinja2

# Try to import plotly, but provide fallbacks if not available
try:
    import plotly
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Using mock implementations for visualization.")
    
    # Mock classes for plotly
    class MockFigure:
        def __init__(self, *args, **kwargs):
            self.data = []
            self.layout = {}
        
        def update_layout(self, *args, **kwargs):
            pass
            
        def update_yaxes(self, *args, **kwargs):
            pass
            
        def add_trace(self, *args, **kwargs):
            pass
            
        def to_json(self):
            return '{}'
    
    class MockGraphObjects:
        @staticmethod
        def Figure(*args, **kwargs):
            return MockFigure()
            
        @staticmethod
        def Indicator(*args, **kwargs):
            return {}
            
        @staticmethod
        def Pie(*args, **kwargs):
            return {}
            
        @staticmethod
        def Bar(*args, **kwargs):
            return {}
            
        @staticmethod
        def Scatter(*args, **kwargs):
            return {}
    
    class MockSubplots:
        @staticmethod
        def make_subplots(*args, **kwargs):
            return MockFigure()
    
    # Create mock objects
    go = MockGraphObjects()
    px = MockGraphObjects()
    make_subplots = MockSubplots.make_subplots

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("circuit_breaker_visualization")


class CircuitBreakerVisualization:
    """Visualization components for circuit breaker states and metrics."""
    
    def __init__(self, output_dir: str = "./dashboards/circuit_breakers"):
        """
        Initialize the circuit breaker visualization.
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up Jinja2 environment for templates
        module_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(module_dir, "templates")
        
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Historical data
        self.state_history = []
        self.metric_history = []
        self.max_history_points = 1000
        
        logger.info(f"Circuit breaker visualization initialized with output directory: {output_dir}")
    
    def _generate_circuit_state_indicators(self, metrics: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate indicator components for circuit breaker states.
        
        Args:
            metrics: Circuit breaker metrics
            
        Returns:
            Dictionary containing indicator components by category
        """
        indicators = {
            "workers": [],
            "tasks": [],
            "endpoints": [],
            "browsers": []
        }
        
        # Process worker circuits
        for worker_id, worker_metrics in metrics.get("worker_circuits", {}).items():
            state = worker_metrics.get("state", "UNKNOWN")
            health = worker_metrics.get("health_percentage", 0.0)
            
            # Determine color based on state
            color = "green"
            if state == "OPEN":
                color = "red"
            elif state == "HALF_OPEN":
                color = "yellow"
            
            indicators["workers"].append({
                "id": worker_id,
                "state": state,
                "health": health,
                "color": color,
                "metrics": worker_metrics
            })
        
        # Process task circuits
        for task_type, task_metrics in metrics.get("task_circuits", {}).items():
            state = task_metrics.get("state", "UNKNOWN")
            health = task_metrics.get("health_percentage", 0.0)
            
            # Determine color based on state
            color = "green"
            if state == "OPEN":
                color = "red"
            elif state == "HALF_OPEN":
                color = "yellow"
            
            indicators["tasks"].append({
                "id": task_type,
                "state": state,
                "health": health,
                "color": color,
                "metrics": task_metrics
            })
        
        # Process endpoint circuits
        for endpoint, endpoint_metrics in metrics.get("endpoint_circuits", {}).items():
            state = endpoint_metrics.get("state", "UNKNOWN")
            health = endpoint_metrics.get("health_percentage", 0.0)
            
            # Determine color based on state
            color = "green"
            if state == "OPEN":
                color = "red"
            elif state == "HALF_OPEN":
                color = "yellow"
            
            indicators["endpoints"].append({
                "id": endpoint,
                "state": state,
                "health": health,
                "color": color,
                "metrics": endpoint_metrics
            })
            
        # Process browser circuits
        for browser_id, browser_metrics in metrics.get("browser_circuits", {}).items():
            state = browser_metrics.get("state", "UNKNOWN")
            health = browser_metrics.get("health_percentage", 0.0)
            
            # Determine color based on state
            color = "green"
            if state == "OPEN":
                color = "red"
            elif state == "HALF_OPEN":
                color = "yellow"
            
            indicators["browsers"].append({
                "id": browser_id,
                "state": state,
                "health": health,
                "color": color,
                "metrics": browser_metrics
            })
        
        return indicators
    
    def _generate_global_health_gauge(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a global health gauge component.
        
        Args:
            metrics: Circuit breaker metrics
            
        Returns:
            Dictionary containing the gauge component
        """
        global_health = metrics.get("global_health", 100.0)
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=global_health,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Global Circuit Health"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        
        # Update layout
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Convert to JSON
        gauge_json = json.loads(fig.to_json())
        
        return {
            "figure": gauge_json,
            "value": global_health
        }
    
    def _generate_state_distribution_chart(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a chart showing distribution of circuit breaker states.
        
        Args:
            metrics: Circuit breaker metrics
            
        Returns:
            Dictionary containing the chart component
        """
        # Count states across all circuit types
        state_counts = {
            "CLOSED": 0,
            "OPEN": 0,
            "HALF_OPEN": 0
        }
        
        # Count worker states
        for worker_metrics in metrics.get("worker_circuits", {}).values():
            state = worker_metrics.get("state", "UNKNOWN")
            if state in state_counts:
                state_counts[state] += 1
        
        # Count task states
        for task_metrics in metrics.get("task_circuits", {}).values():
            state = task_metrics.get("state", "UNKNOWN")
            if state in state_counts:
                state_counts[state] += 1
        
        # Count endpoint states
        for endpoint_metrics in metrics.get("endpoint_circuits", {}).values():
            state = endpoint_metrics.get("state", "UNKNOWN")
            if state in state_counts:
                state_counts[state] += 1
                
        # Count browser states
        for browser_metrics in metrics.get("browser_circuits", {}).values():
            state = browser_metrics.get("state", "UNKNOWN")
            if state in state_counts:
                state_counts[state] += 1
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(state_counts.keys()),
            values=list(state_counts.values()),
            hole=.3,
            marker_colors=['green', 'red', 'yellow']
        )])
        
        # Update layout
        fig.update_layout(
            title_text="Circuit Breaker State Distribution",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Convert to JSON
        chart_json = json.loads(fig.to_json())
        
        return {
            "figure": chart_json,
            "counts": state_counts
        }
    
    def _generate_failure_rate_chart(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a chart showing failure rates by circuit type.
        
        Args:
            metrics: Circuit breaker metrics
            
        Returns:
            Dictionary containing the chart component
        """
        # Calculate failure rates
        failure_rates = {
            "workers": [],
            "tasks": [],
            "endpoints": [],
            "browsers": []
        }
        
        # Calculate worker failure rates
        for worker_id, worker_metrics in metrics.get("worker_circuits", {}).items():
            total_successes = worker_metrics.get("total_successes", 0)
            total_failures = worker_metrics.get("total_failures", 0)
            total_calls = total_successes + total_failures
            
            failure_rate = 0.0
            if total_calls > 0:
                failure_rate = (total_failures / total_calls) * 100.0
            
            failure_rates["workers"].append({
                "id": worker_id,
                "rate": failure_rate,
                "total_calls": total_calls,
                "total_failures": total_failures
            })
        
        # Calculate task failure rates
        for task_type, task_metrics in metrics.get("task_circuits", {}).items():
            total_successes = task_metrics.get("total_successes", 0)
            total_failures = task_metrics.get("total_failures", 0)
            total_calls = total_successes + total_failures
            
            failure_rate = 0.0
            if total_calls > 0:
                failure_rate = (total_failures / total_calls) * 100.0
            
            failure_rates["tasks"].append({
                "id": task_type,
                "rate": failure_rate,
                "total_calls": total_calls,
                "total_failures": total_failures
            })
        
        # Calculate endpoint failure rates
        for endpoint, endpoint_metrics in metrics.get("endpoint_circuits", {}).items():
            total_successes = endpoint_metrics.get("total_successes", 0)
            total_failures = endpoint_metrics.get("total_failures", 0)
            total_calls = total_successes + total_failures
            
            failure_rate = 0.0
            if total_calls > 0:
                failure_rate = (total_failures / total_calls) * 100.0
            
            failure_rates["endpoints"].append({
                "id": endpoint,
                "rate": failure_rate,
                "total_calls": total_calls,
                "total_failures": total_failures
            })
            
        # Calculate browser failure rates
        for browser_id, browser_metrics in metrics.get("browser_circuits", {}).items():
            total_successes = browser_metrics.get("success_count", 0)  # different field name in browser metrics
            total_failures = browser_metrics.get("failure_count", 0)   # different field name in browser metrics
            total_calls = total_successes + total_failures
            
            failure_rate = 0.0
            if total_calls > 0:
                failure_rate = (total_failures / total_calls) * 100.0
            
            failure_rates["browsers"].append({
                "id": browser_id,
                "rate": failure_rate,
                "total_calls": total_calls,
                "total_failures": total_failures
            })
        
        # Create bar chart - only include circuits with at least 1 call
        worker_ids = [w["id"] for w in failure_rates["workers"] if w["total_calls"] > 0]
        worker_rates = [w["rate"] for w in failure_rates["workers"] if w["total_calls"] > 0]
        
        task_ids = [t["id"] for t in failure_rates["tasks"] if t["total_calls"] > 0]
        task_rates = [t["rate"] for t in failure_rates["tasks"] if t["total_calls"] > 0]
        
        endpoint_ids = [e["id"] for e in failure_rates["endpoints"] if e["total_calls"] > 0]
        endpoint_rates = [e["rate"] for e in failure_rates["endpoints"] if e["total_calls"] > 0]
        
        browser_ids = [b["id"] for b in failure_rates["browsers"] if b["total_calls"] > 0]
        browser_rates = [b["rate"] for b in failure_rates["browsers"] if b["total_calls"] > 0]
        
        # Create figure with subplots
        fig = make_subplots(rows=4, cols=1, 
                            subplot_titles=("Worker Failure Rates", "Task Failure Rates", 
                                           "Endpoint Failure Rates", "Browser Failure Rates"),
                            vertical_spacing=0.1)
        
        # Add worker failure rates
        if worker_ids:
            fig.add_trace(
                go.Bar(x=worker_ids, y=worker_rates, name="Worker Failure Rates", marker_color="blue"),
                row=1, col=1
            )
        
        # Add task failure rates
        if task_ids:
            fig.add_trace(
                go.Bar(x=task_ids, y=task_rates, name="Task Failure Rates", marker_color="orange"),
                row=2, col=1
            )
        
        # Add endpoint failure rates
        if endpoint_ids:
            fig.add_trace(
                go.Bar(x=endpoint_ids, y=endpoint_rates, name="Endpoint Failure Rates", marker_color="green"),
                row=3, col=1
            )
            
        # Add browser failure rates
        if browser_ids:
            fig.add_trace(
                go.Bar(x=browser_ids, y=browser_rates, name="Browser Failure Rates", marker_color="purple"),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=700,
            showlegend=False,
            title_text="Failure Rates by Circuit Type",
            margin=dict(l=50, r=20, t=70, b=20)
        )
        
        # Update y-axis ranges
        fig.update_yaxes(title_text="Failure Rate (%)", range=[0, 100])
        
        # Convert to JSON
        chart_json = json.loads(fig.to_json())
        
        return {
            "figure": chart_json,
            "failure_rates": failure_rates
        }
    
    def _update_history(self, metrics: Dict[str, Any]) -> None:
        """
        Update historical data with current metrics.
        
        Args:
            metrics: Current circuit breaker metrics
        """
        current_time = datetime.now()
        
        # Collect state data
        state_data = {
            "timestamp": current_time.isoformat(),
            "states": {
                "CLOSED": 0,
                "OPEN": 0,
                "HALF_OPEN": 0
            }
        }
        
        # Count worker states
        for worker_metrics in metrics.get("worker_circuits", {}).values():
            state = worker_metrics.get("state", "UNKNOWN")
            if state in state_data["states"]:
                state_data["states"][state] += 1
        
        # Count task states
        for task_metrics in metrics.get("task_circuits", {}).values():
            state = task_metrics.get("state", "UNKNOWN")
            if state in state_data["states"]:
                state_data["states"][state] += 1
        
        # Count endpoint states
        for endpoint_metrics in metrics.get("endpoint_circuits", {}).values():
            state = endpoint_metrics.get("state", "UNKNOWN")
            if state in state_data["states"]:
                state_data["states"][state] += 1
        
        # Collect metric data
        metric_data = {
            "timestamp": current_time.isoformat(),
            "global_health": metrics.get("global_health", 100.0),
            "worker_health_avg": 0.0,
            "task_health_avg": 0.0,
            "endpoint_health_avg": 0.0
        }
        
        # Calculate average worker health
        worker_healths = [w.get("health_percentage", 100.0) for w in metrics.get("worker_circuits", {}).values()]
        if worker_healths:
            metric_data["worker_health_avg"] = sum(worker_healths) / len(worker_healths)
            
        # Calculate average task health
        task_healths = [t.get("health_percentage", 100.0) for t in metrics.get("task_circuits", {}).values()]
        if task_healths:
            metric_data["task_health_avg"] = sum(task_healths) / len(task_healths)
            
        # Calculate average endpoint health
        endpoint_healths = [e.get("health_percentage", 100.0) for e in metrics.get("endpoint_circuits", {}).values()]
        if endpoint_healths:
            metric_data["endpoint_health_avg"] = sum(endpoint_healths) / len(endpoint_healths)
        
        # Add to history
        self.state_history.append(state_data)
        self.metric_history.append(metric_data)
        
        # Limit history size
        if len(self.state_history) > self.max_history_points:
            self.state_history = self.state_history[-self.max_history_points:]
        
        if len(self.metric_history) > self.max_history_points:
            self.metric_history = self.metric_history[-self.max_history_points:]
    
    def _generate_history_chart(self) -> Dict[str, Any]:
        """
        Generate a chart showing historical circuit breaker states and metrics.
        
        Returns:
            Dictionary containing the chart component
        """
        if not self.state_history or not self.metric_history:
            return {"error": "No historical data available"}
        
        # Extract timestamp data
        timestamps = [datetime.fromisoformat(data["timestamp"]) for data in self.metric_history]
        
        # Extract state data
        closed_counts = [data["states"]["CLOSED"] for data in self.state_history]
        open_counts = [data["states"]["OPEN"] for data in self.state_history]
        half_open_counts = [data["states"]["HALF_OPEN"] for data in self.state_history]
        
        # Extract metric data
        global_health = [data["global_health"] for data in self.metric_history]
        worker_health = [data["worker_health_avg"] for data in self.metric_history]
        task_health = [data["task_health_avg"] for data in self.metric_history]
        
        # Create figure with subplots
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=("Circuit Breaker States", "Health Metrics"),
                           vertical_spacing=0.1,
                           shared_xaxes=True)
        
        # Add state traces
        fig.add_trace(
            go.Scatter(x=timestamps, y=closed_counts, name="Closed", line=dict(color="green")),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=open_counts, name="Open", line=dict(color="red")),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=half_open_counts, name="Half-Open", line=dict(color="yellow")),
            row=1, col=1
        )
        
        # Add health traces
        fig.add_trace(
            go.Scatter(x=timestamps, y=global_health, name="Global Health", line=dict(color="blue")),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=worker_health, name="Worker Health", line=dict(color="purple")),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=task_health, name="Task Health", line=dict(color="orange")),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            title_text="Circuit Breaker Historical Metrics",
            margin=dict(l=50, r=20, t=70, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update y-axis for health metrics
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Health %", range=[0, 100], row=2, col=1)
        
        # Convert to JSON
        chart_json = json.loads(fig.to_json())
        
        return {
            "figure": chart_json
        }
    
    def generate_dashboard(self, metrics: Dict[str, Any], theme: str = "light") -> str:
        """
        Generate a complete dashboard for circuit breaker visualization.
        
        Args:
            metrics: Circuit breaker metrics
            theme: Dashboard theme ('light' or 'dark')
            
        Returns:
            HTML content of the dashboard
        """
        # Update historical data
        self._update_history(metrics)
        
        # Generate components
        indicators = self._generate_circuit_state_indicators(metrics)
        health_gauge = self._generate_global_health_gauge(metrics)
        state_chart = self._generate_state_distribution_chart(metrics)
        failure_chart = self._generate_failure_rate_chart(metrics)
        history_chart = self._generate_history_chart()
        
        # Load template
        try:
            template = self.env.get_template("circuit_breaker_dashboard.html")
        except jinja2.exceptions.TemplateNotFound:
            # Create a default template if not found
            template_str = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Circuit Breaker Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background-color: {{ 'white' if theme == 'light' else '#121212' }};
                        color: {{ 'black' if theme == 'light' else 'white' }};
                    }
                    .dashboard-container {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                        gap: 20px;
                    }
                    .dashboard-item {
                        border: 1px solid {{ '#ddd' if theme == 'light' else '#444' }};
                        border-radius: 5px;
                        padding: 15px;
                        background-color: {{ '#f9f9f9' if theme == 'light' else '#1e1e1e' }};
                    }
                    .indicator-container {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 10px;
                    }
                    .indicator {
                        display: inline-block;
                        padding: 10px;
                        border-radius: 5px;
                        text-align: center;
                        min-width: 120px;
                    }
                    .indicator.green {
                        background-color: #dff0d8;
                        color: #3c763d;
                        border: 1px solid #d6e9c6;
                    }
                    .indicator.yellow {
                        background-color: #fcf8e3;
                        color: #8a6d3b;
                        border: 1px solid #faebcc;
                    }
                    .indicator.red {
                        background-color: #f2dede;
                        color: #a94442;
                        border: 1px solid #ebccd1;
                    }
                    h1, h2, h3 {
                        color: {{ 'black' if theme == 'light' else 'white' }};
                    }
                    .full-width {
                        grid-column: 1 / -1;
                    }
                </style>
            </head>
            <body>
                <h1>Circuit Breaker Dashboard</h1>
                <div class="dashboard-container">
                    <div class="dashboard-item">
                        <h2>Global Health</h2>
                        <div id="health-gauge"></div>
                    </div>
                    <div class="dashboard-item">
                        <h2>State Distribution</h2>
                        <div id="state-chart"></div>
                    </div>
                    <div class="dashboard-item full-width">
                        <h2>Workers</h2>
                        <div class="indicator-container">
                            {% for worker in indicators.workers %}
                            <div class="indicator {{ worker.color }}">
                                <h3>{{ worker.id }}</h3>
                                <p>State: {{ worker.state }}</p>
                                <p>Health: {{ "%.1f"|format(worker.health) }}%</p>
                            </div>
                            {% else %}
                            <p>No worker circuits found</p>
                            {% endfor %}
                        </div>
                    </div>
                    <div class="dashboard-item full-width">
                        <h2>Tasks</h2>
                        <div class="indicator-container">
                            {% for task in indicators.tasks %}
                            <div class="indicator {{ task.color }}">
                                <h3>{{ task.id }}</h3>
                                <p>State: {{ task.state }}</p>
                                <p>Health: {{ "%.1f"|format(task.health) }}%</p>
                            </div>
                            {% else %}
                            <p>No task circuits found</p>
                            {% endfor %}
                        </div>
                    </div>
                    <div class="dashboard-item full-width">
                        <h2>Browsers</h2>
                        <div class="indicator-container">
                            {% for browser in indicators.browsers %}
                            <div class="indicator {{ browser.color }}">
                                <h3>{{ browser.id }}</h3>
                                <p>State: {{ browser.state }}</p>
                                <p>Health: {{ "%.1f"|format(browser.health) }}%</p>
                            </div>
                            {% else %}
                            <p>No browser circuits found</p>
                            {% endfor %}
                        </div>
                    </div>
                    <div class="dashboard-item full-width">
                        <h2>Failure Rates</h2>
                        <div id="failure-chart"></div>
                    </div>
                    <div class="dashboard-item full-width">
                        <h2>Historical Metrics</h2>
                        <div id="history-chart"></div>
                    </div>
                </div>
                <script>
                    // Render health gauge
                    var healthGauge = {{ health_gauge.figure|tojson }};
                    Plotly.newPlot('health-gauge', healthGauge.data, healthGauge.layout);
                    
                    // Render state chart
                    var stateChart = {{ state_chart.figure|tojson }};
                    Plotly.newPlot('state-chart', stateChart.data, stateChart.layout);
                    
                    // Render failure chart
                    var failureChart = {{ failure_chart.figure|tojson }};
                    Plotly.newPlot('failure-chart', failureChart.data, failureChart.layout);
                    
                    // Render history chart
                    var historyChart = {{ history_chart.figure|tojson }};
                    Plotly.newPlot('history-chart', historyChart.data, historyChart.layout);
                </script>
            </body>
            </html>
            """
            template = jinja2.Template(template_str)
        
        # Render template
        html_content = template.render(
            indicators=indicators,
            health_gauge=health_gauge,
            state_chart=state_chart,
            failure_chart=failure_chart,
            history_chart=history_chart,
            theme=theme,
            last_update=metrics.get("last_update", datetime.now().isoformat())
        )
        
        # Save dashboard to file
        dashboard_file = os.path.join(self.output_dir, "circuit_breaker_dashboard.html")
        with open(dashboard_file, "w") as f:
            f.write(html_content)
        
        logger.info(f"Generated circuit breaker dashboard: {dashboard_file}")
        
        return html_content
    
    def generate_iframe_html(self, width: str = "100%", height: str = "800px") -> str:
        """
        Generate HTML for embedding the dashboard in an iframe.
        
        Args:
            width: Width of the iframe
            height: Height of the iframe
            
        Returns:
            HTML string for embedding
        """
        dashboard_path = os.path.join(self.output_dir, "circuit_breaker_dashboard.html")
        dashboard_url = f"dashboards/circuit_breakers/circuit_breaker_dashboard.html"
        
        iframe_html = f'<iframe src="{dashboard_url}" width="{width}" height="{height}" frameborder="0"></iframe>'
        
        return iframe_html
    
    def get_latest_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the latest metrics.
        
        Returns:
            Dictionary containing summary metrics
        """
        if not self.metric_history:
            return {"error": "No metrics history available"}
        
        latest_metrics = self.metric_history[-1]
        latest_states = self.state_history[-1]
        
        return {
            "timestamp": latest_metrics["timestamp"],
            "global_health": latest_metrics["global_health"],
            "worker_health": latest_metrics["worker_health_avg"],
            "task_health": latest_metrics["task_health_avg"],
            "endpoint_health": latest_metrics["endpoint_health_avg"],
            "state_counts": latest_states["states"]
        }


class CircuitBreakerDashboardIntegration:
    """Integration between Circuit Breaker Visualization and Monitoring Dashboard."""
    
    def __init__(self, coordinator=None, output_dir: str = "./dashboards/circuit_breakers"):
        """
        Initialize the dashboard integration.
        
        Args:
            coordinator: Coordinator server instance with circuit breaker integration
            output_dir: Directory to save visualization files
        """
        self.coordinator = coordinator
        self.output_dir = output_dir
        
        # Create visualization
        self.visualization = CircuitBreakerVisualization(output_dir)
        
        # Get circuit breaker integration from coordinator
        self.circuit_breaker_integration = None
        if self.coordinator:
            self.circuit_breaker_integration = getattr(
                self.coordinator, 'circuit_breaker_integration', None
            )
        
        logger.info("Circuit breaker dashboard integration initialized")
    
    def get_circuit_breaker_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from the circuit breaker integration.
        
        Returns:
            Dictionary containing circuit breaker metrics
        """
        if not self.circuit_breaker_integration:
            return {
                "error": "Circuit breaker integration not available",
                "worker_circuits": {},
                "task_circuits": {},
                "endpoint_circuits": {},
                "global_health": 100.0,
                "last_update": datetime.now().isoformat()
            }
        
        return self.circuit_breaker_integration.get_circuit_breaker_metrics()
    
    def generate_dashboard(self, theme: str = "light") -> str:
        """
        Generate the circuit breaker dashboard.
        
        Args:
            theme: Dashboard theme ('light' or 'dark')
            
        Returns:
            HTML content of the dashboard
        """
        # Get metrics
        metrics = self.get_circuit_breaker_metrics()
        
        # Generate dashboard
        return self.visualization.generate_dashboard(metrics, theme)
    
    def get_dashboard_iframe_html(self, width: str = "100%", height: str = "800px") -> str:
        """
        Get HTML for embedding the dashboard in an iframe.
        
        Args:
            width: Width of the iframe
            height: Height of the iframe
            
        Returns:
            HTML string for embedding
        """
        return self.visualization.generate_iframe_html(width, height)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the latest metrics.
        
        Returns:
            Dictionary containing summary metrics
        """
        return self.visualization.get_latest_metrics_summary()