# API Management UI

## Overview

The API Management UI provides an interactive web-based dashboard for visualizing predictive analytics data generated from API monitoring within the IPFS Accelerate Distributed Testing Framework. It enables users to explore time series forecasts, anomaly predictions, pattern analysis, and cost optimization recommendations through a user-friendly interface built with Dash and Plotly.

This UI is a key component of the Distributed Testing Framework, providing actionable insights from the collected API performance metrics and helping users make data-driven decisions about API usage optimization.

## Features

- **Time Series Forecasts**: View historical API metrics with ML-based predictions for future performance
- **Anomaly Detection**: Identify and visualize detected and predicted anomalies in API metrics with confidence scoring
- **Pattern Analysis**: Analyze trends, seasonality, and patterns in API performance data with decomposition
- **Cost Optimization**: Get actionable recommendations for improving API cost efficiency with projected savings
- **Real-time Integration**: Connect to the live monitoring dashboard for up-to-date metrics
- **Data Export**: Export visualizations and data in multiple formats (CSV, JSON, HTML)
- **Interactive Filtering**: Filter data by API provider, metric type, time range, and more
- **Responsive Design**: Optimized for both desktop and mobile viewing

## Requirements

The management UI requires the following Python packages:

```
pandas>=1.5.0
numpy>=1.22.0
plotly>=5.10.0
dash>=2.7.0
dash-bootstrap-components>=1.3.0
scikit-learn>=1.1.0
fastapi>=0.88.0  # For API server integration
websockets>=10.4  # For real-time updates
```

## Installation

Install the required dependencies:

```bash
pip install -r api_management_ui_requirements.txt
```

Or install dependencies directly:

```bash
pip install pandas numpy plotly dash dash-bootstrap-components scikit-learn fastapi websockets
```

## Usage

### Basic Usage

Run the management UI with sample data:

```bash
python run_api_management_ui.py --generate-sample
```

This will:
1. Generate sample API performance data for OpenAI, Anthropic, Cohere, Groq, and Mistral APIs
2. Start a web server on port 8050
3. Open your browser to http://localhost:8050

### Command-Line Options

```
python run_api_management_ui.py --help
```

Available options:

- `-p, --port PORT`: Specify the port to run the server on (default: 8050)
- `-d, --data FILE`: Path to JSON data file to load
- `--generate-sample`: Generate sample data for demonstration
- `--sample-path PATH`: Path to save/load sample data (default: ./sample_api_data.json)
- `--connect-dashboard`: Connect to live API monitoring dashboard
- `--fastapi-integration`: Enable FastAPI integration for external access (default: False)
- `--fastapi-port PORT`: Port for FastAPI server if enabled (default: 8000)
- `--debug`: Enable debug mode with more detailed logging
- `--export-format FORMAT`: Default export format (html, png, svg, pdf)
- `--theme THEME`: UI theme to use (cosmo, darkly, flatly, etc.)

### Integration with Distributed Testing

To connect the UI to the live distributed testing framework:

```bash
python run_api_management_ui.py --connect-dashboard
```

This integrates with the existing `APIMonitoringDashboard` to display real-time data collected from distributed testing nodes, along with predictions and anomaly detection.

### FastAPI Integration

Enable the FastAPI backend for programmatic access:

```bash
python run_api_management_ui.py --fastapi-integration --fastapi-port 8000
```

This allows other components of the distributed testing framework to interact with the management UI programmatically, and provides RESTful endpoints for data access and dashboard management.

## Dashboard Sections

### 1. Time Series Forecasts

View historical API performance metrics with predictive forecasts:
- Select different APIs and metrics from the dropdown
- Adjust forecast horizon (1-30 days) and confidence intervals (50-99%)
- Analyze performance trends with interactive zooming and tooltips
- Export forecast data and visualizations
- Compare multiple APIs or metrics side-by-side

### 2. Anomaly Predictions

Identify and investigate anomalies in API performance:
- Visualize detected anomalies with categorization by type
- Adjust sensitivity (1-10) to control anomaly detection threshold
- Filter anomalies by type (spikes, trend breaks, oscillations, seasonal deviations)
- See detailed anomaly information in the table below the chart
- Receive proactive notifications for critical anomalies
- Export anomaly reports for further analysis

### 3. Pattern Analysis

Analyze performance patterns and decompose time series data:
- View trend, seasonality, and residual components with interactive toggles
- See pattern classification with confidence scores for detected patterns
- Examine weekly and hourly patterns through heatmaps
- Analyze cyclical patterns and correlations between metrics
- Identify performance drivers and influencing factors
- Compare patterns across different API providers

### 4. Optimization Recommendations

Get actionable recommendations for optimizing API usage:
- View cost optimization potential with projected savings and ROI
- See specific recommendations with impact assessment and effort required
- Analyze cost efficiency metrics over time with trend visualization
- Sort and filter recommendations by impact, effort, or implementation time
- Track recommendation implementation status
- Export recommendations for planning purposes

### 5. Comparative Analysis (New)

Compare performance across different API providers:
- Side-by-side comparison of key metrics (latency, cost, throughput)
- Relative performance scoring with percentile rankings
- Feature and capability matrix comparison
- Cost-benefit analysis for multi-API strategies
- Historical performance stability comparison
- Export comparative reports for vendor selection

## Data Format

The UI can load data from a JSON file with the following structure:

```json
{
  "historical_data": {
    "metric_type": {
      "api_name": [
        {"timestamp": "ISO-format", "value": numeric_value},
        ...
      ]
    }
  },
  "predictions": {
    "metric_type": {
      "api_name": [
        {
          "timestamp": "ISO-format", 
          "value": numeric_value,
          "lower_bound": numeric_value,
          "upper_bound": numeric_value
        },
        ...
      ]
    }
  },
  "anomalies": {
    "metric_type": {
      "api_name": [
        {
          "timestamp": "ISO-format",
          "value": numeric_value,
          "type": "spike|trend_break|oscillation|seasonal",
          "confidence": numeric_value,
          "description": "description_text",
          "severity": "low|medium|high|critical"
        },
        ...
      ]
    }
  },
  "recommendations": {
    "api_name": [
      {
        "title": "recommendation_title",
        "description": "recommendation_text",
        "impact": numeric_value,
        "effort": "Low|Medium|High",
        "implementation_time": "Hours|Days|Weeks",
        "roi_period": "Days|Weeks|Months",
        "status": "New|In Progress|Implemented|Verified"
      },
      ...
    ]
  },
  "comparative_data": {
    "metric_type": {
      "timestamp": "ISO-format",
      "values": {
        "api_name_1": numeric_value,
        "api_name_2": numeric_value,
        ...
      }
    }
  }
}
```

## Integration with Existing Monitoring

The Management UI is designed to work alongside the API Monitoring Dashboard, providing a focused interface for visualizing predictive analytics data while the monitoring dashboard continues to track real-time metrics.

### Real-time Updates

When connected to the live monitoring dashboard with `--connect-dashboard`, the UI will:

1. Immediately pull current data from the monitoring system
2. Periodically refresh (every 5 minutes by default) to show new data
3. Display real-time alerts for new anomalies or significant pattern changes
4. Update recommendations based on the latest performance metrics

### Database Integration

The UI integrates with the DuckDB database used by the distributed testing framework:

```bash
python run_api_management_ui.py --db-path /path/to/distributed_testing.duckdb
```

This allows for:
- Advanced querying of historical performance data
- Integration with benchmark results
- Long-term trend analysis
- Custom report generation

## Development

### Extending the UI

To add new visualizations or features:

1. Create a new tab in the `_setup_layout()` method of the `PredictiveAnalyticsUI` class
2. Add the tab content creation function
3. Implement the required callbacks for the new tab
4. Update CSS styles if needed

Example extending the UI with a new tab:

```python
def _setup_layout(self):
    # ... existing code ...
    
    # Add a new tab
    tabs = dbc.Tabs([
        # ... existing tabs ...
        dbc.Tab(self._create_new_feature_tab(), label="New Feature"),
    ])
    
    # ... rest of the method ...

def _create_new_feature_tab(self):
    """Create a new custom visualization tab."""
    return html.Div([
        html.Br(),
        dbc.Row([
            dbc.Col([
                html.H5("My New Feature"),
                html.P("Description of the new feature"),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="new-feature-graph"),
            ], width=12),
        ]),
        # Add controls and additional elements
    ])

def _setup_callbacks(self):
    # ... existing callbacks ...
    
    # Add callback for the new feature
    @self.app.callback(
        Output("new-feature-graph", "figure"),
        [Input("api-dropdown", "value"), Input("metric-dropdown", "value")]
    )
    def update_new_feature_graph(api, metric):
        # Implement your visualization logic here
        fig = go.Figure()
        # ...
        return fig
```

### Adding New Metric Types

To support additional metric types:

1. Update the data generation in `generate_sample_data()` to include the new metric
2. Add appropriate visualization logic for the new metric type
3. Update pattern detection logic if needed

Example adding a new metric type:

```python
def generate_sample_data():
    # ... existing code ...
    
    # Add a new metric type
    metrics = ["latency", "cost", "throughput", "success_rate", "tokens_per_second", "my_new_metric"]
    
    # ... in the metrics generation loop ...
    elif metric == "my_new_metric":
        base = 50  # base value for the new metric
        noise_scale = 10
        trend = 0.03  # slight upward trend
    
    # ... rest of the function ...
```

### FastAPI Integration

To extend the FastAPI integration:

1. Add new endpoints in `api_management_ui_server.py`
2. Implement corresponding functionality in the UI class
3. Add authentication if needed
4. Document the new endpoints

Example adding a new API endpoint:

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/management", tags=["management"])

class FilterRequest(BaseModel):
    api_name: str
    metric_type: str
    start_date: str
    end_date: str

@router.post("/filter_data")
async def filter_data(request: FilterRequest):
    """Filter dashboard data based on criteria."""
    try:
        # Implement data filtering logic
        return {"status": "success", "data": filtered_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error filtering data: {str(e)}")
```

## Troubleshooting

Common issues and solutions:

- **UI shows "No data selected"**: Ensure you've either connected to a dashboard or provided a valid data file
- **Charts not displaying properly**: Check that your data format matches the expected structure
- **High memory usage**: For large datasets, consider filtering or aggregating data before loading it into the UI
- **Connection to dashboard fails**: Verify that the API monitoring dashboard is running and accessible
- **Slow UI performance**: Consider enabling data caching with `--enable-caching` option
- **Missing metrics or APIs**: Check that the data source includes all expected metrics and API providers
- **FastAPI endpoints return 404**: Ensure the FastAPI server is running with `--fastapi-integration`
- **Database connection errors**: Verify the database path with `--db-path` and check permissions

## API Documentation

When running with `--fastapi-integration`, the following RESTful endpoints are available:

- **GET /api/metrics**: List available metrics
- **GET /api/providers**: List available API providers
- **GET /api/data/{metric}/{provider}**: Get data for specific metric and provider
- **GET /api/anomalies/{metric}/{provider}**: Get anomalies for specific metric and provider
- **GET /api/recommendations/{provider}**: Get recommendations for specific provider
- **POST /api/filter**: Filter data by custom criteria
- **GET /api/export/{format}**: Export dashboard data in specified format

The API documentation is available at http://localhost:8000/docs when running with FastAPI integration enabled.

## Performance Considerations

For optimal performance with large datasets:

1. Enable data caching with `--enable-caching`
2. Use database integration rather than JSON files
3. Implement data aggregation for long time periods
4. Consider using a production ASGI server like Uvicorn or Hypercorn
5. Run the FastAPI and Dash servers on separate processes

## Security Considerations

When deploying in a production environment:

1. Enable authentication with `--require-auth`
2. Use HTTPS with `--ssl-cert` and `--ssl-key`
3. Set appropriate CORS policies with `--cors-origins`
4. Restrict access to sensitive endpoints
5. Consider using a reverse proxy like Nginx