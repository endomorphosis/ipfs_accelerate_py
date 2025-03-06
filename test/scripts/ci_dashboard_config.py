#!/usr/bin/env python
"""
CI Dashboard Configuration

This module contains configuration settings for the CI/CD test results dashboard.
It defines dashboard layouts, visualization settings, and report templates.
"""

# Dashboard configuration
DASHBOARD_CONFIG = {
    # General settings
    "title": "IPFS Accelerate Performance Dashboard",
    "refresh_interval": 3600,  # seconds
    "theme": "light",  # light or dark
    
    # Navigation sections
    "navigation": [
        {"name": "Overview", "path": "/", "icon": "dashboard"},
        {"name": "Hardware Compatibility", "path": "/compatibility", "icon": "devices"},
        {"name": "Performance Metrics", "path": "/performance", "icon": "speed"},
        {"name": "Regression Analysis", "path": "/regressions", "icon": "trending_down"},
        {"name": "Historical Data", "path": "/history", "icon": "history"}
    ],
    
    # Dashboard sections
    "sections": {
        "overview": {
            "title": "Performance Dashboard Overview",
            "description": "Key metrics and status for model performance across hardware platforms",
            "charts": [
                {
                    "type": "summary_cards",
                    "title": "Performance Summary",
                    "items": [
                        {"name": "Total Models", "query": "SELECT COUNT(DISTINCT model_id) FROM models"},
                        {"name": "Hardware Platforms", "query": "SELECT COUNT(DISTINCT hardware_id) FROM hardware_platforms"},
                        {"name": "Total Benchmarks", "query": "SELECT COUNT(*) FROM performance_results"},
                        {"name": "Last Updated", "query": "SELECT MAX(test_timestamp) FROM performance_results"}
                    ]
                },
                {
                    "type": "bar_chart",
                    "title": "Hardware Platform Usage",
                    "query": """
                        SELECT hardware_type, COUNT(*) as test_count
                        FROM performance_results pr
                        JOIN hardware_platforms h ON pr.hardware_id = h.hardware_id
                        GROUP BY hardware_type
                        ORDER BY test_count DESC
                    """,
                    "x_field": "hardware_type",
                    "y_field": "test_count"
                },
                {
                    "type": "line_chart",
                    "title": "Benchmark Trends (Last 30 Days)",
                    "query": """
                        SELECT DATE_TRUNC('day', test_timestamp) as date, COUNT(*) as test_count
                        FROM performance_results
                        WHERE test_timestamp >= CURRENT_DATE - INTERVAL '30 days'
                        GROUP BY DATE_TRUNC('day', test_timestamp)
                        ORDER BY date
                    """,
                    "x_field": "date",
                    "y_field": "test_count"
                }
            ]
        },
        "compatibility": {
            "title": "Hardware Compatibility Matrix",
            "description": "Compatibility status for models across different hardware platforms",
            "charts": [
                {
                    "type": "heatmap",
                    "title": "Model-Hardware Compatibility",
                    "query": """
                        SELECT 
                            m.model_name,
                            h.hardware_type,
                            CASE WHEN MAX(tr.success) = 1 THEN 1 ELSE 0 END as is_compatible
                        FROM 
                            models m
                        CROSS JOIN 
                            hardware_platforms h
                        LEFT JOIN 
                            test_results tr ON m.model_id = tr.model_id AND h.hardware_id = tr.hardware_id
                        GROUP BY 
                            m.model_name, h.hardware_type
                        ORDER BY 
                            m.model_name, h.hardware_type
                    """,
                    "x_field": "hardware_type",
                    "y_field": "model_name",
                    "z_field": "is_compatible",
                    "color_scale": ["#FF6B6B", "#4ECB71"]
                }
            ]
        },
        "performance": {
            "title": "Performance Metrics",
            "description": "Detailed performance metrics for models across hardware platforms",
            "charts": [
                {
                    "type": "bar_chart",
                    "title": "Throughput by Hardware Platform",
                    "query": """
                        SELECT 
                            m.model_name,
                            h.hardware_type,
                            AVG(pr.throughput_items_per_second) as avg_throughput
                        FROM 
                            performance_results pr
                        JOIN 
                            models m ON pr.model_id = m.model_id
                        JOIN 
                            hardware_platforms h ON pr.hardware_id = h.hardware_id
                        GROUP BY 
                            m.model_name, h.hardware_type
                        ORDER BY 
                            m.model_name, avg_throughput DESC
                    """,
                    "x_field": "model_name",
                    "y_field": "avg_throughput",
                    "color_field": "hardware_type"
                },
                {
                    "type": "bar_chart",
                    "title": "Latency by Hardware Platform",
                    "query": """
                        SELECT 
                            m.model_name,
                            h.hardware_type,
                            AVG(pr.average_latency_ms) as avg_latency
                        FROM 
                            performance_results pr
                        JOIN 
                            models m ON pr.model_id = m.model_id
                        JOIN 
                            hardware_platforms h ON pr.hardware_id = h.hardware_id
                        GROUP BY 
                            m.model_name, h.hardware_type
                        ORDER BY 
                            m.model_name, avg_latency ASC
                    """,
                    "x_field": "model_name",
                    "y_field": "avg_latency",
                    "color_field": "hardware_type"
                },
                {
                    "type": "bar_chart",
                    "title": "Memory Usage by Hardware Platform",
                    "query": """
                        SELECT 
                            m.model_name,
                            h.hardware_type,
                            AVG(pr.memory_peak_mb) as avg_memory
                        FROM 
                            performance_results pr
                        JOIN 
                            models m ON pr.model_id = m.model_id
                        JOIN 
                            hardware_platforms h ON pr.hardware_id = h.hardware_id
                        GROUP BY 
                            m.model_name, h.hardware_type
                        ORDER BY 
                            m.model_name, avg_memory ASC
                    """,
                    "x_field": "model_name",
                    "y_field": "avg_memory",
                    "color_field": "hardware_type"
                }
            ]
        },
        "regressions": {
            "title": "Regression Analysis",
            "description": "Analysis of performance regressions across test runs",
            "charts": [
                {
                    "type": "table",
                    "title": "Recent Regressions",
                    "query": """
                        WITH regression_data AS (
                            SELECT 
                                m.model_name,
                                h.hardware_type,
                                'throughput' as metric,
                                pr1.throughput_items_per_second as current_value,
                                pr2.throughput_items_per_second as previous_value,
                                (pr1.throughput_items_per_second - pr2.throughput_items_per_second) / pr2.throughput_items_per_second as change
                            FROM 
                                (SELECT * FROM performance_results WHERE run_id = (SELECT MAX(run_id) FROM performance_results)) pr1
                            JOIN 
                                (SELECT * FROM performance_results WHERE run_id = (SELECT MAX(run_id) FROM performance_results WHERE run_id != (SELECT MAX(run_id) FROM performance_results))) pr2
                                ON pr1.model_id = pr2.model_id AND pr1.hardware_id = pr2.hardware_id
                            JOIN 
                                models m ON pr1.model_id = m.model_id
                            JOIN 
                                hardware_platforms h ON pr1.hardware_id = h.hardware_id
                            WHERE
                                pr1.throughput_items_per_second < pr2.throughput_items_per_second * 0.9
                        )
                        SELECT
                            model_name,
                            hardware_type,
                            metric,
                            current_value,
                            previous_value,
                            change * 100 as percent_change
                        FROM
                            regression_data
                        ORDER BY
                            change ASC
                        LIMIT 10
                    """,
                    "columns": [
                        {"name": "Model", "field": "model_name"},
                        {"name": "Hardware", "field": "hardware_type"},
                        {"name": "Metric", "field": "metric"},
                        {"name": "Current", "field": "current_value", "format": "0.00"},
                        {"name": "Previous", "field": "previous_value", "format": "0.00"},
                        {"name": "Change", "field": "percent_change", "format": "0.00%"}
                    ]
                }
            ]
        },
        "history": {
            "title": "Historical Performance",
            "description": "Historical performance trends for models across hardware platforms",
            "charts": [
                {
                    "type": "line_chart",
                    "title": "Throughput History",
                    "query": """
                        SELECT 
                            DATE_TRUNC('day', pr.test_timestamp) as date,
                            m.model_name,
                            h.hardware_type,
                            AVG(pr.throughput_items_per_second) as avg_throughput
                        FROM 
                            performance_results pr
                        JOIN 
                            models m ON pr.model_id = m.model_id
                        JOIN 
                            hardware_platforms h ON pr.hardware_id = h.hardware_id
                        WHERE
                            pr.test_timestamp >= CURRENT_DATE - INTERVAL '30 days'
                        GROUP BY 
                            DATE_TRUNC('day', pr.test_timestamp), m.model_name, h.hardware_type
                        ORDER BY 
                            date
                    """,
                    "x_field": "date",
                    "y_field": "avg_throughput",
                    "color_field": "model_name",
                    "style_field": "hardware_type"
                }
            ]
        }
    }
}

# Report templates
REPORT_TEMPLATES = {
    "compatibility_matrix": {
        "title": "Hardware Compatibility Matrix",
        "description": "Model compatibility status across different hardware platforms",
        "format": "markdown",
        "template": """
# Hardware Compatibility Matrix

Generated: {{generated_date}}

## Model Compatibility

| Model | Type | {{#hardware_types}} {{.}} | {{/hardware_types}}
|-------|------|{{#hardware_types}}--------|{{/hardware_types}}
{{#models}}
| {{model_name}} | {{model_type}} | {{#compatibilities}} {{#compatible}}✅{{/compatible}}{{^compatible}}⚠️{{/compatible}} | {{/compatibilities}}
{{/models}}

## Hardware Recommendations

| Model Type | Recommended Hardware | Notes |
|------------|---------------------|-------|
{{#recommendations}}
| {{model_type}} | {{recommended_hardware}} | {{notes}} |
{{/recommendations}}
        """
    },
    "performance_report": {
        "title": "Performance Report",
        "description": "Performance metrics for models across hardware platforms",
        "format": "html",
        "template": """
<!DOCTYPE html>
<html>
<head>
    <title>Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .chart { max-width: 100%; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Performance Report</h1>
    <p>Generated: {{generated_date}}</p>
    
    <h2>Throughput (items/second)</h2>
    <table>
        <tr>
            <th>Model</th>
            {{#hardware_types}}<th>{{.}}</th>{{/hardware_types}}
        </tr>
        {{#throughput_data}}
        <tr>
            <td>{{model_name}}</td>
            {{#values}}<td>{{.}}</td>{{/values}}
        </tr>
        {{/throughput_data}}
    </table>
    
    <h2>Latency (ms)</h2>
    <table>
        <tr>
            <th>Model</th>
            {{#hardware_types}}<th>{{.}}</th>{{/hardware_types}}
        </tr>
        {{#latency_data}}
        <tr>
            <td>{{model_name}}</td>
            {{#values}}<td>{{.}}</td>{{/values}}
        </tr>
        {{/latency_data}}
    </table>
    
    <h2>Memory Usage (MB)</h2>
    <table>
        <tr>
            <th>Model</th>
            {{#hardware_types}}<th>{{.}}</th>{{/hardware_types}}
        </tr>
        {{#memory_data}}
        <tr>
            <td>{{model_name}}</td>
            {{#values}}<td>{{.}}</td>{{/values}}
        </tr>
        {{/memory_data}}
    </table>
</body>
</html>
        """
    },
    "regression_report": {
        "title": "Regression Report",
        "description": "Analysis of performance regressions",
        "format": "html",
        "template": """
<!DOCTYPE html>
<html>
<head>
    <title>Regression Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .regression { color: red; }
        .improvement { color: green; }
        .high-severity { font-weight: bold; background-color: #ffcccc; }
        .medium-severity { background-color: #fff2cc; }
        .chart { max-width: 100%; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>Performance Regression Report</h1>
    <p>Run ID: {{run_id}}</p>
    <p>Generated: {{generated_date}}</p>
    <p><strong>{{regression_count}}</strong> performance regressions detected.</p>
    
    <h2>Summary by Severity</h2>
    <ul>
        <li><strong>High Severity:</strong> {{high_count}} regressions</li>
        <li><strong>Medium Severity:</strong> {{medium_count}} regressions</li>
    </ul>
    
    <h2>Detailed Regressions</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Hardware</th>
            <th>Metric</th>
            <th>Current Value</th>
            <th>Historical Value</th>
            <th>Change</th>
            <th>Severity</th>
        </tr>
        {{#regressions}}
        <tr class="{{severity}}-severity">
            <td>{{model_name}}</td>
            <td>{{hardware_type}}</td>
            <td>{{metric_display_name}}</td>
            <td>{{current_value}}</td>
            <td>{{historical_value}}</td>
            <td class="regression">{{change_percent}}</td>
            <td>{{severity}}</td>
        </tr>
        {{/regressions}}
    </table>
</body>
</html>
        """
    }
}

# Chart configurations
CHART_CONFIGS = {
    "performance_comparison": {
        "chart_type": "bar",
        "title": "Performance Comparison: Throughput by Hardware",
        "x_axis_label": "Model",
        "y_axis_label": "Throughput (items/s)",
        "legend_title": "Hardware Platform",
        "colors": ["#4285F4", "#34A853", "#FBBC05", "#EA4335", "#8142FF"]
    },
    "latency_comparison": {
        "chart_type": "bar",
        "title": "Performance Comparison: Latency by Hardware",
        "x_axis_label": "Model",
        "y_axis_label": "Latency (ms)",
        "legend_title": "Hardware Platform",
        "colors": ["#4285F4", "#34A853", "#FBBC05", "#EA4335", "#8142FF"]
    },
    "memory_comparison": {
        "chart_type": "bar",
        "title": "Performance Comparison: Memory Usage by Hardware",
        "x_axis_label": "Model",
        "y_axis_label": "Memory (MB)",
        "legend_title": "Hardware Platform",
        "colors": ["#4285F4", "#34A853", "#FBBC05", "#EA4335", "#8142FF"]
    },
    "throughput_history": {
        "chart_type": "line",
        "title": "Throughput History",
        "x_axis_label": "Date",
        "y_axis_label": "Throughput (items/s)",
        "legend_title": "Model",
        "colors": ["#4285F4", "#34A853", "#FBBC05", "#EA4335", "#8142FF"]
    },
    "compatibility_heatmap": {
        "chart_type": "heatmap",
        "title": "Model-Hardware Compatibility Matrix",
        "x_axis_label": "Hardware Platform",
        "y_axis_label": "Model",
        "color_scale": ["#FF6B6B", "#4ECB71"],
        "legend_title": "Compatible"
    }
}

# GitHub Pages settings
GITHUB_PAGES_CONFIG = {
    "site_title": "IPFS Accelerate Performance Dashboard",
    "site_description": "Benchmark results and hardware compatibility for the IPFS Accelerate Python Framework",
    "repo_url": "https://github.com/ipfs/ipfs_accelerate_py",
    "base_url": "/ipfs_accelerate_py/",
    "theme": "minima",
    "update_frequency": "daily",
    "navigation": [
        {"title": "Home", "url": "/"},
        {"title": "Compatibility Matrix", "url": "/compatibility"},
        {"title": "Performance Report", "url": "/performance"},
        {"title": "Regression Analysis", "url": "/regressions"},
        {"title": "About", "url": "/about"}
    ]
}

if __name__ == "__main__":
    print("This is a configuration module and is not meant to be run directly.")
    print("Import this module in your dashboard generation scripts.")
    sys.exit(0)