"""
Monitoring Dashboard Routes

This module defines the web routes for the Monitoring Dashboard.
"""

import json
import logging
import time
from typing import Dict, Any, Optional

from aiohttp import web

logger = logging.getLogger(__name__)

async def handle_index(request: web.Request) -> web.Response:
    """Handle requests to the index page."""
    dashboard = request.app['dashboard']
    theme = request.query.get('theme', dashboard.theme)
    
    # Get current status data
    coordinator_status = await dashboard.get_coordinator_status()
    workers_status = await dashboard.get_workers_status()
    tasks_status = await dashboard.get_tasks_status()
    
    # Convert any complex objects to JSON-serializable format
    coordinator_status = json.loads(json.dumps(coordinator_status, default=str))
    workers_status = json.loads(json.dumps(workers_status, default=str))
    tasks_status = json.loads(json.dumps(tasks_status, default=str))
    
    # Get embedded dashboards for this page
    embedded_dashboards = None
    if dashboard.enable_visualization_integration and dashboard.visualization_integration:
        try:
            embedded_dashboards = dashboard.visualization_integration.get_embedded_dashboards_for_page('index')
        except Exception as e:
            logger.error(f"Error getting embedded dashboards: {e}")
    
    # Get HTML for embedded dashboards
    embedded_dashboard_html = {}
    if embedded_dashboards:
        for name, dash_details in embedded_dashboards.items():
            try:
                iframe_html = dashboard.visualization_integration.get_dashboard_iframe_html(
                    name=name,
                    width="100%",
                    height="600px"
                )
                embedded_dashboard_html[name] = {
                    "html": iframe_html,
                    "position": dash_details.get("position", "below"),
                    "title": dash_details.get("title", "Dashboard")
                }
            except Exception as e:
                logger.error(f"Error getting dashboard iframe HTML for {name}: {e}")
    
    # Render template
    return web.Response(
        text=dashboard.env.get_template('index.html').render(
            coordinator_status=coordinator_status,
            workers_status=workers_status,
            tasks_status=tasks_status,
            theme=theme,
            refresh_interval=dashboard.refresh_interval,
            embedded_dashboards=embedded_dashboard_html,
            visualization_enabled=dashboard.enable_visualization_integration
        ),
        content_type='text/html'
    )

async def handle_workers(request: web.Request) -> web.Response:
    """Handle requests to the workers page."""
    dashboard = request.app['dashboard']
    theme = request.query.get('theme', dashboard.theme)
    
    # Get workers status
    workers_status = await dashboard.get_workers_status()
    workers_status = json.loads(json.dumps(workers_status, default=str))
    
    # Get additional worker details
    workers_details = await dashboard.get_workers_details()
    workers_details = json.loads(json.dumps(workers_details, default=str))
    
    # Render template
    return web.Response(
        text=dashboard.env.get_template('workers.html').render(
            workers_status=workers_status,
            workers_details=workers_details,
            theme=theme,
            refresh_interval=dashboard.refresh_interval
        ),
        content_type='text/html'
    )

async def handle_tasks(request: web.Request) -> web.Response:
    """Handle requests to the tasks page."""
    dashboard = request.app['dashboard']
    theme = request.query.get('theme', dashboard.theme)
    
    # Get tasks status
    tasks_status = await dashboard.get_tasks_status()
    tasks_status = json.loads(json.dumps(tasks_status, default=str))
    
    # Get task history
    task_history = await dashboard.get_task_history()
    task_history = json.loads(json.dumps(task_history, default=str))
    
    # Render template
    return web.Response(
        text=dashboard.env.get_template('tasks.html').render(
            tasks_status=tasks_status,
            task_history=task_history,
            theme=theme,
            refresh_interval=dashboard.refresh_interval
        ),
        content_type='text/html'
    )

async def handle_results_view(request: web.Request) -> web.Response:
    """Handle requests to the results page."""
    dashboard = request.app['dashboard']
    theme = request.query.get('theme', dashboard.theme)
    
    # Check if result aggregator integration is enabled
    if not dashboard.enable_result_aggregator_integration:
        return web.Response(
            text=dashboard.env.get_template('results_disabled.html').render(
                theme=theme
            ),
            content_type='text/html'
        )
    
    # Get result aggregator integration
    result_aggregator = dashboard.result_aggregator_integration
    
    # Get time range from query
    time_range = request.query.get('time_range', '1d')
    
    # Get result summary
    result_summary = await result_aggregator.get_dashboard_result_summary(time_range)
    result_summary = json.loads(json.dumps(result_summary, default=str))
    
    # Get embedded dashboards for this page
    embedded_dashboards = None
    if dashboard.enable_visualization_integration and dashboard.visualization_integration:
        try:
            embedded_dashboards = dashboard.visualization_integration.get_embedded_dashboards_for_page('results')
        except Exception as e:
            logger.error(f"Error getting embedded dashboards: {e}")
    
    # Get HTML for embedded dashboards
    embedded_dashboard_html = {}
    if embedded_dashboards:
        for name, dash_details in embedded_dashboards.items():
            try:
                iframe_html = dashboard.visualization_integration.get_dashboard_iframe_html(
                    name=name,
                    width="100%",
                    height="600px"
                )
                embedded_dashboard_html[name] = {
                    "html": iframe_html,
                    "position": dash_details.get("position", "below"),
                    "title": dash_details.get("title", "Dashboard")
                }
            except Exception as e:
                logger.error(f"Error getting dashboard iframe HTML for {name}: {e}")
    
    # Try to generate a dashboard from result data if we have visualization but no embedded dashboards
    if (dashboard.enable_visualization_integration and dashboard.visualization_integration and 
        not embedded_dashboard_html and dashboard.visualization_integration.visualization_available):
        try:
            # Convert result summary to a format suitable for visualization
            performance_data = {
                "metrics": {
                    "throughput_items_per_second": result_summary.get("overall_stats", {}).get("avg_throughput", 0),
                    "average_latency_ms": result_summary.get("overall_stats", {}).get("avg_latency", 0),
                    "memory_peak_mb": result_summary.get("overall_stats", {}).get("avg_memory", 0)
                },
                "dimensions": {
                    "model_family": result_summary.get("model_families", []),
                    "hardware_type": result_summary.get("hardware_types", [])
                }
            }
            
            # Generate a dashboard from this data
            dashboard_name = f"results_dashboard_{int(time.time())}"
            dashboard_path = dashboard.visualization_integration.generate_dashboard_from_performance_data(
                performance_data=performance_data,
                name=dashboard_name,
                title=f"Results Dashboard for {time_range} Time Range"
            )
            
            if dashboard_path:
                # Register as an embedded dashboard
                dashboard_details = dashboard.visualization_integration.create_embedded_dashboard(
                    name=dashboard_name,
                    page="results",
                    title=f"Results Dashboard for {time_range} Time Range",
                    description=f"Automatically generated dashboard from result data for {time_range} time range",
                    position="below"
                )
                
                if dashboard_details:
                    # Get the iframe HTML
                    iframe_html = dashboard.visualization_integration.get_dashboard_iframe_html(
                        name=dashboard_name,
                        width="100%",
                        height="600px"
                    )
                    
                    embedded_dashboard_html[dashboard_name] = {
                        "html": iframe_html,
                        "position": "below",
                        "title": f"Results Dashboard for {time_range} Time Range"
                    }
        except Exception as e:
            logger.error(f"Error generating dashboard from result data: {e}")
    
    # Render template
    return web.Response(
        text=dashboard.env.get_template('results.html').render(
            result_summary=result_summary,
            theme=theme,
            refresh_interval=dashboard.refresh_interval,
            time_range=time_range,
            embedded_dashboards=embedded_dashboard_html,
            visualization_enabled=dashboard.enable_visualization_integration
        ),
        content_type='text/html'
    )

async def handle_e2e_test_results(request: web.Request) -> web.Response:
    """Handle requests to the E2E test results page."""
    dashboard = request.app['dashboard']
    theme = request.query.get('theme', dashboard.theme)
    
    # Check if E2E test integration is enabled
    if not dashboard.enable_e2e_test_integration:
        return web.Response(
            text=dashboard.env.get_template('e2e_test_results_disabled.html').render(
                theme=theme
            ),
            content_type='text/html'
        )
    
    # Get E2E test integration
    e2e_integration = dashboard.e2e_test_integration
    
    # Get list of tests
    tests = e2e_integration.get_test_list()
    
    # Get selected test ID from URL path or use the latest test
    test_id = request.match_info.get('test_id', None)
    selected_test = test_id
    
    if not test_id and tests:
        # Use the latest test if none specified
        test_id = tests[0]['id']
        selected_test = test_id
    
    # Get test details and visualizations
    current_test = None
    visualizations = {}
    
    if test_id:
        current_test = e2e_integration.get_test_details(test_id)
        visualizations = e2e_integration.get_test_visualizations(test_id)
    
    # Render template
    return web.Response(
        text=dashboard.env.get_template('e2e_test_results.html').render(
            tests=tests,
            selected_test=selected_test,
            current_test=current_test,
            visualizations=visualizations,
            theme=theme
        ),
        content_type='text/html'
    )

async def handle_api_e2e_test_results(request: web.Request) -> web.Response:
    """Handle API requests to add E2E test results."""
    dashboard = request.app['dashboard']
    
    # Check if E2E test integration is enabled
    if not dashboard.enable_e2e_test_integration:
        return web.json_response({"error": "E2E test integration is disabled"}, status=400)
    
    # Get E2E test integration
    e2e_integration = dashboard.e2e_test_integration
    
    # Check request method
    if request.method != 'POST':
        return web.json_response({"error": "Only POST method is allowed"}, status=405)
    
    try:
        # Parse request data
        data = await request.json()
        
        # Validate required fields
        required_fields = ['test_id', 'visualizations']
        for field in required_fields:
            if field not in data:
                return web.json_response({"error": f"Missing required field: {field}"}, status=400)
        
        # Extract data
        test_id = data['test_id']
        visualizations = data['visualizations']
        
        # Add test result
        result = e2e_integration.add_test_result(test_id, data, visualizations)
        
        if result:
            return web.json_response({"status": "success", "test_id": test_id})
        else:
            return web.json_response({"error": "Failed to add test result"}, status=500)
    
    except Exception as e:
        logger.error(f"Error handling API request: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_e2e_test_monitoring(request: web.Request) -> web.Response:
    """Handle requests to the E2E test monitoring page."""
    dashboard = request.app['dashboard']
    theme = request.query.get('theme', dashboard.theme)
    
    # Render template
    return web.Response(
        text=dashboard.env.get_template('e2e_test_monitoring.html').render(
            theme=theme
        ),
        content_type='text/html'
    )

async def handle_start_e2e_test_monitoring(request: web.Request) -> web.Response:
    """Handle requests to start a new E2E test with monitoring."""
    dashboard = request.app['dashboard']
    
    # Generate test ID
    from datetime import datetime
    test_id = f"e2e_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start the monitoring process
    import subprocess
    import sys
    import os
    
    cmd = [
        sys.executable, 
        "-m", 
        "duckdb_api.distributed_testing.tests.realtime_monitoring",
        "--dashboard-url", f"http://{dashboard.host}:{dashboard.port}",
        "--test-id", test_id
    ]
    
    # Start the process in the background
    subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True  # Detach from parent process
    )
    
    # Redirect to monitoring page
    raise web.HTTPFound(f"/e2e-test-monitoring?test_id={test_id}")

async def handle_performance_analytics(request: web.Request) -> web.Response:
    """Handle requests to the performance analytics page."""
    dashboard = request.app['dashboard']
    theme = request.query.get('theme', dashboard.theme)
    time_range = int(request.query.get('time_range', '30'))
    
    # Get performance analytics data
    analytics_data = dashboard.get_performance_analytics_data(time_range)
    
    # Define metric display names
    metric_display = {
        'latency_ms': 'Latency (ms)',
        'throughput_items_sec': 'Throughput (items/sec)',
        'memory_usage_mb': 'Memory Usage (MB)',
        'cpu_usage_percent': 'CPU Usage (%)'
    }
    
    # Get embedded dashboards for this page
    embedded_dashboards = None
    if dashboard.enable_visualization_integration and dashboard.visualization_integration:
        try:
            embedded_dashboards = dashboard.visualization_integration.get_embedded_dashboards_for_page('performance-analytics')
        except Exception as e:
            logger.error(f"Error getting embedded dashboards: {e}")
    
    # Get HTML for embedded dashboards
    embedded_dashboard_html = {}
    if embedded_dashboards:
        for name, dash_details in embedded_dashboards.items():
            try:
                iframe_html = dashboard.visualization_integration.get_dashboard_iframe_html(
                    name=name,
                    width="100%",
                    height="600px"
                )
                embedded_dashboard_html[name] = {
                    "html": iframe_html,
                    "position": dash_details.get("position", "below"),
                    "title": dash_details.get("title", "Dashboard")
                }
            except Exception as e:
                logger.error(f"Error getting dashboard iframe HTML for {name}: {e}")
    
    # Try to generate a dashboard from analytics data if we have visualization but no embedded dashboards
    if (dashboard.enable_visualization_integration and dashboard.visualization_integration and 
        not embedded_dashboard_html and dashboard.visualization_integration.visualization_available and analytics_data):
        try:
            # Generate a dashboard from this data
            dashboard_name = f"analytics_dashboard_{int(time.time())}"
            dashboard_path = dashboard.visualization_integration.generate_dashboard_from_performance_data(
                performance_data=analytics_data,
                name=dashboard_name,
                title=f"Performance Analytics Dashboard ({time_range} days)"
            )
            
            if dashboard_path:
                # Register as an embedded dashboard
                dashboard_details = dashboard.visualization_integration.create_embedded_dashboard(
                    name=dashboard_name,
                    page="performance-analytics",
                    title=f"Performance Analytics Dashboard ({time_range} days)",
                    description=f"Automatically generated dashboard from performance analytics data for {time_range} day time range",
                    position="below"
                )
                
                if dashboard_details:
                    # Get the iframe HTML
                    iframe_html = dashboard.visualization_integration.get_dashboard_iframe_html(
                        name=dashboard_name,
                        width="100%",
                        height="600px"
                    )
                    
                    embedded_dashboard_html[dashboard_name] = {
                        "html": iframe_html,
                        "position": "below",
                        "title": f"Performance Analytics Dashboard ({time_range} days)"
                    }
        except Exception as e:
            logger.error(f"Error generating dashboard from analytics data: {e}")
    
    # Render template
    return web.Response(
        text=dashboard.env.get_template('performance_analytics.html').render(
            theme=theme,
            time_range=time_range,
            analytics_data=analytics_data,
            metric_display=metric_display,
            embedded_dashboards=embedded_dashboard_html,
            visualization_enabled=dashboard.enable_visualization_integration
        ),
        content_type='text/html'
    )

async def handle_run_performance_analytics(request: web.Request) -> web.Response:
    """Handle requests to run performance analytics."""
    dashboard = request.app['dashboard']
    
    # Start the performance analytics process
    import subprocess
    import sys
    import os
    
    time_range = request.query.get('time_range', '30')
    
    cmd = [
        sys.executable, 
        "-m", 
        "duckdb_api.distributed_testing.tests.performance_analytics",
        "--time-range", time_range,
        "--generate-report",
        "--upload-to-dashboard",
        "--dashboard-url", f"http://{dashboard.host}:{dashboard.port}"
    ]
    
    # Start the process in the background
    subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True  # Detach from parent process
    )
    
    # Redirect back to performance analytics page
    raise web.HTTPFound(f"/performance-analytics?time_range={time_range}")

async def handle_api_performance_analytics(request: web.Request) -> web.Response:
    """Handle API requests to add performance analytics data."""
    dashboard = request.app['dashboard']
    
    # Check request method
    if request.method != 'POST':
        return web.json_response({"error": "Only POST method is allowed"}, status=405)
    
    try:
        # Parse request data
        data = await request.json()
        
        # Store performance analytics data
        dashboard.store_performance_analytics_data(data)
        
        return web.json_response({"status": "success"})
    except Exception as e:
        logger.error(f"Failed to process performance analytics data: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def handle_dashboard_management(request: web.Request) -> web.Response:
    """Handle requests to the dashboard management page."""
    dashboard = request.app['dashboard']
    theme = request.query.get('theme', dashboard.theme)
    
    # Check if visualization integration is enabled
    if not dashboard.enable_visualization_integration or not dashboard.visualization_integration:
        return web.Response(
            text=f"""
            <html>
                <head>
                    <title>Dashboard Management</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; padding: 20px; background-color: {theme == 'dark' and '#1a1a1a' or '#f5f5f5'}; color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                        .message {{ padding: 20px; background-color: {theme == 'dark' and '#333' or '#fff'}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                        h1 {{ color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                        a {{ color: {theme == 'dark' and '#4dabf7' or '#007bff'}; text-decoration: none; }}
                        a:hover {{ text-decoration: underline; }}
                    </style>
                </head>
                <body>
                    <div class="message">
                        <h1>Dashboard Management</h1>
                        <p>Advanced Visualization System integration is not enabled.</p>
                        <p>To enable it, restart the monitoring dashboard with the <code>--enable-visualization-integration</code> flag.</p>
                        <p><a href="/">Back to Dashboard</a></p>
                    </div>
                </body>
            </html>
            """,
            content_type='text/html'
        )
    
    # Get action from query
    action = request.query.get('action', 'list')
    
    # Handle list action (default)
    if action == 'list':
        # Get all embedded dashboards
        all_dashboards = dashboard.visualization_integration.embedded_dashboards
        
        # Get available templates and components
        templates = dashboard.visualization_integration.list_available_templates()
        components = dashboard.visualization_integration.list_available_components()
        
        # Render template
        return web.Response(
            text=dashboard.env.get_template('dashboard_management.html').render(
                theme=theme,
                dashboards=all_dashboards,
                templates=templates,
                components=components,
                pages=[
                    {"id": "index", "name": "Overview"},
                    {"id": "workers", "name": "Workers"},
                    {"id": "tasks", "name": "Tasks"},
                    {"id": "results", "name": "Results"},
                    {"id": "performance-analytics", "name": "Performance Analytics"},
                    {"id": "e2e-test-results", "name": "E2E Test Results"}
                ],
                positions=[
                    {"id": "above", "name": "Above Content"},
                    {"id": "below", "name": "Below Content"},
                    {"id": "tab", "name": "In Tab"}
                ]
            ),
            content_type='text/html'
        )
    
    # Handle create action
    elif action == 'create':
        # Get parameters from query
        name = request.query.get('name')
        page = request.query.get('page')
        template = request.query.get('template')
        title = request.query.get('title')
        description = request.query.get('description')
        position = request.query.get('position', 'below')
        
        # Validate required parameters
        if not name or not page or not template:
            return web.Response(
                text=f"""
                <html>
                    <head>
                        <title>Create Dashboard Error</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; padding: 20px; background-color: {theme == 'dark' and '#1a1a1a' or '#f5f5f5'}; color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                            .message {{ padding: 20px; background-color: {theme == 'dark' and '#333' or '#fff'}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                            h1 {{ color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                            .error {{ color: #dc3545; }}
                            a {{ color: {theme == 'dark' and '#4dabf7' or '#007bff'}; text-decoration: none; }}
                            a:hover {{ text-decoration: underline; }}
                        </style>
                    </head>
                    <body>
                        <div class="message">
                            <h1>Create Dashboard Error</h1>
                            <p class="error">Missing required parameters: name, page, and template are required.</p>
                            <p><a href="/dashboards">Back to Dashboard Management</a></p>
                        </div>
                    </body>
                </html>
                """,
                content_type='text/html'
            )
        
        # Create dashboard
        try:
            result = dashboard.visualization_integration.create_embedded_dashboard(
                name=name,
                page=page,
                template=template,
                title=title,
                description=description,
                position=position
            )
            
            if result:
                # Redirect to dashboard management page
                raise web.HTTPFound('/dashboards?message=Dashboard created successfully')
            else:
                return web.Response(
                    text=f"""
                    <html>
                        <head>
                            <title>Create Dashboard Error</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; padding: 20px; background-color: {theme == 'dark' and '#1a1a1a' or '#f5f5f5'}; color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                                .message {{ padding: 20px; background-color: {theme == 'dark' and '#333' or '#fff'}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                                h1 {{ color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                                .error {{ color: #dc3545; }}
                                a {{ color: {theme == 'dark' and '#4dabf7' or '#007bff'}; text-decoration: none; }}
                                a:hover {{ text-decoration: underline; }}
                            </style>
                        </head>
                        <body>
                            <div class="message">
                                <h1>Create Dashboard Error</h1>
                                <p class="error">Failed to create dashboard. Please check logs for details.</p>
                                <p><a href="/dashboards">Back to Dashboard Management</a></p>
                            </div>
                        </body>
                    </html>
                    """,
                    content_type='text/html'
                )
        except Exception as e:
            return web.Response(
                text=f"""
                <html>
                    <head>
                        <title>Create Dashboard Error</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; padding: 20px; background-color: {theme == 'dark' and '#1a1a1a' or '#f5f5f5'}; color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                            .message {{ padding: 20px; background-color: {theme == 'dark' and '#333' or '#fff'}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                            h1 {{ color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                            .error {{ color: #dc3545; }}
                            a {{ color: {theme == 'dark' and '#4dabf7' or '#007bff'}; text-decoration: none; }}
                            a:hover {{ text-decoration: underline; }}
                        </style>
                    </head>
                    <body>
                        <div class="message">
                            <h1>Create Dashboard Error</h1>
                            <p class="error">Error: {str(e)}</p>
                            <p><a href="/dashboards">Back to Dashboard Management</a></p>
                        </div>
                    </body>
                </html>
                """,
                content_type='text/html'
            )
    
    # Handle remove action
    elif action == 'remove':
        # Get parameters from query
        name = request.query.get('name')
        
        # Validate required parameters
        if not name:
            return web.Response(
                text=f"""
                <html>
                    <head>
                        <title>Remove Dashboard Error</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; padding: 20px; background-color: {theme == 'dark' and '#1a1a1a' or '#f5f5f5'}; color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                            .message {{ padding: 20px; background-color: {theme == 'dark' and '#333' or '#fff'}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                            h1 {{ color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                            .error {{ color: #dc3545; }}
                            a {{ color: {theme == 'dark' and '#4dabf7' or '#007bff'}; text-decoration: none; }}
                            a:hover {{ text-decoration: underline; }}
                        </style>
                    </head>
                    <body>
                        <div class="message">
                            <h1>Remove Dashboard Error</h1>
                            <p class="error">Missing required parameter: name is required.</p>
                            <p><a href="/dashboards">Back to Dashboard Management</a></p>
                        </div>
                    </body>
                </html>
                """,
                content_type='text/html'
            )
        
        # Remove dashboard
        try:
            result = dashboard.visualization_integration.remove_embedded_dashboard(name)
            
            if result:
                # Redirect to dashboard management page
                raise web.HTTPFound('/dashboards?message=Dashboard removed successfully')
            else:
                return web.Response(
                    text=f"""
                    <html>
                        <head>
                            <title>Remove Dashboard Error</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; padding: 20px; background-color: {theme == 'dark' and '#1a1a1a' or '#f5f5f5'}; color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                                .message {{ padding: 20px; background-color: {theme == 'dark' and '#333' or '#fff'}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                                h1 {{ color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                                .error {{ color: #dc3545; }}
                                a {{ color: {theme == 'dark' and '#4dabf7' or '#007bff'}; text-decoration: none; }}
                                a:hover {{ text-decoration: underline; }}
                            </style>
                        </head>
                        <body>
                            <div class="message">
                                <h1>Remove Dashboard Error</h1>
                                <p class="error">Failed to remove dashboard. Dashboard not found or could not be removed.</p>
                                <p><a href="/dashboards">Back to Dashboard Management</a></p>
                            </div>
                        </body>
                    </html>
                    """,
                    content_type='text/html'
                )
        except Exception as e:
            return web.Response(
                text=f"""
                <html>
                    <head>
                        <title>Remove Dashboard Error</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; padding: 20px; background-color: {theme == 'dark' and '#1a1a1a' or '#f5f5f5'}; color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                            .message {{ padding: 20px; background-color: {theme == 'dark' and '#333' or '#fff'}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                            h1 {{ color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                            .error {{ color: #dc3545; }}
                            a {{ color: {theme == 'dark' and '#4dabf7' or '#007bff'}; text-decoration: none; }}
                            a:hover {{ text-decoration: underline; }}
                        </style>
                    </head>
                    <body>
                        <div class="message">
                            <h1>Remove Dashboard Error</h1>
                            <p class="error">Error: {str(e)}</p>
                            <p><a href="/dashboards">Back to Dashboard Management</a></p>
                        </div>
                    </body>
                </html>
                """,
                content_type='text/html'
            )
    
    # Handle update action
    elif action == 'update':
        # Get parameters from query
        name = request.query.get('name')
        title = request.query.get('title')
        description = request.query.get('description')
        position = request.query.get('position')
        page = request.query.get('page')
        
        # Validate required parameters
        if not name:
            return web.Response(
                text=f"""
                <html>
                    <head>
                        <title>Update Dashboard Error</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; padding: 20px; background-color: {theme == 'dark' and '#1a1a1a' or '#f5f5f5'}; color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                            .message {{ padding: 20px; background-color: {theme == 'dark' and '#333' or '#fff'}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                            h1 {{ color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                            .error {{ color: #dc3545; }}
                            a {{ color: {theme == 'dark' and '#4dabf7' or '#007bff'}; text-decoration: none; }}
                            a:hover {{ text-decoration: underline; }}
                        </style>
                    </head>
                    <body>
                        <div class="message">
                            <h1>Update Dashboard Error</h1>
                            <p class="error">Missing required parameter: name is required.</p>
                            <p><a href="/dashboards">Back to Dashboard Management</a></p>
                        </div>
                    </body>
                </html>
                """,
                content_type='text/html'
            )
        
        # Update dashboard
        try:
            result = dashboard.visualization_integration.update_embedded_dashboard(
                name=name,
                title=title,
                description=description,
                position=position,
                page=page
            )
            
            if result:
                # Redirect to dashboard management page
                raise web.HTTPFound('/dashboards?message=Dashboard updated successfully')
            else:
                return web.Response(
                    text=f"""
                    <html>
                        <head>
                            <title>Update Dashboard Error</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; padding: 20px; background-color: {theme == 'dark' and '#1a1a1a' or '#f5f5f5'}; color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                                .message {{ padding: 20px; background-color: {theme == 'dark' and '#333' or '#fff'}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                                h1 {{ color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                                .error {{ color: #dc3545; }}
                                a {{ color: {theme == 'dark' and '#4dabf7' or '#007bff'}; text-decoration: none; }}
                                a:hover {{ text-decoration: underline; }}
                            </style>
                        </head>
                        <body>
                            <div class="message">
                                <h1>Update Dashboard Error</h1>
                                <p class="error">Failed to update dashboard. Dashboard not found or could not be updated.</p>
                                <p><a href="/dashboards">Back to Dashboard Management</a></p>
                            </div>
                        </body>
                    </html>
                    """,
                    content_type='text/html'
                )
        except Exception as e:
            return web.Response(
                text=f"""
                <html>
                    <head>
                        <title>Update Dashboard Error</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; padding: 20px; background-color: {theme == 'dark' and '#1a1a1a' or '#f5f5f5'}; color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                            .message {{ padding: 20px; background-color: {theme == 'dark' and '#333' or '#fff'}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                            h1 {{ color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                            .error {{ color: #dc3545; }}
                            a {{ color: {theme == 'dark' and '#4dabf7' or '#007bff'}; text-decoration: none; }}
                            a:hover {{ text-decoration: underline; }}
                        </style>
                    </head>
                    <body>
                        <div class="message">
                            <h1>Update Dashboard Error</h1>
                            <p class="error">Error: {str(e)}</p>
                            <p><a href="/dashboards">Back to Dashboard Management</a></p>
                        </div>
                    </body>
                </html>
                """,
                content_type='text/html'
            )
    
    # Handle unknown action
    else:
        return web.Response(
            text=f"""
            <html>
                <head>
                    <title>Dashboard Management Error</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; padding: 20px; background-color: {theme == 'dark' and '#1a1a1a' or '#f5f5f5'}; color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                        .message {{ padding: 20px; background-color: {theme == 'dark' and '#333' or '#fff'}; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                        h1 {{ color: {theme == 'dark' and '#f5f5f5' or '#333'}; }}
                        .error {{ color: #dc3545; }}
                        a {{ color: {theme == 'dark' and '#4dabf7' or '#007bff'}; text-decoration: none; }}
                        a:hover {{ text-decoration: underline; }}
                    </style>
                </head>
                <body>
                    <div class="message">
                        <h1>Dashboard Management Error</h1>
                        <p class="error">Unknown action: {action}</p>
                        <p><a href="/dashboards">Back to Dashboard Management</a></p>
                    </div>
                </body>
            </html>
            """,
            content_type='text/html'
        )

def setup_routes(app: web.Application) -> None:
    """Set up routes for the monitoring dashboard."""
    app.router.add_get('/', handle_index)
    app.router.add_get('/workers', handle_workers)
    app.router.add_get('/tasks', handle_tasks)
    app.router.add_get('/results', handle_results_view)
    app.router.add_get('/e2e-test-results', handle_e2e_test_results)
    app.router.add_get('/e2e-test-results/{test_id}', handle_e2e_test_results)
    app.router.add_post('/api/e2e-test-results', handle_api_e2e_test_results)
    app.router.add_get('/e2e-test-monitoring', handle_e2e_test_monitoring)
    app.router.add_get('/start-e2e-test-monitoring', handle_start_e2e_test_monitoring)
    app.router.add_get('/performance-analytics', handle_performance_analytics)
    app.router.add_get('/run-performance-analytics', handle_run_performance_analytics)
    app.router.add_post('/api/performance-analytics', handle_api_performance_analytics)
    app.router.add_get('/dashboards', handle_dashboard_management)
    
    # Set up static routes
    app.router.add_static('/static/', app['dashboard'].static_dir)