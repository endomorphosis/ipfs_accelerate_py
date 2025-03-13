"""
Monitoring Dashboard Routes

This module defines the web routes for the Monitoring Dashboard.
"""

import json
import logging
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
    
    # Render template
    return web.Response(
        text=dashboard.env.get_template('index.html').render(
            coordinator_status=coordinator_status,
            workers_status=workers_status,
            tasks_status=tasks_status,
            theme=theme,
            refresh_interval=dashboard.refresh_interval
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
    
    # Render template
    return web.Response(
        text=dashboard.env.get_template('results.html').render(
            result_summary=result_summary,
            theme=theme,
            refresh_interval=dashboard.refresh_interval,
            time_range=time_range
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
    
    # Render template
    return web.Response(
        text=dashboard.env.get_template('performance_analytics.html').render(
            theme=theme,
            time_range=time_range,
            analytics_data=analytics_data,
            metric_display=metric_display
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
    
    # Set up static routes
    app.router.add_static('/static/', app['dashboard'].static_dir)