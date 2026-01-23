#!/usr/bin/env python3
"""
Error Visualization Integration for the Distributed Testing Dashboard

This module integrates worker error reporting data with the monitoring dashboard,
providing comprehensive error visualization, pattern detection, and analysis.
"""

import os
import json
import logging
import anyio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import traceback

# Set up logging
logger = logging.getLogger(__name__)

# Try to import plotting libraries
try:
    import plotly
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    has_plotly = True
except ImportError:
    # Fall back to empty implementations if plotly is not available
    has_plotly = False
    logger.warning("Plotly not available. Visualization features will be limited.")
    # Create dummy plotly module with required components
    class DummyPlotlyUtils:
        class PlotlyJSONEncoder:
            def encode(self, obj):
                return "{}"
    
    class DummyPlotly:
        utils = DummyPlotlyUtils()
    
    plotly = DummyPlotly()

try:
    import pandas as pd
    import numpy as np
    has_pandas = True
except ImportError:
    has_pandas = False
    logger.warning("Pandas not available. Data processing features will be limited.")
    # Create dummy pandas and numpy modules
    class DummyPandas:
        def DataFrame(self, data):
            return data
    
    class DummyNumPy:
        pass
    
    pd = DummyPandas()
    np = DummyNumPy()

class ErrorVisualizationIntegration:
    """Integrates error data with the monitoring dashboard."""
    
    def __init__(self, 
                 output_dir: str = "./error_visualizations",
                 db_path: Optional[str] = None,
                 coordinator_url: Optional[str] = None,
                 websocket_manager: Optional[Any] = None):
        """Initialize the error visualization integration.
        
        Args:
            output_dir: Directory to store generated visualizations
            db_path: Path to the DuckDB database file (optional)
            coordinator_url: URL of the coordinator service (optional)
            websocket_manager: WebSocket manager for real-time updates (optional)
        """
        self.output_dir = output_dir
        self.db_path = db_path
        self.coordinator_url = coordinator_url
        self.websocket_manager = websocket_manager
        self.error_cache = {}
        self.last_update = datetime.now()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Error Visualization Integration initialized with output directory: {output_dir}")
        
    async def get_error_data(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get error data for the specified time range.
        
        Args:
            time_range_hours: Time range in hours
            
        Returns:
            Dictionary containing error data for visualization
        """
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_range_hours)
        
        # Check if cached data is available and still valid
        cache_key = f"errors_{time_range_hours}"
        if cache_key in self.error_cache:
            cache_time, cache_data = self.error_cache[cache_key]
            # Use cache if it's less than 5 minutes old
            if (datetime.now() - cache_time).total_seconds() < 300:
                return cache_data
        
        # Collect error data from various sources
        try:
            # If DuckDB is available, query from database
            if self.db_path:
                error_data = await self._get_error_data_from_db(start_time, end_time)
            # If coordinator URL is available, fetch from coordinator API
            elif self.coordinator_url:
                error_data = await self._get_error_data_from_coordinator(start_time, end_time)
            else:
                # Fallback to collected local files (e.g., from log directories)
                error_data = await self._get_error_data_from_files(start_time, end_time)
            
            # Process and analyze error data
            processed_data = await self._process_error_data(error_data, time_range_hours)
            
            # Cache the processed data
            self.error_cache[cache_key] = (datetime.now(), processed_data)
            self.last_update = datetime.now()
            
            return processed_data
        except Exception as e:
            logger.error(f"Error getting error data: {e}")
            logger.error(traceback.format_exc())
            
            # Return empty data structure on error
            return {
                "summary": None,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _get_error_data_from_db(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get error data from DuckDB database.
        
        Args:
            start_time: Start time for the query
            end_time: End time for the query
            
        Returns:
            List of error records
        """
        try:
            import duckdb
            
            # Connect to database
            conn = duckdb.connect(self.db_path)
            
            # Convert datetimes to strings for SQL query
            start_str = start_time.isoformat()
            end_str = end_time.isoformat()
            
            # Query error reports from database
            query = f"""
                SELECT * FROM worker_error_reports
                WHERE timestamp >= '{start_str}' AND timestamp <= '{end_str}'
                ORDER BY timestamp DESC
            """
            
            # Execute query and convert to dictionaries
            result = conn.execute(query).fetchall()
            column_names = [desc[0] for desc in conn.description]
            
            # Convert to list of dictionaries with parsed JSON fields
            errors = []
            for row in result:
                error_dict = dict(zip(column_names, row))
                
                # Parse JSON fields
                for field in ['system_context', 'hardware_context', 'error_frequency']:
                    if field in error_dict and isinstance(error_dict[field], str):
                        try:
                            error_dict[field] = json.loads(error_dict[field])
                        except json.JSONDecodeError:
                            error_dict[field] = {}
                
                errors.append(error_dict)
            
            conn.close()
            logger.info(f"Retrieved {len(errors)} error records from database")
            return errors
        except Exception as e:
            logger.error(f"Error querying database: {e}")
            logger.error(traceback.format_exc())
            return []
    
    async def _get_error_data_from_coordinator(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get error data from coordinator API.
        
        Args:
            start_time: Start time for the query
            end_time: End time for the query
            
        Returns:
            List of error records
        """
        try:
            import aiohttp
            
            # Format the date range parameters
            params = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
            
            # Create HTTP session
            async with aiohttp.ClientSession() as session:
                # Fetch error data from coordinator API
                async with session.get(f"{self.coordinator_url}/api/errors", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Retrieved {len(data['errors'])} error records from coordinator API")
                        return data['errors']
                    else:
                        logger.error(f"Error response from coordinator API: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching from coordinator API: {e}")
            logger.error(traceback.format_exc())
            return []
    
    async def _get_error_data_from_files(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get error data from local log files.
        
        Args:
            start_time: Start time for the query
            end_time: End time for the query
            
        Returns:
            List of error records
        """
        try:
            # Look for log directories in a common parent directory
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dirs = [d for d in os.listdir(parent_dir) if d.startswith('e2e_test_logs_')]
            
            # Collect errors from all log directories
            all_errors = []
            
            for log_dir in log_dirs:
                log_dir_path = os.path.join(parent_dir, log_dir)
                
                # Skip if not a directory
                if not os.path.isdir(log_dir_path):
                    continue
                
                # Look for worker log files
                worker_logs = [f for f in os.listdir(log_dir_path) if f.startswith('worker-') and f.endswith('.log')]
                
                for log_file in worker_logs:
                    log_file_path = os.path.join(log_dir_path, log_file)
                    
                    # Extract worker ID from filename
                    worker_id = log_file.replace('worker-', '').replace('.log', '')
                    
                    # Parse log file for error reports
                    with open(log_file_path, 'r') as f:
                        log_content = f.read()
                        
                        # Look for error report JSON blocks
                        error_blocks = self._extract_error_blocks(log_content)
                        
                        for error_block in error_blocks:
                            try:
                                error_data = json.loads(error_block)
                                
                                # Parse timestamp and check if it's within the time range
                                if 'timestamp' in error_data:
                                    try:
                                        error_time = datetime.fromisoformat(error_data['timestamp'])
                                        if start_time <= error_time <= end_time:
                                            # Add to list
                                            all_errors.append(error_data)
                                    except ValueError:
                                        # Skip errors with invalid timestamps
                                        pass
                            except json.JSONDecodeError:
                                # Skip invalid JSON
                                pass
            
            logger.info(f"Retrieved {len(all_errors)} error records from log files")
            return all_errors
        except Exception as e:
            logger.error(f"Error extracting errors from files: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def _extract_error_blocks(self, log_content: str) -> List[str]:
        """Extract error report JSON blocks from log content.
        
        Args:
            log_content: Content of log file
            
        Returns:
            List of JSON error blocks
        """
        error_blocks = []
        
        # Look for ERROR_REPORT: markers followed by JSON
        lines = log_content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'ERROR_REPORT:' in line:
                # Start capturing the JSON block
                json_block = ''
                i += 1
                # Assume JSON block starts with { and ends with }
                while i < len(lines) and '{' in lines[i]:
                    json_start = lines[i].find('{')
                    json_block = lines[i][json_start:]
                    
                    # Find the matching closing brace
                    brace_count = 1
                    j = i + 1
                    while j < len(lines) and brace_count > 0:
                        for char in lines[j]:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    # Found the end of the JSON block
                                    json_block += lines[j].split('}')[0] + '}'
                                    break
                        if brace_count > 0:
                            json_block += lines[j] + '\n'
                        j += 1
                    
                    i = j
                    break
                
                if json_block:
                    error_blocks.append(json_block)
            i += 1
        
        return error_blocks
    
    async def report_error(self, error_data: Dict[str, Any]) -> bool:
        """Report a new error for real-time monitoring.
        
        This method stores the error in the database and broadcasts it to WebSocket clients.
        
        Args:
            error_data: Error data to report
            
        Returns:
            True if error was successfully reported, False otherwise
        """
        try:
            # Ensure timestamp is present
            if 'timestamp' not in error_data:
                error_data['timestamp'] = datetime.now().isoformat()
            
            # Ensure worker_id is present
            if 'worker_id' not in error_data:
                error_data['worker_id'] = 'unknown'
                
            # Add the error to the database if available
            if self.db_path:
                await self._store_error_in_db(error_data)
            
            # Prepare the error for display
            display_error = self._prepare_error_for_display(error_data)
            
            # Determine which time ranges this error applies to
            time_ranges = [1, 6, 24, 168]  # 1h, 6h, 24h, 7d
            
            # Send real-time update to WebSocket clients if available
            if self.websocket_manager:
                for time_range in time_ranges:
                    # Send update for each time range
                    await self.websocket_manager.broadcast(
                        topic=f"error_visualization:{time_range}",
                        message={
                            "type": "error_visualization_update",
                            "data": {
                                "error": display_error,
                                "time_range": time_range
                            }
                        }
                    )
                
                # Also send to the general error visualization topic
                await self.websocket_manager.broadcast(
                    topic="error_visualization",
                    message={
                        "type": "error_visualization_update",
                        "data": {
                            "error": display_error
                        }
                    }
                )
                
                logger.info(f"Error broadcast to WebSocket clients: {error_data.get('error_type', 'Unknown error')}")
            
            # Invalidate cache for all time ranges
            for time_range in time_ranges:
                cache_key = f"errors_{time_range}"
                if cache_key in self.error_cache:
                    del self.error_cache[cache_key]
            
            return True
        except Exception as e:
            logger.error(f"Error reporting new error: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def _store_error_in_db(self, error_data: Dict[str, Any]) -> bool:
        """Store an error in the database.
        
        Args:
            error_data: Error data to store
            
        Returns:
            True if error was successfully stored, False otherwise
        """
        try:
            import duckdb
            
            # Connect to database
            conn = duckdb.connect(self.db_path)
            
            # Check if the worker_error_reports table exists
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='worker_error_reports'"
            ).fetchone()
            
            # Create table if it doesn't exist
            if not table_exists:
                conn.execute("""
                    CREATE TABLE worker_error_reports (
                        id INTEGER PRIMARY KEY,
                        timestamp TIMESTAMP,
                        worker_id VARCHAR,
                        type VARCHAR,
                        error_category VARCHAR,
                        message VARCHAR,
                        traceback VARCHAR,
                        system_context JSON,
                        hardware_context JSON,
                        error_frequency JSON
                    )
                """)
            
            # Convert JSON fields to strings
            system_context = json.dumps(error_data.get('system_context', {}))
            hardware_context = json.dumps(error_data.get('hardware_context', {}))
            error_frequency = json.dumps(error_data.get('error_frequency', {}))
            
            # Insert error into database
            conn.execute("""
                INSERT INTO worker_error_reports 
                (timestamp, worker_id, type, error_category, message, traceback, 
                 system_context, hardware_context, error_frequency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                error_data.get('timestamp'),
                error_data.get('worker_id'),
                error_data.get('type', 'Unknown'),
                error_data.get('error_category', 'UNKNOWN_ERROR'),
                error_data.get('message', ''),
                error_data.get('traceback', ''),
                system_context,
                hardware_context,
                error_frequency
            ))
            
            conn.close()
            logger.info(f"Error stored in database: {error_data.get('error_type', 'Unknown error')}")
            return True
        except Exception as e:
            logger.error(f"Error storing error in database: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def _process_error_data(self, error_data: List[Dict[str, Any]], time_range_hours: int) -> Dict[str, Any]:
        """Process error data for visualization.
        
        Args:
            error_data: List of error records
            time_range_hours: Time range in hours
            
        Returns:
            Processed error data ready for visualization
        """
        if not error_data:
            return {
                "summary": None,
                "timestamp": datetime.now().isoformat()
            }
        
        # Initialize result structure
        result = {
            "summary": {},
            "timestamp": datetime.now().isoformat(),
            "recent_errors": [],
            "error_distribution": None,
            "error_patterns": None,
            "worker_errors": None,
            "hardware_errors": None
        }
        
        # Calculate summary statistics
        total_errors = len(error_data)
        recurring_errors = sum(1 for e in error_data if self._is_recurring_error(e))
        
        # Count errors by category
        error_categories = Counter(e.get('error_category', 'UNKNOWN') for e in error_data)
        resource_errors = sum(error_categories.get(cat, 0) for cat in ['RESOURCE_EXHAUSTED', 'RESOURCE_UNAVAILABLE', 'RESOURCE_NOT_FOUND'])
        network_errors = sum(error_categories.get(cat, 0) for cat in ['NETWORK_CONNECTION_ERROR', 'NETWORK_TIMEOUT', 'NETWORK_SERVER_ERROR'])
        hardware_errors = sum(error_categories.get(cat, 0) for cat in ['HARDWARE_NOT_AVAILABLE', 'HARDWARE_MISMATCH', 'HARDWARE_COMPATIBILITY_ERROR'])
        
        # Count critical hardware errors
        critical_hardware_errors = sum(1 for e in error_data 
                                       if e.get('error_category') in ['HARDWARE_NOT_AVAILABLE', 'HARDWARE_MISMATCH', 'HARDWARE_COMPATIBILITY_ERROR'] 
                                       and self._is_critical_error(e))
        
        # Summary data
        result["summary"] = {
            "total_errors": total_errors,
            "recurring_errors": recurring_errors,
            "resource_errors": resource_errors,
            "network_errors": network_errors,
            "hardware_errors": hardware_errors,
            "critical_hardware_errors": critical_hardware_errors,
            "time_range_hours": time_range_hours
        }
        
        # Recent errors (limited to last 100)
        result["recent_errors"] = [self._prepare_error_for_display(e) for e in sorted(error_data, key=lambda x: x.get('timestamp', ''), reverse=True)[:100]]
        
        try:
            # Generate error distribution visualization
            result["error_distribution"] = await self._generate_error_distribution(error_data)
            if result["error_distribution"] is None:
                # Provide fallback data if generation fails
                result["error_distribution"] = {
                    'chart_data': {'data': [], 'layout': {}},
                    'categories': []
                }
            
            # Generate error pattern analysis
            result["error_patterns"] = await self._generate_error_patterns(error_data)
            if result["error_patterns"] is None:
                # Provide fallback data if generation fails
                result["error_patterns"] = {
                    'chart_data': {'data': [], 'layout': {}},
                    'top_patterns': []
                }
            
            # Generate worker error analysis
            result["worker_errors"] = await self._generate_worker_error_analysis(error_data)
            if result["worker_errors"] is None:
                # Provide fallback data if generation fails
                result["worker_errors"] = {
                    'chart_data': {'data': [], 'layout': {}},
                    'worker_stats': []
                }
            
            # Generate hardware error analysis
            result["hardware_errors"] = await self._generate_hardware_error_analysis(error_data)
            if result["hardware_errors"] is None:
                # Provide fallback data if generation fails
                result["hardware_errors"] = {
                    'chart_data': {'data': [], 'layout': {}},
                    'hardware_status': {},
                    'recent_errors': []
                }
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            logger.error(traceback.format_exc())
            # Ensure minimal data structure is maintained even if visualization fails
            if result["error_distribution"] is None:
                result["error_distribution"] = {'chart_data': {'data': [], 'layout': {}}, 'categories': []}
            if result["error_patterns"] is None:
                result["error_patterns"] = {'chart_data': {'data': [], 'layout': {}}, 'top_patterns': []}
            if result["worker_errors"] is None:
                result["worker_errors"] = {'chart_data': {'data': [], 'layout': {}}, 'worker_stats': []}
            if result["hardware_errors"] is None:
                result["hardware_errors"] = {'chart_data': {'data': [], 'layout': {}}, 'hardware_status': {}, 'recent_errors': []}
        
        return result
    
    def _is_recurring_error(self, error: Dict[str, Any]) -> bool:
        """Check if an error is recurring.
        
        Args:
            error: Error record
            
        Returns:
            True if the error is recurring, False otherwise
        """
        if 'error_frequency' in error:
            freq = error['error_frequency']
            if 'recurring' in freq and freq['recurring']:
                return True
            
            # Check if error has occurred multiple times
            if 'similar_message' in freq:
                if freq['similar_message'].get('last_1h', 0) > 2:
                    return True
                if freq['similar_message'].get('last_6h', 0) > 5:
                    return True
        
        return False
    
    def _is_critical_error(self, error: Dict[str, Any]) -> bool:
        """Check if an error is critical.
        
        Args:
            error: Error record
            
        Returns:
            True if the error is critical, False otherwise
        """
        # Check error category
        critical_categories = [
            'HARDWARE_NOT_AVAILABLE',
            'RESOURCE_EXHAUSTED',
            'WORKER_CRASHED'
        ]
        
        if error.get('error_category') in critical_categories:
            return True
        
        # Check if hardware status indicates issues
        if 'hardware_context' in error and 'hardware_status' in error['hardware_context']:
            hw_status = error['hardware_context']['hardware_status']
            if hw_status.get('overheating') or hw_status.get('memory_pressure'):
                return True
        
        # Check system context for high resource usage
        if 'system_context' in error and 'metrics' in error['system_context']:
            metrics = error['system_context']['metrics']
            
            # Check CPU usage
            if 'cpu' in metrics and metrics['cpu'].get('percent', 0) > 90:
                return True
                
            # Check memory usage
            if 'memory' in metrics and metrics['memory'].get('used_percent', 0) > 95:
                return True
                
            # Check disk usage
            if 'disk' in metrics and metrics['disk'].get('used_percent', 0) > 95:
                return True
        
        return False
    
    def _prepare_error_for_display(self, error: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare error record for display in the UI.
        
        Args:
            error: Error record
            
        Returns:
            Processed error record for display
        """
        # Create a copy of the error to avoid modifying the original
        display_error = error.copy()
        
        # Format timestamp for display
        if 'timestamp' in display_error:
            try:
                timestamp = datetime.fromisoformat(display_error['timestamp'])
                display_error['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                display_error['timestamp'] = str(display_error['timestamp'])
        
        # Add flag for recurring errors
        display_error['is_recurring'] = self._is_recurring_error(error)
        
        # Add flag for critical errors
        display_error['is_critical'] = self._is_critical_error(error)
        
        return display_error
    
    async def _generate_error_distribution(self, error_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate error distribution visualization.
        
        Args:
            error_data: List of error records
            
        Returns:
            Error distribution visualization data
        """
        try:
            # Count errors by category
            error_categories = Counter(e.get('error_category', 'UNKNOWN_ERROR') for e in error_data)
            
            # Create categories dictionary for the return value
            categories_dict = [
                {'Category': category, 'Count': count}
                for category, count in sorted(error_categories.items(), key=lambda x: x[1], reverse=True)
            ]
            
            # If plotly is not available, return simplified data
            if not has_plotly:
                return {
                    'chart_data': {'data': [], 'layout': {}},
                    'categories': categories_dict
                }
            
            # If pandas is available, use dataframe for sorting
            if has_pandas:
                # Create dataframe
                df = pd.DataFrame({
                    'Category': list(error_categories.keys()),
                    'Count': list(error_categories.values())
                })
                
                # Sort by count descending
                df = df.sort_values('Count', ascending=False)
                
                # Update categories_dict to use dataframe
                categories_dict = df.to_dict('records')
            
            # Create figure
            fig = go.Figure(data=[
                go.Bar(
                    x=[item['Category'] for item in categories_dict],
                    y=[item['Count'] for item in categories_dict],
                    marker_color='rgba(158, 202, 225, 0.8)',
                    marker_line_color='rgb(8, 48, 107)',
                    marker_line_width=1.5,
                    opacity=0.8
                )
            ])
            
            fig.update_layout(
                title='Error Distribution by Category',
                xaxis_title='Error Category',
                yaxis_title='Count',
                template='plotly_white',
                xaxis={'categoryorder': 'total descending'}
            )
            
            # Convert to JSON for template rendering
            chart_json = json.dumps({
                'data': fig.data,
                'layout': fig.layout
            }, cls=plotly.utils.PlotlyJSONEncoder)
            
            return {
                'chart_data': json.loads(chart_json),
                'categories': categories_dict
            }
        except Exception as e:
            logger.error(f"Error generating error distribution: {e}")
            logger.error(traceback.format_exc())
            return {
                'chart_data': {'data': [], 'layout': {}},
                'categories': []
            }
    
    async def _generate_error_patterns(self, error_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate error pattern analysis.
        
        Args:
            error_data: List of error records
            
        Returns:
            Error pattern analysis data
        """
        try:
            # Extract error messages and types
            error_messages = [e.get('message', '') for e in error_data]
            error_types = [e.get('type', '') for e in error_data]
            
            # Create error patterns by analyzing message content
            patterns = []
            
            # Group by similar message patterns
            message_groups = self._group_similar_messages(error_messages)
            
            # Extract timestamps for pattern analysis
            timestamps = []
            for e in error_data:
                try:
                    if 'timestamp' in e:
                        timestamps.append(datetime.fromisoformat(e['timestamp']))
                    else:
                        timestamps.append(None)
                except (ValueError, TypeError):
                    timestamps.append(None)
            
            # Create pattern records
            for pattern, indices in message_groups.items():
                if len(indices) < 2:
                    continue  # Skip patterns with only one occurrence
                
                # Get related errors
                related_errors = [error_data[i] for i in indices]
                
                # Count occurrences by category
                categories = Counter(e.get('error_category', 'UNKNOWN') for e in related_errors)
                most_common_category = categories.most_common(1)[0][0] if categories else 'UNKNOWN'
                
                # Get time range
                pattern_timestamps = [timestamps[i] for i in indices if timestamps[i] is not None]
                first_seen = min(pattern_timestamps).strftime('%Y-%m-%d %H:%M:%S') if pattern_timestamps else 'Unknown'
                last_seen = max(pattern_timestamps).strftime('%Y-%m-%d %H:%M:%S') if pattern_timestamps else 'Unknown'
                
                patterns.append({
                    'pattern': pattern,
                    'occurrences': len(indices),
                    'category': most_common_category,
                    'first_seen': first_seen,
                    'last_seen': last_seen
                })
            
            # Sort patterns by occurrences (descending)
            patterns = sorted(patterns, key=lambda x: x['occurrences'], reverse=True)
            
            # Calculate pattern distribution by category
            pattern_categories = Counter(p['category'] for p in patterns)
            
            # If plotly is not available, return simplified data
            if not has_plotly:
                return {
                    'chart_data': {'data': [], 'layout': {}},
                    'top_patterns': patterns[:10]  # Limit to top 10 patterns
                }
            
            # Create category data for visualization
            category_names = list(pattern_categories.keys())
            category_values = list(pattern_categories.values())
            
            # Create figure for pattern distribution
            if has_pandas:
                df_patterns = pd.DataFrame({
                    'Category': category_names,
                    'Patterns': category_values
                })
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=df_patterns['Category'],
                        values=df_patterns['Patterns'],
                        hole=.3,
                        marker_colors=px.colors.qualitative.Set3
                    )
                ])
            else:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=category_names,
                        values=category_values,
                        hole=.3
                    )
                ])
            
            fig.update_layout(
                title='Error Pattern Distribution by Category',
                template='plotly_white'
            )
            
            # Convert to JSON for template rendering
            chart_json = json.dumps({
                'data': fig.data,
                'layout': fig.layout
            }, cls=plotly.utils.PlotlyJSONEncoder)
            
            return {
                'chart_data': json.loads(chart_json),
                'top_patterns': patterns[:10]  # Limit to top 10 patterns
            }
        except Exception as e:
            logger.error(f"Error generating error patterns: {e}")
            logger.error(traceback.format_exc())
            return {
                'chart_data': {'data': [], 'layout': {}},
                'top_patterns': []
            }
    
    def _group_similar_messages(self, messages: List[str]) -> Dict[str, List[int]]:
        """Group similar error messages.
        
        Args:
            messages: List of error messages
            
        Returns:
            Dictionary mapping pattern to list of indices
        """
        # Simple pattern extraction by removing specific details like IDs, timestamps, etc.
        def extract_pattern(message):
            # Remove numbers, UUIDs, timestamps
            import re
            pattern = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '<UUID>', message)
            pattern = re.sub(r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?\b', '<TIMESTAMP>', pattern)
            pattern = re.sub(r'\b\d+\b', '<NUM>', pattern)
            
            # Tokenize and keep the first 8-10 tokens (approximate pattern)
            tokens = pattern.split()
            if len(tokens) > 10:
                pattern = ' '.join(tokens[:10]) + '...'
            
            return pattern
        
        # Group messages by pattern
        pattern_groups = defaultdict(list)
        
        for i, message in enumerate(messages):
            pattern = extract_pattern(message)
            pattern_groups[pattern].append(i)
        
        return pattern_groups
    
    async def _generate_worker_error_analysis(self, error_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate worker error analysis.
        
        Args:
            error_data: List of error records
            
        Returns:
            Worker error analysis data
        """
        try:
            # Group errors by worker
            worker_errors = defaultdict(list)
            
            for error in error_data:
                worker_id = error.get('worker_id', 'unknown')
                worker_errors[worker_id].append(error)
            
            # Calculate worker statistics
            worker_stats = []
            
            for worker_id, errors in worker_errors.items():
                # Count errors
                error_count = len(errors)
                
                # Find most common error type
                error_types = Counter(e.get('type', 'Unknown') for e in errors)
                most_common_error = error_types.most_common(1)[0][0] if error_types else 'Unknown'
                
                # Get last error time
                last_error_time = "Unknown"
                try:
                    timestamps = [datetime.fromisoformat(e['timestamp']) for e in errors if 'timestamp' in e]
                    if timestamps:
                        last_error_time = max(timestamps).strftime('%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError):
                    pass
                
                # Determine worker status
                status = "STABLE"
                if error_count > 10:
                    status = "WARNING"
                
                # Check for critical errors
                critical_errors = sum(1 for e in errors if self._is_critical_error(e))
                if critical_errors > 0:
                    status = "CRITICAL"
                
                worker_stats.append({
                    'worker_id': worker_id,
                    'error_count': error_count,
                    'most_common_error': most_common_error,
                    'last_error_time': last_error_time,
                    'status': status,
                    'critical_errors': critical_errors
                })
            
            # Sort by error count (descending)
            worker_stats = sorted(worker_stats, key=lambda x: x['error_count'], reverse=True)
            
            # If plotly is not available, return simplified data
            if not has_plotly:
                return {
                    'chart_data': {'data': [], 'layout': {}},
                    'worker_stats': worker_stats
                }
            
            # Create chart based on worker stats
            worker_ids = [w['worker_id'] for w in worker_stats]
            total_errors = [w['error_count'] for w in worker_stats]
            critical_errors = [w['critical_errors'] for w in worker_stats]
            
            # Create figure for worker error distribution
            if has_pandas:
                df_workers = pd.DataFrame(worker_stats)
                
                # Create bar chart of error counts by worker
                fig = go.Figure()
                
                # Add bars for total errors
                fig.add_trace(go.Bar(
                    x=df_workers['worker_id'],
                    y=df_workers['error_count'],
                    name='Total Errors',
                    marker_color='lightblue'
                ))
                
                # Add bars for critical errors
                fig.add_trace(go.Bar(
                    x=df_workers['worker_id'],
                    y=df_workers['critical_errors'],
                    name='Critical Errors',
                    marker_color='red'
                ))
            else:
                # Create figure without pandas
                fig = go.Figure()
                
                # Add bars for total errors
                fig.add_trace(go.Bar(
                    x=worker_ids,
                    y=total_errors,
                    name='Total Errors',
                    marker_color='lightblue'
                ))
                
                # Add bars for critical errors
                fig.add_trace(go.Bar(
                    x=worker_ids,
                    y=critical_errors,
                    name='Critical Errors',
                    marker_color='red'
                ))
            
            fig.update_layout(
                title='Error Count by Worker',
                xaxis_title='Worker ID',
                yaxis_title='Error Count',
                barmode='group',
                template='plotly_white'
            )
            
            # Convert to JSON for template rendering
            chart_json = json.dumps({
                'data': fig.data,
                'layout': fig.layout
            }, cls=plotly.utils.PlotlyJSONEncoder)
            
            return {
                'chart_data': json.loads(chart_json),
                'worker_stats': worker_stats
            }
        except Exception as e:
            logger.error(f"Error generating worker error analysis: {e}")
            logger.error(traceback.format_exc())
            return {
                'chart_data': {'data': [], 'layout': {}},
                'worker_stats': []
            }
    
    async def _generate_hardware_error_analysis(self, error_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate hardware error analysis.
        
        Args:
            error_data: List of error records
            
        Returns:
            Hardware error analysis data
        """
        try:
            # Extract hardware-related errors
            hardware_errors = [e for e in error_data if e.get('error_category') in [
                'HARDWARE_NOT_AVAILABLE',
                'HARDWARE_MISMATCH',
                'HARDWARE_COMPATIBILITY_ERROR'
            ]]
            
            # Group by hardware type
            hardware_types = defaultdict(list)
            
            for error in hardware_errors:
                # Extract hardware context
                hardware_context = error.get('hardware_context', {})
                hardware_type = hardware_context.get('hardware_type', 'unknown')
                
                hardware_types[hardware_type].append(error)
            
            # Calculate hardware status information
            hardware_status = {}
            
            for hw_type, errors in hardware_types.items():
                error_count = len(errors)
                
                # Total tasks for this hardware type
                # This would ideally come from a task database, but we'll estimate based on error rate
                estimated_total_tasks = max(error_count * 10, 100)  # Assume error rate of ~10%
                
                error_rate = (error_count / estimated_total_tasks) * 100
                
                # Find most common error type
                error_messages = [e.get('message', '') for e in errors]
                error_patterns = self._group_similar_messages(error_messages)
                
                most_common_pattern = "Unknown"
                max_occurrences = 0
                
                for pattern, indices in error_patterns.items():
                    if len(indices) > max_occurrences:
                        most_common_pattern = pattern
                        max_occurrences = len(indices)
                
                # Determine hardware status
                status = "Stable"
                if error_rate > 1:
                    status = "Warning"
                if error_rate > 5:
                    status = "Critical"
                
                # Check for overheating or memory pressure
                for error in errors:
                    if error.get('hardware_context', {}).get('hardware_status', {}).get('overheating'):
                        status = "Critical"
                        break
                    if error.get('hardware_context', {}).get('hardware_status', {}).get('memory_pressure'):
                        status = "Critical"
                        break
                
                hardware_status[hw_type] = {
                    'error_count': error_count,
                    'error_rate': round(error_rate, 2),
                    'most_common_error': most_common_pattern,
                    'status': status
                }
            
            # Get recent hardware errors (last 50)
            recent_hardware_errors = []
            
            for error in sorted(hardware_errors, key=lambda x: x.get('timestamp', ''), reverse=True)[:50]:
                # Format for display
                hw_type = error.get('hardware_context', {}).get('hardware_type', 'unknown')
                error_type = error.get('type', 'Unknown')
                message = error.get('message', 'No message')
                worker_id = error.get('worker_id', 'unknown')
                
                # Format timestamp
                timestamp = "Unknown"
                if 'timestamp' in error:
                    try:
                        timestamp = datetime.fromisoformat(error['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    except (ValueError, TypeError):
                        pass
                
                recent_hardware_errors.append({
                    'hardware_type': hw_type,
                    'error_type': error_type,
                    'message': message,
                    'worker_id': worker_id,
                    'timestamp': timestamp
                })
            
            # If plotly is not available, return simplified data
            if not has_plotly:
                return {
                    'chart_data': {'data': [], 'layout': {}},
                    'hardware_status': hardware_status,
                    'recent_errors': recent_hardware_errors
                }
            
            # Create data for visualization
            hw_data = []
            for hw_type, stats in hardware_status.items():
                hw_data.append({
                    'hardware_type': hw_type,
                    'error_count': stats['error_count'],
                    'error_rate': stats['error_rate'],
                    'status': stats['status']
                })
            
            # Create color mapping for status
            color_map = {
                'Critical': 'red',
                'Warning': 'orange',
                'Stable': 'green'
            }
            
            # Create visualization based on hardware status
            if has_pandas:
                df_hw = pd.DataFrame(hw_data)
                
                # Create scatter plot of error rate by hardware type
                fig = go.Figure()
                
                for status in ['Critical', 'Warning', 'Stable']:
                    df_subset = df_hw[df_hw['status'] == status]
                    
                    if not df_subset.empty:
                        fig.add_trace(go.Bar(
                            x=df_subset['hardware_type'],
                            y=df_subset['error_rate'],
                            name=status,
                            marker_color=color_map[status]
                        ))
            else:
                # Create figure without pandas
                fig = go.Figure()
                
                # Group data by status
                status_groups = defaultdict(list)
                for item in hw_data:
                    status_groups[item['status']].append(item)
                
                for status, items in status_groups.items():
                    if items:
                        fig.add_trace(go.Bar(
                            x=[item['hardware_type'] for item in items],
                            y=[item['error_rate'] for item in items],
                            name=status,
                            marker_color=color_map.get(status, 'blue')
                        ))
            
            fig.update_layout(
                title='Hardware Error Rate by Type',
                xaxis_title='Hardware Type',
                yaxis_title='Error Rate (%)',
                template='plotly_white',
                barmode='group'
            )
            
            # Convert to JSON for template rendering
            chart_json = json.dumps({
                'data': fig.data,
                'layout': fig.layout
            }, cls=plotly.utils.PlotlyJSONEncoder)
            
            return {
                'chart_data': json.loads(chart_json),
                'hardware_status': hardware_status,
                'recent_errors': recent_hardware_errors
            }
        except Exception as e:
            logger.error(f"Error generating hardware error analysis: {e}")
            logger.error(traceback.format_exc())
            return {
                'chart_data': {'data': [], 'layout': {}},
                'hardware_status': {},
                'recent_errors': []
            }