#!/usr/bin/env python3
"""
Enhanced Error Handling Integration for Distributed Testing Framework

This module integrates the performance-based error recovery system with the coordinator
and provides a centralized API for error handling, recovery, and performance tracking.

Usage:
    Import this module in coordinator.py to enhance error recovery capabilities.
"""

import logging
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("enhanced_error_handling")

# Import error handling and recovery components
from distributed_error_handler import DistributedErrorHandler, ErrorReport
from error_recovery_strategies import EnhancedErrorRecoveryManager
from error_recovery_with_performance_tracking import PerformanceBasedErrorRecovery


class EnhancedErrorHandlingIntegration:
    """
    Enhanced error handling integration for the Distributed Testing Framework.
    
    This class provides a cohesive interface for:
    - Distributed error handling
    - Recovery strategy management
    - Performance-based error recovery
    - Error metrics and analytics
    """
    
    def __init__(self, coordinator):
        """
        Initialize the enhanced error handling integration.
        
        Args:
            coordinator: The coordinator instance
        """
        self.coordinator = coordinator
        
        # Initialize error handler
        self.error_handler = DistributedErrorHandler()
        
        # Initialize recovery manager
        self.recovery_manager = EnhancedErrorRecoveryManager(coordinator)
        
        # Initialize performance-based recovery
        self.performance_recovery = PerformanceBasedErrorRecovery(
            error_handler=self.error_handler,
            recovery_manager=self.recovery_manager,
            coordinator=coordinator,
            db_connection=getattr(coordinator, 'db', None)
        )
        
        # Set up error reporting hooks
        self._setup_error_hooks()
        
        logger.info("Enhanced error handling integration initialized")
    
    def _setup_error_hooks(self):
        """Set up error reporting hooks."""
        # Connect error handler with recovery manager
        self.error_handler.register_error_hook("*", self._error_notification_hook)
        
        # Add specialized hooks for critical systems
        self.error_handler.register_error_hook("network", self._network_error_hook)
        self.error_handler.register_error_hook("db_connection", self._database_error_hook)
        self.error_handler.register_error_hook("coordinator", self._coordinator_error_hook)
    
    async def _error_notification_hook(self, error_report: ErrorReport):
        """
        General error notification hook.
        
        Args:
            error_report: The error report
        """
        # Log the error
        logger.debug(f"Error notification: {error_report.error_id} ({error_report.error_type})")
    
    async def _network_error_hook(self, error_report: ErrorReport):
        """
        Network error hook.
        
        Args:
            error_report: The error report
        """
        # Handle network errors specially
        logger.info(f"Network error detected: {error_report.error_id}")
    
    async def _database_error_hook(self, error_report: ErrorReport):
        """
        Database error hook.
        
        Args:
            error_report: The error report
        """
        # Handle database errors specially
        logger.info(f"Database error detected: {error_report.error_id}")
    
    async def _coordinator_error_hook(self, error_report: ErrorReport):
        """
        Coordinator error hook.
        
        Args:
            error_report: The error report
        """
        # Handle coordinator errors specially
        logger.info(f"Coordinator error detected: {error_report.error_id}")
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Handle an error with automatic recovery.
        
        Args:
            error: The exception
            context: Error context information
            
        Returns:
            Tuple of (success, recovery_info)
        """
        # Create error report
        error_report = self.error_handler.create_error_report(error, context)
        
        # Attempt recovery
        success, recovery_info = await self.performance_recovery.recover(error_report)
        
        # Log recovery attempt
        if success:
            logger.info(f"Successfully recovered from error {error_report.error_id} using strategy {recovery_info['strategy_name']}")
        else:
            logger.warning(f"Failed to recover from error {error_report.error_id} after using strategy {recovery_info['strategy_name']}")
        
        # Return recovery results
        return success, recovery_info
    
    async def retry_operation(self, operation: Callable, args: Tuple = None, kwargs: Dict[str, Any] = None, 
                         context: Dict[str, Any] = None) -> Tuple[Any, Optional[ErrorReport]]:
        """
        Execute an operation with automatic retry based on retry policy.
        
        Args:
            operation: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            context: Additional context information
            
        Returns:
            Tuple of (result, error_report)
        """
        # Delegate to error handler's retry operation
        return await self.error_handler.retry_operation(operation, args, kwargs, context)
    
    def reset_recovery_level(self, error_id: str) -> bool:
        """
        Reset recovery level for an error.
        
        Args:
            error_id: The error ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.performance_recovery.reset_recovery_level(error_id)
    
    def reset_all_recovery_levels(self) -> bool:
        """
        Reset all recovery levels.
        
        Returns:
            True if successful, False otherwise
        """
        return self.performance_recovery.reset_all_recovery_levels()
    
    def get_performance_metrics(self, error_type: Optional[str] = None, 
                            strategy_id: Optional[str] = None, 
                            days: int = 30) -> Dict[str, Any]:
        """
        Get performance metrics for recovery strategies.
        
        Args:
            error_type: Optional filter by error type
            strategy_id: Optional filter by strategy
            days: Number of days to include in analysis
            
        Returns:
            Dictionary with performance metrics
        """
        return self.performance_recovery.get_performance_metrics(error_type, strategy_id, days)
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """
        Get error metrics from the error handler.
        
        Returns:
            Dictionary with error metrics
        """
        return self.error_handler.get_error_metrics()
    
    def get_strategy_recommendations(self, error_type: str) -> List[Dict[str, Any]]:
        """
        Get strategy recommendations for an error type.
        
        Args:
            error_type: The error type
            
        Returns:
            List of recommended strategies sorted by score
        """
        return self.performance_recovery.get_strategy_recommendations(error_type)
    
    def get_recovery_history(self, error_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get recovery history for all errors or a specific error.
        
        Args:
            error_id: Optional filter by error ID
            
        Returns:
            Dictionary with recovery history
        """
        return self.performance_recovery.get_recovery_history(error_id)
    
    def get_unresolved_errors(self, component: Optional[str] = None, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get unresolved errors, optionally filtered by component or severity.
        
        Args:
            component: Optional component to filter by
            severity: Optional minimum severity to filter by
            
        Returns:
            List of unresolved error reports
        """
        reports = self.error_handler.get_unresolved_errors(component, severity)
        
        # Convert to dictionaries for easier use
        result = []
        for report in reports:
            result.append({
                "error_id": report.error_id,
                "error_type": report.error_type.value,
                "error_severity": report.error_severity.value,
                "message": report.message,
                "component": report.context.component,
                "operation": report.context.operation,
                "timestamp": report.context.timestamp.isoformat(),
                "retry_count": report.retry_count,
                "recovery_level": self.performance_recovery.error_recovery_levels.get(report.error_id, 1)
            })
        
        return result
    
    def resolve_error(self, error_id: str, resolution: str, details: Dict[str, Any] = None) -> bool:
        """
        Mark an error as resolved.
        
        Args:
            error_id: ID of the error to resolve
            resolution: Resolution description
            details: Additional resolution details
            
        Returns:
            True if error was found and resolved, False otherwise
        """
        # Let error handler do the resolution
        resolved = self.error_handler.resolve_error(error_id, resolution, details)
        
        # Additionally reset recovery level
        if resolved:
            self.reset_recovery_level(error_id)
        
        return resolved
    
    async def run_diagnostics(self) -> Dict[str, Any]:
        """
        Run diagnostics on the error handling system.
        
        Returns:
            Dictionary with diagnostic information
        """
        # Get error metrics
        error_metrics = self.error_handler.get_error_metrics() 
        
        # Get recovery metrics
        recovery_metrics = self.performance_recovery.get_performance_metrics()
        
        # Get recovery levels
        recovery_levels = self.performance_recovery.get_recovery_levels()
        
        # Get timeouts
        timeouts = self.performance_recovery.get_timeouts()
        
        # Check for issues
        issues = []
        
        # Check for high unresolved error count
        unresolved_count = error_metrics.get("unresolved_errors", 0)
        if unresolved_count > 10:
            issues.append(f"High number of unresolved errors: {unresolved_count}")
        
        # Check for level 5 escalations
        level5_count = sum(1 for level in recovery_levels.values() if level >= 5)
        if level5_count > 0:
            issues.append(f"Critical recovery level reached for {level5_count} errors")
        
        # Check for low success rates
        for strategy_id, strategy in recovery_metrics.get("strategies", {}).items():
            if strategy.get("total_executions", 0) > 5 and strategy.get("success_rate", 1.0) < 0.5:
                issues.append(f"Low success rate for strategy {strategy.get('strategy_name', strategy_id)}: {strategy.get('success_rate', 0):.2f}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "error_metrics": error_metrics,
            "recovery_metrics": recovery_metrics,
            "active_recovery_levels": len(recovery_levels),
            "issues": issues,
            "status": "critical" if issues else "healthy"
        }


# Function to install the enhanced error handling system into the coordinator
def install_enhanced_error_handling(coordinator):
    """
    Install the enhanced error handling system into the coordinator.
    
    Args:
        coordinator: The coordinator instance
        
    Returns:
        The enhanced error handling integration instance
    """
    # Create enhanced error handling
    enhanced_error_handling = EnhancedErrorHandlingIntegration(coordinator)
    
    # Store in coordinator
    coordinator.enhanced_error_handling = enhanced_error_handling
    
    # Initialize error handling endpoints if needed
    if hasattr(coordinator, 'app') and hasattr(coordinator, 'app.router'):
        _setup_error_handling_endpoints(coordinator)
    
    logger.info("Enhanced error handling system installed in coordinator")
    return enhanced_error_handling


def _setup_error_handling_endpoints(coordinator):
    """Set up error handling API endpoints."""
    from aiohttp import web
    
    # Add API routes
    coordinator.app.router.add_get('/api/errors', handle_list_errors)
    coordinator.app.router.add_get('/api/errors/{error_id}', handle_get_error)
    coordinator.app.router.add_post('/api/errors/{error_id}/resolve', handle_resolve_error)
    coordinator.app.router.add_get('/api/recovery/metrics', handle_recovery_metrics)
    coordinator.app.router.add_get('/api/recovery/history', handle_recovery_history)
    coordinator.app.router.add_post('/api/recovery/reset', handle_reset_recovery)
    coordinator.app.router.add_get('/api/diagnostics', handle_diagnostics)


# API Endpoint Handlers
async def handle_list_errors(request):
    """Handle error listing request."""
    coordinator = request.app['coordinator']
    if not hasattr(coordinator, 'enhanced_error_handling'):
        return web.json_response({"error": "Enhanced error handling not installed"}, status=500)
    
    # Get query parameters
    component = request.query.get('component')
    severity = request.query.get('severity')
    resolved = request.query.get('resolved', 'false').lower() == 'true'
    
    if resolved:
        # Get all errors from error handler
        errors = coordinator.error_handler.errors.values()
        result = []
        for error in errors:
            result.append({
                "error_id": error.error_id,
                "error_type": error.error_type.value,
                "error_severity": error.error_severity.value,
                "message": error.message,
                "component": error.context.component,
                "operation": error.context.operation,
                "timestamp": error.context.timestamp.isoformat(),
                "resolution_status": error.resolution_status
            })
    else:
        # Get unresolved errors
        result = coordinator.enhanced_error_handling.get_unresolved_errors(component, severity)
    
    return web.json_response({"errors": result})


async def handle_get_error(request):
    """Handle get error details request."""
    coordinator = request.app['coordinator']
    if not hasattr(coordinator, 'enhanced_error_handling'):
        return web.json_response({"error": "Enhanced error handling not installed"}, status=500)
    
    # Get error ID
    error_id = request.match_info.get('error_id')
    
    # Get error report
    error_report = coordinator.error_handler.get_error_report(error_id)
    if not error_report:
        return web.json_response({"error": f"Error {error_id} not found"}, status=404)
    
    # Get recovery history
    recovery_history = coordinator.enhanced_error_handling.get_recovery_history(error_id)
    
    # Get summary
    summary = coordinator.error_handler.get_error_report_summary(error_id)
    
    # Add recovery history
    summary['recovery_history'] = recovery_history.get('history', [])
    summary['current_recovery_level'] = recovery_history.get('current_level', 1)
    
    return web.json_response(summary)


async def handle_resolve_error(request):
    """Handle error resolution request."""
    coordinator = request.app['coordinator']
    if not hasattr(coordinator, 'enhanced_error_handling'):
        return web.json_response({"error": "Enhanced error handling not installed"}, status=500)
    
    # Get error ID
    error_id = request.match_info.get('error_id')
    
    # Get request body
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    
    # Get resolution and details
    resolution = data.get('resolution', 'manually resolved')
    details = data.get('details', {})
    
    # Resolve error
    resolved = coordinator.enhanced_error_handling.resolve_error(error_id, resolution, details)
    
    if resolved:
        return web.json_response({"status": "success", "message": f"Error {error_id} resolved"})
    else:
        return web.json_response({"error": f"Error {error_id} not found or already resolved"}, status=404)


async def handle_recovery_metrics(request):
    """Handle recovery metrics request."""
    coordinator = request.app['coordinator']
    if not hasattr(coordinator, 'enhanced_error_handling'):
        return web.json_response({"error": "Enhanced error handling not installed"}, status=500)
    
    # Get query parameters
    error_type = request.query.get('error_type')
    strategy_id = request.query.get('strategy_id')
    days = int(request.query.get('days', 30))
    
    # Get metrics
    metrics = coordinator.enhanced_error_handling.get_performance_metrics(error_type, strategy_id, days)
    
    return web.json_response(metrics)


async def handle_recovery_history(request):
    """Handle recovery history request."""
    coordinator = request.app['coordinator']
    if not hasattr(coordinator, 'enhanced_error_handling'):
        return web.json_response({"error": "Enhanced error handling not installed"}, status=500)
    
    # Get query parameters
    error_id = request.query.get('error_id')
    
    # Get history
    history = coordinator.enhanced_error_handling.get_recovery_history(error_id)
    
    return web.json_response(history)


async def handle_reset_recovery(request):
    """Handle reset recovery levels request."""
    coordinator = request.app['coordinator']
    if not hasattr(coordinator, 'enhanced_error_handling'):
        return web.json_response({"error": "Enhanced error handling not installed"}, status=500)
    
    # Get request body
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    
    # Get error ID
    error_id = data.get('error_id')
    
    if error_id:
        # Reset specific error
        result = coordinator.enhanced_error_handling.reset_recovery_level(error_id)
        message = f"Recovery level for error {error_id} reset" if result else f"Error {error_id} not found"
    else:
        # Reset all errors
        result = coordinator.enhanced_error_handling.reset_all_recovery_levels()
        message = "All recovery levels reset"
    
    return web.json_response({"status": "success", "message": message})


async def handle_diagnostics(request):
    """Handle diagnostics request."""
    coordinator = request.app['coordinator']
    if not hasattr(coordinator, 'enhanced_error_handling'):
        return web.json_response({"error": "Enhanced error handling not installed"}, status=500)
    
    # Run diagnostics
    diagnostics = await coordinator.enhanced_error_handling.run_diagnostics()
    
    return web.json_response(diagnostics)


if __name__ == "__main__":
    # Example usage
    class MockCoordinator:
        def __init__(self):
            import duckdb
            self.db = duckdb.connect(":memory:")
            self.tasks = {}
            self.running_tasks = {}
            self.pending_tasks = set()
            
    async def run_example():
        # Create mock coordinator
        coordinator = MockCoordinator()
        
        # Install enhanced error handling
        enhanced_error_handling = install_enhanced_error_handling(coordinator)
        
        # Example error handling
        try:
            # Simulate an error
            raise ConnectionError("Database connection failed")
        except Exception as e:
            # Handle the error
            success, info = await enhanced_error_handling.handle_error(e, {
                "component": "database",
                "operation": "connect"
            })
            
            print(f"Recovery {'succeeded' if success else 'failed'}")
            print(f"Strategy: {info['strategy_name']}")
            print(f"Execution time: {info['execution_time']:.2f}s")
        
        # Get metrics
        metrics = enhanced_error_handling.get_performance_metrics()
        print("\nPerformance Metrics:")
        print(f"Total executions: {metrics['overall']['total_executions']}")
        
        # Run diagnostics
        diagnostics = await enhanced_error_handling.run_diagnostics()
        print("\nDiagnostics:")
        print(f"Status: {diagnostics['status']}")
        if diagnostics['issues']:
            print("Issues:")
            for issue in diagnostics['issues']:
                print(f"- {issue}")
    
    # Run the example
    asyncio.run(run_example())