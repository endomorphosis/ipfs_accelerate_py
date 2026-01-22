#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Reporting API Endpoint

This module provides a REST API endpoint for receiving error reports
from JavaScript clients and creating GitHub issues.

Author: IPFS Accelerate Python Framework Team
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    flask_available = True
except ImportError:
    flask_available = False

try:
    from utils.error_reporter import get_error_reporter
    error_reporter_available = True
except ImportError:
    error_reporter_available = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_error_reporting_api(app: Optional[Any] = None) -> Any:
    """
    Create and configure the error reporting API.
    
    Args:
        app: Existing Flask app instance, or None to create a new one
        
    Returns:
        Flask app instance with error reporting endpoints
    """
    if not flask_available:
        logger.error("Flask not available, cannot create error reporting API")
        return None
    
    if app is None:
        app = Flask(__name__)
        CORS(app)  # Enable CORS for all routes
    
    @app.route('/api/report-error', methods=['POST'])
    def report_error():
        """
        API endpoint to receive error reports from clients and create GitHub issues.
        
        Expected JSON payload:
        {
            "title": "Error title",
            "body": "Error description",
            "labels": ["bug", "automated-report"],
            "error_info": {
                "error_type": "TypeError",
                "error_message": "Cannot read property 'x' of undefined",
                "source_component": "dashboard",
                "timestamp": "2025-11-06T08:00:00Z",
                ...
            }
        }
        
        Returns:
            JSON response with issue URL or error message
        """
        try:
            # Get request data
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Validate required fields
            error_info = data.get('error_info', {})
            if not error_info:
                return jsonify({'error': 'Missing error_info'}), 400
            
            # Check if error reporter is available
            if not error_reporter_available:
                logger.warning("Error reporter not available")
                return jsonify({
                    'error': 'Error reporting not configured',
                    'issue_url': None
                }), 503
            
            # Get error reporter
            reporter = get_error_reporter()
            
            if not reporter.enabled:
                logger.warning("Error reporter is disabled")
                return jsonify({
                    'error': 'Error reporting is disabled',
                    'issue_url': None
                }), 503
            
            # Report the error
            issue_url = reporter.report_error(
                error_type=error_info.get('error_type'),
                error_message=error_info.get('error_message'),
                traceback_str=error_info.get('stack', ''),
                source_component=error_info.get('source_component', 'unknown'),
                context=error_info.get('context', {})
            )
            
            if issue_url:
                logger.info(f"Created GitHub issue: {issue_url}")
                return jsonify({
                    'success': True,
                    'issue_url': issue_url,
                    'message': 'Error reported successfully'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'issue_url': None,
                    'message': 'Error already reported or failed to create issue'
                }), 200
                
        except Exception as e:
            logger.error(f"Error in report_error endpoint: {e}")
            return jsonify({
                'error': str(e),
                'issue_url': None
            }), 500
    
    @app.route('/api/error-reporter/status', methods=['GET'])
    def error_reporter_status():
        """
        Get the status of the error reporting system.
        
        Returns:
            JSON response with error reporter status
        """
        try:
            if not error_reporter_available:
                return jsonify({
                    'enabled': False,
                    'reason': 'Error reporter module not available'
                }), 200
            
            reporter = get_error_reporter()
            
            return jsonify({
                'enabled': reporter.enabled,
                'github_repo': reporter.github_repo if reporter.enabled else None,
                'reported_errors_count': len(reporter.reported_errors)
            }), 200
            
        except Exception as e:
            logger.error(f"Error in error_reporter_status endpoint: {e}")
            return jsonify({
                'error': str(e)
            }), 500
    
    @app.route('/api/error-reporter/test', methods=['POST'])
    def test_error_reporter():
        """
        Test endpoint to verify error reporting is working.
        
        Returns:
            JSON response with test result
        """
        try:
            if not error_reporter_available:
                return jsonify({
                    'success': False,
                    'reason': 'Error reporter module not available'
                }), 200
            
            reporter = get_error_reporter()
            
            if not reporter.enabled:
                return jsonify({
                    'success': False,
                    'reason': 'Error reporter is disabled'
                }), 200
            
            # Create a test error report
            issue_url = reporter.report_error(
                error_type='TestError',
                error_message='This is a test error report',
                source_component='test',
                context={
                    'test': True,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            return jsonify({
                'success': bool(issue_url),
                'issue_url': issue_url,
                'message': 'Test error reported successfully' if issue_url else 'Test error not reported (may be duplicate)'
            }), 200
            
        except Exception as e:
            logger.error(f"Error in test_error_reporter endpoint: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    logger.info("Error reporting API endpoints registered")
    return app


if __name__ == '__main__':
    # Create standalone app for testing
    app = create_error_reporting_api()
    if app:
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Failed to create error reporting API")
