#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Docker Container Error Reporter

This script wraps Python execution in Docker containers to automatically
report errors to GitHub.

Author: IPFS Accelerate Python Framework Team
"""

import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.error_reporter import get_error_reporter, install_global_exception_handler
    error_reporting_available = True
except ImportError:
    error_reporting_available = False
    print("Warning: Error reporting not available in Docker container", file=sys.stderr)


def setup_docker_error_reporting():
    """
    Set up error reporting for Docker container.
    """
    if not error_reporting_available:
        return False
    
    try:
        # Install global exception handler
        install_global_exception_handler(source_component='docker-container')
        
        # Check configuration
        reporter = get_error_reporter()
        if reporter.enabled:
            print(f"✓ Error reporting enabled for Docker container -> {reporter.github_repo}")
            return True
        else:
            print("! Error reporting disabled (missing GITHUB_TOKEN or GITHUB_REPO environment variables)")
            return False
    except Exception as e:
        print(f"Warning: Failed to set up error reporting: {e}", file=sys.stderr)
        return False


def main():
    """
    Main entry point for Docker error reporting wrapper.
    """
    # Set up error reporting
    setup_docker_error_reporting()
    
    # If no arguments provided, show usage
    if len(sys.argv) < 2:
        print("Usage: docker_error_wrapper.py <python_script> [args...]")
        print("       docker_error_wrapper.py -m <module> [args...]")
        sys.exit(1)
    
    # Extract script/module to run
    if sys.argv[1] == '-m':
        # Module mode
        if len(sys.argv) < 3:
            print("Error: -m flag requires a module name")
            sys.exit(1)
        
        module_name = sys.argv[2]
        sys.argv = [sys.argv[0]] + sys.argv[3:]
        
        # Run the module
        import runpy
        try:
            runpy.run_module(module_name, run_name='__main__')
        except Exception as e:
            if error_reporting_available:
                reporter = get_error_reporter()
                reporter.report_error(
                    exception=e,
                    source_component='docker-container',
                    context={
                        'module': module_name,
                        'arguments': sys.argv
                    }
                )
            raise
    else:
        # Script mode
        script_path = sys.argv[1]
        sys.argv = sys.argv[1:]
        
        # Run the script
        try:
            with open(script_path) as f:
                code = compile(f.read(), script_path, 'exec')
                exec(code, {'__name__': '__main__', '__file__': script_path})
        except Exception as e:
            if error_reporting_available:
                reporter = get_error_reporter()
                reporter.report_error(
                    exception=e,
                    source_component='docker-container',
                    context={
                        'script': script_path,
                        'arguments': sys.argv
                    }
                )
            raise


if __name__ == '__main__':
    main()
