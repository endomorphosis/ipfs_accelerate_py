#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error Reporting Demo

This example demonstrates how to use the automated error reporting system
in various scenarios.

Author: IPFS Accelerate Python Framework Team
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.error_reporter import ErrorReporter, get_error_reporter, report_error, install_global_exception_handler


def demo_basic_error_reporting():
    """Demonstrate basic error reporting"""
    print("=" * 60)
    print("Demo 1: Basic Error Reporting")
    print("=" * 60)
    
    # Create error reporter (will use env vars if available)
    reporter = ErrorReporter(
        github_token=os.environ.get('GITHUB_TOKEN', 'test_token'),
        github_repo=os.environ.get('GITHUB_REPO', 'test_owner/test_repo'),
        enabled=bool(os.environ.get('GITHUB_TOKEN'))  # Only enable if token is set
    )
    
    print(f"Reporter enabled: {reporter.enabled}")
    print(f"GitHub repo: {reporter.github_repo}")
    print()
    
    # Report a simple error
    try:
        # Simulate an error
        raise ValueError("This is a demonstration error")
    except Exception as e:
        print(f"Caught exception: {e}")
        issue_url = reporter.report_error(
            exception=e,
            source_component='demo-script',
            context={
                'demo': 'basic_error_reporting',
                'user': os.environ.get('USER', 'unknown')
            }
        )
        
        if issue_url:
            print(f"✓ Error reported to GitHub: {issue_url}")
        else:
            print("✗ Error not reported (may be duplicate or disabled)")
    
    print()


def demo_manual_error_reporting():
    """Demonstrate manual error reporting without exception object"""
    print("=" * 60)
    print("Demo 2: Manual Error Reporting")
    print("=" * 60)
    
    # Use convenience function
    issue_url = report_error(
        error_type='CustomError',
        error_message='This is a manually created error report',
        traceback_str='Stack trace line 1\nStack trace line 2',
        source_component='demo-manual',
        context={
            'operation': 'manual_reporting',
            'severity': 'low'
        }
    )
    
    if issue_url:
        print(f"✓ Error reported to GitHub: {issue_url}")
    else:
        print("✗ Error not reported (may be duplicate or disabled)")
    
    print()


def demo_global_exception_handler():
    """Demonstrate global exception handler"""
    print("=" * 60)
    print("Demo 3: Global Exception Handler")
    print("=" * 60)
    
    # Install global exception handler
    install_global_exception_handler('demo-global-handler')
    print("✓ Global exception handler installed")
    print("  Any uncaught exceptions will now be reported automatically")
    print()
    
    # Note: We can't actually trigger an uncaught exception in this demo
    # without terminating the script, so we'll just show it's installed
    print("  To test: run a script that raises an uncaught exception")
    print()


def demo_context_information():
    """Demonstrate adding context to error reports"""
    print("=" * 60)
    print("Demo 4: Adding Context Information")
    print("=" * 60)
    
    reporter = get_error_reporter()
    
    try:
        # Simulate an error with lots of context
        user_input = {'field1': 'value1', 'field2': 'value2'}
        raise TypeError(f"Invalid input: expected string, got {type(user_input)}")
    except Exception as e:
        issue_url = reporter.report_error(
            exception=e,
            source_component='demo-context',
            context={
                'user_input': str(user_input),
                'operation': 'process_user_data',
                'step': 'validation',
                'environment': {
                    'python_version': sys.version,
                    'os': os.name,
                    'cwd': os.getcwd()
                }
            }
        )
        
        if issue_url:
            print(f"✓ Error with context reported: {issue_url}")
        else:
            print("✗ Error not reported")
    
    print()


def demo_duplicate_prevention():
    """Demonstrate duplicate error prevention"""
    print("=" * 60)
    print("Demo 5: Duplicate Error Prevention")
    print("=" * 60)
    
    reporter = get_error_reporter()
    
    # Report the same error multiple times
    for i in range(3):
        try:
            raise RuntimeError("This is a duplicate error for testing")
        except Exception as e:
            issue_url = reporter.report_error(
                exception=e,
                source_component='demo-duplicate',
                context={'attempt': i + 1}
            )
            
            if issue_url:
                print(f"Attempt {i+1}: ✓ Error reported: {issue_url}")
            else:
                print(f"Attempt {i+1}: ✗ Duplicate detected, not reported")
    
    print()


def demo_different_components():
    """Demonstrate reporting errors from different components"""
    print("=" * 60)
    print("Demo 6: Errors from Different Components")
    print("=" * 60)
    
    reporter = get_error_reporter()
    
    components = ['mcp-server', 'dashboard', 'docker-container', 'cli']
    
    for component in components:
        try:
            raise Exception(f"Error in {component}")
        except Exception as e:
            issue_url = reporter.report_error(
                exception=e,
                source_component=component,
                context={'component': component}
            )
            
            if issue_url:
                print(f"{component}: ✓ Reported")
            else:
                print(f"{component}: ✗ Not reported")
    
    print()


def demo_check_status():
    """Demonstrate checking error reporter status"""
    print("=" * 60)
    print("Demo 7: Check Error Reporter Status")
    print("=" * 60)
    
    reporter = get_error_reporter()
    
    print(f"Enabled: {reporter.enabled}")
    print(f"GitHub Token: {'Set' if reporter.github_token else 'Not set'}")
    print(f"GitHub Repo: {reporter.github_repo or 'Not set'}")
    print(f"Include System Info: {reporter.include_system_info}")
    print(f"Auto Label: {reporter.auto_label}")
    print(f"Reported Errors Count: {len(reporter.reported_errors)}")
    print(f"Cache File: {reporter.error_cache_file}")
    
    if reporter.error_cache_file.exists():
        print(f"Cache File Size: {reporter.error_cache_file.stat().st_size} bytes")
    else:
        print("Cache File: Does not exist yet")
    
    print()


def main():
    """Run all demos"""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  IPFS Accelerate Error Reporting System - Demo".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Check if GitHub credentials are set
    if not os.environ.get('GITHUB_TOKEN'):
        print("⚠️  WARNING: GITHUB_TOKEN environment variable not set")
        print("   Error reporting will be in demo mode only")
        print("   To enable actual reporting, set GITHUB_TOKEN and GITHUB_REPO")
        print()
    else:
        print("✓ GitHub credentials detected")
        print(f"  Repository: {os.environ.get('GITHUB_REPO', 'not set')}")
        print()
    
    input("Press Enter to continue...")
    print()
    
    # Run demos
    try:
        demo_basic_error_reporting()
        input("Press Enter to continue...")
        
        demo_manual_error_reporting()
        input("Press Enter to continue...")
        
        demo_global_exception_handler()
        input("Press Enter to continue...")
        
        demo_context_information()
        input("Press Enter to continue...")
        
        demo_duplicate_prevention()
        input("Press Enter to continue...")
        
        demo_different_components()
        input("Press Enter to continue...")
        
        demo_check_status()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        return
    
    print()
    print("=" * 60)
    print("All demos completed!")
    print("=" * 60)
    print()
    print("To enable actual error reporting:")
    print("1. Set GITHUB_TOKEN environment variable")
    print("2. Set GITHUB_REPO environment variable (e.g., 'owner/repo')")
    print("3. Run this demo again")
    print()


if __name__ == '__main__':
    main()
