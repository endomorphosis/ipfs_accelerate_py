#!/usr/bin/env python3
"""
Unified Database API Example

This example demonstrates how to use the Unified API Server to access database operations
across multiple components, showing cross-component aggregation and unified data views.

Usage:
    python unified_db_example.py --model bert-base-uncased --api-key your-api-key
"""

import os
import sys
import json
import argparse
import requests
from typing import Dict, Any, List, Optional

def get_db_overview(base_url: str, api_key: str) -> Dict[str, Any]:
    """
    Get overview of database statistics across all components.
    
    Args:
        base_url: Base URL of the Unified API Server
        api_key: API key for authentication
        
    Returns:
        Dictionary containing database overview
    """
    headers = {"X-API-Key": api_key}
    response = requests.get(f"{base_url}/api/db/overview", headers=headers)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return {}
    
    return response.json()

def get_model_unified_data(base_url: str, api_key: str, model_name: str) -> Dict[str, Any]:
    """
    Get unified data for a specific model across all components.
    
    Args:
        base_url: Base URL of the Unified API Server
        api_key: API key for authentication
        model_name: Name of the model to get data for
        
    Returns:
        Dictionary containing unified model data
    """
    headers = {"X-API-Key": api_key}
    response = requests.get(f"{base_url}/api/db/model/{model_name}", headers=headers)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return {}
    
    return response.json()

def search_across_components(base_url: str, api_key: str, query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search across all database components.
    
    Args:
        base_url: Base URL of the Unified API Server
        api_key: API key for authentication
        query: Search query
        limit: Maximum number of results per component
        
    Returns:
        Dictionary containing search results
    """
    headers = {"X-API-Key": api_key}
    response = requests.post(
        f"{base_url}/api/db/search",
        params={"query": query, "limit": limit},
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return {}
    
    return response.json()

def get_component_specific_data(base_url: str, api_key: str, component: str, endpoint: str) -> Dict[str, Any]:
    """
    Get data from a specific component's database API.
    
    Args:
        base_url: Base URL of the Unified API Server
        api_key: API key for authentication
        component: Component name (test, generator, benchmark)
        endpoint: Endpoint path
        
    Returns:
        Dictionary containing component-specific data
    """
    headers = {"X-API-Key": api_key}
    response = requests.get(f"{base_url}/api/db/{component}/{endpoint}", headers=headers)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return {}
    
    return response.json()

def print_section(title: str):
    """Print a section title."""
    print(f"\n{'-' * 80}")
    print(f"{title}")
    print(f"{'-' * 80}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified Database API Example")
    parser.add_argument("--base-url", default="http://localhost:8080", help="Base URL of the Unified API Server")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--model", default="bert-base-uncased", help="Model name for unified queries")
    parser.add_argument("--query", default="bert", help="Search query for unified search")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()
    
    # Print example header
    print("\nUnified Database API Example")
    print("===========================\n")
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Search Query: {args.query}")
    
    # Get database overview
    print_section("Database Overview")
    overview = get_db_overview(args.base_url, args.api_key)
    
    if overview:
        # Print test stats
        if "test_stats" in overview and overview["test_stats"] and not isinstance(overview["test_stats"], dict):
            test_stats = overview["test_stats"]
            print(f"Test Stats: {len(test_stats)} models")
            for i, stat in enumerate(test_stats[:3]):  # Show top 3
                print(f"  - {stat.get('model_name', 'Unknown')}: {stat.get('total_tasks', 0)} tasks, " 
                      f"{stat.get('completed_tasks', 0)} completed")
            if len(test_stats) > 3:
                print(f"  - ... and {len(test_stats) - 3} more models")
        else:
            print("Test Stats: Not available or empty")
            
        # Print generator stats
        if "generator_stats" in overview and overview["generator_stats"] and not isinstance(overview["generator_stats"], dict):
            gen_stats = overview["generator_stats"]
            print(f"Generator Stats: {len(gen_stats)} models")
            for i, stat in enumerate(gen_stats[:3]):  # Show top 3
                print(f"  - {stat.get('model_name', 'Unknown')}: {stat.get('total_tasks', 0)} tasks, "
                      f"{stat.get('completed_tasks', 0)} completed")
            if len(gen_stats) > 3:
                print(f"  - ... and {len(gen_stats) - 3} more models")
        else:
            print("Generator Stats: Not available or empty")
            
        # Print benchmark stats
        if "benchmark_stats" in overview and overview["benchmark_stats"] and not isinstance(overview["benchmark_stats"], dict):
            bench_stats = overview["benchmark_stats"]
            print(f"Benchmark Stats: {len(bench_stats)} models")
            for i, stat in enumerate(bench_stats[:3]):  # Show top 3
                print(f"  - {stat.get('model_name', 'Unknown')}: {stat.get('total_runs', 0)} runs, "
                      f"{stat.get('completed_runs', 0)} completed")
            if len(bench_stats) > 3:
                print(f"  - ... and {len(bench_stats) - 3} more models")
        else:
            print("Benchmark Stats: Not available or empty")
    else:
        print("Failed to retrieve database overview")
    
    # Get unified model data
    print_section(f"Unified Model Data: {args.model}")
    model_data = get_model_unified_data(args.base_url, args.api_key, args.model)
    
    if model_data and "overview" in model_data:
        overview = model_data["overview"]
        print(f"Total Test Runs: {overview.get('total_test_runs', 0)}")
        print(f"Total Generator Tasks: {overview.get('total_generator_tasks', 0)}")
        print(f"Total Benchmark Runs: {overview.get('total_benchmark_runs', 0)}")
        print(f"Test Success Rate: {overview.get('test_success_rate', 0) * 100:.1f}%")
        print(f"Generator Success Rate: {overview.get('generator_success_rate', 0) * 100:.1f}%")
        print(f"Benchmark Success Rate: {overview.get('benchmark_success_rate', 0) * 100:.1f}%")
        
        # Print recent test runs
        if "recent_test_runs" in model_data and model_data["recent_test_runs"]:
            recent_tests = model_data["recent_test_runs"]
            print(f"\nRecent Test Runs ({len(recent_tests)}):")
            for i, run in enumerate(recent_tests[:2]):  # Show top 2
                print(f"  - {run.get('run_id', 'Unknown')}: Status: {run.get('status', 'Unknown')}, "
                      f"Duration: {run.get('duration', 0):.2f}s")
            if len(recent_tests) > 2:
                print(f"  - ... and {len(recent_tests) - 2} more runs")
                
        # Print recent generator tasks
        if "recent_generator_tasks" in model_data and model_data["recent_generator_tasks"]:
            recent_gens = model_data["recent_generator_tasks"]
            print(f"\nRecent Generator Tasks ({len(recent_gens)}):")
            for i, task in enumerate(recent_gens[:2]):  # Show top 2
                print(f"  - {task.get('task_id', 'Unknown')}: Status: {task.get('status', 'Unknown')}, "
                      f"Duration: {task.get('duration', 0):.2f}s")
            if len(recent_gens) > 2:
                print(f"  - ... and {len(recent_gens) - 2} more tasks")
                
        # Print recent benchmark runs
        if "recent_benchmark_runs" in model_data and model_data["recent_benchmark_runs"]:
            recent_benchs = model_data["recent_benchmark_runs"]
            print(f"\nRecent Benchmark Runs ({len(recent_benchs)}):")
            for i, run in enumerate(recent_benchs[:2]):  # Show top 2
                print(f"  - {run.get('run_id', 'Unknown')}: Status: {run.get('status', 'Unknown')}, "
                      f"Duration: {run.get('duration', 0):.2f}s")
            if len(recent_benchs) > 2:
                print(f"  - ... and {len(recent_benchs) - 2} more runs")
    else:
        print(f"Failed to retrieve unified data for model {args.model}")
    
    # Search across components
    print_section(f"Unified Search: '{args.query}'")
    search_results = search_across_components(args.base_url, args.api_key, args.query)
    
    if search_results and "result_counts" in search_results:
        counts = search_results["result_counts"]
        print(f"Total Results: {counts.get('total', 0)}")
        print(f"Test Results: {counts.get('test', 0)}")
        print(f"Generator Results: {counts.get('generator', 0)}")
        print(f"Benchmark Results: {counts.get('benchmark', 0)}")
        
        # Print some test results
        if "test_results" in search_results and search_results["test_results"]:
            test_results = search_results["test_results"]
            print(f"\nTest Results ({len(test_results)}):")
            for i, result in enumerate(test_results[:2]):  # Show top 2
                print(f"  - {result.get('run_id', 'Unknown')}: Model: {result.get('model_name', 'Unknown')}, "
                      f"Status: {result.get('status', 'Unknown')}")
            if len(test_results) > 2:
                print(f"  - ... and {len(test_results) - 2} more results")
    else:
        print(f"No search results found for query '{args.query}'")
    
    # Get component-specific data
    if args.verbose:
        print_section("Component-Specific Data: Test Suite")
        test_runs = get_component_specific_data(args.base_url, args.api_key, "test", "runs?limit=5")
        
        if test_runs:
            print(f"Test Runs ({len(test_runs)}):")
            for i, run in enumerate(test_runs[:3]):  # Show top 3
                print(f"  - {run.get('run_id', 'Unknown')}: Model: {run.get('model_name', 'Unknown')}, "
                      f"Status: {run.get('status', 'Unknown')}")
            if len(test_runs) > 3:
                print(f"  - ... and {len(test_runs) - 3} more runs")
        else:
            print("Failed to retrieve test runs")
            
        print_section("Component-Specific Data: Generator")
        gen_tasks = get_component_specific_data(args.base_url, args.api_key, "generator", "tasks?limit=5")
        
        if gen_tasks:
            print(f"Generator Tasks ({len(gen_tasks)}):")
            for i, task in enumerate(gen_tasks[:3]):  # Show top 3
                print(f"  - {task.get('task_id', 'Unknown')}: Model: {task.get('model_name', 'Unknown')}, "
                      f"Status: {task.get('status', 'Unknown')}")
            if len(gen_tasks) > 3:
                print(f"  - ... and {len(gen_tasks) - 3} more tasks")
        else:
            print("Failed to retrieve generator tasks")
    
    print("\nExample completed!\n")

if __name__ == "__main__":
    main()