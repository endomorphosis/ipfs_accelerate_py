#!/usr/bin/env python3
"""
Update Unified API Server

This script updates the Unified API Server to integrate the Predictive Performance API.
It modifies the necessary configuration and gateway code to include the new component.
"""

import os
import sys
import re
import shutil
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("update_unified_api")

def find_unified_api_server():
    """Find the path to the unified_api_server.py file.
    
    Returns:
        Path to the unified_api_server.py file, or None if not found
    """
    # Start from the current directory
    current_dir = Path(__file__).resolve().parent
    
    # Check in parent directory
    parent_dir = current_dir.parent
    unified_api_path = parent_dir / "unified_api_server.py"
    
    if unified_api_path.exists():
        return unified_api_path
    
    # Check in repository root
    repo_root = parent_dir.parent
    unified_api_path = repo_root / "test" / "unified_api_server.py"
    
    if unified_api_path.exists():
        return unified_api_path
    
    return None

def backup_file(file_path):
    """Create a backup of a file.
    
    Args:
        file_path: Path to the file to backup
        
    Returns:
        Path to the backup file
    """
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    return backup_path

def update_default_config(content):
    """Update the DEFAULT_CONFIG in the unified_api_server.py file.
    
    Args:
        content: Content of the unified_api_server.py file
        
    Returns:
        Updated content
    """
    pattern = r'DEFAULT_CONFIG = \{(.*?)\}'
    config_match = re.search(pattern, content, re.DOTALL)
    
    if not config_match:
        logger.error("Could not find DEFAULT_CONFIG in unified_api_server.py")
        return content
    
    config_str = config_match.group(1)
    # Find the last key in the config
    last_key = re.findall(r'"(\w+)": \{.*?\},?', config_str, re.DOTALL)[-1]
    last_key_pattern = f'"{last_key}": {{.*?}}'
    last_key_match = re.search(last_key_pattern, config_str, re.DOTALL)
    
    if not last_key_match:
        logger.error(f"Could not find last key '{last_key}' in DEFAULT_CONFIG")
        return content
    
    # Insert our config after the last key
    predictive_config = """
    "predictive_performance_api": {
        "enabled": True,
        "port": 8500,
        "host": "0.0.0.0",
        "module": "test.api_server.predictive_performance_api_server",
        "args": []
    },"""
    
    updated_config = config_str.replace(
        last_key_match.group(0), 
        last_key_match.group(0) + "," + predictive_config
    )
    
    updated_content = content.replace(config_match.group(1), updated_config)
    return updated_content

def update_gateway_file_generation(content):
    """Update the _generate_gateway_file method to include Predictive Performance API.
    
    Args:
        content: Content of the unified_api_server.py file
        
    Returns:
        Updated content
    """
    # Find the _generate_gateway_file method
    method_pattern = r'def _generate_gateway_file\(self, file_path, service_config\):(.*?)\n    def'
    method_match = re.search(method_pattern, content, re.DOTALL)
    
    if not method_match:
        logger.error("Could not find _generate_gateway_file method in unified_api_server.py")
        return content
    
    method_body = method_match.group(1)
    
    # Find the service configuration section
    config_pattern = r'# Get service configurations(.*?)# Create gateway file content'
    config_match = re.search(config_pattern, method_body, re.DOTALL)
    
    if not config_match:
        logger.error("Could not find service configuration section in _generate_gateway_file method")
        return content
    
    config_section = config_match.group(1)
    
    # Add our configuration
    predictive_config = """
        predictive_performance_api_config = self.config.get("predictive_performance_api", {})
        
        predictive_performance_api_host = predictive_performance_api_config.get("host", "0.0.0.0")
        predictive_performance_api_port = predictive_performance_api_config.get("port", 8500)
    """
    
    updated_config = config_section + predictive_config
    updated_method_body = method_body.replace(config_match.group(1), updated_config)
    
    # Find the gateway URL configuration section
    url_pattern = r'# Configure service endpoints(.*?)# Define API key security'
    url_match = re.search(url_pattern, method_body, re.DOTALL)
    
    if not url_match:
        logger.error("Could not find URL configuration section in _generate_gateway_file method")
        return content
    
    url_section = url_match.group(1)
    
    # Add our URL configuration
    predictive_url = """
# Configure Predictive Performance API endpoint
PREDICTIVE_PERFORMANCE_API_URL = "http://{predictive_performance_api_host}:{predictive_performance_api_port}"
"""
    
    updated_url = url_section + predictive_url
    updated_method_body = updated_method_body.replace(url_match.group(1), updated_url)
    
    # Find the root endpoint services section
    services_pattern = r'"services": \[(.*?)\],'
    services_match = re.search(services_pattern, method_body, re.DOTALL)
    
    if not services_match:
        logger.error("Could not find services section in _generate_gateway_file method")
        return content
    
    services_section = services_match.group(1)
    
    # Add our service
    predictive_service = """
            {"name": "Predictive Performance API", "url": f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance"},"""
    
    updated_services = services_section + predictive_service
    updated_method_body = updated_method_body.replace(services_match.group(1), updated_services)
    
    # Find the database endpoints section
    db_pattern = r'"database_endpoints": \[(.*?)\],'
    db_match = re.search(db_pattern, method_body, re.DOTALL)
    
    if not db_match:
        logger.error("Could not find database_endpoints section in _generate_gateway_file method")
        return content
    
    db_section = db_match.group(1)
    
    # Add our database endpoint
    predictive_db = """
            {"name": "Predictive Performance Database API", "url": "/api/db/predictive-performance"},"""
    
    updated_db = db_section + predictive_db
    updated_method_body = updated_method_body.replace(db_match.group(1), updated_db)
    
    # Find the component API routes section
    routes_pattern = r'# Component API routes(.*?)# WebSocket routes'
    routes_match = re.search(routes_pattern, method_body, re.DOTALL)
    
    if not routes_match:
        logger.error("Could not find component API routes section in _generate_gateway_file method")
        return content
    
    routes_section = routes_match.group(1)
    
    # Add our route
    predictive_route = """
@app.api_route("/api/predictive-performance{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"])
async def predictive_performance_api_route(request: Request, path: str):
    """Route requests to the Predictive Performance API."""
    url = f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance{path}"
    return await forward_request(url, request)
"""
    
    updated_routes = routes_section + predictive_route
    updated_method_body = updated_method_body.replace(routes_match.group(1), updated_routes)
    
    # Find the WebSocket routes section
    ws_pattern = r'# WebSocket routes(.*?)# Database API routes'
    ws_match = re.search(ws_pattern, method_body, re.DOTALL)
    
    if not ws_match:
        logger.error("Could not find WebSocket routes section in _generate_gateway_file method")
        return content
    
    ws_section = ws_match.group(1)
    
    # Add our WebSocket route
    predictive_ws = """
@app.websocket("/api/predictive-performance/ws/{task_id}")
async def predictive_performance_api_websocket(websocket: WebSocket, task_id: str):
    """WebSocket connection for the Predictive Performance API."""
    await websocket_forward(websocket, f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance/ws/{task_id}")
"""
    
    updated_ws = ws_section + predictive_ws
    updated_method_body = updated_method_body.replace(ws_match.group(1), updated_ws)
    
    # Find the Database API routes section
    db_routes_pattern = r'# Database API routes - Benchmark(.*?)# Cross-component database operations'
    db_routes_match = re.search(db_routes_pattern, method_body, re.DOTALL)
    
    if not db_routes_match:
        logger.error("Could not find Database API routes section in _generate_gateway_file method")
        return content
    
    db_routes_section = db_routes_match.group(1)
    
    # Add our database route
    predictive_db_route = """
# Database API routes - Predictive Performance
@app.api_route("/api/db/predictive-performance{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def predictive_performance_db_route(request: Request, path: str, api_key: str = Depends(get_api_key)):
    """
    Route requests to the Predictive Performance Database API.
    
    All database operations require API key authentication.
    """
    url = f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance/db{path}"
    return await forward_request(url, request)
"""
    
    updated_db_routes = db_routes_section + predictive_db_route
    updated_method_body = updated_method_body.replace(db_routes_match.group(1), updated_db_routes)
    
    # Find the get_db_overview method
    db_overview_pattern = r'async def get_db_overview\(\):(.*?)# Process responses'
    db_overview_match = re.search(db_overview_pattern, method_body, re.DOTALL)
    
    if db_overview_match:
        db_overview_section = db_overview_match.group(1)
        
        # Add our stats
        predictive_stats = """
        # Gather data from all component databases
        test_stats = await client.get(f"{TEST_API_URL}/api/test/db/models/stats")
        generator_stats = await client.get(f"{GENERATOR_API_URL}/api/generator/db/models/stats")
        benchmark_stats = await client.get(f"{BENCHMARK_API_URL}/api/benchmark/db/models/stats")
        predictive_stats = await client.get(f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance/stats")
        """
        
        updated_db_overview = db_overview_section.replace(
            "        # Gather data from all component databases\n"
            "        test_stats = await client.get(f\"{TEST_API_URL}/api/test/db/models/stats\")\n"
            "        generator_stats = await client.get(f\"{GENERATOR_API_URL}/api/generator/db/models/stats\")\n"
            "        benchmark_stats = await client.get(f\"{BENCHMARK_API_URL}/api/benchmark/db/models/stats\")\n",
            predictive_stats
        )
        updated_method_body = updated_method_body.replace(db_overview_match.group(1), updated_db_overview)
        
        # Find the process responses section
        responses_pattern = r'# Process responses(.*?)# Combine data'
        responses_match = re.search(responses_pattern, method_body, re.DOTALL)
        
        if responses_match:
            responses_section = responses_match.group(1)
            
            # Add our response
            predictive_response = """
        # Process responses
        test_data = test_stats.json() if test_stats.status_code == 200 else {"error": "Failed to fetch test stats"}
        generator_data = generator_stats.json() if generator_stats.status_code == 200 else {"error": "Failed to fetch generator stats"}
        benchmark_data = benchmark_stats.json() if benchmark_stats.status_code == 200 else {"error": "Failed to fetch benchmark stats"}
        predictive_data = predictive_stats.json() if predictive_stats.status_code == 200 else {"error": "Failed to fetch predictive performance stats"}
        """
            
            updated_responses = responses_section.replace(
                "        # Process responses\n"
                "        test_data = test_stats.json() if test_stats.status_code == 200 else {\"error\": \"Failed to fetch test stats\"}\n"
                "        generator_data = generator_stats.json() if generator_stats.status_code == 200 else {\"error\": \"Failed to fetch generator stats\"}\n"
                "        benchmark_data = benchmark_stats.json() if benchmark_stats.status_code == 200 else {\"error\": \"Failed to fetch benchmark stats\"}\n",
                predictive_response
            )
            updated_method_body = updated_method_body.replace(responses_match.group(1), updated_responses)
        
        # Find the combine data section
        combine_pattern = r'# Combine data(.*?)except Exception as e:'
        combine_match = re.search(combine_pattern, method_body, re.DOTALL)
        
        if combine_match:
            combine_section = combine_match.group(1)
            
            # Add our data
            predictive_combine = """
        # Combine data
        return {
            "test_stats": test_data,
            "generator_stats": generator_data,
            "benchmark_stats": benchmark_data,
            "predictive_stats": predictive_data,
            "timestamp": import_datetime.datetime.now().isoformat()
        }
        """
            
            updated_combine = combine_section.replace(
                "        # Combine data\n"
                "        return {\n"
                "            \"test_stats\": test_data,\n"
                "            \"generator_stats\": generator_data,\n"
                "            \"benchmark_stats\": benchmark_data,\n"
                "            \"timestamp\": import_datetime.datetime.now().isoformat()\n"
                "        }\n",
                predictive_combine
            )
            updated_method_body = updated_method_body.replace(combine_match.group(1), updated_combine)
    
    # Update the method body in the content
    updated_content = content.replace(method_match.group(1), updated_method_body)
    return updated_content

def update_start_stop_services(content):
    """Update the start_all and stop_all methods to include Predictive Performance API.
    
    Args:
        content: Content of the unified_api_server.py file
        
    Returns:
        Updated content
    """
    # Find the start_all method
    start_all_pattern = r'def start_all\(self\):(.*?)def stop_service\('
    start_all_match = re.search(start_all_pattern, content, re.DOTALL)
    
    if not start_all_match:
        logger.error("Could not find start_all method in unified_api_server.py")
        return content
    
    start_all_body = start_all_match.group(1)
    
    # Find the component APIs section
    component_pattern = r'# Start component APIs(.*?)# Start API gateway'
    component_match = re.search(component_pattern, start_all_body, re.DOTALL)
    
    if not component_match:
        logger.error("Could not find component APIs section in start_all method")
        return content
    
    component_section = component_match.group(1)
    
    # Add our service
    predictive_service = """
        # Start component APIs
        for service_name in ["test_api", "generator_api", "benchmark_api", "predictive_performance_api"]:
            self.start_service(service_name)
        """
    
    updated_component = component_section.replace(
        "        # Start component APIs\n"
        "        for service_name in [\"test_api\", \"generator_api\", \"benchmark_api\"]:\n"
        "            self.start_service(service_name)\n",
        predictive_service
    )
    updated_start_all = start_all_body.replace(component_match.group(1), updated_component)
    updated_content = content.replace(start_all_match.group(1), updated_start_all)
    
    # Find the stop_all method
    stop_all_pattern = r'def stop_all\(self\):(.*?)def check_status\('
    stop_all_match = re.search(stop_all_pattern, updated_content, re.DOTALL)
    
    if not stop_all_match:
        logger.error("Could not find stop_all method in unified_api_server.py")
        return updated_content
    
    stop_all_body = stop_all_match.group(1)
    
    # Find the stop in reverse order section
    reverse_pattern = r'# Stop in reverse order(.*?)# Clean up temp files'
    reverse_match = re.search(reverse_pattern, stop_all_body, re.DOTALL)
    
    if not reverse_match:
        logger.error("Could not find stop in reverse order section in stop_all method")
        return updated_content
    
    reverse_section = reverse_match.group(1)
    
    # Add our service
    predictive_reverse = """
        # Stop in reverse order (gateway first, then component APIs)
        for service_name in ["gateway", "predictive_performance_api", "benchmark_api", "generator_api", "test_api"]:
            self.stop_service(service_name)
        """
    
    updated_reverse = reverse_section.replace(
        "        # Stop in reverse order (gateway first, then component APIs)\n"
        "        for service_name in [\"gateway\", \"benchmark_api\", \"generator_api\", \"test_api\"]:\n"
        "            self.stop_service(service_name)\n",
        predictive_reverse
    )
    updated_stop_all = stop_all_body.replace(reverse_match.group(1), updated_reverse)
    updated_content = updated_content.replace(stop_all_match.group(1), updated_stop_all)
    
    return updated_content

def update_main_function(content):
    """Update the main function to include Predictive Performance API command line arguments.
    
    Args:
        content: Content of the unified_api_server.py file
        
    Returns:
        Updated content
    """
    # Find the main function
    main_pattern = r'def main\(\):(.*?)if __name__ == "__main__":'
    main_match = re.search(main_pattern, content, re.DOTALL)
    
    if not main_match:
        logger.error("Could not find main function in unified_api_server.py")
        return content
    
    main_body = main_match.group(1)
    
    # Find the command line arguments section
    args_pattern = r'parser.add_argument\("--benchmark-api-port",(.*?)args = parser.parse_args\(\)'
    args_match = re.search(args_pattern, main_body, re.DOTALL)
    
    if not args_match:
        logger.error("Could not find command line arguments section in main function")
        return content
    
    args_section = args_match.group(1)
    
    # Add our argument
    predictive_arg = """    parser.add_argument("--predictive-performance-api-port", type=int, default=8500, help="Port for the Predictive Performance API")
    """
    
    updated_args = args_section + predictive_arg
    updated_main = main_body.replace(args_match.group(1), updated_args)
    
    # Find the override ports section
    ports_pattern = r'# Override ports from command line arguments(.*?)# Create service manager'
    ports_match = re.search(ports_pattern, main_body, re.DOTALL)
    
    if not ports_match:
        logger.error("Could not find override ports section in main function")
        return content
    
    ports_section = ports_match.group(1)
    
    # Add our port
    predictive_port = """    # Override ports from command line arguments
    config["gateway"]["port"] = args.gateway_port
    config["test_api"]["port"] = args.test_api_port
    config["generator_api"]["port"] = args.generator_api_port
    config["benchmark_api"]["port"] = args.benchmark_api_port
    config["predictive_performance_api"]["port"] = args.predictive_performance_api_port
    """
    
    updated_ports = ports_section.replace(
        "    # Override ports from command line arguments\n"
        "    config[\"gateway\"][\"port\"] = args.gateway_port\n"
        "    config[\"test_api\"][\"port\"] = args.test_api_port\n"
        "    config[\"generator_api\"][\"port\"] = args.generator_api_port\n"
        "    config[\"benchmark_api\"][\"port\"] = args.benchmark_api_port\n",
        predictive_port
    )
    updated_main = updated_main.replace(ports_match.group(1), updated_ports)
    
    updated_content = content.replace(main_match.group(1), updated_main)
    
    return updated_content

def update_unified_api_server():
    """Update the unified_api_server.py file to integrate the Predictive Performance API.
    
    Returns:
        True if the update was successful, False otherwise
    """
    # Find the unified_api_server.py file
    unified_api_path = find_unified_api_server()
    
    if unified_api_path is None:
        logger.error("Could not find unified_api_server.py")
        return False
    
    logger.info(f"Found unified_api_server.py at {unified_api_path}")
    
    # Read the file
    with open(unified_api_path, 'r') as f:
        content = f.read()
    
    # Create a backup
    backup_file(unified_api_path)
    
    # Update the content
    updated_content = update_default_config(content)
    updated_content = update_gateway_file_generation(updated_content)
    updated_content = update_start_stop_services(updated_content)
    updated_content = update_main_function(updated_content)
    
    # Write the updated content
    with open(unified_api_path, 'w') as f:
        f.write(updated_content)
    
    logger.info(f"Updated {unified_api_path}")
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Update Unified API Server")
    parser.add_argument("--restore", action="store_true", help="Restore from backup")
    args = parser.parse_args()
    
    if args.restore:
        # Find the unified_api_server.py file
        unified_api_path = find_unified_api_server()
        
        if unified_api_path is None:
            logger.error("Could not find unified_api_server.py")
            return 1
        
        # Find the backup
        backup_path = unified_api_path.with_suffix(unified_api_path.suffix + ".bak")
        
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return 1
        
        # Restore from backup
        shutil.copy2(backup_path, unified_api_path)
        logger.info(f"Restored {unified_api_path} from backup")
        return 0
    
    # Update the unified_api_server.py file
    if update_unified_api_server():
        logger.info("Successfully updated unified_api_server.py")
        return 0
    else:
        logger.error("Failed to update unified_api_server.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())