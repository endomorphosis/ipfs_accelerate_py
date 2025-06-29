#!/usr/bin/env python3
"""
Update Unified API Server with Hardware Optimization endpoints

This script updates the Unified API Server to include endpoints for
hardware optimization recommendations.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("update_hardware_optimization")

def update_unified_api(
    server_file_path: str = "test/api_server/unified_api_server.py",
    benchmark_db_path: str = "benchmark_db.duckdb",
    api_url: str = "http://localhost:8080",
    api_key: str = None
):
    """
    Update the Unified API Server with Hardware Optimization integration.
    
    Args:
        server_file_path: Path to the Unified API Server file
        benchmark_db_path: Path to benchmark database
        api_url: API base URL
        api_key: Optional API key
    """
    server_path = Path(server_file_path)
    
    if not server_path.exists():
        logger.error(f"Server file not found: {server_file_path}")
        return False
    
    try:
        # Read the server file
        with open(server_path, 'r') as f:
            content = f.read()
        
        # Check if the integration is already added
        if "hardware_optimization_integration" in content:
            logger.info("Hardware Optimization integration already added")
            return True
        
        # Find the imports section
        imports_section = content.find("# Import integrations")
        if imports_section == -1:
            logger.error("Could not find imports section")
            return False
        
        # Find the register integrations section
        register_section = content.find("# Register integrations")
        if register_section == -1:
            logger.error("Could not find register integrations section")
            return False
        
        # Find the cleanup section
        cleanup_section = content.find("# Cleanup integrations")
        if cleanup_section == -1:
            logger.error("Could not find cleanup section")
            return False
        
        # Update imports
        imports_update = """# Import integrations
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import integrations
try:
    from test.api_server.integrations.predictive_performance_integration import create_integration as create_pp_integration
    PREDICTIVE_PERFORMANCE_AVAILABLE = True
except ImportError:
    PREDICTIVE_PERFORMANCE_AVAILABLE = False
    logger.warning("Predictive Performance integration not available")

try:
    from test.api_server.integrations.hardware_optimization_integration import create_integration as create_hw_opt_integration
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    logger.warning("Hardware Optimization integration not available")
"""
        
        # Update register section
        register_update = """# Register integrations
    # Predictive Performance API
    if PREDICTIVE_PERFORMANCE_AVAILABLE:
        try:
            pp_integration = create_pp_integration(
                database_path=database_path,
                api_key=api_key
            )
            pp_integration.register_routes(app)
            integrations.append(pp_integration)
            logger.info("Predictive Performance integration registered")
        except Exception as e:
            logger.error(f"Error registering Predictive Performance integration: {e}")
    
    # Hardware Optimization API
    if HARDWARE_OPTIMIZATION_AVAILABLE:
        try:
            hw_opt_integration = create_hw_opt_integration(
                benchmark_db_path=database_path,
                api_url=f"http://{host}:{port}",
                api_key=api_key
            )
            hw_opt_integration.register_routes(app)
            integrations.append(hw_opt_integration)
            logger.info("Hardware Optimization integration registered")
        except Exception as e:
            logger.error(f"Error registering Hardware Optimization integration: {e}")
"""
        
        # Update cleanup section
        cleanup_update = """# Cleanup integrations
    @app.on_event("shutdown")
    def shutdown_event():
        logger.info("Shutting down Unified API Server")
        for integration in integrations:
            if hasattr(integration, "close"):
                integration.close()
"""
        
        # Replace sections
        new_content = content[:imports_section] + imports_update
        new_content += content[imports_section + len("# Import integrations"):register_section] + register_update
        new_content += content[register_section + len("# Register integrations"):cleanup_section] + cleanup_update
        new_content += content[cleanup_section + len("# Cleanup integrations"):]
        
        # Write updated content
        with open(server_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Updated {server_file_path} with Hardware Optimization integration")
        return True
        
    except Exception as e:
        logger.error(f"Error updating Unified API Server: {e}")
        return False

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update Unified API Server with Hardware Optimization integration")
    parser.add_argument("--server-file", type=str, default="test/api_server/unified_api_server.py",
                      help="Path to the Unified API Server file")
    parser.add_argument("--benchmark-db", type=str, default="benchmark_db.duckdb",
                      help="Path to benchmark database")
    parser.add_argument("--api-url", type=str, default="http://localhost:8080",
                      help="API base URL")
    parser.add_argument("--api-key", type=str, help="Optional API key")
    
    args = parser.parse_args()
    
    success = update_unified_api(
        server_file_path=args.server_file,
        benchmark_db_path=args.benchmark_db,
        api_url=args.api_url,
        api_key=args.api_key
    )
    
    if success:
        print("Unified API Server updated successfully!")
    else:
        print("Failed to update Unified API Server")
        sys.exit(1)

if __name__ == "__main__":
    main()