#!/usr/bin/env python3
"""
Update Unified API Server with Export Optimization endpoints

This script updates the Unified API Server to include endpoints for
exporting hardware optimization recommendations to deployable files.
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
logger = logging.getLogger("update_export_optimization")

def update_unified_api(
    server_file_path: str = "test/api_server/unified_api_server.py",
    benchmark_db_path: str = "benchmark_db.duckdb",
    api_url: str = "http://localhost:8080",
    api_key: str = None,
    output_dir: str = "./optimization_exports"
):
    """
    Update the Unified API Server with Export Optimization integration.
    
    Args:
        server_file_path: Path to the Unified API Server file
        benchmark_db_path: Path to benchmark database
        api_url: API base URL
        api_key: Optional API key
        output_dir: Directory for optimization exports
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
        if "export_optimization_integration" in content:
            logger.info("Export Optimization integration already added")
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
        
        # Extract the existing imports
        imports_end = content.find("\n\n", imports_section)
        if imports_end == -1:
            imports_end = register_section
        
        existing_imports = content[imports_section:imports_end]
        
        # Add our import to the imports section
        imports_update = existing_imports + "\n"
        if "EXPORT_OPTIMIZATION_AVAILABLE" not in existing_imports:
            imports_update += """
try:
    from test.api_server.integrations.export_optimization_integration import create_integration as create_export_integration
    EXPORT_OPTIMIZATION_AVAILABLE = True
except ImportError:
    EXPORT_OPTIMIZATION_AVAILABLE = False
    logger.warning("Export Optimization integration not available")
"""
        
        # Find where to insert our registration code
        if "Hardware Optimization API" in content:
            # Add after hardware optimization
            insert_point = content.find("except Exception as e:", register_section)
            insert_point = content.find("\n", insert_point) + 1
        else:
            # Add at the end of the register section
            insert_point = content.find("\n", register_section + len("# Register integrations")) + 1
        
        # Create registration code
        register_code = """
    # Export Optimization API
    if EXPORT_OPTIMIZATION_AVAILABLE:
        try:
            output_dir = os.path.join(os.path.dirname(database_path), "optimization_exports")
            os.makedirs(output_dir, exist_ok=True)
            
            export_integration = create_export_integration(
                benchmark_db_path=database_path,
                api_url=f"http://{host}:{port}",
                api_key=api_key,
                output_dir=output_dir
            )
            export_integration.register_routes(app)
            integrations.append(export_integration)
            logger.info("Export Optimization integration registered")
        except Exception as e:
            logger.error(f"Error registering Export Optimization integration: {e}")
"""
        
        # Update the content
        new_content = content[:imports_section] + imports_update
        new_content += content[imports_end:insert_point] + register_code
        new_content += content[insert_point:]
        
        # Make sure os module is imported
        if "import os" not in new_content[:100]:
            # Find the first import statement
            first_import = new_content.find("import ")
            if first_import != -1:
                new_content = new_content[:first_import] + "import os\n" + new_content[first_import:]
        
        # Write updated content
        with open(server_path, 'w') as f:
            f.write(new_content)
        
        logger.info(f"Updated {server_file_path} with Export Optimization integration")
        return True
        
    except Exception as e:
        logger.error(f"Error updating Unified API Server: {e}")
        return False

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Update Unified API Server with Export Optimization integration")
    parser.add_argument("--server-file", type=str, default="test/api_server/unified_api_server.py",
                      help="Path to the Unified API Server file")
    parser.add_argument("--benchmark-db", type=str, default="benchmark_db.duckdb",
                      help="Path to benchmark database")
    parser.add_argument("--api-url", type=str, default="http://localhost:8080",
                      help="API base URL")
    parser.add_argument("--api-key", type=str, help="Optional API key")
    parser.add_argument("--output-dir", type=str, default="./optimization_exports",
                      help="Directory for optimization exports")
    
    args = parser.parse_args()
    
    success = update_unified_api(
        server_file_path=args.server_file,
        benchmark_db_path=args.benchmark_db,
        api_url=args.api_url,
        api_key=args.api_key,
        output_dir=args.output_dir
    )
    
    if success:
        print("Unified API Server updated successfully!")
    else:
        print("Failed to update Unified API Server")
        sys.exit(1)

if __name__ == "__main__":
    main()