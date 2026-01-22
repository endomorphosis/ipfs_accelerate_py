#!/usr/bin/env python3
"""
CLI Validation and Fix Script

This script validates the ipfs-accelerate CLI functionality and provides
solutions for common installation issues.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def test_cli_direct():
    """Test CLI by running the module directly"""
    print("🧪 Testing CLI direct execution...")
    try:
        result = subprocess.run([
            sys.executable, "cli.py", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ Direct CLI execution: SUCCESS")
            return True
        else:
            print(f"❌ Direct CLI execution failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Direct CLI execution error: {e}")
        return False

def test_mcp_server():
    """Test MCP server functionality"""
    print("🧪 Testing MCP server functionality...")
    try:
        result = subprocess.run([
            sys.executable, "cli.py", "mcp", "start", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ MCP server commands: SUCCESS")
            return True
        else:
            print(f"❌ MCP server commands failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ MCP server test error: {e}")
        return False

def test_dashboard_functionality():
    """Test dashboard functionality without full startup"""
    print("🧪 Testing dashboard functionality...")
    try:
        # Import the dashboard HTML generation
        sys.path.insert(0, str(Path(__file__).parent))
        from cli import dashboard_html_content
        
        # Verify HTML content is generated
        html = dashboard_html_content()
        if len(html) > 10000 and "IPFS Accelerate" in html:
            print("✅ Dashboard HTML generation: SUCCESS")
            return True
        else:
            print("❌ Dashboard HTML generation: FAILED")
            return False
    except Exception as e:
        print(f"❌ Dashboard test error: {e}")
        return False

def create_entry_point_script():
    """Create a standalone entry point script"""
    print("🔧 Creating standalone entry point script...")
    
    script_content = '''#!/usr/bin/env python3
"""
IPFS Accelerate CLI Entry Point

This script provides the ipfs-accelerate command functionality.
"""
import sys
import os
from pathlib import Path

# Add the package directory to the Python path
package_dir = Path(__file__).parent
sys.path.insert(0, str(package_dir))

# Import and run the CLI
try:
    from cli import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"Error importing CLI module: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -e .")
    sys.exit(1)
'''
    
    entry_point_path = Path(__file__).parent / "ipfs_accelerate_cli.py"
    with open(entry_point_path, 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(entry_point_path, 0o755)
    print(f"✅ Created entry point script: {entry_point_path}")
    return str(entry_point_path)

def test_entry_point_script():
    """Test the standalone entry point script"""
    print("🧪 Testing standalone entry point script...")
    
    # Create the entry point script
    script_path = create_entry_point_script()
    
    try:
        result = subprocess.run([
            sys.executable, script_path, "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Standalone entry point: SUCCESS")
            return True
        else:
            print(f"❌ Standalone entry point failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Entry point test error: {e}")
        return False
    finally:
        # Clean up the created script
        try:
            if os.path.exists(script_path):
                os.remove(script_path)
        except:
            pass

def provide_installation_guidance():
    """Provide installation guidance for users"""
    print("\n" + "="*60)
    print("📋 INSTALLATION GUIDANCE")
    print("="*60)
    
    print("\n🚀 To use ipfs-accelerate CLI, you have several options:")
    
    print("\n1️⃣ DIRECT EXECUTION (Recommended for development):")
    print("   cd /path/to/ipfs_accelerate_py")
    print("   python cli.py mcp start --dashboard")
    
    print("\n2️⃣ PYTHON MODULE EXECUTION:")
    print("   cd /path/to/ipfs_accelerate_py")
    print("   python -m cli mcp start --dashboard")
    
    print("\n3️⃣ INSTALL IN DEVELOPMENT MODE:")
    print("   cd /path/to/ipfs_accelerate_py")
    print("   pip install -e .")
    print("   ipfs-accelerate mcp start --dashboard")
    
    print("\n4️⃣ USE STANDALONE SCRIPT:")
    print("   python ipfs_accelerate_cli.py mcp start --dashboard")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("   - If you get import errors, ensure you're in the correct directory")
    print("   - If dependencies are missing, install them:")
    print("     pip install fastmcp uvicorn psutil numpy torch")
    print("   - For virtual environments, activate before running:")
    print("     source venv/bin/activate  # or venv\\Scripts\\activate on Windows")

def main():
    """Main validation function"""
    print("🚀 IPFS Accelerate CLI Validation Script")
    print("="*50)
    
    # Test results
    results = []
    
    # Test direct CLI execution
    results.append(("Direct CLI", test_cli_direct()))
    
    # Test MCP server functionality
    results.append(("MCP Server", test_mcp_server()))
    
    # Test dashboard functionality
    results.append(("Dashboard", test_dashboard_functionality()))
    
    # Create and test entry point script
    script_path = create_entry_point_script()
    results.append(("Entry Point", test_entry_point_script(script_path)))
    
    # Summary
    print("\n" + "="*50)
    print("📊 VALIDATION RESULTS")
    print("="*50)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:15} {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\n📈 Overall: {total_passed}/{len(results)} tests passed")
    
    # Provide guidance
    provide_installation_guidance()
    
    return total_passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)