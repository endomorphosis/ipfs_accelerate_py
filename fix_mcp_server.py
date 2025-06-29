#!/usr/bin/env python3
"""
MCP Server Diagnostic and Fix Tool

This script diagnoses and fixes common issues with the IPFS Accelerate MCP server.
"""

import os
import sys
import json
import subprocess
import time
import shutil
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
RESET = "\033[0m"

# Configuration
SERVER_FILE = "final_mcp_server.py"
LOG_FILE = "diagnostic_mcp_test.log"
REQUIRED_MODULES = ["aiohttp", "ipfshttpclient", "fastapi", "uvicorn", "pydantic"]
PORT = 8002

def print_colored(color, message):
    """Print a colored message."""
    print(f"{color}{message}{RESET}")

def check_file(filename):
    """Check if a file exists and is readable."""
    if not os.path.isfile(filename):
        print_colored(RED, f"❌ File not found: {filename}")
        return False
    
    if not os.access(filename, os.R_OK):
        print_colored(RED, f"❌ File not readable: {filename}")
        return False
    
    print_colored(GREEN, f"✅ File exists and is readable: {filename}")
    return True

def check_executable(filename):
    """Check if a file is executable."""
    if not os.path.isfile(filename):
        print_colored(RED, f"❌ File not found: {filename}")
        return False
    
    if not os.access(filename, os.X_OK):
        print_colored(RED, f"❌ File not executable: {filename}")
        return False
    
    print_colored(GREEN, f"✅ File is executable: {filename}")
    return True

def make_executable(filename):
    """Make a file executable."""
    try:
        current_mode = os.stat(filename).st_mode
        os.chmod(filename, current_mode | 0o111)
        print_colored(GREEN, f"✅ Made file executable: {filename}")
        return True
    except Exception as e:
        print_colored(RED, f"❌ Error making file executable: {e}")
        return False

def check_module(module_name):
    """Check if a Python module is installed."""
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")
        print_colored(GREEN, f"✅ Module {module_name} is installed (version: {version})")
        return True
    except ImportError:
        print_colored(RED, f"❌ Module {module_name} is not installed")
        return False
    except Exception as e:
        print_colored(RED, f"❌ Error checking module {module_name}: {e}")
        return False

def check_port(port):
    """Check if a port is in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(("127.0.0.1", port))
        if result == 0:
            print_colored(YELLOW, f"⚠️ Port {port} is in use")
            return True
        else:
            print_colored(GREEN, f"✅ Port {port} is available")
            return False

def kill_process_on_port(port):
    """Kill any process using a specific port."""
    try:
        # Find processes using the port
        if sys.platform == "win32":
            cmd = f"netstat -ano | findstr :{port}"
            output = subprocess.check_output(cmd, shell=True).decode()
            lines = output.strip().split("\n")
            for line in lines:
                if "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True)
                    print_colored(GREEN, f"✅ Killed process {pid} using port {port}")
        else:
            cmd = f"lsof -i :{port} -t"
            output = subprocess.check_output(cmd, shell=True).decode()
            pids = output.strip().split("\n")
            for pid in pids:
                if pid:
                    subprocess.run(["kill", "-9", pid])
                    print_colored(GREEN, f"✅ Killed process {pid} using port {port}")
        return True
    except subprocess.CalledProcessError:
        # No process using the port
        return True
    except Exception as e:
        print_colored(RED, f"❌ Error killing process on port {port}: {e}")
        return False

def backup_file(filename):
    """Create a backup of a file."""
    backup = f"{filename}.bak"
    try:
        shutil.copy2(filename, backup)
        print_colored(GREEN, f"✅ Created backup: {backup}")
        return True
    except Exception as e:
        print_colored(RED, f"❌ Error creating backup: {e}")
        return False

def fix_python_script(filename):
    """Fix common issues in Python scripts."""
    if not os.path.isfile(filename):
        return False
    
    # Create a backup
    backup_file(filename)
    
    try:
        with open(filename, "r") as f:
            content = f.read()
        
        # Fix common issues:
        # 1. Make sure the script has the proper shebang
        if not content.startswith("#!/usr/bin/env python"):
            content = "#!/usr/bin/env python3\n" + content
            print_colored(YELLOW, "⚠️ Added shebang to script")
        
        # 2. Check for common import issues
        if "import sys" not in content:
            # Find the first import statement
            import_pos = content.find("import ")
            if import_pos > 0:
                content = content[:import_pos] + "import sys\n" + content[import_pos:]
                print_colored(YELLOW, "⚠️ Added missing sys import")
        
        # 3. Add debug output
        if "if __name__ == \"__main__\":" in content and "print(\"Starting server\")" not in content:
            content = content.replace(
                "if __name__ == \"__main__\":",
                "if __name__ == \"__main__\":\n    print(\"Starting server\")"
            )
            print_colored(YELLOW, "⚠️ Added debug output to script")
        
        # Write fixed content
        with open(filename, "w") as f:
            f.write(content)
        
        print_colored(GREEN, f"✅ Fixed and saved: {filename}")
        return True
    except Exception as e:
        print_colored(RED, f"❌ Error fixing script: {e}")
        # Restore backup
        try:
            shutil.copy2(f"{filename}.bak", filename)
            print_colored(YELLOW, f"⚠️ Restored backup due to error")
        except:
            pass
        return False

def fix_server_script():
    """Fix issues in the server script."""
    print_colored(YELLOW, f"Attempting to fix {SERVER_FILE}...")
    
    if not check_file(SERVER_FILE):
        print_colored(RED, f"❌ Cannot fix {SERVER_FILE} - file not found")
        return False
    
    # Make executable
    if not check_executable(SERVER_FILE):
        make_executable(SERVER_FILE)
    
    # Fix script issues
    fix_python_script(SERVER_FILE)
    
    return True

def install_requirements():
    """Install required Python modules."""
    print_colored(YELLOW, "Installing required modules...")
    
    if os.path.isfile("requirements.txt"):
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print_colored(GREEN, "✅ Installed requirements from requirements.txt")
            return True
        except Exception as e:
            print_colored(RED, f"❌ Error installing requirements: {e}")
    
    # Install individual modules if requirements.txt installation failed
    try:
        for module in REQUIRED_MODULES:
            subprocess.run([sys.executable, "-m", "pip", "install", module], check=True)
        print_colored(GREEN, "✅ Installed required modules individually")
        return True
    except Exception as e:
        print_colored(RED, f"❌ Error installing modules: {e}")
        return False

def test_server():
    """Test if the server works."""
    print_colored(YELLOW, f"Testing {SERVER_FILE}...")
    
    # Kill any process using the port
    if check_port(PORT):
        if not kill_process_on_port(PORT):
            print_colored(RED, f"❌ Could not free port {PORT}")
            return False
    
    # Run the server
    try:
        process = subprocess.Popen(
            [sys.executable, SERVER_FILE, "--debug"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it time to start
        time.sleep(3)
        
        # Check if it's still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print_colored(RED, f"❌ Server exited with code {process.returncode}")
            print_colored(RED, f"Output:\n{stdout}")
            print_colored(RED, f"Errors:\n{stderr}")
            return False
        
        # Try to connect
        import urllib.request
        try:
            response = urllib.request.urlopen(f"http://localhost:{PORT}/health")
            if response.status == 200:
                print_colored(GREEN, "✅ Server is working!")
            else:
                print_colored(RED, f"❌ Server returned status {response.status}")
        except Exception as e:
            print_colored(RED, f"❌ Error connecting to server: {e}")
            return False
        finally:
            # Stop the server
            process.terminate()
            process.wait(timeout=5)
        
        return True
    except Exception as e:
        print_colored(RED, f"❌ Error testing server: {e}")
        return False

def main():
    print("\n" + "="*60)
    print(" IPFS Accelerate MCP Server Diagnostic and Fix Tool ")
    print("="*60 + "\n")
    
    print_colored(YELLOW, "Step 1: Checking server script...")
    check_file(SERVER_FILE)
    
    print_colored(YELLOW, "\nStep 2: Checking required modules...")
    modules_ok = True
    for module in REQUIRED_MODULES:
        if not check_module(module):
            modules_ok = False
    
    print_colored(YELLOW, "\nStep 3: Checking port availability...")
    check_port(PORT)
    
    print_colored(YELLOW, "\nApplying fixes if needed...")
    if not check_executable(SERVER_FILE):
        make_executable(SERVER_FILE)
    
    if not modules_ok:
        install_requirements()
    
    fix_server_script()
    
    print_colored(YELLOW, "\nStep 4: Testing server...")
    if test_server():
        print_colored(GREEN, "\n✅ The MCP server is now working correctly!")
    else:
        print_colored(RED, "\n❌ The MCP server is still not working correctly.")
        print_colored(RED, "Please check the logs for more information.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
