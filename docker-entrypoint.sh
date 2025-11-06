#!/bin/bash
set -e

# Docker Container Entrypoint for IPFS Accelerate
# This script runs dependency validation before starting the actual service

echo "========================================="
echo "IPFS Accelerate Container Starting"
echo "========================================="
echo ""

# Display environment
echo "Environment Information:"
echo "  Architecture: $(uname -m)"
echo "  OS: $(uname -s) $(uname -r)"
echo "  Python: $(python3 --version)"
echo "  User: $(whoami)"
echo "  Working Directory: $(pwd)"
echo "  PYTHONPATH: ${PYTHONPATH:-<not set>}"
echo ""

# Set PYTHONPATH to ensure package can be found
export PYTHONPATH=/app:${PYTHONPATH:-}

# Function to install missing dependencies
install_missing_dependencies() {
    echo "Checking and installing missing dependencies..."
    
    # Check for Flask and flask-cors
    if ! python3 -c "import flask" 2>/dev/null; then
        echo "  Installing flask..."
        pip install --quiet --no-cache-dir "flask>=3.0.0" || echo "  Warning: Could not install flask"
    fi
    
    if ! python3 -c "import flask_cors" 2>/dev/null; then
        echo "  Installing flask-cors..."
        pip install --quiet --no-cache-dir "flask-cors>=4.0.0" || echo "  Warning: Could not install flask-cors"
    fi
    
    if ! python3 -c "import werkzeug" 2>/dev/null; then
        echo "  Installing werkzeug..."
        pip install --quiet --no-cache-dir "werkzeug>=3.0.0" || echo "  Warning: Could not install werkzeug"
    fi
    
    if ! python3 -c "import jinja2" 2>/dev/null; then
        echo "  Installing jinja2..."
        pip install --quiet --no-cache-dir "jinja2>=3.1.0" || echo "  Warning: Could not install jinja2"
    fi
    
    if ! python3 -c "import fastmcp" 2>/dev/null; then
        echo "  Installing fastmcp..."
        pip install --quiet --no-cache-dir "fastmcp>=0.1.0" || echo "  Warning: Could not install fastmcp"
    fi
    
    echo "  Dependency check complete"
    echo ""
}

# Install any missing dependencies
install_missing_dependencies

# Run startup validation
echo "Running startup validation checks..."
echo ""

# Allow the validation to run without failing the container
# We'll capture the exit code but continue even if there are warnings
if python3 /app/docker_startup_check.py --verbose; then
    echo ""
    echo "✅ All validation checks passed"
    echo ""
else
    EXIT_CODE=$?
    echo ""
    if [ $EXIT_CODE -eq 1 ]; then
        echo "❌ Validation failed with critical errors"
        echo "   Review the errors above. Container will continue but may not function correctly."
    else
        echo "⚠️  Validation completed with warnings (exit code: $EXIT_CODE)"
        echo "   Continuing with startup - check logs above for details"
    fi
    echo ""
fi

# Function to execute ipfs-accelerate CLI
run_ipfs_cli() {
    # Try different methods to run the CLI
    if python3 -m ipfs_accelerate_py.cli_entry "$@" 2>/dev/null; then
        return 0
    elif python3 -c "from cli import main; import sys; sys.exit(main())" "$@" 2>/dev/null; then
        return 0
    elif [ -f "/app/cli.py" ]; then
        python3 /app/cli.py "$@"
        return $?
    else
        echo "Error: Could not find CLI entry point"
        return 1
    fi
}

# If the first argument is a known command, execute it directly
case "$1" in
    mcp|inference|files|models|network|github|copilot|help|--help|-h|--version)
        echo "Executing command: ipfs-accelerate $@"
        echo "========================================="
        echo ""
        exec python3 -m ipfs_accelerate_py.cli_entry "$@"
        ;;
    
    python|python3)
        echo "Executing: $@"
        echo "========================================="
        echo ""
        exec "$@"
        ;;
    
    bash|sh)
        echo "Starting interactive shell"
        echo "========================================="
        echo ""
        exec "$@"
        ;;
    
    validate|check)
        # Special command to just run validation
        echo "Running validation only..."
        echo "========================================="
        echo ""
        exec python3 /app/docker_startup_check.py --verbose
        ;;
    
    *)
        # If no command or unknown command, check if it looks like a python module
        if [ "$1" = "-m" ] || [ -z "$1" ]; then
            echo "Executing: python3 $@"
            echo "========================================="
            echo ""
            exec python3 "$@"
        else
            # Default: assume it's arguments for ipfs-accelerate CLI
            echo "Executing command: ipfs-accelerate $@"
            echo "========================================="
            echo ""
            exec python3 -m ipfs_accelerate_py.cli_entry "$@"
        fi
        ;;
esac
