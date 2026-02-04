#!/bin/bash
#
# Setup script for the Export Visualization components
#

set -e  # Exit on error

echo "Setting up Export Visualization dependencies..."

# Create directories
mkdir -p exports

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r export_visualization_requirements.txt

# Install Playwright browser
echo "Installing Playwright browser for animation export capabilities..."
python -m playwright install chromium

# Install ImageMagick for GIF creation
echo "Checking for ImageMagick (required for GIF export)..."
if command -v convert >/dev/null 2>&1; then
    echo "ImageMagick is already installed."
else
    echo "ImageMagick is not installed. Attempting to install..."
    
    # Check OS and install accordingly
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update
            sudo apt-get install -y imagemagick
        elif command -v dnf >/dev/null 2>&1; then
            sudo dnf install -y ImageMagick
        elif command -v yum >/dev/null 2>&1; then
            sudo yum install -y ImageMagick
        else
            echo "Unsupported Linux distribution. Please install ImageMagick manually."
            echo "See: https://imagemagick.org/script/download.php"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew >/dev/null 2>&1; then
            brew install imagemagick
        else
            echo "Homebrew not found. Please install ImageMagick manually."
            echo "See: https://imagemagick.org/script/download.php"
        fi
    else
        echo "Unsupported OS. Please install ImageMagick manually."
        echo "See: https://imagemagick.org/script/download.php"
    fi
fi

# Verify installation
echo "Verifying installation..."

python -c "import plotly, pandas, sklearn, duckdb, playwright, PIL, requests, numpy, kaleido" || {
    echo "Error: Some Python packages are not installed correctly."
    exit 1
}

echo "Creating test export directory..."
mkdir -p ./exports/test

echo "Installation complete!"
echo ""
echo "You can now use the export functionality with:"
echo "python run_export_visualization.py export --viz-type 3d"
echo "python run_export_visualization.py export-animation --format mp4"
echo "python run_export_visualization.py export-all"
echo ""
echo "For more options, run:"
echo "python run_export_visualization.py --help"