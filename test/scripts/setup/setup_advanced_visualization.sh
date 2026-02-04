#!/bin/bash
# Setup script for Advanced Visualization System dependencies

echo "Setting up Advanced Visualization System dependencies..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip not found. Please install Python and pip first."
    exit 1
fi

# Install required packages
echo "Installing required packages from advanced_visualization_requirements.txt..."
pip install -r advanced_visualization_requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
    echo "You can now use the Advanced Visualization System."
    echo "Try running: python test_advanced_visualization.py --viz-type all"
else
    echo "❌ Error: Failed to install dependencies."
    echo "Please check your internet connection and try again."
    exit 1
fi

# Create output directory
echo "Creating output directory for visualizations..."
mkdir -p advanced_visualizations

echo "Setup complete! 🚀"
echo "For usage instructions, see ADVANCED_VISUALIZATION_GUIDE.md"