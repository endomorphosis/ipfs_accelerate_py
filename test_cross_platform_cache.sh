#!/bin/bash
# Cross-Platform Cache Test - Linux/Mac Script
# This script helps Linux/Mac users run the cross-platform cache test

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Cross-Platform Cache Test (Linux/Mac)${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} Python 3 is not installed"
    echo ""
    echo "Please install Python 3.8+ from your package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "  macOS: brew install python3"
    exit 1
fi

echo -e "${GREEN}[OK]${NC} Python found"
python3 --version
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}[INFO]${NC} Virtual environment not found, creating one..."
    python3 -m venv venv
    echo -e "${GREEN}[OK]${NC} Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo -e "${YELLOW}[INFO]${NC} Activating virtual environment..."
source venv/bin/activate

echo -e "${GREEN}[OK]${NC} Virtual environment activated"
echo ""

# Check if dependencies are installed
echo -e "${YELLOW}[INFO]${NC} Checking dependencies..."
if ! python -c "import cryptography" &> /dev/null; then
    echo -e "${YELLOW}[WARN]${NC} Dependencies not installed, installing now..."
    echo ""
    pip install --upgrade pip
    pip install cryptography py-multiformats-cid
    
    # Try to install libp2p
    echo -e "${YELLOW}[INFO]${NC} Installing libp2p..."
    if pip install libp2p>=0.4.0 pymultihash>=0.8.2; then
        echo -e "${GREEN}[OK]${NC} libp2p installed successfully"
    else
        echo -e "${YELLOW}[WARN]${NC} libp2p installation failed"
        echo -e "${YELLOW}[INFO]${NC} Will continue without P2P support"
    fi
    echo ""
fi

echo -e "${GREEN}[OK]${NC} Dependencies checked"
echo ""

# Run the cross-platform test
echo -e "${YELLOW}[INFO]${NC} Running cross-platform test..."
echo -e "${BLUE}============================================${NC}"
echo ""

if python test_cross_platform_cache.py; then
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${GREEN}[SUCCESS]${NC} Test completed"
    echo ""
    echo "Review the Platform Compatibility Report above"
    echo ""
    echo "Next steps:"
    echo "  1. Test on your other platform (Windows/Linux)"
    echo "  2. If both pass, proceed to Docker testing"
    echo "  3. See CROSS_PLATFORM_TESTING_GUIDE.md for more info"
    echo -e "${BLUE}============================================${NC}"
    exit 0
else
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${RED}[ERROR]${NC} Test failed with errors"
    echo ""
    echo "Review the output above for details."
    echo ""
    echo "Common issues:"
    echo "  1. Missing build tools (install build-essential)"
    echo "  2. libp2p compilation errors (may need development packages)"
    echo "  3. Permission issues (check file/directory permissions)"
    echo ""
    echo "See CROSS_PLATFORM_TESTING_GUIDE.md for help"
    echo -e "${BLUE}============================================${NC}"
    exit 1
fi
