#!/bin/bash

# System Check for GitHub Actions Runner Installation
# Checks system readiness for runner installation

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}GitHub Actions Runner System Check${NC}"
echo "=================================="
echo ""

# Check OS
echo -e "${BLUE}Operating System:${NC}"
echo "  OS: $(lsb_release -d 2>/dev/null | cut -f2 || echo "$(uname -s) $(uname -r)")"
echo "  Architecture: $(uname -m)"
echo ""

# Check required tools
echo -e "${BLUE}Required Tools:${NC}"
for tool in curl wget git tar; do
    if command -v $tool &> /dev/null; then
        echo -e "  ✓ $tool: ${GREEN}installed${NC}"
    else
        echo -e "  ✗ $tool: ${RED}not installed${NC}"
    fi
done
echo ""

# Check Docker
echo -e "${BLUE}Docker:${NC}"
if command -v docker &> /dev/null; then
    echo -e "  ✓ Docker: ${GREEN}installed${NC} ($(docker --version | cut -d' ' -f3))"
    if docker ps &> /dev/null; then
        echo -e "  ✓ Docker access: ${GREEN}available${NC}"
    else
        echo -e "  ✗ Docker access: ${RED}denied${NC} (user not in docker group or service not running)"
    fi
else
    echo -e "  ✗ Docker: ${RED}not installed${NC}"
fi
echo ""

# Check Python
echo -e "${BLUE}Python:${NC}"
if command -v python3 &> /dev/null; then
    echo -e "  ✓ Python 3: ${GREEN}installed${NC} ($(python3 --version | cut -d' ' -f2))"
    if python3 -m pip --version &> /dev/null; then
        echo -e "  ✓ pip: ${GREEN}available${NC}"
    else
        echo -e "  ✗ pip: ${RED}not available${NC}"
    fi
else
    echo -e "  ✗ Python 3: ${RED}not installed${NC}"
fi
echo ""

# Check GPU support
echo -e "${BLUE}GPU Support:${NC}"
gpu_found=false

if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo -e "  ✓ NVIDIA GPU: ${GREEN}detected${NC} (CUDA support available)"
        gpu_found=true
    else
        echo -e "  ⚠ NVIDIA drivers: ${YELLOW}installed but not working${NC}"
    fi
elif lspci | grep -i nvidia &> /dev/null; then
    echo -e "  ⚠ NVIDIA GPU: ${YELLOW}detected but drivers not installed${NC}"
fi

if command -v rocm-smi &> /dev/null; then
    echo -e "  ✓ AMD GPU: ${GREEN}detected${NC} (ROCm support available)"
    gpu_found=true
elif lspci | grep -i amd | grep -i vga &> /dev/null; then
    echo -e "  ⚠ AMD GPU: ${YELLOW}detected but ROCm not installed${NC}"
fi

if command -v intel_gpu_top &> /dev/null; then
    echo -e "  ✓ Intel GPU: ${GREEN}detected${NC} (OpenVINO support available)"
    gpu_found=true
elif lspci | grep -i intel | grep -i vga &> /dev/null; then
    echo -e "  ⚠ Intel GPU: ${YELLOW}detected but tools not installed${NC}"
fi

if [ "$gpu_found" = false ]; then
    echo -e "  ℹ GPU: ${BLUE}CPU-only configuration${NC}"
fi
echo ""

# Check system resources
echo -e "${BLUE}System Resources:${NC}"
echo "  CPU cores: $(nproc)"
echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "  Disk space (root): $(df -h / | awk 'NR==2 {print $4}') available"
echo ""

# Check existing runners
echo -e "${BLUE}Existing Runners:${NC}"
if [ -d "$HOME/actions-runner" ]; then
    echo -e "  ⚠ Primary runner: ${YELLOW}directory exists${NC} ($HOME/actions-runner)"
else
    echo -e "  ✓ Primary runner: ${GREEN}not installed${NC}"
fi

if [ -d "$HOME/actions-runner-backup" ]; then
    echo -e "  ⚠ Backup runner: ${YELLOW}directory exists${NC} ($HOME/actions-runner-backup)"
else
    echo -e "  ✓ Backup runner: ${GREEN}not installed${NC}"
fi

# Check services
if systemctl is-active --quiet github-actions-runner 2>/dev/null; then
    echo -e "  ⚠ Primary service: ${YELLOW}running${NC}"
elif systemctl list-unit-files | grep -q github-actions-runner 2>/dev/null; then
    echo -e "  ⚠ Primary service: ${YELLOW}installed but not running${NC}"
else
    echo -e "  ✓ Primary service: ${GREEN}not installed${NC}"
fi

if systemctl is-active --quiet github-actions-runner-backup 2>/dev/null; then
    echo -e "  ⚠ Backup service: ${YELLOW}running${NC}"
elif systemctl list-unit-files | grep -q github-actions-runner-backup 2>/dev/null; then
    echo -e "  ⚠ Backup service: ${YELLOW}installed but not running${NC}"
else
    echo -e "  ✓ Backup service: ${GREEN}not installed${NC}"
fi
echo ""

# Recommendations
echo -e "${BLUE}Recommendations:${NC}"

# Check if dependencies need to be installed
missing_deps=false
for tool in curl wget git tar; do
    if ! command -v $tool &> /dev/null; then
        missing_deps=true
        break
    fi
done

if [ "$missing_deps" = true ]; then
    echo -e "  • ${YELLOW}Install missing dependencies${NC}: Use --install-deps flag"
fi

if ! command -v docker &> /dev/null; then
    echo -e "  • ${YELLOW}Install Docker${NC}: Use --install-deps flag or install manually"
elif ! docker ps &> /dev/null; then
    echo -e "  • ${YELLOW}Fix Docker access${NC}: Add user to docker group or start Docker service"
fi

if [ ! -d "$HOME/actions-runner" ]; then
    echo -e "  • ${GREEN}Ready for primary runner installation${NC}"
else
    echo -e "  • ${YELLOW}Primary runner exists${NC}: Consider backup runner only"
fi

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Get your GitHub token from: https://github.com/endomorphosis/ipfs_accelerate_py/settings/actions/runners"
echo "2. Run installation script:"

if [ ! -d "$HOME/actions-runner" ]; then
    echo "   ./scripts/setup-github-runner.sh --token YOUR_TOKEN --install-deps"
else
    echo "   ./scripts/setup-backup-runner.sh YOUR_TOKEN"
fi

echo "3. Verify installation in GitHub repository settings"
echo ""
echo "For detailed instructions, see: GITHUB_RUNNER_INSTALLATION.md"