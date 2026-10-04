#!/bin/bash

# Verify Runner Coverage
# Ensures at least one runner is configured for each repository

set -e

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  GitHub Actions Runner Coverage Verification                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Expected repositories (add/remove as needed)
EXPECTED_REPOS=(
    "https://github.com/endomorphosis/ipfs_accelerate_py"
    "https://github.com/endomorphosis/ipfs_datasets_py"
    "https://github.com/endomorphosis/ipfs_kit_py"
    "https://github.com/endomorphosis/swissknife"
    "https://github.com/navichat"
)

# Find all runner directories
RUNNER_DIRS=(
    "/home/barberb/actions-runner"
    "/home/barberb/actions-runner-ipfs_datasets_py"
    "/home/barberb/actions-runners/endomorphosis-ipfs_kit_py"
    "/home/barberb/swissknife/actions-runner"
    "/home/barberb/motion/actions-runner"
)

# Additional potential runner locations
for dir in /home/barberb/actions-runner* /home/barberb/*/actions-runner /home/barberb/actions-runners/*; do
    if [ -f "$dir/.runner" ]; then
        # Check if not already in list
        already_listed=false
        for existing in "${RUNNER_DIRS[@]}"; do
            if [ "$dir" = "$existing" ]; then
                already_listed=true
                break
            fi
        done
        if [ "$already_listed" = false ]; then
            RUNNER_DIRS+=("$dir")
        fi
    fi
done

# Build repository -> runner mapping
declare -A REPO_RUNNERS
declare -A RUNNER_STATUS

echo -e "${BLUE}Scanning configured runners...${NC}"
echo ""

for runner_dir in "${RUNNER_DIRS[@]}"; do
    if [ -f "$runner_dir/.runner" ]; then
        # Extract repository URL
        repo_url=$(cat "$runner_dir/.runner" 2>/dev/null | jq -r '.gitHubUrl // empty' 2>/dev/null)
        runner_name=$(cat "$runner_dir/.runner" 2>/dev/null | jq -r '.agentName // empty' 2>/dev/null)
        
        if [ -n "$repo_url" ]; then
            # Check if runner is actually running
            is_running=false
            if pgrep -f "$runner_dir/bin/Runner.Listener" > /dev/null; then
                is_running=true
                RUNNER_STATUS["$runner_dir"]="✓ Running"
            else
                RUNNER_STATUS["$runner_dir"]="✗ Stopped"
            fi
            
            # Add to mapping
            if [ -z "${REPO_RUNNERS[$repo_url]}" ]; then
                REPO_RUNNERS["$repo_url"]="$runner_name ($runner_dir)"
            else
                REPO_RUNNERS["$repo_url"]="${REPO_RUNNERS[$repo_url]}, $runner_name ($runner_dir)"
            fi
            
            # Print runner info
            if [ "$is_running" = true ]; then
                echo -e "${GREEN}✓${NC} $runner_name"
            else
                echo -e "${RED}✗${NC} $runner_name"
            fi
            echo "  Repository: $repo_url"
            echo "  Location: $runner_dir"
            if [ "$is_running" = true ]; then
                echo -e "  Status: ${GREEN}Running${NC}"
            else
                echo -e "  Status: ${RED}Stopped${NC}"
            fi
            echo ""
        fi
    fi
done

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Repository Coverage Analysis                                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

missing_runners=0
coverage_ok=0

for repo in "${EXPECTED_REPOS[@]}"; do
    if [ -n "${REPO_RUNNERS[$repo]}" ]; then
        # Count running runners for this repo
        running_count=0
        total_count=0
        
        for runner_dir in "${RUNNER_DIRS[@]}"; do
            if [ -f "$runner_dir/.runner" ]; then
                repo_url=$(cat "$runner_dir/.runner" 2>/dev/null | jq -r '.gitHubUrl // empty' 2>/dev/null)
                if [ "$repo_url" = "$repo" ]; then
                    ((total_count++))
                    if pgrep -f "$runner_dir/bin/Runner.Listener" > /dev/null; then
                        ((running_count++))
                    fi
                fi
            fi
        done
        
        if [ $running_count -gt 0 ]; then
            echo -e "${GREEN}✓${NC} $repo"
            echo "  Runners: $running_count running / $total_count configured"
            ((coverage_ok++))
        else
            echo -e "${YELLOW}⚠${NC} $repo"
            echo "  Runners: $running_count running / $total_count configured"
            echo -e "  ${YELLOW}Warning: No running runners!${NC}"
            ((missing_runners++))
        fi
    else
        echo -e "${RED}✗${NC} $repo"
        echo -e "  ${RED}No runners configured!${NC}"
        ((missing_runners++))
    fi
    echo ""
done

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Summary                                                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Repositories with coverage: ${GREEN}$coverage_ok${NC}"
echo -e "  Repositories missing runners: ${RED}$missing_runners${NC}"
echo -e "  Total repositories checked: ${#EXPECTED_REPOS[@]}"
echo ""

if [ $missing_runners -gt 0 ]; then
    echo -e "${YELLOW}⚠ Action Required:${NC}"
    echo "  Some repositories are missing runners or have no running runners."
    echo "  Please configure or start runners for the affected repositories."
    echo ""
    echo "  To start a stopped runner:"
    echo "    cd <runner_directory>"
    echo "    sudo ./svc.sh start"
    echo ""
    echo "  To configure a new runner, see: GITHUB_RUNNER_INSTALLATION.md"
    exit 1
else
    echo -e "${GREEN}✓ All repositories have at least one running runner!${NC}"
    exit 0
fi
