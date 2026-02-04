#!/bin/bash
#
# Run comprehensive tests for the Predictive Performance System
#

# Define colors for output formatting
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
NC="\033[0m" # No Color

# Print a header
echo -e "\n${BLUE}===== Predictive Performance System Test Runner =====${NC}\n"

# Define test modes
run_quick_test() {
    echo -e "${YELLOW}Running quick test...${NC}"
    python test_predictive_performance_system.py --quick-test
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}✅ Quick test PASSED${NC}\n"
    else
        echo -e "\n${RED}❌ Quick test FAILED${NC}\n"
    fi
    return $exit_code
}

run_full_test() {
    echo -e "${YELLOW}Running full test suite...${NC}"
    python test_predictive_performance_system.py --full-test
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}✅ All tests PASSED${NC}\n"
    else
        echo -e "\n${RED}❌ Some tests FAILED${NC}\n"
    fi
    return $exit_code
}

run_component_test() {
    component=$1
    echo -e "${YELLOW}Testing component: $component${NC}"
    python test_predictive_performance_system.py --test-component $component
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}✅ $component test PASSED${NC}\n"
    else
        echo -e "\n${RED}❌ $component test FAILED${NC}\n"
    fi
    return $exit_code
}

# Print usage information
print_usage() {
    echo -e "Usage: $0 [options]"
    echo -e ""
    echo -e "Options:"
    echo -e "  -q, --quick       Run a quick test (basic functionality)"
    echo -e "  -f, --full        Run the full test suite"
    echo -e "  -c, --component COMPONENT"
    echo -e "                    Test a specific component:"
    echo -e "                    - prediction: Test prediction functionality"
    echo -e "                    - accuracy: Test prediction accuracy"
    echo -e "                    - recommendation: Test hardware recommendation"
    echo -e "                    - active_learning: Test active learning pipeline"
    echo -e "                    - visualization: Test visualization generation"
    echo -e "                    - scheduler: Test benchmark scheduler integration"
    echo -e "  -h, --help        Display this help message"
    echo -e ""
    echo -e "Examples:"
    echo -e "  $0 --quick                Run a quick test"
    echo -e "  $0 --full                 Run the full test suite"
    echo -e "  $0 --component prediction Test prediction functionality only"
}

# Check for test mode options
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

# Parse command line arguments
case "$1" in
    -q|--quick)
        run_quick_test
        ;;
    -f|--full)
        run_full_test
        ;;
    -c|--component)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Component name required${NC}"
            print_usage
            exit 1
        fi
        component=$2
        valid_components=("prediction" "accuracy" "recommendation" "active_learning" "visualization" "scheduler")
        if [[ ! " ${valid_components[@]} " =~ " ${component} " ]]; then
            echo -e "${RED}Error: Invalid component name: $component${NC}"
            print_usage
            exit 1
        fi
        run_component_test $component
        ;;
    -h|--help)
        print_usage
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid option: $1${NC}"
        print_usage
        exit 1
        ;;
esac

# Exit with the test's exit code
exit $?