#!/bin/bash
# run_docker_tests.sh
#
# This script runs the Distributed Testing Framework tests in a Docker environment,
# which provides isolation and consistency for testing.

# Color codes for output formatting
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Distributed Testing Framework - Docker Test Environment   ${NC}"
echo -e "${BLUE}============================================================${NC}"

# Default values
RUN_ALL=1
TEST_TYPE=""
CLEAN=0
DETACHED=0
BUILD=0
LOGS=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --all)
      RUN_ALL=1
      shift
      ;;
    --type)
      RUN_ALL=0
      TEST_TYPE="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --detached | -d)
      DETACHED=1
      shift
      ;;
    --build | -b)
      BUILD=1
      shift
      ;;
    --logs)
      LOGS=1
      shift
      ;;
    --help)
      echo -e "${GREEN}Usage: $0 [options]${NC}"
      echo ""
      echo "Options:"
      echo "  --all               Run all tests (default)"
      echo "  --type TYPE         Run specific test type: integration, fault, monitoring, stress"
      echo "  --clean             Remove containers and volumes before starting"
      echo "  --detached, -d      Run in detached mode (background)"
      echo "  --build, -b         Force rebuild of Docker images"
      echo "  --logs              Follow logs of the tester container"
      echo "  --help              Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0 --all"
      echo "  $0 --type fault"
      echo "  $0 --clean --build"
      echo "  $0 --detached --logs"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
  echo -e "${RED}Error: Docker is not installed or not in the PATH${NC}"
  exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
  echo -e "${RED}Error: Docker Compose is not installed or not in the PATH${NC}"
  exit 1
fi

# Clean up if requested
if [ $CLEAN -eq 1 ]; then
  echo -e "${YELLOW}Cleaning up Docker environment...${NC}"
  docker-compose -f docker-compose.test.yml down -v
  echo -e "${GREEN}Docker environment cleaned${NC}"
fi

# Build or start containers
if [ $DETACHED -eq 1 ]; then
  # Run in detached mode
  if [ $BUILD -eq 1 ]; then
    echo -e "${YELLOW}Building and starting Docker containers in detached mode...${NC}"
    docker-compose -f docker-compose.test.yml up -d --build
  else
    echo -e "${YELLOW}Starting Docker containers in detached mode...${NC}"
    docker-compose -f docker-compose.test.yml up -d
  fi
  
  # Wait for containers to be ready
  echo -e "${YELLOW}Waiting for containers to be ready...${NC}"
  sleep 10
  
  # Modify the tester command based on test type
  if [ $RUN_ALL -eq 0 ] && [ -n "$TEST_TYPE" ]; then
    # Run specific test type
    echo -e "${YELLOW}Running $TEST_TYPE tests...${NC}"
    docker exec dtf-tester bash -c "cd /app/test/duckdb_api/distributed_testing && ./run_all_tests.sh --type $TEST_TYPE"
  else
    # Run all tests
    echo -e "${YELLOW}Running all tests...${NC}"
    docker exec dtf-tester bash -c "cd /app/test/duckdb_api/distributed_testing && ./run_all_tests.sh"
  fi
  
  # Follow logs if requested
  if [ $LOGS -eq 1 ]; then
    echo -e "${YELLOW}Following tester logs...${NC}"
    docker logs -f dtf-tester
  fi
else
  # Run in interactive mode
  # Modify the tester command based on test type
  if [ $RUN_ALL -eq 0 ] && [ -n "$TEST_TYPE" ]; then
    # Update the command in the docker-compose file
    sed -i.bak "s|./run_all_tests.sh|./run_all_tests.sh --type $TEST_TYPE|g" docker-compose.test.yml
    echo -e "${YELLOW}Running $TEST_TYPE tests...${NC}"
  else
    # Make sure the command is set to run all tests
    sed -i.bak "s|./run_all_tests.sh --type.*|./run_all_tests.sh|g" docker-compose.test.yml
    echo -e "${YELLOW}Running all tests...${NC}"
  fi
  
  # Run the containers
  if [ $BUILD -eq 1 ]; then
    echo -e "${YELLOW}Building and starting Docker containers...${NC}"
    docker-compose -f docker-compose.test.yml up --build
  else
    echo -e "${YELLOW}Starting Docker containers...${NC}"
    docker-compose -f docker-compose.test.yml up
  fi
  
  # Restore the original docker-compose file
  mv docker-compose.test.yml.bak docker-compose.test.yml 2>/dev/null || true
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}Test run completed. Check logs for details.${NC}"
echo -e "${BLUE}============================================================${NC}"