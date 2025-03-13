#!/usr/bin/env python3
"""
Run script for CI/CD Integration tests and examples

This script demonstrates the CI/CD Integration capabilities of the
Distributed Testing Framework by running the CI integration example.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure proper imports by adding parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the example
from distributed_testing.examples.ci_integration_example import run_example

# Main function to run the example
async def main():
    """Run the CI integration example."""
    logger.info("Running CI/CD Integration example...")
    await run_example()
    logger.info("CI/CD Integration example complete")

# Run the example
if __name__ == "__main__":
    asyncio.run(main())