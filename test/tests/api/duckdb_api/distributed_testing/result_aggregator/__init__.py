"""
Result Aggregator Module for Distributed Testing Framework

This module provides components for aggregating and analyzing results from distributed tests.
"""

from test.tests.api.duckdb_api.distributed_testing.result_aggregator.aggregator import ResultAggregator
from test.tests.api.duckdb_api.distributed_testing.result_aggregator.service import (
    ResultAggregatorService,
    RESULT_TYPE_PERFORMANCE,
    RESULT_TYPE_COMPATIBILITY,
    RESULT_TYPE_INTEGRATION,
    RESULT_TYPE_WEB_PLATFORM,
    AGGREGATION_LEVEL_TEST_RUN,
    AGGREGATION_LEVEL_MODEL,
    AGGREGATION_LEVEL_HARDWARE,
    AGGREGATION_LEVEL_MODEL_HARDWARE,
    AGGREGATION_LEVEL_TASK_TYPE,
    AGGREGATION_LEVEL_WORKER,
)

__all__ = [
    'ResultAggregator',
    'ResultAggregatorService',
    'RESULT_TYPE_PERFORMANCE',
    'RESULT_TYPE_COMPATIBILITY',
    'RESULT_TYPE_INTEGRATION',
    'RESULT_TYPE_WEB_PLATFORM',
    'AGGREGATION_LEVEL_TEST_RUN',
    'AGGREGATION_LEVEL_MODEL',
    'AGGREGATION_LEVEL_HARDWARE',
    'AGGREGATION_LEVEL_MODEL_HARDWARE',
    'AGGREGATION_LEVEL_TASK_TYPE',
    'AGGREGATION_LEVEL_WORKER',
]