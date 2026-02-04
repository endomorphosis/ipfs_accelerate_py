"""
Performance metrics collection modules for the refactored benchmark suite.
"""

from test.tools.skills.refactored_benchmark_suite.metrics.timing import LatencyMetric, ThroughputMetric
from test.tools.skills.refactored_benchmark_suite.metrics.memory import MemoryMetric
from test.tools.skills.refactored_benchmark_suite.metrics.flops import FLOPsMetric
from test.tools.skills.refactored_benchmark_suite.metrics.power import PowerMetric
from test.tools.skills.refactored_benchmark_suite.metrics.bandwidth import BandwidthMetric

def get_available_metrics():
    """Get the list of available metrics."""
    return ["latency", "throughput", "memory", "flops", "power", "bandwidth"]

__all__ = [
    "LatencyMetric",
    "ThroughputMetric",
    "MemoryMetric",
    "FLOPsMetric",
    "PowerMetric",
    "BandwidthMetric",
    "get_available_metrics"
]