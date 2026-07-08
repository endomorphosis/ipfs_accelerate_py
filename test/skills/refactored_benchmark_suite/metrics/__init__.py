"""
Performance metrics collection modules for the refactored benchmark suite.
"""

from .timing import LatencyMetric, ThroughputMetric
from .memory import MemoryMetric
from .flops import FLOPsMetric
from .power import PowerMetric
from .bandwidth import BandwidthMetric

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