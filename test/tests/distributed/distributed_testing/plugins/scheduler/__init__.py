"""
Scheduler Plugin Module for Distributed Testing Framework

This module provides extensibility for custom task scheduling algorithms through plugins.
"""

from test.tests.distributed.distributed_testing.plugins.scheduler.scheduler_plugin_interface import SchedulerPluginInterface, SchedulingStrategy
from test.tests.distributed.distributed_testing.plugins.scheduler.scheduler_plugin_registry import SchedulerPluginRegistry

__all__ = [
    'SchedulerPluginInterface',
    'SchedulingStrategy',
    'SchedulerPluginRegistry',
]