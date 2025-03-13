"""
Scheduler Plugin Module for Distributed Testing Framework

This module provides extensibility for custom task scheduling algorithms through plugins.
"""

from .scheduler_plugin_interface import SchedulerPluginInterface, SchedulingStrategy
from .scheduler_plugin_registry import SchedulerPluginRegistry

__all__ = [
    'SchedulerPluginInterface',
    'SchedulingStrategy',
    'SchedulerPluginRegistry',
]