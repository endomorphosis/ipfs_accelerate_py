"""Explicit common Harbor resources and bounded execution budgets.

The extended development profile is a distinct experiment. Select it for every
comparison arm; existing task defaults and the original profile stay unchanged.
"""
from copy import deepcopy
import math
from ipfs_accelerate_py.agent_supervisor.runtime.benchmark_budget_policy import (
    FORMALIZATION_BUDGET_PROFILE, FORMALIZATION_NATIVE_LIFETIME_SECONDS,
)

SOURCE384_PROFILE = "source384-5cpu-12gib@1"
EXTENDED_SOURCE384_PROFILE = "source384-5cpu-16gib-extended@1"
PLANNER180_SOURCE384_PROFILE = "source384-5cpu-16gib-planner180@1"
PLANNER180_20GIB_SOURCE384_PROFILE = "source384-5cpu-20gib-planner180@1"
CODING600_SOURCE384_PROFILE = "source384-5cpu-20gib-coding600@1"
FORMALIZATION_SOURCE384_PROFILE = FORMALIZATION_BUDGET_PROFILE
EXTENDED_PROFILES = (EXTENDED_SOURCE384_PROFILE, PLANNER180_SOURCE384_PROFILE,
                     PLANNER180_20GIB_SOURCE384_PROFILE, CODING600_SOURCE384_PROFILE,
                     FORMALIZATION_SOURCE384_PROFILE)
PROFILES = (SOURCE384_PROFILE, *EXTENDED_PROFILES)
SOURCE384_ENVIRONMENT = dict(override_cpus=5, override_memory_mb=12288,
    cpu_enforcement_policy="limit", memory_enforcement_policy="limit")


def resource_environment(profile):
    if profile not in PROFILES:
        raise ValueError("unknown benchmark resource profile")
    result = dict(SOURCE384_ENVIRONMENT)
    if profile in (PLANNER180_20GIB_SOURCE384_PROFILE, CODING600_SOURCE384_PROFILE,
                   FORMALIZATION_SOURCE384_PROFILE):
        result["override_memory_mb"] = 20480
    elif profile in EXTENDED_PROFILES:
        result["override_memory_mb"] = 16384
    return result


def execution_budget(profile=None):
    if profile not in (None, *PROFILES):
        raise ValueError("unknown benchmark resource profile")
    if profile == FORMALIZATION_SOURCE384_PROFILE:
        return dict(driver_seconds=6420, cleanup_seconds=120, source384_seconds=3600,
                    harbor_seconds=6480, exec_seconds=6450, qualification_seconds=3900,
                    qualification_exec_seconds=3930, native_start_seconds=120,
                    preparation_seconds=3600, planning_seconds=660,
                    native_setup_seconds=180, implementation_seconds=1860,
                    native_lifetime_seconds=FORMALIZATION_NATIVE_LIFETIME_SECONDS)
    if profile in EXTENDED_PROFILES:
        return dict(driver_seconds=900, cleanup_seconds=60, source384_seconds=180,
                    harbor_seconds=960, exec_seconds=910, qualification_seconds=600,
                    qualification_exec_seconds=630, native_start_seconds=120)
    return dict(driver_seconds=285, cleanup_seconds=40, source384_seconds=90,
                harbor_seconds=300, exec_seconds=295, qualification_seconds=270,
                qualification_exec_seconds=300, native_start_seconds=20)


def planner_timeout_seconds(profile=None):
    """Explicit provider allowance within the selected enclosing planner budget."""
    execution_budget(profile)
    if profile == FORMALIZATION_SOURCE384_PROFILE:
        return 600
    return 180 if profile in (PLANNER180_SOURCE384_PROFILE, PLANNER180_20GIB_SOURCE384_PROFILE,
                             CODING600_SOURCE384_PROFILE) else 90


def coding_timeout_seconds(profile=None):
    """Provider cap; only the formalization profile reserves it independently."""
    execution_budget(profile)
    if profile == FORMALIZATION_SOURCE384_PROFILE:
        return 1800
    return 600 if profile == CODING600_SOURCE384_PROFILE else 300


def implementation_watchdog_seconds(profile=None):
    """Allow bounded provider settlement without extending the enclosing run."""
    return coding_timeout_seconds(profile) + 60


class ReservedPhaseBudget:
    """Cumulative preparation and independent later phases under one outer bound.

    Time in planning cannot refill preparation. Every transition checks the
    preceding deadline, including when an interrupt was caught by a component.
    The caller installs the returned deadlines as process alarms.
    """
    _ORDER = ("preparation", "planning", "preparation", "native_setup", "implementation")

    def __init__(self, *, started, profile):
        if profile != FORMALIZATION_SOURCE384_PROFILE:
            raise ValueError("reserved phases require the explicit formalization profile")
        budget = execution_budget(profile)
        self.limits = {name: budget[name + "_seconds"] for name in set(self._ORDER)}
        self.consumed = dict.fromkeys(self.limits, 0.)
        self.outer_deadline = started + budget["driver_seconds"] - budget["cleanup_seconds"]
        self.phase = None
        self.index = 0
        self.entered = started
        self.deadline = self.outer_deadline

    def enter(self, phase, *, now):
        if self.index >= len(self._ORDER) or phase != self._ORDER[self.index]:
            raise ValueError("benchmark phase transition out of order")
        if self.phase is not None:
            self.remaining(now=now)
            self.consumed[self.phase] += now - self.entered
        else:
            # State/report setup and initial scope imports are preparation too;
            # leaving them outside a phase would steal from the last reserve.
            self.consumed[phase] += now - self.entered
        self.phase, self.entered = phase, now
        self.index += 1
        self.deadline = min(self.outer_deadline, now + self.limits[phase] - self.consumed[phase])
        self.remaining(now=now)
        return self.deadline

    def remaining(self, *, now):
        value = self.deadline - now
        if value <= 0:
            raise TimeoutError("the benchmark " + str(self.phase) + " budget is exhausted")
        return value


def native_start_timeout_ms(profile=None, *, remaining_work_seconds):
    """Select an explicit START allowance without extending the work deadline.

    Legacy profiles omit the override and retain the driver's shared 20-second
    START/STOP timeout. The extended profile separates START's admission and
    bootstrap work from STOP's existing cleanup allowance. The work alarm still
    covers runtime construction and every subsequent operation.
    """
    budget = execution_budget(profile)
    if (type(remaining_work_seconds) not in (int, float)
            or not math.isfinite(remaining_work_seconds) or remaining_work_seconds < 0):
        raise ValueError("remaining work must be finite non-negative seconds")
    if profile not in EXTENDED_PROFILES:
        return None
    timeout = int(min(budget["native_start_seconds"], remaining_work_seconds) * 1000)
    if timeout < 2000:
        raise TimeoutError("insufficient work budget for bounded native START")
    return timeout


def admission_environment(profile=None):
    execution_budget(profile)  # Reject unknown profiles even without admission changes.
    return ({"IPFS_DATASETS_PROOF_RESOURCE_PROFILE": "local-benchmark@1",
             "IPFS_DATASETS_RESOURCE_SCHEDULER_PATH": "/opt/ipfs-supervisor/state/resource-scheduler.json"}
            if profile in EXTENDED_PROFILES else {})


def apply_resource_profile(config, profile=None):
    if profile is None:
        return config
    result = deepcopy(config)
    environment = result["environment"]
    for key, value in resource_environment(profile).items():
        if key in environment and (type(environment[key]) is not type(value) or environment[key] != value):
            raise ValueError("benchmark resource profile conflicts with declared limits")
        environment[key] = value
    return result


def validate_resource_profile(config, profile):
    if profile not in PROFILES:
        raise ValueError("Source384 requires an explicit common resource profile")
    expected = resource_environment(profile)
    environment = config.get("environment", {})
    if any(type(environment.get(key)) is not type(value) or environment[key] != value
           for key, value in expected.items()):
        raise ValueError("Source384 declared resource limits differ from the selected common profile")
    budget = execution_budget(profile)
    for agent in config.get("agents", []):
        if any(agent.get(key) != budget["harbor_seconds"] for key in ("override_timeout_sec", "max_timeout_sec")):
            raise ValueError("benchmark time limits differ from the selected common profile")
    return profile
