"""Explicit common Harbor resources and bounded execution budgets.

The extended development profile is a distinct experiment. Select it for every
comparison arm; existing task defaults and the original profile stay unchanged.
"""
from copy import deepcopy

SOURCE384_PROFILE = "source384-5cpu-12gib@1"
EXTENDED_SOURCE384_PROFILE = "source384-5cpu-16gib-extended@1"
PROFILES = (SOURCE384_PROFILE, EXTENDED_SOURCE384_PROFILE)
SOURCE384_ENVIRONMENT = dict(override_cpus=5, override_memory_mb=12288,
    cpu_enforcement_policy="limit", memory_enforcement_policy="limit")


def resource_environment(profile):
    if profile not in PROFILES:
        raise ValueError("unknown benchmark resource profile")
    result = dict(SOURCE384_ENVIRONMENT)
    if profile == EXTENDED_SOURCE384_PROFILE:
        result["override_memory_mb"] = 16384
    return result


def execution_budget(profile=None):
    if profile not in (None, *PROFILES):
        raise ValueError("unknown benchmark resource profile")
    if profile == EXTENDED_SOURCE384_PROFILE:
        return dict(driver_seconds=900, cleanup_seconds=60, source384_seconds=180,
                    harbor_seconds=960, exec_seconds=910, qualification_seconds=600,
                    qualification_exec_seconds=630)
    return dict(driver_seconds=285, cleanup_seconds=40, source384_seconds=90,
                harbor_seconds=300, exec_seconds=295, qualification_seconds=270,
                qualification_exec_seconds=300)


def admission_environment(profile=None):
    execution_budget(profile)  # Reject unknown profiles even without admission changes.
    return ({"IPFS_DATASETS_PROOF_RESOURCE_PROFILE": "local-benchmark@1",
             "IPFS_DATASETS_RESOURCE_SCHEDULER_PATH": "/opt/ipfs-supervisor/state/resource-scheduler.json"}
            if profile == EXTENDED_SOURCE384_PROFILE else {})


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
