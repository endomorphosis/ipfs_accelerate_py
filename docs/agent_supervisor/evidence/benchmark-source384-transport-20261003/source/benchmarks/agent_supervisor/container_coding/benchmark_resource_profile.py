"""Explicit common Harbor resources; default task limits remain untouched."""
from copy import deepcopy

SOURCE384_PROFILE = "source384-5cpu-8gib@1"
PROFILES = (SOURCE384_PROFILE,)
SOURCE384_ENVIRONMENT = dict(override_cpus=5, override_memory_mb=8192,
    cpu_enforcement_policy="limit", memory_enforcement_policy="limit")


def apply_resource_profile(config, profile=None):
    if profile is None:
        return config
    if profile != SOURCE384_PROFILE:
        raise ValueError("unknown benchmark resource profile")
    result = deepcopy(config)
    environment = result["environment"]
    for key, value in SOURCE384_ENVIRONMENT.items():
        if key in environment and environment[key] != value:
            raise ValueError("benchmark resource profile conflicts with declared limits")
        environment[key] = value
    return result


def validate_resource_profile(config, profile):
    if profile != SOURCE384_PROFILE:
        raise ValueError("Source384 requires the explicit common source384-5cpu-8gib@1 profile")
    environment = config.get("environment", {})
    if any(type(environment.get(key)) is not type(value) or environment[key] != value
           for key, value in SOURCE384_ENVIRONMENT.items()):
        raise ValueError("Source384 declared resource limits differ from the selected common profile")
    return profile
