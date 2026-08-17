"""Deterministic Plan Doctor proposals."""
from types import MappingProxyType
class PlanDoctorError(ValueError): pass
def diagnose(record):
    if record.get("mutate_accepted"):
        raise PlanDoctorError("cannot mutate accepted history")
    return MappingProxyType({"proposal": True, "accepted": False, "finding": record.get("finding") or "none"})
