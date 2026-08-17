"""Bounded semantic refill proposals."""
from types import MappingProxyType
class RefillError(ValueError): pass
def propose_refill(record):
    if record.get("rewrite_accepted"):
        raise RefillError("accepted history is immutable")
    if int(record.get("bound", 1)) > int(record.get("max_bound", 4)):
        raise RefillError("refill exceeds bound")
    return MappingProxyType({"proposal": True, "accepted": False, "bound": int(record.get("bound", 1))})
