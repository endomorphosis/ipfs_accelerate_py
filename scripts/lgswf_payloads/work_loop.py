"""Pre-execution and provisional semantic work loop."""
from types import MappingProxyType
class WorkLoopError(ValueError): pass
def run_provisional_loop(record):
    if record.get("publish_canonical"):
        raise WorkLoopError("provisional loop cannot publish canonical")
    return MappingProxyType({"provisional": True, "canonical": False, "scanned": True})
