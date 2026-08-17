"""Decision receipts and highest-level machine output."""
from types import MappingProxyType
def emit_decision(record):
    return MappingProxyType({"schema": "lgswf/decision-receipt@1", "decision": record.get("decision"), "metrics": record.get("metrics") or {}})
