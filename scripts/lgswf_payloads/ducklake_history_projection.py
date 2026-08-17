"""Optional non-authoritative DuckLake history projection."""
from types import MappingProxyType
def project_history(record):
    return MappingProxyType({"authoritative": False, "observed": bool(record.get("receipt"))})
