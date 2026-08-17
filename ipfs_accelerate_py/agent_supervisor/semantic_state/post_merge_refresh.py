"""Pre-merge seal and post-merge canonical refresh."""
from types import MappingProxyType
class RefreshError(ValueError): pass
def refresh_canonical(record):
    if not record.get("accepted_merge") or not record.get("fresh_rescan"):
        raise RefreshError("canonical refresh requires merge+rescan")
    if record.get("provisional"):
        raise RefreshError("provisional root cannot become canonical")
    return MappingProxyType({"canonical": True})
