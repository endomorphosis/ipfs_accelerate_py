"""Dynamic import used only for the opaque-plugin escalation path."""

from importlib import import_module


def load_unaffected_label() -> str:
    """Load the unaffected module by name and read its stable label."""

    module = import_module("pkg.unaffected")
    getter = getattr(module, "stable_label")
    return str(getter())
