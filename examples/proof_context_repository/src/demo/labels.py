"""An unaffected module whose exact proof can be reused safely."""


def label(value: int) -> str:
    """Render a stable label for an integer."""

    return f"value:{value}"


__all__ = ["label"]
