"""Schema serializer used by configuration-sensitive consumers."""

from .schema import MAX_PRODUCED_VALUE


def encode_produced(value: int) -> str:
    """Serialize a produced integer under the current schema bound."""

    if value > MAX_PRODUCED_VALUE:
        raise ValueError("produced value exceeds schema bound")
    return f"produced:{value}"


def decode_produced(payload: str) -> int:
    """Restore a produced integer from the schema serializer."""

    prefix, _, rest = payload.partition(":")
    if prefix != "produced":
        raise ValueError("schema payload is malformed")
    return int(rest)
