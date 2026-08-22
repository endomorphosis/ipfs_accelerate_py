"""Paired checks for the extra representative task classes."""

from pkg.codec import decode_produced, encode_produced
from pkg.compat import produce_v1
from pkg.lock import CONSUMER_LOCK, PRESENTER_LOCK, ordered_locks
from pkg.module_a import produce
from pkg.plugin import load_unaffected_label
from pkg.policy import admit_limit
from pkg.proof import obligation_text
from pkg.unaffected import stable_label


def test_schema_round_trip() -> None:
    assert decode_produced(encode_produced(10)) == 10


def test_security_policy_admits_small_limit() -> None:
    assert admit_limit(5) is True
    assert admit_limit(-1) is False


def test_compat_api_matches_producer() -> None:
    assert produce_v1(5) == produce(5)


def test_lock_order_is_consumer_then_presenter() -> None:
    consumer, presenter = ordered_locks()
    assert consumer is CONSUMER_LOCK
    assert presenter is PRESENTER_LOCK


def test_dynamic_plugin_reads_unaffected_label() -> None:
    assert load_unaffected_label() == stable_label()


def test_proof_obligation_mentions_schema_bound() -> None:
    text = obligation_text()
    assert "MAX_PRODUCED_VALUE" in text
    assert "producer-upper-bound" in text


def test_unaffected_label_is_stable() -> None:
    assert stable_label() == "unaffected"
