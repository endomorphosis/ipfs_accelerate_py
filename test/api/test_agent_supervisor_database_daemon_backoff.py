"""Recovery retains the admitted child and fails closed for other errors."""
import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.database_daemon_backoff import (
    run_database_daemon_with_backoff,
)


@pytest.mark.parametrize("error_name", ["OutOfMemoryException", "MemoryError", "IOException"])
def test_retry_preserves_invocation_and_caps_delay(error_name):
    error = type(error_name, (Exception,), {})
    calls, sleeps = [], []
    def main(argv):
        calls.append(tuple(argv))
        argv.append("local mutation")
        if len(calls) <= 4:
            raise error("secret endpoint token")
        return 7
    assert run_database_daemon_with_backoff(
        main, ["--once"], sleep=sleeps.append, backoff_seconds=(1, 2),
    ) == 7
    assert calls == [("--once",)] * 5
    assert sleeps == [1, 2, 2, 2]


@pytest.mark.parametrize("error", [ValueError("secret token"), RuntimeError("secret token"), SystemExit(78), KeyboardInterrupt()])
def test_nonretryable_errors_preserve_failure(error, caplog):
    calls, sleeps = [], []
    def main(argv):
        calls.append(argv)
        raise error
    with pytest.raises(type(error)) as raised:
        run_database_daemon_with_backoff(main, [], sleep=sleeps.append)
    assert raised.value is error
    assert len(calls) == 1
    assert sleeps == []
    assert "secret token" not in caplog.text


def test_transient_error_does_not_log_credentials(caplog):
    calls = []
    def main(argv):
        calls.append(argv)
        if len(calls) == 1:
            raise MemoryError("secret endpoint token")
    sleeps = []
    assert run_database_daemon_with_backoff(main, [], sleep=sleeps.append, backoff_seconds=()) == 0
    assert sleeps == [30]
    assert "secret endpoint token" not in caplog.text
