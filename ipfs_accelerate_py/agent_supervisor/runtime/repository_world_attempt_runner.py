"""One strict router attempt with independently durable lifecycle records.

The caller constructs a journal, retaining its start before this wrapper can
validate or dispatch. The router snapshots its actual accounting receipt before
historical/output processing. Missing records remain unknown; this wrapper
neither retries nor parses output to reconstruct a receipt or token count.
"""
from __future__ import annotations

from .repository_world_attempt_journal import RepositoryWorldAttemptJournal


SCHEMA = "repository-world-attempt-runner-retention@1"


def _error_type(error):
    try:
        from .router_implementation_runner import RUNNER_FAILURE_TYPES

        name = type(error).__name__
        return name if name in RUNNER_FAILURE_TYPES else "other"
    except BaseException:
        return "other"


def _finish(journal, *, outcome, error_type=None):
    try:
        RepositoryWorldAttemptJournal.finish(journal, outcome=outcome, error_type=error_type)
        return False
    except BaseException:
        # A partial terminal write is evidence to preserve. It cannot replace
        # the provider's return or original exception, or imply completion.
        return True


def _close(journal):
    try:
        RepositoryWorldAttemptJournal.close(journal)
        return False
    except BaseException:
        return True


def _diagnostic(journal, *, terminal_finish_failed, close_failed):
    try:
        retained = RepositoryWorldAttemptJournal.receipt(journal)
        unavailable = False
    except BaseException:
        retained, unavailable = None, True
    return dict(schema=SCHEMA, journal=retained,
        journal_diagnostic_unavailable=unavailable,
        terminal_finish_failed=terminal_finish_failed, close_failed=close_failed,
        provider_outcome_reclassified=False, provider_delivery_verified=False,
        token_counts_verified=False, billing_verified=False,
        independent_campaign_census_verified=False,
        execution_authority=False, completion_authority=False)


def run_repository_world_packet_attempt(*, journal, runner_options: dict):
    """Return ``(output, receipt, retention_diagnostic)`` for one router call.

    Construct the exact journal first: its exclusive durable start exists even
    when the existing router refuses arguments, workspace, packet or provider.
    The journal is checked again by the router before native closing fences.
    The wrapper preserves both provider success and the same raised exception
    when receipt capture, terminal recording or closing is unavailable.

    Receipt/output enrichment belongs after this function. Its terminal outcome
    describes this runner call, never acceptance, proof, billing or completion.
    """
    try:
        RepositoryWorldAttemptJournal.require_open(journal)
        if type(runner_options) is not dict:
            raise ValueError("world attempt runner requires exact keyword options")
        if "world_task_attempt_journal" in runner_options:
            raise ValueError("world attempt runner owns its journal keyword")
        from . import router_implementation_runner

        output, receipt = router_implementation_runner.run(
            **runner_options, world_task_attempt_journal=journal)
    except BaseException as error:
        try:
            interrupted = isinstance(error, (KeyboardInterrupt, SystemExit, GeneratorExit))
            _finish(journal, outcome="interrupted" if interrupted else "raised",
                error_type=_error_type(error))
        finally:
            _close(journal)
        raise
    terminal_finish_failed = _finish(journal, outcome="returned")
    close_failed = _close(journal)
    return output, receipt, _diagnostic(journal,
        terminal_finish_failed=terminal_finish_failed, close_failed=close_failed)
