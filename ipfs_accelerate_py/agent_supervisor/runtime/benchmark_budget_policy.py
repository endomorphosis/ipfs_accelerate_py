"""Explicit development time allowances; these confer no execution authority."""

FORMALIZATION_BUDGET_PROFILE = "source384-5cpu-20gib-formalization@1"
FORMALIZATION_NATIVE_LIFETIME_SECONDS = 2160
FORMALIZATION_WORK_SECONDS = 6300


def validate_budget_profile(profile):
    if profile is not None and (type(profile) is not str or profile != FORMALIZATION_BUDGET_PROFILE):
        raise ValueError("explicit reviewed benchmark budget profile required")
    return profile


def source384_timeout_limit(profile=None):
    validate_budget_profile(profile)
    return 3600 if profile is not None else 180


def local_work_timeout_limit(profile=None):
    validate_budget_profile(profile)
    return FORMALIZATION_WORK_SECONDS if profile is not None else 900


def validate_native_lifetime(seconds, *, budget_profile=None):
    validate_budget_profile(budget_profile)
    if budget_profile is None:
        if type(seconds) is not int or not 120 <= seconds <= 900:
            raise ValueError("signed launch lifetime_seconds must be in 120..900")
    elif type(seconds) is not int or seconds != FORMALIZATION_NATIVE_LIFETIME_SECONDS:
        raise ValueError("formalization launch requires its exact 2160-second lifetime")
