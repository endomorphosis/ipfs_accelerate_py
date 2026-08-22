from pkg.unaffected import stable_label


def test_unselected_unaffected_module() -> None:
    assert stable_label() == "unaffected"
