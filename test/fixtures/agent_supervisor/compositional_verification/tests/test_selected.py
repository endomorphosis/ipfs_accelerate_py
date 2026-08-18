from pkg.module_c import present


def test_selected_contract_path() -> None:
    assert present(5) == 22


def test_selected_exception_path() -> None:
    try:
        present(-1)
    except ValueError:
        return
    raise AssertionError("negative input must preserve ValueError")
