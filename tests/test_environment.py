import pytest

from registry import environment
from registry.environment import EnvironmentMismatch, assert_loadable, capture, differences


def present(monkeypatch, versions: dict[str, str]) -> None:
    """Pretends the process has these versions installed."""
    monkeypatch.setattr(
        environment, "installed", lambda name: versions.get(name, environment.MISSING)
    )


def test_capture_records_python_and_every_named_package(monkeypatch):
    present(monkeypatch, {"scikit-learn": "1.9.0", "numpy": "2.5.2"})

    record = capture(("numpy", "scikit-learn"))

    assert record["packages"] == {"numpy": "2.5.2", "scikit-learn": "1.9.0"}
    assert record["python"].count(".") == 2


def test_a_package_that_is_not_installed_is_recorded_as_absent(monkeypatch):
    present(monkeypatch, {})

    assert capture(("lightgbm",))["packages"] == {"lightgbm": environment.MISSING}


def test_matching_versions_produce_no_differences(monkeypatch):
    present(monkeypatch, {"scikit-learn": "1.9.0"})
    record = {"packages": {"scikit-learn": "1.9.0"}}

    assert differences(record) == []


def test_a_patch_difference_is_reported_without_blocking(monkeypatch):
    present(monkeypatch, {"numpy": "2.5.3"})
    record = {"packages": {"numpy": "2.5.2"}}

    (difference,) = differences(record)

    assert difference.blocking is False
    assert str(difference) == "numpy: recorded 2.5.2, present 2.5.3"


def test_a_minor_difference_blocks(monkeypatch):
    """The failure this exists for: 1.8 reading what 1.9 wrote, without an exception."""
    present(monkeypatch, {"scikit-learn": "1.8.2"})
    record = {"packages": {"scikit-learn": "1.9.0"}}

    (difference,) = differences(record)

    assert difference.blocking is True


def test_a_patch_difference_blocks_for_a_package_named_exact(monkeypatch):
    present(monkeypatch, {"scikit-learn": "1.9.1"})
    record = {"packages": {"scikit-learn": "1.9.0"}}

    (difference,) = differences(record, exact=frozenset({"scikit-learn"}))

    assert difference.blocking is True


def test_a_package_that_disappeared_blocks(monkeypatch):
    present(monkeypatch, {})
    record = {"packages": {"joblib": "1.5.0"}}

    (difference,) = differences(record)

    assert difference.blocking is True
    assert difference.present == environment.MISSING


def test_assert_loadable_passes_and_returns_the_harmless_differences(monkeypatch):
    present(monkeypatch, {"numpy": "2.5.3", "scipy": "1.16.0"})
    record = {"packages": {"numpy": "2.5.2", "scipy": "1.16.0"}}

    remaining = assert_loadable(record)

    assert [difference.package for difference in remaining] == ["numpy"]


def test_assert_loadable_names_every_blocking_package(monkeypatch):
    present(monkeypatch, {"scikit-learn": "1.8.2", "numpy": "2.4.0"})
    record = {"packages": {"scikit-learn": "1.9.0", "numpy": "2.5.2"}}

    with pytest.raises(EnvironmentMismatch) as raised:
        assert_loadable(record)

    message = str(raised.value)
    assert "scikit-learn: recorded 1.9.0, present 1.8.2" in message
    assert "numpy: recorded 2.5.2, present 2.4.0" in message


def test_a_record_without_packages_is_not_a_failure():
    assert differences({}) == []


def test_installed_reads_the_real_metadata():
    """The monkeypatched tests above would pass against a stub that never worked."""
    assert environment.installed("mlflow") == environment.version("mlflow")
    assert environment.installed("not-a-real-distribution") == environment.MISSING
