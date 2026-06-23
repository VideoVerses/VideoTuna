from pathlib import Path

from scripts import CI_SMOKE_TESTS, COVERAGE_GATE_FAIL_UNDER, coverage_gate


def test_ci_smoke_tests_exist():
    repo_root = Path(__file__).resolve().parents[1]
    for rel_path in CI_SMOKE_TESTS:
        assert (repo_root / rel_path).is_file(), f"missing CI smoke test: {rel_path}"


def test_coverage_gate_entrypoint():
    assert callable(coverage_gate)


def test_coverage_gate_threshold_is_modest():
    assert 0 < COVERAGE_GATE_FAIL_UNDER < 50
