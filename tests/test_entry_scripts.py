"""Regression tests for Poetry entry scripts (no sys.path hacks)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY_SCRIPTS = (
    REPO_ROOT / "scripts" / "inference_new.py",
    REPO_ROOT / "scripts" / "train_new.py",
)


def test_entry_scripts_do_not_manipulate_sys_path():
    for script_path in ENTRY_SCRIPTS:
        source = script_path.read_text(encoding="utf-8")
        assert (
            "sys.path.insert" not in source
        ), f"{script_path.name} must not use sys.path.insert"
        assert (
            "/src" not in source
        ), f"{script_path.name} must not reference dead src/ path"
