import importlib
import tomllib
from pathlib import Path


def _poetry_script_entrypoints():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    scripts = data.get("tool", {}).get("poetry", {}).get("scripts", {})
    return scripts


def test_poetry_scripts_resolve():
    for name, target in _poetry_script_entrypoints().items():
        module_name, _, attr_name = target.partition(":")
        assert module_name, f"{name} has invalid target {target!r}"
        assert attr_name, f"{name} has invalid target {target!r}"
        module = importlib.import_module(module_name)
        assert hasattr(module, attr_name), f"{name} -> {target} not found"
