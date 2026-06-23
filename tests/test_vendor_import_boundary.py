"""Guard vendor-only third-party imports stay inside vendored trees."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEOTUNA_ROOT = REPO_ROOT / "videotuna"
WAN_VENDOR_ROOT = VIDEOTUNA_ROOT / "models" / "wan"
WAN_CONFIGS_ROOT = WAN_VENDOR_ROOT / "wan" / "configs"

_EASYDICT_MARKERS = ("from easydict", "import easydict")


def _iter_py_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _file_mentions_easydict(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return any(marker in text for marker in _EASYDICT_MARKERS)


def test_easydict_only_in_wan_vendor_tree():
    """easydict imports must stay inside videotuna/models/wan/."""
    outside_wan = [
        path
        for path in _iter_py_files(VIDEOTUNA_ROOT)
        if WAN_VENDOR_ROOT not in path.parents and _file_mentions_easydict(path)
    ]
    assert outside_wan == [], (
        "easydict is Wan-vendor-only; remove imports from: "
        + ", ".join(str(p.relative_to(REPO_ROOT)) for p in outside_wan)
    )


def test_wan_configs_still_use_easydict():
    """Sanity: upstream Wan config modules still depend on easydict."""
    config_files = _iter_py_files(WAN_CONFIGS_ROOT)
    assert (
        config_files
    ), "expected Wan config modules under videotuna/models/wan/wan/configs/"
    users = [path for path in config_files if _file_mentions_easydict(path)]
    assert users, "expected at least one Wan config module to import easydict"
