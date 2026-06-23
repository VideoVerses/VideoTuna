"""Guard: domain entrypoints must not reference pruned Wan variants."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_TOKENS = (
    "s2v",
    "animate",
    "ti2v",
    "speech2video",
    "textimage2video",
    "WanS2V",
    "WanAnimate",
    "WanTI2V",
)

DOMAIN_PATHS = (
    REPO_ROOT / "configs" / "domain",
    REPO_ROOT / "scripts" / "__init__.py",
    REPO_ROOT / "scripts" / "train_new.py",
    REPO_ROOT / "videotuna" / "flow" / "wanvideo.py",
    REPO_ROOT / "videotuna" / "training" / "wan_lora",
    REPO_ROOT / "videotuna" / "utils" / "wan_training.py",
)


def _iter_text_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*")) if path.is_dir() else []


def _find_forbidden_references() -> list[str]:
    violations: list[str] = []
    for domain_path in DOMAIN_PATHS:
        for file_path in _iter_text_files(domain_path):
            if not file_path.is_file() or file_path.suffix not in {
                ".py",
                ".yaml",
                ".yml",
                ".json",
            }:
                continue
            text = file_path.read_text(encoding="utf-8")
            for token in FORBIDDEN_TOKENS:
                if token in text:
                    rel = file_path.relative_to(REPO_ROOT)
                    violations.append(f"{rel}: contains {token!r}")
    return violations


def test_domain_entrypoints_exclude_pruned_wan_variants():
    """Domain scripts and configs must not reference s2v/animate/ti2v."""
    violations = _find_forbidden_references()
    assert violations == [], "Pruned Wan variant references found:\n" + "\n".join(
        violations
    )
