"""Temporary file for CI lint-autofix verification; remove after merge."""

import os
import sys


def _lint_autofix_smoke() -> tuple[str, str]:
    return (os.getcwd(), sys.version)
