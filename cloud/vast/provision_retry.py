#!/usr/bin/env python3
"""Retry wrapper for Vast bootstrap network steps.

Mirrors provisioning.yaml settings.retry backoff for shell/subprocess steps.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROVISIONING_MANIFEST = SCRIPT_DIR / "provisioning.yaml"
BOOTSTRAP_REQUIREMENTS = SCRIPT_DIR / "bootstrap-requirements.txt"
DOWNLOAD_OK_SENTINEL = ".privtune_download_ok"
HUB_CACHE_SENTINEL_DIR = ".privtune_hub_cache"


def _hub_cache_sentinel_path(repo_id: str) -> Path:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    slug = repo_id.replace("/", "--")
    return hf_home / HUB_CACHE_SENTINEL_DIR / slug / DOWNLOAD_OK_SENTINEL


LOG = logging.getLogger("videotuna-provision-retry")


@dataclass(frozen=True)
class RetrySettings:
    max_attempts: int = 5
    initial_delay: float = 2
    backoff_multiplier: float = 2


def load_retry_settings(manifest_path: Path | None = None) -> RetrySettings:
    path = manifest_path or PROVISIONING_MANIFEST
    if not path.is_file():
        return RetrySettings()
    try:
        import yaml
    except ImportError:
        return RetrySettings()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    retry = (data.get("settings") or {}).get("retry") or {}
    return RetrySettings(
        max_attempts=int(retry.get("max_attempts", 5)),
        initial_delay=float(retry.get("initial_delay", 2)),
        backoff_multiplier=float(retry.get("backoff_multiplier", 2)),
    )


def _wait_seconds(settings: RetrySettings, attempt: int) -> float:
    """Mirror tenacity wait_exponential(multiplier, exp_base, min=multiplier)."""
    return max(
        settings.initial_delay,
        settings.initial_delay * (settings.backoff_multiplier ** (attempt - 1)),
    )


def _make_retry_decorator(settings: RetrySettings):
    from tenacity import (
        before_sleep_log,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    return retry(
        stop=stop_after_attempt(settings.max_attempts),
        wait=wait_exponential(
            multiplier=settings.initial_delay,
            exp_base=settings.backoff_multiplier,
            min=settings.initial_delay,
        ),
        retry=retry_if_exception_type((subprocess.CalledProcessError, OSError)),
        before_sleep=before_sleep_log(LOG, logging.WARNING),
        reraise=True,
    )


def run_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    settings: RetrySettings | None = None,
) -> None:
    retry_settings = settings or load_retry_settings()

    @_make_retry_decorator(retry_settings)
    def _run() -> None:
        LOG.info("Running: %s", " ".join(argv))
        subprocess.run(
            argv,
            check=True,
            cwd=str(cwd) if cwd else None,
            env=env,
        )

    _run()


def install_bootstrap_deps(settings: RetrySettings | None = None) -> None:
    if not BOOTSTRAP_REQUIREMENTS.is_file():
        raise FileNotFoundError(f"Missing {BOOTSTRAP_REQUIREMENTS}")
    retry_settings = settings or load_retry_settings()
    argv = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--user",
        "-q",
        "-r",
        str(BOOTSTRAP_REQUIREMENTS),
    ]
    last_error: BaseException | None = None
    for attempt in range(1, retry_settings.max_attempts + 1):
        try:
            LOG.info("Running: %s", " ".join(argv))
            subprocess.run(argv, check=True)
            return
        except (subprocess.CalledProcessError, OSError) as exc:
            last_error = exc
            if attempt >= retry_settings.max_attempts:
                break
            delay = _wait_seconds(retry_settings, attempt)
            LOG.warning(
                "Attempt %s/%s failed (%s); retrying in %.1fs",
                attempt,
                retry_settings.max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def _resolve_hf_argv(repo_root: Path | None) -> list[str]:
    if shutil.which("hf"):
        return ["hf"]
    if shutil.which("huggingface-cli"):
        return ["huggingface-cli"]
    if repo_root and repo_root.is_dir():
        return ["poetry", "run", "hf"]
    raise RuntimeError("hf / huggingface-cli not found and repo root unavailable")


def hf_download(
    repo_id: str,
    local_dir: Path,
    *,
    repo_root: Path | None = None,
    settings: RetrySettings | None = None,
) -> None:
    local_dir = local_dir.resolve()
    sentinel = local_dir / DOWNLOAD_OK_SENTINEL
    if sentinel.is_file():
        LOG.info("Skipping %s (sentinel %s exists)", repo_id, sentinel)
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    hf_base = _resolve_hf_argv(repo_root)
    argv = [*hf_base, "download", repo_id, "--local-dir", str(local_dir)]
    run_command(argv, cwd=repo_root, env=os.environ.copy(), settings=settings)
    sentinel.write_text(f"{repo_id}\n", encoding="utf-8")
    LOG.info("Download complete: %s -> %s", repo_id, local_dir)


def hf_download_hub_cache(
    repo_id: str,
    *,
    repo_root: Path | None = None,
    settings: RetrySettings | None = None,
) -> None:
    """Download a HF repo into the default hub cache (no --local-dir)."""
    sentinel = _hub_cache_sentinel_path(repo_id)
    if sentinel.is_file():
        LOG.info("Skipping %s (hub cache sentinel %s exists)", repo_id, sentinel)
        return

    hf_base = _resolve_hf_argv(repo_root)
    argv = [*hf_base, "download", repo_id]
    run_command(argv, cwd=repo_root, env=os.environ.copy(), settings=settings)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(f"{repo_id}\n", encoding="utf-8")
    LOG.info("Hub cache download complete: %s", repo_id)


def install_poetry(settings: RetrySettings | None = None) -> None:
    retry_settings = settings or load_retry_settings()

    @_make_retry_decorator(retry_settings)
    def _install() -> None:
        LOG.info("Installing Poetry via official installer...")
        curl = subprocess.run(
            ["curl", "-sSL", "https://install.python-poetry.org"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [sys.executable, "-"],
            input=curl.stdout,
            check=True,
        )

    _install()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[videotuna-provision-retry] %(levelname)s %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(
        description="Retry wrapper for Vast bootstrap steps"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "install-bootstrap-deps", help="pip install bootstrap-requirements.txt"
    )

    run_parser = sub.add_parser("run", help="Run a command with retries")
    run_parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command after -- (e.g. run -- poetry install ...)",
    )

    hf_parser = sub.add_parser("hf-download", help="Download HF repo with retries")
    hf_parser.add_argument("repo_id", help="Hugging Face repo id (org/name)")
    hf_parser.add_argument("local_dir", type=Path, help="Destination directory")
    hf_parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="VideoTuna repo root for poetry run hf fallback",
    )

    hf_cache_parser = sub.add_parser(
        "hf-download-cache", help="Download HF repo into hub cache with retries"
    )
    hf_cache_parser.add_argument("repo_id", help="Hugging Face repo id (org/name)")
    hf_cache_parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="VideoTuna repo root for poetry run hf fallback",
    )

    sub.add_parser("install-poetry", help="Install Poetry via official installer")

    args = parser.parse_args(argv)
    settings = load_retry_settings()

    try:
        if args.command == "install-bootstrap-deps":
            install_bootstrap_deps(settings)
        elif args.command == "run":
            cmd = args.cmd
            if cmd and cmd[0] == "--":
                cmd = cmd[1:]
            if not cmd:
                parser.error("run requires a command after --")
            run_command(cmd, settings=settings)
        elif args.command == "hf-download":
            hf_download(
                args.repo_id,
                args.local_dir,
                repo_root=args.repo_root,
                settings=settings,
            )
        elif args.command == "hf-download-cache":
            hf_download_hub_cache(
                args.repo_id,
                repo_root=args.repo_root,
                settings=settings,
            )
        elif args.command == "install-poetry":
            install_poetry(settings)
    except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
        LOG.error("Failed after retries: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
