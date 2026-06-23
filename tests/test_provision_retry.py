"""CPU tests for cloud/vast/provision_retry.py (no network)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLOUD_VAST = REPO_ROOT / "cloud" / "vast"

# Import from cloud/vast without installing as package
sys.path.insert(0, str(CLOUD_VAST))
import provision_retry  # noqa: E402


@pytest.fixture
def manifest_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "provisioning.yaml"
    path.write_text(
        """
version: 1
settings:
  retry:
    max_attempts: 5
    initial_delay: 2
    backoff_multiplier: 2
""",
        encoding="utf-8",
    )
    return path


def test_load_retry_settings_from_manifest(manifest_yaml: Path):
    settings = provision_retry.load_retry_settings(manifest_yaml)
    assert settings.max_attempts == 5
    assert settings.initial_delay == 2
    assert settings.backoff_multiplier == 2


def test_load_retry_settings_defaults_when_missing():
    settings = provision_retry.load_retry_settings(
        Path("/nonexistent/provisioning.yaml")
    )
    assert settings.max_attempts == 5
    assert settings.initial_delay == 2
    assert settings.backoff_multiplier == 2


def test_wait_seconds_matches_exponential_backoff():
    settings = provision_retry.RetrySettings(
        max_attempts=5, initial_delay=2, backoff_multiplier=2
    )
    assert provision_retry._wait_seconds(settings, 1) == 2
    assert provision_retry._wait_seconds(settings, 2) == 4
    assert provision_retry._wait_seconds(settings, 3) == 8


def _simple_retry_decorator(settings: provision_retry.RetrySettings):
    """Test double for tenacity when tenacity is not installed in Poetry env."""

    def decorator(fn):
        def wrapper(*args, **kwargs):
            last_exc: BaseException | None = None
            for attempt in range(1, settings.max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except (subprocess.CalledProcessError, OSError) as exc:
                    last_exc = exc
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


@patch("provision_retry.subprocess.run")
@patch("provision_retry._make_retry_decorator", side_effect=_simple_retry_decorator)
def test_run_command_retries_then_succeeds(
    _mock_decorator: MagicMock, mock_run: MagicMock
):
    mock_run.side_effect = [
        subprocess.CalledProcessError(1, ["false"]),
        subprocess.CalledProcessError(1, ["false"]),
        MagicMock(returncode=0),
    ]
    settings = provision_retry.RetrySettings(
        max_attempts=5, initial_delay=0, backoff_multiplier=2
    )
    provision_retry.run_command(["echo", "ok"], settings=settings)
    assert mock_run.call_count == 3


@patch("provision_retry.subprocess.run")
@patch("provision_retry._make_retry_decorator", side_effect=_simple_retry_decorator)
def test_run_command_exhausts_retries(_mock_decorator: MagicMock, mock_run: MagicMock):
    mock_run.side_effect = subprocess.CalledProcessError(1, ["false"])
    settings = provision_retry.RetrySettings(
        max_attempts=3, initial_delay=0, backoff_multiplier=2
    )
    with pytest.raises(subprocess.CalledProcessError):
        provision_retry.run_command(["false"], settings=settings)
    assert mock_run.call_count == 3


@patch("provision_retry.run_command")
def test_hf_download_skips_when_sentinel_exists(mock_run: MagicMock, tmp_path: Path):
    local_dir = tmp_path / "model"
    local_dir.mkdir()
    (local_dir / provision_retry.DOWNLOAD_OK_SENTINEL).write_text(
        "ok\n", encoding="utf-8"
    )
    provision_retry.hf_download("org/model", local_dir)
    mock_run.assert_not_called()


@patch("provision_retry.run_command")
def test_hf_download_invokes_hf_and_writes_sentinel(
    mock_run: MagicMock, tmp_path: Path
):
    local_dir = tmp_path / "model"
    with patch("provision_retry.shutil.which", return_value="/usr/bin/hf"):
        provision_retry.hf_download("org/model", local_dir)
    mock_run.assert_called_once()
    argv = mock_run.call_args[0][0]
    assert argv == [
        "hf",
        "download",
        "org/model",
        "--local-dir",
        str(local_dir.resolve()),
    ]
    assert (local_dir / provision_retry.DOWNLOAD_OK_SENTINEL).is_file()


@patch("provision_retry.time.sleep")
@patch("provision_retry.subprocess.run")
def test_install_bootstrap_deps_retries_without_tenacity(
    mock_run: MagicMock,
    mock_sleep: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    req = tmp_path / "bootstrap-requirements.txt"
    req.write_text("tenacity>=9.0.0\n", encoding="utf-8")
    monkeypatch.setattr(provision_retry, "BOOTSTRAP_REQUIREMENTS", req)
    mock_run.side_effect = [
        subprocess.CalledProcessError(1, ["pip"]),
        MagicMock(returncode=0),
    ]
    settings = provision_retry.RetrySettings(
        max_attempts=5, initial_delay=0, backoff_multiplier=2
    )
    provision_retry.install_bootstrap_deps(settings)
    assert mock_run.call_count == 2
    mock_sleep.assert_called_once()


@patch("provision_retry.run_command")
def test_hf_download_hub_cache_skips_when_sentinel_exists(
    mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    sentinel_dir = (
        tmp_path / ".privtune_hub_cache" / "Wan-AI--Wan2.2-T2V-A14B-Diffusers"
    )
    sentinel_dir.mkdir(parents=True)
    (sentinel_dir / provision_retry.DOWNLOAD_OK_SENTINEL).write_text(
        "ok\n", encoding="utf-8"
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    provision_retry.hf_download_hub_cache(repo_id)
    mock_run.assert_not_called()


@patch("provision_retry.run_command")
def test_hf_download_hub_cache_invokes_hf_without_local_dir(
    mock_run: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    with patch("provision_retry.shutil.which", return_value="/usr/bin/hf"):
        provision_retry.hf_download_hub_cache(repo_id)
    mock_run.assert_called_once()
    argv = mock_run.call_args[0][0]
    assert argv == ["hf", "download", repo_id]
    sentinel = provision_retry._hub_cache_sentinel_path(repo_id)
    assert sentinel.is_file()


def test_main_run_subcommand_exit_code():
    with patch("provision_retry.run_command") as mock_run:
        code = provision_retry.main(["run", "--", "echo", "hi"])
    assert code == 0
    mock_run.assert_called_once_with(
        ["echo", "hi"], settings=mock_run.call_args[1]["settings"]
    )


def test_main_run_missing_command_returns_error():
    with pytest.raises(SystemExit):
        provision_retry.main(["run"])
