"""CPU tests for cloud/vast provisioning scripts and configs (no GPU)."""

import os
import re
from pathlib import Path

import yaml
from omegaconf import OmegaConf

from videotuna.training.flux_lora.config import load_train_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CLOUD_VAST = REPO_ROOT / "cloud" / "vast"

EXECUTABLE_SCRIPTS = [
    CLOUD_VAST / "bootstrap.sh",
    CLOUD_VAST / "run-train.sh",
    CLOUD_VAST / "run-smoke-train.sh",
]

PROVISIONING_FILES = [
    CLOUD_VAST / "provisioning.yaml",
    CLOUD_VAST / "bootstrap.sh",
    CLOUD_VAST / "run-train.sh",
    CLOUD_VAST / "run-smoke-train.sh",
]

REQUIRED_POETRY_COMMANDS = [
    "poetry install",
    "train-flux-lora",
    "train-wan2-1-t2v-lora",
    "install-deepspeed",
    "test tests/test_import_smoke.py",
]

SECRET_PATTERNS = [
    re.compile(r"hf_[a-zA-Z0-9]{20,}"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
]

FLUX_CLOUD_SMOKE = REPO_ROOT / "configs" / "domain" / "flux_t2i_cloud_smoke.json"
WAN_CLOUD_SMOKE = REPO_ROOT / "configs" / "domain" / "wan_t2v_lora_cloud_smoke.yaml"
FLUX_CLOUD_SMOKE_LEGACY = REPO_ROOT / "configs" / "006_flux" / "cloud_smoke.json"
WAN_CLOUD_SMOKE_LEGACY = (
    REPO_ROOT / "configs" / "008_wanvideo" / "wan2_1_t2v_14B_lora_cloud_smoke.yaml"
)
FLUX_DATA_CONFIG = REPO_ROOT / "configs" / "domain" / "flux_t2i_data.json"


def test_cloud_scripts_exist_and_are_executable():
    for path in EXECUTABLE_SCRIPTS:
        assert path.is_file(), f"missing {path}"
        assert os.access(path, os.X_OK), f"not executable: {path}"


def test_provisioning_references_valid_poetry_commands():
    combined = ""
    for path in PROVISIONING_FILES:
        combined += path.read_text(encoding="utf-8") + "\n"
    for cmd in REQUIRED_POETRY_COMMANDS:
        assert cmd in combined, f"expected poetry command not found: {cmd}"


def test_no_hardcoded_secrets_in_cloud_vast():
    for path in CLOUD_VAST.rglob("*"):
        if not path.is_file():
            continue
        if path.name.endswith(".example"):
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in SECRET_PATTERNS:
            match = pattern.search(text)
            assert match is None, (
                f"possible secret in {path.relative_to(REPO_ROOT)}: "
                f"{match.group()[:12]}..."
            )


def test_provisioning_yaml_structure():
    prov_path = CLOUD_VAST / "provisioning.yaml"
    data = yaml.safe_load(prov_path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert "git_repos" in data
    assert any(
        "PrivTune" in r.get("dest", "") or "VideoTuna" in r.get("dest", "")
        for r in data["git_repos"]
    )
    assert "post_commands" in data
    assert any("bootstrap.sh" in c for c in data["post_commands"])


def test_flux_cloud_smoke_config_loads():
    train_cfg, data_cfg = load_train_config(FLUX_CLOUD_SMOKE, FLUX_DATA_CONFIG)
    assert train_cfg.max_train_steps == 50
    assert train_cfg.checkpointing_steps == 25
    assert train_cfg.output_dir == "results/train/flux-cloud-smoke"
    assert train_cfg.pretrained_model_name_or_path == "checkpoints/flux/FLUX.1-dev"
    assert data_cfg.instance_data_dir == "data/t2i/domain"


def test_wan_cloud_smoke_yaml_parses():
    cfg = OmegaConf.load(WAN_CLOUD_SMOKE)
    assert cfg.train.name == "train_wan_cloud_smoke"
    assert cfg.train.lightning.trainer.max_epochs == 1
    ckpt_cb = cfg.train.lightning.callbacks.model_checkpoint.params
    assert ckpt_cb.every_n_train_steps == 5


def test_legacy_cloud_smoke_symlinks_resolve():
    assert FLUX_CLOUD_SMOKE_LEGACY.is_file()
    assert FLUX_CLOUD_SMOKE_LEGACY.resolve() == FLUX_CLOUD_SMOKE.resolve()
    assert WAN_CLOUD_SMOKE_LEGACY.is_file()
    assert WAN_CLOUD_SMOKE_LEGACY.resolve() == WAN_CLOUD_SMOKE.resolve()


def test_env_cloud_example_exists():
    example = CLOUD_VAST / ".env.cloud.example"
    assert example.is_file()
    text = example.read_text(encoding="utf-8")
    assert "VIDEOTUNA_COMPUTE_BACKEND=cuda" in text
    assert "TRAIN_PROFILE=" in text
    assert "HF_TOKEN=" in text


def test_supervisor_config_exists():
    conf = CLOUD_VAST / "supervisor" / "videotuna-train.conf"
    assert conf.is_file()
    text = conf.read_text(encoding="utf-8")
    assert "videotuna-train" in text
    assert "run-train.sh" in text


def test_env_cloud_example_documents_fast_hf_download():
    example = CLOUD_VAST / ".env.cloud.example"
    text = example.read_text(encoding="utf-8")
    assert "VIDEOTUNA_FAST_HF_DOWNLOAD" in text
    assert "HF_XET_HIGH_PERFORMANCE" in text


def test_bootstrap_enables_hf_xet_high_performance():
    bootstrap = CLOUD_VAST / "bootstrap.sh"
    text = bootstrap.read_text(encoding="utf-8")
    assert "VIDEOTUNA_FAST_HF_DOWNLOAD" in text
    assert "HF_XET_HIGH_PERFORMANCE" in text
    assert "enable_fast_hf_download" in text
    assert "provision_retry.py" in text


def test_bootstrap_retry_artifacts_exist():
    assert (CLOUD_VAST / "provision_retry.py").is_file()
    assert (CLOUD_VAST / "bootstrap-requirements.txt").is_file()
    reqs = (CLOUD_VAST / "bootstrap-requirements.txt").read_text(encoding="utf-8")
    assert "tenacity" in reqs
    assert "pyyaml" in reqs


def test_provisioning_includes_wan_train_and_validate_hub_ids():
    combined = ""
    for path in PROVISIONING_FILES:
        combined += path.read_text(encoding="utf-8") + "\n"
    for repo in (
        "Wan-AI/Wan2.1-T2V-14B",
        "Wan-AI/Wan2.1-I2V-14B-480P",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    ):
        assert repo in combined, f"expected hub id in cloud bundle: {repo}"


def test_provisioning_yaml_documents_retry_layers():
    prov_path = CLOUD_VAST / "provisioning.yaml"
    text = prov_path.read_text(encoding="utf-8")
    assert "settings:" in text
    assert "on_failure:" in text
    assert "provision_retry.py" in text
    data = yaml.safe_load(text)
    assert data["settings"]["retry"]["max_attempts"] == 5
    assert data["on_failure"]["action"] == "continue"
