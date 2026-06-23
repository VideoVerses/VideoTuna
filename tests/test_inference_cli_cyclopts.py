"""Tests for cyclopts inference CLI groups and Poetry entrypoints."""

from __future__ import annotations

import subprocess
import sys

import pytest

from videotuna.cli.inference_app import (
    PRESET_DOMAIN_T2I,
    PRESET_VALIDATE_T2V,
    PRESET_WAN2_2_T2V_720P,
    app,
)
from videotuna.cli.inference_options import (
    InferenceRunOptions,
    StandardInferenceOptions,
    inference_options_to_namespace,
    validate_preset_requirements,
)


def _help_text(command: list[str]) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ("from videotuna.cli.inference_app import app; " f"app({command!r})"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


@pytest.mark.parametrize(
    "command",
    [
        ["inference-domain-t2i", "--help"],
        ["validate-domain-t2v", "--help"],
        ["inference-wan2.2-t2v-720p", "--help"],
    ],
)
def test_inference_help_lists_shared_flags(command: list[str]) -> None:
    help_text = _help_text(command)
    for flag in (
        "--lorackpt",
        "--memory-preset",
        "--enable_vae_tiling",
        "--num-inference-steps",
    ):
        assert flag in help_text


@pytest.mark.parametrize(
    "command",
    [
        ["inference-domain-t2i", "--help"],
        ["validate-domain-t2v", "--help"],
    ],
)
def test_inference_help_omits_legacy_hunyuan_fp8(command: list[str]) -> None:
    help_text = _help_text(command).lower()
    for forbidden in ("enable_fp8", "enable-fp8", "dit-weight", "hunyuan"):
        assert forbidden not in help_text


def test_flag_parity_across_presets() -> None:
    run = InferenceRunOptions(lorackpt="/tmp/lora", num_inference_steps=8)
    standard = StandardInferenceOptions(memory_preset="balanced", device="cuda:0")
    t2i = vars(
        inference_options_to_namespace(
            run=run,
            standard=standard,
            preset=PRESET_DOMAIN_T2I,
        )
    )
    wan = vars(
        inference_options_to_namespace(
            run=run,
            standard=standard,
            preset=PRESET_WAN2_2_T2V_720P,
        )
    )

    shared_keys = {k for k in t2i if k not in {"config", "enable_model_cpu_offload"}}
    for key in shared_keys:
        assert t2i[key] == wan[key], key

    assert t2i["config"] == PRESET_DOMAIN_T2I.config
    assert wan["config"] == PRESET_WAN2_2_T2V_720P.config
    assert t2i["enable_model_cpu_offload"] is True
    assert wan["enable_model_cpu_offload"] is False


def test_domain_t2i_preset_applies_without_user_config() -> None:
    args = inference_options_to_namespace(preset=PRESET_DOMAIN_T2I)
    assert args.config == PRESET_DOMAIN_T2I.config
    assert args.enable_model_cpu_offload is True


def test_validate_domain_t2v_requires_checkpoint() -> None:
    with pytest.raises(SystemExit) as exc:
        validate_preset_requirements(InferenceRunOptions(), PRESET_VALIDATE_T2V)
    assert exc.value.code == 2


def test_cyclopts_parses_standard_flags() -> None:
    captured: dict[str, object] = {}

    def handler(
        run: InferenceRunOptions | None = None,
        *,
        standard: StandardInferenceOptions | None = None,
    ) -> None:
        captured["args"] = inference_options_to_namespace(run=run, standard=standard)

    probe = app.__class__(name="probe")
    probe.command(name="probe")(handler)
    probe(
        [
            "probe",
            "--lorackpt",
            "/tmp/lora",
            "--memory-preset",
            "low_vram",
            "--enable_vae_tiling",
        ]
    )
    args = captured["args"]
    assert isinstance(args, object)
    assert getattr(args, "lorackpt") == "/tmp/lora"
    assert getattr(args, "memory_preset") == "low_vram"
    assert getattr(args, "enable_vae_tiling") is True
