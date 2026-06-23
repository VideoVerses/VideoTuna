"""cyclopts App for PrivTune inference and validation entrypoints."""

from __future__ import annotations

import sys
from collections.abc import Callable

from cyclopts import App

from videotuna.cli.inference_options import (
    InferencePreset,
    InferenceRunOptions,
    StandardInferenceOptions,
    inference_options_to_namespace,
    validate_preset_requirements,
)

FLUX_DOMAIN_SMOKE_CONFIG = "configs/inference/presets/flux_domain_lora_smoke.yaml"
WAN_DOMAIN_SMOKE_22_CONFIG = "configs/inference/presets/wan_domain_lora_smoke_22.yaml"
WAN_DOMAIN_I2V_SMOKE_22_CONFIG = (
    "configs/inference/presets/wan_domain_i2v_smoke_22.yaml"
)
WAN2_2_T2V_720P_CONFIG = "configs/inference/presets/balanced_wan2_2_720p.yaml"

PRESET_DOMAIN_T2I = InferencePreset(
    cli_name="inference-domain-t2i",
    config=FLUX_DOMAIN_SMOKE_CONFIG,
    enable_model_cpu_offload=True,
)
PRESET_VALIDATE_T2V = InferencePreset(
    cli_name="validate-domain-t2v",
    config=WAN_DOMAIN_SMOKE_22_CONFIG,
    enable_model_cpu_offload=True,
    require_checkpoint=True,
)
PRESET_WAN2_2_T2V_720P = InferencePreset(
    cli_name="inference-wan2.2-t2v-720p",
    config=WAN2_2_T2V_720P_CONFIG,
)
PRESET_VALIDATE_I2V = InferencePreset(
    cli_name="validate-domain-i2v",
    config=WAN_DOMAIN_I2V_SMOKE_22_CONFIG,
    enable_model_cpu_offload=True,
    require_checkpoint=True,
    require_prompt_dir=True,
)
PRESET_WAN2_2_I2V_720P = InferencePreset(
    cli_name="inference-wan2.2-i2v-720p",
    config=WAN_DOMAIN_I2V_SMOKE_22_CONFIG,
)


def _run_inference_with_options(
    run: InferenceRunOptions | None,
    standard: StandardInferenceOptions | None,
    *,
    preset: InferencePreset | None = None,
) -> None:
    from scripts.inference_new import run_inference

    run_opts = run or InferenceRunOptions()
    if preset is not None:
        validate_preset_requirements(run_opts, preset)
    args = inference_options_to_namespace(
        run=run_opts,
        standard=standard,
        preset=preset,
    )
    run_inference(args)


def _register_inference_command(
    app: App,
    preset: InferencePreset | None,
    *,
    name: str | None = None,
    is_default: bool = False,
) -> None:
    command_name = name or (preset.cli_name if preset else "run")

    def handler(
        run: InferenceRunOptions | None = None,
        *,
        standard: StandardInferenceOptions | None = None,
    ) -> None:
        _run_inference_with_options(run, standard, preset=preset)

    handler.__doc__ = (
        f"Run inference with preset {preset.config}."
        if preset is not None
        else "Run inference from a YAML config and CLI overrides."
    )
    if is_default:
        app.default(handler)
    else:
        app.command(name=command_name)(handler)


def _make_app(preset: InferencePreset | None = None, *, name: str | None = None) -> App:
    app = App(
        name=name or (preset.cli_name if preset else "privtune-inference"),
        help="PrivTune domain inference and validation.",
    )
    _register_inference_command(app, preset, is_default=True)
    return app


app = App(name="privtune-inference", help="PrivTune domain inference and validation.")
_register_inference_command(app, PRESET_DOMAIN_T2I, name="inference-domain-t2i")
_register_inference_command(app, PRESET_VALIDATE_T2V, name="validate-domain-t2v")
_register_inference_command(
    app, PRESET_WAN2_2_T2V_720P, name="inference-wan2.2-t2v-720p"
)
_register_inference_command(app, None, name="run")


def _entry_for_preset(preset: InferencePreset) -> Callable[[], None]:
    cli_app = _make_app(preset)

    def entry() -> None:
        raise SystemExit(cli_app(sys.argv[1:]))

    entry.__name__ = preset.cli_name.replace(".", "_").replace("-", "_")
    entry.__doc__ = f"Entry point for {preset.cli_name}."
    return entry


def generic_inference_entry() -> None:
    """Compat shim for ``python scripts/inference_new.py``."""
    raise SystemExit(_make_app(name="inference").__call__(sys.argv[1:]))


def main() -> None:
    """Dispatch subcommands on the shared inference App."""
    raise SystemExit(app(sys.argv[1:]))


inference_domain_t2i_entry = _entry_for_preset(PRESET_DOMAIN_T2I)
validate_domain_t2v_entry = _entry_for_preset(PRESET_VALIDATE_T2V)
inference_wan2_2_t2v_720p_entry = _entry_for_preset(PRESET_WAN2_2_T2V_720P)
inference_flux_lora_entry = inference_domain_t2i_entry
validate_domain_i2v_entry = _entry_for_preset(PRESET_VALIDATE_I2V)
inference_wan2_2_i2v_720p_entry = _entry_for_preset(PRESET_WAN2_2_I2V_720P)
