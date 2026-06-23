"""Tests for PEFT LoRA helpers."""

import pytest
import torch.nn as nn

from videotuna.utils.lora_utils import (
    parameter_matches_lora_target,
    resolve_lora_target_modules,
)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


def test_resolve_all_linear():
    model = _TinyModel()
    assert resolve_lora_target_modules(model, "all-linear") == "all-linear"


def test_resolve_explicit_list():
    model = _TinyModel()
    targets = resolve_lora_target_modules(model, ["linear"])
    assert targets == ["linear"]


@pytest.mark.parametrize(
    ("param_name", "targets", "expected"),
    [
        ("blocks.0.self_attn.q.weight", ["q"], True),
        ("blocks.0.self_attn.to_q.weight", ["q"], False),
        ("blocks.0.ffn.0.weight", ["ffn.0"], True),
        ("blocks.0.ffn.00.weight", ["ffn.0"], False),
        ("blocks.0.unique.weight", ["q"], False),
    ],
)
def test_parameter_matches_lora_target_edge_cases(param_name, targets, expected):
    assert parameter_matches_lora_target(param_name, targets) is expected
