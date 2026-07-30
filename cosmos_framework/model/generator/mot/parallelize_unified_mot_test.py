# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
from cosmos_framework.model.generator.mot.parallelize_unified_mot import (
    _SAC_EXACT_SAVE_OP_NAMES,
    _create_selective_checkpoint_policy,
    _get_default_sac_save_ops,
    apply_ac,
)


class _ToyTransformer(nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8)) for _ in range(num_layers)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            inputs = layer(inputs)
        return inputs


class _NamedOp:
    """Hashable stand-in for an optional op that is not registered on CPU."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


@pytest.mark.parametrize(
    ("mode", "expect_wrapped"),
    [("none", False), ("full", True), ("selective", True)],
)
def test_apply_ac_wraps_full_and_selective_blocks(mode: str, expect_wrapped: bool) -> None:
    model = _ToyTransformer()

    apply_ac(model, ActivationCheckpointingConfig(mode=mode))

    assert all(isinstance(layer, CheckpointWrapper) is expect_wrapped for layer in model.model.layers)
    inputs = torch.randn(2, 4, 8, requires_grad=True)
    model(inputs).sum().backward()
    assert inputs.grad is not None


def test_default_sac_saves_compute_intensive_and_flex_ops() -> None:
    save_ops = _get_default_sac_save_ops()

    assert torch.ops.aten.mm.default in save_ops
    assert torch.ops.aten.bmm.default in save_ops
    assert torch.ops.aten.addmm.default in save_ops
    assert torch.ops.aten._scaled_dot_product_cudnn_attention.default in save_ops
    assert torch.ops.aten._flash_attention_forward.default in save_ops
    assert torch._higher_order_ops.flex_attention in save_ops


def test_sac_recomputes_every_second_mm_with_independent_phase_counters() -> None:
    policy = _create_selective_checkpoint_policy((), {torch.ops.aten.mm.default})
    forward_context = SimpleNamespace(is_recompute=False)
    recompute_context = SimpleNamespace(is_recompute=True)
    mm = torch.ops.aten.mm.default

    assert policy(forward_context, mm) is CheckpointPolicy.MUST_SAVE
    assert policy(forward_context, mm) is CheckpointPolicy.PREFER_RECOMPUTE
    assert policy(forward_context, mm) is CheckpointPolicy.MUST_SAVE
    assert policy(recompute_context, mm) is CheckpointPolicy.MUST_SAVE
    assert policy(recompute_context, mm) is CheckpointPolicy.PREFER_RECOMPUTE


def test_sac_mm_sequence_matches_legacy_and_forward_only_checkpoint_apis() -> None:
    """Exercise the runtime contract used by PyTorch <=2.10 and 2.13+."""
    base_policy = _create_selective_checkpoint_policy((), {torch.ops.aten.mm.default})
    mm_decisions_by_phase: dict[str, list[CheckpointPolicy]] = {
        "forward": [],
        "recompute": [],
    }

    def recording_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
        decision = base_policy(ctx, func, *args, **kwargs)
        if func == torch.ops.aten.mm.default:
            phase = "recompute" if getattr(ctx, "is_recompute", False) else "forward"
            mm_decisions_by_phase[phase].append(decision)
        return decision

    def three_matmuls(
        inputs: torch.Tensor,
        weight_1: torch.Tensor,
        weight_2: torch.Tensor,
        weight_3: torch.Tensor,
    ) -> torch.Tensor:
        return ((inputs @ weight_1) @ weight_2) @ weight_3

    tensors = [torch.randn(8, 8, requires_grad=True) for _ in range(4)]
    output = checkpoint(
        three_matmuls,
        *tensors,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(recording_policy),
        early_stop=False,
    )
    output.sum().backward()

    expected = [
        CheckpointPolicy.MUST_SAVE,
        CheckpointPolicy.PREFER_RECOMPUTE,
        CheckpointPolicy.MUST_SAVE,
    ]
    assert mm_decisions_by_phase["forward"] == expected
    # PyTorch <=2.10 invokes the policy again during recompute; PyTorch 2.13+
    # replays the forward decisions by cache index and never invokes it there.
    assert mm_decisions_by_phase["recompute"] in ([], expected)


@pytest.mark.parametrize(
    "op_name",
    [
        "flash_attn_3._flash_attn_forward.default",
        "cosmos3.cudnn_fused_attn.default",
        "natten.hopper_fmha_forward.default",
        "aten._scaled_dot_product_cudnn_attention.default",
    ],
)
def test_optional_attention_backends_use_exact_default_matches(op_name: str) -> None:
    assert op_name in _SAC_EXACT_SAVE_OP_NAMES
    policy = _create_selective_checkpoint_policy((), set())

    assert policy(SimpleNamespace(is_recompute=False), _NamedOp(op_name)) is CheckpointPolicy.MUST_SAVE
    assert (
        policy(SimpleNamespace(is_recompute=False), _NamedOp(f"{op_name}.unrelated"))
        is CheckpointPolicy.PREFER_RECOMPUTE
    )


def test_user_regex_is_an_additive_namespace_qualified_extension() -> None:
    policy = _create_selective_checkpoint_policy(
        (re.compile(r"^vendor_backend\.custom_attention\.default$"),),
        set(),
    )
    context = SimpleNamespace(is_recompute=False)

    assert policy(context, _NamedOp("vendor_backend.custom_attention.default")) is CheckpointPolicy.MUST_SAVE
    assert (
        policy(context, _NamedOp("vendor_backend.custom_attention.default.extra")) is CheckpointPolicy.PREFER_RECOMPUTE
    )
    assert policy(context, _NamedOp("aten.sin.default")) is CheckpointPolicy.PREFER_RECOMPUTE


@pytest.mark.manual
@pytest.mark.level(1)
@pytest.mark.gpus(1)
def test_flash3_cuda_op_is_saved_by_default_sac() -> None:
    """Optional real-kernel check: ``pytest --manual --levels=1 --num-gpus=1``."""
    from cosmos_framework.model.attention import attention
    from cosmos_framework.model.attention.flash3 import FLASH3_SUPPORTED

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Flash3 CUDA SAC test requires a Hopper (SM90) GPU.")
    if not FLASH3_SUPPORTED:
        pytest.skip("Flash3 is not installed in this environment.")

    save_ops = _get_default_sac_save_ops()
    base_policy = _create_selective_checkpoint_policy((), save_ops)
    decisions: dict[str, list[CheckpointPolicy]] = {}

    def recording_policy(ctx, func, *args, **kwargs) -> CheckpointPolicy:
        decision = base_policy(ctx, func, *args, **kwargs)
        decisions.setdefault(str(func), []).append(decision)
        return decision

    def flash3_forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        output = attention(query, key, value, backend="flash3")
        assert isinstance(output, torch.Tensor)
        return output

    query, key, value = (
        torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)
    )
    output = checkpoint(
        flash3_forward,
        query,
        key,
        value,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(recording_policy),
        early_stop=False,
    )
    output.float().sum().backward()

    flash3_op = "flash_attn_3._flash_attn_forward.default"
    assert flash3_op in decisions
    assert all(decision is CheckpointPolicy.MUST_SAVE for decision in decisions[flash3_op])
