from __future__ import annotations

import copy

import torch

from egms_encoder.models.tile_encoder import SpatialBlock


def reference_forward(block, hidden, mask):
    normed = block.attention_norm(hidden)
    attention, _ = block.attention(normed, normed, normed, key_padding_mask=mask,
                                    need_weights=True)
    hidden = hidden + block.dropout(attention)
    return hidden + block.dropout(block.ffn(block.ffn_norm(hidden)))


def test_training_attention_preserves_outputs_and_gradients():
    torch.manual_seed(7)
    block = SpatialBlock(8, 2, 16, dropout=0.0).double().train()
    reference = copy.deepcopy(block)
    value = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    reference_value = value.detach().clone().requires_grad_(True)
    mask = torch.tensor([[False, False, False, True, True], [False] * 5])
    actual = block(value, mask)
    expected = reference_forward(reference, reference_value, mask)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(value.grad, reference_value.grad)
    for (_, actual_parameter), (_, reference_parameter) in zip(
        block.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(actual_parameter.grad, reference_parameter.grad)


def test_released_inference_path_is_unchanged():
    torch.manual_seed(9)
    block = SpatialBlock(8, 2, 16, dropout=0.1).eval()
    value = torch.randn(2, 5, 8)
    mask = torch.tensor([[False, False, False, True, True], [False] * 5])
    with torch.no_grad():
        torch.testing.assert_close(block(value, mask), reference_forward(block, value, mask),
                                   rtol=0, atol=0)
