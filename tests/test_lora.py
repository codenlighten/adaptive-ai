import torch
import torch.nn as nn

from src.lora import TernaryLoRA


def test_lora_zero_init_matches_base():
    """At init (lora_B == 0), TernaryLoRA forward = base Linear forward."""
    torch.manual_seed(0)
    base = nn.Linear(8, 4)
    lora = TernaryLoRA.from_linear(base, rank=2)
    x = torch.randn(7, 8)
    assert torch.allclose(lora(x), base(x), atol=1e-5)


def test_lora_base_weight_frozen():
    base = nn.Linear(8, 4)
    lora = TernaryLoRA.from_linear(base, rank=4)
    assert not lora.base_weight.requires_grad
    assert lora.base_bias is None or not lora.base_bias.requires_grad
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad


def test_lora_updates_only_adapter():
    torch.manual_seed(0)
    base = nn.Linear(8, 4)
    lora = TernaryLoRA.from_linear(base, rank=4)
    w_before = lora.base_weight.detach().clone()
    A_before = lora.lora_A.detach().clone()
    opt = torch.optim.SGD([p for p in lora.parameters() if p.requires_grad], lr=0.1)
    x = torch.randn(16, 8)
    target = torch.randn(16, 4)
    for _ in range(5):
        opt.zero_grad()
        ((lora(x) - target) ** 2).mean().backward()
        opt.step()
    assert torch.allclose(w_before, lora.base_weight)
    assert not torch.allclose(A_before, lora.lora_A)


def test_lora_trainable_count_correct():
    lora = TernaryLoRA(in_features=128, out_features=128, rank=8)
    assert lora.trainable_param_count() == 8 * 128 * 2
