import torch

from src.moe import TernaryMoE


def test_moe_forward_shape():
    torch.manual_seed(0)
    moe = TernaryMoE(in_dim=3, hidden=16, out_dim=2, n_experts=4, top_k=2, depth=3)
    x = torch.randn(5, 3)
    out = moe(x)
    assert out.shape == (5, 2)


def test_moe_topk_one():
    """top_k=1 should still produce sensible output."""
    moe = TernaryMoE(3, 16, 1, n_experts=4, top_k=1, depth=3)
    x = torch.randn(8, 3)
    out = moe(x)
    assert out.shape == (8, 1)


def test_moe_trains():
    torch.manual_seed(0)
    moe = TernaryMoE(3, 16, 1, n_experts=2, top_k=1, depth=3)
    opt = torch.optim.AdamW(moe.parameters(), lr=1e-2)
    X = torch.randn(64, 3)
    y = torch.randn(64, 1)
    initial = sum(p.detach().sum().item() for p in moe.parameters())
    for _ in range(8):
        opt.zero_grad()
        loss = ((moe(X) - y) ** 2).mean()
        loss.backward()
        opt.step()
    final = sum(p.detach().sum().item() for p in moe.parameters())
    assert initial != final


def test_moe_routing_stats_sum_to_top_k():
    moe = TernaryMoE(3, 8, 1, n_experts=4, top_k=2)
    x = torch.randn(20, 3)
    stats = moe.routing_stats(x)
    # top-2 routing: each input picks 2 of 4 experts, so fractions should sum to 1
    total = sum(stats.values())
    assert abs(total - 1.0) < 1e-6
