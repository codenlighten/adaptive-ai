import torch

from delta_sigma_nn.dsigma_linear import DeltaSigmaLinear, DeltaSigmaMLP


def test_dsigma_linear_forward():
    layer = DeltaSigmaLinear(8, 4, T=16)
    x = torch.randn(7, 8)
    y = layer(x)
    assert y.shape == (7, 4)


def test_dsigma_linear_trains():
    torch.manual_seed(0)
    layer = DeltaSigmaLinear(8, 4, T=8)
    opt = torch.optim.SGD(layer.parameters(), lr=0.05)
    w0 = layer.weight.detach().clone()
    X = torch.randn(32, 8)
    target = torch.randn(32, 4)
    for _ in range(5):
        opt.zero_grad()
        loss = ((layer(X) - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert not torch.allclose(w0, layer.weight)


def test_larger_T_more_accurate():
    """At larger T, the dsigma weight should better approximate the float weight."""
    torch.manual_seed(0)
    layer4 = DeltaSigmaLinear(32, 32, T=4)
    layer64 = DeltaSigmaLinear(32, 32, T=64)
    # use the same underlying weights
    layer64.weight.data.copy_(layer4.weight.data)
    layer64.norm.load_state_dict(layer4.norm.state_dict())
    layer64.bias.data.copy_(layer4.bias.data)

    x = torch.randn(10, 32)
    layer4.eval(); layer64.eval()
    # The fp32 reference computation
    ref = layer4.norm(x) @ layer4.weight.T + layer4.bias
    y4 = layer4(x); y64 = layer64(x)
    err4 = (y4 - ref).abs().mean().item()
    err64 = (y64 - ref).abs().mean().item()
    assert err64 < err4, f"T=64 err {err64} not < T=4 err {err4}"


def test_anytime_inference_stabilizes():
    """Anytime inference should converge to the same value as full forward."""
    torch.manual_seed(0)
    model = DeltaSigmaMLP(3, 32, 1, depth=4, T=32)
    x = torch.randn(8, 3)
    full = model(x)
    early, k = model.anytime_inference(x, T_max=32, stop_eps=1e-2)
    # Early-exit value should be close to the full-T forward
    assert (full - early).abs().max().item() < 0.5
