"""Minimal example: train a DeltaSigmaMLP on synthetic regression data."""

import torch
import torch.nn as nn

from delta_sigma_nn import DeltaSigmaMLP


def main():
    torch.manual_seed(0)

    # Toy regression data
    X = torch.randn(2000, 3)
    y = torch.sin(X[:, 0:1] * 2) + 0.5 * X[:, 1:2] - X[:, 2:3] ** 2

    # Delta-sigma MLP at T=8 time steps
    model = DeltaSigmaMLP(in_dim=3, hidden_dim=64, out_dim=1, depth=4, T=8)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    print(f"Training DeltaSigmaMLP (T=8), {sum(p.numel() for p in model.parameters()):,} params")
    for epoch in range(100):
        model.train()
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                print(f"  epoch {epoch:3d}  train MSE = {loss_fn(model(X), y).item():.5f}")

    # Anytime inference
    print("\nAnytime inference:")
    x_test = X[:5]
    for stop_eps in [0.5, 0.1, 0.01]:
        out, k = model.anytime_inference(x_test, stop_eps=stop_eps)
        print(f"  stop_eps={stop_eps:.2f}  k used = {k}/8  out[:2] = {out[:2].squeeze().tolist()}")


if __name__ == "__main__":
    main()
