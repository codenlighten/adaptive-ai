import torch

from src.hnn import (
    HamiltonianNet,
    leapfrog_step,
    make_pendulum_data,
    pendulum_hamiltonian,
    pendulum_vector_field,
)


def test_pendulum_data_shapes():
    q, p, dq, dp = make_pendulum_data(n_traj=5, steps_per_traj=8, dt=0.1)
    assert q.shape == (40, 1)
    assert dq.shape == (40, 1)


def test_leapfrog_conserves_energy_on_true_H():
    q = torch.tensor([[1.0]])
    p = torch.tensor([[0.0]])
    H0 = pendulum_hamiltonian(q, p).item()
    for _ in range(200):
        q, p = leapfrog_step(q, p, pendulum_vector_field, 0.05)
    H_end = pendulum_hamiltonian(q, p).item()
    # Leapfrog on the true H must conserve energy to high precision.
    assert abs(H_end - H0) / abs(H0) < 1e-3


def test_hnn_vector_field_is_differentiable():
    model = HamiltonianNet(dim=1, hidden=16, depth=3, ternary=False)
    q = torch.randn(4, 1, requires_grad=True)
    p = torch.randn(4, 1, requires_grad=True)
    dq, dp = model(q, p)
    assert dq.shape == (4, 1)
    assert dp.shape == (4, 1)
    loss = (dq.sum() + dp.sum())
    # Need backward through the model parameters, not q/p (which are intermediate)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_ternary_hnn_trains_one_step():
    torch.manual_seed(0)
    model = HamiltonianNet(dim=1, hidden=16, depth=3, ternary=True)
    q, p, dq_t, dp_t = make_pendulum_data(n_traj=4, steps_per_traj=4, dt=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial = sum(x.detach().clone().sum().item() for x in model.parameters())
    for _ in range(3):
        opt.zero_grad()
        dq_p, dp_p = model(q, p)
        loss = ((dq_p - dq_t) ** 2).mean() + ((dp_p - dp_t) ** 2).mean()
        loss.backward()
        opt.step()
    final = sum(x.detach().sum().item() for x in model.parameters())
    assert initial != final
