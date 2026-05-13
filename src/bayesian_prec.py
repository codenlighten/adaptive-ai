"""Bayesian mixed-precision: Dirichlet prior over {Binary, Ternary, Quaternary} per layer.

Each hidden layer has a Dirichlet-distributed mixture weight pi ~ Dir(alpha).
At each forward pass we sample pi and use it as the soft mixture weights.
The model is trained by MAP / SVI on a variational posterior over alpha.

We use a simple variational scheme: maintain a posterior Dir(alpha + n)
where n[k] is the running count of how often level k was best for this
layer, and treat alpha as a trainable parameter (concentration).

At inference: sample pi once per forward pass; ensemble multiple samples
to reduce variance and read off uncertainty over precisions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .learned_mixed import _ste_binarize
from .quaternary import ste_quaternize
from .ternary import ste_ternarize


_LEVELS = ["Binary", "Ternary", "Quaternary"]


class BayesianPrecLinear(nn.Module):
    """Dirichlet posterior over weight-precision mixture.

    alpha shape (3,): variational concentration parameters. At forward we
    sample pi = Dirichlet(alpha).rsample() (reparameterized) and combine
    the three quantized weights by pi.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 alpha_init: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.norm = nn.LayerNorm(in_features)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        # alpha must stay positive; parameterize via softplus.
        self.alpha_raw = nn.Parameter(torch.full((3,), float(alpha_init)).log())

    def alpha(self) -> torch.Tensor:
        return self.alpha_raw.exp().clamp(0.1, 20.0)

    def sample_pi(self) -> torch.Tensor:
        """Reparameterized Dirichlet sample via the Beta-Stick or Gamma trick."""
        a = self.alpha()
        gammas = torch._standard_gamma(a)
        return gammas / gammas.sum()

    def expected_pi(self) -> torch.Tensor:
        a = self.alpha()
        return a / a.sum()

    def forward(self, x):
        x = self.norm(x)
        if self.training:
            pi = self.sample_pi()
        else:
            pi = self.expected_pi()
        w_bin  = _ste_binarize(self.weight)
        w_ter  = ste_ternarize(self.weight)
        w_quat = ste_quaternize(self.weight)
        w = pi[0] * w_bin + pi[1] * w_ter + pi[2] * w_quat
        return F.linear(x, w, self.bias)

    def alpha_dict(self) -> dict[str, float]:
        a = self.alpha().detach().tolist()
        return dict(zip(_LEVELS, a))


class BayesianPrecMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth=5):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 2):
            layers += [BayesianPrecLinear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def alphas(self):
        return [m.alpha_dict() for m in self.net if isinstance(m, BayesianPrecLinear)]

    @torch.no_grad()
    def ensemble_predict(self, x, n_samples=16):
        """Average n samples from the posterior. Activates train mode briefly
        to draw Dirichlet samples."""
        was_training = self.training
        self.train()
        acc = torch.zeros_like(self.forward(x))
        for _ in range(n_samples):
            acc = acc + self.forward(x)
        self.train(was_training)
        return acc / n_samples
