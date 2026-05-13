"""delta_sigma_nn — multiply-free neural networks with anytime inference.

Public API:
    DeltaSigmaLinear   — linear layer with delta-sigma encoded weights
    DeltaSigmaMLP      — convenience MLP wrapper
    DSigmaCharLM       — char-level transformer LM with delta-sigma weights
    confidence_router  — per-query anytime inference with entropy/topk/diff signals
    save_dsigma_mlp    — serialize a trained MLP with packed trit streams
    load_dsigma_arrays, dsigma_inference — pure-NumPy inference engine
    encode_delta_sigma_ternary, encode_delta_sigma_order2 — modulators

Quick start:

    import torch
    from delta_sigma_nn import DeltaSigmaMLP

    model = DeltaSigmaMLP(in_dim=3, hidden_dim=128, out_dim=1, depth=5, T=8)
    # Train with normal PyTorch loops. Each forward pass uses T 1-trit matmuls.

    # At inference, you get a runtime accuracy/compute knob:
    out, k_used = model.anytime_inference(x, stop_eps=0.01)
"""

from .delta_sigma import (
    encode_delta_sigma_binary,
    encode_delta_sigma_order2,
    encode_delta_sigma_ternary,
)
from .dsigma_linear import DeltaSigmaLinear, DeltaSigmaMLP, dsigma_inference_context
from .dsigma_pack import (
    dsigma_inference,
    load_dsigma_arrays,
    pack_dsigma_layer,
    save_dsigma_mlp,
    unpack_dsigma_layer,
)
from .dsigma_router import confidence_router
from .dsigma_transformer import DSigmaCharLM
from .trit_pack import pack_trits, unpack_trits, storage_bytes

__version__ = "0.1.1"

__all__ = [
    "DeltaSigmaLinear",
    "DeltaSigmaMLP",
    "DSigmaCharLM",
    "encode_delta_sigma_ternary",
    "encode_delta_sigma_order2",
    "encode_delta_sigma_binary",
    "save_dsigma_mlp",
    "load_dsigma_arrays",
    "dsigma_inference",
    "pack_dsigma_layer",
    "unpack_dsigma_layer",
    "confidence_router",
    "dsigma_inference_context",
    "pack_trits",
    "unpack_trits",
    "storage_bytes",
    "__version__",
]
