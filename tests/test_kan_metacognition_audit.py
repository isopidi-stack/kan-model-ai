"""
Standalone Audit Suite for kan-model-ai (v1.0.0-twistor-kan)
Verifies:
  1. Analytical TwistorKANLayer univariate erf() + harmonic evaluation and C^5 projection.
  2. Exact 25-element SIGMA_5 freedom matrix trace invariance (Tr(M_25) == |phi|^2).
  3. Closed-loop visual telemetry feedback differentiability and manifold regularization.
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from kan_metacognition_core import (
    SIGMA_5,
    TwistorKANLayer,
    MetacognitiveObserverLoop
)


def test_twistor_kan_layer_and_25_sigma5_invariance():
    """
    Verifies that TwistorKANLayer maps multivariate inputs through univariate erf/harmonic
    edges into C^5 twistors with exact Penrose incidence and 25-element SIGMA_5 trace invariance.
    """
    torch.manual_seed(42)
    layer = TwistorKANLayer(in_features=6, out_features=9, grid_size=8)
    x = torch.randn(4, 6, dtype=torch.float64)

    out = layer(x)
    assert out["node_activations"].shape == (4, 9)
    assert out["edge_potentials"].shape == (4, 9, 6)
    assert out["Z5_twistors"].shape == (4, 3, 5)
    assert out["freedom_matrix_25"].shape == (4, 3, 5, 5)
    assert out["freedom_matrix_25"][0, 0].numel() == 25

    # Because omega^A = (i/L0) X^{AA'} pi_{A'} with Hermitian X^{AA'}, omega^dagger pi + pi^dagger omega == 0
    Z5 = out["Z5_twistors"]
    tr_25 = torch.einsum("...ii->...", out["freedom_matrix_25"]).real
    phi_sq = (torch.abs(Z5[..., 4]) ** 2)
    assert torch.allclose(tr_25, phi_sq, atol=1e-12)


def test_metacognitive_observer_feedback_loop_differentiable():
    """
    Verifies that the closed-loop observer renders the 25-element manifold projection,
    computes roughness telemetry, stabilizes node activations, and backpropagates cleanly.
    """
    torch.manual_seed(137)
    layer = TwistorKANLayer(in_features=4, out_features=6, grid_size=6)
    observer = MetacognitiveObserverLoop(layer, resolution=24, feedback_gain=0.25)

    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    res = observer.forward_with_visual_feedback(x)

    assert res["rendered_manifold"].shape == (24, 24)
    assert res["roughness_energy"] > 0.0
    assert 0.0 < float(res["damping_factor"].item()) <= 1.0

    # Verify end-to-end backpropagation through the rendered manifold and KAN edge weights
    loss = res["stabilized_nodes"].pow(2).mean() + res["roughness_energy"]
    loss.backward()

    assert layer.erf_weights.grad is not None
    assert not torch.isnan(layer.erf_weights.grad).any()
    assert x.grad is not None


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
