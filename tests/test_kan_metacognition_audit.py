import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from kan_metacognition_core import (
    SIGMA_5,
    TwistorKANLayer,
    MetacognitiveObserverLoop,
    ProjectiveCP3MandelbrotEncoder
)


def test_twistor_kan_layer_and_25_sigma5_invariance():
    torch.manual_seed(42)
    layer = TwistorKANLayer(in_features=6, out_features=9, grid_size=8)
    x = torch.randn(4, 6, dtype=torch.float64)

    out = layer(x)
    assert out["node_activations"].shape == (4, 9)
    assert out["edge_potentials"].shape == (4, 9, 6)
    assert out["Z5_twistors"].shape == (4, 3, 5)
    assert out["freedom_matrix_25"].shape == (4, 3, 5, 5)
    assert out["freedom_matrix_25"][0, 0].numel() == 25

    Z5 = out["Z5_twistors"]
    tr_25 = torch.einsum("...ii->...", out["freedom_matrix_25"]).real
    phi_sq = (torch.abs(Z5[..., 4]) ** 2)
    assert torch.allclose(tr_25, phi_sq, atol=1e-12)


def test_metacognitive_observer_feedback_loop_differentiable():
    torch.manual_seed(137)
    layer = TwistorKANLayer(in_features=4, out_features=6, grid_size=6)
    observer = MetacognitiveObserverLoop(layer, resolution=24, feedback_gain=0.25)

    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    res = observer.forward_with_visual_feedback(x)

    assert res["rendered_manifold"].shape == (24, 24)
    assert res["roughness_energy"] > 0.0
    assert 0.0 < float(res["damping_factor"].item()) <= 1.0

    loss = res["stabilized_nodes"].pow(2).mean() + res["roughness_energy"]
    loss.backward()

    assert layer.erf_weights.grad is not None
    assert not torch.isnan(layer.erf_weights.grad).any()
    assert x.grad is not None


def test_projective_cp3_mandelbrot_zero_trig():
    torch.manual_seed(11)
    enc = ProjectiveCP3MandelbrotEncoder(iterations=16)
    tokens = torch.randn(6, 20, dtype=torch.float64)
    out = enc.iterate_cp3_mandelbrot_orbit(tokens)
    assert out["Z5_cp3_mandelbrot"].shape == (6, 5)
    assert out["freedom_matrix_25"].shape == (6, 5, 5)
    assert not torch.isnan(out["Z5_cp3_mandelbrot"]).any()
    assert torch.allclose(torch.linalg.norm(out["Z5_cp3_mandelbrot"], dim=-1), torch.ones(6, dtype=torch.float64), atol=1e-12)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
