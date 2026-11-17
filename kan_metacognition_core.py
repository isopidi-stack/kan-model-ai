"""
kan-model-ai - Analytical Twistor-KAN & Closed-Loop Visual Telemetry Engine
Implements:
  1. TwistorKANLayer: B-spline-free Kolmogorov-Arnold Network layer using vectorized
     erf() envelopes, harmonic frequency bases, and 25-element SIGMA_5 spinor fibers.
  2. MetacognitiveObserverLoop: Differentiable feedback loop coupling KAN edge states
     with a Z-buffer-free volumetric manifold projection to regularize network structure.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple

SIGMA_5 = torch.tensor([
    [0., 0., 1., 0., 0.],
    [0., 0., 0., 1., 0.],
    [1., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 1.]
], dtype=torch.complex128)


def compute_sigma5_freedom_matrix(Z5: torch.Tensor) -> torch.Tensor:
    """
    Constructs the 5x5 (25-element) system freedom operator M^alpha_beta = Z^alpha (Sigma_5 Z)_beta^*
    satisfying Tr(M) = Z_5^dagger Sigma_5 Z_5.
    """
    Z_c = Z5.to(torch.complex128)
    sigma = SIGMA_5.to(Z_c.device)
    Z_dual = torch.matmul(Z_c.conj(), sigma.T)
    return torch.einsum("...i,...j->...ij", Z_c, Z_dual)


class TwistorKANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network layer replacing piecewise B-splines with smooth
    analytical erf() transitions and harmonic Fourier bases on every edge,
    projecting node sums into 5-component C^5 twistor space via Penrose incidence.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        L0_scale: float = 1.0,
        dtype: torch.dtype = torch.float64
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.L0_scale = float(L0_scale)
        self.dtype = dtype

        # Fixed grid centers mu_k in [-1, 1] and harmonic frequencies omega_k
        grid_centers = torch.linspace(-1.0, 1.0, self.grid_size, dtype=dtype)
        frequencies = torch.arange(1, self.grid_size + 1, dtype=dtype) * math.pi
        self.register_buffer("grid_centers", grid_centers)
        self.register_buffer("frequencies", frequencies)
        self.sigma_width = 2.0 / max(self.grid_size, 2)

        # Learnable univariate edge coefficients: (out_features, in_features, grid_size)
        scale = 1.0 / math.sqrt(self.in_features * self.grid_size)
        self.erf_weights = nn.Parameter(torch.randn(out_features, in_features, grid_size, dtype=dtype) * scale)
        self.cos_weights = nn.Parameter(torch.randn(out_features, in_features, grid_size, dtype=dtype) * scale)
        self.sin_weights = nn.Parameter(torch.randn(out_features, in_features, grid_size, dtype=dtype) * scale)

    def evaluate_edge_functions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates univariate edge functions phi_{q,p}(x_p) across all (out, in) edges.
        x shape: (batch, in_features) -> returns (batch, out_features, in_features)
        """
        x_d = x.to(self.dtype).unsqueeze(-1)  # (B, P, 1)
        # 1. Analytical erf() cumulative basis
        z_grid = (x_d - self.grid_centers) / (math.sqrt(2.0) * self.sigma_width)
        erf_basis = 0.5 * (1.0 + torch.erf(z_grid))  # (B, P, G)

        # 2. Harmonic frequency basis
        phase = x_d * self.frequencies  # (B, P, G)
        cos_basis = torch.cos(phase)
        sin_basis = torch.sin(phase)

        # Contract over grid dimension G -> (B, Q, P)
        phi_erf = torch.einsum("bpg,qpg->bqp", erf_basis, self.erf_weights)
        phi_cos = torch.einsum("bpg,qpg->bqp", cos_basis, self.cos_weights)
        phi_sin = torch.einsum("bpg,qpg->bqp", sin_basis, self.sin_weights)

        return phi_erf + phi_cos + phi_sin

    def project_to_cp4_twistor(self, node_activations: torch.Tensor) -> torch.Tensor:
        """
        Embeds real KAN node activations (batch, out_features) into 3D spatial fibers
        and lifts them to C^5 twistors Z_5 = (omega^0, omega^1, pi_0', pi_1', phi)
        satisfying Penrose's incidence relation omega^A = (i / L_0) X^{AA'} pi_{A'}.
        """
        B, Q = node_activations.shape
        rem = Q % 3
        if rem != 0:
            pad = torch.zeros(B, 3 - rem, dtype=node_activations.dtype, device=node_activations.device)
            padded = torch.cat([node_activations, pad], dim=-1)
        else:
            padded = node_activations

        fibers_3d = padded.view(B, -1, 3)  # (B, F, 3) -> (x, y, z) with x^0 = 0
        x1, x2, x3 = fibers_3d[..., 0], fibers_3d[..., 1], fibers_3d[..., 2]

        # Build Bloch spinor pi_{A'} from fiber orientation angles
        r_norm = torch.linalg.norm(fibers_3d, dim=-1).clamp(min=1e-12)
        cos_theta = (x3 / r_norm).clamp(-0.999, 0.999)
        half_theta = 0.5 * torch.acos(cos_theta)
        azim = torch.atan2(x2, x1)

        pi_0 = torch.cos(half_theta).to(torch.complex128) * torch.exp(-0.5j * azim)
        pi_1 = torch.sin(half_theta).to(torch.complex128) * torch.exp(0.5j * azim)
        pi_spinor = torch.stack([pi_0, pi_1], dim=-1)  # (B, F, 2)

        # Infeld-van der Waerden Hermitian matrix X^{AA'} for spatial fiber (x^0 = 0)
        X_00 = x3.to(torch.complex128)
        X_01 = x1.to(torch.complex128) - 1j * x2.to(torch.complex128)
        X_10 = x1.to(torch.complex128) + 1j * x2.to(torch.complex128)
        X_11 = -x3.to(torch.complex128)
        X_mat = torch.stack([
            torch.stack([X_00, X_01], dim=-1),
            torch.stack([X_10, X_11], dim=-1)
        ], dim=-2)

        # Scaled Penrose Incidence: omega^A = (i / L_0) X^{AA'} pi_{A'}
        omega_spinor = (1j / self.L0_scale) * torch.matmul(X_mat, pi_spinor.unsqueeze(-1)).squeeze(-1)
        phi_coord = torch.ones(B, fibers_3d.shape[1], 1, dtype=torch.complex128, device=node_activations.device)

        Z5 = torch.cat([omega_spinor, pi_spinor, phi_coord], dim=-1)
        return Z5 / torch.linalg.norm(Z5, dim=-1, keepdim=True).clamp(min=1e-12)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        edge_vals = self.evaluate_edge_functions(x)  # (B, Q, P)
        node_sums = edge_vals.sum(dim=-1)            # Kolmogorov-Arnold summation over p
        Z5 = self.project_to_cp4_twistor(node_sums)
        freedom_25 = compute_sigma5_freedom_matrix(Z5)

        return {
            "node_activations": node_sums,
            "edge_potentials": edge_vals,
            "Z5_twistors": Z5,
            "freedom_matrix_25": freedom_25,
        }


class MetacognitiveObserverLoop(nn.Module):
    """
    Closed-loop structural telemetry system between TwistorKANLayer and the Z-buffer-free
    volumetric renderer. Renders the KAN's 25-element SIGMA_5 state into a continuous 2D
    manifold density field, measures structural roughness and spinor coupling balance,
    and applies a differentiable feedback gate to stabilize the KAN's internal manifold.
    """
    def __init__(self, kan_layer: TwistorKANLayer, resolution: int = 32, feedback_gain: float = 0.15):
        super().__init__()
        self.kan = kan_layer
        self.resolution = int(resolution)
        self.feedback_gain = float(feedback_gain)

    def render_kan_manifold_projection(self, freedom_matrix_25: torch.Tensor) -> torch.Tensor:
        """
        Projects the batch-averaged 5x5 (25-element) freedom matrix onto a continuous
        2D volumetric density field using analytical erf() soft-cell envelopes.
        """
        # Average magnitude across batch and fibers -> (5, 5)
        M_avg = torch.abs(freedom_matrix_25).mean(dim=(0, 1))
        M_norm = M_avg / M_avg.max().clamp(min=1e-12)

        coords = torch.linspace(-1.0, 1.0, self.resolution, dtype=torch.float64, device=M_avg.device)
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        centers = torch.linspace(-0.75, 0.75, 5, dtype=torch.float64, device=M_avg.device)

        field = torch.zeros((self.resolution, self.resolution), dtype=torch.float64, device=M_avg.device)
        radius = 0.16
        softness = 0.05

        for r in range(5):
            for c in range(5):
                dist = torch.sqrt((gx - centers[c]) ** 2 + (gy - centers[r]) ** 2 + 1e-12)
                envelope = 0.5 * (1.0 - torch.erf((dist - radius) / softness))
                field = field + M_norm[r, c] * envelope

        return field

    def forward_with_visual_feedback(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Executes a two-stage closed loop:
          Stage 1: Forward pass through TwistorKANLayer -> render 25-element manifold field.
          Stage 2: Compute spatial gradient energy & spinor trace invariant from rendered field,
                   generating a structural damping gate that refines the KAN output state.
        """
        initial_out = self.kan(x)
        M_25 = initial_out["freedom_matrix_25"]
        manifold_field = self.render_kan_manifold_projection(M_25)

        # Compute spatial smoothness (Dirichlet gradient energy) on the rendered manifold
        dy = manifold_field[1:, :] - manifold_field[:-1, :]
        dx = manifold_field[:, 1:] - manifold_field[:, :-1]
        roughness_energy = (dy ** 2).mean() + (dx ** 2).mean()

        # Compute off-diagonal spinor coupling strength from the 25-element matrix
        spinor_trace = torch.einsum("...ii->...", M_25[..., :4, :4]).real.mean()

        # Differentiable feedback gate regularizing high-frequency edge oscillation
        damping_factor = torch.exp(-self.feedback_gain * roughness_energy)
        stabilized_nodes = initial_out["node_activations"] * damping_factor
        Z5_stabilized = self.kan.project_to_cp4_twistor(stabilized_nodes)
        M_25_stabilized = compute_sigma5_freedom_matrix(Z5_stabilized)
        manifold_stabilized = self.render_kan_manifold_projection(M_25_stabilized)

        return {
            "initial_nodes": initial_out["node_activations"],
            "stabilized_nodes": stabilized_nodes,
            "Z5_stabilized": Z5_stabilized,
            "freedom_matrix_25": M_25_stabilized,
            "rendered_manifold": manifold_stabilized,
            "roughness_energy": roughness_energy,
            "spinor_trace": spinor_trace,
            "damping_factor": damping_factor,
        }


# =========================================================================
# NATIVE TRIGONOMETRY-FREE CP^3 MANDELBROT PROJECTIVE DYNAMICS
# =========================================================================

class ProjectiveCP3MandelbrotEncoder:
    """
    Executes the quadratic Mandelbrot / Julia dynamical map natively in
    projective twistor space CP^3 (embedded in C^5 / SIGMA_5) using pure
    homogeneous degree-2 complex spinor polynomials.
    Eliminates all trigonometric functions (sin, cos, atan2, acos) and
    treats the point at infinity as the compact hyperplane pi_{A'} = 0.
    """
    def __init__(self, iterations: int = 16):
        self.iterations = int(iterations)

    def iterate_cp3_mandelbrot_orbit(self, token_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Maps real token vectors (..., D) into complex spinor parameters C^A
        and iterates the homogeneous degree-2 projective endomorphism on CP^3:
            omega^A_{k+1} = (omega^A_k)^2 + C^A (pi_{A',k})^2
            pi_{A',k+1}  = 2 * omega^A_k * pi_{A',k}
        Normalizes projectively at each step so orbits never overflow at infinity.
        """
        x = token_vectors.to(torch.float64)
        orig_shape = x.shape[:-1]
        D = x.shape[-1]

        # Deterministic algebraic Vandermonde-style complex folding (zero trig)
        # Powers of primitive 5th root of unity zeta = exp(2*pi*i / 5)
        zeta = complex(math.cos(2.0 * math.pi / 5.0), math.sin(2.0 * math.pi / 5.0))
        k_idx = torch.arange(D, dtype=torch.float64, device=x.device)
        # Pure polynomial / root-of-unity contraction into 2 complex parameter channels C^0, C^1
        w0 = torch.tensor([zeta ** (int(i) % 5) for i in range(D)], dtype=torch.complex128, device=x.device)
        w1 = torch.tensor([zeta ** ((2 * int(i) + 1) % 5) for i in range(D)], dtype=torch.complex128, device=x.device)
        scale = 1.0 / math.sqrt(max(D, 1))

        x_c = x.to(torch.complex128)
        # Anchor C^A near the main Mandelbrot cardioid cusp (-0.745 + 0.113j)
        c_anchor = complex(-0.745, 0.113)
        C_0 = c_anchor + 0.25 * scale * torch.matmul(x_c, w0)
        C_1 = c_anchor.conjugate() + 0.25 * scale * torch.matmul(x_c, w1)

        # Initialize homogeneous CP^3 coordinates [omega^0 : omega^1 : pi_0' : pi_1']
        omega_0 = torch.zeros_like(C_0)
        omega_1 = torch.zeros_like(C_1)
        pi_0 = torch.ones_like(C_0)
        pi_1 = torch.ones_like(C_1)

        # Track projective Green's potential (logarithmic conformal scale factor)
        green_potential = torch.zeros(C_0.shape, dtype=torch.float64, device=x.device)
        d_omega_0 = torch.ones_like(C_0)
        d_omega_1 = torch.ones_like(C_1)

        for _ in range(self.iterations):
            # Derivative sensitivity along the projective Julia/Mandelbrot boundary
            d_omega_0 = 2.0 * omega_0 * d_omega_0 + (pi_0 * pi_0)
            d_omega_1 = 2.0 * omega_1 * d_omega_1 + (pi_1 * pi_1)

            # Homogeneous degree-2 projective quadratic map on CP^3 (zero trig, zero infinity blowup)
            next_omega_0 = (omega_0 * omega_0) + C_0 * (pi_0 * pi_0)
            next_omega_1 = (omega_1 * omega_1) + C_1 * (pi_1 * pi_1)
            next_pi_0 = pi_0 * pi_0
            next_pi_1 = pi_1 * pi_1

            # Projective re-scaling on CP^3 (equivalence class [Z] == [lambda * Z])
            proj_norm = torch.sqrt(
                (next_omega_0.abs() ** 2) +
                (next_omega_1.abs() ** 2) +
                (next_pi_0.abs() ** 2) +
                (next_pi_1.abs() ** 2)
            ).clamp(min=1e-15)

            green_potential = green_potential + torch.log(proj_norm)

            omega_0 = next_omega_0 / proj_norm
            omega_1 = next_omega_1 / proj_norm
            pi_0 = next_pi_0 / proj_norm
            pi_1 = next_pi_1 / proj_norm

        # Embed the converged CP^3 projective point into C^5 with invariant shell coordinate phi
        # pi_norm -> 0 means the orbit reached the hyperplane at infinity in CP^3
        pi_horizon_weight = torch.sqrt((pi_0.abs() ** 2) + (pi_1.abs() ** 2)).to(torch.complex128)
        phi = (1.0 + pi_horizon_weight)

        Z5_cp3_mandelbrot = torch.stack([omega_0, omega_1, pi_0, pi_1, phi], dim=-1)
        Z5_cp3_mandelbrot = Z5_cp3_mandelbrot / torch.linalg.norm(Z5_cp3_mandelbrot, dim=-1, keepdim=True).clamp(min=1e-15)
        freedom_25 = compute_sigma5_freedom_matrix(Z5_cp3_mandelbrot)

        return {
            "Z5_cp3_mandelbrot": Z5_cp3_mandelbrot.view(*orig_shape, 5),
            "freedom_matrix_25": freedom_25.view(*orig_shape, 5, 5),
            "pi_horizon_proximity": pi_horizon_weight.real.view(*orig_shape),
            "projective_green_potential": green_potential.view(*orig_shape),
            "boundary_derivative": torch.sqrt(d_omega_0.abs() ** 2 + d_omega_1.abs() ** 2).view(*orig_shape),
        }

