# KAN Model AI (`kan-model-ai`)

An analytical Kolmogorov-Arnold Network (KAN) and closed-loop visual telemetry suite built on projective $\mathbb{CP}^4$ spinor fibers, $\mathrm{erf}()$ radiative basis functions, and the 25-element $\Sigma_5$ freedom matrix.

---

## Core Architecture (Two Integrated Solutions)

### 1. Analytical Twistor-KAN Layer (`TwistorKANLayer`)
Replaces computationally expensive piecewise B-spline recursion (de Boor's algorithm) with a fully vectorized combination of analytical error-function ($\mathrm{erf}$) radial envelopes and harmonic Fourier frequencies along every network edge:

$$\phi_{q,p}(x_p) = \sum_{k=1}^{G} c_{q,p,k}\,\frac{1}{2}\left[1 + \mathrm{erf}\left(\frac{x_p - \mu_k}{\sqrt{2}\sigma_k}\right)\right] + a_{q,p,k}\cos(\omega_k x_p) + b_{q,p,k}\sin(\omega_k x_p)$$

Each transformed node state is embedded into $\mathbb{C}^5$ via scaled Penrose incidence $\omega^A = \frac{i}{L_0} X^{AA'} \pi_{A'}$ and evaluated across all 25 elements of the indefinite Hermitian metric $\Sigma_5$.

### 2. Closed-Loop Visual Telemetry & Self-Regularization (`MetacognitiveObserverLoop`)
Forms a differentiable feedback loop between `kan-model-ai`, `ai-vector-shield`, and `twistor-render-core`:
1. **Forward Edge Pass:** Computes univariate edge activations and builds the $5 \times 5$ (`25-element`) freedom matrix $M^\alpha{}_\beta = Z^\alpha (\Sigma_5 Z)_\beta^*$.
2. **Z-Buffer-Free Manifold Projection:** Renders the active 25-element KAN state into an analytical 2D volumetric density field using continuous $\mathrm{erf}()$ transmittance.
3. **Structural Telemetry Feedback:** Computes spatial gradient energy, spectral entropy, and $\Sigma_5$ trace defect from the rendered manifold, feeding a corrective modulation tensor back into the KAN edge weights to damp spline oscillation and stabilize training.

---

## Companion Repositories
* **Volumetric & Spinor Ray-Optics:** [twistor-render-core](https://github.com/isopidi-stack/twistor-render-core)
* **Krein-Space Holonomy Shield:** [ai-vector-shield](https://github.com/isopidi-stack/ai-vector-shield)
* **Theoretical Spinor Sandbox:** [penrose-frontier-lab](https://github.com/isopidi-stack/penrose-frontier-lab)

---

## Run Standalone Audit Suite

    python tests\test_kan_metacognition_audit.py
