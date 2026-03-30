# quantkit

A Python library for quantitative finance: stochastic processes, Monte Carlo
simulation, and real-time SDL2 visualisation.

⚠️ This library is currently under development, and expects active future updates

## Installation

```bash
pip install quantkit
```

With the optional SDL2 visualiser:

```bash
pip install "quantkit[visualize]"
```

## Quick start

```python
from quantkit import GBM, CEV
from quantkit.visualize import Renderer, PathPlot

# Simulate 1 000 GBM paths
gbm = GBM(mu=0.05, sigma=0.2, S0=100.0, seed=42)
paths = gbm.simulate(x_start=0, x_end=1, n_paths=1_000, n_points=500)

# Price a European call
price = gbm.compute_option_price(
    paths, K=105.0, position="call", type="european",
    expiry=365, r=0.05, q=0.0,
)
print(f"Call price: {price:.4f}")

# Visualise
renderer = Renderer(title="GBM", width=1280, height=720)
plot = PathPlot(renderer)
plot.show(paths, x_start=0, x_end=1, alpha=40)
```

## Available processes

| Class | Description | Stochastic equation |
|---|---|---|
| `StandardBM` | Standard Brownian motion (µ=0, σ=1) | $\mathrm{d}S_t = \mathrm{d}W_t$ |
| `BrownianMotion` | Arithmetic BM with drift and volatility | $\mathrm{d}S_t = \mu\mathrm{d}t + \sigma\mathrm{d}W_t$ |
| `GBM` | Geometric Brownian Motion | $\mathrm{d}S_t = \mu S_t\mathrm{d}t + \sigma S_t\mathrm{d}W_t$ |
| `RandomWalk` | Discrete random walk | $\Delta S_t = \mu\Delta t\varepsilon_t,\ \varepsilon_t = \pm 1$ |
| `OrnsteinUhlenbeck` | Mean-reverting OU process | $\mathrm{d}S_t = \theta(\mu - S_t)\mathrm{d}t + \sigma\mathrm{d}W_t$ |
| `PoissonProcess` | Homogeneous Poisson process | $\mathrm{d}N_t \sim \mathrm{Poisson}(\lambda\mathrm{d}t)$ |
| `CompoundPoissonProcess` | Compound Poisson with custom jump distribution | $\mathrm{d}S_t = J_t\mathrm{d}N_t$ |
| `LevyFlight` | Lévy flight (Cauchy-distributed steps) | $\Delta S_t = c(\Delta t)^{1/\alpha}\Xi_t,\ \Xi_t \sim \mathrm{Cauchy}(0,1)$ |
| `FractionalBrownianMotion` | fBM with Hurst parameter H | $S_t = B_t^H,\ \text{Cov}(B_t^H,B_s^H)=\tfrac12\left(t^{2H}+s^{2H}-(t-s)^{2H}\right)$ |
| `StableProcess` | α-stable process | $\Delta S_t = c(\Delta t)^{1/\alpha}L_t + \ell\Delta t$ |
| `HestonModel` | Stochastic volatility (Heston) | $\mathrm{d}S_t = \sqrt{v_t}S_t\mathrm{d}W_1,\quad \mathrm{d}v_t = \theta(\omega-v_t)\mathrm{d}t + \sigma\sqrt{v_t}\mathrm{d}W_2,\quad \mathrm{d}W_1\mathrm{d}W_2=\rho\mathrm{d}t$ |
| `CEV` | Constant Elasticity of Variance | $\mathrm{d}S_t = \mu S_t\mathrm{d}t + \sigma S_t^{\beta}\mathrm{d}W_t$ |
| `MertonJumpDiffusion` | Jump-diffusion (Merton) | $\mathrm{d}S_t = \mu S_t\mathrm{d}t + \sigma\sqrt{S_t}\mathrm{d}W_t + J_t\mathrm{d}N_t$ |
| `VarianceGamma` | Variance Gamma process | $\Delta S_t = \mu S_t\Delta t + \sigma\sqrt{S_t}\Delta W_{G_t},\quad \Delta G_t \sim \Gamma(\Delta t/\nu,\nu)$ |
| `ThreeTwoModel` | 3/2 stochastic volatility model | $\mathrm{d}S_t = \sqrt{v_t}S_t\mathrm{d}W_1,\quad \mathrm{d}v_t = \theta(\omega-v_t)\mathrm{d}t + \sigma v_t^{3/2}\mathrm{d}W_2$ |

## Development

```bash
git clone https://github.com/axeldafflon/quantkit
cd quantkit
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
