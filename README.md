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
| `StandardBM` | Standard Brownian motion (µ=0, σ=1) | $dS_t = dW_t$ |
| `BrownianMotion` | Arithmetic BM with drift and volatility | $dS_t = \mu\,dt + \sigma\,dW_t$ |
| `GBM` | Geometric Brownian Motion | $dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$ |
| `RandomWalk` | Discrete random walk | $\Delta S_t = \mu\,\Delta t\,\varepsilon_t,\ \varepsilon_t \in \{-1,+1\}$ |
| `OrnsteinUhlenbeck` | Mean-reverting OU process | $dS_t = \theta(\mu - S_t)\,dt + \sigma\,dW_t$ |
| `PoissonProcess` | Homogeneous Poisson process | $dN_t \sim \mathrm{Poisson}(\lambda\,dt)$ |
| `CompoundPoissonProcess` | Compound Poisson with custom jump distribution | $dS_t = J_t\,dN_t$ |
| `LevyFlight` | Lévy flight (Cauchy-distributed steps) | $\Delta S_t = c\,(\Delta t)^{1/\alpha}\,\Xi_t,\ \Xi_t \sim \mathrm{Cauchy}(0,1)$ |
| `FractionalBrownianMotion` | fBM with Hurst parameter H | $S_t = B_t^H,\ \operatorname{Cov}(B_t^H,B_s^H)=\tfrac12\left(t^{2H}+s^{2H}-(t-s)^{2H}\right)$ |
| `StableProcess` | α-stable process | $\Delta S_t = c\,(\Delta t)^{1/\alpha}L_t + \ell\,\Delta t$ |
| `HestonModel` | Stochastic volatility (Heston) | $dS_t = \sqrt{v_t}\,S_t\,dW_1,\quad dv_t = \theta(\omega-v_t)dt + \sigma\sqrt{v_t}\,dW_2,\quad dW_1dW_2=\rho\,dt$ |
| `CEV` | Constant Elasticity of Variance | $dS_t = \mu S_t\,dt + \sigma S_t^{\beta}\,dW_t$ |
| `MertonJumpDiffusion` | Jump-diffusion (Merton) | $dS_t = \mu S_t\,dt + \sigma\sqrt{S_t}\,dW_t + J_t\,dN_t$ |
| `VarianceGamma` | Variance Gamma process | $\Delta S_t = \mu S_t\,\Delta t + \sigma\sqrt{S_t}\,\Delta W_{G_t},\quad \Delta G_t \sim \Gamma(\Delta t/\nu,\nu)$ |
| `ThreeTwoModel` | 3/2 stochastic volatility model | $dS_t = \sqrt{v_t}\,S_t\,dW_1,\quad dv_t = \theta(\omega-v_t)dt + \sigma v_t^{3/2} dW_2$ |

## Development

```bash
git clone https://github.com/axeldafflon/quantkit
cd quantkit
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
