# quantkit

A Python library for quantitative finance: stochastic processes, Monte Carlo
simulation, and real-time SDL2 visualisation.

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

| Class | Description |
|---|---|
| `StandardBM` | Standard Brownian motion (µ=0, σ=1) |
| `BrownianMotion` | Arithmetic BM with drift and volatility |
| `GBM` | Geometric Brownian Motion |
| `RandomWalk` | Discrete random walk |
| `OrnsteinUhlenbeck` | Mean-reverting OU process |
| `PoissonProcess` | Homogeneous Poisson process |
| `CompoundPoissonProcess` | Compound Poisson with custom jump distribution |
| `LevyFlight` | Lévy flight (Cauchy-distributed steps) |
| `FractionalBrownianMotion` | fBM with Hurst parameter H |
| `StableProcess` | α-stable process |
| `HestonModel` | Stochastic volatility (Heston) |
| `CEV` | Constant Elasticity of Variance |
| `MertonJumpDiffusion` | Jump-diffusion (Merton) |
| `VarianceGamma` | Variance Gamma process |
| `ThreeTwoModel` | 3/2 stochastic volatility model |

## Development

```bash
git clone https://github.com/axeldafflon/quantkit
cd quantkit
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
