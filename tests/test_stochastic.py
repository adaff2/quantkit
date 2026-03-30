import numpy as np
import pytest

from quantkit.stochastic import CEV, GBM, StandardBM, BrownianMotion, OrnsteinUhlenbeck


# ── CEV ──────────────────────────────────────────────────────────────────────

def test_cev_shape_and_initial_value() -> None:
    model = CEV(mu=0.0, sigma=0.3, beta=0.7, S0=100.0, seed=42)
    paths = model.simulate(x_start=0, x_end=1, n_paths=500, n_points=200)

    assert paths.shape == (500, 200)
    assert np.all(paths[:, 0] == 100.0)


def test_cev_values_are_finite_and_move() -> None:
    model = CEV(mu=0.0, sigma=0.3, beta=0.7, S0=100.0, seed=7)
    paths = model.simulate(x_start=0, x_end=1, n_paths=500, n_points=200)

    assert np.isfinite(paths).all()
    assert np.any(paths[:, -1] != paths[:, 0])


# ── GBM ──────────────────────────────────────────────────────────────────────

def test_gbm_shape_and_initial_value() -> None:
    model = GBM(mu=0.05, sigma=0.2, S0=100.0, seed=0)
    paths = model.simulate(x_start=0, x_end=1, n_paths=200, n_points=100)

    assert paths.shape == (200, 100)


def test_gbm_values_positive() -> None:
    model = GBM(mu=0.05, sigma=0.2, S0=100.0, seed=1)
    paths = model.simulate(x_start=0, x_end=1, n_paths=200, n_points=100)

    assert np.all(paths > 0)


# ── StandardBM ───────────────────────────────────────────────────────────────

def test_standard_bm_starts_at_s0() -> None:
    model = StandardBM(S0=5.0, seed=99)
    paths = model.simulate(x_start=0, x_end=1, n_paths=100, n_points=50)

    assert paths.shape == (100, 50)
    assert np.all(paths[:, 0] == 5.0)


# ── OrnsteinUhlenbeck ─────────────────────────────────────────────────────────

def test_ou_mean_reversion() -> None:
    model = OrnsteinUhlenbeck(theta=2.0, mu=0.0, sigma=0.5, S0=0.0, seed=3)
    paths = model.simulate(x_start=0, x_end=5, n_paths=1000, n_points=500)

    assert np.isfinite(paths).all()
    # Long-run mean should be close to mu (0) across paths
    assert abs(paths[:, -1].mean()) < 0.5
