import numpy as np
from numba import njit, prange

@njit(parallel=True, cache=True)
def _simulate_standard_bm(Z, dt, n_paths, n_points):
    W = np.zeros((n_paths, n_points))
    step_scale = np.sqrt(dt)

    for p in prange(n_paths):
        for i in range(1, n_points):
            W[p, i] = W[p, i - 1] + step_scale * Z[p, i - 1]
    return W

@njit(parallel=True, cache=True)
def _simulate_bm(Z, dt, mu, sigma, n_paths, n_points):
    W = np.zeros((n_paths, n_points))
    drift = mu * dt
    vol = sigma * np.sqrt(dt)

    for p in prange(n_paths):
        for i in range(1, n_points):
            W[p, i] = W[p, i - 1] + drift + vol * Z[p, i - 1]
    return W

@njit(parallel=True, cache=True)
def _simulate_gbm(Z, dt, mu, sigma, y_start, n_paths, n_points, t):
    W = np.zeros((n_paths, n_points))
    for i in range(1, n_points):
        W[:, i] = W[:, i - 1] + np.sqrt(dt) * Z[:, i - 1]
    S = np.empty((n_paths, n_points))
    for p in prange(n_paths):
        for j in range(n_points):
            S[p, j] = y_start * np.exp((mu - 0.5 * sigma ** 2) * t[j] + sigma * W[p, j])
    return S


class StandardBM:
    """Standard Brownian motion (Wiener process): mu=0, sigma=1."""

    def __init__(self, S0: float = 0.0, seed=None):
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def _simulate_numpy(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        Z = self.rng.normal(0, 1, (n_paths, n_points))
        W = np.zeros((n_paths, n_points))
        dt = (x_end - x_start) / (n_points - 1)
        step_scale = np.sqrt(dt)

        for i in range(1, n_points):
            W[:, i] = W[:, i - 1] + step_scale * Z[:, i - 1]
        return self.S0 + W

    def _simulate_numba(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        Z = self.rng.normal(0, 1, (n_paths, n_points))
        dt = (x_end - x_start) / (n_points - 1)
        return self.S0 + _simulate_standard_bm(Z, dt, n_paths, n_points)

    def simulate(
        self,
        x_start: int,
        x_end: int,
        n_paths: int,
        n_points: int,
        use_numba: bool | None = None,
        numba_threshold: int = 500_000,
    ):
        """
        Creates an array of (n_paths) Standard Brownian motions (mu=0, sigma=1)
        defined on the interval [x_start, x_end].
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :param use_numba: Whether to use numba. If None, determined by numba_threshold.
        :param numba_threshold: Minimum n_paths to trigger numba automatically.
        :return: (n_paths x n_points) matrix
        """
        if use_numba is None:
            use_numba = n_paths >= numba_threshold
        if use_numba:
            return self._simulate_numba(x_start, x_end, n_paths, n_points)
        return self._simulate_numpy(x_start, x_end, n_paths, n_points)


class BrownianMotion:
    """Arithmetic Brownian Motion with arbitrary drift (mu) and volatility (sigma)."""

    def __init__(self, mu: float, sigma: float, S0: float = 0.0, seed=None):
        self.mu = mu
        self.sigma = sigma
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def _simulate_numpy(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        Z = self.rng.normal(0, 1, (n_paths, n_points))
        W = np.zeros((n_paths, n_points))
        dt = (x_end - x_start) / (n_points - 1)
        drift = self.mu * dt
        vol = self.sigma * np.sqrt(dt)

        for i in range(1, n_points):
            W[:, i] = W[:, i - 1] + drift + vol * Z[:, i - 1]
        return self.S0 + W

    def _simulate_numba(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        Z = self.rng.normal(0, 1, (n_paths, n_points))
        dt = (x_end - x_start) / (n_points - 1)
        return self.S0 + _simulate_bm(Z, dt, self.mu, self.sigma, n_paths, n_points)

    def simulate(
        self,
        x_start: int,
        x_end: int,
        n_paths: int,
        n_points: int,
        use_numba: bool | None = None,
        numba_threshold: int = 500_000,
    ):
        """
        Creates an array of (n_paths) Brownian motions with drift (mu) and volatility (sigma)
        defined on the interval [x_start, x_end].
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :param use_numba: Whether to use numba. If None, determined by numba_threshold.
        :param numba_threshold: Minimum n_paths to trigger numba automatically.
        :return: (n_paths x n_points) matrix
        """
        if use_numba is None:
            use_numba = n_paths >= numba_threshold
        if use_numba:
            return self._simulate_numba(x_start, x_end, n_paths, n_points)
        return self._simulate_numpy(x_start, x_end, n_paths, n_points)


class GBM:
    """Geometric Brownian Motion."""

    def __init__(self, mu, sigma, S0, seed=None):
        self.mu = mu
        self.sigma = sigma
        self.y_start = S0
        self.rng = np.random.default_rng(seed)

    def _simulate_numpy(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        Z = self.rng.normal(0, 1, (n_paths, n_points))
        W = np.zeros((n_paths, n_points))
        dt = (x_end - x_start) / (n_points - 1)

        for i in range(1, n_points):
            W[:, i] = W[:, i - 1] + np.sqrt(dt) * Z[:, i - 1]

        t = np.linspace(x_start, x_end, n_points)
        return self.y_start * np.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * W)

    def simulate_numba(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        Z = self.rng.normal(0, 1, (n_paths, n_points))
        dt = (x_end - x_start) / (n_points - 1)
        t = np.linspace(x_start, x_end, n_points)
        return _simulate_gbm(Z, dt, self.mu, self.sigma, self.y_start, n_paths, n_points, t)

    def simulate(
        self,
        x_start: int,
        x_end: int,
        n_paths: int,
        n_points: int,
        use_numba: bool | None = None,
        numba_threshold: int = 500_000,
    ):
        """
        Creates an array of (n_paths) Geometric Brownian motions defined on the interval [x_start, x_end].
        Control the granularity with the n_points parameter.
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :param use_numba: Whether to use numba for simulation. If None, it will be determined based on numba_threshold.
        :param numba_threshold: Threshold for using numba. If the number of paths is
        :return: (n_paths x n_points) matrix
        """
        if use_numba is None:
            use_numba = n_paths >= numba_threshold
        if use_numba:
            return self.simulate_numba(x_start, x_end, n_paths, n_points)
        return self._simulate_numpy(x_start, x_end, n_paths, n_points)

    def compute_option_price(self, paths, K: float, position: str, type: str, expiry: float, r: float, q: float, barrier: dict = None) -> float:
        """
        Computes the arbitrage-free Monte Carlo price of an option, from a series of given GBMs.
        :param K: strike
        :param position: call or put
        :param type: european or asian
        :param expiry: number of days until maturity
        :param r: risk-free rate e.g. 0.05 for 5%
        :param q: dividends, e.g. 0.05 for 5%
        :param barrier: dict with keys "type" (up-and-out, down-and-out, up-and-in, down-and-in) and "level" (barrier level)
        :return: Discounted payoff
        """

        final_values = paths[:, -1]

        if barrier is not None:
            if barrier["type"] == "up-and-out":
                mask = np.max(paths, axis=1) < barrier["level"]
            elif barrier["type"] == "down-and-out":
                mask = np.min(paths, axis=1) > barrier["level"]
            elif barrier["type"] == "up-and-in":
                mask = np.max(paths, axis=1) >= barrier["level"]
            elif barrier["type"] == "down-and-in":
                mask = np.min(paths, axis=1) <= barrier["level"]
            else:
                raise ValueError("Invalid barrier type. Must be one of: up-and-out, down-and-out, up-and-in, down-and-in.")
            final_values = final_values[mask]

        # we only store the last value
        paths_above = final_values[final_values >= K]
        paths_below = final_values[final_values < K]

        average_above = np.mean(paths_above)
        average_below = np.mean(paths_below)

        T = expiry / 365.0
        discount = np.exp(-(r - q) * T)

        if type == "european":
            if position == "call":
                expected_value = average_above / (len(paths))
                return discount * expected_value
            elif position == "put":
                expected_value = average_below / (len(paths))
                return discount * expected_value

        elif type == "asian":
            average = paths.mean(axis=1)
            if position == "call":
                payoff = np.maximum(average - K, 0)
                return discount * np.mean(payoff)
            elif position == "put":
                payoff = np.maximum(K - average, 0)
                return discount * np.mean(payoff)


class RandomWalk:
    """Random walk with arbitrary drift (mu), but no volatility (sigma=0).
    Designed to randomly move up or down by a fixed amount (mu) at each step.
    """

    def __init__(self, mu: float, sigma: float, S0: float = 0.0, seed=None):
        self.mu = mu
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def simulate(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        """
        Creates an array of (n_paths) Random Walks with drift (mu) and no volatility (sigma=0)
        defined on the interval [x_start, x_end].
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :return: (n_paths x n_points) matrix
        """
        Z = self.rng.choice([-1, 1], (n_paths, n_points))
        W = np.zeros((n_paths, n_points))
        dt = (x_end - x_start) / (n_points - 1)
        step_scale = self.mu * dt

        for i in range(1, n_points):
            W[:, i] = W[:, i - 1] + step_scale * Z[:, i - 1]
        return self.S0 + W


class OrnsteinUhlenbeck:
    """Ornstein-Uhlenbeck process with mean reversion speed (theta), long-term mean (mu), and volatility (sigma)."""

    def __init__(self, theta: float, mu: float, sigma: float, S0: float = 0.0, seed=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def simulate(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        """
        Creates an array of (n_paths) Ornstein-Uhlenbeck processes defined on the interval [x_start, x_end].
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :return: (n_paths x n_points) matrix
        """
        Z = self.rng.normal(0, 1, (n_paths, n_points))
        W = np.zeros((n_paths, n_points))
        dt = (x_end - x_start) / (n_points - 1)

        assert(self.theta * dt < 1), "Mean reversion speed (theta) must be less than 1/dt to ensure stability of the process."

        for i in range(1, n_points):
            W[:, i] = W[:, i - 1] + self.theta * (self.mu - W[:, i - 1]) * dt + self.sigma * np.sqrt(dt) * Z[:, i - 1]

        return self.S0 + W


class PoissonProcess:
    """Poisson process with rate (lambda)."""

    def __init__(self, lam: float, S0: float = 0.0, seed=None):
        self.lam = lam
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def simulate(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        """
        Creates an array of (n_paths) Poisson processes defined on the interval [x_start, x_end].
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :return: (n_paths x n_points) matrix
        """
        dt = (x_end - x_start) / (n_points - 1)
        lam_dt = self.lam * dt
        return self.S0 + self.rng.poisson(lam_dt, (n_paths, n_points)).cumsum(axis=1)


class CompoundPoissonProcess:
    """Compound Poisson process with rate (lambda) and jump size distribution (jump_dist)."""

    def __init__(self, lam: float, jump_dist: callable, S0: float = 0.0, seed=None):
        self.lam = lam
        self.jump_dist = jump_dist
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def simulate(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        """
        Creates an array of (n_paths) Compound Poisson processes defined on the interval [x_start, x_end].
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :return: (n_paths x n_points) matrix
        """
        dt = (x_end - x_start) / (n_points - 1)
        lam_dt = self.lam * dt
        jumps = self.rng.poisson(lam_dt, (n_paths, n_points))
        jump_sizes = self.jump_dist(size=(n_paths, n_points)) * jumps
        return self.S0 + jump_sizes.cumsum(axis=1)


class LevyFlight:
    """Levy flight with stability parameter (alpha) and scale parameter (scale)."""

    def __init__(self, alpha: float, scale: float, S0: float = 0.0, seed=None):
        self.alpha = alpha
        self.scale = scale
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def simulate(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        """
        Creates an array of (n_paths) Levy flights defined on the interval [x_start, x_end].
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :return: (n_paths x n_points) matrix
        """
        dt = (x_end - x_start) / (n_points - 1)
        step_scale = self.scale * dt**(1/self.alpha)
        steps = self.rng.standard_cauchy((n_paths, n_points)) * step_scale
        return self.S0 + steps.cumsum(axis=1)


class FractionalBrownianMotion:
    """Fractional Brownian motion with Hurst parameter (H)."""

    def __init__(self, H: float, S0: float = 0.0, seed=None):
        self.H = H
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def simulate(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        """
        Creates an array of (n_paths) Fractional Brownian motions defined on the interval [x_start, x_end].
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :return: (n_paths x n_points) matrix
        """
        dt = (x_end - x_start) / (n_points - 1)
        t = np.linspace(x_start, x_end, n_points)
        cov = 0.5 * (t[:, None]**(2*self.H) + t[None, :]**(2*self.H) - np.abs(t[:, None] - t[None, :])**(2*self.H))
        L = np.linalg.cholesky(cov)
        Z = self.rng.normal(0, 1, (n_paths, n_points))
        W = Z @ L.T
        return self.S0 + W


class StableProcess:
    """Stable process with stability parameter (alpha), skewness parameter (beta), scale parameter (scale), and location parameter (loc)."""

    def __init__(self, alpha: float, beta: float, scale: float, loc: float, S0: float = 0.0, seed=None):
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.loc = loc
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def simulate(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        """
        Creates an array of (n_paths) Stable processes defined on the interval [x_start, x_end].
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :return: (n_paths x n_points) matrix
        """
        dt = (x_end - x_start) / (n_points - 1)
        step_scale = self.scale * dt**(1/self.alpha)
        steps = self.rng.standard_cauchy((n_paths, n_points)) * step_scale + self.loc * dt
        return self.S0 + steps.cumsum(axis=1)


class HestonModel:
    """Heston model with mean reversion speed (theta), long-term variance (omega), volatility of variance (sigma), correlation (rho), and initial variance (v0)."""

    def __init__(self, theta: float, omega: float, sigma: float, rho: float, v0: float, S0: float = 0.0, seed=None):
        self.theta = theta
        self.omega = omega
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def simulate(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        """
        Creates an array of (n_paths) Heston model paths defined on the interval [x_start, x_end].
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :return: (n_paths x n_points) matrix
        """
        dt = (x_end - x_start) / (n_points - 1)
        S = np.zeros((n_paths, n_points))
        v = np.zeros((n_paths, n_points))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        for i in range(1, n_points):
            Z1 = self.rng.normal(0, 1, n_paths)
            Z2 = self.rng.normal(0, 1, n_paths)
            W1 = Z1 * np.sqrt(dt)
            W2 = (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2) * np.sqrt(dt)

            v[:, i] = np.abs(v[:, i - 1] + self.theta * (self.omega - v[:, i - 1]) * dt + self.sigma * np.sqrt(v[:, i - 1]) * W2)
            S[:, i] = S[:, i - 1] * np.exp(-0.5 * v[:, i - 1] * dt + np.sqrt(v[:, i - 1]) * W1)

        return S


class CEV:
    r"""Constant Elasticity of Variance (CEV) model with elasticity parameter ``beta``.

    Stochastic differential equation (SDE):

    .. math::
         dS_t = \mu S_t\,dt + \sigma S_t^{\beta}\,dW_t

    where :math:`W_t` is a standard Brownian motion.

    ASCII fallback:
        dS_t = mu S_t dt + sigma S_t^beta dW_t

    """

    def __init__(self, mu: float, sigma: float, beta: float, S0: float = 0.0, seed=None):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def simulate(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        """
        Creates an array of (n_paths) CEV model paths defined on the interval [x_start, x_end].
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :return: (n_paths x n_points) matrix
        """
        dt = (x_end - x_start) / (n_points - 1)
        S = np.zeros((n_paths, n_points))
        S[:, 0] = self.S0

        for i in range(1, n_points):
            Z = self.rng.normal(0, 1, n_paths)
            S[:, i] = S[:, i - 1] + self.mu * S[:, i - 1] * dt + self.sigma * np.power(S[:, i - 1], self.beta) * np.sqrt(dt) * Z

        return S


class MertonJumpDiffusion:
    """Merton jump diffusion model with drift (mu), volatility (sigma), jump intensity (lam), and jump size distribution (jump_dist)."""

    def __init__(self, mu: float, sigma: float, lam: float, jump_dist: callable, S0: float = 0.0, seed=None):
        self.mu = mu
        self.sigma = sigma
        self.lam = lam
        self.jump_dist = jump_dist
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def simulate(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        """
        Creates an array of (n_paths) Merton jump diffusion paths defined on the interval [x_start, x_end].
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :return: (n_paths x n_points) matrix
        """
        dt = (x_end - x_start) / (n_points - 1)
        S = np.zeros((n_paths, n_points))
        S[:, 0] = self.S0

        for i in range(1, n_points):
            Z = self.rng.normal(0, 1, n_paths)
            jumps = self.rng.poisson(self.lam * dt, n_paths)
            jump_sizes = self.jump_dist(size=n_paths) * jumps
            S[:, i] = S[:, i - 1] + self.mu * S[:, i - 1] * dt + self.sigma * np.sqrt(S[:, i - 1]) * np.sqrt(dt) * Z + jump_sizes

        return S


class VarianceGamma:
    """Variance Gamma process with drift (mu), volatility (sigma), and variance of the gamma process (nu)."""

    def __init__(self, mu: float, sigma: float, nu: float, S0: float = 0.0, seed=None):
        self.mu = mu
        self.sigma = sigma
        self.nu = nu
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def simulate(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        """
        Creates an array of (n_paths) Variance Gamma processes defined on the interval [x_start, x_end].
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :return: (n_paths x n_points) matrix
        """
        dt = (x_end - x_start) / (n_points - 1)
        S = np.zeros((n_paths, n_points))
        S[:, 0] = self.S0

        for i in range(1, n_points):
            Z = self.rng.normal(0, 1, n_paths)
            G = self.rng.gamma(dt / self.nu, scale=self.nu, size=n_paths)
            S[:, i] = S[:, i - 1] + self.mu * S[:, i - 1] * dt + self.sigma * np.sqrt(S[:, i - 1]) * np.sqrt(G) * Z

        return S


class ThreeTwoModel:
    """3/2 model with mean reversion speed (theta), long-term variance (omega), volatility of variance (sigma), and initial variance (v0)."""

    def __init__(self, theta: float, omega: float, sigma: float, v0: float, S0: float = 0.0, seed=None):
        self.theta = theta
        self.omega = omega
        self.sigma = sigma
        self.v0 = v0
        self.S0 = S0
        self.rng = np.random.default_rng(seed)

    def simulate(self, x_start: int, x_end: int, n_paths: int, n_points: int):
        """
        Creates an array of (n_paths) 3/2 model paths defined on the interval [x_start, x_end].
        :param x_start: Interval start
        :param x_end: Interval end
        :param n_paths: Number of paths
        :param n_points: Number of points
        :return: (n_paths x n_points) matrix
        """
        dt = (x_end - x_start) / (n_points - 1)
        S = np.zeros((n_paths, n_points))
        v = np.zeros((n_paths, n_points))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        for i in range(1, n_points):
            Z1 = self.rng.normal(0, 1, n_paths)
            Z2 = self.rng.normal(0, 1, n_paths)
            W1 = Z1 * np.sqrt(dt)
            W2 = Z2 * np.sqrt(dt)

            v[:, i] = np.abs(v[:, i - 1] + self.theta * (self.omega - v[:, i - 1]) * dt + self.sigma * np.power(v[:, i - 1], 1.5) * W2)
            S[:, i] = S[:, i - 1] * np.exp(-0.5 * v[:, i - 1] * dt + np.sqrt(v[:, i - 1]) * W1)

        return S
