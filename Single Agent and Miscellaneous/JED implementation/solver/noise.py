"""
solver/noise.py — Additive Gaussian process noise d_k ~ N(mu, Sigma).

Nonzero mean, general (anisotropic) covariance, over the FULL state
[x, y, vx, vy] — see config.NOISE_MU / config.NOISE_SIGMA. Everything else
(barriers.py's LSE relaxation, pu_bound.py's delta) reads mu/Sigma from here
so the "assumed" noise model used for planning safety and the "real" noise
model used for rollouts are, by construction, the same object unless the
caller deliberately passes a scaled/perturbed Sigma (e.g. the tuning
dashboard's sigma-multiplier).
"""

from __future__ import annotations

import numpy as np

from . import config


class GaussianProcessNoise:
    """d_k ~ N(mu, Sigma), i.i.d. across timesteps."""

    def __init__(self, mu: np.ndarray = config.NOISE_MU,
                 Sigma: np.ndarray = config.NOISE_SIGMA,
                 rng: np.random.Generator | None = None) -> None:
        self.mu = np.asarray(mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample(self) -> np.ndarray:
        return self.rng.multivariate_normal(self.mu, self.Sigma)

    def scaled(self, sigma_mult: float) -> "GaussianProcessNoise":
        """Return a copy with Sigma_used = sigma_mult^2 * Sigma (dashboard use)."""
        return GaussianProcessNoise(self.mu, (sigma_mult ** 2) * self.Sigma, self.rng)
