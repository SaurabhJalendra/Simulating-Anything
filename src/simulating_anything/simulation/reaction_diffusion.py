"""Gray-Scott reaction-diffusion simulation in pure JAX."""

from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


def _laplacian_2d(field: "jnp.ndarray", dx: float) -> "jnp.ndarray":
    """Compute 2D Laplacian with periodic boundary conditions."""
    return (
        jnp.roll(field, 1, axis=0)
        + jnp.roll(field, -1, axis=0)
        + jnp.roll(field, 1, axis=1)
        + jnp.roll(field, -1, axis=1)
        - 4.0 * field
    ) / (dx * dx)


def _gray_scott_step(
    u: "jnp.ndarray",
    v: "jnp.ndarray",
    D_u: float,
    D_v: float,
    f: float,
    k: float,
    dt: float,
    dx: float,
) -> tuple["jnp.ndarray", "jnp.ndarray"]:
    """Single Gray-Scott timestep (forward Euler)."""
    lap_u = _laplacian_2d(u, dx)
    lap_v = _laplacian_2d(v, dx)

    uvv = u * v * v
    du = D_u * lap_u - uvv + f * (1.0 - u)
    dv = D_v * lap_v + uvv - (f + k) * v

    u_new = u + dt * du
    v_new = v + dt * dv
    return u_new, v_new


class GrayScottSimulation(SimulationEnvironment):
    """Gray-Scott reaction-diffusion model.

    State is a 2-channel (u, v) concentration field on a 2D grid.
    Uses periodic boundary conditions and finite-difference Laplacian.
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.D_u = p.get("D_u", 0.16)
        self.D_v = p.get("D_v", 0.08)
        self.f = p.get("f", 0.035)
        self.k = p.get("k", 0.065)
        self.nx, self.ny = config.grid_resolution[:2]
        lx = config.domain_size[0] if len(config.domain_size) > 0 else 2.5
        self.dx = lx / self.nx

        if _HAS_JAX:
            self._jit_step = jax.jit(_gray_scott_step, static_argnums=())
        else:
            self._jit_step = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize u=1, v=0 everywhere with a small square perturbation."""
        rng = np.random.default_rng(seed or self.config.seed)

        u = np.ones((self.nx, self.ny), dtype=np.float64)
        v = np.zeros((self.nx, self.ny), dtype=np.float64)

        # Seed a square perturbation in the center
        cx, cy = self.nx // 2, self.ny // 2
        r = max(self.nx // 10, 2)
        u[cx - r : cx + r, cy - r : cy + r] = 0.50
        v[cx - r : cx + r, cy - r : cy + r] = 0.25

        # Add small noise
        u += 0.01 * rng.standard_normal(u.shape)
        v += 0.01 * rng.standard_normal(v.shape)

        self._u = u
        self._v = v
        self._step_count = 0
        self._state = np.stack([u, v], axis=-1)
        return self._state

    def step(self) -> np.ndarray:
        """Advance one Gray-Scott timestep."""
        if _HAS_JAX and self._jit_step is not None:
            u_jax = jnp.array(self._u)
            v_jax = jnp.array(self._v)
            u_new, v_new = self._jit_step(
                u_jax, v_jax, self.D_u, self.D_v, self.f, self.k, self.config.dt, self.dx
            )
            self._u = np.asarray(u_new)
            self._v = np.asarray(v_new)
        else:
            # Pure numpy fallback
            self._numpy_step()

        self._step_count += 1
        self._state = np.stack([self._u, self._v], axis=-1)
        return self._state

    def _numpy_step(self) -> None:
        """Numpy fallback for Gray-Scott step."""
        dx = self.dx
        dt = self.config.dt

        def lap(f: np.ndarray) -> np.ndarray:
            return (
                np.roll(f, 1, axis=0)
                + np.roll(f, -1, axis=0)
                + np.roll(f, 1, axis=1)
                + np.roll(f, -1, axis=1)
                - 4.0 * f
            ) / (dx * dx)

        uvv = self._u * self._v * self._v
        du = self.D_u * lap(self._u) - uvv + self.f * (1.0 - self._u)
        dv = self.D_v * lap(self._v) + uvv - (self.f + self.k) * self._v

        self._u = self._u + dt * du
        self._v = self._v + dt * dv

    def observe(self) -> np.ndarray:
        """Return current (u, v) field."""
        return self._state
