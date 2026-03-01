"""2D Oregonator PDE for Belousov-Zhabotinsky spiral waves.

Implements the two-variable Oregonator reaction-diffusion model:

    du/dt = D_u * nabla^2(u) + (1/eps) * (u - u^2 - f*v*(u - q)/(u + q))
    dv/dt = u - v

where u = [HBrO2] (activator), v = [Ce4+] (catalyst/inhibitor).
Only u diffuses (D_v = 0 by default). The system supports spiral waves,
target patterns, and excitable pulse propagation.

Grid: Nx x Ny with spacing dx, Neumann (zero-flux) boundary conditions.
Integration: Forward Euler with finite-difference Laplacian.

CFL stability requirement: dt < dx^2 / (4 * D_u).
For dx=0.5, D_u=1.0: dt < 0.0625, so dt=0.01 is safe.
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


def _laplacian_2d_neumann(field: np.ndarray, dx: float) -> np.ndarray:
    """Compute 2D Laplacian with Neumann (zero-flux) boundary conditions.

    Uses second-order finite differences. At boundaries, the ghost cell
    mirrors the interior value (df/dn = 0).

    Args:
        field: 2D array of shape (Nx, Ny).
        dx: Grid spacing.

    Returns:
        Laplacian of the field, same shape as input.
    """
    # Pad with reflected boundary (Neumann BC)
    padded = np.pad(field, 1, mode="edge")
    lap = (
        padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        + padded[1:-1, 2:]
        + padded[1:-1, :-2]
        - 4.0 * padded[1:-1, 1:-1]
    ) / (dx * dx)
    return lap


class BZSpiralSimulation(SimulationEnvironment):
    """2D Oregonator PDE for BZ reaction spiral waves.

    State: u and v on a 2D grid, flattened to shape (2 * Nx * Ny,).
    The u field (HBrO2 activator) diffuses with coefficient D_u.
    The v field (Ce4+ inhibitor) does not diffuse (D_v = 0 by default).

    The Oregonator kinetics with eps ~ 0.01 produce a stiff fast-slow
    system where u jumps quickly between excited and rest states, while
    v recovers slowly. This generates excitable dynamics that support
    spiral wave solutions.

    Parameters:
        eps: Timescale ratio (default 0.01, u evolves 100x faster).
        f: Stoichiometric factor (default 1.0).
        q: Excitability parameter (default 0.002).
        D_u: Activator diffusion coefficient (default 1.0).
        D_v: Inhibitor diffusion coefficient (default 0.0).
        Nx: Grid points in x (default 128).
        Ny: Grid points in y (default 128).
        dx: Grid spacing (default 0.5).
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.eps = p.get("eps", 0.01)
        self.f = p.get("f", 1.0)
        self.q = p.get("q", 0.002)
        self.D_u = p.get("D_u", 1.0)
        self.D_v = p.get("D_v", 0.0)
        self.Nx = int(p.get("Nx", 128))
        self.Ny = int(p.get("Ny", 128))
        self.dx = p.get("dx", 0.5)

        # Verify CFL condition
        if self.D_u > 0:
            dt_cfl = self.dx ** 2 / (4.0 * self.D_u)
            if config.dt > dt_cfl:
                raise ValueError(
                    f"dt={config.dt} violates CFL condition dt < {dt_cfl:.6f} "
                    f"for dx={self.dx}, D_u={self.D_u}. Reduce dt."
                )

        self._u: np.ndarray | None = None
        self._v: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize with a broken wavefront to seed spiral formation.

        Creates a planar wavefront (excited strip along x-axis) with a
        gap cut in the middle. The broken end curls into a spiral tip.
        """
        rng = np.random.default_rng(seed or self.config.seed)

        # Start at rest state: u ~ q, v ~ 0
        u = np.full((self.Nx, self.Ny), self.q, dtype=np.float64)
        v = np.zeros((self.Nx, self.Ny), dtype=np.float64)

        # Create a broken planar wavefront to seed spiral
        # Excited strip in the left half (high u)
        half_x = self.Nx // 2
        u[:half_x, :] = 0.8

        # Cut a gap in the upper half to break the wavefront
        half_y = self.Ny // 2
        u[:half_x, half_y:] = self.q

        # Set recovery variable behind the wavefront
        # (refractory tail in upper-left quadrant)
        v[:half_x, half_y:] = 0.3

        # Small noise to break perfect symmetry
        u += 0.001 * rng.standard_normal(u.shape)
        v += 0.001 * np.abs(rng.standard_normal(v.shape))

        # Clamp to physical range
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        self._u = u
        self._v = v
        self._step_count = 0
        self._state = np.concatenate([u.ravel(), v.ravel()])
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using forward Euler with Neumann BC."""
        dt = self.config.dt
        u = self._u
        v = self._v

        # Diffusion (only u diffuses by default)
        lap_u = _laplacian_2d_neumann(u, self.dx)
        diffusion_u = self.D_u * lap_u

        if self.D_v > 0:
            lap_v = _laplacian_2d_neumann(v, self.dx)
            diffusion_v = self.D_v * lap_v
        else:
            diffusion_v = 0.0

        # Oregonator reaction kinetics
        # du/dt = D_u*lap(u) + (1/eps)*(u - u^2 - f*v*(u-q)/(u+q))
        # dv/dt = D_v*lap(v) + u - v
        u_safe = np.maximum(u, 0.0)
        denom = u_safe + self.q
        reaction_u = (
            u_safe - u_safe ** 2 - self.f * v * (u_safe - self.q) / denom
        ) / self.eps
        reaction_v = u_safe - v

        u_new = u + dt * (diffusion_u + reaction_u)
        v_new = v + dt * (diffusion_v + reaction_v)

        # Clamp to non-negative
        u_new = np.maximum(u_new, 0.0)
        v_new = np.maximum(v_new, 0.0)

        self._u = u_new
        self._v = v_new
        self._step_count += 1
        self._state = np.concatenate([u_new.ravel(), v_new.ravel()])
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [u_flat, v_flat]."""
        return self._state

    def get_u_field(self) -> np.ndarray:
        """Return u (activator) as a 2D array of shape (Nx, Ny)."""
        return self._u.copy()

    def get_v_field(self) -> np.ndarray:
        """Return v (inhibitor) as a 2D array of shape (Nx, Ny)."""
        return self._v.copy()

    def detect_spiral_tip(self, u_field: np.ndarray | None = None) -> tuple[int, int]:
        """Find spiral tip location using phase singularity detection.

        The spiral tip is where both u and v contours cross, forming a
        phase singularity. We detect it by finding the point where the
        u-field crosses a threshold and the gradient direction rotates
        by 2pi around it.

        Uses a simpler approach: find the intersection of the u = u_thresh
        and v = v_thresh contour lines by looking for grid cells where
        both u and v are near their respective thresholds.

        Args:
            u_field: Optional u field to analyze. Uses current if None.

        Returns:
            (ix, iy) grid indices of the spiral tip, or (-1, -1) if none.
        """
        if u_field is None:
            u_field = self._u
        v_field = self._v

        # Threshold values: roughly mid-range of the excited/rest states
        u_thresh = 0.3
        v_thresh = 0.15

        # Find cells near both thresholds simultaneously
        u_near = np.abs(u_field - u_thresh)
        v_near = np.abs(v_field - v_thresh)

        # Combined proximity score
        score = u_near + v_near

        # The spiral tip is where this score is minimized
        min_idx = np.unravel_index(np.argmin(score), score.shape)

        # Verify it is actually a reasonable tip (not just uniform)
        u_range = np.max(u_field) - np.min(u_field)
        if u_range < 0.1:
            return (-1, -1)

        return (int(min_idx[0]), int(min_idx[1]))

    def compute_spiral_frequency(
        self, n_steps: int = 5000
    ) -> float:
        """Track spiral tip rotation and compute its frequency.

        Runs the simulation for n_steps, tracking the spiral tip angle
        relative to the grid center. The frequency is estimated from
        the number of full rotations.

        Args:
            n_steps: Number of steps to track.

        Returns:
            Spiral rotation frequency (rotations per time unit),
            or 0.0 if no spiral detected.
        """
        cx, cy = self.Nx // 2, self.Ny // 2
        angles = []

        for _ in range(n_steps):
            self.step()
            tip = self.detect_spiral_tip()
            if tip[0] < 0:
                continue
            dx_tip = tip[0] - cx
            dy_tip = tip[1] - cy
            if dx_tip == 0 and dy_tip == 0:
                continue
            angles.append(np.arctan2(dy_tip, dx_tip))

        if len(angles) < 10:
            return 0.0

        # Unwrap angles and compute total rotation
        angles = np.array(angles)
        unwrapped = np.unwrap(angles)
        total_rotation = abs(unwrapped[-1] - unwrapped[0])
        total_time = n_steps * self.config.dt

        # Frequency = total rotations / total time
        frequency = total_rotation / (2 * np.pi * total_time)
        return float(frequency)
