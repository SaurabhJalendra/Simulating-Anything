"""2D ideal gas (Boltzmann gas) simulation -- statistical mechanics.

Target rediscoveries:
- Ideal gas law: PV = NkT (with k_B = 1)
- Maxwell-Boltzmann speed distribution: f(v) = (m*v/kT)*exp(-m*v^2/(2kT))
- Pressure proportional to temperature at fixed N, V
- Energy equipartition: <KE> = N*k*T (2D, k_B=1)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class BoltzmannGas2D(SimulationEnvironment):
    """2D ideal gas with elastic hard-sphere collisions in a box.

    N particles with mass m bounce elastically off walls and each other.
    State vector: flattened [x1, y1, x2, y2, ..., vx1, vy1, vx2, vy2, ...]
    of shape (4*N,).

    Parameters:
        N: number of particles (default 100)
        L: side length of square box (default 10.0)
        T: initial temperature (default 1.0, with k_B=1)
        particle_radius: hard-sphere radius (default 0.1)
        m: particle mass (default 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N = int(p.get("N", 100))
        self.L = p.get("L", 10.0)
        self.T_init = p.get("T", 1.0)
        self.radius = p.get("particle_radius", 0.1)
        self.m = p.get("m", 1.0)

        # Positions and velocities stored as (N, 2) arrays
        self._pos: np.ndarray = np.zeros((self.N, 2))
        self._vel: np.ndarray = np.zeros((self.N, 2))

        # Wall impulse accumulator for pressure measurement
        self._wall_impulse = 0.0
        self._wall_time = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize random positions (no overlap) and MB velocities."""
        rng = np.random.default_rng(seed)

        # Place particles on a grid to avoid overlaps, then jitter
        self._pos = self._place_particles(rng)

        # Maxwell-Boltzmann velocity initialization (k_B = 1):
        # Each velocity component ~ N(0, sqrt(kT/m))
        sigma_v = np.sqrt(self.T_init / self.m)
        self._vel = rng.normal(0.0, sigma_v, size=(self.N, 2))

        # Remove net momentum so center of mass is stationary
        self._vel -= self._vel.mean(axis=0)

        # Rescale to exact target temperature
        self._rescale_to_temperature(self.T_init)

        self._wall_impulse = 0.0
        self._wall_time = 0.0
        self._step_count = 0
        return self.observe()

    def step(self) -> np.ndarray:
        """Advance one timestep: move, wall reflect, particle collisions."""
        dt = self.config.dt

        # Move particles
        self._pos += self._vel * dt

        # Wall collisions (elastic reflection)
        self._handle_wall_collisions(dt)

        # Particle-particle collisions (O(N^2) detection)
        self._handle_particle_collisions()

        self._wall_time += dt
        self._step_count += 1
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return flattened state: [x1,y1,...,xN,yN, vx1,vy1,...,vxN,vyN]."""
        return np.concatenate([
            self._pos.ravel(),
            self._vel.ravel(),
        ])

    def _place_particles(self, rng: np.random.Generator) -> np.ndarray:
        """Place N particles in the box avoiding overlaps."""
        margin = self.radius
        pos = np.zeros((self.N, 2))

        # Try grid placement first
        n_side = int(np.ceil(np.sqrt(self.N)))
        spacing = (self.L - 2 * margin) / max(n_side, 1)

        # If particles fit on a regular grid, use it
        if spacing > 2 * self.radius:
            idx = 0
            for i in range(n_side):
                for j in range(n_side):
                    if idx >= self.N:
                        break
                    pos[idx, 0] = margin + (i + 0.5) * spacing
                    pos[idx, 1] = margin + (j + 0.5) * spacing
                    idx += 1
                if idx >= self.N:
                    break
            # Add small jitter
            jitter = spacing * 0.1
            pos += rng.uniform(-jitter, jitter, size=pos.shape)
        else:
            # Fallback: random placement with rejection
            for i in range(self.N):
                for _ in range(1000):
                    candidate = rng.uniform(
                        margin, self.L - margin, size=2
                    )
                    if i == 0:
                        pos[i] = candidate
                        break
                    dists = np.linalg.norm(
                        pos[:i] - candidate, axis=1
                    )
                    if np.all(dists > 2 * self.radius):
                        pos[i] = candidate
                        break
                else:
                    # Give up on no-overlap: just place randomly
                    pos[i] = rng.uniform(margin, self.L - margin, size=2)

        return pos

    def _rescale_to_temperature(self, target_T: float) -> None:
        """Rescale velocities to match target temperature exactly.

        In 2D: T = m * <v^2> / (2 * k_B) per particle, with k_B = 1.
        So <v^2> = 2*T/m. Total KE = 0.5*m*N*<v^2> = N*T.
        """
        current_T = self.temperature
        if current_T > 0:
            scale = np.sqrt(target_T / current_T)
            self._vel *= scale

    def _handle_wall_collisions(self, dt: float) -> None:
        """Reflect particles off walls and accumulate impulse."""
        for dim in range(2):
            # Lower wall (coordinate < radius)
            mask_low = self._pos[:, dim] < self.radius
            if np.any(mask_low):
                self._pos[mask_low, dim] = 2 * self.radius - self._pos[
                    mask_low, dim
                ]
                impulse = np.sum(
                    np.abs(self._vel[mask_low, dim])
                ) * 2 * self.m
                self._wall_impulse += impulse
                self._vel[mask_low, dim] = np.abs(
                    self._vel[mask_low, dim]
                )

            # Upper wall (coordinate > L - radius)
            mask_high = self._pos[:, dim] > self.L - self.radius
            if np.any(mask_high):
                self._pos[mask_high, dim] = (
                    2 * (self.L - self.radius)
                    - self._pos[mask_high, dim]
                )
                impulse = np.sum(
                    np.abs(self._vel[mask_high, dim])
                ) * 2 * self.m
                self._wall_impulse += impulse
                self._vel[mask_high, dim] = -np.abs(
                    self._vel[mask_high, dim]
                )

    def _handle_particle_collisions(self) -> None:
        """Detect and resolve elastic hard-sphere collisions (O(N^2))."""
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dx = self._pos[j] - self._pos[i]
                dist_sq = dx[0] ** 2 + dx[1] ** 2
                min_dist = 2 * self.radius

                if dist_sq < min_dist ** 2 and dist_sq > 0:
                    dist = np.sqrt(dist_sq)
                    # Unit normal from i to j
                    n_hat = dx / dist

                    # Relative velocity
                    dv = self._vel[i] - self._vel[j]
                    dvn = np.dot(dv, n_hat)

                    # Only collide if approaching
                    if dvn > 0:
                        # Equal mass elastic collision: exchange
                        # normal velocity components
                        self._vel[i] -= dvn * n_hat
                        self._vel[j] += dvn * n_hat

                        # Separate particles to avoid overlap
                        overlap = min_dist - dist
                        self._pos[i] -= 0.5 * overlap * n_hat
                        self._pos[j] += 0.5 * overlap * n_hat

    @property
    def temperature(self) -> float:
        """Kinetic temperature: T = m * <v^2> / (2 * k_B * d).

        In 2D with k_B = 1: T = m * sum(v^2) / (2 * N).
        We use 2*N degrees of freedom (2D).
        """
        if self.N == 0:
            return 0.0
        v_sq = np.sum(self._vel ** 2)
        return self.m * v_sq / (2.0 * self.N)

    @property
    def pressure(self) -> float:
        """Pressure from wall impulse transfer.

        P = total_impulse / (perimeter * time).
        The box perimeter in 2D is 4*L.
        """
        if self._wall_time <= 0:
            return 0.0
        perimeter = 4.0 * self.L
        return self._wall_impulse / (perimeter * self._wall_time)

    @property
    def total_energy(self) -> float:
        """Total kinetic energy: sum of 0.5 * m * v_i^2."""
        return 0.5 * self.m * np.sum(self._vel ** 2)

    def speeds(self) -> np.ndarray:
        """Return array of particle speeds."""
        return np.sqrt(np.sum(self._vel ** 2, axis=1))

    def reset_pressure(self) -> None:
        """Reset the pressure accumulator."""
        self._wall_impulse = 0.0
        self._wall_time = 0.0
