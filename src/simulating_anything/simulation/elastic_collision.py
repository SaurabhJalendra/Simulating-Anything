"""1D elastic collision chain simulation.

A chain of N particles on a line undergoing elastic (or inelastic) collisions.
Particles move freely between collisions; when two adjacent particles meet
(x_i >= x_{i+1}), their velocities are updated using the 1D collision
formula with a coefficient of restitution.

For perfectly elastic collisions (restitution=1.0):
    v1' = ((m1 - m2)*v1 + 2*m2*v2) / (m1 + m2)
    v2' = ((m2 - m1)*v2 + 2*m1*v1) / (m1 + m2)

For general restitution e:
    v1' = (m1*v1 + m2*v2 + m2*e*(v2 - v1)) / (m1 + m2)
    v2' = (m1*v1 + m2*v2 + m1*e*(v1 - v2)) / (m1 + m2)

Target rediscoveries:
- Momentum conservation: sum(m_i * v_i) = const
- Energy conservation (elastic): sum(0.5 * m_i * v_i^2) = const
- Newton's cradle effect (1 ball hits N-1 stationary, equal masses)
- Energy loss vs restitution: KE_ratio = f(e)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class ElasticCollisionSimulation(SimulationEnvironment):
    """1D chain of N particles undergoing elastic/inelastic collisions.

    State vector: [x_1, v_1, x_2, v_2, ..., x_N, v_N] of shape (2*N,).
    Particles are ordered on a line: x_1 < x_2 < ... < x_N initially.

    Parameters:
        n_particles: number of particles (default 5)
        masses: comma-separated mass string or single mass for all
            (default all 1.0)
        restitution: coefficient of restitution, 1.0 = perfectly elastic
            (default 1.0)
        spacing: initial spacing between adjacent particles (default 2.0)
        init_velocity_0: initial velocity of first particle (default 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.n_particles = int(p.get("n_particles", 5))
        self.restitution = p.get("restitution", 1.0)
        self._spacing = p.get("spacing", 2.0)
        self._init_v0 = p.get("init_velocity_0", 1.0)

        # Parse masses: either a single float applied to all particles,
        # or stored from set_masses() after construction.
        m_val = p.get("mass", 1.0)
        self._masses = np.full(self.n_particles, m_val, dtype=np.float64)

        # Internal arrays for positions and velocities
        self._pos = np.zeros(self.n_particles, dtype=np.float64)
        self._vel = np.zeros(self.n_particles, dtype=np.float64)

    def set_masses(self, masses: list[float] | np.ndarray) -> None:
        """Set individual particle masses.

        Args:
            masses: array of length n_particles with each particle's mass.
        """
        masses_arr = np.asarray(masses, dtype=np.float64)
        if len(masses_arr) != self.n_particles:
            raise ValueError(
                f"Expected {self.n_particles} masses, got {len(masses_arr)}"
            )
        self._masses = masses_arr.copy()

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize particles equally spaced with first particle moving.

        Default: particle 0 has velocity init_velocity_0, all others at rest.
        Positions: x_i = i * spacing.
        """
        self._step_count = 0

        # Equally spaced positions
        self._pos = np.arange(self.n_particles, dtype=np.float64) * self._spacing

        # Only first particle is moving initially
        self._vel = np.zeros(self.n_particles, dtype=np.float64)
        self._vel[0] = self._init_v0

        self._state = self._pack_state()
        return self._state.copy()

    def step(self) -> np.ndarray:
        """Advance one timestep: move particles, detect and resolve collisions.

        Uses sub-stepping to handle multiple collisions per timestep
        if particles are close together.
        """
        dt = self.config.dt
        self._advance(dt)
        self._step_count += 1
        self._state = self._pack_state()
        return self._state.copy()

    def observe(self) -> np.ndarray:
        """Return current state [x1, v1, x2, v2, ..., xN, vN]."""
        return self._pack_state()

    def _pack_state(self) -> np.ndarray:
        """Interleave positions and velocities into state vector."""
        state = np.zeros(2 * self.n_particles, dtype=np.float64)
        state[0::2] = self._pos
        state[1::2] = self._vel
        return state

    def _unpack_state(self, state: np.ndarray) -> None:
        """Extract positions and velocities from state vector."""
        self._pos = state[0::2].copy()
        self._vel = state[1::2].copy()

    def _advance(self, dt: float) -> None:
        """Advance positions and handle collisions.

        Uses multiple collision passes per timestep to resolve
        simultaneous or cascading collisions (e.g., Newton's cradle).
        """
        # Move all particles
        self._pos += self._vel * dt

        # Resolve collisions iteratively until no more overlaps
        # Limit iterations to avoid infinite loops
        max_passes = self.n_particles * 2
        for _ in range(max_passes):
            any_collision = False
            for i in range(self.n_particles - 1):
                if self._pos[i] >= self._pos[i + 1]:
                    self._resolve_collision(i, i + 1)
                    any_collision = True
            if not any_collision:
                break

    def _resolve_collision(self, i: int, j: int) -> None:
        """Resolve a 1D collision between particles i and j.

        Uses the general restitution formula:
            v1' = (m1*v1 + m2*v2 + m2*e*(v2 - v1)) / (m1 + m2)
            v2' = (m1*v1 + m2*v2 + m1*e*(v1 - v2)) / (m1 + m2)

        Also separates overlapping particles.
        """
        m1 = self._masses[i]
        m2 = self._masses[j]
        v1 = self._vel[i]
        v2 = self._vel[j]
        e = self.restitution
        M = m1 + m2

        # General 1D collision with restitution
        self._vel[i] = (m1 * v1 + m2 * v2 + m2 * e * (v2 - v1)) / M
        self._vel[j] = (m1 * v1 + m2 * v2 + m1 * e * (v1 - v2)) / M

        # Separate overlapping particles: push apart by half the overlap
        overlap = self._pos[i] - self._pos[j]
        if overlap >= 0:
            half_sep = (overlap + 1e-10) * 0.5
            self._pos[i] -= half_sep
            self._pos[j] += half_sep

    def compute_momentum(self) -> float:
        """Total momentum: sum(m_i * v_i)."""
        return float(np.sum(self._masses * self._vel))

    def compute_kinetic_energy(self) -> float:
        """Total kinetic energy: sum(0.5 * m_i * v_i^2)."""
        return float(0.5 * np.sum(self._masses * self._vel ** 2))

    def detect_collisions(self) -> list[tuple[int, int]]:
        """Return list of (i, j) pairs where particles overlap (x_i >= x_j).

        Only checks adjacent pairs since particles are ordered.
        """
        collisions = []
        for i in range(self.n_particles - 1):
            if self._pos[i] >= self._pos[i + 1]:
                collisions.append((i, i + 1))
        return collisions

    @property
    def positions(self) -> np.ndarray:
        """Current particle positions."""
        return self._pos.copy()

    @property
    def velocities(self) -> np.ndarray:
        """Current particle velocities."""
        return self._vel.copy()

    @property
    def masses(self) -> np.ndarray:
        """Particle masses."""
        return self._masses.copy()

    def set_state(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> None:
        """Manually set particle positions and velocities.

        Args:
            positions: array of length n_particles.
            velocities: array of length n_particles.
        """
        self._pos = np.asarray(positions, dtype=np.float64).copy()
        self._vel = np.asarray(velocities, dtype=np.float64).copy()
        self._state = self._pack_state()

    def newtons_cradle_setup(self, n_moving: int = 1) -> None:
        """Set up a Newton's cradle configuration.

        n_moving particles on the left are launched toward the right.
        The remaining particles are touching (spacing=0.01) and stationary.

        Args:
            n_moving: number of particles to launch from the left.
        """
        # Stationary group: tightly packed on the right
        n_stationary = self.n_particles - n_moving
        pos = np.zeros(self.n_particles, dtype=np.float64)
        vel = np.zeros(self.n_particles, dtype=np.float64)

        # Moving particles start further left
        gap = self._spacing
        for i in range(n_moving):
            pos[i] = i * 0.01 - gap
            vel[i] = self._init_v0

        # Stationary group
        for i in range(n_stationary):
            pos[n_moving + i] = i * 0.01

        self._pos = pos
        self._vel = vel
        self._state = self._pack_state()
