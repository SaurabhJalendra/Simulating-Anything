"""Lennard-Jones molecular dynamics simulation.

Classical molecular dynamics with pairwise Lennard-Jones potential.
This serves as the JAX-MD equivalent domain -- uses the same physics
but implemented in pure numpy for CPU compatibility.

Discovery targets:
- Equation of state P(rho, T)
- Phase transitions: gas-liquid-solid
- Radial distribution function g(r)
- Diffusion coefficient D(T) from mean square displacement
- Melting temperature estimation
- Lindemann criterion for melting (delta_L ~ 0.1)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class LennardJonesSimulation(SimulationEnvironment):
    """Lennard-Jones molecular dynamics in 2D with periodic boundaries.

    N particles interact via LJ potential:
        V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]

    Uses velocity Verlet integrator for symplectic integration.

    State: flattened [x1, y1, ..., xN, yN, vx1, vy1, ..., vxN, vyN]
           of shape (4*N,).

    Parameters:
        N: number of particles (default: 64)
        L: box side length (default: 10.0)
        T: target temperature (default: 1.0, LJ units)
        epsilon: LJ well depth (default: 1.0)
        sigma: LJ diameter (default: 1.0)
        r_cut: cutoff radius (default: 2.5 * sigma)
        m: particle mass (default: 1.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N = int(p.get("N", 64))
        self.L = p.get("L", 10.0)
        self.T_target = p.get("T", 1.0)
        self.eps = p.get("epsilon", 1.0)
        self.sigma = p.get("sigma", 1.0)
        self.r_cut = p.get("r_cut", 2.5) * self.sigma
        self.m = p.get("m", 1.0)
        self.dt = config.dt

        self._rng = np.random.RandomState(config.seed)

        # Pre-compute LJ cutoff shift
        sr6 = (self.sigma / self.r_cut) ** 6
        self._v_shift = 4.0 * self.eps * (sr6 ** 2 - sr6)

        # State arrays
        self._pos = np.zeros((self.N, 2))
        self._vel = np.zeros((self.N, 2))
        self._forces = np.zeros((self.N, 2))

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._step_count = 0

        # Place particles on a grid
        n_side = int(np.ceil(np.sqrt(self.N)))
        spacing = self.L / n_side
        idx = 0
        for i in range(n_side):
            for j in range(n_side):
                if idx >= self.N:
                    break
                self._pos[idx] = [(i + 0.5) * spacing, (j + 0.5) * spacing]
                idx += 1

        # Maxwell-Boltzmann velocities
        speed_scale = np.sqrt(self.T_target / self.m)
        self._vel = self._rng.randn(self.N, 2) * speed_scale

        # Remove center of mass velocity
        self._vel -= np.mean(self._vel, axis=0)

        # Compute initial forces
        self._forces = self._compute_forces()
        self._state = self._build_state()
        return self._state.copy()

    def step(self) -> np.ndarray:
        """Velocity Verlet integration step."""
        dt = self.dt

        # Half-step velocity
        self._vel += 0.5 * dt * self._forces / self.m

        # Full-step position with PBC
        self._pos += dt * self._vel
        self._pos %= self.L

        # Compute new forces
        self._forces = self._compute_forces()

        # Half-step velocity
        self._vel += 0.5 * dt * self._forces / self.m

        self._step_count += 1
        self._state = self._build_state()
        return self._state.copy()

    def observe(self) -> np.ndarray:
        return self._state.copy()

    def _build_state(self) -> np.ndarray:
        return np.concatenate([self._pos.ravel(), self._vel.ravel()])

    def _compute_forces(self) -> np.ndarray:
        """Compute LJ forces with minimum image convention."""
        forces = np.zeros((self.N, 2))
        half_L = self.L / 2.0
        r_cut2 = self.r_cut ** 2

        for i in range(self.N):
            for j in range(i + 1, self.N):
                dr = self._pos[j] - self._pos[i]
                # Minimum image convention
                dr -= self.L * np.round(dr / self.L)
                r2 = np.dot(dr, dr)

                if r2 < r_cut2 and r2 > 1e-10:
                    sr2 = (self.sigma ** 2) / r2
                    sr6 = sr2 ** 3
                    sr12 = sr6 ** 2
                    f_mag = 24.0 * self.eps * (2.0 * sr12 - sr6) / r2
                    f = f_mag * dr
                    forces[i] += f
                    forces[j] -= f

        return forces

    def kinetic_energy(self) -> float:
        """Total kinetic energy."""
        return 0.5 * self.m * np.sum(self._vel ** 2)

    def potential_energy(self) -> float:
        """Total LJ potential energy."""
        pe = 0.0
        half_L = self.L / 2.0
        r_cut2 = self.r_cut ** 2

        for i in range(self.N):
            for j in range(i + 1, self.N):
                dr = self._pos[j] - self._pos[i]
                dr -= self.L * np.round(dr / self.L)
                r2 = np.dot(dr, dr)

                if r2 < r_cut2 and r2 > 1e-10:
                    sr6 = (self.sigma ** 2 / r2) ** 3
                    pe += 4.0 * self.eps * (sr6 ** 2 - sr6) - self._v_shift

        return pe

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()

    def temperature(self) -> float:
        """Instantaneous temperature from kinetic energy (2D)."""
        # <KE> = N_dof * k_B * T / 2, N_dof = 2*(N-1) in 2D with COM constraint
        n_dof = 2 * (self.N - 1)
        if n_dof <= 0:
            return 0.0
        return 2.0 * self.kinetic_energy() / n_dof

    def pressure(self) -> float:
        """Instantaneous pressure via virial theorem (2D)."""
        T = self.temperature()
        rho = self.N / self.L ** 2

        # Virial contribution
        virial = 0.0
        half_L = self.L / 2.0
        r_cut2 = self.r_cut ** 2

        for i in range(self.N):
            for j in range(i + 1, self.N):
                dr = self._pos[j] - self._pos[i]
                dr -= self.L * np.round(dr / self.L)
                r2 = np.dot(dr, dr)

                if r2 < r_cut2 and r2 > 1e-10:
                    sr2 = (self.sigma ** 2) / r2
                    sr6 = sr2 ** 3
                    f_mag = 24.0 * self.eps * (2.0 * sr6 ** 2 - sr6) / r2
                    virial += f_mag * r2

        return rho * T + virial / (2.0 * self.L ** 2)

    def radial_distribution(self, n_bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Compute radial distribution function g(r)."""
        r_max = self.L / 2.0
        dr = r_max / n_bins
        bins = np.arange(n_bins) * dr + dr / 2.0
        hist = np.zeros(n_bins)

        for i in range(self.N):
            for j in range(i + 1, self.N):
                d = self._pos[j] - self._pos[i]
                d -= self.L * np.round(d / self.L)
                r = np.sqrt(np.dot(d, d))
                b = int(r / dr)
                if 0 <= b < n_bins:
                    hist[b] += 2  # Count both i-j and j-i

        # Normalize: g(r) = hist / (N * rho * 2*pi*r*dr)
        rho = self.N / self.L ** 2
        for i in range(n_bins):
            r = bins[i]
            shell_area = 2 * np.pi * r * dr
            hist[i] /= self.N * rho * shell_area

        return bins, hist

    def mean_square_displacement(self, positions_history: np.ndarray) -> np.ndarray:
        """Compute MSD from position history array of shape (T, N, 2)."""
        n_frames = len(positions_history)
        msd = np.zeros(n_frames)
        ref = positions_history[0]
        for t in range(n_frames):
            diff = positions_history[t] - ref
            msd[t] = np.mean(np.sum(diff ** 2, axis=1))
        return msd
