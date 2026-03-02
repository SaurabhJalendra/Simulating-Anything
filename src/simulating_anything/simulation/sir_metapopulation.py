"""SIR Metapopulation simulation -- multi-patch SIR with migration.

Standard SIR dynamics on each patch coupled by symmetric migration between
all patch pairs. Demonstrates epidemic wave propagation, migration-induced
synchronization, and spatial spread of disease.

Equations (per patch i):
    dS_i/dt = -beta * S_i * I_i / N_i + m * (mean(S) - S_i)
    dI_i/dt =  beta * S_i * I_i / N_i - gamma * I_i + m * (mean(I) - I_i)
    dR_i/dt =  gamma * I_i + m * (mean(R) - R_i)

where mean(X) = sum_j(X_j) / n_patches and m is the migration rate.

Target rediscoveries:
- Migration-induced synchronization of epidemic peaks across patches
- Epidemic wave propagation delay between patches
- Total epidemic size converges to single-patch SIR as m -> inf
- Zero migration isolates patches (only seeded patch has outbreak)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class SIRMetapopulationSimulation(SimulationEnvironment):
    """SIR epidemic on multiple patches with symmetric migration coupling.

    State vector: [S_0, I_0, R_0, S_1, I_1, R_1, ..., S_{n-1}, I_{n-1}, R_{n-1}]
    Length = 3 * n_patches.

    Parameters:
        n_patches: number of spatial patches (default: 5)
        beta: transmission rate within each patch (default: 0.5)
        gamma: recovery rate (default: 0.1)
        m: migration rate between all patch pairs (default: 0.01)
        N_per_patch: initial population per patch (default: 1000.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.n_patches = int(p.get("n_patches", 5))
        self.beta = p.get("beta", 0.5)
        self.gamma = p.get("gamma", 0.1)
        self.m = p.get("m", 0.01)
        self.N_per_patch = p.get("N_per_patch", 1000.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize compartments: patch 0 seeded with 10 infected."""
        n = self.n_patches
        self._state = np.zeros(3 * n, dtype=np.float64)
        for i in range(n):
            s_idx = 3 * i
            if i == 0:
                # Seed patch 0 with infected individuals
                self._state[s_idx] = self.N_per_patch - 10.0     # S_0
                self._state[s_idx + 1] = 10.0                     # I_0
            else:
                self._state[s_idx] = self.N_per_patch             # S_i
                self._state[s_idx + 1] = 0.0                      # I_i
            self._state[s_idx + 2] = 0.0                          # R_i
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state vector."""
        return self._state

    def _rk4_step(self) -> None:
        """Classical Runge-Kutta 4th order step."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Non-negativity clamp
        self._state = np.maximum(self._state, 0.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Compute derivatives for all patches.

        Within-patch SIR dynamics plus mean-field migration coupling.
        Migration: m * (mean(X) - X_i) for each compartment X.
        """
        dy = np.zeros_like(y)

        # Extract compartments for all patches
        S = y[0::3]   # [S_0, S_1, ..., S_{n-1}]
        Ip = y[1::3]  # [I_0, I_1, ..., I_{n-1}] (Ip to avoid ambiguous name)
        R = y[2::3]   # [R_0, R_1, ..., R_{n-1}]

        # Per-patch population
        N = S + Ip + R
        # Avoid division by zero
        N_safe = np.maximum(N, 1e-12)

        # Within-patch SIR dynamics
        force_of_infection = self.beta * S * Ip / N_safe
        dS_local = -force_of_infection
        dI_local = force_of_infection - self.gamma * Ip
        dR_local = self.gamma * Ip

        # Migration: mean-field coupling m * (mean(X) - X_i)
        S_mean = np.mean(S)
        I_mean = np.mean(Ip)
        R_mean = np.mean(R)

        dS_migration = self.m * (S_mean - S)
        dI_migration = self.m * (I_mean - Ip)
        dR_migration = self.m * (R_mean - R)

        # Combine
        dy[0::3] = dS_local + dS_migration
        dy[1::3] = dI_local + dI_migration
        dy[2::3] = dR_local + dR_migration

        return dy

    def get_patch_state(self, patch_idx: int) -> np.ndarray:
        """Return [S, I, R] for a specific patch."""
        base = 3 * patch_idx
        return self._state[base:base + 3].copy()

    def get_all_infected(self) -> np.ndarray:
        """Return infected counts for all patches."""
        return self._state[1::3].copy()

    def get_all_susceptible(self) -> np.ndarray:
        """Return susceptible counts for all patches."""
        return self._state[0::3].copy()

    def get_all_recovered(self) -> np.ndarray:
        """Return recovered counts for all patches."""
        return self._state[2::3].copy()

    def total_infected(self) -> float:
        """Total infected across all patches."""
        return float(np.sum(self._state[1::3]))

    def total_population(self) -> float:
        """Total population across all patches."""
        return float(np.sum(self._state))

    def patch_population(self, patch_idx: int) -> float:
        """Population of a specific patch: S_i + I_i + R_i."""
        base = 3 * patch_idx
        return float(self._state[base] + self._state[base + 1] + self._state[base + 2])

    @property
    def R0(self) -> float:
        """Basic reproduction number (single-patch): R0 = beta / gamma."""
        return self.beta / self.gamma

    def peak_infected_per_patch(
        self,
        n_steps: int = 2000,
    ) -> dict[str, np.ndarray]:
        """Run simulation and record peak infected and time-to-peak per patch.

        Returns dict with 'peak_I' and 'peak_time' arrays of shape (n_patches,).
        """
        self.reset()
        dt = self.config.dt
        peak_I = np.zeros(self.n_patches)
        peak_time = np.zeros(self.n_patches)

        for step in range(1, n_steps + 1):
            self.step()
            I_all = self.get_all_infected()
            for i in range(self.n_patches):
                if I_all[i] > peak_I[i]:
                    peak_I[i] = I_all[i]
                    peak_time[i] = step * dt

        return {"peak_I": peak_I, "peak_time": peak_time}
