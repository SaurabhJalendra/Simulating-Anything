"""Coupled Rossler systems simulation -- phase and complete synchronization.

Two Rossler oscillators coupled symmetrically via diffusive x-coupling:

System 1:
    dx1/dt = -(y1 + z1) + epsilon*(x2 - x1)
    dy1/dt = x1 + a*y1
    dz1/dt = b + z1*(x1 - c)

System 2:
    dx2/dt = -(y2 + z2) + epsilon*(x1 - x2)
    dy2/dt = x2 + a*y2
    dz2/dt = b + z2*(x2 - c)

Target rediscoveries:
- Synchronization transition as epsilon increases
- Phase synchronization before complete synchronization
- Conditional Lyapunov exponent determines sync stability
- Independent chaotic Rossler attractors at epsilon=0
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CoupledRosslerSimulation(SimulationEnvironment):
    """Two diffusively-coupled Rossler systems for synchronization studies.

    State vector: [x1, y1, z1, x2, y2, z2]

    Parameters:
        a: Rossler parameter controlling oscillation shape (classic: 0.2)
        b: Rossler parameter controlling z-dynamics (classic: 0.2)
        c: Rossler parameter controlling chaos onset (classic: 5.7)
        epsilon: coupling strength between oscillators (default: 0.05)
        x1_0, y1_0, z1_0: initial conditions for system 1
        x2_0, y2_0, z2_0: initial conditions for system 2
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.a = p.get("a", 0.2)
        self.b = p.get("b", 0.2)
        self.c = p.get("c", 5.7)
        self.epsilon = p.get("epsilon", 0.05)
        self.x1_0 = p.get("x1_0", 1.0)
        self.y1_0 = p.get("y1_0", 0.0)
        self.z1_0 = p.get("z1_0", 0.0)
        self.x2_0 = p.get("x2_0", 0.5)
        self.y2_0 = p.get("y2_0", 0.5)
        self.z2_0 = p.get("z2_0", 0.5)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize both Rossler systems with offset initial conditions."""
        self._state = np.array(
            [self.x1_0, self.y1_0, self.z1_0,
             self.x2_0, self.y2_0, self.z2_0],
            dtype=np.float64,
        )
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        self._rk4_step()
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state [x1, y1, z1, x2, y2, z2]."""
        return self._state

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta integration."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Coupled Rossler equations with symmetric diffusive x-coupling.

        System 1:
            dx1/dt = -(y1 + z1) + epsilon*(x2 - x1)
            dy1/dt = x1 + a*y1
            dz1/dt = b + z1*(x1 - c)

        System 2:
            dx2/dt = -(y2 + z2) + epsilon*(x1 - x2)
            dy2/dt = x2 + a*y2
            dz2/dt = b + z2*(x2 - c)
        """
        x1, y1, z1, x2, y2, z2 = state

        dx1 = -(y1 + z1) + self.epsilon * (x2 - x1)
        dy1 = x1 + self.a * y1
        dz1 = self.b + z1 * (x1 - self.c)

        dx2 = -(y2 + z2) + self.epsilon * (x1 - x2)
        dy2 = x2 + self.a * y2
        dz2 = self.b + z2 * (x2 - self.c)

        return np.array([dx1, dy1, dz1, dx2, dy2, dz2])

    def sync_error(self) -> float:
        """Compute instantaneous synchronization error ||sys1 - sys2||."""
        s = self._state
        diff = s[:3] - s[3:]
        return float(np.linalg.norm(diff))

    def compute_sync_error(self, states: np.ndarray) -> float:
        """Compute mean synchronization error over a trajectory.

        Args:
            states: Array of shape (n_steps, 6) with columns
                [x1, y1, z1, x2, y2, z2].

        Returns:
            Mean Euclidean distance between system 1 and system 2 states.
        """
        diff = states[:, :3] - states[:, 3:]
        norms = np.linalg.norm(diff, axis=1)
        return float(np.mean(norms))

    def sync_error_trajectory(
        self, n_steps: int = 10000,
    ) -> dict[str, np.ndarray]:
        """Run simulation and record synchronization error vs time.

        Args:
            n_steps: Number of integration steps.

        Returns:
            Dict with 'time' and 'error' arrays.
        """
        dt = self.config.dt
        errors = np.empty(n_steps + 1)
        times = np.empty(n_steps + 1)

        errors[0] = self.sync_error()
        times[0] = 0.0

        for i in range(1, n_steps + 1):
            self.step()
            errors[i] = self.sync_error()
            times[i] = i * dt

        return {"time": times, "error": errors}

    def sync_transition_data(
        self,
        epsilon_range: tuple[float, float] = (0.0, 0.3),
        n_eps: int = 30,
        n_transient: int = 5000,
        n_measure: int = 2000,
    ) -> dict[str, np.ndarray]:
        """Sweep epsilon and measure steady-state sync error at each value.

        For each epsilon, creates a fresh simulation from the same initial
        conditions, discards transient, and computes the time-averaged
        synchronization error.

        Args:
            epsilon_range: Tuple of (eps_min, eps_max).
            n_eps: Number of epsilon values to sample.
            n_transient: Steps to discard as transient.
            n_measure: Steps over which to measure average sync error.

        Returns:
            Dict with 'epsilon' and 'mean_error' arrays.
        """
        eps_values = np.linspace(epsilon_range[0], epsilon_range[1], n_eps)
        mean_errors = np.empty(n_eps)

        for i, eps_val in enumerate(eps_values):
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_measure,
                parameters={
                    "a": self.a,
                    "b": self.b,
                    "c": self.c,
                    "epsilon": float(eps_val),
                    "x1_0": self.x1_0,
                    "y1_0": self.y1_0,
                    "z1_0": self.z1_0,
                    "x2_0": self.x2_0,
                    "y2_0": self.y2_0,
                    "z2_0": self.z2_0,
                },
            )
            sim = CoupledRosslerSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Measure average error
            error_sum = 0.0
            for _ in range(n_measure):
                sim.step()
                error_sum += sim.sync_error()

            mean_errors[i] = error_sum / n_measure

        return {"epsilon": eps_values, "mean_error": mean_errors}

    def compute_lyapunov(
        self,
        n_transient: int = 2000,
        n_compute: int = 5000,
    ) -> float:
        """Compute max Lyapunov exponent of the full 6D coupled system.

        Uses the Wolf et al. (1985) method: track two nearby trajectories,
        periodically renormalize the perturbation vector.

        Args:
            n_transient: Steps to skip before measurement.
            n_compute: Steps over which to accumulate Lyapunov estimate.

        Returns:
            Estimated largest Lyapunov exponent.
        """
        dt = self.config.dt
        eps = 1e-8

        # Skip transient on main trajectory
        for _ in range(n_transient):
            self.step()

        state1 = self._state.copy()
        state2 = state1.copy()
        state2[0] += eps

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_compute):
            # Advance state1 with RK4
            k1 = self._derivatives(state1)
            k2 = self._derivatives(state1 + 0.5 * dt * k1)
            k3 = self._derivatives(state1 + 0.5 * dt * k2)
            k4 = self._derivatives(state1 + dt * k3)
            state1 = state1 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Advance state2 with RK4
            k1 = self._derivatives(state2)
            k2 = self._derivatives(state2 + 0.5 * dt * k1)
            k3 = self._derivatives(state2 + 0.5 * dt * k2)
            k4 = self._derivatives(state2 + dt * k3)
            state2 = state2 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Measure divergence and renormalize
            diff = state2 - state1
            dist = np.linalg.norm(diff)
            if dist > 0:
                lyap_sum += np.log(dist / eps)
                n_renorm += 1
                state2 = state1 + eps * diff / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)

    def conditional_lyapunov(
        self,
        epsilon: float | None = None,
        n_steps: int = 30000,
        n_transient: int = 5000,
    ) -> float:
        """Estimate the conditional Lyapunov exponent for synchronization.

        The conditional (transverse) Lyapunov exponent determines whether the
        synchronization manifold x1=x2, y1=y2, z1=z2 is stable. When negative,
        perturbations transverse to the manifold decay and sync is stable.

        Uses two copies of system 2 driven by the same system 1 trajectory,
        started from slightly different initial conditions.

        Args:
            epsilon: Coupling strength (uses self.epsilon if None).
            n_steps: Integration steps for Lyapunov estimation.
            n_transient: Steps to skip before measuring.

        Returns:
            Estimated conditional Lyapunov exponent.
        """
        if epsilon is None:
            epsilon = self.epsilon
        dt = self.config.dt

        # Create a simulation to generate the drive trajectory
        config = SimulationConfig(
            domain=self.config.domain,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "epsilon": epsilon,
                "x1_0": self.x1_0,
                "y1_0": self.y1_0,
                "z1_0": self.z1_0,
                "x2_0": self.x2_0,
                "y2_0": self.y2_0,
                "z2_0": self.z2_0,
            },
        )
        sim = CoupledRosslerSimulation(config)
        sim.reset()

        # Skip transient on the full coupled system
        for _ in range(n_transient):
            sim.step()

        # Extract system 1 state as drive, system 2 as reference response
        drive = sim._state[:3].copy()
        resp_ref = sim._state[3:].copy()

        # Perturb the response
        perturbation = 1e-8
        resp_pert = resp_ref.copy()
        resp_pert[0] += perturbation

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance drive (uncoupled Rossler)
            drive = self._rk4_single(drive, dt)

            # Advance both response copies with same drive
            resp_ref = self._rk4_response(drive, resp_ref, epsilon, dt)
            resp_pert = self._rk4_response(drive, resp_pert, epsilon, dt)

            # Measure divergence
            diff = resp_pert - resp_ref
            dist = np.linalg.norm(diff)
            if dist > 0:
                lyap_sum += np.log(dist / perturbation)
                n_renorm += 1
                resp_pert = resp_ref + perturbation * diff / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)

    def _rk4_single(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Advance a single uncoupled Rossler system one RK4 step.

        Args:
            state: [x, y, z] state vector.
            dt: Timestep.

        Returns:
            New state after one RK4 step.
        """
        k1 = self._single_derivatives(state)
        k2 = self._single_derivatives(state + 0.5 * dt * k1)
        k3 = self._single_derivatives(state + 0.5 * dt * k2)
        k4 = self._single_derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _single_derivatives(self, state: np.ndarray) -> np.ndarray:
        """Compute derivatives for a single uncoupled Rossler system.

        Args:
            state: [x, y, z] state vector.

        Returns:
            Derivatives [dx, dy, dz].
        """
        x, y, z = state
        dx = -(y + z)
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return np.array([dx, dy, dz])

    def _rk4_response(
        self,
        drive: np.ndarray,
        response: np.ndarray,
        eps_val: float,
        dt: float,
    ) -> np.ndarray:
        """Advance the response system one RK4 step with fixed drive.

        The drive state is held constant during the RK4 step.

        Args:
            drive: [x1, y1, z1] drive state (treated as constant).
            response: [x2, y2, z2] response state.
            eps_val: Coupling strength.
            dt: Timestep.

        Returns:
            Updated response state.
        """
        k1 = self._response_derivatives(drive, response, eps_val)
        k2 = self._response_derivatives(drive, response + 0.5 * dt * k1, eps_val)
        k3 = self._response_derivatives(drive, response + 0.5 * dt * k2, eps_val)
        k4 = self._response_derivatives(drive, response + dt * k3, eps_val)
        return response + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _response_derivatives(
        self,
        drive: np.ndarray,
        response: np.ndarray,
        eps_val: float,
    ) -> np.ndarray:
        """Compute derivatives of the response system given a fixed drive.

        Args:
            drive: [x1, y1, z1] drive state (constant).
            response: [x2, y2, z2] response state.
            eps_val: Coupling strength.

        Returns:
            Derivatives [dx2, dy2, dz2].
        """
        x1 = drive[0]
        x2, y2, z2 = response
        dx2 = -(y2 + z2) + eps_val * (x1 - x2)
        dy2 = x2 + self.a * y2
        dz2 = self.b + z2 * (x2 - self.c)
        return np.array([dx2, dy2, dz2])
