"""Coupled Lorenz systems simulation -- chaos synchronization.

Two Lorenz attractors coupled via diffusive x-coupling:

System 1 (drive):
    dx1/dt = sigma*(y1 - x1)
    dy1/dt = x1*(rho - z1) - y1
    dz1/dt = x1*y1 - beta*z1

System 2 (response):
    dx2/dt = sigma*(y2 - x2) + eps*(x1 - x2)
    dy2/dt = x2*(rho - z2) - y2
    dz2/dt = x2*y2 - beta*z2

Target rediscoveries:
- Critical coupling eps_c for synchronization transition
- Exponential decay of synchronization error for eps > eps_c
- Conditional Lyapunov exponent as a function of eps
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CoupledLorenzSimulation(SimulationEnvironment):
    """Two diffusively-coupled Lorenz systems for chaos synchronization.

    State vector: [x1, y1, z1, x2, y2, z2]

    Parameters:
        sigma: Prandtl number (classic value: 10)
        rho: Rayleigh number (classic value: 28)
        beta: geometric factor (classic value: 8/3)
        eps: coupling strength (default: 5.0, above synchronization threshold)
        x1_0, y1_0, z1_0: initial conditions for system 1
        x2_0, y2_0, z2_0: initial conditions for system 2
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.sigma = p.get("sigma", 10.0)
        self.rho = p.get("rho", 28.0)
        self.beta = p.get("beta", 8.0 / 3.0)
        self.eps = p.get("eps", 5.0)
        self.x1_0 = p.get("x1_0", 1.0)
        self.y1_0 = p.get("y1_0", 1.0)
        self.z1_0 = p.get("z1_0", 1.0)
        self.x2_0 = p.get("x2_0", -5.0)
        self.y2_0 = p.get("y2_0", 5.0)
        self.z2_0 = p.get("z2_0", 25.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize both Lorenz systems with different initial conditions."""
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
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(y + 0.5 * dt * k1)
        k3 = self._derivatives(y + 0.5 * dt * k2)
        k4 = self._derivatives(y + dt * k3)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """Coupled Lorenz equations with diffusive x-coupling on system 2."""
        x1, y1, z1, x2, y2, z2 = state

        # Drive system (standard Lorenz)
        dx1 = self.sigma * (y1 - x1)
        dy1 = x1 * (self.rho - z1) - y1
        dz1 = x1 * y1 - self.beta * z1

        # Response system with coupling term eps*(x1 - x2)
        dx2 = self.sigma * (y2 - x2) + self.eps * (x1 - x2)
        dy2 = x2 * (self.rho - z2) - y2
        dz2 = x2 * y2 - self.beta * z2

        return np.array([dx1, dy1, dz1, dx2, dy2, dz2])

    def sync_error(self) -> float:
        """Compute the instantaneous synchronization error ||sys1 - sys2||."""
        s = self._state
        diff = s[:3] - s[3:]
        return float(np.linalg.norm(diff))

    def sync_error_trajectory(self, n_steps: int = 10000) -> dict[str, np.ndarray]:
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

    def sync_sweep(
        self,
        eps_values: np.ndarray,
        n_steps: int = 10000,
        n_transient: int = 5000,
    ) -> dict[str, np.ndarray]:
        """Measure steady-state synchronization error vs coupling strength.

        For each eps, runs the simulation, discards transient, and computes
        the time-averaged sync error in the remaining steps.

        Args:
            eps_values: Array of coupling strengths to sweep.
            n_steps: Number of measurement steps after transient.
            n_transient: Number of transient steps to discard.

        Returns:
            Dict with 'eps' and 'mean_error' arrays.
        """
        mean_errors = np.empty(len(eps_values))

        for i, eps_val in enumerate(eps_values):
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    "sigma": self.sigma,
                    "rho": self.rho,
                    "beta": self.beta,
                    "eps": eps_val,
                    "x1_0": self.x1_0,
                    "y1_0": self.y1_0,
                    "z1_0": self.z1_0,
                    "x2_0": self.x2_0,
                    "y2_0": self.y2_0,
                    "z2_0": self.z2_0,
                },
            )
            sim = CoupledLorenzSimulation(config)
            sim.reset()

            # Skip transient
            for _ in range(n_transient):
                sim.step()

            # Measure average error
            error_sum = 0.0
            for _ in range(n_steps):
                sim.step()
                error_sum += sim.sync_error()

            mean_errors[i] = error_sum / n_steps

        return {"eps": eps_values, "mean_error": mean_errors}

    def _response_derivatives(
        self, drive: np.ndarray, response: np.ndarray, eps_val: float,
    ) -> np.ndarray:
        """Compute derivatives of the response system given a fixed drive state.

        Args:
            drive: [x1, y1, z1] drive state (treated as constant).
            response: [x2, y2, z2] response state.
            eps_val: Coupling strength.

        Returns:
            Derivatives [dx2, dy2, dz2].
        """
        x1 = drive[0]
        x2, y2, z2 = response
        dx2 = self.sigma * (y2 - x2) + eps_val * (x1 - x2)
        dy2 = x2 * (self.rho - z2) - y2
        dz2 = x2 * y2 - self.beta * z2
        return np.array([dx2, dy2, dz2])

    def _drive_derivatives(self, drive: np.ndarray) -> np.ndarray:
        """Compute derivatives of the drive system (standard Lorenz).

        Args:
            drive: [x1, y1, z1] drive state.

        Returns:
            Derivatives [dx1, dy1, dz1].
        """
        x, y, z = drive
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz])

    def _rk4_drive(self, drive: np.ndarray, dt: float) -> np.ndarray:
        """Advance the drive system one RK4 step."""
        k1 = self._drive_derivatives(drive)
        k2 = self._drive_derivatives(drive + 0.5 * dt * k1)
        k3 = self._drive_derivatives(drive + 0.5 * dt * k2)
        k4 = self._drive_derivatives(drive + dt * k3)
        return drive + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _rk4_response(
        self, drive: np.ndarray, response: np.ndarray,
        eps_val: float, dt: float,
    ) -> np.ndarray:
        """Advance the response system one RK4 step with fixed drive.

        The drive state is held constant during the entire RK4 step to
        properly estimate the conditional (transverse) Lyapunov exponent.
        """
        k1 = self._response_derivatives(drive, response, eps_val)
        k2 = self._response_derivatives(drive, response + 0.5 * dt * k1, eps_val)
        k3 = self._response_derivatives(drive, response + 0.5 * dt * k2, eps_val)
        k4 = self._response_derivatives(drive, response + dt * k3, eps_val)
        return response + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def conditional_lyapunov(
        self,
        eps: float | None = None,
        n_steps: int = 50000,
        n_transient: int = 10000,
    ) -> float:
        """Estimate the conditional Lyapunov exponent for the response system.

        The conditional (or transverse) Lyapunov exponent determines whether
        the synchronization manifold is stable. When it is negative, the
        response system synchronizes to the drive.

        This is estimated by integrating two copies of the response system
        (both driven by the same drive trajectory) from slightly different
        initial conditions. The drive is advanced independently and shared.

        Args:
            eps: Coupling strength (uses self.eps if None).
            n_steps: Number of integration steps for Lyapunov estimation.
            n_transient: Steps to skip before measuring.

        Returns:
            Estimated conditional Lyapunov exponent.
        """
        if eps is None:
            eps = self.eps
        dt = self.config.dt

        # Create a simulation to generate the drive trajectory
        config = SimulationConfig(
            domain=self.config.domain,
            dt=dt,
            n_steps=n_steps,
            parameters={
                "sigma": self.sigma,
                "rho": self.rho,
                "beta": self.beta,
                "eps": eps,
            },
        )
        sim = CoupledLorenzSimulation(config)
        sim.reset()

        # Skip transient on the full coupled system to reach the attractor
        for _ in range(n_transient):
            sim.step()

        # Extract drive and response from the synchronized (or not) state
        drive = sim._state[:3].copy()
        resp_ref = sim._state[3:6].copy()

        # Perturb the response
        perturbation = 1e-8
        resp_pert = resp_ref.copy()
        resp_pert[0] += perturbation

        lyap_sum = 0.0
        n_renorm = 0

        for _ in range(n_steps):
            # Advance drive independently (standard Lorenz, no coupling)
            drive = self._rk4_drive(drive, dt)

            # Advance both response copies with the same drive
            resp_ref = self._rk4_response(drive, resp_ref, eps, dt)
            resp_pert = self._rk4_response(drive, resp_pert, eps, dt)

            # Measure divergence
            diff = resp_pert - resp_ref
            dist = np.linalg.norm(diff)
            if dist > 0:
                lyap_sum += np.log(dist / perturbation)
                n_renorm += 1
                # Renormalize
                resp_pert = resp_ref + perturbation * diff / dist

        if n_renorm == 0:
            return 0.0
        return lyap_sum / (n_renorm * dt)
