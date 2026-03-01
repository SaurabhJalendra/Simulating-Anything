"""Kapitza pendulum simulation -- inverted pendulum stabilized by rapid pivot oscillation.

Target rediscoveries:
- Stability criterion: a^2 * omega^2 > 2 * g * L for inverted position
- Effective potential: V_eff(theta) = -g*L*cos(theta) + (a*omega)^2/(4*L)*sin^2(theta)
- Inverted position (theta=pi) becomes stable when criterion is satisfied
- Normal position (theta=0) is always stable
- Bifurcation as a*omega increases: theta=pi transitions from unstable to stable
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class KapitzaPendulumSimulation(SimulationEnvironment):
    """Kapitza pendulum: pivot oscillates vertically with amplitude a and frequency omega.

    Equation of motion:
        theta'' + gamma*theta' + (g/L)*sin(theta) = (a*omega^2/L)*cos(omega*t)*sin(theta)

    State vector: [theta, theta_dot, t] where theta = angle from downward vertical,
    theta_dot = angular velocity, t = current time.

    The key phenomenon is dynamic stabilization: when a*omega is large enough
    (a^2*omega^2 > 2*g*L), the inverted position theta=pi becomes a stable
    equilibrium -- the Kapitza effect.

    Parameters:
        L: pendulum length (default 1.0)
        g: gravitational acceleration (default 9.81)
        a: pivot oscillation amplitude (default 0.1)
        omega: pivot oscillation frequency (default 50.0)
        gamma: damping coefficient (default 0.1)
        theta_0: initial angle (default 0.1)
        theta_dot_0: initial angular velocity (default 0.0)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.L = p.get("L", 1.0)
        self.g = p.get("g", 9.81)
        self.a = p.get("a", 0.1)
        self.omega = p.get("omega", 50.0)
        self.gamma = p.get("gamma", 0.1)
        self.theta_0 = p.get("theta_0", 0.1)
        self.theta_dot_0 = p.get("theta_dot_0", 0.0)
        self._t = 0.0

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize angle, angular velocity, and time."""
        self._state = np.array(
            [self.theta_0, self.theta_dot_0], dtype=np.float64
        )
        self._step_count = 0
        self._t = 0.0
        return self.observe()

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4 with time-dependent forcing."""
        self._rk4_step()
        self._step_count += 1
        return self.observe()

    def observe(self) -> np.ndarray:
        """Return current state [theta, theta_dot, t]."""
        return np.array(
            [self._state[0], self._state[1], self._t], dtype=np.float64
        )

    def _rk4_step(self) -> None:
        """Fourth-order Runge-Kutta with explicit time dependence."""
        dt = self.config.dt
        y = self._state
        t = self._t

        k1 = self._derivatives(y, t)
        k2 = self._derivatives(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self._derivatives(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self._derivatives(y + dt * k3, t + dt)

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._t += dt

    def _derivatives(self, y: np.ndarray, t: float) -> np.ndarray:
        """Compute dy/dt for the Kapitza pendulum.

        theta'' = -(g/L)*sin(theta) - gamma*theta' + (a*omega^2/L)*cos(omega*t)*sin(theta)

        Args:
            y: State vector [theta, theta_dot].
            t: Current time (needed for oscillating pivot term).

        Returns:
            Derivatives [dtheta/dt, dtheta_dot/dt].
        """
        theta, theta_dot = y
        gravity_term = -(self.g / self.L) * np.sin(theta)
        damping_term = -self.gamma * theta_dot
        forcing_term = (
            (self.a * self.omega**2 / self.L)
            * np.cos(self.omega * t)
            * np.sin(theta)
        )
        dtheta_dt = theta_dot
        dtheta_dot_dt = gravity_term + damping_term + forcing_term
        return np.array([dtheta_dt, dtheta_dot_dt])

    def stability_criterion(self) -> float:
        """Compute the Kapitza stability parameter: a^2*omega^2 / (2*g*L).

        When this value > 1, the inverted position (theta=pi) is stable.

        Returns:
            Stability parameter (>1 means inverted position is stable).
        """
        return (self.a**2 * self.omega**2) / (2 * self.g * self.L)

    def effective_potential(self, theta_values: np.ndarray) -> np.ndarray:
        """Compute the time-averaged effective potential V_eff(theta).

        V_eff(theta) = -g*L*cos(theta) + (a*omega)^2/(4*L) * sin^2(theta)

        The first term is the gravitational potential, the second arises from
        time-averaging the rapid oscillation. When the second term dominates,
        a local minimum appears at theta=pi.

        Args:
            theta_values: Array of angles at which to evaluate V_eff.

        Returns:
            Array of effective potential values.
        """
        V_grav = -self.g * self.L * np.cos(theta_values)
        V_vib = (self.a * self.omega) ** 2 / (4 * self.L) * np.sin(theta_values) ** 2
        return V_grav + V_vib

    def inverted_stability_sweep(
        self,
        a_omega_values: np.ndarray,
        n_steps: int = 100000,
        perturbation: float = 0.05,
    ) -> dict[str, np.ndarray]:
        """Sweep a*omega values and test whether the inverted position is stable.

        For each a*omega value, starts near theta=pi with a small perturbation,
        runs the simulation, and checks if it stays near pi.

        Args:
            a_omega_values: Array of a*omega products to test.
            n_steps: Number of integration steps per test.
            perturbation: Initial perturbation from pi (radians).

        Returns:
            Dict with a_omega, is_stable (bool), final_deviation arrays.
        """
        is_stable = []
        final_deviations = []

        for a_omega in a_omega_values:
            # Compute a and omega such that a*omega = a_omega, keeping omega high
            omega_test = max(50.0, a_omega / 0.5)
            a_test = a_omega / omega_test

            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    "L": self.L,
                    "g": self.g,
                    "a": a_test,
                    "omega": omega_test,
                    "gamma": self.gamma,
                    "theta_0": np.pi - perturbation,
                    "theta_dot_0": 0.0,
                },
            )
            sim = KapitzaPendulumSimulation(config)
            sim.reset()

            for _ in range(n_steps):
                sim.step()

            # Check deviation from pi (use slow envelope, not fast oscillation)
            # Average over several fast periods to get the slow dynamics
            avg_period_steps = max(1, int(2 * np.pi / (omega_test * self.config.dt)))
            thetas = []
            for _ in range(min(avg_period_steps * 5, 1000)):
                sim.step()
                thetas.append(sim._state[0])

            mean_theta = np.mean(thetas)
            deviation = abs(mean_theta - np.pi)
            # Wrap deviation to [0, pi]
            deviation = min(deviation, 2 * np.pi - deviation)

            stable = deviation < 0.5
            is_stable.append(stable)
            final_deviations.append(deviation)

        return {
            "a_omega": a_omega_values,
            "is_stable": np.array(is_stable),
            "final_deviation": np.array(final_deviations),
        }

    def compute_slow_dynamics(self, n_steps: int = 100000) -> dict[str, np.ndarray]:
        """Compute the slow (averaged) dynamics by filtering out fast oscillation.

        Runs the simulation and extracts the slow envelope by averaging
        over windows of the fast oscillation period.

        Args:
            n_steps: Total number of integration steps.

        Returns:
            Dict with time, theta_slow, theta_dot_slow arrays.
        """
        dt = self.config.dt
        fast_period = 2 * np.pi / self.omega
        window_steps = max(1, int(fast_period / dt))

        all_theta = []
        all_time = []

        for _ in range(n_steps):
            self.step()
            all_theta.append(self._state[0])
            all_time.append(self._t)

        all_theta = np.array(all_theta)
        all_time = np.array(all_time)

        # Moving average with window = fast period
        if window_steps > 1 and len(all_theta) > window_steps:
            kernel = np.ones(window_steps) / window_steps
            theta_slow = np.convolve(all_theta, kernel, mode="valid")
            time_slow = all_time[window_steps // 2: window_steps // 2 + len(theta_slow)]
            # Numerical derivative for slow theta_dot
            theta_dot_slow = np.gradient(theta_slow, time_slow)
        else:
            theta_slow = all_theta
            time_slow = all_time
            theta_dot_slow = np.gradient(theta_slow, time_slow)

        return {
            "time": time_slow,
            "theta_slow": theta_slow,
            "theta_dot_slow": theta_dot_slow,
        }

    def check_inverted_stability(self, n_steps: int = 200000) -> dict[str, float]:
        """Start near theta=pi and check if the pendulum stays there.

        This directly tests the Kapitza effect by initializing near the
        inverted position and measuring how far it drifts.

        Args:
            n_steps: Number of integration steps to run.

        Returns:
            Dict with initial and final deviation from pi, and stability flag.
        """
        # Start near inverted position
        self._state = np.array([np.pi - 0.05, 0.0], dtype=np.float64)
        self._t = 0.0
        self._step_count = 0

        initial_deviation = abs(self._state[0] - np.pi)

        # Run simulation
        for _ in range(n_steps):
            self._rk4_step()
            self._step_count += 1
            self._t += 0.0  # _t already updated in _rk4_step

        # Average over a few fast periods to get slow angle
        fast_period = 2 * np.pi / self.omega
        avg_steps = max(1, int(fast_period / self.config.dt))
        thetas = []
        for _ in range(avg_steps * 3):
            self._rk4_step()
            self._step_count += 1
            thetas.append(self._state[0])

        mean_theta = np.mean(thetas)
        final_deviation = abs(mean_theta - np.pi)
        final_deviation = min(final_deviation, 2 * np.pi - final_deviation)

        return {
            "initial_deviation": initial_deviation,
            "final_deviation": final_deviation,
            "is_stable": final_deviation < 0.5,
            "stability_parameter": self.stability_criterion(),
        }

    @property
    def mechanical_energy(self) -> float:
        """Compute the instantaneous mechanical energy (not conserved with forcing).

        E = 0.5*L^2*theta_dot^2 - g*L*cos(theta) + a*omega^2*L*cos(omega*t)*cos(theta)

        The last term is from the moving pivot, but the time-averaged energy
        is approximately:
        <E> ~ 0.5*L^2*theta_dot^2 + V_eff(theta)
        """
        theta, theta_dot = self._state
        KE = 0.5 * self.L**2 * theta_dot**2
        PE_grav = -self.g * self.L * np.cos(theta)
        return float(KE + PE_grav)
