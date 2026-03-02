"""Age-structured predator-prey simulation (Leslie-Gower type).

Four-compartment ODE model with juvenile/adult age classes for both prey
and predator species:

    dN_j/dt = b_N * N_a - mu_N * N_j - m_j_N * N_j
    dN_a/dt = mu_N * N_j - d_N * N_a - a * N_a * P_a / (1 + h * N_a)
    dP_j/dt = e * a * N_a * P_a / (1 + h * N_a) - mu_P * P_j - m_j_P * P_j
    dP_a/dt = mu_P * P_j - d_P * P_a

where:
    N_j, N_a = juvenile and adult prey
    P_j, P_a = juvenile and adult predator
    b_N  = prey birth rate (from adults)
    mu_N = prey maturation rate (juvenile -> adult)
    m_j_N = juvenile prey mortality
    d_N  = adult prey natural death rate
    a    = predation rate (Holling Type II attack rate)
    h    = handling time
    e    = conversion efficiency
    mu_P = predator maturation rate
    m_j_P = juvenile predator mortality
    d_P  = adult predator death rate

Target rediscoveries:
- Equilibrium age ratios N_j*/N_a* and P_j*/P_a*
- Maturation rate effects on stability
- Functional response saturation
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class AgeStructuredSimulation(SimulationEnvironment):
    """Age-structured predator-prey with juvenile/adult classes.

    State vector: [N_j, N_a, P_j, P_a]
        N_j = juvenile prey, N_a = adult prey
        P_j = juvenile predator, P_a = adult predator

    Uses Holling Type II functional response for predation on adult prey.
    RK4 integration with non-negativity clamping.
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        # Prey parameters
        self.b_N = p.get("b_N", 2.0)       # Birth rate from adult prey
        self.mu_N = p.get("mu_N", 0.5)     # Prey maturation rate (juv -> adult)
        self.m_j_N = p.get("m_j_N", 0.1)   # Juvenile prey mortality
        self.d_N = p.get("d_N", 0.1)       # Adult prey natural death rate
        # Predation parameters
        self.a = p.get("a", 0.02)           # Attack rate
        self.h = p.get("h", 0.01)           # Handling time
        self.e = p.get("e", 0.5)            # Conversion efficiency
        # Predator parameters
        self.mu_P = p.get("mu_P", 0.3)     # Predator maturation rate
        self.m_j_P = p.get("m_j_P", 0.15)  # Juvenile predator mortality
        self.d_P = p.get("d_P", 0.2)       # Adult predator death rate
        # Initial conditions
        self.N_j_0 = p.get("N_j_0", 30.0)
        self.N_a_0 = p.get("N_a_0", 50.0)
        self.P_j_0 = p.get("P_j_0", 5.0)
        self.P_a_0 = p.get("P_a_0", 10.0)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize populations [N_j, N_a, P_j, P_a]."""
        self._state = np.array(
            [self.N_j_0, self.N_a_0, self.P_j_0, self.P_a_0],
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
        """Return current populations [N_j, N_a, P_j, P_a]."""
        return self._state

    def _rk4_step(self) -> None:
        """Classical Runge-Kutta 4th order step with non-negativity clamp."""
        dt = self.config.dt
        y = self._state

        k1 = self._derivatives(y)
        k2 = self._derivatives(np.maximum(y + 0.5 * dt * k1, 0.0))
        k3 = self._derivatives(np.maximum(y + 0.5 * dt * k2, 0.0))
        k4 = self._derivatives(np.maximum(y + dt * k3, 0.0))

        self._state = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state = np.maximum(self._state, 0.0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """Age-structured predator-prey right-hand side.

        dN_j/dt = b_N * N_a - mu_N * N_j - m_j_N * N_j
        dN_a/dt = mu_N * N_j - d_N * N_a - a * N_a * P_a / (1 + h * N_a)
        dP_j/dt = e * a * N_a * P_a / (1 + h * N_a) - mu_P * P_j - m_j_P * P_j
        dP_a/dt = mu_P * P_j - d_P * P_a
        """
        N_j, N_a, P_j, P_a = y

        # Holling Type II functional response on adult prey
        functional_resp = self.a * N_a / (1.0 + self.h * N_a)

        dN_j = self.b_N * N_a - self.mu_N * N_j - self.m_j_N * N_j
        dN_a = self.mu_N * N_j - self.d_N * N_a - functional_resp * P_a
        dP_j = (
            self.e * functional_resp * P_a
            - self.mu_P * P_j
            - self.m_j_P * P_j
        )
        dP_a = self.mu_P * P_j - self.d_P * P_a

        return np.array([dN_j, dN_a, dP_j, dP_a])

    def functional_response(self, N_a: float) -> float:
        """Holling Type II functional response at given adult prey density.

        f(N_a) = a * N_a / (1 + h * N_a)
        """
        return self.a * N_a / (1.0 + self.h * N_a)

    @property
    def total_prey(self) -> float:
        """Total prey population (juvenile + adult)."""
        if self._state is None:
            return 0.0
        return float(self._state[0] + self._state[1])

    @property
    def total_predator(self) -> float:
        """Total predator population (juvenile + adult)."""
        if self._state is None:
            return 0.0
        return float(self._state[2] + self._state[3])

    @property
    def total_population(self) -> float:
        """Sum of all four compartments."""
        if self._state is None:
            return 0.0
        return float(np.sum(self._state))

    @property
    def prey_age_ratio(self) -> float:
        """Ratio N_j / N_a (juvenile-to-adult prey ratio).

        Returns 0.0 if adult prey population is zero.
        """
        if self._state is None or self._state[1] <= 0:
            return 0.0
        return float(self._state[0] / self._state[1])

    @property
    def predator_age_ratio(self) -> float:
        """Ratio P_j / P_a (juvenile-to-adult predator ratio).

        Returns 0.0 if adult predator population is zero.
        """
        if self._state is None or self._state[3] <= 0:
            return 0.0
        return float(self._state[2] / self._state[3])

    @property
    def is_coexisting(self) -> bool:
        """True if all four compartments are above the extinction threshold."""
        if self._state is None:
            return False
        threshold = 1e-6
        return bool(np.all(self._state > threshold))

    def prey_equilibrium_ratio(self) -> float:
        """Theoretical juvenile/adult prey ratio at equilibrium.

        At a steady state where dN_j/dt = 0:
        b_N * N_a = (mu_N + m_j_N) * N_j
        => N_j/N_a = b_N / (mu_N + m_j_N)

        This holds at any true equilibrium (zero net growth), including
        the coexistence equilibrium where predation balances prey growth.
        """
        return self.b_N / (self.mu_N + self.m_j_N)

    def prey_asymptotic_ratio(self) -> float:
        """Asymptotic juvenile/adult prey ratio on the growing eigenvector.

        When prey grows exponentially (no predators), the age ratio converges
        to the eigenvector of the dominant eigenvalue lambda:
            (lambda + mu_N + m_j_N) * N_j = b_N * N_a
            => N_j/N_a = b_N / (lambda + mu_N + m_j_N)

        where lambda is the dominant eigenvalue of the prey subsystem.
        """
        lam = self.prey_net_growth_rate()
        denom = lam + self.mu_N + self.m_j_N
        if denom <= 0:
            return 0.0
        return self.b_N / denom

    def predator_equilibrium_ratio(self) -> float:
        """Theoretical juvenile/adult predator ratio at equilibrium.

        At equilibrium: dP_a/dt = 0 gives mu_P * P_j = d_P * P_a
        => P_j/P_a = d_P / mu_P
        """
        if self.mu_P <= 0:
            return 0.0
        return self.d_P / self.mu_P

    def prey_net_growth_rate(self) -> float:
        """Net prey growth rate in absence of predators.

        The prey subsystem without predators has growth rate determined by
        the dominant eigenvalue of the juvenile-adult transition matrix:
            A = [[-mu_N - m_j_N,  b_N],
                 [ mu_N,         -d_N]]

        The characteristic equation is:
            lambda^2 - tr(A)*lambda + det(A) = 0

        where tr(A) = -(mu_N + m_j_N + d_N) and
              det(A) = (mu_N + m_j_N)*d_N - b_N*mu_N.

        Returns the larger eigenvalue (dominant growth rate).
        """
        tr = -(self.mu_N + self.m_j_N + self.d_N)
        det = (self.mu_N + self.m_j_N) * self.d_N - self.b_N * self.mu_N
        disc = tr * tr - 4.0 * det
        if disc < 0:
            return tr / 2.0
        return (tr + np.sqrt(disc)) / 2.0

    def predator_invasion_rate(self, N_a_star: float) -> float:
        """Growth rate of predator at prey-only equilibrium.

        The predator can invade if the dominant eigenvalue of its subsystem
        is positive when evaluated at the prey-only equilibrium.

        The predator juvenile-adult matrix at prey equilibrium N_a*:
            B = [[-mu_P - m_j_P,  e * f(N_a*)],
                 [ mu_P,         -d_P         ]]

        The dominant eigenvalue is positive when:
            mu_P * e * f(N_a*) > (mu_P + m_j_P) * d_P
        """
        f_star = self.functional_response(N_a_star)
        tr = -(self.mu_P + self.m_j_P + self.d_P)
        det = (self.mu_P + self.m_j_P) * self.d_P - self.mu_P * self.e * f_star
        disc = tr * tr - 4.0 * det
        if disc < 0:
            return tr / 2.0
        return (tr + np.sqrt(disc)) / 2.0

    def maturation_sweep(
        self,
        mu_N_values: np.ndarray,
        n_steps: int = 20000,
        n_transient: int = 10000,
    ) -> dict[str, np.ndarray]:
        """Sweep prey maturation rate and measure equilibrium properties.

        Args:
            mu_N_values: Array of maturation rates to test.
            n_steps: Total simulation steps per trial.
            n_transient: Steps to discard as transient.

        Returns:
            Dict with arrays of maturation rates, mean populations, and age ratios.
        """
        n_measure = n_steps - n_transient
        results: dict[str, list[float]] = {
            "mu_N": [], "N_j_avg": [], "N_a_avg": [],
            "P_j_avg": [], "P_a_avg": [],
            "prey_ratio": [], "predator_ratio": [],
            "amplitude_N": [],
        }

        for mu_N_val in mu_N_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    **{k: v for k, v in self.config.parameters.items()},
                    "mu_N": mu_N_val,
                },
            )
            sim = AgeStructuredSimulation(config)
            sim.reset()

            # Transient
            for _ in range(n_transient):
                sim.step()

            # Measure
            N_j_vals, N_a_vals, P_j_vals, P_a_vals = [], [], [], []
            for _ in range(n_measure):
                state = sim.step()
                N_j_vals.append(state[0])
                N_a_vals.append(state[1])
                P_j_vals.append(state[2])
                P_a_vals.append(state[3])

            N_j_mean = np.mean(N_j_vals)
            N_a_mean = np.mean(N_a_vals)
            P_j_mean = np.mean(P_j_vals)
            P_a_mean = np.mean(P_a_vals)

            results["mu_N"].append(float(mu_N_val))
            results["N_j_avg"].append(float(N_j_mean))
            results["N_a_avg"].append(float(N_a_mean))
            results["P_j_avg"].append(float(P_j_mean))
            results["P_a_avg"].append(float(P_a_mean))
            results["prey_ratio"].append(
                float(N_j_mean / N_a_mean) if N_a_mean > 1e-10 else 0.0
            )
            results["predator_ratio"].append(
                float(P_j_mean / P_a_mean) if P_a_mean > 1e-10 else 0.0
            )
            results["amplitude_N"].append(float(max(N_a_vals) - min(N_a_vals)))

        return {k: np.array(v) for k, v in results.items()}

    @staticmethod
    def _fft_period(signal: np.ndarray, dt: float) -> float:
        """Estimate dominant period from FFT of a signal.

        Returns 0.0 if no clear periodicity is detected.
        """
        sig = signal - np.mean(signal)
        if np.std(sig) < 1e-12:
            return 0.0
        n = len(sig)
        fft_vals = np.fft.rfft(sig)
        power = np.abs(fft_vals) ** 2
        # Skip DC component
        power[0] = 0.0
        if len(power) < 2:
            return 0.0
        peak_idx = np.argmax(power)
        if peak_idx == 0:
            return 0.0
        freq = peak_idx / (n * dt)
        return 1.0 / freq if freq > 0 else 0.0

    def compute_period(self, n_steps: int = 30000) -> float:
        """Compute the oscillation period of adult prey from FFT.

        Discards the first half as transient.
        """
        n_transient = n_steps // 2
        self.reset()
        for _ in range(n_transient):
            self.step()

        N_a_vals = []
        for _ in range(n_steps - n_transient):
            state = self.step()
            N_a_vals.append(state[1])

        return self._fft_period(np.array(N_a_vals), self.config.dt)
