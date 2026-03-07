"""Composable dynamics module library.

Build simulations by composing reusable dynamics modules rather than
writing monolithic SimulationEnvironment subclasses.

Example:
    sim = ComposedSimulation.from_modules([
        HarmonicForce(var="x", param_k="k"),
        LinearDamping(var="v", param_c="c"),
        NewtonianDynamics(pos_var="x", vel_var="v", param_m="m"),
    ], defaults={"k": 4.0, "c": 0.2, "m": 1.0, "x_0": 1.0, "v_0": 0.0})
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import Domain, SimulationConfig


class DynamicsModule(ABC):
    """A reusable building block for simulation dynamics.

    Each module contributes acceleration/rate terms to one or more
    state variables. Modules are composed additively: the total
    derivative of each variable is the sum of contributions from
    all modules that affect it.
    """

    @abstractmethod
    def variables(self) -> list[str]:
        """Return names of state variables this module affects."""

    @abstractmethod
    def compute(
        self, state: dict[str, float], params: dict[str, float], t: float,
    ) -> dict[str, float]:
        """Compute rate contributions for affected variables.

        Args:
            state: Current values of all state variables.
            params: Simulation parameters.
            t: Current time.

        Returns:
            Dict mapping variable names to rate contributions (additive).
        """


# --- Force modules ---

class HarmonicForce(DynamicsModule):
    """Restoring force F = -k * x."""

    def __init__(self, var: str = "x", accel_var: str = "a_x", param_k: str = "k"):
        self.var = var
        self.accel_var = accel_var
        self.param_k = param_k

    def variables(self) -> list[str]:
        return [self.accel_var]

    def compute(self, state, params, t):
        return {self.accel_var: -params[self.param_k] * state[self.var]}


class CubicForce(DynamicsModule):
    """Nonlinear restoring force F = -beta * x^3 (Duffing-type)."""

    def __init__(self, var: str = "x", accel_var: str = "a_x", param_beta: str = "beta"):
        self.var = var
        self.accel_var = accel_var
        self.param_beta = param_beta

    def variables(self) -> list[str]:
        return [self.accel_var]

    def compute(self, state, params, t):
        return {self.accel_var: -params[self.param_beta] * state[self.var] ** 3}


class PeriodicForcing(DynamicsModule):
    """External periodic force F = gamma * cos(omega * t)."""

    def __init__(
        self,
        accel_var: str = "a_x",
        param_gamma: str = "gamma_f",
        param_omega: str = "omega",
    ):
        self.accel_var = accel_var
        self.param_gamma = param_gamma
        self.param_omega = param_omega

    def variables(self) -> list[str]:
        return [self.accel_var]

    def compute(self, state, params, t):
        gamma = params.get(self.param_gamma, 0.0)
        omega = params.get(self.param_omega, 1.0)
        return {self.accel_var: gamma * np.cos(omega * t)}


class GravitationalForce(DynamicsModule):
    """Constant gravitational acceleration F = -g."""

    def __init__(self, accel_var: str = "a_y", param_g: str = "g"):
        self.accel_var = accel_var
        self.param_g = param_g

    def variables(self) -> list[str]:
        return [self.accel_var]

    def compute(self, state, params, t):
        return {self.accel_var: -params.get(self.param_g, 9.81)}


class PendulumForce(DynamicsModule):
    """Pendulum restoring force: -g/L * sin(theta)."""

    def __init__(
        self,
        angle_var: str = "theta",
        accel_var: str = "a_theta",
        param_g: str = "g",
        param_L: str = "L",
    ):
        self.angle_var = angle_var
        self.accel_var = accel_var
        self.param_g = param_g
        self.param_L = param_L

    def variables(self) -> list[str]:
        return [self.accel_var]

    def compute(self, state, params, t):
        g = params.get(self.param_g, 9.81)
        L = params.get(self.param_L, 1.0)
        return {self.accel_var: -g / L * np.sin(state[self.angle_var])}


# --- Damping modules ---

class LinearDamping(DynamicsModule):
    """Linear damping: F = -c * v."""

    def __init__(self, vel_var: str = "v", accel_var: str = "a_x", param_c: str = "c"):
        self.vel_var = vel_var
        self.accel_var = accel_var
        self.param_c = param_c

    def variables(self) -> list[str]:
        return [self.accel_var]

    def compute(self, state, params, t):
        return {self.accel_var: -params.get(self.param_c, 0.0) * state[self.vel_var]}


class VanDerPolDamping(DynamicsModule):
    """Van der Pol damping: F = mu * (1 - x^2) * v."""

    def __init__(
        self,
        pos_var: str = "x",
        vel_var: str = "v",
        accel_var: str = "a_x",
        param_mu: str = "mu",
    ):
        self.pos_var = pos_var
        self.vel_var = vel_var
        self.accel_var = accel_var
        self.param_mu = param_mu

    def variables(self) -> list[str]:
        return [self.accel_var]

    def compute(self, state, params, t):
        mu = params.get(self.param_mu, 1.0)
        x = state[self.pos_var]
        v = state[self.vel_var]
        return {self.accel_var: mu * (1 - x ** 2) * v}


# --- Dynamics integrators ---

class NewtonianDynamics(DynamicsModule):
    """Converts acceleration to velocity change: dv/dt = a/m, dx/dt = v."""

    def __init__(
        self,
        pos_var: str = "x",
        vel_var: str = "v",
        accel_var: str = "a_x",
        param_m: str = "m",
    ):
        self.pos_var = pos_var
        self.vel_var = vel_var
        self.accel_var = accel_var
        self.param_m = param_m

    def variables(self) -> list[str]:
        return [self.pos_var, self.vel_var]

    def compute(self, state, params, t):
        m = params.get(self.param_m, 1.0)
        return {
            self.pos_var: state[self.vel_var],
            self.vel_var: state.get(self.accel_var, 0.0) / m,
        }


# --- Population dynamics modules ---

class LogisticGrowth(DynamicsModule):
    """Logistic growth: dN/dt = r*N*(1 - N/K)."""

    def __init__(self, var: str = "N", param_r: str = "r", param_K: str = "K"):
        self.var = var
        self.param_r = param_r
        self.param_K = param_K

    def variables(self) -> list[str]:
        return [self.var]

    def compute(self, state, params, t):
        N = state[self.var]
        r = params[self.param_r]
        K = params[self.param_K]
        return {self.var: r * N * (1 - N / K)}


class PredatorPreyInteraction(DynamicsModule):
    """Lotka-Volterra interaction terms.

    dN/dt += -beta * N * P
    dP/dt += delta * N * P - gamma * P
    """

    def __init__(
        self,
        prey_var: str = "N",
        pred_var: str = "P",
        param_beta: str = "beta",
        param_delta: str = "delta",
        param_gamma: str = "gamma",
    ):
        self.prey_var = prey_var
        self.pred_var = pred_var
        self.param_beta = param_beta
        self.param_delta = param_delta
        self.param_gamma = param_gamma

    def variables(self) -> list[str]:
        return [self.prey_var, self.pred_var]

    def compute(self, state, params, t):
        N = state[self.prey_var]
        P = state[self.pred_var]
        beta = params[self.param_beta]
        delta = params[self.param_delta]
        gamma = params[self.param_gamma]
        return {
            self.prey_var: -beta * N * P,
            self.pred_var: delta * N * P - gamma * P,
        }


# --- Epidemic modules ---

class SIRDynamics(DynamicsModule):
    """SIR epidemic dynamics.

    dS/dt = -beta * S * I / N
    dI/dt = beta * S * I / N - gamma * I
    dR/dt = gamma * I
    """

    def __init__(
        self,
        s_var: str = "S",
        i_var: str = "I",
        r_var: str = "R",
        param_beta: str = "beta",
        param_gamma: str = "gamma",
        param_N: str = "N_pop",
    ):
        self.s_var = s_var
        self.i_var = i_var
        self.r_var = r_var
        self.param_beta = param_beta
        self.param_gamma = param_gamma
        self.param_N = param_N

    def variables(self) -> list[str]:
        return [self.s_var, self.i_var, self.r_var]

    def compute(self, state, params, t):
        S = state[self.s_var]
        I = state[self.i_var]
        N = params.get(self.param_N, S + I + state[self.r_var])
        beta = params[self.param_beta]
        gamma = params[self.param_gamma]
        infection = beta * S * I / N
        recovery = gamma * I
        return {
            self.s_var: -infection,
            self.i_var: infection - recovery,
            self.r_var: recovery,
        }


# --- Chemical kinetics modules ---

class BrusselatorKinetics(DynamicsModule):
    """Brusselator reaction kinetics: du/dt = a - (b+1)u + u^2*v, dv/dt = b*u - u^2*v."""

    def __init__(
        self,
        u_var: str = "u",
        v_var: str = "v",
        param_a: str = "a",
        param_b: str = "b",
    ):
        self.u_var = u_var
        self.v_var = v_var
        self.param_a = param_a
        self.param_b = param_b

    def variables(self) -> list[str]:
        return [self.u_var, self.v_var]

    def compute(self, state, params, t):
        u = state[self.u_var]
        v = state[self.v_var]
        a = params[self.param_a]
        b = params[self.param_b]
        return {
            self.u_var: a - (b + 1) * u + u ** 2 * v,
            self.v_var: b * u - u ** 2 * v,
        }


# --- Composed simulation ---

class ComposedSimulation(SimulationEnvironment):
    """A simulation built by composing DynamicsModule instances.

    The composed simulation uses RK4 integration over the sum of all
    module rate contributions for each state variable.
    """

    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        self._modules: list[DynamicsModule] = []
        self._var_names: list[str] = []
        self._initial_values: dict[str, float] = {}
        self._accel_vars: set[str] = set()

    @classmethod
    def from_modules(
        cls,
        modules: list[DynamicsModule],
        defaults: dict[str, float] | None = None,
        initial_values: dict[str, float] | None = None,
        dt: float = 0.01,
        n_steps: int = 1000,
        accel_vars: set[str] | None = None,
    ) -> ComposedSimulation:
        """Create a composed simulation from a list of modules.

        Args:
            modules: List of DynamicsModule instances.
            defaults: Default parameter values.
            initial_values: Initial state variable values (keys ending in _0
                in defaults are also treated as initial values).
            dt: Timestep.
            n_steps: Number of simulation steps.
            accel_vars: Set of variable names that are intermediate
                accelerations (not part of the state vector).

        Returns:
            A configured ComposedSimulation instance.
        """
        defaults = defaults or {}
        initial_values = initial_values or {}

        # Extract initial values from defaults (keys ending in _0)
        params = {}
        for k, v in defaults.items():
            if k.endswith("_0"):
                var_name = k[:-2]
                initial_values.setdefault(var_name, v)
            else:
                params[k] = v

        config = SimulationConfig(
            domain=Domain.CUSTOM,
            parameters=params,
            dt=dt,
            n_steps=n_steps,
        )

        sim = cls(config)
        sim._modules = list(modules)
        sim._initial_values = dict(initial_values)
        sim._accel_vars = accel_vars or set()

        # Collect all state variable names (excluding acceleration intermediates)
        var_set: list[str] = []
        for mod in modules:
            for v in mod.variables():
                if v not in var_set and v not in sim._accel_vars:
                    var_set.append(v)
        sim._var_names = var_set

        return sim

    def _state_to_dict(self, state: np.ndarray, t: float = 0.0) -> dict[str, float]:
        """Convert state array to named dict."""
        d: dict[str, float] = {}
        for i, name in enumerate(self._var_names):
            d[name] = float(state[i])
        return d

    def _compute_rates(
        self,
        state_dict: dict[str, float],
        params: dict[str, float],
        t: float,
    ) -> np.ndarray:
        """Compute total rates for all state variables."""
        # First compute acceleration intermediates
        accel_accum: dict[str, float] = {v: 0.0 for v in self._accel_vars}
        for mod in self._modules:
            contrib = mod.compute(state_dict, params, t)
            for v, rate in contrib.items():
                if v in self._accel_vars:
                    accel_accum[v] += rate

        # Add acceleration values to state dict for dynamics modules
        state_with_accel = dict(state_dict)
        state_with_accel.update(accel_accum)

        # Now compute rates for actual state variables
        rates = np.zeros(len(self._var_names))
        for mod in self._modules:
            contrib = mod.compute(state_with_accel, params, t)
            for v, rate in contrib.items():
                if v in self._accel_vars:
                    continue
                if v in self._var_names:
                    idx = self._var_names.index(v)
                    rates[idx] += rate

        return rates

    def reset(self, seed: int | None = None) -> np.ndarray:
        state = np.array([self._initial_values.get(v, 0.0) for v in self._var_names])
        self._state = state
        self._step_count = 0
        return state

    def step(self) -> np.ndarray:
        dt = self.config.dt
        t = self._step_count * dt
        params = self.config.parameters
        s = self._state

        # RK4 integration
        def f(state, t_val):
            sd = self._state_to_dict(state, t_val)
            return self._compute_rates(sd, params, t_val)

        k1 = f(s, t)
        k2 = f(s + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(s + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(s + dt * k3, t + dt)

        self._state = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        return self._state

    def get_variable_names(self) -> list[str]:
        """Return ordered list of state variable names."""
        return list(self._var_names)

    def get_module_names(self) -> list[str]:
        """Return list of module class names."""
        return [type(m).__name__ for m in self._modules]


# --- Pre-built compositions (recipes) ---

def make_harmonic_oscillator(
    k: float = 4.0, m: float = 1.0, c: float = 0.0,
    x_0: float = 1.0, v_0: float = 0.0,
    dt: float = 0.01, n_steps: int = 1000,
) -> ComposedSimulation:
    """Damped harmonic oscillator: m*x'' + c*x' + k*x = 0."""
    return ComposedSimulation.from_modules(
        modules=[
            HarmonicForce(var="x", accel_var="a_x", param_k="k"),
            LinearDamping(vel_var="v", accel_var="a_x", param_c="c"),
            NewtonianDynamics(pos_var="x", vel_var="v", accel_var="a_x", param_m="m"),
        ],
        defaults={"k": k, "m": m, "c": c, "x_0": x_0, "v_0": v_0},
        accel_vars={"a_x"},
        dt=dt, n_steps=n_steps,
    )


def make_duffing_oscillator(
    alpha: float = 1.0, beta: float = 1.0, delta: float = 0.2,
    gamma_f: float = 0.3, omega: float = 1.0,
    x_0: float = 0.5, v_0: float = 0.0,
    dt: float = 0.005, n_steps: int = 10000,
) -> ComposedSimulation:
    """Duffing oscillator: x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)."""
    return ComposedSimulation.from_modules(
        modules=[
            HarmonicForce(var="x", accel_var="a_x", param_k="alpha"),
            CubicForce(var="x", accel_var="a_x", param_beta="beta"),
            LinearDamping(vel_var="v", accel_var="a_x", param_c="delta"),
            PeriodicForcing(accel_var="a_x", param_gamma="gamma_f", param_omega="omega"),
            NewtonianDynamics(pos_var="x", vel_var="v", accel_var="a_x", param_m="m"),
        ],
        defaults={
            "alpha": alpha, "beta": beta, "delta": delta,
            "gamma_f": gamma_f, "omega": omega, "m": 1.0,
            "x_0": x_0, "v_0": v_0,
        },
        accel_vars={"a_x"},
        dt=dt, n_steps=n_steps,
    )


def make_van_der_pol(
    mu: float = 1.0, x_0: float = 0.1, v_0: float = 0.0,
    dt: float = 0.01, n_steps: int = 5000,
) -> ComposedSimulation:
    """Van der Pol oscillator: x'' - mu*(1-x^2)*x' + x = 0."""
    return ComposedSimulation.from_modules(
        modules=[
            HarmonicForce(var="x", accel_var="a_x", param_k="k"),
            VanDerPolDamping(pos_var="x", vel_var="v", accel_var="a_x", param_mu="mu"),
            NewtonianDynamics(pos_var="x", vel_var="v", accel_var="a_x", param_m="m"),
        ],
        defaults={"k": 1.0, "mu": mu, "m": 1.0, "x_0": x_0, "v_0": v_0},
        accel_vars={"a_x"},
        dt=dt, n_steps=n_steps,
    )


def make_pendulum(
    g: float = 9.81, L: float = 1.0, c: float = 0.0,
    theta_0: float = 0.5, omega_0: float = 0.0,
    dt: float = 0.01, n_steps: int = 2000,
) -> ComposedSimulation:
    """Damped pendulum: theta'' + c*theta' + g/L*sin(theta) = 0."""
    return ComposedSimulation.from_modules(
        modules=[
            PendulumForce(
                angle_var="theta", accel_var="a_theta", param_g="g", param_L="L",
            ),
            LinearDamping(vel_var="omega", accel_var="a_theta", param_c="c"),
            NewtonianDynamics(
                pos_var="theta", vel_var="omega", accel_var="a_theta", param_m="m",
            ),
        ],
        defaults={"g": g, "L": L, "c": c, "m": 1.0, "theta_0": theta_0, "omega_0": omega_0},
        accel_vars={"a_theta"},
        dt=dt, n_steps=n_steps,
    )


def make_lotka_volterra(
    alpha: float = 1.1, beta_lv: float = 0.4,
    gamma_lv: float = 0.4, delta_lv: float = 0.1,
    N_0: float = 10.0, P_0: float = 5.0,
    dt: float = 0.01, n_steps: int = 5000,
) -> ComposedSimulation:
    """Lotka-Volterra predator-prey: dN/dt = alpha*N - beta*N*P, dP/dt = delta*N*P - gamma*P."""
    return ComposedSimulation.from_modules(
        modules=[
            LogisticGrowth(var="N", param_r="alpha", param_K="K"),
            PredatorPreyInteraction(
                prey_var="N", pred_var="P",
                param_beta="beta_lv", param_delta="delta_lv", param_gamma="gamma_lv",
            ),
        ],
        defaults={
            "alpha": alpha, "beta_lv": beta_lv, "gamma_lv": gamma_lv,
            "delta_lv": delta_lv, "K": 1e10,
            "N_0": N_0, "P_0": P_0,
        },
        dt=dt, n_steps=n_steps,
    )


def make_sir(
    beta: float = 0.3, gamma: float = 0.1, N_pop: float = 1000.0,
    S_0: float = 990.0, I_0: float = 10.0, R_0: float = 0.0,
    dt: float = 0.1, n_steps: int = 2000,
) -> ComposedSimulation:
    """SIR epidemic model."""
    return ComposedSimulation.from_modules(
        modules=[
            SIRDynamics(
                param_beta="beta", param_gamma="gamma", param_N="N_pop",
            ),
        ],
        defaults={
            "beta": beta, "gamma": gamma, "N_pop": N_pop,
            "S_0": S_0, "I_0": I_0, "R_0": R_0,
        },
        dt=dt, n_steps=n_steps,
    )
