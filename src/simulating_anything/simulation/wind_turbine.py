"""Novel Wind Turbine Dynamics simulation (4D).

Coupled aerodynamic rotor with drivetrain and pitch control.
Wind speed variations drive rotor torque via blade element theory.
Generator applies braking torque. Pitch controller regulates
power output.

State: [omega, theta_p, P_gen, v_wind] where:
  omega = rotor angular velocity
  theta_p = blade pitch angle
  P_gen = generator power output
  v_wind = wind speed (turbulent)
"""
from __future__ import annotations
import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

class WindTurbineSimulation(SimulationEnvironment):
    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.J = p.get("J_rotor", 1e6)
        self.R_blade = p.get("R_blade", 40.0)
        self.rho_air = p.get("rho_air", 1.225)
        self.Cp_max = p.get("Cp_max", 0.48)
        self.lambda_opt = p.get("lambda_opt", 8.0)
        self.K_gen = p.get("K_gen", 1e4)
        self.tau_pitch = p.get("tau_pitch", 2.0)
        self.omega_rated = p.get("omega_rated", 1.2)
        self.P_rated = p.get("P_rated", 2e6)
        self.v_mean = p.get("v_mean", 10.0)
        self.v_turb = p.get("v_turb", 2.0)
        self.dt = config.dt
        self._state = np.array([p.get("omega_0", 1.0), p.get("theta_0", 0.0),
                                0.0, self.v_mean], dtype=np.float64)
        self._t = 0.0

    def _Cp(self, lam, theta):
        lam_eff = max(lam - 0.2 * theta, 0.1)
        return self.Cp_max * (1 - ((lam_eff - self.lambda_opt) / self.lambda_opt) ** 2) * np.exp(-0.1 * theta)

    def _rhs(self, state, t):
        omega, theta, Pgen, vw = state
        A = np.pi * self.R_blade ** 2
        lam = omega * self.R_blade / max(vw, 0.1)
        Cp = max(self._Cp(lam, theta), 0)
        P_aero = 0.5 * self.rho_air * A * Cp * vw ** 3
        T_aero = P_aero / max(omega, 0.01)
        T_gen = self.K_gen * omega
        domega = (T_aero - T_gen) / self.J
        theta_ref = max(0, 10 * (omega - self.omega_rated))
        dtheta = (theta_ref - theta) / self.tau_pitch
        dPgen = T_gen * omega - Pgen
        vw_new = self.v_mean + self.v_turb * np.sin(0.5 * t) + self.v_turb * 0.5 * np.sin(1.7 * t)
        dvw = vw_new - vw
        return np.array([domega, dtheta, dPgen, dvw])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                max(1.0 + rng.normal(0, 0.1), 0.1), max(rng.normal(0, 1), 0),
                0.0, self.v_mean + rng.normal(0, 1)], dtype=np.float64)
        else:
            self._state = np.array([1.0, 0.0, 0.0, self.v_mean], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        t = self._t
        k1 = self._rhs(self._state, t)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1, t + 0.5 * self.dt)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2, t + 0.5 * self.dt)
        k4 = self._rhs(self._state + self.dt * k3, t + self.dt)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[0] = max(self._state[0], 0)
        self._state[1] = np.clip(self._state[1], 0, 30)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
