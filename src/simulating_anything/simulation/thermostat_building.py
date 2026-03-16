"""Novel Building Thermostat simulation (4D).

Coupled building thermal dynamics with HVAC control.
Building gains/loses heat through walls, windows, and ventilation.
Thermostat controls heating/cooling to maintain setpoint.
Occupancy and solar gain create disturbances.

State: [T_in, T_wall, Q_hvac, E_cum] where:
  T_in = indoor air temperature
  T_wall = wall temperature (thermal mass)
  Q_hvac = HVAC heat delivery rate
  E_cum = cumulative energy consumption
"""
from __future__ import annotations
import numpy as np
from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

class ThermostatBuildingSimulation(SimulationEnvironment):
    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.UA = p.get("UA", 200.0)
        self.C_air = p.get("C_air", 1000.0)
        self.C_wall = p.get("C_wall", 5000.0)
        self.h_wall = p.get("h_wall", 50.0)
        self.T_out = p.get("T_out", 5.0)
        self.T_set = p.get("T_set", 21.0)
        self.K_p = p.get("K_p", 500.0)
        self.Q_max = p.get("Q_max", 5000.0)
        self.Q_solar = p.get("Q_solar", 500.0)
        self.dt = config.dt
        self._state = np.array([p.get("Tin_0", 20.0), p.get("Twall_0", 15.0),
                                0.0, 0.0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state, t):
        Tin, Tw, Qh, E = state
        Q_sol = self.Q_solar * max(np.sin(2 * np.pi * 0.04 * t), 0)
        Q_ctrl = self.K_p * (self.T_set - Tin)
        Q_ctrl = np.clip(Q_ctrl, -self.Q_max, self.Q_max)
        dTin = (Q_ctrl + Q_sol - self.UA * (Tin - self.T_out) + self.h_wall * (Tw - Tin)) / self.C_air
        dTw = (self.h_wall * (Tin - Tw) - 0.5 * self.UA * (Tw - self.T_out)) / self.C_wall
        dQh = Q_ctrl - Qh
        dE = abs(Q_ctrl) / 1000.0
        return np.array([dTin, dTw, dQh, dE])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([20 + rng.normal(0, 1), 15 + rng.normal(0, 1), 0, 0], dtype=np.float64)
        else:
            self._state = np.array([20.0, 15.0, 0.0, 0.0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        t = self._t
        k1 = self._rhs(self._state, t)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1, t + 0.5 * self.dt)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2, t + 0.5 * self.dt)
        k4 = self._rhs(self._state + self.dt * k3, t + self.dt)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
