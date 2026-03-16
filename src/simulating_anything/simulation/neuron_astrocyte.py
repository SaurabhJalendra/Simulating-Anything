"""Novel coupled Neuron-Astrocyte simulation (4D).

Couples FitzHugh-Nagumo neuron dynamics with astrocyte calcium
oscillations. Astrocytes release gliotransmitters that modulate
synaptic strength, while neuronal activity triggers astrocyte
calcium waves. This bidirectional coupling is a key unsolved
problem in computational neuroscience.

State: [v, w, Ca, IP3] where:
  v = neuronal membrane potential (FHN)
  w = neuronal recovery variable (FHN)
  Ca = astrocyte intracellular calcium
  IP3 = inositol trisphosphate (second messenger)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class NeuronAstrocyteSimulation(SimulationEnvironment):
    """Coupled FHN neuron + astrocyte calcium dynamics."""

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # FHN neuron
        self.a_n = p.get("a_n", 0.7)
        self.b_n = p.get("b_n", 0.8)
        self.eps_n = p.get("eps_n", 0.08)
        self.I_ext = p.get("I_ext", 0.5)

        # Astrocyte calcium
        self.v_ca = p.get("v_ca", 0.5)
        self.K_ca = p.get("K_ca", 0.5)
        self.d_ca = p.get("d_ca", 0.1)
        self.v_ip3 = p.get("v_ip3", 0.3)
        self.d_ip3 = p.get("d_ip3", 0.2)

        # Coupling
        self.coupling_na = p.get("coupling_na", 0.2)  # neuron -> astrocyte IP3
        self.coupling_an = p.get("coupling_an", 0.1)  # astrocyte Ca -> neuron

        self.v_0 = p.get("v_0", -1.0)
        self.w_0 = p.get("w_0", -0.5)
        self.Ca_0 = p.get("Ca_0", 0.3)
        self.IP3_0 = p.get("IP3_0", 0.2)

        self.dt = config.dt
        self._state = np.array([self.v_0, self.w_0, self.Ca_0, self.IP3_0], dtype=np.float64)
        self._t = 0.0

    def _rhs(self, state):
        v, w, Ca, IP3 = state

        # Gliotransmitter modulation of excitability
        I_glio = self.coupling_an * Ca

        # FHN neuron with astrocyte modulation
        dv = v - v ** 3 / 3.0 - w + self.I_ext + I_glio
        dw = self.eps_n * (v + self.a_n - self.b_n * w)

        # IP3 production from neuronal activity
        v_clipped = max(v, -2.0)
        ip3_prod = self.coupling_na * (1.0 / (1.0 + np.exp(-v_clipped)))

        # Astrocyte calcium: IP3-dependent release from ER
        ca_release = self.v_ca * IP3 ** 2 / (self.K_ca ** 2 + IP3 ** 2)
        dCa = ca_release - self.d_ca * Ca
        dIP3 = self.v_ip3 + ip3_prod - self.d_ip3 * IP3

        return np.array([dv, dw, dCa, dIP3])

    def reset(self, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._state = np.array([
                self.v_0 + rng.normal(0, 0.1),
                self.w_0 + rng.normal(0, 0.05),
                max(self.Ca_0 + rng.normal(0, 0.05), 0.01),
                max(self.IP3_0 + rng.normal(0, 0.03), 0.01),
            ], dtype=np.float64)
        else:
            self._state = np.array([self.v_0, self.w_0, self.Ca_0, self.IP3_0], dtype=np.float64)
        self._t = 0.0
        return self.observe()

    def step(self):
        k1 = self._rhs(self._state)
        k2 = self._rhs(self._state + 0.5 * self.dt * k1)
        k3 = self._rhs(self._state + 0.5 * self.dt * k2)
        k4 = self._rhs(self._state + self.dt * k3)
        self._state += (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._state[2:] = np.clip(self._state[2:], 0.0, None)
        self._t += self.dt
        return self.observe()

    def observe(self):
        return self._state.copy()
