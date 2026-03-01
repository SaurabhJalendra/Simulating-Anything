"""Complex Ginzburg-Landau equation (CGLE) simulation.

The CGLE is a universal amplitude equation near pattern-forming instabilities:

    dA/dt = A + (1 + i*c1) * d^2A/dx^2 - (1 + i*c2) * |A|^2 * A

where A(x,t) is a complex amplitude field on a periodic 1D domain.

Target rediscoveries:
- Benjamin-Feir instability threshold: 1 + c1*c2 < 0
- Amplitude statistics: mean |A| vs parameters
- Phase defect (topological defect) counting
- Plane wave solutions: A = sqrt(1 - q^2) * exp(i*q*x - i*omega*t)
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class GinzburgLandau(SimulationEnvironment):
    """1D Complex Ginzburg-Landau equation with spectral ETD-RK4 solver.

    State: [Re(A_1), ..., Re(A_N), Im(A_1), ..., Im(A_N)] shape (2*N,)
    The complex field A(x) is stored as separate real and imaginary parts
    to satisfy the real-valued numpy array interface.

    Solver: Exponential Time Differencing RK4 (ETD-RK4) via Cox & Matthews.
    The linear part L = 1 + (1+ic1)*d^2/dx^2 is handled exactly in Fourier
    space, while the nonlinear part N(A) = -(1+ic2)*|A|^2*A is integrated
    with a 4th-order scheme.

    Parameters:
        c1: linear dispersion coefficient (default 1.0)
        c2: nonlinear dispersion coefficient (default -1.2)
        L: domain length (default 50.0)
        N: number of grid points (default 128)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.c1 = p.get("c1", 1.0)
        self.c2 = p.get("c2", -1.2)
        self.L = p.get("L", 50.0)
        self.N = int(p.get("N", 128))

        # Spatial grid
        self.dx = self.L / self.N
        self.x = np.linspace(0, self.L, self.N, endpoint=False)

        # Wavenumbers for spectral derivatives
        self.k = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        self.k_sq = self.k ** 2

        # Linear operator in Fourier space: L_hat = 1 - (1+ic1)*k^2
        # Applied to A_hat: each mode grows/decays as exp(L_hat * dt)
        self._L_hat = 1.0 - (1.0 + 1j * self.c1) * self.k_sq

        # Pre-compute ETD coefficients for current dt
        self._setup_etd(config.dt)

    def _setup_etd(self, dt: float) -> None:
        """Pre-compute ETD-RK4 coefficients (Cox & Matthews scheme)."""
        L = self._L_hat
        E = np.exp(L * dt)
        E2 = np.exp(L * dt / 2.0)

        # Number of contour integration points for stable coefficient computation
        M = 32
        r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)

        # Contour integrals for ETD coefficients
        # L is shape (N,), r is shape (M,) -> LR is shape (N, M)
        LR = dt * L[:, None] + r[None, :]

        # f1 = (e^(hL) - 1) / (hL)
        # f2 = (-4 - hL + e^(hL)*(4 - 3*hL + hL^2)) / (hL)^3
        # f3 = (2 + hL + e^(hL)*(-2 + hL)) / (hL)^3
        # f4 = (-4 - 3*hL - hL^2 + e^(hL)*(4 - hL)) / (hL)^3

        Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))

        f1 = dt * np.real(np.mean(
            (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3,
            axis=1
        ))
        f2 = dt * np.real(np.mean(
            (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3,
            axis=1
        ))
        f3 = dt * np.real(np.mean(
            (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3,
            axis=1
        ))

        self._E = E
        self._E2 = E2
        self._Q = Q
        self._f1 = f1
        self._f2 = f2
        self._f3 = f3
        self._dt = dt

    def _nonlinear(self, A: np.ndarray) -> np.ndarray:
        """Compute the nonlinear term N(A) = -(1+ic2)*|A|^2*A in physical space,
        then return its FFT."""
        abs_sq = np.abs(A) ** 2
        NL = -(1.0 + 1j * self.c2) * abs_sq * A
        return np.fft.fft(NL)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize with small random perturbation of the uniform A=1 solution."""
        rng = np.random.default_rng(seed if seed is not None else self.config.seed)

        # Uniform solution A=1 plus small noise
        perturbation = 0.01 * (
            rng.standard_normal(self.N) + 1j * rng.standard_normal(self.N)
        )
        self._A = np.ones(self.N, dtype=complex) + perturbation

        self._state = np.concatenate([self._A.real, self._A.imag])
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using ETD-RK4 (Cox & Matthews)."""
        A = self._A
        A_hat = np.fft.fft(A)

        # ETD-RK4 stages
        N_hat_a = self._nonlinear(A)

        a = self._E2 * A_hat + self._Q * N_hat_a
        A_a = np.fft.ifft(a)
        N_hat_b = self._nonlinear(A_a)

        b = self._E2 * A_hat + self._Q * N_hat_b
        A_b = np.fft.ifft(b)
        N_hat_c = self._nonlinear(A_b)

        c = self._E2 * a + self._Q * (2.0 * N_hat_c - N_hat_a)
        A_c = np.fft.ifft(c)
        N_hat_d = self._nonlinear(A_c)

        # Combine
        A_hat_new = (
            self._E * A_hat
            + N_hat_a * self._f1
            + 2.0 * (N_hat_b + N_hat_c) * self._f2
            + N_hat_d * self._f3
        )

        self._A = np.fft.ifft(A_hat_new)
        self._state = np.concatenate([self._A.real, self._A.imag])
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state as [Re(A), Im(A)]."""
        return self._state

    @property
    def amplitude(self) -> float:
        """Mean amplitude |A| across the domain."""
        return float(np.mean(np.abs(self._A)))

    @property
    def energy(self) -> float:
        """L2 norm (energy) of the field: integral |A|^2 dx / L."""
        return float(np.mean(np.abs(self._A) ** 2))

    @property
    def phase_coherence(self) -> float:
        """Phase coherence: mean cos(arg(A)).

        Measures how aligned the phases are across the domain.
        Value near 1 means uniform phase, near 0 means disordered.
        """
        phases = np.angle(self._A)
        return float(np.mean(np.cos(phases)))

    @property
    def benjamin_feir_parameter(self) -> float:
        """The Benjamin-Feir parameter: 1 + c1*c2.

        When negative, the uniform solution A=1 is unstable (spatio-temporal chaos).
        """
        return 1.0 + self.c1 * self.c2

    @property
    def is_benjamin_feir_unstable(self) -> bool:
        """Whether the Benjamin-Feir instability condition is met: 1 + c1*c2 < 0."""
        return self.benjamin_feir_parameter < 0

    def count_phase_defects(self) -> int:
        """Count phase defects (points where the phase winds by 2*pi).

        A phase defect occurs where the amplitude goes through zero and the
        phase jumps by approximately 2*pi. We detect these by looking for
        large phase gradients.
        """
        phases = np.angle(self._A)
        # Phase differences (mod 2*pi wrapped to [-pi, pi])
        dphase = np.diff(phases)
        dphase = np.mod(dphase + np.pi, 2 * np.pi) - np.pi
        # A defect occurs when |dphase| > pi (winding)
        n_defects = int(np.sum(np.abs(dphase) > np.pi * 0.9))
        return n_defects

    def spatial_std(self) -> float:
        """Spatial standard deviation of |A|, measuring non-uniformity."""
        return float(np.std(np.abs(self._A)))
