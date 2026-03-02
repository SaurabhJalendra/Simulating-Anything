"""Amari neural field simulation (Wilson-Cowan-Amari type).

Implements the integro-differential equation on a 1D spatial domain:

    tau * du(x,t)/dt = -u(x,t) + integral[w(x-y)*f(u(y,t))]dy + h + I_ext(x)

where:
    f(u) = 1 / (1 + exp(-beta*(u - theta)))   -- sigmoid firing rate
    w(x) = A*exp(-|x|/sigma_e) - B*exp(-|x|/sigma_i)  -- Mexican hat kernel

The Mexican hat connectivity produces lateral excitation and surround inhibition,
enabling the formation of stable localized bumps of neural activity.

Target rediscoveries:
- Bump existence threshold in h (resting level)
- Bump width vs sigma_e/sigma_i ratio
- Mexican hat kernel structure (excitation/inhibition balance)
- Bump amplitude and stability analysis
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class AmariNeuralFieldSimulation(SimulationEnvironment):
    """Amari neural field: 1D continuum neural field with Mexican hat connectivity.

    State: u(x) at N grid points representing membrane potential field.

    The model supports stable bump solutions when excitation dominates locally
    and inhibition dominates at longer range (Mexican hat kernel). Bump
    existence depends on the balance between kernel shape, sigmoid steepness,
    and resting level h.

    Parameters:
        tau: membrane time constant (default 1.0)
        A: excitatory kernel amplitude (default 5.0)
        B: inhibitory kernel amplitude (default 2.0)
        sigma_e: excitatory spatial scale (default 1.0)
        sigma_i: inhibitory spatial scale (default 3.0)
        beta: sigmoid steepness (default 50.0)
        theta: sigmoid threshold (default 0.5)
        h: resting level (default -0.5)
        N: number of grid points (default 128)
        L_half: half-domain length, domain is [-L_half, L_half] (default 10.0)

    For localized bump existence, the kernel must have strong local excitation
    (A > B) with inhibition dominating globally (2*B*sigma_i > 2*A*sigma_e).
    Default: A*sigma_e = 5 < B*sigma_i = 6, so total integral is negative (-2),
    but local excitation (A-B = 3.0 at center) is strong enough for self-sustaining bumps.
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters

        # Neural field parameters
        self.tau = p.get("tau", 1.0)
        self.A = p.get("A", 5.0)
        self.B = p.get("B", 2.0)
        self.sigma_e = p.get("sigma_e", 1.0)
        self.sigma_i = p.get("sigma_i", 3.0)
        self.beta = p.get("beta", 50.0)
        self.theta = p.get("theta", 0.5)
        self.h = p.get("h", -0.5)

        # Spatial grid
        self.N = int(p.get("N", 128))
        self.L_half = p.get("L_half", 10.0)
        self.L = 2.0 * self.L_half
        self.dx = self.L / self.N
        self.x = np.linspace(-self.L_half, self.L_half, self.N, endpoint=False)

        # Pre-compute the kernel in Fourier space for efficient convolution.
        # The kernel is defined on the same periodic grid, centered at x=0.
        self._kernel = self._build_kernel()
        self._kernel_fft = np.fft.fft(self._kernel)

        # External input: zero by default
        self._I_ext = np.zeros(self.N, dtype=np.float64)

    def _build_kernel(self) -> np.ndarray:
        """Build the Mexican hat connectivity kernel w(x).

        w(x) = A * exp(-|x| / sigma_e) - B * exp(-|x| / sigma_i)

        The kernel is computed on the periodic grid, wrapping distances
        properly for FFT-based circular convolution.
        """
        # Compute distances with periodic wrapping
        dist = np.abs(self.x)
        # For periodic domain, use minimum distance
        dist = np.minimum(dist, self.L - dist)

        kernel = (
            self.A * np.exp(-dist / self.sigma_e)
            - self.B * np.exp(-dist / self.sigma_i)
        )
        # Scale by dx for numerical integration (trapezoidal-like)
        kernel *= self.dx
        return kernel

    def sigmoid(self, u: np.ndarray | float) -> np.ndarray | float:
        """Sigmoid firing rate function.

        f(u) = 1 / (1 + exp(-beta * (u - theta)))
        """
        return 1.0 / (1.0 + np.exp(-self.beta * (np.asarray(u) - self.theta)))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize with a localized Gaussian bump perturbation.

        The initial condition is a Gaussian centered at x=0 on top of the
        resting level h, providing a seed for bump formation.
        """
        rng = np.random.default_rng(seed)
        # Start with a strong Gaussian perturbation above the sigmoid threshold,
        # providing a seed for self-sustaining bump formation.
        # Amplitude must push peak well above theta for the excitatory feedback
        # to sustain the bump against decay.
        perturbation = 2.0 * np.exp(-self.x ** 2 / (2.0 * self.sigma_e ** 2))
        self._state = self.h + perturbation
        # Add tiny noise for symmetry breaking if needed
        self._state += rng.normal(0, 1e-6, self.N)
        self._state = self._state.astype(np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep using forward Euler with FFT convolution.

        du/dt = (-u + conv(w, f(u)) + h + I_ext) / tau
        """
        dt = self.config.dt
        u = self._state

        # Compute firing rate
        f_u = self.sigmoid(u)

        # Convolution via FFT: w * f(u)
        f_u_fft = np.fft.fft(f_u)
        conv_result = np.real(np.fft.ifft(self._kernel_fft * f_u_fft))

        # Time derivative
        dudt = (-u + conv_result + self.h + self._I_ext) / self.tau

        # Forward Euler update
        self._state = u + dt * dudt
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current membrane potential field u(x)."""
        return self._state

    def set_external_input(self, I_ext: np.ndarray) -> None:
        """Set spatially varying external input I_ext(x)."""
        self._I_ext = I_ext.copy()

    def set_localized_input(
        self, center: float, width: float, amplitude: float
    ) -> None:
        """Set a localized Gaussian external input.

        Args:
            center: center position of the input.
            width: standard deviation of the Gaussian.
            amplitude: peak amplitude.
        """
        self._I_ext = amplitude * np.exp(
            -(self.x - center) ** 2 / (2.0 * width ** 2)
        )

    def compute_bump_width(self) -> float:
        """Compute the width of the activity bump.

        Width is defined as the spatial extent where u(x) > theta (the
        sigmoid threshold). Returns 0 if no bump is detected.
        """
        above_threshold = self._state > self.theta
        if not np.any(above_threshold):
            return 0.0

        # Find contiguous above-threshold regions
        indices = np.where(above_threshold)[0]
        if len(indices) == 0:
            return 0.0

        # Handle wrapping: find the largest contiguous block
        # Split at gaps > 1
        gaps = np.diff(indices)
        split_points = np.where(gaps > 1)[0]

        if len(split_points) == 0:
            # Single contiguous block
            width = (indices[-1] - indices[0] + 1) * self.dx
        else:
            # Multiple blocks -- check if first and last connect via wrapping
            blocks = np.split(indices, split_points + 1)
            block_sizes = [len(b) for b in blocks]

            # Check periodic wrap: first block starts at 0 and last ends at N-1
            if blocks[0][0] == 0 and blocks[-1][-1] == self.N - 1:
                # Wrapped block
                wrap_size = block_sizes[0] + block_sizes[-1]
                inner_max = max(block_sizes[1:-1]) if len(blocks) > 2 else 0
                width = max(wrap_size, inner_max) * self.dx
            else:
                width = max(block_sizes) * self.dx

        return float(width)

    def compute_bump_amplitude(self) -> float:
        """Compute the peak amplitude of the activity bump.

        Returns the maximum value of u(x) minus the resting level h.
        """
        return float(np.max(self._state) - self.h)

    def is_bump_stable(self, n_check_steps: int = 500) -> bool:
        """Check if the current bump is stable by evolving further.

        A bump is considered stable if its width and amplitude remain
        approximately constant over additional integration steps.

        Args:
            n_check_steps: number of additional steps to evolve.

        Returns:
            True if the bump width and amplitude are stable.
        """
        width_before = self.compute_bump_width()
        amp_before = self.compute_bump_amplitude()

        # Save state
        state_save = self._state.copy()
        step_save = self._step_count

        # Evolve forward
        for _ in range(n_check_steps):
            self.step()

        width_after = self.compute_bump_width()
        amp_after = self.compute_bump_amplitude()

        # Restore state
        self._state = state_save
        self._step_count = step_save

        # Check stability: both width and amplitude should not change much
        if width_before < 1e-6 and width_after < 1e-6:
            # No bump in either case -- stable (trivially)
            return True
        if width_before < 1e-6 or width_after < 1e-6:
            # Bump appeared or disappeared -- not stable
            return False

        width_change = abs(width_after - width_before) / max(width_before, 1e-10)
        amp_change = abs(amp_after - amp_before) / max(amp_before, 1e-10)

        return width_change < 0.1 and amp_change < 0.1

    def get_kernel(self) -> np.ndarray:
        """Return the connectivity kernel w(x) for analysis."""
        return self._kernel / self.dx  # Return unscaled kernel

    def compute_total_activity(self) -> float:
        """Compute the total integrated firing rate over the domain."""
        return float(np.sum(self.sigmoid(self._state)) * self.dx)

    def bump_existence_sweep(
        self,
        h_values: np.ndarray | None = None,
        n_steps: int = 2000,
    ) -> dict[str, np.ndarray]:
        """Sweep resting level h and detect bump existence threshold.

        Args:
            h_values: array of h values to test.
            n_steps: integration steps per trial.

        Returns:
            Dict with h values, bump widths, and bump amplitudes.
        """
        if h_values is None:
            h_values = np.linspace(-2.0, 0.5, 25)

        widths = []
        amplitudes = []

        for h_val in h_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    **self.config.parameters,
                    "h": h_val,
                },
            )
            sim = AmariNeuralFieldSimulation(config)
            sim.reset(seed=42)

            for _ in range(n_steps):
                sim.step()

            widths.append(sim.compute_bump_width())
            amplitudes.append(sim.compute_bump_amplitude())

        return {
            "h_values": h_values.copy(),
            "bump_widths": np.array(widths),
            "bump_amplitudes": np.array(amplitudes),
        }

    def sigma_ratio_sweep(
        self,
        sigma_e_values: np.ndarray | None = None,
        n_steps: int = 2000,
    ) -> dict[str, np.ndarray]:
        """Sweep excitatory scale sigma_e and measure bump width.

        Args:
            sigma_e_values: array of sigma_e values to test.
            n_steps: integration steps per trial.

        Returns:
            Dict with sigma_e values, bump widths, and sigma ratios.
        """
        if sigma_e_values is None:
            sigma_e_values = np.linspace(0.5, 3.0, 15)

        widths = []

        for sig_e in sigma_e_values:
            config = SimulationConfig(
                domain=self.config.domain,
                dt=self.config.dt,
                n_steps=n_steps,
                parameters={
                    **self.config.parameters,
                    "sigma_e": sig_e,
                },
            )
            sim = AmariNeuralFieldSimulation(config)
            sim.reset(seed=42)

            for _ in range(n_steps):
                sim.step()

            widths.append(sim.compute_bump_width())

        return {
            "sigma_e_values": sigma_e_values.copy(),
            "sigma_ratios": sigma_e_values / self.sigma_i,
            "bump_widths": np.array(widths),
        }
