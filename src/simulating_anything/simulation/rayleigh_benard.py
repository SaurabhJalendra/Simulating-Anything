"""2D Rayleigh-Benard convection simulation -- buoyancy-driven flow.

Solves the Boussinesq equations in 2D using the vorticity-streamfunction
formulation with spectral methods:

  d(omega)/dt = Pr * nabla^2(omega) + Pr * Ra * dT/dx + J(psi, omega)
  dT/dt = nabla^2(T) + J(psi, T)

where omega = vorticity, psi = streamfunction (nabla^2 psi = -omega),
T = temperature perturbation from conduction profile,
Pr = Prandtl number, Ra = Rayleigh number,
J(f,g) = df/dx * dg/dz - df/dz * dg/dx (Jacobian/advection).

Boundary conditions: periodic in x, free-slip in z (sine/cosine series).
Critical Rayleigh number for free-slip: Ra_c = (27/4)*pi^4 ~ 657.5.

Target rediscoveries:
- Critical Rayleigh number Ra_c ~ 657.5 (free-slip)
- Nusselt number Nu(Ra) scaling above onset
- Convection roll wavelength ~ 2*H at onset
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig

# Theoretical critical Rayleigh number for free-slip BCs
RA_CRITICAL_FREE_SLIP = (27.0 / 4.0) * np.pi**4  # ~657.51


class RayleighBenardSimulation(SimulationEnvironment):
    """2D Rayleigh-Benard convection via vorticity-streamfunction method.

    Uses a pseudospectral approach: FFT in x (periodic) and sine/cosine
    series in z (free-slip boundary conditions). The temperature is
    decomposed as T_total = T_cond + T', where T_cond = 1 - z/H is
    the conduction profile and T' is the perturbation.

    Parameters:
        Ra: Rayleigh number (default: 1000)
        Pr: Prandtl number (default: 1.0)
        Lx: horizontal domain size (default: 2.0)
        H: vertical domain height (default: 1.0)
        Nx: horizontal grid points (default: 64)
        Nz: vertical grid points (default: 32)
        perturbation_amp: initial perturbation amplitude (default: 0.01)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.Ra = p.get("Ra", 1000.0)
        self.Pr = p.get("Pr", 1.0)
        self.Lx = p.get("Lx", 2.0)
        self.H = p.get("H", 1.0)
        self.Nx = int(p.get("Nx", 64))
        self.Nz = int(p.get("Nz", 32))
        self.perturbation_amp = p.get("perturbation_amp", 0.01)

        self.dx = self.Lx / self.Nx
        self.dz = self.H / (self.Nz + 1)  # Interior points only for sine series

        # z grid: interior points for sine series (excludes z=0 and z=H)
        self.z_interior = np.linspace(self.dz, self.H - self.dz, self.Nz)
        self.x = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        self.X, self.Z = np.meshgrid(self.x, self.z_interior)

        self._setup_spectral()

    def _setup_spectral(self) -> None:
        """Setup wavenumber arrays for spectral differentiation."""
        # Horizontal wavenumbers (periodic, FFT)
        self.kx = np.fft.fftfreq(self.Nx, d=self.dx) * 2 * np.pi

        # Vertical wavenumbers (sine series: k_z = n*pi/H, n=1,2,...,Nz)
        self.nz = np.arange(1, self.Nz + 1)
        self.kz = self.nz * np.pi / self.H

        # 2D wavenumber grids for spectral Laplacian
        # Shape: (Nz, Nx//2+1) in the (sine, rfft) transform space
        kx_rfft = np.fft.rfftfreq(self.Nx, d=self.dx) * 2 * np.pi
        KX, KZ = np.meshgrid(kx_rfft, self.kz)
        self.k_sq = KX**2 + KZ**2

        # Inverse Laplacian for Poisson solve (avoid division by zero)
        self.k_sq_inv = np.zeros_like(self.k_sq)
        mask = self.k_sq > 0
        self.k_sq_inv[mask] = 1.0 / self.k_sq[mask]

        # Horizontal wavenumber grid for derivatives
        self.kx_grid = KX

        # Sine/cosine transform matrices (for efficiency with small Nz)
        n_col = self.nz.reshape(-1, 1)
        j_row = np.arange(1, self.Nz + 1).reshape(1, -1)
        # DST-I: f_n = (2/(Nz+1)) * sum_j f(z_j) * sin(n*pi*j/(Nz+1))
        self.sin_matrix = np.sin(n_col * np.pi * j_row / (self.Nz + 1))
        self.dst_norm = 2.0 / (self.Nz + 1)

    def _dst_forward(self, f: np.ndarray) -> np.ndarray:
        """Forward discrete sine transform along z-axis (axis=0).

        Input shape: (Nz, Nx). Output shape: (Nz, Nx).
        Transform: f_hat[n] = (2/(Nz+1)) * sum_j f[j] * sin(n*pi*j/(Nz+1))
        """
        return self.dst_norm * (self.sin_matrix @ f)

    def _dst_inverse(self, f_hat: np.ndarray) -> np.ndarray:
        """Inverse discrete sine transform along z-axis (axis=0).

        Input shape: (Nz, Nx). Output shape: (Nz, Nx).
        Transform: f[j] = sum_n f_hat[n] * sin(n*pi*j/(Nz+1))
        """
        return self.sin_matrix.T @ f_hat

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize fields: T' = small perturbation, omega = 0."""
        rng = np.random.default_rng(seed or self.config.seed)

        # Temperature perturbation: small random perturbation
        # Use a single convection-cell mode plus noise
        self._T_pert = self.perturbation_amp * (
            np.sin(np.pi * self.Z / self.H)
            * np.sin(2 * np.pi * self.X / self.Lx)
            + 0.1 * rng.standard_normal((self.Nz, self.Nx))
        )

        # Vorticity: initially zero
        self._omega = np.zeros((self.Nz, self.Nx))

        self._step_count = 0
        self._state = self._pack_state()
        return self._state

    def _pack_state(self) -> np.ndarray:
        """Pack omega and T_pert into a single flat array."""
        return np.concatenate([self._omega.ravel(), self._T_pert.ravel()])

    def step(self) -> np.ndarray:
        """Advance one timestep using RK4."""
        dt = self.config.dt
        omega = self._omega
        T = self._T_pert

        k1_o, k1_T = self._rhs(omega, T)
        k2_o, k2_T = self._rhs(omega + 0.5 * dt * k1_o, T + 0.5 * dt * k1_T)
        k3_o, k3_T = self._rhs(omega + 0.5 * dt * k2_o, T + 0.5 * dt * k2_T)
        k4_o, k4_T = self._rhs(omega + dt * k3_o, T + dt * k3_T)

        self._omega = omega + (dt / 6.0) * (k1_o + 2 * k2_o + 2 * k3_o + k4_o)
        self._T_pert = T + (dt / 6.0) * (k1_T + 2 * k2_T + 2 * k3_T + k4_T)

        self._step_count += 1
        self._state = self._pack_state()
        return self._state

    def observe(self) -> np.ndarray:
        """Return current state (omega, T_pert) flattened."""
        return self._state

    def _rhs(
        self, omega: np.ndarray, T_pert: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute right-hand side of the Boussinesq equations.

        Vorticity equation:
          d(omega)/dt = Pr * nabla^2(omega) + Pr * Ra * dT'/dx + J(psi, omega)

        Temperature equation:
          dT'/dt = nabla^2(T') - w * dT_cond/dz + J(psi, T')
          where dT_cond/dz = -1/H, so -w * (-1/H) = w/H

        Streamfunction: nabla^2(psi) = -omega => solve in spectral space
        Velocity: u = d(psi)/dz, w = -d(psi)/dx
        """
        # Forward transforms: DST in z, then rfft in x
        omega_sz = self._dst_forward(omega)
        omega_hat = np.fft.rfft(omega_sz, axis=1)

        T_sz = self._dst_forward(T_pert)
        T_hat = np.fft.rfft(T_sz, axis=1)

        # Poisson solve for streamfunction: psi_hat = omega_hat / k^2
        psi_hat = omega_hat * self.k_sq_inv

        # Compute velocity in spectral space, then transform back
        # u = d(psi)/dz: in sine space, d/dz of sine(kz*z) = kz*cos(kz*z)
        # This converts sine coefficients to cosine coefficients
        # w = -d(psi)/dx: in Fourier space, d/dx -> i*kx
        # w stays in sine space (sine * Fourier derivative stays sine)

        # Horizontal velocity u = d(psi)/dz
        # psi is in sine basis; derivative maps sine -> cosine
        # u_cosine_hat[n] = kz[n] * psi_hat[n]
        u_cos_hat = self.kz.reshape(-1, 1) * psi_hat
        # Transform cosine coefficients back to physical space
        # For cosine: f[j] = sum_n a_n * cos(n*pi*j/(Nz+1))
        # Use DCT via the cosine matrix
        n_col = self.nz.reshape(-1, 1)
        j_row = np.arange(1, self.Nz + 1).reshape(1, -1)
        cos_matrix = np.cos(n_col * np.pi * j_row / (self.Nz + 1))
        u_cos_x = np.fft.irfft(u_cos_hat, n=self.Nx, axis=1)
        u = cos_matrix.T @ u_cos_x

        # Vertical velocity w = -d(psi)/dx
        # d/dx in Fourier space: multiply by i*kx
        w_hat = -1j * self.kx_grid * psi_hat
        w_sz = np.fft.irfft(w_hat, n=self.Nx, axis=1)
        w = self._dst_inverse(w_sz)

        # Temperature gradient in x (for buoyancy term in vorticity eq)
        dT_dx_hat = 1j * self.kx_grid * T_hat
        dT_dx_sz = np.fft.irfft(dT_dx_hat, n=self.Nx, axis=1)
        dT_dx = self._dst_inverse(dT_dx_sz)

        # Compute gradients for advection (Jacobian terms)
        # d(omega)/dx
        domega_dx_hat = 1j * self.kx_grid * omega_hat
        domega_dx_sz = np.fft.irfft(domega_dx_hat, n=self.Nx, axis=1)
        domega_dx = self._dst_inverse(domega_dx_sz)

        # d(omega)/dz: sine -> cosine derivative
        domega_dz_cos_hat = self.kz.reshape(-1, 1) * omega_hat
        domega_dz_cos_x = np.fft.irfft(domega_dz_cos_hat, n=self.Nx, axis=1)
        domega_dz = cos_matrix.T @ domega_dz_cos_x

        # d(T')/dx
        dT_pert_dx_hat = 1j * self.kx_grid * T_hat
        dT_pert_dx_sz = np.fft.irfft(dT_pert_dx_hat, n=self.Nx, axis=1)
        dT_pert_dx = self._dst_inverse(dT_pert_dx_sz)

        # d(T')/dz: sine -> cosine derivative
        dT_dz_cos_hat = self.kz.reshape(-1, 1) * T_hat
        dT_dz_cos_x = np.fft.irfft(dT_dz_cos_hat, n=self.Nx, axis=1)
        dT_pert_dz = cos_matrix.T @ dT_dz_cos_x

        # Jacobian J(psi, omega) = u * d(omega)/dx + w * d(omega)/dz
        # But in the vorticity equation it appears as J(psi, omega) on RHS
        # which equals u * domega/dx + w * domega/dz (advection)
        adv_omega = u * domega_dx + w * domega_dz

        # Jacobian J(psi, T') = u * dT'/dx + w * dT'/dz
        adv_T = u * dT_pert_dx + w * dT_pert_dz

        # Diffusion terms (computed spectrally)
        # nabla^2(omega) in spectral space: -k^2 * omega_hat
        diff_omega_hat = -self.k_sq * omega_hat
        diff_omega_sz = np.fft.irfft(diff_omega_hat, n=self.Nx, axis=1)
        diff_omega = self._dst_inverse(diff_omega_sz)

        # nabla^2(T') in spectral space
        diff_T_hat = -self.k_sq * T_hat
        diff_T_sz = np.fft.irfft(diff_T_hat, n=self.Nx, axis=1)
        diff_T = self._dst_inverse(diff_T_sz)

        # RHS of vorticity equation
        rhs_omega = (
            self.Pr * diff_omega
            + self.Pr * self.Ra * dT_dx
            - adv_omega
        )

        # RHS of temperature equation
        # dT_cond/dz = -1/H, so -w*dT_cond/dz = w/H (buoyancy forcing)
        rhs_T = diff_T + w / self.H - adv_T

        return rhs_omega, rhs_T

    @property
    def total_temperature(self) -> np.ndarray:
        """Total temperature field: T_cond + T_pert."""
        T_cond = 1.0 - self.Z / self.H
        return T_cond + self._T_pert

    @property
    def vertical_velocity(self) -> np.ndarray:
        """Compute vertical velocity w from streamfunction."""
        omega_sz = self._dst_forward(self._omega)
        omega_hat = np.fft.rfft(omega_sz, axis=1)
        psi_hat = omega_hat * self.k_sq_inv
        w_hat = -1j * self.kx_grid * psi_hat
        w_sz = np.fft.irfft(w_hat, n=self.Nx, axis=1)
        return self._dst_inverse(w_sz)

    def compute_nusselt(self) -> float:
        """Compute the Nusselt number.

        Nu = 1 + <w*T'>_vol / (kappa * Delta_T / H)
        For the nondimensionalized equations (Delta_T=1, H=1, kappa=1):
        Nu = 1 + <w * T'> * H

        In our nondimensionalization, kappa=1, so:
        Nu = 1 + H * mean(w * T')
        """
        w = self.vertical_velocity
        wT_mean = np.mean(w * self._T_pert)
        return 1.0 + self.H * wT_mean

    def convection_amplitude(self) -> float:
        """Measure convection amplitude as max |psi| or max |w|."""
        w = self.vertical_velocity
        return float(np.max(np.abs(w)))

    def get_roll_wavelength(self) -> float:
        """Measure dominant horizontal wavelength from FFT of w at mid-height.

        Returns the wavelength of the dominant horizontal mode.
        """
        w = self.vertical_velocity
        # Take mid-height slice
        mid_z = self.Nz // 2
        w_mid = w[mid_z, :]

        # FFT and find dominant mode
        w_hat = np.fft.rfft(w_mid)
        amplitudes = np.abs(w_hat)
        # Skip k=0 (mean)
        amplitudes[0] = 0
        if np.max(amplitudes) < 1e-15:
            return self.Lx  # No convection
        dominant_k = np.argmax(amplitudes)
        if dominant_k == 0:
            return self.Lx
        wavelength = self.Lx / dominant_k
        return float(wavelength)

    def convection_onset_sweep(
        self,
        Ra_values: np.ndarray | list[float],
        n_steps: int = 5000,
    ) -> dict[str, np.ndarray]:
        """Sweep Ra and measure convection amplitude at each value.

        Args:
            Ra_values: Array of Rayleigh numbers to test.
            n_steps: Steps to run at each Ra for equilibration.

        Returns:
            Dict with Ra_values, amplitudes, nusselt numbers.
        """
        amplitudes = []
        nusselt_numbers = []

        for Ra in Ra_values:
            self.Ra = Ra
            self.reset(seed=42)

            for _ in range(n_steps):
                self.step()

            amp = self.convection_amplitude()
            nu_ = self.compute_nusselt()
            amplitudes.append(amp)
            nusselt_numbers.append(nu_)

        return {
            "Ra": np.array(Ra_values, dtype=float),
            "amplitude": np.array(amplitudes),
            "nusselt": np.array(nusselt_numbers),
        }
