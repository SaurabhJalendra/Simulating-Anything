"""Chirikov standard map simulation (area-preserving 2D discrete map).

Target rediscoveries:
- Area preservation: Jacobian determinant = 1 (symplectic map)
- KAM tori at low K, chaos at high K
- Critical stochasticity parameter K_c ~ 0.9716 (Greene's criterion)
- Lyapunov exponent lambda(K): zero for regular, positive for chaotic
- Phase space portraits showing KAM tori breakup
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class StandardMapSimulation(SimulationEnvironment):
    """Chirikov standard map: p_{n+1} = p_n + K*sin(theta_n), theta_{n+1} = theta_n + p_{n+1}.

    Both coordinates are taken mod 2*pi.

    State vector: shape (2 * n_particles,), interleaved as
    [theta_0, p_0, theta_1, p_1, ...].

    Parameters:
        K: stochasticity parameter (default 0.9716, critical value)
        n_particles: number of particles to track (default 100)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.K = p.get("K", 0.9716)
        self.n_particles = int(p.get("n_particles", 100))

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize particles on a uniform grid in [0, 2*pi) x [0, 2*pi)."""
        n_side = int(np.ceil(np.sqrt(self.n_particles)))
        theta_vals = np.linspace(0, 2 * np.pi, n_side, endpoint=False)
        p_vals = np.linspace(0, 2 * np.pi, n_side, endpoint=False)
        theta_grid, p_grid = np.meshgrid(theta_vals, p_vals)
        theta_flat = theta_grid.ravel()[:self.n_particles]
        p_flat = p_grid.ravel()[:self.n_particles]

        # Interleave: [theta_0, p_0, theta_1, p_1, ...]
        self._state = np.empty(2 * self.n_particles, dtype=np.float64)
        self._state[0::2] = theta_flat
        self._state[1::2] = p_flat
        self._step_count = 0
        return self._state.copy()

    def step(self) -> np.ndarray:
        """Apply one iteration of the standard map to all particles."""
        thetas = self._state[0::2]
        ps = self._state[1::2]

        # Standard map equations
        ps_new = (ps + self.K * np.sin(thetas)) % (2 * np.pi)
        thetas_new = (thetas + ps_new) % (2 * np.pi)

        self._state[0::2] = thetas_new
        self._state[1::2] = ps_new
        self._step_count += 1
        return self._state.copy()

    def observe(self) -> np.ndarray:
        """Return current state [theta_0, p_0, theta_1, p_1, ...]."""
        return self._state.copy()

    def get_particles(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (thetas, momenta) as separate arrays."""
        return self._state[0::2].copy(), self._state[1::2].copy()

    def compute_lyapunov(
        self,
        K: float | None = None,
        n_steps: int = 10000,
        n_transient: int = 1000,
        theta_0: float = 0.5,
        p_0: float = 0.5,
    ) -> float:
        """Compute the maximal Lyapunov exponent for the standard map.

        Uses the QR method: track a tangent vector along a single orbit,
        periodically re-normalizing via QR decomposition to prevent overflow.

        Args:
            K: stochasticity parameter (defaults to self.K).
            n_steps: number of iterations for accumulation.
            n_transient: transient iterations before accumulation.
            theta_0: initial theta for the test orbit.
            p_0: initial momentum for the test orbit.

        Returns:
            Maximal Lyapunov exponent (nats per iteration).
        """
        if K is None:
            K = self.K

        theta, p = theta_0, p_0

        # Transient to settle onto typical orbit
        for _ in range(n_transient):
            p = (p + K * np.sin(theta)) % (2 * np.pi)
            theta = (theta + p) % (2 * np.pi)

        # QR-based Lyapunov computation
        q = np.eye(2)
        lyap_sum = np.zeros(2)

        for _ in range(n_steps):
            # Jacobian of the standard map at (theta, p):
            # dp_new/dtheta = K*cos(theta), dp_new/dp = 1
            # dtheta_new/dtheta = K*cos(theta) + 1, dtheta_new/dp = 1
            # But we order state as (theta, p), so:
            cos_theta = np.cos(theta)
            jac = np.array([
                [1.0 + K * cos_theta, 1.0],
                [K * cos_theta, 1.0],
            ])

            # Propagate tangent vectors
            m = jac @ q
            q, r = np.linalg.qr(m)
            lyap_sum += np.log(np.abs(np.diag(r)))

            # Advance orbit
            p = (p + K * np.sin(theta)) % (2 * np.pi)
            theta = (theta + p) % (2 * np.pi)

        return float(lyap_sum[0] / n_steps)

    def phase_portrait(
        self,
        K: float | None = None,
        n_particles: int = 500,
        n_steps: int = 200,
    ) -> dict[str, np.ndarray]:
        """Generate phase space portrait data.

        Args:
            K: stochasticity parameter (defaults to self.K).
            n_particles: number of initial conditions.
            n_steps: iterations per particle.

        Returns:
            Dict with 'theta' and 'p' arrays of all visited points.
        """
        if K is None:
            K = self.K

        all_theta = []
        all_p = []

        # Initialize particles on a grid
        n_side = int(np.ceil(np.sqrt(n_particles)))
        theta_vals = np.linspace(0, 2 * np.pi, n_side, endpoint=False)
        p_vals = np.linspace(0, 2 * np.pi, n_side, endpoint=False)
        theta_grid, p_grid = np.meshgrid(theta_vals, p_vals)
        thetas = theta_grid.ravel()[:n_particles]
        ps = p_grid.ravel()[:n_particles]

        # Record initial positions
        all_theta.append(thetas.copy())
        all_p.append(ps.copy())

        # Iterate all particles
        for _ in range(n_steps):
            ps = (ps + K * np.sin(thetas)) % (2 * np.pi)
            thetas = (thetas + ps) % (2 * np.pi)
            all_theta.append(thetas.copy())
            all_p.append(ps.copy())

        return {
            "theta": np.concatenate(all_theta),
            "p": np.concatenate(all_p),
        }

    def stochasticity_sweep(
        self,
        K_values: np.ndarray,
        n_particles: int = 200,
        n_steps: int = 500,
        lyapunov_threshold: float = 0.01,
    ) -> dict[str, np.ndarray]:
        """Sweep K and measure the fraction of chaotic orbits.

        A particle is classified as chaotic if its Lyapunov exponent exceeds
        the threshold.

        Args:
            K_values: array of K values to sweep.
            n_particles: particles per K value.
            n_steps: iterations for each Lyapunov computation.
            lyapunov_threshold: minimum Lyapunov to count as chaotic.

        Returns:
            Dict with K_values, chaos_fractions, and mean_lyapunovs.
        """
        chaos_fractions = []
        mean_lyapunovs = []

        rng = np.random.default_rng(42)

        for K in K_values:
            # Sample random initial conditions
            theta_ics = rng.uniform(0, 2 * np.pi, n_particles)
            p_ics = rng.uniform(0, 2 * np.pi, n_particles)

            lyaps = []
            for theta_0, p_0 in zip(theta_ics, p_ics):
                lam = self.compute_lyapunov(
                    K=K,
                    n_steps=n_steps,
                    n_transient=200,
                    theta_0=theta_0,
                    p_0=p_0,
                )
                lyaps.append(lam)

            lyaps_arr = np.array(lyaps)
            frac = float(np.mean(lyaps_arr > lyapunov_threshold))
            chaos_fractions.append(frac)
            mean_lyapunovs.append(float(np.mean(lyaps_arr)))

        return {
            "K_values": K_values,
            "chaos_fractions": np.array(chaos_fractions),
            "mean_lyapunovs": np.array(mean_lyapunovs),
        }
