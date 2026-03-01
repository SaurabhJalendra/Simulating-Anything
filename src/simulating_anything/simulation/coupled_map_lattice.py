"""Coupled Map Lattice (CML) simulation -- spatiotemporal chaos.

A 1D lattice of N coupled logistic maps:
    x_i(n+1) = (1 - eps) * f(x_i(n)) + eps/2 * (f(x_{i-1}(n)) + f(x_{i+1}(n)))

where f(x) = r * x * (1 - x) is the logistic map.

Target rediscoveries:
- Coupling-dependent order parameter: spatial variance vs eps
- Synchronization transition: eps -> 1 gives complete sync
- Lyapunov exponent spectrum characterizing spatiotemporal complexity
- Space-time pattern classification: chaos, frozen random, synchronization
"""
from __future__ import annotations

import numpy as np

from simulating_anything.simulation.base import SimulationEnvironment
from simulating_anything.types.simulation import SimulationConfig


class CoupledMapLatticeSimulation(SimulationEnvironment):
    """Coupled Map Lattice: 1D lattice of coupled logistic maps.

    State vector: [x_1, x_2, ..., x_N] with periodic boundary conditions.

    Parameters:
        N: lattice size (default 100)
        r: logistic map parameter (default 3.9, chaotic regime)
        eps: coupling strength in [0, 1] (default 0.3)
    """

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__(config)
        p = config.parameters
        self.N = int(p.get("N", 100))
        self.r = p.get("r", 3.9)
        self.eps = p.get("eps", 0.3)

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Initialize lattice with random values in (0, 1)."""
        rng = np.random.default_rng(seed)
        self._state = rng.uniform(0.01, 0.99, self.N).astype(np.float64)
        self._step_count = 0
        return self._state

    def step(self) -> np.ndarray:
        """Advance one timestep: apply coupled logistic map."""
        f = self._logistic(self._state)
        # Periodic boundary: left neighbor is roll(+1), right is roll(-1)
        f_left = np.roll(f, 1)
        f_right = np.roll(f, -1)
        self._state = (
            (1.0 - self.eps) * f
            + (self.eps / 2.0) * (f_left + f_right)
        )
        self._step_count += 1
        return self._state

    def observe(self) -> np.ndarray:
        """Return current lattice state."""
        return self._state

    def _logistic(self, x: np.ndarray) -> np.ndarray:
        """Apply logistic map f(x) = r * x * (1 - x)."""
        return self.r * x * (1.0 - x)

    def compute_spatial_variance(self) -> float:
        """Compute spatial variance of the lattice -- order parameter."""
        return float(np.var(self._state))

    def compute_mean_field(self) -> float:
        """Compute the spatial mean of the lattice."""
        return float(np.mean(self._state))

    def synchronization_order_parameter(
        self,
        var_uncoupled: float | None = None,
    ) -> float:
        """Compute synchronization order parameter: 1 - var(x) / var_uncoupled.

        A value near 1 indicates synchronization; near 0 indicates
        independence. var_uncoupled is the variance of the uncoupled system;
        if not provided, the theoretical variance for a single chaotic logistic
        map is estimated via simulation.
        """
        if var_uncoupled is None:
            var_uncoupled = self._estimate_uncoupled_variance()
        if var_uncoupled < 1e-15:
            return 0.0
        return 1.0 - np.var(self._state) / var_uncoupled

    def _estimate_uncoupled_variance(
        self,
        n_iterations: int = 5000,
        n_transient: int = 1000,
    ) -> float:
        """Estimate variance of a single uncoupled logistic map orbit."""
        x = 0.4
        for _ in range(n_transient):
            x = self.r * x * (1.0 - x)
        orbit = np.empty(n_iterations)
        for i in range(n_iterations):
            x = self.r * x * (1.0 - x)
            orbit[i] = x
        return float(np.var(orbit))

    def coupling_sweep(
        self,
        eps_values: np.ndarray,
        n_transient: int = 2000,
        n_measure: int = 500,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """Sweep coupling strength and measure spatial variance.

        Args:
            eps_values: array of coupling strengths to test.
            n_transient: steps to discard before measuring.
            n_measure: steps over which to average measurements.
            seed: random seed for reproducibility.

        Returns:
            Dict with keys 'eps', 'variance_mean', 'variance_std',
            'mean_field_mean', 'sync_order'.
        """
        var_means = []
        var_stds = []
        mf_means = []
        sync_orders = []

        var_uncoupled = self._estimate_uncoupled_variance()

        for eps_val in eps_values:
            self.eps = eps_val
            self.reset(seed=seed)

            # Transient
            for _ in range(n_transient):
                self.step()

            # Measure
            variances = []
            mean_fields = []
            for _ in range(n_measure):
                self.step()
                variances.append(self.compute_spatial_variance())
                mean_fields.append(self.compute_mean_field())

            var_means.append(float(np.mean(variances)))
            var_stds.append(float(np.std(variances)))
            mf_means.append(float(np.mean(mean_fields)))
            sync_order = 1.0 - np.mean(variances) / var_uncoupled if var_uncoupled > 1e-15 else 0.0
            sync_orders.append(float(sync_order))

        return {
            "eps": np.array(eps_values),
            "variance_mean": np.array(var_means),
            "variance_std": np.array(var_stds),
            "mean_field_mean": np.array(mf_means),
            "sync_order": np.array(sync_orders),
        }

    def compute_lyapunov(
        self,
        n_steps: int = 5000,
        n_transient: int = 1000,
        seed: int | None = None,
    ) -> float:
        """Compute the largest Lyapunov exponent of the CML.

        Uses the tangent-space method: track a perturbation vector and
        renormalize at each step. The Lyapunov exponent is the average
        logarithmic growth rate.
        """
        self.reset(seed=seed)

        # Transient
        for _ in range(n_transient):
            self.step()

        # Initialize perturbation vector (random unit vector)
        rng = np.random.default_rng(seed)
        delta = rng.standard_normal(self.N)
        delta /= np.linalg.norm(delta)

        log_growth_sum = 0.0
        for _ in range(n_steps):
            # Jacobian of the CML map applied to perturbation
            df = self.r * (1.0 - 2.0 * self._state)  # derivative of logistic at each site

            # Apply the linearized CML to the perturbation:
            # J * delta where J_{ii} = (1-eps)*df_i, J_{i,i-1} = eps/2 * df_{i-1}, etc.
            f_delta = df * delta
            f_delta_left = np.roll(f_delta, 1)
            f_delta_right = np.roll(f_delta, -1)
            new_delta = (
                (1.0 - self.eps) * f_delta
                + (self.eps / 2.0) * (f_delta_left + f_delta_right)
            )

            # Advance the state
            self.step()

            # Renormalize and accumulate
            norm = np.linalg.norm(new_delta)
            if norm > 0:
                log_growth_sum += np.log(norm)
                delta = new_delta / norm
            else:
                # Perturbation collapsed -- strongly negative exponent
                log_growth_sum += -100.0
                delta = rng.standard_normal(self.N)
                delta /= np.linalg.norm(delta)

        return log_growth_sum / n_steps

    def space_time_diagram(
        self,
        n_steps: int = 200,
        n_transient: int = 500,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate a space-time diagram (n_steps x N matrix).

        Rows correspond to time steps, columns to lattice sites.
        """
        self.reset(seed=seed)
        for _ in range(n_transient):
            self.step()

        diagram = np.empty((n_steps, self.N))
        for t in range(n_steps):
            self.step()
            diagram[t, :] = self._state.copy()

        return diagram
