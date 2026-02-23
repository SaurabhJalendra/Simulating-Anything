"""Uncertainty-driven exploration using MC dropout."""

from __future__ import annotations

from typing import Any

import numpy as np

from simulating_anything.exploration.base import Explorer
from simulating_anything.types.trajectory import TrajectoryData, TrajectoryMetadata


class UncertaintyDrivenExplorer(Explorer):
    """Explorer that prioritizes parameters with high epistemic uncertainty.

    Uses a grid of parameter points and selects the next point based on
    a combination of uncertainty (from world model predictions) and
    novelty (distance from previously explored points).

    In V1, uncertainty is estimated via MC dropout in the world model.
    """

    def __init__(
        self,
        sweep_ranges: dict[str, tuple[float, float]],
        n_points_per_dim: int = 10,
        uncertainty_threshold: float = 0.3,
        novelty_weight: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.sweep_ranges = sweep_ranges
        self.uncertainty_threshold = uncertainty_threshold
        self.novelty_weight = novelty_weight
        self.rng = np.random.default_rng(seed)

        # Build parameter grid using Sobol-like quasi-random sequence
        self.param_names = sorted(sweep_ranges.keys())
        n_dims = len(self.param_names)

        # Generate grid points
        points_1d = np.linspace(0, 1, n_points_per_dim)
        if n_dims == 1:
            self._grid_unit = points_1d.reshape(-1, 1)
        elif n_dims == 2:
            g0, g1 = np.meshgrid(points_1d, points_1d)
            self._grid_unit = np.column_stack([g0.ravel(), g1.ravel()])
        else:
            # Quasi-random for higher dims
            self._grid_unit = self.rng.random((n_points_per_dim**min(n_dims, 3), n_dims))

        # Scale to physical ranges
        self._grid_physical = np.zeros_like(self._grid_unit)
        for i, name in enumerate(self.param_names):
            lo, hi = sweep_ranges[name]
            self._grid_physical[:, i] = lo + (hi - lo) * self._grid_unit[:, i]

        self._uncertainties = np.ones(len(self._grid_physical))
        self._visited = np.zeros(len(self._grid_physical), dtype=bool)
        self._trajectories: list[TrajectoryData] = []
        self._explored_points: list[np.ndarray] = []

    def propose_parameters(self) -> dict[str, float]:
        """Select next parameters maximizing uncertainty + novelty."""
        if not np.any(~self._visited):
            # All visited — pick highest uncertainty for re-evaluation
            idx = int(np.argmax(self._uncertainties))
        else:
            # Score = uncertainty * (1 - novelty_weight) + novelty * novelty_weight
            scores = self._uncertainties.copy()

            if self._explored_points:
                explored = np.array(self._explored_points)
                for i in range(len(self._grid_unit)):
                    if self._visited[i]:
                        scores[i] = -1.0
                        continue
                    # Novelty: min distance to explored points
                    dists = np.linalg.norm(explored - self._grid_unit[i], axis=1)
                    novelty = float(np.min(dists))
                    scores[i] = (
                        (1 - self.novelty_weight) * self._uncertainties[i]
                        + self.novelty_weight * novelty
                    )

            idx = int(np.argmax(scores))

        self._visited[idx] = True
        self._explored_points.append(self._grid_unit[idx].copy())

        params = {}
        for i, name in enumerate(self.param_names):
            params[name] = float(self._grid_physical[idx, i])
        return params

    def update(self, trajectory: TrajectoryData) -> None:
        """Store trajectory and update uncertainty estimates."""
        self._trajectories.append(trajectory)

        # Find nearest grid point and reduce its uncertainty
        point = np.array([trajectory.parameters.get(n, 0.0) for n in self.param_names])
        dists = np.linalg.norm(self._grid_physical - point, axis=1)
        nearest = int(np.argmin(dists))
        self._uncertainties[nearest] *= 0.5  # Decay uncertainty after observation

    def set_uncertainties(self, uncertainties: np.ndarray) -> None:
        """Update uncertainty estimates from world model MC dropout."""
        assert len(uncertainties) == len(self._grid_physical)
        self._uncertainties = uncertainties

    def get_interesting_trajectories(self) -> list[TrajectoryData]:
        """Return trajectories with high novelty or unexpected behavior."""
        interesting = []
        for traj in self._trajectories:
            if traj.metadata.novelty_score > self.uncertainty_threshold:
                interesting.append(traj)
            elif traj.metadata.interesting:
                interesting.append(traj)
        return interesting if interesting else self._trajectories[-5:]  # fallback: most recent

    def get_exploration_progress(self) -> dict[str, Any]:
        """Return exploration coverage metrics."""
        return {
            "total_grid_points": len(self._grid_physical),
            "visited": int(np.sum(self._visited)),
            "coverage_fraction": float(np.mean(self._visited)),
            "mean_uncertainty": float(np.mean(self._uncertainties)),
            "max_uncertainty": float(np.max(self._uncertainties)),
            "n_trajectories": len(self._trajectories),
        }
