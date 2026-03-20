"""Discovery Campaign Runner: sweep → observe → detect → discover → validate.

Takes a domain + scientific question and produces phase diagrams,
bifurcation points, scaling laws, and validated predictions.
"""
from __future__ import annotations

import importlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from simulating_anything.analysis.bifurcation_detector import (
    BifurcationResult,
    detect_bifurcations_1d,
)
from simulating_anything.analysis.observable_extractor import (
    TrajectoryObservables,
    extract_observables,
)
from simulating_anything.types.simulation import Domain, SimulationConfig

logger = logging.getLogger(__name__)


@dataclass
class CampaignConfig:
    """Configuration for a discovery campaign."""
    domain_name: str
    sim_module: str
    sim_class: str
    question: str
    sweep_params: dict[str, tuple[float, float]]  # param -> (lo, hi)
    n_points: int = 50
    n_steps: int = 2000
    dt: float = 0.01
    base_params: dict[str, float] = field(default_factory=dict)


@dataclass
class CampaignDiscovery:
    """A single discovery from a campaign."""
    discovery_type: str  # bifurcation, scaling_law, phase_boundary, invariant
    description: str
    parameter: str
    critical_value: float | None = None
    equation: str | None = None
    r_squared: float | None = None
    evidence: dict = field(default_factory=dict)
    validated: bool = False


@dataclass
class CampaignResult:
    """Complete results from a discovery campaign."""
    config: CampaignConfig
    bifurcations: list[BifurcationResult]
    discoveries: list[CampaignDiscovery]
    phase_classifications: dict[str, list[str]]
    observable_data: dict
    runtime_seconds: float
    narrative: str


class DiscoveryCampaignRunner:
    """Runs a complete discovery campaign on an existing domain."""

    def __init__(self, config: CampaignConfig, output_dir: str | Path = "output/discoveries"):
        self.config = config
        self.output_dir = Path(output_dir) / config.domain_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> CampaignResult:
        """Execute the full campaign."""
        t0 = time.time()
        sim_class = self._load_sim_class()

        sweep_params = list(self.config.sweep_params.keys())
        n_sweep = len(sweep_params)

        all_bifurcations = []
        all_discoveries = []
        phase_classifications = {}
        observable_data = {}

        if n_sweep == 1:
            # 1D sweep
            param = sweep_params[0]
            lo, hi = self.config.sweep_params[param]
            values = np.linspace(lo, hi, self.config.n_points)

            logger.info(f"Running 1D sweep: {param} in [{lo}, {hi}], {self.config.n_points} points")
            observables = self._run_1d_sweep(sim_class, param, values)

            # Detect bifurcations
            bif_result = detect_bifurcations_1d(values, observables, param)
            all_bifurcations.append(bif_result)
            phase_classifications[param] = bif_result.classifications
            observable_data[param] = {
                k: v.tolist() for k, v in bif_result.observable_series.items()
            }

            # Log bifurcations
            for bp in bif_result.bifurcation_points:
                logger.info(
                    f"  BIFURCATION: {bp.bifurcation_type} at {param}={bp.critical_value:.4f} "
                    f"({bp.before_classification} → {bp.after_classification})"
                )
                all_discoveries.append(CampaignDiscovery(
                    discovery_type="bifurcation",
                    description=f"{bp.bifurcation_type} bifurcation at {param}={bp.critical_value:.4f}",
                    parameter=param,
                    critical_value=bp.critical_value,
                    evidence={
                        "type": bp.bifurcation_type,
                        "ci": bp.confidence_interval,
                        "before": bp.before_classification,
                        "after": bp.after_classification,
                    },
                ))

            # Fit scaling laws on observable series
            self._fit_scaling_laws(values, observables, param, all_discoveries)

        elif n_sweep == 2:
            # 2D sweep
            p1, p2 = sweep_params
            lo1, hi1 = self.config.sweep_params[p1]
            lo2, hi2 = self.config.sweep_params[p2]
            v1 = np.linspace(lo1, hi1, self.config.n_points)
            v2 = np.linspace(lo2, hi2, self.config.n_points)

            logger.info(f"Running 2D sweep: {p1}×{p2}, {self.config.n_points}² = {self.config.n_points**2} points")
            grid = self._run_2d_sweep(sim_class, p1, v1, p2, v2)

            # Classify each point
            class_grid = [[o.classification for o in row] for row in grid]
            phase_classifications[f"{p1}_x_{p2}"] = class_grid

            # Count regions
            unique_classes = set()
            for row in class_grid:
                unique_classes.update(row)
            logger.info(f"  Phase regions found: {unique_classes}")

            # Store observable data for the grid
            obs_grid = {}
            for obs_name in ["mean_x0", "amplitude_x0", "peak_value"]:
                arr = np.zeros((len(v1), len(v2)))
                for i, row in enumerate(grid):
                    for j, o in enumerate(row):
                        if obs_name == "mean_x0" and len(o.mean) > 0:
                            arr[i, j] = o.mean[0]
                        elif obs_name == "amplitude_x0" and len(o.amplitude) > 0:
                            arr[i, j] = o.amplitude[0]
                        elif obs_name == "peak_value" and len(o.peak_value) > 0:
                            arr[i, j] = o.peak_value[0]
                obs_grid[obs_name] = arr.tolist()
            observable_data[f"{p1}_x_{p2}"] = obs_grid

            # Detect boundaries
            if len(unique_classes) > 1:
                all_discoveries.append(CampaignDiscovery(
                    discovery_type="phase_boundary",
                    description=f"Phase diagram with {len(unique_classes)} regions: {unique_classes}",
                    parameter=f"{p1} × {p2}",
                    evidence={"regions": list(unique_classes), "grid_size": self.config.n_points},
                ))

            # 1D slices for detailed bifurcation analysis
            mid_idx = len(v2) // 2
            slice_obs = [grid[i][mid_idx] for i in range(len(v1))]
            bif_slice = detect_bifurcations_1d(v1, slice_obs, p1)
            all_bifurcations.append(bif_slice)
            for bp in bif_slice.bifurcation_points:
                logger.info(f"  BIFURCATION (slice): {bp.bifurcation_type} at {p1}={bp.critical_value:.4f}")
                all_discoveries.append(CampaignDiscovery(
                    discovery_type="bifurcation",
                    description=f"{bp.bifurcation_type} at {p1}={bp.critical_value:.4f} (at {p2}={v2[mid_idx]:.3f})",
                    parameter=p1,
                    critical_value=bp.critical_value,
                    evidence={"slice_param": p2, "slice_value": float(v2[mid_idx])},
                ))

        # Generate narrative
        narrative = self._generate_narrative(all_discoveries)
        runtime = time.time() - t0

        result = CampaignResult(
            config=self.config,
            bifurcations=all_bifurcations,
            discoveries=all_discoveries,
            phase_classifications=phase_classifications,
            observable_data=observable_data,
            runtime_seconds=runtime,
            narrative=narrative,
        )

        # Save
        self._save_results(result)
        self._generate_figures(result)

        logger.info(f"\nCampaign complete in {runtime:.1f}s")
        logger.info(f"Discoveries: {len(all_discoveries)}")
        logger.info(f"Narrative:\n{narrative}")

        return result

    def _load_sim_class(self):
        mod = importlib.import_module(f"simulating_anything.simulation.{self.config.sim_module}")
        return getattr(mod, self.config.sim_class)

    def _run_1d_sweep(self, sim_class, param_name, param_values):
        observables = []
        for i, pval in enumerate(param_values):
            params = dict(self.config.base_params)
            params[param_name] = float(pval)
            config = SimulationConfig(
                domain=Domain.CUSTOM, dt=self.config.dt,
                n_steps=self.config.n_steps, parameters=params,
            )
            try:
                sim = sim_class(config)
                sim.reset(seed=0)
                states = [sim.observe().copy()]
                for _ in range(self.config.n_steps):
                    states.append(sim.step().copy())
                obs = extract_observables(np.array(states), self.config.dt)
            except Exception:
                obs = TrajectoryObservables(
                    mean=np.array([0]), std=np.array([0]),
                    amplitude=np.array([0]), is_divergent=True,
                    classification="divergent",
                )
            observables.append(obs)
            if (i + 1) % 50 == 0:
                logger.info(f"  {i+1}/{len(param_values)} points computed")
        return observables

    def _run_2d_sweep(self, sim_class, p1, v1, p2, v2):
        grid = []
        total = len(v1) * len(v2)
        count = 0
        for i, pv1 in enumerate(v1):
            row = []
            for j, pv2 in enumerate(v2):
                params = dict(self.config.base_params)
                params[p1] = float(pv1)
                params[p2] = float(pv2)
                config = SimulationConfig(
                    domain=Domain.CUSTOM, dt=self.config.dt,
                    n_steps=self.config.n_steps, parameters=params,
                )
                try:
                    sim = sim_class(config)
                    sim.reset(seed=0)
                    states = [sim.observe().copy()]
                    for _ in range(self.config.n_steps):
                        states.append(sim.step().copy())
                    obs = extract_observables(np.array(states), self.config.dt)
                except Exception:
                    obs = TrajectoryObservables(
                        mean=np.array([0]), std=np.array([0]),
                        amplitude=np.array([0]), is_divergent=True,
                        classification="divergent",
                    )
                row.append(obs)
                count += 1
            grid.append(row)
            if (i + 1) % 10 == 0:
                logger.info(f"  Row {i+1}/{len(v1)} ({count}/{total} points)")
        return grid

    def _fit_scaling_laws(self, param_values, observables, param_name, discoveries):
        """Fit scaling laws on parameter-observable relationships."""
        for obs_name in ["mean_x0", "amplitude_x0"]:
            if obs_name == "mean_x0":
                values = np.array([o.mean[0] if len(o.mean) > 0 else np.nan for o in observables])
            else:
                values = np.array([o.amplitude[0] if len(o.amplitude) > 0 else np.nan for o in observables])

            valid = ~np.isnan(values) & ~np.isinf(values)
            if np.sum(valid) < 10:
                continue

            x = param_values[valid]
            y = values[valid]

            # Try polynomial fits (degree 1-3) and pick best
            best_r2 = -1
            best_eq = ""
            best_deg = 0
            for deg in [1, 2, 3]:
                try:
                    coeffs = np.polyfit(x, y, deg)
                    pred = np.polyval(coeffs, x)
                    ss_res = np.sum((y - pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - ss_res / max(ss_tot, 1e-12)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_deg = deg
                        terms = []
                        for k, c in enumerate(coeffs):
                            power = deg - k
                            if abs(c) > 1e-6:
                                if power == 0:
                                    terms.append(f"{c:.4f}")
                                elif power == 1:
                                    terms.append(f"{c:.4f}*{param_name}")
                                else:
                                    terms.append(f"{c:.4f}*{param_name}^{power}")
                        best_eq = " + ".join(terms)
                except Exception:
                    continue

            if best_r2 > 0.8:
                discoveries.append(CampaignDiscovery(
                    discovery_type="scaling_law",
                    description=f"{obs_name} = {best_eq} (R²={best_r2:.4f})",
                    parameter=param_name,
                    equation=best_eq,
                    r_squared=best_r2,
                    evidence={"degree": best_deg, "n_points": int(np.sum(valid))},
                ))
                logger.info(f"  SCALING LAW: {obs_name} = {best_eq} (R²={best_r2:.4f})")

    def _generate_narrative(self, discoveries):
        if not discoveries:
            return "No significant discoveries found in this parameter range."

        lines = [f"## Discovery Campaign: {self.config.question}\n"]
        for i, d in enumerate(discoveries, 1):
            lines.append(f"{i}. **{d.discovery_type.upper()}**: {d.description}")
            if d.critical_value is not None:
                lines.append(f"   Critical value: {d.parameter} = {d.critical_value:.4f}")
            if d.equation:
                lines.append(f"   Equation: {d.equation}")
            if d.r_squared is not None:
                lines.append(f"   R² = {d.r_squared:.4f}")
        return "\n".join(lines)

    def _save_results(self, result: CampaignResult):
        out = {
            "domain": result.config.domain_name,
            "question": result.config.question,
            "n_discoveries": len(result.discoveries),
            "discoveries": [
                {
                    "type": d.discovery_type,
                    "description": d.description,
                    "parameter": d.parameter,
                    "critical_value": d.critical_value,
                    "equation": d.equation,
                    "r_squared": d.r_squared,
                }
                for d in result.discoveries
            ],
            "runtime_seconds": result.runtime_seconds,
            "narrative": result.narrative,
        }
        path = self.output_dir / "campaign_results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")

    def _generate_figures(self, result: CampaignResult):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            for bif in result.bifurcations:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                for ax, (name, series) in zip(axes.flatten(), bif.observable_series.items()):
                    ax.plot(bif.parameter_values, series, "b-", linewidth=1.5)
                    for bp in bif.bifurcation_points:
                        ax.axvline(x=bp.critical_value, color="r", linestyle="--", alpha=0.7)
                    ax.set_xlabel(bif.parameter_name)
                    ax.set_ylabel(name)
                    ax.set_title(name)

                plt.suptitle(f"Discovery Campaign: {self.config.domain_name}", fontsize=14)
                plt.tight_layout()
                fig_path = self.output_dir / "bifurcation_diagram.png"
                plt.savefig(fig_path, dpi=150)
                plt.close()
                logger.info(f"Figure saved to {fig_path}")

        except Exception as e:
            logger.warning(f"Figure generation failed: {e}")
