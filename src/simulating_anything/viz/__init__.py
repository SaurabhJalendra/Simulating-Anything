"""Publication-quality visualization functions."""

from __future__ import annotations

from simulating_anything.viz.figures import (
    plot_dream_comparison,
    plot_gs_pattern_gallery,
    plot_gs_phase_diagram,
    plot_gs_wavelength_scaling,
    plot_lv_equilibrium_fit,
    plot_lv_phase_portrait,
    plot_lv_sindy_comparison,
    plot_projectile_equation_fit,
    plot_projectile_trajectories,
    plot_rediscovery_summary,
    plot_training_curves,
    setup_paper_style,
)

__all__ = [
    "setup_paper_style",
    "plot_projectile_trajectories",
    "plot_projectile_equation_fit",
    "plot_lv_phase_portrait",
    "plot_lv_equilibrium_fit",
    "plot_lv_sindy_comparison",
    "plot_gs_phase_diagram",
    "plot_gs_wavelength_scaling",
    "plot_gs_pattern_gallery",
    "plot_training_curves",
    "plot_dream_comparison",
    "plot_rediscovery_summary",
]
