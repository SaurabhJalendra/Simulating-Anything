---
name: The Well
description: 15 TB collection of 16 diverse physics simulations released by Polymathic AI for ML training. NeurIPS 2024.
type: source
sources: ["arXiv:2412.00568"]
last_updated: 2026-04-29
---

# The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning

**Authors:** Ohana et al.
**Venue:** NeurIPS 2024 Datasets & Benchmarks Track
**arXiv:** [2412.00568](https://arxiv.org/abs/2412.00568)
**Project:** [polymathic-ai.org/the_well](https://polymathic-ai.org/the_well/)
**Code:** [github.com/PolymathicAI/the_well](https://github.com/PolymathicAI/the_well)
**Hosting:** HuggingFace `polymathic-ai/the-well` + S3
**License:** CC BY 4.0

## One-line claim

Largest open collection of high-fidelity physics simulations curated for ML training, designed to enable cross-domain transfer studies.

## Scale

- **Total:** 15 TB across 16 datasets
- **Per-dataset:** 6.9 GB to 5.1 TB
- **Format:** uniform-grid HDF5 with self-documenting metadata
- **Access:** streaming via `the-well-download` CLI; HF `datasets` library

## 16 datasets (domain coverage)

Spans linear wave → reaction-diffusion → radiative → relativistic MHD:

- Active matter / biological systems
- Acoustic scattering
- MHD of extra-galactic fluids
- Supernova explosions
- Shear flow
- Reaction-diffusion (Gray-Scott class)
- Turbulent hydrodynamics
- Radiative cooling
- Relativistic MHD
- Gravitational collapse
- Viscoelastic flows
- Convective envelopes (stellar)
- Helmholtz instability
- Planetary atmosphere
- Post-neutron-star merger
- Rayleigh-Taylor / Rayleigh-Bénard

## Benchmarks established

- FNO, U-Net, CNextU-Net baselines
- 2025 SOTA: PDE-FM (arXiv:2511.21861) — 46% VRMSE reduction on 6/12 domains
- Also benchmarked: PDE-Transformer (TUM), PhysiX (NeurIPS ML4PS 2025)

## Why it matters for Simulating-Anything

The project's **honest weakness** (per CLAUDE.md): "1285 template stress tests" alongside 261 real domains — reviewers will catch this. The Well is the cheapest external grounding that converts the universality argument from "we built domains that look similar" to "our SINDy/PySR pipeline rediscovers known physics on the same fields the foundation-model community trains on."

**Recommended subset to ingest first:**
- **Rayleigh-Bénard** (~500 GB) — overlaps with project's existing reaction-diffusion + Navier-Stokes 2D
- **MHD** (~1 TB) — adds a new physics class entirely

## Integration plan for this project

1. New `src/simulating_anything/simulation/well_bridge.py` — wraps HF streaming as `SimulationEnvironment` subclass (~150 LOC, extends pattern in `external_bridge.py`)
2. HDF5 → `TrajectoryData` schema mapper in `types/trajectory.py`
3. Add HDF5 reader sibling to `knowledge/trajectory_store.py`
4. Adopt MPP's evaluation protocol for direct paper-grade comparison

## Related entities

- [Polymathic AI](#entities/polymathic-ai) — releasing organization
- [Flatiron Institute](#entities/flatiron-institute) — Polymathic's home institution

## Related concepts

- [Multi-physics pretraining](#concepts/multi-physics-pretraining) — MPP, the canonical use of this dataset
- [Foundation models for science](#concepts/foundation-models-for-science)
- [PDE benchmarks](#concepts/pde-benchmarks)

## Related sources

- [PDEBench](#sources/pdebench-2022) — the predecessor benchmark
- [MPP (Multiple Physics Pretraining)](#sources/mpp-2024) — uses Well for cross-domain transfer
