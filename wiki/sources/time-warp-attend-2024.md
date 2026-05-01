---
name: Time-Warp-Attend (TWA)
description: Topology-aware CNN with attention that learns the supercritical Hopf normal form once and transfers to any 2-D system. Talmon et al, ICLR 2024.
type: source
sources: ["arXiv:2312.09234"]
last_updated: 2026-04-29
---

# Time-Warp-Attend: Learning Topological Invariants of Dynamical Systems

**Venue:** ICLR 2024
**arXiv:** [2312.09234](https://arxiv.org/html/2312.09234v2)
**Code:** [github.com/nitzanlab/time-warp-attend](https://github.com/nitzanlab/time-warp-attend)
**License:** MIT

## One-line claim

A topology-aware CNN with attention learns the supercritical Hopf normal form via diffeomorphic data augmentation, then transfers to any 2-D dynamical system without retraining.

## Method

1. Train on synthetic trajectories from the Hopf normal form, augmented with arbitrary diffeomorphisms (time-warping, smooth coordinate changes)
2. Render trajectory as 64×64 phase-plane patch
3. CNN with attention outputs Hopf-class probability + bifurcation locus

The diffeomorphic augmentation is the key: it teaches the network the *topological invariants* of Hopf (limit cycle birth, eigenvalue crossing) rather than the geometric specifics of any one system.

## Empirical results

| System | Accuracy at σ=0.1 noise |
|---|---|
| **Mean across 8 test systems** (Brusselator, van der Pol, Selkov glycolysis, Liénard, BZ-class, ...) | **87%** |
| Simple harmonic oscillator | 93% |
| **Real pancreatic single-cell data** | **94%** |

## Compute

- Train: hours on one GPU (one-time cost; pretrained weights ship with code)
- Inference: <1 s per parameter point on a 64×64 patch

## Why it matters for Simulating-Anything

**Direct attack on documented Brusselator 9–28% Hopf detection error** (CLAUDE.md). Brusselator's failure mode is supercritical Hopf with vanishing limit-cycle amplitude near the bifurcation — exactly the case TWA was trained on. The 87% accuracy on Brusselator-class systems at σ=0.1 closes most of that 28% gap.

## Integration plan

In `src/simulating_anything/analysis/bifurcation_detector.py`:

When gradient z-score flags a Hopf candidate at parameter `μ`:
1. Project trajectory onto top-2 PCA dims (or 2 most-oscillatory observables from `observable_extractor.py`)
2. Render 64×64 phase-plane patch
3. Pass through pretrained TWA → Hopf probability `p_TWA`
4. Pass full time series through Bury 2021 deep-EWS classifier → tipping score `s_EWS`
5. **Accept Hopf only if** `p_TWA > 0.85` **AND** `s_EWS rising` **AND** gradient z-score significant

Three orthogonal signals = ensemble that should drive Hopf error below 5%.

## Limitations

- Trained on 2-D systems — for higher-D systems must project first, which can lose information
- Supercritical Hopf only; subcritical and degenerate Hopf less reliable
- Phase-plane patch rendering needs careful normalization (covered in repo)

## Related concepts

- [Hopf bifurcation](#concepts/hopf-bifurcation)
- [Critical slowing down](#concepts/critical-slowing-down)
- [Topological invariants of dynamical systems](#concepts/topological-invariants)

## Related sources

- [Bury et al. 2021 deep EWS](#sources/bury-deep-ews-2021) — complementary classifier
- [TDA Hopf detection (Barrios 2026)](#sources/tda-hopf-2026) — slower backstop using persistent homology

## Related entities

- [Ronen Talmon](#entities/ronen-talmon) — senior author
- [Nitzan lab](#entities/nitzan-lab)
