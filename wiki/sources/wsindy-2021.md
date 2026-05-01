---
name: Weak SINDy (WSINDy)
description: Weak-form SINDy variant that replaces pointwise derivatives with Galerkin integration; consistent at 10-20% noise. Messenger & Bortz, 2021.
type: source
sources: ["SIAM MMS 2021", "JCP 2021"]
last_updated: 2026-04-29
---

# Weak SINDy / WSINDy

**Authors:** Daniel A. Messenger, David M. Bortz
**Venues:**
- ODE version: SIAM Multiscale Modeling & Simulation 2021 — [doi:10.1137/20M1343166](https://epubs.siam.org/doi/10.1137/20M1343166)
- PDE version: Journal of Computational Physics 2021 — [PMC8570254](https://pmc.ncbi.nlm.nih.gov/articles/PMC8570254/)
**Code:** [github.com/MathBioCU/WSINDy_ODE](https://github.com/MathBioCU/WSINDy_ODE), `WSINDy_PDE` (MATLAB primary; Python ports exist; integrated into `pysindy` as `WeakPDELibrary`)

## One-line claim

Replace pointwise derivative estimation in SINDy with Galerkin-style integration against compactly-supported test functions; FFT-accelerated; coefficient identification reliable at 10–20% noise (vs vanilla SINDy's 0.01–0.05 ceiling).

## Why it works

Pointwise derivatives blow up under noise. Integration against smooth test functions transfers the derivative onto the test function (integration by parts), which is computed analytically — exactly. Result: the noise-multiplication factor that kills SINDy is gone.

## Empirical results

- **Asymptotically consistent** for Navier-Stokes, Kuramoto-Sivashinsky
- **Noise tolerance:** reliable identification at noise ratios > 0.1 (vanilla SINDy fails here)
- **Coefficient error scales linearly with noise** (vs SINDy's quadratic)
- **Compute:** FFT-fast, comparable to or cheaper than vanilla SINDy

## Why it matters for Simulating-Anything

**Direct fix for the documented honest limitation** (CLAUDE.md): "Noise tolerance: viable at 0.1%, degrades >5%". WSINDy lifts the ceiling 100×.

**Zero-migration adoption path:** already integrated into `pysindy` as `WeakPDELibrary`. Single import + config change in `src/simulating_anything/analysis/equation_discovery.py`.

```python
# Before
model = pysindy.SINDy(optimizer=STLSQ(threshold=0.001))

# After
model = pysindy.SINDy(
    feature_library=WeakPDELibrary(...),
    optimizer=EnsembleOptimizer(STLSQ(threshold=0.001), n_models=200),
)
```

## Stacking with Ensemble-SINDy

Combining WSINDy with [Ensemble-SINDy](#sources/ensemble-sindy-2022) (Fasel/Kutz, Proc. Roy. Soc. A 2022) gives:
- Noise robustness (WSINDy)
- Library-term inclusion probabilities + coefficient uncertainty (Ensemble)
- Trivially parallel — N× SINDy where N = bootstrap count

Both ship in the same `pysindy` library. Set `ensemble=True, n_models=200, library_ensemble=True` in `model.fit()`.

## Limitations

- Same library-based ceiling as SINDy — if the true equation isn't expressible in the candidate library, WSINDy can't recover it (this is the "Brusselator Hopf needs sqrt scaling" failure mode → use [KAN-SR](#sources/kan-sr-2025) for those cases)
- Test-function selection requires care; defaults work well empirically

## Related concepts

- [SINDy](#concepts/sindy) — the parent algorithm
- [Ensemble-SINDy](#concepts/ensemble-sindy)
- [Symbolic regression](#concepts/symbolic-regression)

## Related entities

- [Daniel Messenger](#entities/daniel-messenger)
- [David Bortz](#entities/david-bortz)
- [MathBioCU](#entities/mathbiocu)
