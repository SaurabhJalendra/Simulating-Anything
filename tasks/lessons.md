# Self-Improvement Log — Simulating Anything

Append-only log of patterns Claude should learn from. After ANY correction from the user or surprising discovery, add an entry here.

> Note: project-specific gotchas and validated lessons live in `.claude/lessons.md` and CLAUDE.md Section 10. This file is for new patterns observed within a working session.

---

## Format
Each entry:

```
## [YYYY-MM-DD] short-title

**Context:** what was being attempted
**What went wrong:** the specific mistake or surprise
**Root cause:** why it happened
**Rule:** preventive rule for future sessions
```

---

## Entries

## [2026-05-01] verify-literature-claims-against-actual-behavior

**Context:** ADR-0001 Change #1 — flip pysindy to ensemble + WSINDy. The research-agent
report claimed "noise tolerance 0.1% to 10-20%" citing Fasel/Kutz and Messenger/Bortz.

**What went wrong:** Initial test asserted ensemble fitting would succeed at 5% noise.
It didn't. Median coefficient error at 5% noise was 50%, not 5%.

**Root cause:** The "10-20% noise tolerance" number is for **WSINDy** (weak-form),
not for ensemble alone. Ensemble-SINDy (Fasel/Kutz 2022) reports ~2x noise tolerance
vs single-fit. Two orders of magnitude difference.

**Rule:** When an agent's research report cites a specific empirical bound,
verify it against actual behavior on project data BEFORE writing tests against
that bound. The literature has multiple methods bundled in similar-sounding
phrases; the bound applies to one specific method, not the family.

## [2026-05-01] pysindy-ensemble-randomness-needs-numpy-seed

**Context:** Tests of ensemble SINDy were intermittent at the same data, same args.

**What went wrong:** EnsembleOptimizer in pysindy 2.1 uses np.random.* internally
without exposing its own RNG. Same code, different runs gave different errors.

**Root cause:** No constructor arg for random_state; uses global numpy state.

**Rule:** For reproducible ensemble SINDy fits, call `np.random.seed(seed)` before
`model.fit(...)`. For multi-seed tests, average across multiple np.random seeds and
assert on median, not single-run.

## [2026-05-01] honest-perf-tradeoff-on-default-changes

**Context:** ADR-0001 Change #1 changes a default (`ensemble=True`). 110+ files
call `run_sindy`.

**What went wrong:** Default ensemble fitting is ~20x slower per call (n_models=20).
A campaign that fit 100 domains in 1 minute now fits in 20 minutes.

**Root cause:** The architectural commitment is correct (ensemble robustness as
default). But changing a default has hidden cost surface area.

**Rule:** When changing a default that affects 100+ callsites, document the
performance trade-off in the commit message AND surface an opt-out path (here:
callers can pass `ensemble=False` for legacy speed).
