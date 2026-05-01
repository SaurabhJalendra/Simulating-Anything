# CLAUDE.md -- Simulating Anything

## 0. End Goal (North Star)

**Make genuine scientific discoveries that help humanity.** Not equation fitting.
Not inflating domain counts. Not R² numbers. ACTUAL discoveries:
- Critical thresholds that guide policy (e.g., "polarization > 0.35 prevents epidemic control")
- Phase boundaries that predict regime changes (e.g., "climate coupling > 0.29 causes population collapse")
- Scaling laws that transfer to the real world (e.g., "peak infection is invariant to climate amplitude")

Everything in this project serves this goal. Architecture, world models, SINDy,
exploration — all tools in service of discovery.

---

## 0.1 Autonomous Execution Protocol

Claude Code operates **fully autonomously** on this research. The human sets
direction; Claude executes without stopping.

### Workflow Orchestration (from global CLAUDE.md)
1. **Plan Mode Default** — Enter plan mode for ANY non-trivial task (3+ steps).
   If something goes sideways, STOP and re-plan. Don't keep pushing.
2. **Subagent Strategy** — Offload research, exploration, parallel analysis to
   subagents. For complex problems, throw more compute at it. One task per subagent.
3. **Self-Improvement Loop** — After ANY correction: update lessons. Write rules
   that prevent the same mistake. Ruthlessly iterate until mistake rate drops.
4. **Verification Before Done** — Never mark complete without proving it works.
   Ask: "Would a staff engineer approve this?" Run tests, check logs, demonstrate.
5. **Demand Elegance** — For non-trivial changes: "is there a more elegant way?"
   Skip this for simple fixes. Challenge your own work before presenting it.
6. **Autonomous Bug Fixing** — When given a bug: just fix it. Zero context
   switching required from the user.

### Continuous Improvement Mandate
- After EVERY discovery campaign: assess what worked, what didn't, improve the pipeline
- After EVERY architecture change: verify it improves discovery quality, not just metrics
- After EVERY session: update CLAUDE.md with new gotchas, lessons, and stats
- **Depth over breadth** — 1 genuine discovery > 1000 template domains
- **Honest results only** — Report failures as future work, never inflate numbers
- **Quality bar** — Would this discovery survive peer review at NeurIPS?

### Discovery Loop (Continuous)
```
LOOP FOREVER:
  1. Pick a novel domain with high discovery potential
  2. Ask a scientific question about its coupling behavior
  3. Run discovery campaign: sweep → observe → detect bifurcations → fit scaling laws
  4. Validate: extrapolation, seed robustness, dt invariance
  5. If genuine discovery found → commit with evidence, update paper
  6. If not → analyze why, improve pipeline, try different approach
  7. Log everything to output/discoveries/experiment_log.jsonl
  8. Improve architecture based on what failed
  9. NEVER STOP until the human says stop
```

---

## 1. Project Vision & Research Thesis

**Simulating Anything** is an autonomous scientific discovery engine. Given a
natural language description of any phenomenon, the system automatically builds
a simulation, trains a world model on it, explores the parameter space, and
extracts human-interpretable discoveries -- governing equations, phase
boundaries, scaling laws, and critical thresholds.

**The core claim:** any real-world phenomenon is a dynamical system; any
dynamical system can be simulated; any simulation can train a world model;
and discoveries from world models transfer back to the real world. One pipeline
handles all of science.

**Why this is novel:** No existing system combines world model training +
uncertainty-driven exploration + symbolic regression + multi-agent orchestration
for scientific *discovery* (not control). DreamerV3 uses world models for
policies. We use them for equations and phase boundaries.

**Goals:**
1. **Make genuine scientific discoveries** — Find critical thresholds, phase
   boundaries, and scaling laws in novel coupled systems with no known solutions
2. **Research paper** — AI4Science workshops (NeurIPS/ICML/ICLR) with honest results
3. **Open-source discovery tool** — Scientists input a question, get equations
4. **Continuously improve** — Every session makes the architecture better

**Current State (honest):**
- 261 real domains (14 core + 247 hand-crafted) with genuine physics
- 277 RSSM world models trained on RTX 5090
- 1285 template stress-test domains (appendix, not main results)
- Extrapolation validation shows SINDy does polynomial fitting (limitation)
- Noise tolerance: viable at 0.1%, degrades >5% (honest finding)
- Dream-based discovery: not yet accurate enough (future work)
- **Genuine discoveries: 3 VALIDATED, 89 total across 19 domains**
  - Neuron-Astrocyte: Inverse Hopf at coupling=0.20 (gliotransmitters silence neurons)
  - Social-Epidemic: Inverse Hopf at v_max=0.08 (vaccination threshold)
  - Ocean-Carbon: Inverse Hopf at k_mix=0.022 (mixing stabilizes carbon pump)

---

## 2. What We're Trying to Prove (Rediscovery Targets)

Success means the system autonomously rediscovers known physics across 14
domains spanning 6 mathematical classes -- proving universality with concrete evidence.

### Projectile (rigid body) -- REDISCOVERED
- **Target:** Recover R = v²sin(2θ)/g from simulation data via PySR
- **Result:** PySR found `v0² * 0.1019 * sin(2*theta)` with R² = 0.9999
- The coefficient 0.1019 matches 1/g = 1/9.81 = 0.10194 to 4 significant figures
- 225 data points (15 speeds x 15 angles), simulation error vs theory: 0.04%

### Lotka-Volterra (agent-based) -- REDISCOVERED
- **Target:** Recover equilibrium point (γ/δ, α/β) from population dynamics
- **Result (PySR):** Found `g_/d_` (γ/δ, R²=0.9999) and `a_/b_` (α/β, R²=0.9999)
- **Result (SINDy):** Recovered exact ODE coefficients with R² = 1.0:
  - `d(prey)/dt = 1.100 prey - 0.400 prey*pred` (true: α=1.1, β=0.4)
  - `d(pred)/dt = -0.400 pred + 0.100 prey*pred` (true: γ=0.4, δ=0.1)
- 200 parameter sweeps, time-average error vs theory: 0.31% prey, 0.19% pred

### Gray-Scott (reaction-diffusion) -- ANALYZED
- **Target:** Turing instability threshold, wavelength scaling λ ~ sqrt(D_v)
- **Result:** Phase diagram with 4 pattern types (uniform, spots, stripes, complex)
- 35 Turing instability boundary points mapped in (f, k) space
- Wavelength scaling: correlation with √(D_v) = 0.927
- PySR wavelength equation R² = 0.985 from 9 D_v variation data points

### SIR Epidemic (epidemiological) -- REDISCOVERED
- **Target:** Recover R0 = β/γ and SIR ODEs from simulation data
- **Result (PySR):** Found `b_/g_` (β/γ, R²=1.0) for basic reproduction number
- **Result (SINDy):** Recovered `dR/dt = 0.100*I` exactly (true γ=0.1)
- 200 parameter sweeps covering R0 range [0.33, 40]
- Final epidemic size and peak infected relationships captured

### Double Pendulum (chaotic ODE) -- REDISCOVERED
- **Target:** Energy conservation and small-angle period T = 2π√(L/g)
- **Result:** Energy conservation verified: drift < 1e-7 over 10,000 RK4 steps
- **Result (PySR):** Found `sqrt(L * 4.0298)` with R² = 0.999993
  - Constant 4.0298 matches 4π²/g = 4.0254 (0.1% error)
- 50 energy trajectories, 100 period measurements across L1 range [0.3, 3.0]

### Harmonic Oscillator (linear ODE) -- REDISCOVERED
- **Target:** Recover ω₀ = √(k/m), damping rate = c/(2m), and ODE
- **Result (PySR):** Found `sqrt(k/m)` equivalent with R² = 1.0
- **Result (PySR):** Found `c/(2m)` damping rate with R² = 1.0
- **Result (SINDy):** Recovered `d(v)/dt = -4.000*x - 0.400*v` exactly (k=4, c=0.4)
- 200 frequency measurements, 100 damping measurements

### Lorenz Attractor (chaotic ODE) -- REDISCOVERED
- **Target:** Recover Lorenz ODEs, chaos onset rho_c, Lyapunov exponent
- **Result (SINDy):** Recovered all three Lorenz equations with R² = 0.99999:
  - `d(x)/dt = -9.977 x + 9.977 y` (true: sigma=10)
  - `d(y)/dt = 27.804 x - 0.962 y - 0.994 x*z` (true: rho=28)
  - `d(z)/dt = -2.659 z + 0.997 x*y` (true: beta=8/3=2.667)
- **Chaos transition:** 50-point rho sweep, critical rho ~ 24.4 (true: 24.74)
- **Lyapunov exponent:** 0.9155 at classic parameters (known: 0.9056, 1.1% error)
- 3 fixed points verified, fine Lyapunov sweep with zero-crossing detection

### Navier-Stokes 2D (PDE) -- REDISCOVERED
- **Target:** Viscous decay rate λ = 2ν|k|² = 4ν for Taylor-Green vortex mode (1,1)
- **Simulation:** Vorticity-streamfunction formulation, FFT Poisson solver, 2/3 dealiasing, RK4
- **Result (PySR):** Found `nu * 4.0` with R² = 1.0
  - Coefficient 4.0 = 2|k|² where |k|² = kx² + ky² = 2 for mode (1,1)
- Energy vs analytical: 4.8% mean relative error over 500 steps
- 30 viscosity sweeps, correlation with theory = 1.0

### Van der Pol Oscillator (nonlinear ODE) -- REDISCOVERED
- **Target:** Limit cycle amplitude A~2, period scaling T(mu)
- **Result (PySR):** Period: `mu*1.662 + 8.09 - sqrt(sqrt(mu))*3.16` R²=0.99996
  - Coefficient 1.662 close to theoretical (3-2ln(2)) = 1.614 for large mu
- Mean amplitude = 2.0098 (theory: 2.0 exact)
- 30 mu values from 0.1 to 31.6, period range [6.3, 53.1]

### Kuramoto Coupled Oscillators (collective dynamics) -- REDISCOVERED
- **Target:** Synchronization transition r(K), critical coupling K_c
- **Result (PySR):** Found `sqrt(K / (K + (((K-2.77)/K)^2)^2))` with R² = 0.9695
- K_c estimate: 1.10 (theory: 4/pi = 1.27, 14% error -- finite-size effect)
- 40-point K sweep, max order parameter r = 0.989
- Finite-size scaling: N = [10, 20, 50, 100, 200, 500]

### Brusselator (chemical oscillator) -- REDISCOVERED
- **Target:** Hopf bifurcation b_c = 1 + a², ODE recovery
- **Result (PySR):** Found `a² + 0.911` with R² = 0.9960 (theory: b_c = 1 + a²)
  - Best expression: `(a-0.119/a)² + 1.131` with R² = 0.9964
- **Result (SINDy):** Recovered both ODEs with R² = 0.9999:
  - `d(u)/dt = -3.686u + 0.513v - 0.070v² + 0.960u²v`
  - `d(v)/dt = 3.000u - 1.000u²v` (true: b=3, u²v term)
- b_c estimate: 1.948 (theory: 1+1²=2.0, 2.6% error)

### FitzHugh-Nagumo (neuroscience) -- REDISCOVERED
- **Target:** ODE recovery and f-I curve
- **Result (SINDy):** Recovered exact ODE coefficients with R² = 0.99999999:
  - `d(v)/dt = 0.500 + 1.000v - 1.000w - 0.333v³` (true: I=0.5, v-v³/3-w+I)
  - `d(w)/dt = 0.056 + 0.080v - 0.064w` (true: eps*(v+a-b*w), eps=0.08, a=0.7, b=0.8)
- **f-I curve:** Critical current I_c ~ 0.362, max firing frequency 0.027
- 21 oscillatory I values detected across sweep

### Heat Equation 1D (pure diffusion PDE) -- REDISCOVERED
- **Target:** Mode decay rate λ_k = D*k²
- **Result (PySR):** Found `D` with R² = 1.0 for mode k=1 on [0,2π]
  - Decay rate = D matches theory exactly (k=2π/L=1, so D*k²=D)
- Mean relative error: 1.5e-13 (machine precision, spectral solver is exact)
- 25 diffusion coefficient sweeps, correlation = 1.0

### Logistic Map (discrete chaos) -- ANALYZED
- **Target:** Feigenbaum delta~4.669, chaos onset r_c~3.57, Lyapunov at r=4
- **Bifurcation:** 4 period-doubling points detected at r = [2.99, 3.45, 3.54, 3.57]
  - Feigenbaum delta estimates: [4.75, 4.0] (theory: 4.669)
- **Chaos onset:** r_c estimate = 3.576 (theory: 3.5699, 0.2% error)
- **Lyapunov:** Max = 1.386 at r=4 (exact: ln(4) = 1.386, from all-positive orbit)
- **PySR Lyapunov fit:** `r*216.1 * (r/617.7 - 0.0056)` R² = 0.629
  - Chaotic Lyapunov spectrum is fractal -- low R² expected and informative

---

## 3. The Universality Argument

Only the `SimulationEnvironment` subclass is domain-specific. Everything
else -- problem parsing, world model, exploration, analysis, reporting --
operates on generic tensors. Adding a domain = one new class (~50-200 lines).

**Cross-domain analogy engine** detects 586+ mathematical isomorphisms across 200+ domains.
The project now spans 1500+ domains. Representative analogies:
- LV ↔ SIR (bilinear interaction terms)
- Pendulum ↔ Oscillator (harmonic restoring force, T ~ √(inertia/force))
- Projectile ↔ Oscillator (energy conservation)
- Gray-Scott wavelength ↔ Oscillator period (same dimensional scaling)
- Lorenz ↔ Double Pendulum (chaotic ODEs with strange attractors)
- Gray-Scott ↔ Navier-Stokes (PDE diffusion operators)
- VdP ↔ Lotka-Volterra (limit cycles)
- Brusselator ↔ VdP (Hopf bifurcation)
- FHN ↔ VdP (same mathematical origin)
- Heat equation ↔ NS (linear vs nonlinear diffusion)
- Logistic map ↔ Lorenz (chaos, positive Lyapunov)
- Kuramoto ↔ SIR (threshold/phase transitions)
- Cart-pole ↔ Double pendulum (Lagrangian coupled DOFs)
- Cart-pole ↔ Harmonic oscillator (linearized small-angle oscillation)
- Three-species ↔ LV (trophic cascade extension)
- Three-species ↔ SIR (3-compartment coupled nonlinear ODEs)
- Elastic pendulum ↔ Harmonic oscillator (radial mode omega_r=sqrt(k/m))
- Rossler ↔ Lorenz (3D chaotic attractors)
- Brusselator-diffusion ↔ Gray-Scott (Turing instability RD-PDEs)
- Henon map ↔ Logistic map (discrete chaotic maps)

Full argument with 40+ concrete domains: `docs/RESEARCH.md` Section 4.
Domain expansion architecture: `docs/DESIGN.md` Section 11.

---

## 4. Setup & Environment

- **Python 3.12** on Windows 11
- **JAX GPU requires WSL2 Ubuntu 24.04** (JAX CUDA doesn't run on native Windows)
- **RTX 5090 32GB** visible as `cuda:0` inside WSL
- **Venv:** `.venv` in project root (WSL path: `/mnt/d/Git Repos/Simulating-Anything/.venv`)

### Install (inside WSL)

```bash
cd /mnt/d/'Git Repos'/Simulating-Anything
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install "jax[cuda12]" equinox optax diffrax pandas
```

### Additional Dependencies

- **Julia** needed for PySR symbolic regression (TODO: not yet installed)
- **Claude Code CLI** in WSL for LLM agents: `sudo npm install -g @anthropic-ai/claude-code`
- **Node.js 22** in WSL: `curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs`

---

## 5. GPU & Training

**ALL JAX/GPU work MUST run through WSL2.** Native Windows = CPU only.

- RTX 5090 (32GB VRAM) handles all V1 workloads locally -- no cloud GPU needed
- RSSM observe step: ~6ms/step, dream step: ~20ms/step on RTX 5090
- World model training: ~7 min/domain on RTX 5090 (200 epochs)
- 277 RSSM world models trained, median dream MSE=0.07

### WSL Command Pattern

```bash
wsl.exe -d Ubuntu -- bash -lc "cd '/mnt/d/Git Repos/Simulating-Anything' && source .venv/bin/activate && <your_command>"
```

Never use `wsl.exe -d Ubuntu -e bash -c "..."` — Windows PATH with parentheses breaks it.
Never fall back to CPU for training or pipeline runs. Always use WSL2.

---

## 6. Key Commands

### Tests
```bash
# Full suite in WSL (7900+ passing, 380 skipped):
wsl.exe -d Ubuntu -- bash -lc "cd '/mnt/d/Git Repos/Simulating-Anything' && source .venv/bin/activate && python3 -m pytest tests/unit/ -v"

# Windows (CPU only, world model tests also pass):
python -m pytest tests/unit/ -v
```

### Pipeline
```python
# Must run inside WSL for GPU + Claude Code CLI
from simulating_anything import Pipeline
pipeline = Pipeline()
report = pipeline.run("How do patterns form in a two-chemical activator-inhibitor system?")
```

### Rediscovery
```python
# Run all fourteen domain rediscoveries (requires WSL + Julia + PySR)
from simulating_anything.rediscovery.runner import run_all_rediscoveries
results = run_all_rediscoveries(pysr_iterations=50)

# Or run individually:
from simulating_anything.rediscovery.projectile import run_projectile_rediscovery
from simulating_anything.rediscovery.lotka_volterra import run_lotka_volterra_rediscovery
from simulating_anything.rediscovery.gray_scott import run_gray_scott_analysis
```

### Lint
```bash
ruff check src/ tests/
```

---

## 7. Architecture Quick Reference

### 7-Stage Pipeline
```
Problem Architect (LLM) → Domain Classifier (rules + LLM fallback)
  → Simulation Builder (LLM) → Ground-Truth Simulation (JAX)
  → Exploration (uncertainty-driven) → Analysis (PySR + SINDy + ablation)
  → Communication Agent (LLM) → Markdown Report
```

### Key Patterns

**Simulation:** Subclass `SimulationEnvironment` from `simulation/base.py`:
- `reset(seed) -> np.ndarray` -- initial state
- `step() -> np.ndarray` -- advance one timestep
- `observe() -> np.ndarray` -- current observable state
- `run(n_steps) -> TrajectoryData` -- collect full trajectory

**Agent:** Subclass `Agent` from `agents/base.py`. LLM agents use
`ClaudeCodeBackend` which calls Claude Code CLI via subprocess:
```python
cmd = ["claude", "-p", prompt, "--output-format", "json"]
```
System prompt is prepended to user prompt (CLI has no --system flag).

**Types:** All Pydantic v2 BaseModel in `src/simulating_anything/types/`:
- `ProblemSpec` -- parsed problem definition
- `SimulationConfig` -- domain, parameters, grid, dt
- `TrajectoryData` -- states array + metadata
- `Discovery` -- equation, confidence, evidence

**Config:** YAML in `configs/`, loaded via `load_config()` / `load_domain_config()`
from `utils/config.py`.

**World Model:** RSSM (Equinox) in `world_model/rssm.py`:
- 512 GRU deterministic + 32×32 categorical stochastic = 1536 latent dims
- Encoder: CNN (spatial) or MLP (vector)
- Decoder: Transposed CNN or MLP with symlog output
- Trainer in `world_model/trainer.py`: symlog MSE + KL loss, Adam + cosine decay

---

## 8. Code Style

- `from __future__ import annotations` in every file
- **Ruff:** line-length 99, target py311, select E/F/I/W
- Type hints on all functions using `|` union syntax (not `Optional`)
- Google-style docstrings
- No emojis in code or documentation
- No `Co-Authored-By` lines in git commits

---

## 9. Git Conventions

- **Branch:** `main` only (no feature branches in V1)
- **Commit after** each logical checkpoint, push immediately
- **Message style:** Imperative ("Add X", "Fix Y"), descriptive body
- **Remote:** `https://github.com/SaurabhJalendra/Simulating-Anything.git`
- **User:** SaurabhJalendra / saurabh@users.noreply.github.com

---

## 10. Critical Gotchas

These are things that broke in previous sessions. Do not repeat them:

| Issue | What Went Wrong | Correct Approach |
|-------|----------------|-----------------|
| JAX on Windows | No GPU support | Always use WSL2 |
| Gray-Scott NaN | dt too large for grid | CFL: `dt < dx²/(4·D_max)`. 128×128 grid → dt < 0.0006 |
| Lotka-Volterra hang | diffrax called per-step (5000× JIT overhead) | RK4 for `step()`, diffrax only in `solve_trajectory()` |
| Claude CLI crash | `--system` flag doesn't exist | Prepend system prompt to user prompt |
| TrajectoryData | Tried to index it like array | `run()` returns Pydantic object, use `.states` for numpy |
| RSSM action | Passed `jnp.zeros(1)` with action_size=0 | Use scalar `jnp.float32(0)` for no-action case |
| Parquet load | Missing pandas | `pip install pandas` (pyarrow alone can't do `to_pandas()`) |
| Projectile params | Used `v0` instead of `initial_speed` | Check exact param names in simulation `__init__` |
| PySR var names | `alpha`, `beta` conflict with sympy | Use `a_`, `b_`, `g_`, `d_` as PySR variable names |
| PySINDy v2.1.0 | `feature_names` moved from `__init__` to `fit()` | Pass `feature_names` to `model.fit()`, not `SINDy()` |
| PySR `variable_names` | FutureWarning in PySR 1.5.9 | Pass `variable_names` to `model.fit()`, not constructor |
| Gray-Scott convention | Pearson D_u=2e-5 gives unresolvable wavelengths | Use Karl Sims convention: D_u=0.16, D_v=0.08, unscaled Laplacian |
| WSL bash -c PATH | Windows PATH with parentheses breaks bash -c | Use `wsl.exe -d Ubuntu -- bash -lc "..."` instead |
| PySINDy differentiate | `model.differentiate()` doesn't exist in v2.1 | Use `np.gradient()` for derivatives, pass as `x_dot` to `fit()` |
| RSSM model loading | Tried standalone `RSSM(obs_shape=...)` | Use `WorldModelTrainer` to rebuild `(encoder, rssm, decoder)` tree, then `eqx.tree_deserialise_leaves` |
| RSSM constructor | Passed `obs_shape` to `RSSM.__init__` | RSSM takes `(action_size, embed_size, hidden_size, stoch_vars, stoch_classes, key)` — NOT obs_shape |
| SINDy threshold | Threshold 0.01 kills small coefficients | Lower to 0.001-0.0001 for slow dynamics (battery, soil, aquifer) |
| GPU contention | 2 JAX training processes share 32GB | Train sequentially; kill redundant processes with `kill PID` |
| Sim class names | Assumed all classes end in "Simulation" | Varies: `DoublePendulumSimulation`, `KeplerOrbit`, `ElasticPendulum`, `CoupledOscillators`, `DrivenPendulum` |
| Batch SINDy | Some auto-generated domains overflow (NaN) | Filter `np.isnan/np.isinf` before fitting; use `np.clip(state, 0, None)` in step() |

---

## 11. Roadmap

### Completed (V2-V8)
- **V2-V6:** 14 core rediscoveries, 192 hand-crafted domains, world models, exploration — DONE
- **V7:** 5 novel coupled systems, CKA latent analysis, cross-domain transfer — DONE
- **V8:** Scaled to 261 real domains + 1285 template stress tests, 277 world models — DONE
- **Honest Assessment:** Domain classification (261 real, 1285 template), extrapolation
  validation (SINDy degrades at 1.5-3x range), noise robustness (viable at 0.1%), dream
  discovery (not yet accurate enough) — ALL DONE

### V9 (Genuine Discovery) — IN PROGRESS
The pivot from breadth to depth. Make real discoveries.

**Phase 9A: Discovery Infrastructure**
- [ ] Observable Extractor (mean, std, amplitude, period, Lyapunov, classification)
- [ ] Bifurcation Detector (gradient z-scores, type classification, confidence intervals)
- [ ] Phase Diagram Generator (2D classification grid, boundary detection, figures)
- [ ] Discovery Campaign Runner (sweep → observe → detect → fit → validate)

**Phase 9B: Discovery Campaigns (3 targets)**
- [ ] Social-Epidemic: polarization threshold for epidemic control
- [ ] Climate-Epidemic: phase-locking mechanism, invariant peak magnitude
- [ ] Predator-Prey-Climate: bifurcation type at coupling=0.29, prey extinction threshold

**Phase 9C: Validation & Improvement Loop**
- [ ] Extrapolation validation on all discoveries
- [ ] Seed robustness (5 seeds), dt invariance (3 values)
- [ ] Improve pipeline based on what fails
- [ ] Run on next 5 novel domains from the 35 hand-crafted set

**Phase 9D: Paper Rewrite**
- [ ] Honest domain count (261 real)
- [ ] Discovery results as main contribution
- [ ] Limitations section (extrapolation, noise, dreams)
- [ ] End-to-end autonomous demo

### V10 (Autoresearch Integration) — PLANNED
- [ ] Fixed-budget experiment loops (autoresearch pattern)
- [ ] Per-domain SINDy threshold optimization
- [ ] Per-domain-class RSSM architecture search
- [ ] Overnight autonomous discovery loop

### Current Stats
- **261 real domains** (14 core + 247 hand-crafted) — verified by `scripts/classify_domains.py`
- **1285 template domains** (stress test only, NOT main results)
- **281 RSSM world models** on RTX 5090
- **316 discovery campaigns** across 35+ novel coupled systems
- **1267 discoveries** (bifurcations + scaling laws + phase boundaries + 43 2D phase diagrams)
- **118 validated bifurcations** (55 calibrated with literature params, 5-seed unanimous)
- **11 policy equations / structural invariants**
- **Literature calibration**: NPB burst 7.2% error, Brusselator 9-28%, Tumor 24%
- **dt-invariance**: 24/51 pass <5% deviation (47%), battery perfect
- **6 testable predictions** (3 high confidence, 2 medium, 1 validated)
- **Paper:** needs rewrite with calibration + prediction contributions

---

## 12. Directory Map

```
src/simulating_anything/
  __init__.py              # Exports Pipeline, __version__
  pipeline.py              # 7-stage orchestrator (entry point)
  agents/
    base.py                # ClaudeCodeBackend + Agent ABC
    problem_architect.py   # NL → ProblemSpec (LLM)
    domain_classifier.py   # Rules + LLM fallback → Domain
    simulation_builder.py  # Domain → SimulationConfig (LLM)
    communicator.py        # DiscoveryReport → Markdown (LLM)
  simulation/
    base.py                # SimulationEnvironment ABC
    # --- 1498 simulation files total (see directory for full list) ---
    # Core V1-V6 domains (192 hand-crafted with domain-specific physics):
    lorenz.py, navier_stokes.py, harmonic_oscillator.py, sir_epidemic.py,
    lotka_volterra.py, brusselator.py, fitzhugh_nagumo.py, heat_equation.py,
    van_der_pol.py, kuramoto.py, logistic_map.py, duffing.py, rossler.py,
    chua.py, hodgkin_huxley.py, lorenz96.py, ising_model.py, ... (192 total)
    # V7 novel coupled systems (5 domains):
    stochastic_resonance.py, replicator_mutator.py, lorenz_stommel.py,
    climate_epidemic.py, neural_cardiac.py
    # V8 novel coupled systems (35+ hand-crafted with real physics):
    predator_prey_climate.py, epidemic_economy.py, neural_ecosystem.py,
    tumor_immune.py, gene_metabolism.py, plankton_ocean.py,
    social_epidemic.py, predator_prey_pollution.py, circadian_metabolism.py,
    prey_disease_predator.py, vegetation_hydrology.py, neuron_astrocyte.py,
    infection_immunity.py, resource_consumer_waste.py, laser_absorber.py,
    atmosphere_vegetation.py, earthquake_aftershock.py, calcium_signaling.py,
    ocean_carbon.py, hormone_glucose.py, soil_carbon_nitrogen.py,
    dopamine_reward.py, coral_reef.py, antibiotic_resistance.py,
    forest_fire.py, supply_chain.py, urban_heat_island.py, ...
    # V8 batch-generated domains (1000+ with unique coefficients):
    # Spanning ecology, medicine, engineering, climate, geology, etc.
    composable.py          # ComposedSimulation + 12 DynamicsModules
    equation_parser.py     # EquationSimulation from NL equations
    external_bridge.py     # File/Socket/Subprocess/PythonModule bridges
  world_model/
    rssm.py                # RSSM (Equinox) — 1536 latent dims
    rssm_v2.py             # RSSMv2 (DreamerV4-style mixed stochastic)
    ensemble.py            # EnsembleRSSM (epistemic uncertainty)
    encoder.py             # CNNEncoder, MLPEncoder
    decoder.py             # CNNDecoder, MLPDecoder, symlog
    advanced_encoders.py   # GNNEncoder, CNN3DEncoder, SetEncoder
    trainer.py             # WorldModelTrainer (Adam + cosine)
    trainer_v2.py          # WorldModelTrainerV2 (RSSMv2 trainer)
  exploration/
    base.py                # Explorer ABC
    uncertainty_driven.py  # MC-dropout uncertainty explorer
  analysis/
    symbolic_regression.py # PySR wrapper (variable_names in fit())
    equation_discovery.py  # PySINDy wrapper (v2.1.0 API)
    ablation.py            # Single-factor ablation studies
    pipeline_ablation.py   # Pipeline component ablation (sampling, method, data)
    sensitivity.py         # Noise/data/range sensitivity analysis
    cross_domain.py        # Cross-domain analogy engine (365 isomorphisms)
    dream_debate.py        # Adversarial dream debate (divergence metrics)
    domain_statistics.py   # Runtime benchmarks for all domains
    error_analysis.py      # Bootstrap R², coefficient uncertainty
    scaling_analysis.py    # Runtime vs steps, dimension, data quantity
    baselines.py           # Formal baseline comparisons (5 domains, LaTeX)
    significance.py        # Permutation tests, bootstrap CI, Wilcoxon, Cohen's d
    robustness.py          # Noise tolerance, sample efficiency, extrapolation
    computational_cost.py  # Wall-clock timing per pipeline stage
  rediscovery/
    __init__.py            # Exports all rediscovery runners
    runner.py              # Unified runner for all domains
    # 192 core rediscovery scripts (projectile.py through neural_cardiac.py)
    # + 35 V8 novel domain scripts (predator_prey_climate.py, etc.)
    # Results saved to output/rediscovery/{domain}/results.json
  knowledge/
    trajectory_store.py    # Parquet + JSON sidecar storage
    discovery_log.py       # JSONL discovery persistence
    knowledge_base.py      # Persistent knowledge across sessions
  discovery/
    open_problems.py       # 6 open problems registry
    discovery_runner.py    # Parameter sweep + equation fitting
  verification/
    dimensional.py         # Dimensional analysis checks
    conservation.py        # Mass, energy, positivity, boundedness
    transfer_validation.py # Sim-to-real validation (12 metrics)
    simulation_validator.py # 7-check auto-sim validation
  types/
    problem_spec.py        # ProblemSpec, Variable, Objective
    simulation.py          # SimulationConfig, Domain (incl CUSTOM, EXTERNAL)
    trajectory.py          # TrajectoryData, TrajectoryMetadata
    discovery.py           # Discovery, Evidence, DiscoveryReport
    campaign.py            # Experiment, ResearchPlan, CampaignReport
  utils/
    config.py              # load_config(), load_domain_config()
  campaign/
    manager.py             # CampaignManager (full autonomous loop)
    notebook.py            # ResearchNotebook (append-only markdown log)
  agents/
    simulation_generator.py # LLM generates simulation code
    research_planner.py    # Decomposes questions into experiments

configs/
  default.yaml             # Global defaults
  domains/
    reaction_diffusion.yaml
    rigid_body.yaml
    agent_based.yaml

tests/unit/                # 7900+ tests across 233 files
  # 150+ test files for V1-V6 domains (test_lorenz.py, test_navier_stokes.py, etc.)
  # V8 test files:
  test_novel_v8_domains.py      # 24 tests — PPC, EpiEcon, NeuEco
  test_novel_v8_extended.py     # 21 tests — TumorImmune, GeneMet, etc.
  test_novel_v8_batch3.py       # 19 tests — Laser, Battery, Infection, etc.
  # Infrastructure tests:
  test_types.py, test_config.py, test_pipeline.py, test_world_model.py,
  test_reproducibility.py, test_cross_domain.py, test_composable.py,
  test_advanced_encoders.py, test_knowledge_base.py

output/rediscovery/          # 1497 rediscovery results (not committed to git)
  {domain}/results.json      # SINDy/PySR equations + R² for each domain

output/world_models/         # 238 trained RSSM checkpoints
  {domain}/model.eqx         # Equinox model weights (all ~32.0 loss)
  {domain}/training_results.json  # Loss, dream MSE, training time
  {domain}/dream_comparison.npz   # Ground truth vs dreamed trajectories
  training_summary_all.json  # Comprehensive summary of all models

scripts/
  train_world_models_generic.py  # Train RSSM on any domain (auto-discovery)
  results_dashboard.py       # Comprehensive results dashboard
  latent_space_analysis.py   # CKA + PCA latent space analysis
  cross_domain_transfer.py   # Cross-domain world model transfer
  dream_accuracy_analysis.py # Dream quality across all models
  ensemble_uncertainty_analysis.py # Ensemble disagreement
  generate_figures.py        # Publication-quality figures
  run_everything.py          # One-command reproduction of all results

docs/
  RESEARCH.md              # Vision, universality argument (Section 4), contributions
  DESIGN.md                # Architecture, domain expansion (Section 11), evaluation
  conf.py                  # Sphinx configuration
  index.rst                # Documentation index
  quickstart.rst           # Installation and usage guide
  architecture.rst         # System architecture overview
  domains.rst              # All 192 domains listed
  api/                     # API reference (autodoc)

Dockerfile                 # Python 3.12-slim container
.dockerignore              # Docker build exclusions

notebooks/
  demos/demo.ipynb           # Three-domain demo
  rediscovery_results.ipynb  # Flagship 5-domain notebook (43 cells, 14 figures)
  seven_domain_rediscovery.ipynb # 7-domain notebook (35 cells)
  world_model_training.ipynb # RSSM training results
  cross_domain_analysis.ipynb # Cross-domain comparison
  showcase_14domain.ipynb    # 14-domain interactive showcase (24 cells)
```

---

## 13. LLM Wiki Operations (Karpathy Pattern)

This project uses the Karpathy LLM Wiki pattern for accumulating external knowledge (papers, articles, references). `raw/` holds immutable sources; `wiki/` holds LLM-maintained knowledge.

### Structure
```
raw/                    # Immutable source documents (papers, articles, clippings)
  assets/              # Images, PDFs, referenced files
wiki/                   # LLM-maintained markdown knowledge base
  index.md             # Content catalog (all pages, by category)
  log.md               # Append-only activity record with timestamps
  entities/            # People, organizations, products, institutions
  concepts/            # Ideas, theories, dynamical-systems methods
  sources/             # One page per ingested source (summary + key takeaways)
  syntheses/           # Cross-cutting analyses, comparisons, theses
```

### Ingest (when user drops a file in raw/)
1. Read the source thoroughly
2. Write a summary page in `wiki/sources/[source-name].md`
3. Update 10-15 related pages in `wiki/entities/` and `wiki/concepts/`
4. Update `wiki/index.md` with new entries
5. Append entry to `wiki/log.md`: `## [YYYY-MM-DD] ingest | [source name]`
6. Report to user: what was ingested, which pages were updated

### Query (when user asks a question)
1. Read `wiki/index.md` first to find relevant pages
2. Drill into those pages
3. Synthesize an answer with citations to source pages
4. **If the answer is valuable**, file it back as a new page in `wiki/syntheses/`
5. Append to `wiki/log.md`: `## [YYYY-MM-DD] query | [topic]`

### Lint (periodic health check — user runs /wiki-lint)
Check for:
- Contradictions between pages (flag for user decision)
- Stale claims that newer sources have superseded
- Orphan pages with no inbound links
- Important concepts mentioned but lacking their own page
- Missing cross-references between related pages
- Data gaps that could be filled with web search

### Rules
- Never modify files in `raw/` — those are immutable
- Every wiki page should link to its sources
- Use YAML frontmatter on wiki pages: name, description, type, sources, last_updated
- Keep `index.md` under 200 lines — one line per page
- For this project, prioritize ingesting: dynamical-systems papers, scientific discovery papers, world-model literature, symbolic-regression methods

