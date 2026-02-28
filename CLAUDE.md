# CLAUDE.md -- Simulating Anything

## 1. Project Vision & Research Thesis

**Simulating Anything** is a multi-agent scientific discovery engine. Given a
natural language description of any phenomenon, the system automatically builds
a simulation, trains a world model on it, explores the parameter space, and
extracts human-interpretable discoveries -- governing equations, phase
boundaries, scaling laws, and optimal strategies.

**The core claim:** any real-world phenomenon is a dynamical system; any
dynamical system can be simulated; any simulation can train a world model;
and discoveries from world models transfer back to the real world. One pipeline
handles all of science.

**Why this is novel:** No existing system combines world model training +
uncertainty-driven exploration + symbolic regression + multi-agent orchestration
for scientific *discovery* (not control). DreamerV3 uses world models for
policies. We use them for equations.

**Goals:** Research paper (AI4Science workshops at NeurIPS/ICML/ICLR),
open-source tool for scientists, and portfolio piece demonstrating ML research
capability.

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

**Cross-domain analogy engine** detects 17 mathematical isomorphisms across 14 domains:
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
- Full pipeline (with Claude Code CLI): ~6.5 minutes end-to-end
- World model training target: < 60 minutes on RTX 5090

### WSL Command Pattern

```bash
wsl.exe -d Ubuntu -e bash -c "cd /mnt/d/'Git Repos'/Simulating-Anything && source .venv/bin/activate && <your_command>"
```

Never fall back to CPU for training or pipeline runs. Always use WSL2.

---

## 6. Key Commands

### Tests
```bash
# Full suite in WSL (284 passing, 14 skipped):
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

---

## 11. Future Roadmap

### V2 (Near-term)
- ~~Install Julia + PySR for symbolic regression~~ DONE
- ~~Demonstrate 3 rediscoveries~~ DONE (projectile R²=0.9999, LV R²=1.0, GS boundary+scaling)
- ~~Add SIR epidemic domain~~ DONE (R0 = β/γ, R²=1.0)
- ~~Add double pendulum domain~~ DONE (T = 2π√(L/g), R²=0.999993)
- ~~Add harmonic oscillator domain~~ DONE (ω₀=√(k/m), R²=1.0)
- ~~Train RSSM world models on all 3 V1 domains~~ DONE
- ~~Uncertainty-driven exploration demo~~ DONE (LV + SIR, R0 boundary detection)
- ~~Dream-based discovery pipeline~~ DONE (dreamed vs simulated comparison)
- ~~Cross-Domain Analogy Engine~~ DONE (11 isomorphisms across 8 domains)
- ~~Add Lorenz attractor domain~~ DONE (SINDy R²=0.99999, Lyapunov 1.1% error)
- ~~Add Navier-Stokes 2D domain~~ DONE (decay_rate=4*nu, R²=1.0)
- ~~Adversarial Dream Debate~~ DONE (simulation debate + divergence metrics)
- ~~Add Van der Pol oscillator~~ DONE (period R²=0.99996, amplitude~2.01)
- ~~Add Kuramoto oscillators~~ DONE (sync transition, K_c detection)
- ~~Add Brusselator~~ DONE (Hopf bifurcation b_c=1+a²)
- ~~Add FitzHugh-Nagumo~~ DONE (f-I curve, neural spiking)
- ~~Add Heat Equation 1D~~ DONE (exact spectral, D*k² decay)
- ~~Add Logistic Map~~ DONE (Feigenbaum, ln(2) Lyapunov at r=4)
- Ablation studies with PySR (data generated, awaiting evaluation)
- Add more JAX-native domains: molecular dynamics (JAX-MD), robotics (Brax)

### V3 (Medium-term)
- Bridge to non-JAX simulators: OpenFOAM (CFD), GROMACS (MD), SUMO (traffic)
- Graph neural network encoders for molecular structures
- 3D CNN encoders for volumetric data
- Persistent knowledge across sessions

### V4 (Long-term)
- Auto-generated simulation code from natural language equations
- Composable dynamics module library
- Real sim-to-real transfer validation

### Paper
- Target: AI4Science workshops (NeurIPS, ICML, ICLR)
- Core contribution: domain-agnostic discovery architecture + rediscovery evidence
- Baseline comparisons: PySR alone, SINDy alone, manual simulation

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
    reaction_diffusion.py  # Gray-Scott (JAX finite differences)
    rigid_body.py          # Projectile (symplectic Euler + drag)
    agent_based.py         # Lotka-Volterra (RK4 + diffrax batch)
    epidemiological.py     # SIR epidemic model (RK4)
    chaotic_ode.py         # Double pendulum (Lagrangian + RK4)
    harmonic_oscillator.py # Damped harmonic oscillator (RK4)
    lorenz.py              # Lorenz strange attractor (RK4 + Lyapunov)
    navier_stokes.py       # 2D incompressible NS (spectral vorticity-streamfunction)
    van_der_pol.py         # Van der Pol oscillator (limit cycle, RK4)
    kuramoto.py            # Kuramoto coupled oscillators (sync transition)
    brusselator.py         # Brusselator chemical oscillator (Hopf bifurcation)
    fitzhugh_nagumo.py     # FitzHugh-Nagumo neuron model (excitable)
    heat_equation.py       # 1D heat equation (spectral FFT)
    logistic_map.py        # Logistic map (discrete chaos, Feigenbaum)
  world_model/
    rssm.py                # RSSM (Equinox) — 1536 latent dims
    encoder.py             # CNNEncoder, MLPEncoder
    decoder.py             # CNNDecoder, MLPDecoder, symlog
    trainer.py             # WorldModelTrainer (Adam + cosine)
  exploration/
    base.py                # Explorer ABC
    uncertainty_driven.py  # MC-dropout uncertainty explorer
  analysis/
    symbolic_regression.py # PySR wrapper (variable_names in fit())
    equation_discovery.py  # PySINDy wrapper (v2.1.0 API)
    ablation.py            # Single-factor ablation studies
    cross_domain.py        # Cross-domain analogy engine (17 isomorphisms)
    dream_debate.py        # Adversarial dream debate (divergence metrics)
  rediscovery/
    __init__.py            # Exports all rediscovery runners
    projectile.py          # Range equation R=v²sin(2θ)/g recovery
    lotka_volterra.py      # Equilibrium + ODE recovery via PySR/SINDy
    gray_scott.py          # Phase diagram + wavelength scaling analysis
    sir_epidemic.py        # R0 = β/γ + SIR ODE recovery
    double_pendulum.py     # Period T = 2π√(L/g) + energy conservation
    harmonic_oscillator.py # ω₀ = √(k/m) + damping + ODE recovery
    lorenz.py              # Lorenz ODE recovery + chaos transition
    navier_stokes.py       # NS 2D viscous decay rate recovery
    van_der_pol.py         # VdP period/amplitude + SINDy ODE
    kuramoto.py            # Sync transition + order parameter r(K)
    brusselator.py         # Hopf bifurcation b_c = 1+a^2
    fitzhugh_nagumo.py     # f-I curve + SINDy ODE
    heat_equation.py       # Mode decay rate D*k^2
    logistic_map.py        # Feigenbaum + Lyapunov + chaos onset
    runner.py              # Unified runner for all 14 domains
  knowledge/
    trajectory_store.py    # Parquet + JSON sidecar storage
    discovery_log.py       # JSONL discovery persistence
  verification/
    dimensional.py         # Dimensional analysis checks
    conservation.py        # Mass, energy, positivity, boundedness
  types/
    problem_spec.py        # ProblemSpec, Variable, Objective
    simulation.py          # SimulationConfig, Domain, DomainClassification
    trajectory.py          # TrajectoryData, TrajectoryMetadata
    discovery.py           # Discovery, Evidence, DiscoveryReport
  utils/
    config.py              # load_config(), load_domain_config()

configs/
  default.yaml             # Global defaults
  domains/
    reaction_diffusion.yaml
    rigid_body.yaml
    agent_based.yaml

tests/unit/                # 284 tests across 22 files
  test_types.py            # 28 tests — Pydantic model validation
  test_config.py           # 14 tests — Config loading
  test_simulation.py       # 14 tests — 3 V1 simulation engines
  test_world_model.py      # 11 tests — RSSM shapes, gradients
  test_agents.py           # 11 tests — Backend, classifier, communicator
  test_pipeline.py         # 20 tests — Verification, stores, exploration
  test_rediscovery.py      # 15 tests — Data gen, PySR, PySINDy integration
  test_new_domains.py      # 18 tests — SIR epidemic + double pendulum
  test_exploration.py      # 13 tests — Explorer + ablation module
  test_harmonic_oscillator.py # 14 tests — Oscillator sim + rediscovery data
  test_cross_domain.py     # 12 tests — Analogy detection + similarity
  test_lorenz.py           # 20 tests — Lorenz sim, fixed points, Lyapunov
  test_dream_debate.py     # 9 tests — Adversarial dream debate
  test_navier_stokes.py    # 13 tests — NS 2D spectral solver
  test_van_der_pol.py      # 12 tests — VdP limit cycle, period
  test_kuramoto.py         # 13 tests — Sync transition, order parameter
  test_brusselator.py      # 11 tests — Hopf bifurcation
  test_fitzhugh_nagumo.py  # 10 tests — FHN neuron model
  test_heat_equation.py    # 12 tests — Heat equation 1D spectral
  test_logistic_map.py     # 13 tests — Logistic map, Lyapunov, periods

output/rediscovery/          # Rediscovery results (not committed to git)
  projectile/results.json    # R = v²sin(2θ)/g recovered
  lotka_volterra/results.json # Equilibrium + ODE equations recovered
  gray_scott/results.json    # Phase diagram + wavelength scaling
  sir_epidemic/results.json  # R0 = β/γ + SIR ODEs
  double_pendulum/results.json # Period T = 2π√(L/g) + energy
  harmonic_oscillator/results.json # ω₀ = √(k/m), c/(2m), SINDy ODE
  lorenz/results.json      # Lorenz ODEs, chaos transition, Lyapunov
  navier_stokes/results.json # NS 2D viscous decay rate

output/world_models/         # Trained RSSM checkpoints (not committed)
  projectile/model.eqx      # Recon MSE 0.38, dream MSE 0.61
  lotka_volterra/model.eqx   # Recon MSE 0.39, dream MSE 0.22
  gray_scott/model.eqx       # Recon MSE 0.06, dream MSE 0.10

scripts/
  generate_figures.py        # 14 publication-quality figures
  build_notebook.py          # Builds flagship rediscovery notebook
  train_world_models.py      # RSSM training on all domains (WSL)
  build_wm_notebook.py       # World model training notebook
  build_crossdomain_notebook.py # Cross-domain analysis notebook
  run_exploration_demo.py    # Uncertainty exploration demo
  run_dream_discovery.py     # Dream-based discovery pipeline
  run_ablation_studies.py    # Systematic ablation studies
  generate_paper_figures.py  # 8 publication-quality figures (all 7 domains)
  build_7domain_notebook.py  # Builds 7-domain rediscovery notebook

docs/
  RESEARCH.md              # Vision, universality argument (Section 4), contributions
  DESIGN.md                # Architecture, domain expansion (Section 11), evaluation

notebooks/
  demos/demo.ipynb           # Three-domain demo
  rediscovery_results.ipynb  # Flagship 5-domain notebook (43 cells, 14 figures)
  seven_domain_rediscovery.ipynb # 7-domain notebook (35 cells)
  world_model_training.ipynb # RSSM training results
  cross_domain_analysis.ipynb # Cross-domain comparison
```
