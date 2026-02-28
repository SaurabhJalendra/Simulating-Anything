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

Success means the system autonomously rediscovers known physics across 3
unrelated domains -- proving the universality claim with concrete evidence.

### Projectile (rigid body)
- **Target:** Recover R = v²sin(2θ)/g from simulation data via PySR
- **Also:** Pendulum period T = 2π√(L/g), optimal angle < 45° with drag
- **Current:** Simulation validated to 0.14% error against theory

### Lotka-Volterra (agent-based)
- **Target:** Recover equilibrium point (γ/δ, α/β) from population dynamics
- **Also:** SIR R0 = β/γ, herd immunity threshold 1 - 1/R0
- **Current:** Time-averages match equilibrium within 1.7%

### Gray-Scott (reaction-diffusion)
- **Target:** Turing instability threshold, wavelength scaling λ ~ sqrt(D/k)
- **Also:** Phase diagram with 4+ of 12 known Pearson regimes
- **Current:** Simulation stable, patterns forming with CFL-safe dt

---

## 3. The Universality Argument

Only the `SimulationEnvironment` subclass is domain-specific. Everything
else -- problem parsing, world model, exploration, analysis, reporting --
operates on generic tensors. Adding a domain = one new class (~50-200 lines).

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
# Full suite in WSL (98 passing, 0 skipped):
wsl.exe -d Ubuntu -e bash -c "cd /mnt/d/'Git Repos'/Simulating-Anything && source .venv/bin/activate && python3 -m pytest tests/unit/ -v"

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

---

## 11. Future Roadmap

### V2 (Near-term)
- Install Julia + PySR for symbolic regression
- Demonstrate 3 rediscoveries (projectile, Lotka-Volterra, Gray-Scott)
- Add more JAX-native domains: molecular dynamics (JAX-MD), robotics (Brax)
- Adversarial Dream Debate: two world models validating each other
- Cross-Domain Analogy Engine: detect mathematical isomorphisms

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
  world_model/
    rssm.py                # RSSM (Equinox) — 1536 latent dims
    encoder.py             # CNNEncoder, MLPEncoder
    decoder.py             # CNNDecoder, MLPDecoder, symlog
    trainer.py             # WorldModelTrainer (Adam + cosine)
  exploration/
    base.py                # Explorer ABC
    uncertainty_driven.py  # MC-dropout uncertainty explorer
  analysis/
    symbolic_regression.py # PySR wrapper
    equation_discovery.py  # PySINDy wrapper
    ablation.py            # Single-factor ablation studies
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

tests/unit/                # 98 tests across 6 files
  test_types.py            # 28 tests — Pydantic model validation
  test_config.py           # 14 tests — Config loading
  test_simulation.py       # 14 tests — All 3 simulation engines
  test_world_model.py      # 11 tests — RSSM shapes, gradients
  test_agents.py           # 11 tests — Backend, classifier, communicator
  test_pipeline.py         # 20 tests — Verification, stores, exploration

docs/
  RESEARCH.md              # Vision, universality argument (Section 4), contributions
  DESIGN.md                # Architecture, domain expansion (Section 11), evaluation

notebooks/demos/demo.ipynb # Three-domain demo
```
