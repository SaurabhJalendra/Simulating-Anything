# Simulating Anything -- System Design

**Version:** 0.1.0
**Last Updated:** 2026-02-23

This document consolidates the system architecture, V1 scope, technology
choices, implementation roadmap, and evaluation strategy into a single
reference. For the research motivation, novel contributions, and positioning,
see [RESEARCH.md](RESEARCH.md).

---

## Table of Contents

1. [V1 Scope](#1-v1-scope)
2. [Multi-Agent Architecture](#2-multi-agent-architecture)
3. [Three-Tier Simulation Hierarchy](#3-three-tier-simulation-hierarchy)
4. [World Model Architecture](#4-world-model-architecture)
5. [Data Flow and Pipeline](#5-data-flow-and-pipeline)
6. [Knowledge Infrastructure](#6-knowledge-infrastructure)
7. [Verification Architecture](#7-verification-architecture)
8. [Technology Stack](#8-technology-stack)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Evaluation Strategy](#10-evaluation-strategy)
11. [Domain Expansion Architecture](#11-domain-expansion-architecture)
12. [Known Problems and Solutions](#12-known-problems-and-solutions)

---

## 1. V1 Scope

All decisions in this section are final for V1. If something is not listed,
it ships in V2 or later. V1 constraints are scope decisions, not architectural
limitations -- the system is designed to support any domain (see
[RESEARCH.md Section 4](RESEARCH.md#4-the-universality-argument) and
[Section 11](#11-domain-expansion-architecture) below).

### 1.1 V1 Problem Domains

V1 supports exactly three problem classes:

| Domain | Backend | Example Problem | Known-Answer Validation |
|--------|---------|-----------------|------------------------|
| Reaction-Diffusion | PhiFlow (JAX) | Pattern formation in activator-inhibitor systems | Turing instability threshold, wavelength selection |
| Rigid-Body Mechanics | Brax / MJX | Projectile optimization, pendulum dynamics | Range formula, period formula, energy conservation |
| Agent-Based / Population | Custom JAX + diffrax | Predator-prey dynamics, epidemic spread | Lotka-Volterra equilibrium, SIR R0 threshold |

### 1.2 V1 Agent Roster (8 Agents)

| # | Agent | LLM? | Responsibility |
|---|-------|------|---------------|
| 1 | Problem Architect | Yes | Natural language -> formal ProblemSpec (YAML) |
| 2 | Domain Classifier | Yes | Route to domain + select simulation backend |
| 3 | Simulation Builder | Yes | Generate simulation code from templates |
| 4 | Simulation Validator | No | Dimensional, conservation, stability checks |
| 5 | World Model Trainer | No | Train RSSM on simulation trajectories |
| 6 | Explorer | No | Uncertainty-driven parameter space exploration |
| 7 | Analyst | No | PySR symbolic regression, PySINDy, ablation |
| 8 | Communication | Yes | Generate human-readable report with plots |

**Deferred to V2:** Coordinator, Skeptic, Auditor, Real-Data Anchoring,
Counterfactual, Cross-Domain Analogy agents.

### 1.3 V1 Success Criteria

| Criterion | Threshold |
|-----------|-----------|
| Pipeline completeness | 3/3 demo problems complete autonomously |
| Known-result rediscovery | >= 3 known results across 3 domains |
| Equation accuracy | Median relative error < 5%, no individual > 15% |
| World model fidelity | < 10% normalized MSE over 50-step rollouts |
| Time to discovery | < 4 hours per demo on single A100 GPU |
| Report quality | Correct equations, meaningful plots, no false claims |

### 1.4 What V1 Does NOT Do (But the Architecture Supports)

V1 deliberately constrains scope to three domains. This is a scope decision,
not an architectural limitation. The items below are deferred, not impossible:

- Arbitrary problem domains (only 3 in V1; architecture supports unlimited -- see Section 11)
- Real-time interaction (batch processing only)
- Sim-to-real transfer (assumption tracking provides the foundation)
- Cross-domain analogy (mathematical structure library in V2)
- Adversarial verification (no second world model)
- Multi-physics coupling (composable dynamics modules in V3)
- 3D fluid dynamics (requires 3D encoder/decoder, not architectural change)
- Ensemble world models (uses MC dropout instead)
- Interactive dashboard (static Markdown + PNG)
- Distributed execution (single GPU)
- Persistent knowledge across sessions
- Auto-generated simulation code from equations (V4 goal)

---

## 2. Multi-Agent Architecture

### Agent Topology

```
                    +-------------------------+
                    |   Human / Notebook      |
                    +------------+------------+
                                 |
                +----------------+----------------+
                |                                 |
      +---------v----------+           +----------v---------+
      | Problem Architect  |           |  Communication     |
      | Agent              |           |  Agent             |
      +---------+----------+           +----------+---------+
                |                                 ^
                v                                 |
      +-------------------+                       |
      | Domain Classifier |                       |
      +--------+----------+                       |
               |                                  |
               v                                  |
    +----------+----------+                       |
    | Simulation Builder  |                       |
    +----------+----------+                       |
               |                                  |
               v                                  |
    +----------+----------+                       |
    | Simulation Validator|                       |
    +----------+----------+                       |
               |                                  |
               v                                  |
    +----------+----------+                       |
    | World Model Trainer |                       |
    +----------+----------+                       |
               |                                  |
               v                                  |
    +----------+----------+                       |
    | Explorer Agent      |                       |
    | (uncertainty-driven)|                       |
    +----------+----------+                       |
               |                                  |
               v                                  |
    +----------+----------+                       |
    | Analyst Agent       |                       |
    | (PySR, SINDy)       +--------->-------------+
    +---------------------+
```

### Agent Communication

All agents communicate via typed messages (Pydantic models) through
in-process async queues. No external message bus in V1.

| From | To | Message Type | Format |
|------|----|-------------|--------|
| Problem Architect | Domain Classifier | `ProblemSpec` | YAML |
| Domain Classifier | Simulation Builder | `DomainClassification` | YAML |
| Simulation Builder | Simulation Validator | `SimulationCode` | Python + config |
| Simulation Validator | World Model Trainer | `ValidationReport` (PASS) | YAML |
| World Model Trainer | Explorer | `TrainedModelCheckpoint` | File path + report |
| Explorer | Analyst | `TrajectoryBundle` | Parquet + metadata |
| Analyst | Communication | `DiscoveryReport` | YAML + equations |
| Communication | User | `FinalReport` | Markdown + PNG |

### LLM Backend

The four LLM-powered agents use **Claude Code CLI** as their backend, not
the Anthropic Python SDK. This avoids API key management and leverages
Claude Code's tool-use capabilities.

```python
class ClaudeCodeBackend:
    """LLM backend using Claude Code CLI subprocess calls."""

    def ask(self, prompt: str, system: str | None = None) -> str:
        cmd = ["claude", "-p", prompt, "--output-format", "json"]
        if system:
            cmd.extend(["--system", system])
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        response = json.loads(result.stdout)
        return response["result"]
```

---

## 3. Three-Tier Simulation Hierarchy

V1 uses two tiers (Tier 3 full-fidelity solvers deferred):

| Tier | Engine | Speed | Use in V1 |
|------|--------|-------|-----------|
| 1 | RSSM world model (neural surrogate) | ~1ms/step | Dreaming, broad exploration |
| 2 | JAX differentiable sim (PhiFlow, Brax, custom) | ~100ms/step | Training data, ground-truth validation |
| 3 | Full-fidelity solver (OpenFOAM, FEniCS, LAMMPS) | ~10s/step | **Deferred to V2** |

### Tier Promotion Protocol

```
For each dream trajectory:
  1. Run on Tier 1 (neural surrogate)
  2. If model confidence >= 0.95: accept at Tier 1
  3. Else: promote to Tier 2 (differentiable sim)
  4. If Tier 2 diverges from Tier 1 by > tolerance:
       Flag region, add to retraining data
  5. Accept at Tier 2
```

---

## 4. World Model Architecture

### Hybrid Physics + Neural Architecture

```
Input: state(t), params
    |
    +---------------------------+
    |                           |
    v                           v
+-----------+            +--------------+
| Physics   |            | Neural       |
| Engine    |            | Residual     |
| Backbone  |            | (RSSM)       |
+-----------+            +--------------+
    |                           |
    | state_phys(t+1)           | residual(t+1)
    |                           |
    +-------------+-------------+
                  |
                  v
         +------------------+
         | Hard Constraint  |
         | Projection Layer |
         +------------------+
                  |
                  v
         state(t+1) [conservation laws guaranteed]
```

### RSSM Architecture (DreamerV3-style)

- **Deterministic path:** GRU cell, hidden size 512
- **Stochastic path:** 32 categorical variables x 32 classes = 1024 dims
- **Full latent state:** 512 + 1024 = 1536 dimensions
- **Encoder:** CNN (4 layers, 32->64->128->256 channels) for spatial data;
  MLP for vector data
- **Decoder:** Transposed CNN / MLP mirroring encoder, symlog output transform
- **Uncertainty:** Monte Carlo dropout at inference time

### Hard Constraint Projection

| Constraint | Projection Method |
|------------|-------------------|
| Mass conservation | Rescale density field |
| Energy conservation | Lagrange multiplier projection |
| Momentum conservation | Subtract mean drift |
| Divergence-free (incompressible) | Helmholtz decomposition |
| Positivity (density, temperature) | Softplus clamp |

---

## 5. Data Flow and Pipeline

### End-to-End V1 Pipeline

```
User: "Describe your problem in natural language"
    |
    v
Problem Architect Agent
    -> Produces ProblemSpec (YAML)
    |
    v
Domain Classifier Agent
    -> Routes to: reaction-diffusion / rigid-body / agent-based
    -> Selects simulation backend
    |
    v
Simulation Builder Agent
    -> Loads domain template
    -> LLM customizes parameters, BCs, ICs
    -> Generates SimulationConfig
    |
    v
Simulation Validator Agent
    -> Dimensional consistency
    -> Conservation law tests
    -> Known-limit tests
    -> Short stability run (100 steps)
    -> Order-of-magnitude check
    -> GATE: all critical checks must pass
    |
    v
World Model Trainer Agent
    -> Generate training trajectories (Sobol sampling)
    -> Train RSSM (JAX + Equinox, 300-500 epochs)
    -> Validate: prediction error < threshold
    |
    v
Explorer Agent (uncertainty-driven)
    -> MC dropout for epistemic uncertainty
    -> Target high-uncertainty parameter regions
    -> Periodic ground-truth validation
    -> Log all trajectories to Dream Journal
    |
    v
Analyst Agent
    -> PySR symbolic regression
    -> PySINDy equation discovery
    -> Single-factor ablation studies
    -> HDBSCAN phase boundary detection
    |
    v
Communication Agent
    -> Structured report: findings, equations, plots
    -> Confidence assessments
    -> Plain-language summary
    |
    v
User reviews findings (basic Socratic redirect)
```

### Pipeline Timing Budget (single GPU)

| Stage | Target Time |
|-------|-------------|
| Problem Architect + Domain Classifier | < 2 min |
| Simulation Builder + Validator | < 10 min |
| World Model Training | < 60 min |
| Exploration | < 120 min |
| Analysis (PySR + PySINDy + ablation) | < 30 min |
| Report generation | < 5 min |
| **Total** | **< 4 hours** |

### MVP Demo: Turing Pattern Discovery

User input: "I have a two-chemical system where chemical A activates itself
and chemical B, while chemical B inhibits chemical A. Both chemicals diffuse,
but B diffuses faster. What patterns emerge?"

Expected output: Report discovering the Turing instability threshold,
wavelength selection formula, and three pattern regimes (spots, stripes,
labyrinths) with >0.92 confidence.

---

## 6. Knowledge Infrastructure

V1 implements lightweight versions of three knowledge stores:

### Dream Journal (Trajectory Store)

```yaml
trajectory:
  id: string (UUID)
  problem_id: string
  model_id: string
  explorer_id: string
  tier: int (1 or 2)
  states: array[state]
  actions: array[action]
  metadata:
    confidence: float
    novelty_score: float
    validated: bool
  provenance:
    code_version: string (git hash)
    random_seed: int
    hardware: string
```

Storage: Chunked Parquet files with JSON metadata sidecar. Immutable once
written.

### Discovery Log

```yaml
discovery:
  id: string
  type: governing_equation | phase_boundary | scaling_law
  confidence: float
  expression: string
  evidence:
    trajectories: list[string]
    fit_r_squared: float
  assumptions: list[string]
```

### Assumption Tracker

Every simplification is logged with justification, evidence status
(SUPPORTED / UNSUPPORTED / CONTRADICTED), and impact-if-wrong assessment.

---

## 7. Verification Architecture

Six layers, ordered cheapest to most expensive:

| Layer | What | When | Cost |
|-------|------|------|------|
| 1 | Dimensional analysis | Simulation construction | Negligible |
| 2 | Conservation law enforcement | Every timestep | Negligible |
| 3 | Symmetry detection/enforcement | Every timestep | Low |
| 4 | Order-of-magnitude validation | Per simulation stage | Low |
| 5 | Cross-model validation | Per discovery | Medium |
| 6 | Ground-truth spot-checking | Periodic | High |

### Progressive Fidelity Pipeline

Before committing expensive compute, validate through progressively
more expensive models:

1. **Dimensional analysis** -- form dimensionless groups, estimate scales
2. **Simplest analytical model** -- ODE, closed-form, lumped parameter
3. **Reduced-complexity numerical** -- 1D/2D, coarse mesh, first-order
4. **Full-complexity simulation** -- 3D transient, grid convergence study

Each stage has an explicit validation gate that must pass.

---

## 8. Technology Stack

### Core (Locked for V1)

| Component | Choice | Reason |
|-----------|--------|--------|
| Language | Python 3.11 | Maximum compatibility |
| Compute framework | JAX | End-to-end differentiability, JIT, vmap, pmap |
| Neural networks | Equinox | Clean JAX-native API |
| Optimizer | Optax | JAX-native, Adam + gradient clipping |
| ODE integration | diffrax | JAX-native ODE/SDE solver |

### Simulation

| Domain | Backend |
|--------|---------|
| Reaction-diffusion | PhiFlow 3.x (JAX backend) or direct JAX finite differences |
| Rigid-body | Brax / MJX |
| Agent-based (ODE) | Custom JAX + diffrax |

### Analysis

| Tool | Purpose |
|------|---------|
| PySR | Symbolic regression (Julia backend) |
| PySINDy | Sparse equation discovery |
| scikit-learn (HDBSCAN) | Clustering for phase boundaries |
| Matplotlib | Static visualization |

### Infrastructure

| Component | Choice |
|-----------|--------|
| LLM backbone | Claude Code CLI (subprocess) |
| Data types | Pydantic v2 |
| Config | YAML + Pydantic validation |
| Trajectory storage | Parquet (PyArrow) |
| Experiment tracking | Weights & Biases (optional, local fallback) |
| Data versioning | DVC (local backend for V1) |
| Agent communication | Python asyncio (in-process) |
| Report format | Markdown + PNG |

### V1 Dependencies

```
# Core
jax[cuda12]
equinox
optax
diffrax

# Simulation
phiflow
brax
mujoco

# Analysis
pysr          # requires Julia 1.10+
pysindy
scikit-learn

# Infrastructure
pydantic>=2.0
pyyaml
pyarrow
matplotlib
numpy
```

### Hardware Requirements

- **Minimum (dev):** NVIDIA GPU 16GB VRAM, 32GB RAM, 100GB storage
- **Target (demo):** NVIDIA A100 40GB, 64GB RAM, 500GB SSD

---

## 9. Implementation Roadmap

9 phases over ~30 weeks. Each phase produces a runnable deliverable.

| Phase | Weeks | Deliverable | Key Risk |
|-------|-------|-------------|----------|
| 0: Foundation | 1-2 | Project skeleton, types, config, CI | JAX/CUDA compatibility |
| 1: Reaction-Diffusion | 3-5 | Working PDE simulation + validation | PhiFlow integration |
| 2: World Model | 6-9 | Trained RSSM, accurate 50-step dreams | RSSM may not suit PDEs |
| 3: Exploration | 10-12 | Automated parameter-space mapping | Uncertainty calibration |
| 4: Analysis | 13-15 | Rediscovered equations from data | PySR speed/complexity |
| 5: LLM Agents | 16-19 | NL input -> report pipeline | LLM output brittleness |
| 6: More Domains | 20-24 | Rigid-body + agent-based end-to-end | Scope creep |
| 7: Validation | 25-27 | Benchmarked, documented | Architecture-level failures |
| 8: Release | 28-30 | Docker image, notebooks, paper draft | Perfectionism delays |

### Directory Structure

```
simulating_anything/
+-- pyproject.toml
+-- configs/
|   +-- default.yaml
|   +-- domains/
|       +-- reaction_diffusion.yaml
|       +-- rigid_body.yaml
|       +-- agent_based.yaml
+-- src/simulating_anything/
|   +-- __init__.py
|   +-- types/           # Pydantic models
|   +-- agents/          # LLM + computation agents
|   +-- simulation/      # Domain simulation backends
|   +-- world_model/     # RSSM implementation
|   +-- exploration/     # Uncertainty-driven explorer
|   +-- analysis/        # PySR, SINDy, ablation
|   +-- verification/    # Dimensional, conservation checks
|   +-- knowledge/       # Trajectory store, discovery log
|   +-- utils/           # JAX utilities, tracking
|   +-- pipeline.py      # End-to-end orchestrator
+-- templates/           # Simulation templates per domain
+-- tests/
|   +-- unit/
|   +-- integration/
|   +-- benchmarks/
+-- notebooks/
    +-- demo.ipynb
```

### Dependency Graph

```
Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5 -> Phase 6 -> Phase 7 -> Phase 8
```

Phases 5 (LLM Agents) can start in parallel with Phases 3-4. Domain
backends (Phase 6 tasks 6.1, 6.2) can start in parallel with Phase 4-5.

---

## 10. Evaluation Strategy

### Benchmark Suite

#### Reaction-Diffusion

| Benchmark | Known Answer | Target |
|-----------|-------------|--------|
| Turing instability threshold | Analytical formula | < 10% relative error |
| Pattern wavelength | lambda ~ sqrt(D/k) | Scaling relationship |
| Gray-Scott phase diagram | 12 known regimes (Pearson 1993) | Identify >= 4 |
| Fisher-KPP wave speed | c = 2*sqrt(D*r) | Exact formula recovery |

#### Rigid-Body

| Benchmark | Known Answer | Target |
|-----------|-------------|--------|
| Projectile range | R = v^2*sin(2*theta)/g | Exact formula |
| Pendulum period | T = 2*pi*sqrt(L/g) | Exact formula |
| Energy conservation | KE + PE = const | Drift < 0.1% |
| Optimal angle with drag | < 45 degrees | Correct qualitative finding |

#### Agent-Based

| Benchmark | Known Answer | Target |
|-----------|-------------|--------|
| Lotka-Volterra equilibrium | (d/c, a/b) | < 5% error |
| SIR basic reproduction number | R0 = beta/gamma | Formula recovery |
| SIR herd immunity threshold | 1 - 1/R0 | Formula recovery |
| Logistic growth carrying capacity | K = r/alpha | Formula recovery |

### Component Metrics

| Component | Key Metric | V1 Threshold |
|-----------|-----------|-------------|
| Simulation | Conservation error | < 1e-3 (Tier 2) |
| Simulation | Analytical agreement | < 5% (Tier 2) |
| World Model | 50-step prediction MSE | < 60% state variance |
| World Model | Uncertainty calibration | Spearman > 0.7 |
| Exploration | Parameter space coverage | > 80% |
| Analysis | Equation accuracy | < 5% mean relative error |
| Analysis | False positive rate | < 10% |
| Report | Human evaluation (1-5) | Mean >= 3.5 |

### Progressive Difficulty Levels

| Level | Characteristics | V1 Target |
|-------|----------------|-----------|
| 1 | Single equation, 1D, steady-state | > 80% success |
| 2 | Coupled equations, 2D, time-dependent | > 60% success |
| 3 | Multi-parameter sweeps, phase transitions | > 30% success |
| 4 | Beyond training distribution | Track only |
| 5 | Open-ended (no known answer) | Track only |

---

## 11. Domain Expansion Architecture

The system is designed from the ground up to support any domain. V1 proves
the pipeline on three domains; V2+ expands to unlimited domains without
architectural changes. This section documents how.

### 11.1 The Only Domain-Specific Component

The entire pipeline is domain-agnostic except for one component: the
`SimulationEnvironment` subclass. Everything else -- problem parsing, world
model training, exploration, analysis, reporting -- operates on generic tensors
and parameter dictionaries.

```
Domain-agnostic (no changes needed per domain):
  - Problem Architect (LLM: parses any natural language)
  - Domain Classifier (routing only, not constraining)
  - World Model (RSSM: trains on any state sequence)
  - Explorer (operates on parameter ranges and uncertainty scores)
  - Analyst (PySR/SINDy: fits equations to any numerical data)
  - Communication Agent (reports findings from any domain)

Domain-specific (one class per domain):
  - SimulationEnvironment subclass
    - reset() -> initial_state
    - step() -> next_state
    - observe() -> observation
```

### 11.2 Adding a New Domain

Adding support for a new problem class requires exactly these steps:

1. **Write a `SimulationEnvironment` subclass** implementing `reset()`,
   `step()`, and `observe()`. For domains with existing JAX-compatible
   simulators, this is a thin wrapper (~50-200 lines).

2. **Add a domain config YAML** specifying default parameters, sweep ranges,
   and known-answer benchmarks.

3. **Register the domain** in the `Domain` enum and `DomainClassifier`
   keyword table.

4. **(Optional)** Add a specialized encoder/decoder if the state representation
   requires one (e.g., graph neural network for molecular structures).

No changes to the pipeline, world model, explorer, analyst, or communicator.

### 11.3 Encoder/Decoder Architecture Selection

The state representation determines which encoder/decoder to use:

| State Representation | Encoder | Decoder | Example Domains |
|---------------------|---------|---------|-----------------|
| Scalar fields on regular grid | CNN (Conv2d layers) | Transposed CNN | Reaction-diffusion, fluid dynamics, weather, acoustics |
| Vector state (low-dimensional) | MLP | MLP | Rigid body, ODEs, population dynamics, circuit models |
| 3D volumetric fields | 3D CNN | 3D Transposed CNN | Structural mechanics, 3D fluid dynamics, MRI |
| Point clouds / particles | PointNet / Set encoder | Set decoder | Molecular dynamics, SPH fluids, N-body |
| Graphs / networks | GNN (message-passing) | GNN decoder | Molecules, social networks, supply chains |
| Sequences / time series | Transformer / 1D CNN | Autoregressive decoder | Financial markets, speech, signal processing |
| Multi-resolution / AMR | U-Net / hierarchical | Hierarchical decoder | Climate, astrophysics, adaptive mesh simulations |

V1 implements CNN and MLP encoders/decoders. The RSSM core is unchanged
regardless of encoder choice -- it always operates on a fixed-size latent
vector.

### 11.4 Domain Expansion Roadmap

#### Near-Term (V2): JAX-Native Domains

These domains have existing JAX-compatible simulators and can be added with
minimal effort:

| Domain | Simulator | State Shape | Effort |
|--------|-----------|-------------|--------|
| Molecular dynamics | JAX-MD | (N_atoms, 3) positions + velocities | Low -- direct wrapper |
| Robotics (articulated) | Brax / MJX | (N_joints,) angles + velocities | Low -- already in stack |
| ODEs (arbitrary) | diffrax | (N_vars,) | Low -- generic ODE wrapper |
| Fluid dynamics (2D) | JAX-CFD | (Nx, Ny, 3) velocity + pressure | Medium -- grid setup |
| Quantum circuits | PennyLane (JAX) | (2^N,) state vector | Medium -- qubit encoding |

#### Medium-Term (V3): Bridge Domains

These require bridging to non-JAX simulators via file I/O or subprocess:

| Domain | Simulator | Bridge Method |
|--------|-----------|--------------|
| Structural FEM | FEniCS / FreeFEM | Python API |
| Full 3D CFD | OpenFOAM | File-based I/O |
| Molecular (large-scale) | GROMACS / LAMMPS | Trajectory file parsing |
| Climate | CESM / E3SM | Netcdf output processing |
| Traffic networks | SUMO | TraCI Python API |
| Power grids | PyPSA | Direct Python integration |

#### Long-Term (V4+): Auto-Generated Domains

The ultimate goal: given a natural language description and no pre-existing
simulator, the system generates the simulation code itself using the LLM
agents. The Simulation Builder Agent already does this in simplified form
for V1 templates. V4+ generalizes this to arbitrary domain equations.

### 11.5 Cross-Domain Transfer

As the system studies more domains, it accumulates reusable knowledge:

- **Composable dynamics modules**: Gravity, diffusion, advection, reaction,
  friction, viscosity, electrostatics -- once implemented for one domain,
  reusable in others.
- **Mathematical structure library**: Bifurcations, conservation laws,
  symmetry groups, scaling relationships -- recognized across domains.
- **Cross-domain analogies**: The mathematical structure of epidemic spread
  (SIR) is identical to chemical reaction kinetics. Predator-prey (Lotka-
  Volterra) has the same form as competing economic firms. The system can
  transfer discoveries between isomorphic domains automatically.
- **Encoder/decoder reuse**: A CNN encoder trained on reaction-diffusion
  patterns transfers to any 2D scalar field problem. An MLP encoder for
  Lotka-Volterra transfers to any low-dimensional ODE system.

The 100th domain the system studies is dramatically easier than the 10th,
because the component library, analogy database, and training heuristics
all compound.

---

## 12. Known Problems and Solutions

| # | Problem | Severity | V1 Solution |
|---|---------|----------|-------------|
| 1 | NL -> formal simulation | Critical | Template library + progressive building for 3 domains |
| 2 | World model drift | High | Hybrid physics+neural, hard constraint enforcement |
| 3 | Exploration scalability | High | Uncertainty-driven search, two-tier hierarchy |
| 4 | Computational cost | High | Tier 1 dreaming + Tier 2 validation |
| 5 | Insight extraction | Medium-High | PySR + PySINDy + LLM narration |
| 6 | Discovery validation | Medium-High | Ground-truth spot-checks, dimensional analysis |
| 7 | Sim-to-real gap | High | Assumption tracking, confidence labeling |
| 8 | Problem ambiguity | Medium | Interactive Problem Architect, structured templates |
| 9 | Physical consistency | Medium | Hard constraint enforcement (energy, mass) |
| 10 | Correlation vs. causation | Medium | Ablation studies, interventional dreaming (V2) |
| 11 | Generalization vs. specialization | Medium | Modular architecture, curated V1 domains |
| 12 | Hallucinated consensus | Medium | Tool-grounded verification, physics backstop |

### Cross-Cutting Solutions

- **Hard constraint enforcement** addresses problems 2, 9, 12
- **Three-tier hierarchy** addresses problems 3, 4, 6
- **Template-based construction** addresses problems 1, 8, 11

---

*This document is the primary technical reference for the Simulating Anything
project. For the universality argument and research motivation, see
[RESEARCH.md](RESEARCH.md). For domain expansion details, see
[Section 11](#11-domain-expansion-architecture).*
