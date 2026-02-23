# Simulating Anything

**World Models as a General-Purpose Scientific Discovery Engine**

Given **any** problem -- from reaction-diffusion chemistry to self-driving cars
to protein folding to climate modeling -- build a custom simulation, train a
world model on it, dream through millions of scenarios, and surface optimal
solutions, governing equations, phase boundaries, and scaling laws that humans
might never find.

**The core insight:** any real-world phenomenon is a dynamical system; any
dynamical system can be simulated; any simulation can train a world model; and
discoveries from world models transfer back to the real world. The architecture
is domain-agnostic. Only the simulation backend changes per domain.

---

## Core Pipeline

```
Problem Definition (natural language)
       |
       v
Simulation Environment Construction (auto-generated JAX code)
       |
       v
World Model Training (physics backbone + neural residual)
       |
       v
Dreaming / Exploration (uncertainty-driven)
       |
       v
Pattern Discovery & Insight Extraction (PySR, SINDy)
       |
       v
Validation (ground-truth checks, conservation laws)
       |
       v
Human-Readable Scientific Report
```

## Key Principles

- **Multi-agent architecture**: 8 specialized agents (V1), each owning one step
- **Verification-first**: Every discovery validated before reporting
- **Progressive fidelity**: Start simple, escalate only as needed
- **Physics-grounded**: Conservation laws enforced as hard constraints
- **LLM as conductor**: Agents orchestrate specialized tools -- never do math

## Why "Any" Problem?

The system is not limited to specific domains. The architecture is genuinely
universal:

| Component | Domain-Specific? | What It Sees |
|-----------|-----------------|-------------|
| Problem Architect (LLM) | No | Natural language -- any domain |
| Domain Classifier | No | Routing only -- keywords |
| World Model (RSSM) | No | Tensors evolving over time -- any state sequence |
| Explorer | No | Parameter ranges and uncertainty scores |
| Analyst (PySR/SINDy) | No | Numerical arrays -- fits equations to any data |
| Communicator | No | Equations, plots, text |
| **Simulation Environment** | **Yes** | **This is the only component that changes** |

Adding a new domain = writing one `SimulationEnvironment` subclass (~50-200
lines). The rest of the pipeline works unchanged. See
[RESEARCH.md Section 4](docs/RESEARCH.md#4-the-universality-argument) for the
full universality argument.

### Domains the System Can Address

| Category | Example Domains |
|----------|----------------|
| **Physical sciences** | Fluid dynamics, structural mechanics, thermodynamics, electromagnetics, plasma physics, quantum systems, acoustics |
| **Robotics & autonomous systems** | Robotic manipulation, legged locomotion, self-driving cars, drones, swarm robotics |
| **Life sciences** | Protein dynamics, drug interactions, gene networks, epidemiology, ecology, neuroscience |
| **Earth & climate** | Weather prediction, climate modeling, ocean dynamics, seismology, hydrology |
| **Social sciences & economics** | Financial markets, urban traffic, supply chains, social dynamics, energy systems |
| **Materials science** | Molecular dynamics, crystal growth, battery chemistry, polymer dynamics, catalysis |

For each domain, the system discovers governing equations, phase boundaries,
scaling laws, and optimal strategies -- not by being told the physics, but by
learning from simulation data.

## V1 Scope

V1 proves the pipeline on three domains. This is a scope decision, not an
architectural limitation:

| Domain | Backend | Validation |
|--------|---------|-----------|
| Reaction-Diffusion | PhiFlow / JAX | Turing instability, wavelength selection |
| Rigid-Body Mechanics | Brax / MJX | Projectile range, pendulum period |
| Agent-Based / Population | Custom JAX + diffrax | Lotka-Volterra, SIR R0 |

## Architecture

```
Problem Architect --> Domain Classifier --> Simulation Builder
                                                    |
                                                    v
                                           Simulation Validator
                                                    |
                                                    v
                                           World Model Trainer (RSSM)
                                                    |
                                                    v
                                           Explorer (uncertainty-driven)
                                                    |
                                                    v
                                           Analyst (PySR + SINDy)
                                                    |
                                                    v
                                           Communication Agent --> Report
```

## Three-Tier Simulation Hierarchy

| Tier | Engine | Speed | Use |
|------|--------|-------|-----|
| 1 | Neural Surrogate (RSSM) | ~1ms/step | Exploration |
| 2 | Differentiable Sim (JAX) | ~100ms/step | Training + Validation |
| 3 | Full-Fidelity Solver | ~10s/step | V2 |

## Documentation

| Document | Description |
|----------|-------------|
| [Research](docs/RESEARCH.md) | Vision, motivation, research landscape, novel contributions, positioning |
| [Design](docs/DESIGN.md) | Architecture, V1 scope, tech stack, roadmap, evaluation |

## Technology Stack

```
Core:          Python 3.11 | JAX | Equinox | Optax | diffrax
Simulation:    PhiFlow | Brax/MJX | Custom JAX
World Model:   DreamerV3-style RSSM (JAX + Equinox)
Discovery:     PySR | PySINDy | HDBSCAN
LLM:           Claude Code CLI (subprocess)
Data:          Pydantic | PyArrow (Parquet) | YAML | Matplotlib
```

## Novel Contributions

1. **Domain-Agnostic Discovery Architecture** -- one pipeline for any simulatable phenomenon
2. **Adversarial Dream Debate** -- competing world models resolve discoveries
3. **Cross-Domain Analogy Engine** -- isomorphic structures across fields
4. **FunSearch-Style Program Discovery** -- evolving verifiable solutions
5. **Dream Journaling** -- mining trajectories for recurring patterns
6. **Socratic Discovery Mode** -- bidirectional human-AI exploration
7. **Composable Dynamics Modules** -- reusable physics building blocks
8. **Uncertainty-Driven Exploration** -- Bayesian exploration at knowledge boundaries
9. **Automated Ablation Studies** -- causal isolation of discovery factors
10. **Progressive Trust Architecture** -- reliability tracking across components

## Concrete Example: Self-Driving Cars

To illustrate the universality, consider autonomous vehicles:

1. **Simulation**: CARLA/SUMO models vehicle dynamics, traffic, weather, sensors
2. **World Model**: RSSM trains on (vehicle_state, traffic_state, action, next_state) trajectories
3. **System Discovers**:
   - Braking distance equations as f(speed, surface, tire_condition, mass)
   - Traffic flow phase transitions (free flow -> congestion thresholds)
   - Collision probability boundaries in (speed, distance, reaction_time) space
   - Fuel-optimal speed profiles for given route/traffic patterns
   - Sensor degradation scaling laws vs rain/fog/sun angle

These are scientific discoveries about driving physics -- not a driving policy,
but the understanding that informs better policies, testing, and safety analysis.

## Status

V1 implementation complete. All simulation engines, world model, exploration,
analysis, and agent pipeline implemented and tested (87 tests passing).

## License

TBD
