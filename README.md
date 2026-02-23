# Simulating Anything

**World Models as a General-Purpose Scientific Discovery Engine**

Given any problem, build a custom simulation, train a world model on it,
dream through millions of scenarios, and surface optimal solutions or novel
patterns that humans might never find.

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

## V1 Scope

Three domains, eight agents, one pipeline:

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

1. **Adversarial Dream Debate** -- competing world models resolve discoveries
2. **Cross-Domain Analogy Engine** -- isomorphic structures across fields
3. **FunSearch-Style Program Discovery** -- evolving verifiable solutions
4. **Dream Journaling** -- mining trajectories for recurring patterns
5. **Socratic Discovery Mode** -- bidirectional human-AI exploration
6. **Composable Dynamics Modules** -- reusable physics building blocks
7. **Uncertainty-Driven Exploration** -- Bayesian exploration at knowledge boundaries
8. **Automated Ablation Studies** -- causal isolation of discovery factors
9. **Progressive Trust Architecture** -- reliability tracking across components

## Status

Research design phase complete. Implementation beginning.

## License

TBD
