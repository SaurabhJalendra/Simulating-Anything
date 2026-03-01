# Simulating Anything

**World Models as a General-Purpose Scientific Discovery Engine**

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction and Motivation](#2-introduction-and-motivation)
3. [Core Idea and Pipeline](#3-core-idea-and-pipeline)
4. [The Universality Argument](#4-the-universality-argument)
5. [What Makes This Different](#5-what-makes-this-different)
6. [Research Landscape](#6-research-landscape)
7. [Critical Gaps in Current Research](#7-critical-gaps-in-current-research)
8. [Novel Contributions](#8-novel-contributions)
9. [Positioning and Comparison](#9-positioning-and-comparison)
10. [Target Capabilities](#10-target-capabilities)
11. [Long-Term Vision](#11-long-term-vision)
12. [References](#12-references)

---

## 1. Abstract

Scientific progress is constrained less by computational power than by the number of hypotheses a human scientist can formulate, test, and interpret. This project proposes a general-purpose scientific discovery engine that, given a natural language description of any problem or phenomenon, automatically constructs a simulation environment, trains a neural world model on its dynamics, explores the resulting state space through structured imagination, and extracts human-interpretable insights -- governing equations, causal graphs, phase diagrams, and optimal strategies.

The system is not a simulation framework. Simulation frameworks help run models that are already understood. This system discovers things that are not yet known, using simulations it builds for itself. It integrates automated simulation construction, hybrid physics-neural world models, uncertainty-driven exploration, symbolic insight extraction, and multi-agent verification into a single pipeline. Nine novel architectural contributions -- including adversarial dream debate, cross-domain analogy detection, composable dynamics modules, and progressive trust tracking -- address critical gaps in the current research landscape where no existing system provides satisfactory solutions.

The result is a computational research partner: a system that amplifies human scientists by automating the computationally intensive and combinatorially explosive aspects of the scientific method while keeping the human in the loop for judgment, direction, and domain expertise.

---

## 2. Introduction and Motivation

### The Bottleneck Is Human Bandwidth

Scientific progress is constrained less by computational power than by the number of hypotheses a human scientist can formulate, test, and interpret in a career. A domain expert might spend years developing intuition for a system, building a simulation, running parameter sweeps, and extracting meaning from the results. The intellectual labor at each stage is enormous, and it does not parallelize.

World models offer a path around this bottleneck. A trained world model can explore a simulation's state space orders of magnitude faster than the simulation itself runs, because it has learned a compressed, differentiable approximation of the dynamics. Where a traditional simulation might evaluate thousands of scenarios over days, a world model can dream through millions of counterfactual trajectories in hours. The exploration is not random; it can be steered by curiosity, uncertainty, and explicit research questions.

### Simulation Expertise Should Not Be a Prerequisite for Scientific Inquiry

Building a high-quality simulation today requires deep domain knowledge, numerical methods expertise, and significant engineering effort. This creates an access barrier: many researchers with important questions lack the simulation infrastructure to explore them computationally. A system that can construct simulations from natural language specifications, validate them, and progressively refine them lowers that barrier dramatically.

### Counterfactual Exploration Is Uniquely Powerful

Physical experiments are bound by reality. You cannot rewind a supernova, replay an epidemic with different parameters, or run a planetary climate for ten thousand years to see what happens. Simulations can. World models trained on those simulations inherit this freedom and extend it further: they can interpolate between scenarios, extrapolate beyond the training distribution (with appropriate uncertainty quantification), and explore conditions that are physically impossible but mathematically informative.

### Mathematical Structure Transfers Across Domains

The same differential equations appear in population dynamics, chemical kinetics, electrical circuits, and epidemiology. Diffusion governs heat transfer, particle motion, opinion spread, and financial markets. A system that learns to recognize these structural similarities can transfer insights across domains -- applying a technique discovered in fluid dynamics to a problem in urban traffic flow, or recognizing that a biological regulatory network has the same feedback topology as a well-studied control system.

This cross-domain transfer is not metaphorical. It is grounded in the shared mathematical structure of the underlying dynamics. A world model that learns to represent dynamics in terms of composable mathematical primitives can recognize and exploit these connections automatically.

---

## 3. Core Idea and Pipeline

The premise is direct: given any problem or situation, build a custom simulation environment and train a world model on it to autonomously discover solutions or scientific insights through imagined exploration.

The pipeline has five stages:

```
Problem --> Simulation --> World Model --> Dreaming/Exploration --> Insight Extraction
```

1. **Problem specification.** A researcher describes a phenomenon, a question, or an optimization target in natural language. No simulation code, no mathematical formulation -- just a clear statement of what they want to understand.

2. **Simulation construction.** The system translates that specification into a working simulation environment. It selects the right physics, the right abstractions, the right level of fidelity. It writes the code, validates it against known baselines, and exposes a clean interface for the world model to interact with.

3. **World model training.** A neural world model learns the dynamics of the simulation by observing trajectories through it. It builds an internal compressed representation of how the system behaves -- what causes what, what is stable, what is sensitive.

4. **Dreaming and exploration.** The world model generates millions of imagined trajectories, guided by curiosity signals, uncertainty estimates, and goal-directed search. It explores counterfactual scenarios, stress-tests edge cases, and systematically maps the space of possibilities far faster than running the original simulation would allow.

5. **Insight extraction.** The system distills its exploration into human-interpretable findings: governing equations, causal graphs, phase diagrams, optimal strategies, anomalous regimes, and plain-language explanations of what it discovered and why it matters.

### The Scientific Method, Automated

The system implements a computational analog of the scientific method, with each stage owned by a specialized agent:

```
Observe --> Hypothesize --> Experiment --> Analyze --> Validate --> Communicate
```

- **Observe.** The observation agent examines the problem domain, gathers relevant background knowledge, identifies key variables, and characterizes the scope and constraints.
- **Hypothesize.** The hypothesis agent generates testable predictions, ranked by expected information value.
- **Experiment.** The experiment agent designs and executes targeted investigations in both the full simulation (for ground truth) and the world model (for rapid exploration).
- **Analyze.** The analysis agent fits models, discovers equations, builds causal graphs, identifies phase boundaries, and detects anomalies.
- **Validate.** The validation agent checks extracted equations against held-out data, tests causal claims by intervention, and compares findings against known results.
- **Communicate.** The communication agent translates validated findings into figures, reports, and plain-language summaries at multiple levels of detail.

These stages form a loop, not a linear pipeline. Each cycle refines the system's understanding. A meta-cognitive layer monitors overall progress and decides when to continue iterating, when to refine the simulation, and when the question has been adequately answered.

---

## 3.1 Experimental Validation: Three-Domain Rediscovery

The following results demonstrate that the system can autonomously rediscover
known scientific laws across three unrelated domains using the same pipeline
infrastructure. Only the simulation backend changes.

### Projectile Range Equation (Rigid Body)

**Target:** R = v²sin(2θ)/g

**Method:** Generated 225 trajectories (15 speeds × 15 angles, no drag),
computed landing range, ran PySR symbolic regression (50 iterations).

**Result:** PySR discovered `v0² × 0.1019 × sin(2 × theta)` with R² = 0.9999999.
The coefficient 0.1019 matches 1/g = 1/9.81 = 0.10194 to 4 significant figures.
Simulation vs analytical solution mean error: 0.04%.

### Lotka-Volterra Equilibrium and ODE (Population Dynamics)

**Target:** Equilibrium prey* = γ/δ, pred* = α/β; ODE coefficients.

**Method:** Generated 200 trajectories with randomized (α, β, γ, δ), computed
time-averaged populations (skipping initial transient), ran PySR. Also ran
SINDy on a single reference trajectory (α=1.1, β=0.4, γ=0.4, δ=0.1).

**PySR Results:**
- Prey equilibrium: `g_/d_` (γ/δ) with R² = 0.9999 (from 200 parameter sweeps)
- Predator equilibrium: `a_/b_` (α/β) with R² = 0.9999
- Time-average error vs theory: 0.31% prey, 0.19% predator

**SINDy Results (ODE recovery):**
- `d(prey)/dt = 1.100 prey − 0.400 prey·pred` (true α=1.1, β=0.4) R² = 1.0
- `d(pred)/dt = −0.400 pred + 0.100 prey·pred` (true γ=0.4, δ=0.1) R² = 1.0

All four ODE coefficients recovered exactly.

### Gray-Scott Pattern Analysis (Reaction-Diffusion)

**Target:** Turing instability boundary, wavelength scaling λ ~ √D_v.

**Method:** Scanned 121 (f, k) parameter combinations on 128×128 grid (10,000
timesteps each). Classified patterns via FFT power spectrum. Varied D_v at
fixed (f=0.035, k=0.065) for wavelength scaling.

**Results:**
- Phase diagram: 83 uniform, 26 spots, 6 stripes, 6 complex (4 pattern types)
- 35 Turing instability boundary points mapped in (f, k) space
- D_v wavelength scaling: correlation with √D_v = 0.927 (9 data points)
- PySR wavelength equation: R² = 0.985

---

## 4. The Universality Argument

This section makes the explicit case for why this system is not limited to any particular domain. The claim is strong and deliberate: **any real-world phenomenon that can be simulated can be learned, explored, and understood by this system.** The architecture is domain-agnostic. Only the simulation backend changes.

### 4.1 The Logical Chain

The argument rests on four observations, each well-established independently:

1. **Every real-world system is a dynamical system.** Whether it is a car on a highway, a protein folding in solution, a hurricane forming over the Atlantic, or an economy responding to policy changes -- every system has a state that evolves over time according to some dynamics. Those dynamics may be deterministic or stochastic, continuous or discrete, well-understood or unknown. But they exist, and they can be described mathematically. This is not a philosophical claim. It is the foundational assumption of all physics, biology, chemistry, economics, and engineering.

2. **Every dynamical system can be simulated.** If a system has state and dynamics, it can be simulated on a computer. The simulation may be approximate. The simulation may be expensive. The simulation may require simplifying assumptions. But a simulation can be constructed. This is also well-established: we simulate weather systems, nuclear reactors, protein dynamics, galactic collisions, traffic networks, financial markets, epidemics, and quantum systems. For virtually every domain of human inquiry, simulators already exist -- often multiple competing ones at different fidelity levels.

3. **Every simulation can train a world model.** A world model learns from sequences of states. It observes state(t), state(t+1), state(t+2), and learns the transition function that generates these sequences. It does not care what those states represent. A 128x128 concentration field, a 12-dimensional robot joint configuration, a 1000-agent economic network, a 3D protein backbone -- to the world model, they are all tensors evolving over time. The RSSM architecture (and its successors) has been proven across 150+ diverse environments in DreamerV3. The learning algorithm is genuinely domain-agnostic.

4. **Discoveries from the world model transfer to the real world.** If the simulation faithfully captures the relevant physics, then equations, phase boundaries, and scaling laws discovered within the simulation are equations, phase boundaries, and scaling laws of the real system. This is how computational science has always worked: we trust simulations because they are grounded in validated physical laws. A governing equation discovered from Gray-Scott simulation data is the actual Gray-Scott equation. A scaling law discovered from projectile simulation data is the actual projectile scaling law. The world model adds efficiency (dreaming is faster than simulating), not a new layer of approximation.

The chain is: **Real World -> Dynamical System -> Simulation -> World Model -> Discovery -> Back to Real World.**

The system does not need to understand the domain. It needs a simulator. Everything else follows.

### 4.2 Why the Architecture Is Domain-Agnostic

Every component in the pipeline operates on generic mathematical objects, not domain-specific representations:

| Component | What It Sees | Domain-Specific? |
|-----------|-------------|-----------------|
| Problem Architect | Natural language text | No -- LLM handles any domain |
| Domain Classifier | Keywords and physics descriptors | Routing only -- does not constrain learning |
| Simulation Builder | Parameter dictionaries and config | Template selection only |
| Simulation Environment | State arrays, timesteps, parameters | **Yes -- this is the only domain-specific component** |
| World Model (RSSM) | Tensors of shape (T, *state_shape) | No -- learns from any state sequence |
| Encoder/Decoder | Input/output tensors | Architecture selection (CNN vs MLP), not learning algorithm |
| Explorer | Parameter ranges, uncertainty scores | No -- operates on scalar parameter values |
| Analyst (PySR/SINDy) | Numerical arrays (features, targets) | No -- fits equations to any data |
| Ablation Study | Metric function over parameters | No -- domain enters only through the metric |
| Communicator | Equations, plots, text | No -- reports findings from any domain |

**The only domain-specific component is the simulation backend itself.** Everything upstream (problem parsing) and downstream (learning, exploration, analysis, reporting) is generic. Adding a new domain means writing one new class: a `SimulationEnvironment` subclass with `reset()`, `step()`, and `observe()` methods. The rest of the pipeline works unchanged.

### 4.3 Concrete Domain Applications

The following table is not speculative. For every domain listed, production-quality simulators already exist, the state representation is well-defined, and the type of discoveries the system would make are scientifically meaningful.

#### Physical Sciences and Engineering

| Domain | State Representation | Existing Simulators | What the System Would Discover |
|--------|---------------------|--------------------|-----------------------------|
| **Fluid dynamics** | Velocity/pressure fields on 3D grid | OpenFOAM, PhiFlow, JAX-CFD | Turbulence scaling laws, drag coefficients, vortex shedding frequencies, Reynolds number transitions |
| **Structural engineering** | Stress/strain tensors, displacement fields | FEniCS, Abaqus, ANSYS | Failure mode boundaries, load-bearing capacity scaling, resonance frequencies, fatigue life equations |
| **Thermodynamics** | Temperature/entropy fields, phase fractions | COMSOL, OpenFOAM | Phase transition boundaries, heat transfer coefficients, critical point equations, efficiency scaling |
| **Electromagnetics** | Electric/magnetic field vectors on grid | MEEP, COMSOL, CST | Antenna gain patterns, waveguide mode equations, resonance conditions, scattering cross-sections |
| **Acoustics** | Pressure fields, wave amplitudes | k-Wave, COMSOL | Room resonance modes, absorption coefficients, diffraction patterns, noise reduction scaling |
| **Plasma physics** | Particle distributions, field potentials | GENE, GS2, BOUT++ | Instability thresholds, confinement scaling laws, turbulent transport coefficients |
| **Quantum systems** | Wave functions, density matrices | QuTiP, Qiskit Aer, PennyLane | Entanglement dynamics, decoherence rates, optimal control pulses, phase diagram of quantum materials |

#### Robotics and Autonomous Systems

| Domain | State Representation | Existing Simulators | What the System Would Discover |
|--------|---------------------|--------------------|-----------------------------|
| **Robotic manipulation** | Joint angles, velocities, contact forces | MuJoCo, Isaac Sim, PyBullet | Grasp stability boundaries, force scaling laws, optimal trajectory equations, workspace reachability maps |
| **Legged locomotion** | Body pose, joint torques, ground contacts | MuJoCo, Brax, Isaac Gym | Gait phase diagrams, energy-optimal speed equations, stability margins, terrain adaptation scaling |
| **Autonomous vehicles** | Vehicle state (x, y, v, theta), traffic state | CARLA, SUMO, LGSVL | Braking distance equations, traffic flow phase transitions, collision avoidance boundaries, fuel-optimal speed profiles |
| **Drone dynamics** | 6-DOF pose, rotor speeds, wind field | AirSim, Flightmare, PyFlyt | Lift-to-drag scaling, stability boundaries vs wind speed, energy-optimal hover equations, payload capacity curves |
| **Swarm robotics** | Multi-agent positions, velocities, communications | NetLogo, Mesa, custom JAX | Emergent formation equations, consensus convergence rates, optimal swarm density, communication range scaling |

#### Life Sciences

| Domain | State Representation | Existing Simulators | What the System Would Discover |
|--------|---------------------|--------------------|-----------------------------|
| **Protein dynamics** | Atomic coordinates, velocities | OpenMM, GROMACS, AMBER | Folding rate equations, stability phase diagrams, binding affinity scaling, allosteric pathway identification |
| **Drug interactions** | Molecular conformations, binding energies | AutoDock, Schrodinger, RDKit | Dose-response curves, synergy/antagonism boundaries, selectivity equations, pharmacokinetic scaling |
| **Gene regulatory networks** | Expression levels, promoter states | BioNetGen, COPASI, custom ODE | Bistability conditions, oscillation period equations, noise filtering thresholds, network motif functions |
| **Epidemiology** | Compartment populations (S, I, R, ...) | EpiModel, GLEAM, Mesa | R0 formulas, herd immunity thresholds, intervention timing optimization, variant competition dynamics |
| **Ecology** | Species populations, resource levels | NetLogo, EcoSim, custom ODE | Coexistence boundaries, trophic cascade equations, biodiversity-stability relationships, extinction thresholds |
| **Neuroscience** | Neuron membrane potentials, synaptic weights | NEURON, Brian2, NEST | Firing rate equations, synchronization boundaries, learning rule scaling, information capacity formulas |

#### Earth and Climate Sciences

| Domain | State Representation | Existing Simulators | What the System Would Discover |
|--------|---------------------|--------------------|-----------------------------|
| **Weather prediction** | Atmospheric fields (T, P, wind, humidity) | WRF, MPAS, FV3 | Storm formation conditions, jet stream scaling, precipitation thresholds, teleconnection equations |
| **Climate modeling** | Global temperature, CO2, ice extent, ocean state | CESM, GFDL, E3SM | Climate sensitivity equations, tipping point boundaries, carbon cycle feedback strengths, sea level scaling |
| **Ocean dynamics** | Current velocities, temperature, salinity | MOM6, NEMO, ROMS | Thermohaline circulation stability, eddy transport scaling, upwelling condition equations |
| **Seismology** | Stress fields, fault slip, wave propagation | SPECFEM3D, SW4 | Earthquake scaling laws (Gutenberg-Richter), rupture propagation speed equations, aftershock rate formulas |
| **Hydrology** | Water table, soil moisture, river flow | ParFlow, MODFLOW, HEC-RAS | Flood threshold equations, groundwater recharge rates, drought propagation scaling |

#### Social Sciences and Economics

| Domain | State Representation | Existing Simulators | What the System Would Discover |
|--------|---------------------|--------------------|-----------------------------|
| **Financial markets** | Asset prices, order books, agent wealth | Agent-based models, custom ODE | Volatility scaling laws, crash precursor equations, market microstructure phase transitions |
| **Urban traffic** | Vehicle counts, signal states, road network | SUMO, MATSim, VISSIM | Congestion phase transitions, optimal signal timing equations, capacity scaling laws, route choice equilibria |
| **Supply chains** | Inventory levels, demand, transport state | AnyLogic, SimPy, custom | Bullwhip effect equations, optimal inventory formulas, disruption propagation speed, resilience scaling |
| **Social dynamics** | Opinion vectors, network connections | Mesa, NetLogo, custom | Polarization tipping points, consensus convergence rates, information cascade thresholds, influence scaling |
| **Energy systems** | Grid load, generation, storage levels | PyPSA, PLEXOS, GridLAB-D | Renewable integration thresholds, storage sizing equations, grid stability boundaries, pricing equilibria |

#### Materials Science and Chemistry

| Domain | State Representation | Existing Simulators | What the System Would Discover |
|--------|---------------------|--------------------|-----------------------------|
| **Molecular dynamics** | Atomic positions and velocities | LAMMPS, GROMACS, JAX-MD | Phase transition temperatures, diffusion coefficients, crystal growth rates, mechanical property scaling |
| **Crystal growth** | Lattice site occupancies, temperature field | KMC codes, phase-field models | Growth rate equations, morphology phase diagrams, defect formation thresholds, supersaturation scaling |
| **Battery chemistry** | Ion concentrations, electrode potentials | PyBaMM, COMSOL | Degradation rate equations, capacity fade scaling, optimal charging profiles, thermal runaway boundaries |
| **Polymer dynamics** | Chain conformations, entanglement state | HOOMD-blue, LAMMPS | Viscosity scaling laws, glass transition equations, self-assembly phase diagrams |
| **Catalysis** | Surface coverage, reaction intermediates | CANTERA, custom KMC | Reaction rate equations, selectivity phase diagrams, poisoning thresholds, turnover frequency scaling |

### 4.4 The Self-Driving Car Example

To make the universality argument concrete, consider autonomous vehicles in detail.

**The dynamical system.** A car on a road has state: position (x, y), heading (theta), velocity (v), steering angle (delta), plus the state of surrounding vehicles, pedestrians, traffic signals, and road geometry. The dynamics: tire-road friction models, engine torque curves, aerodynamic drag, suspension response, and the decision-making of other road users. All of this is well-characterized physics and well-studied behavioral models.

**The simulation.** CARLA, SUMO, and LGSVL are production-quality open-source simulators. They model vehicle dynamics, sensor physics (camera, lidar, radar), traffic behavior, weather effects, and road networks. Billions of simulation miles have been run by Waymo, Cruise, and Tesla for autonomous vehicle development.

**What the world model learns.** The RSSM trains on trajectories: sequences of (vehicle_state, traffic_state, action, next_state). It learns a compressed representation of how the driving environment evolves. From this, it can dream: "What happens if I brake now? What if the car ahead swerves? What if it starts raining?"

**What the system discovers.** Not a driving policy (that is a control problem, not a discovery problem). Instead:
- **Braking distance equations** as a function of speed, road surface, tire condition, and vehicle mass. The system would rediscover and refine the standard braking distance formula, and potentially find corrections for conditions not well-characterized analytically.
- **Traffic flow phase transitions.** At what vehicle density does free flow transition to congested flow? The system would discover the fundamental diagram of traffic flow and its dependence on road geometry, speed limits, and driver behavior distributions.
- **Collision probability boundaries.** For a given speed, following distance, and reaction time, what is the boundary between safe and unsafe? The system would map this as a phase diagram in (speed, distance, reaction_time) space.
- **Fuel/energy-optimal speed profiles.** What velocity trajectory minimizes energy consumption for a given route, traffic pattern, and vehicle? The system would discover the governing equations of eco-driving.
- **Sensor degradation scaling.** How does perception accuracy degrade with rain intensity, fog density, or sun angle? The system would find the scaling laws that govern sensor reliability.

These discoveries are genuinely useful for autonomous vehicle engineering. They are not the driving policy itself, but the scientific understanding that informs better policies, better testing, and better safety analysis.

### 4.5 The Robotics Example

Consider a robotic arm performing pick-and-place tasks.

**The dynamical system.** State: 7 joint angles, 7 joint velocities, end-effector pose, gripper state, object pose, contact forces. Dynamics: rigid body mechanics, joint actuator models, contact and friction physics, object inertia.

**The simulation.** MuJoCo, Isaac Sim, and PyBullet provide high-fidelity physics simulation with differentiable contact models. Thousands of robotics researchers use these daily.

**What the system discovers:**
- **Grasp stability phase diagrams.** For a given object geometry and friction coefficient, what gripper configurations produce stable grasps? The system maps this as a phase boundary in (grip_force, contact_angle, friction) space.
- **Workspace reachability equations.** What is the analytical boundary of the robot's reachable workspace as a function of link lengths? The system rediscovers forward kinematics equations.
- **Energy-optimal trajectory scaling.** How does the minimum-energy trajectory between two points scale with distance, payload mass, and speed constraint? The system finds the governing equations.
- **Force-accuracy tradeoffs.** What is the relationship between applied force, contact stiffness, and positioning accuracy? The system discovers the compliance equations that govern precision manipulation.
- **Failure mode boundaries.** At what combination of speed, payload, and joint angle does the robot lose stability or exceed torque limits? The system maps these as phase boundaries.

### 4.6 Why "Any Situation" Is Not an Exaggeration

The claim "any situation" has a precise meaning: **any situation whose dynamics can be simulated on a computer.** This covers:

- **Any physical system** governed by known or partially known laws of physics. This includes everything from quantum mechanics to cosmology, and every engineering discipline: mechanical, electrical, chemical, civil, aerospace, biomedical, nuclear, environmental.

- **Any biological system** that can be modeled as interacting components with quantifiable state. This includes molecular biology, cell biology, physiology, ecology, epidemiology, and neuroscience.

- **Any social/economic system** that can be modeled as agents with defined behaviors and interactions. This includes markets, traffic, supply chains, social networks, elections, and urban systems.

- **Any abstract mathematical system** that can be evolved forward in time. This includes cellular automata, graph dynamics, game theory, and any computable dynamical system.

The only situations excluded are those that cannot be simulated at all:
- Systems whose fundamental laws are unknown AND no empirical model exists (e.g., quantum gravity -- but even here, proposed models can be simulated)
- Systems that require exponential computational resources just to represent the state (e.g., exact quantum simulation of >50 qubits -- though approximate methods exist)
- Systems where the relevant variables cannot be identified or measured

For everything else -- which encompasses virtually all of science, engineering, medicine, economics, and social science -- the system applies. The V1 implementation supports three domains. The architecture supports all of them. Adding each new domain requires writing one simulation backend class. The learning, exploration, analysis, and communication pipeline works unchanged.

### 4.7 From Three Domains to Unlimited

The path from V1 (3 domains) to universal coverage is not a change in architecture. It is a matter of:

1. **Adding simulation backends.** Each new domain needs a `SimulationEnvironment` subclass. For domains with existing JAX-compatible simulators (Brax for robotics, JAX-MD for molecular dynamics, diffrax for ODEs), this is straightforward wrapping. For other domains, it requires a JAX reimplementation or a bridge to external simulators.

2. **Expanding encoder/decoder architectures.** The CNN encoder handles spatial fields. The MLP encoder handles vector states. Future domains may need graph neural network encoders (for molecular structures), point cloud encoders (for particle systems), or sequence encoders (for time series). The RSSM core and training loop remain unchanged.

3. **Growing the composable dynamics library.** Each domain studied adds reusable dynamics modules (gravity, diffusion, reaction, friction, advection) that accelerate future domain construction. The 100th domain is easier than the 10th.

4. **Accumulating cross-domain analogies.** As the system studies more domains, its library of mathematical structures grows. The Cross-Domain Analogy Engine (Section 8.2) automatically identifies when a new domain shares mathematical structure with a previously studied one, enabling transfer of solution techniques and discovered equations.

The architecture was designed for this scaling from day one. V1 proves it works on three domains. The universality is inherent in the design.

---

## 5. What Makes This Different

### Multi-Agent Architecture with Specialized Roles

The system is a coordinated ensemble of specialized agents, each responsible for a distinct phase of the discovery process. A simulation construction agent builds environments. A world model training agent manages the learning loop. An exploration agent drives the search through state space. An analysis agent extracts patterns and equations. A validation agent checks results against independent evidence.

This decomposition matters because each agent can be optimized for its specific task, equipped with the right tools, and evaluated against clear criteria. A failure in one component can be diagnosed and corrected without destabilizing the whole system.

### Meta-Cognitive Reasoning

The system reasons about its own process. It can recognize when its simulation is too coarse to answer the question being asked, when its world model is uncertain in a region that matters, when its exploration is stuck in a local basin, or when its extracted insights are not yet sufficiently validated. This self-awareness drives adaptive behavior: the system refines its approach in response to what it discovers about its own limitations.

This is not a fixed pipeline. It is a loop with reflection.

### Progressive Fidelity

Not every question requires a high-fidelity simulation from the start. The system begins with the simplest model that could plausibly capture the relevant dynamics -- a point-mass approximation, a mean-field model, a linearized system. It uses the world model trained on this simple simulation to identify which aspects of the dynamics matter most. Then it selectively adds complexity where needed: resolving spatial structure in the regions where it matters, adding nonlinearities where they dominate, introducing stochastic effects where deterministic models fail.

This progressive refinement is computationally efficient and scientifically sound. It mirrors how good scientists work: start with the simplest viable model, understand it thoroughly, then add complexity with purpose.

### Verification Before Communication

Every claimed discovery passes through a validation pipeline before being reported. The system checks whether an extracted equation actually predicts the simulation dynamics. It tests whether a claimed causal relationship holds under intervention. It verifies whether an identified optimal strategy is robust to perturbation. It flags the confidence level and the scope of applicability for every finding.

The goal is not to generate plausible-sounding insights. It is to generate correct ones.

### Composable Dynamics Modules

Rather than building each simulation from scratch, the system maintains a library of composable dynamics modules: integrators, force fields, reaction networks, transport operators, boundary conditions, coupling mechanisms. New simulations are assembled by selecting and connecting the right modules, configuring their parameters, and validating the assembled system.

This composability accelerates simulation construction and ensures that well-tested numerical components are reused. It also provides a natural vocabulary for the system to reason about dynamics at a structural level.

### LLM as Conductor, Specialized Tools as Orchestra

A large language model serves as the central coordinator -- interpreting the problem specification, planning the research strategy, dispatching tasks to specialized agents, and synthesizing results into coherent findings. But the LLM does not do the numerical work itself. It orchestrates purpose-built tools: simulation engines, neural network trainers, symbolic regression systems, causal inference algorithms, and visualization generators.

This separation preserves the LLM's strengths (reasoning, planning, communication, cross-domain knowledge) while delegating its weaknesses (precise numerical computation, long-horizon optimization) to tools designed for those tasks.

---

## 6. Research Landscape

This section surveys the key areas of active research that form the technical foundation of the project: world models, AI for scientific discovery, equation discovery, simulation infrastructure, and exploration methods.

### 6.1 World Models (State of the Art 2024-2025)

World models learn to predict future states of an environment given actions, enabling planning, imagination, and data-efficient reinforcement learning. The field has seen rapid architectural diversification since 2023, moving beyond recurrent state-space models toward transformers, diffusion models, and hybrid approaches.

**Key architectures:**

- **DreamerV3** (Hafner et al., 2023). RSSM combining deterministic and stochastic latent states. A single set of hyperparameters works across 150+ tasks. Imagination horizon limited to approximately 15 steps; discrete latent categories may underrepresent continuous physical quantities.
- **DreamerV4** (Hafner et al., 2025). Next-generation world model with improved long-horizon dreaming, better uncertainty calibration, and more expressive latent representations. A natural upgrade path for our RSSM -- the modular architecture means only `world_model/rssm.py` and `trainer.py` need updating. Key advantages for scientific discovery: longer imagination horizons enable better parameter space exploration, and improved uncertainty estimates drive more efficient exploration.
- **IRIS** (Micheli et al., 2023). VQ-VAE discrete autoencoder with autoregressive Transformer dynamics. Highly sample-efficient on Atari 100K. Quadratic attention cost limits rollout horizon; codebook collapse reduces representational capacity over time.
- **DIAMOND** (Alonso et al., 2024). Conditional diffusion model generating future frames via iterative denoising. State-of-the-art on Atari 100K with visually sharp generation. Slow inference due to multi-step denoising; temporal consistency not inherently enforced.
- **Genie 1/2** (Bruce et al., 2024; DeepMind, 2024). Learns interactive environments from unlabeled video with latent action discovery. No action labels needed during training. Low frame rate (~1 FPS in Genie 1); no guarantee of physical consistency.
- **UniSim** (Yang et al., 2023). Unified real-world simulator across diverse action modalities. Handles multiple action types in a single model. Approximate physics only; quality degrades beyond ~10-20 seconds.
- **GameNGen** (Valevski et al., 2024) and **Oasis** (Decart, 2024). Neural game engines demonstrating real-time interactive generation. Single-domain; no persistent world state; no physics model.
- **Cosmos** (NVIDIA, 2024-2025). Family of world foundation models for Physical AI. Open weights with multiple model sizes. Early stage; not yet physics-accurate for safety-critical applications.

**Architecture comparison:**

| Model | Architecture Type | Action Space | Resolution | Horizon | Inference Speed |
|-------|------------------|--------------|------------|---------|-----------------|
| DreamerV3 | RSSM (recurrent) | Continuous/discrete | 64x64 | ~15 steps | Real-time |
| DreamerV4 | RSSM (improved) | Continuous/discrete | 64x64+ | ~50+ steps | Real-time |
| IRIS | Transformer + VQ-VAE | Discrete | 64x64 | ~50 tokens | Near real-time |
| DIAMOND | Diffusion (U-Net) | Discrete | 64x64 | ~100 steps | Slow (multi-step) |
| Genie 1 | Transformer + VQ-VAE | Latent (discovered) | 256x256 | Seconds | ~1 FPS |
| UniSim | Video diffusion | Multi-modal | 256x256 | ~10-20s | Slow |
| GameNGen | Fine-tuned diffusion | Discrete (game) | 320x200 | ~3s memory | Real-time |
| Oasis | Transformer | Discrete (game) | 360p+ | Context window | Real-time |
| Cosmos | Hybrid (AR + diffusion) | Configurable | Up to 1080p | Variable | Variable |

**Known failure modes across current world models:**

| Failure Mode | Description | Typical Onset |
|---|---|---|
| Compounding error drift | Predictions diverge from reality over time | 50-200 steps |
| Object permanence failure | Objects disappear or teleport when occluded | When objects leave frame |
| Counting errors | Cannot track exact counts | Immediate for non-trivial counts |
| Novel physics / OOD dynamics | Incorrect predictions for unseen scenarios | Immediate upon encountering OOD |
| Spatial consistency in 3D | View-dependent inconsistencies, geometry violations | Multi-view or navigation |
| Hallucination of plausible states | Visually plausible but physically impossible states | Increases with horizon |

### 6.2 AI for Scientific Discovery

- **FunSearch** (Romera-Paredes et al., 2024). Combines LLMs with evolutionary search to discover new mathematical constructions by searching in program space. Discovered results for the cap set problem and online bin packing surpassing previously known bounds. Requires automatically computable evaluation functions, limiting applicability to broader science.
- **AlphaFold 2/3** (Jumper et al., 2021; Abramson et al., 2024). Near-experimental accuracy for protein structure prediction (AF2) and biomolecular complex prediction (AF3, using diffusion-based generation). Predicts static structures only; does not model dynamics or function.
- **GNoME** (Merchant et al., 2023). Graph neural networks predicting stability of inorganic crystals. 2.2 million new stable structures predicted, 380,000 DFT-validated. Stability does not equal synthesizability; limited to inorganic crystals.
- **AI Scientist** (Lu et al., 2024). End-to-end automated research from idea to paper at ~$15 per paper. Quality well below human standards with factual errors, trivial contributions, and misinterpreted results. Demonstrates automatable pipeline; exposes the quality ceiling.

**SciML ecosystem:**

| Technique | Description | Maturity |
|---|---|---|
| PINNs (Physics-Informed Neural Networks) | Neural nets trained with PDE residual as loss | Research-mature |
| Neural Operators (FNO, DeepONet) | Learn mappings between function spaces | Research-mature |
| Universal Differential Equations (UDEs) | Hybrid ODE/PDE with neural network components | Research-mature |
| Equation-Free Methods | Neural nets as surrogates within multiscale simulation | Experimental |

Julia SciML is the most complete and integrated ecosystem. NVIDIA Modulus is the primary production-oriented framework. Key challenge: training PINNs is notoriously finicky; neural operators require large training datasets; UDEs require differentiable solvers.

### 6.3 Equation Discovery

| Tool | Method | Speed | Interpretability | Variable Scalability | Novelty Potential |
|---|---|---|---|---|---|
| PySR | Evolutionary search | Slow (hours) | High | Low (~10) | High |
| SINDy/PySINDy | Sparse regression | Fast (seconds) | High | Medium (~50) | Low (library-limited) |
| AI Feynman | Physics-prior + search | Medium | High | Low (~10) | Medium |
| LLM-SR | LLM proposal + eval | Fast per-iter | High | Medium | Medium (prior-biased) |
| KANs | Gradient-based training | Medium | Medium-High | Unknown | Medium |

PySR is the current best-in-class tool for symbolic regression, using multi-population evolutionary search with Pareto front optimization. SINDy discovers governing equations by sparse regression over candidate function libraries -- fast but limited to functions in the library. KANs (Kolmogorov-Arnold Networks) replace fixed MLP activations with learnable edge functions, offering improved interpretability but remaining unproven at scale.

### 6.4 Simulation Infrastructure and Transfer

**Differentiable simulation frameworks:**

| Framework | Physics Domains | Differentiable | Maturity |
|---|---|---|---|
| JAX-MD | Molecular dynamics | Yes (full) | Research-mature |
| DiffTaichi | MPM, FEM, fluids, rigid body | Yes (full) | Research-mature |
| PhiFlow | Fluids (incompressible, smoke) | Yes (full) | Research-mature |
| Brax / MJX | Rigid-body robotics | Yes (full) | Production-ready |
| NVIDIA Warp | General GPU physics | Yes (full) | Production-ready |
| Genesis | Multi-physics | Yes (partial) | Early production |

Key limitation: gradient quality degrades over long horizons. In chaotic systems, gradients through more than approximately 100 time steps become numerically unreliable.

**Sim-to-real transfer maturity:**

| Domain | Status | Key Challenge |
|---|---|---|
| Legged locomotion | Production | Solved for flat/moderate terrain |
| Wheeled navigation | Production | Solved for structured environments |
| Rigid-body manipulation | Experimental | Contact modeling, precise force control |
| Dexterous manipulation | Experimental | High-DOF, contact-rich |
| Deformable objects | Unsolved | Modeling deformation accuracy |
| Fluid interaction | Unsolved | Turbulence, splashing, surface tension |

**Neural surrogates for computational cost reduction:**

| Approach | Speedup | Domain | Maturity |
|---|---|---|---|
| Fourier Neural Operators (FNO) | 1,000-10,000x | PDE-governed systems | Research-mature |
| Graph Neural Network Simulators | 1,000-10,000x | Particle-based physics | Research-mature |
| Weather foundation models | 1,000-10,000x | Global weather prediction | Production (deployed) |
| Neural radiance / rendering | 10-100x | Visual rendering | Production |

Weather foundation models (GraphCast, GenCast, Aurora, Pangu-Weather) are the flagship success: operationally deployed at weather services, producing skillful 10-day forecasts in minutes rather than hours.

### 6.5 Exploration and Causal Discovery

**Exploration methods:** Random Network Distillation (RND) is the most widely used curiosity method due to simplicity and effectiveness. Go-Explore separates "returning to frontier" from "exploring beyond frontier," solving hard exploration problems where all other methods failed. LLM-guided exploration combines language-level world knowledge with RL exploration but remains experimental.

**Causal discovery methods:**

| Method Class | Examples | Maturity |
|---|---|---|
| Constraint-based | PC, FCI, RFCI | Production-ready |
| Score-based | GES, FGES | Production-ready |
| Continuous optimization | NOTEARS, DAGMA | Research-mature (~100 variables) |
| LLM + data hybrid | LLM proposes graph, data refines | Experimental |

Production frameworks include DoWhy (Microsoft), EconML, CausalML, and causal-learn. The hybrid approach of LLM-proposed causal structures refined by data is the most promising direction for automated simulator construction.

---

## 7. Critical Gaps in Current Research

These are areas where no existing research program provides a satisfactory solution, ordered roughly by importance to the project's mission.

### 7.1 Natural Language to Validated Simulation

**Maturity: Very Low.** No existing system takes a natural language description of a physical system and produces a validated, quantitatively accurate simulation. Individual pieces exist (LLMs for code generation, physics engines, testing frameworks) but nobody has assembled and validated the end-to-end pipeline for scientific simulation. The core gap is validation and verification of generated simulations.

### 7.2 Composable Dynamics Modules

**Maturity: Very Low.** Current world models are trained end-to-end on specific domains. There is no modular system where learned "rigid body dynamics," "fluid dynamics," and "electromagnetic field" modules can be composed to simulate a new multi-physics scenario. Genesis (2024) integrates multiple physics engines but through engineering, not learned composition. The gap is learned modules with interface contracts that ensure physical consistency.

### 7.3 Automated Model Selection and Theory Competition

**Maturity: Very Low.** No system automatically generates multiple competing models, evaluates them on held-out data, and selects or synthesizes the best explanation. Bayesian model selection is well-established but requires human-specified candidate models. The gap is an automated system that maintains a portfolio of candidate models and identifies discriminating experiments.

### 7.4 Cross-Domain Transfer of Mathematical Patterns

**Maturity: Very Low.** When symbolic regression discovers that a biological system follows diffusion dynamics, that insight could transfer to other domains with similar structure. No existing system captures and transfers structural mathematical patterns across domains. The gap is a mathematical pattern library with tools for recognizing and applying structural analogies.

### 7.5 Symbolic-Numeric Computation Bridging

**Maturity: Low.** Symbolic computation and numeric computation remain largely separate worlds. Julia's ModelingToolkit.jl comes closest to bridging them, but no system provides a seamless AI-driven bridge that automatically decides when symbolic versus numeric approaches are appropriate.

### 7.6 Automated Progressive Fidelity

**Maturity: Low.** Starting with the simplest possible model and automatically escalating fidelity only when needed. Multi-fidelity optimization combines fidelity levels but does not automatically select or escalate. The gap is a meta-controller that manages a hierarchy of models and decides when to invest in higher fidelity.

### 7.7 Reproducibility Infrastructure for AI-Generated Science

**Maturity: Very Low.** AI-generated scientific claims lack standardized infrastructure for reproducibility. Existing experiment tracking tools (MLflow, Weights & Biases) cover general ML but nothing specific to AI-generated scientific claims, provenance tracking, or automated verification.

### 7.8 Collaborative AI-Human Discovery Interface

**Maturity: Low.** Current AI science tools are either fully automated or fully manual. The middle ground -- where AI and human scientists collaborate as peers -- lacks dedicated interface design, interaction patterns, and workflow support. Notebook assistants are tool-use patterns, not collaboration patterns.

**Gap maturity summary:**

| Gap | Maturity | Nearest Existing Work |
|---|---|---|
| Natural language to validated simulation | Very Low | LLM code generation (unvalidated) |
| Composable dynamics modules | Very Low | Genesis multi-physics (engineered, not learned) |
| Automated model selection / theory competition | Very Low | Bayesian model selection (human-specified models) |
| Cross-domain mathematical pattern transfer | Very Low | LLM analogical reasoning (informal) |
| Symbolic-numeric computation bridging | Low | Julia ModelingToolkit.jl (partial) |
| Automated progressive fidelity | Low | Multi-fidelity optimization (not automated selection) |
| AI science reproducibility infrastructure | Very Low | MLflow / W&B (general ML, not AI-science-specific) |
| Collaborative AI-human discovery interface | Low | Jupyter + AI assistants (tool-use, not collaboration) |

---

## 8. Novel Contributions

This section catalogs nine original architectural contributions that address the critical gaps identified above. None of these exist in published literature as described here, though they build on and extend established foundations. Each idea addresses a specific failure mode or opens a new capability. Together, they form a discovery engine qualitatively different from anything in the literature.

### 8.1 Adversarial Dream Debate

Two world models, trained independently on the same problem (different architectures, random seeds, data orderings), propose competing hypotheses about the system's behavior. When their predictions diverge, a judge agent designs discriminating experiments within the ground-truth simulation to determine which model is correct. The losing model is corrected; the winning model's reliability is reinforced.

This addresses the hallucinated consensus problem in multi-agent AI systems. When multiple agents share training data and architecture, they tend to agree even when wrong -- their agreement is correlated error, not independent evidence. Adversarial Dream Debate breaks this by construction: disagreement identifies regions where at least one model has learned something incorrect, and ground-truth arbitration resolves the dispute.

**Mechanism:** (1) Independent training of two world models with deliberate architectural and procedural differences. (2) Parallel exploration without communication. (3) Systematic disagreement detection classified by magnitude and relevance. (4) Minimum-cost discriminating experiment design. (5) Ground-truth simulation arbitration. (6) Model correction for the losing model. (7) Iterative refinement until remaining disagreements concentrate in genuinely ambiguous regions.

**Research gap filled:** No existing system pits world models against each other for scientific verification. Ensemble methods average predictions rather than resolving disagreements. GAN adversarial training targets realistic generation, not scientific accuracy. AI safety debate operates in language/reasoning contexts, not physical world models. This combines the independence of ensembles, the adversarial pressure of GANs, and the verification logic of debate protocols.

### 8.2 Cross-Domain Analogy Engine

When the system discovers a mathematical pattern in one domain -- a governing equation, a dynamical structure, a symmetry -- it automatically searches for isomorphic structures across every other domain it has studied. Many of the most important scientific breakthroughs came from recognizing such structural analogies: Maxwell's equations and fluid dynamics, Black-Scholes and the heat equation, Shannon entropy and Boltzmann entropy, predator-prey dynamics appearing in ecology, epidemiology, chemical kinetics, and economics. These are exact mathematical isomorphisms, not metaphors, and solution techniques transfer directly across them.

**Mechanism:** (1) Every discovered equation is stored in canonical form, stripped of domain-specific variable names. (2) Structural similarity search compares new patterns against the library using algebraic isomorphism, topological equivalence, symmetry matching, and dimensionless group alignment. (3) Explicit mappings are constructed between matched domains. (4) Cross-domain hypotheses are generated as testable predictions. (5) Analogies are reported with confidence levels and known limitations.

**Research gap filled:** No existing system searches across scientific domains for structural mathematical analogies. Symbolic regression discovers equations within a single domain. Transfer learning transfers statistical features, not mathematical structure. The Cross-Domain Analogy Engine maintains a growing library of mathematical structures and systematically searches for isomorphisms.

### 8.3 FunSearch-Style Program Discovery for Physical Problems

Instead of discovering numerical solutions or symbolic equations, the system discovers programs: executable algorithms, control strategies, and optimization policies. The world model serves as a fast evaluation function, and evolutionary search over LLM-generated programs provides the exploration mechanism.

This extends DeepMind's FunSearch from pure mathematics to physical and scientific problems. The key enabler is the world model as fast surrogate evaluator: where FunSearch requires problems with instant exact evaluation, a trained world model can evaluate candidate programs in milliseconds rather than the seconds or minutes required by full simulation. Ground-truth simulation provides periodic validation to prevent exploitation of model artifacts.

**Research gap filled:** FunSearch demonstrated LLM-guided evolutionary search for mathematical constructions, but only where evaluation is exact and instant. No one has combined FunSearch-style program evolution with learned world models to make this tractable for physical problems.

### 8.4 Dream Journaling and Pattern Mining

Every trajectory imagined by the world model is logged in a structured database with full metadata. A background process continuously mines this archive for patterns that recur across many independent dreams, flagging recurring structures as candidate discoveries and one-off patterns as potential artifacts.

Current world model systems treat each dream as an isolated event. This is wasteful: patterns appearing consistently across hundreds of independent dreams (generated under different conditions and different world model versions) are almost certainly capturing real structure, while patterns appearing once might be artifacts. This distinction -- between observation and replicated finding -- is fundamental to the scientific method.

**Mechanism:** (1) Structured logging of every dream with full context (problem, model version, exploration strategy, parameters, trajectory, detected patterns). (2) Continuous pattern mining: symbolic regression on aggregate data, trajectory clustering, anomaly detection, frequency analysis, temporal trend analysis. (3) Confidence scoring proportional to replication count, weighted by independence. (4) Artifact detection (patterns appearing only in specific model versions flagged). (5) Retrospective cross-problem analysis as the archive grows.

**Research gap filled:** No world model system maintains and mines a historical archive of imagined trajectories. MuZero, Dreamer, and curiosity-driven exploration all treat dreaming as stateless. Dream Journaling adds a memory layer that transforms the system from a stateless exploration engine into a cumulative knowledge-building system.

### 8.5 Socratic Discovery Mode

Instead of operating as fully autonomous, the system periodically pauses its exploration to present its current understanding to the human researcher and ask targeted questions. It seeks human input on where to explore next, which hypotheses to prioritize, and whether intermediate findings align with domain intuition.

The landscape of AI-assisted science is polarized between fully autonomous systems (which lack scientific judgment and produce shallow results) and passive tools (which require the human to drive every decision). The productive middle ground is genuine collaboration during the discovery process, not just before and after it.

**Mechanism:** (1) Autonomous exploration for a configured period. (2) Pause at natural breakpoints (significant pattern found, decision point with multiple directions, high uncertainty in relevant region) to present findings. (3) Integration of human guidance: prioritization adjustments, hypothesis injection, constraint addition, validation shortcuts. (4) Adaptive interaction frequency based on user response patterns. (5) Transparent reasoning at every pause, explaining not just what was found but why the system is asking.

**Research gap filled:** No existing AI science tool supports genuine bidirectional collaboration during the discovery process. Fully autonomous systems have no human-in-the-loop. Notebook assistants respond to requests but do not proactively contribute hypotheses or ask questions. Active learning asks for labels, not strategic scientific questions.

### 8.6 Uncertainty-Driven Exploration

The world model's own predictive uncertainty is used as the primary signal for directing exploration, with a critical distinction between epistemic uncertainty (lack of knowledge, worth exploring) and aleatoric uncertainty (inherent stochasticity, not reducible by more data).

Curiosity-driven exploration conflates these: an agent can get permanently stuck trying to predict inherently chaotic dynamics, wasting compute on regions where no amount of additional data will improve predictions. Random exploration cannot focus compute on the vanishingly small fraction of state space containing scientifically important regions (phase transitions, bifurcations, resonances).

**Mechanism:** (1) Calibrated uncertainty estimation via ensembles, Bayesian neural networks, or Monte Carlo dropout. (2) Epistemic/aleatoric decomposition using established techniques (epistemic from variance of mean predictions across ensemble members; aleatoric from mean of variance predictions). (3) Exploration targeting regions of highest epistemic uncertainty. (4) Adaptive resource allocation as epistemic uncertainty decreases in explored regions. (5) Principled stopping criterion when remaining epistemic uncertainty is low everywhere relevant.

**Research gap filled:** Bayesian optimization uses uncertainty for function optimization, not world model training. RND and curiosity methods do not distinguish epistemic from aleatoric uncertainty. Active learning operates in supervised contexts, not world model dreaming. The novel contribution is principled epistemic/aleatoric decomposition applied to world model dreaming for systematically expanding the frontier of reliable knowledge about a physical system.

### 8.7 Composable Dynamics Modules

Rather than training a monolithic world model from scratch for every new problem, the system maintains a library of reusable learned dynamics components. Each module captures an isolated physical phenomenon -- gravity, friction, diffusion, reaction kinetics -- with a standardized interface. New problems are solved by composing relevant modules and training only a small residual network to capture their interactions.

Every world model in the current literature is trained from scratch on a single domain. A world model for robotic manipulation learns nothing that helps a world model for fluid dynamics, even though both involve rigid body physics, contact forces, and conservation laws.

**Mechanism:** (1) Module pre-training on isolated phenomena (gravity from orbital simulations; diffusion from heat conduction; friction from sliding contact). (2) Standard interface: input is current state vector, output is state derivative contribution, parameters are configurable constants. (3) Automatic module selection by a Domain Classifier agent. (4) Composition by summing derivative contributions (additive dynamics) or complex coupling schemes, with a small residual network for interaction effects. (5) Fast fine-tuning on specific problem data. (6) Library growth when new phenomena are encountered.

**Research gap filled:** No one is building world models from reusable, composable physics components. Monolithic world models (Dreamer, IRIS) have no cross-domain reuse. PINNs hardcode physics as loss constraints, not reusable modules. Foundation model proposals aim for one large model; composable modules offer the modular alternative with independent validation, updating, and debugging.

### 8.8 Automated Ablation Studies

Whenever the system makes a discovery, it automatically conducts ablation studies -- systematically removing or modifying individual factors to determine which are essential and which are incidental. This transforms correlational observations into causal claims.

**Mechanism:** (1) Discovery specification under a set of conditions. (2) Single-factor ablation: remove, modify, or randomize each condition. (3) Multi-factor ablation testing combinations of important factors. (4) Necessity versus sufficiency analysis identifying mechanistic requirements, minimal sufficient subsets, modulatory factors, and irrelevant factors. (5) Robustness quantification measuring sensitivity (sharp threshold versus smooth degradation). (6) Report generation alongside the original discovery.

**Research gap filled:** Ablation studies are standard in ML research but have never been automated for scientific discoveries made by AI systems. AI Scientist reports findings without testing which factors are essential. Symbolic regression discovers equations without testing which terms are necessary. Causal discovery algorithms infer from observation; automated ablation tests through direct intervention. The two are complementary.

### 8.9 Progressive Trust Architecture

The system maintains explicit, quantitative trust scores for every component: each world model (and each region of each world model's state space), each dynamics module, each agent, each analysis type. Trust scores increase when predictions match ground truth and decrease when they fail. These scores directly influence resource allocation and validation requirements.

**Mechanism:** (1) Trust initialization reflecting prior expectations (moderate for new models, high for validated library modules, low for new agents). (2) Bayesian-inspired trust updates: large updates for surprising outcomes, small for expected ones. (3) Trust-informed resource allocation affecting validation intensity, compute budgets, reporting thresholds, and ensemble weighting. (4) Regional trust maps for world models (spatially resolved reliability). (5) User-visible trust scores with full history. (6) Trust decay over time if not reinforced by new evidence.

**Research gap filled:** No existing AI system maintains explicit, quantitative reliability tracking across its components. Ensemble methods compute per-prediction uncertainty without persistent trust history. Multi-agent systems treat all agents as equally reliable. Bayesian optimization maintains uncertainty about the objective function, not about its own surrogate model. The Progressive Trust Architecture treats component reliability as a first-class quantity that is estimated, tracked, and used for decision-making.

### Summary Matrix

| Idea | Complexity | V1 Priority | Builds On |
|------|-----------|-------------|-----------|
| 1. Adversarial Dream Debate | Medium | High | AI safety debate, ensembles |
| 2. Cross-Domain Analogy Engine | High | Medium | Symbolic regression, structure mapping |
| 3. FunSearch Program Discovery | Medium | Medium | FunSearch, program synthesis |
| 4. Dream Journaling | Medium | High | Meta-analysis methodology |
| 5. Socratic Discovery Mode | Low-Medium | Critical | Active learning, human-in-the-loop ML |
| 6. Uncertainty-Driven Exploration | Medium | Critical | Bayesian optimization, epistemic uncertainty |
| 7. Composable Dynamics Modules | High | Medium | PINNs, modular networks |
| 8. Automated Ablation Studies | Low-Medium | High | Causal inference, ML ablation practice |
| 9. Progressive Trust Architecture | Medium | High | Bayesian updating, safe RL |

**Recommended implementation order.** Phase 1 (core discovery loop): Uncertainty-Driven Exploration, Socratic Discovery Mode, Automated Ablation Studies, Dream Journaling, Progressive Trust Architecture, Adversarial Dream Debate. Phase 2 (extended capabilities): FunSearch Program Discovery, Composable Dynamics Modules, Cross-Domain Analogy Engine.

---

## 9. Positioning and Comparison

### 9.1 Comparison Matrix

| Dimension | Simulating Anything | AI Scientist (Sakana) | FunSearch (DeepMind) | DreamerV3 | SciML.jl | Traditional Simulation |
|-----------|--------------------|-----------------------|---------------------|-----------|----------|----------------------|
| **Input type** | Natural language problem description | Natural language research idea | Well-defined objective function (code) | Pre-built environment (code) | Equations + model structure (Julia code) | Domain expertise + manual code |
| **Simulation construction** | Automatic (LLM + template library + progressive fidelity) | None | None | None | Manual (user specifies equations) | Fully manual (months of expert effort) |
| **Dynamics learning** | Hybrid world model (physics backbone + neural residual + hard constraints) | None | None | Monolithic RSSM (no physics priors) | Neural ODE / UDE (differentiable, physics-hybrid) | No learning (hand-specified equations) |
| **Exploration strategy** | Uncertainty-driven (epistemic/aleatoric) + curiosity + goal-conditioned | LLM proposes incremental variations | Evolutionary search over LLM programs | Actor-critic in imagination (reward-maximizing) | Manual parameter sweeps or gradient-based | Human-guided parameter sweeps |
| **Insight extraction** | Symbolic regression, sparse dynamics, causal analysis, ablation studies | LLM writes analysis section | Program output is the insight | None (outputs are policies) | Manual analysis by user | Manual (plot inspection, curve fitting) |
| **Validation** | Multi-layered: dimensional analysis, conservation enforcement, ground-truth spot-checks, adversarial debate | LLM-based reviewing (poorly calibrated) | Automatic evaluator (exact, limited) | Task performance (reward) | Manual verification | Manual V&V, peer review |
| **Interpretability** | Symbolic equations, phase diagrams, causal graphs, plain-language reports | Papers (may contain errors) | Executable programs | Black box (latent states) | Equations are the input | Full (human wrote the model) |
| **Domain generality** | Multi-domain (expanding) | ML experiments only | Pure math and combinatorics | Single environment per run (no cross-domain transfer) | Any domain as differential equations | Single domain per project |
| **Human involvement** | Collaborative: Socratic check-ins, human redirects | Minimal: fully autonomous | Minimal: human defines objective | None during training | High: human specifies everything | Total: human drives every step |
| **Reproducibility** | Built-in: provenance ledger, dream journal, versioned checkpoints | Ad-hoc | Reproducible (deterministic programs) | Standard ML reproducibility | Standard computational reproducibility | Often poor |

### 9.2 Detailed Comparisons

#### vs. AI Scientist (Sakana AI, 2024)

AI Scientist is an end-to-end LLM research pipeline that generates ML experiment papers at approximately $15 per paper. Key differences: AI Scientist writes ML papers by making incremental variations on fixed experiment templates; this project discovers scientific insights across arbitrary domains. AI Scientist has no simulation capability and no world model; every experiment is a full code execution. AI Scientist validates through an LLM-based reviewer that is poorly calibrated, accepting papers with factual errors; this project validates through ground-truth simulation, dimensional analysis, conservation law enforcement, and adversarial dream debate. AI Scientist produces papers; this project produces validated equations, phase diagrams, and causal graphs with explicit confidence scores.

What is learned: end-to-end automation is possible, but quality suffers without computation-grounded validation. LLM-based reviewing is not a substitute for simulation-based verification.

#### vs. FunSearch (DeepMind, 2024)

FunSearch searches program space for pure mathematical problems with exact evaluation functions. Key differences: this project searches simulation space for physical problems requiring simulation to evaluate. FunSearch requires well-defined scalar objectives; this project explores open-ended problems where the evaluation function is itself discovered. FunSearch has no dynamics learning or simulation. FunSearch's discoveries are provably correct via exact evaluators; physical insights require multi-stage validation because no single evaluator can certify a physical claim.

What is learned: the LLM + evolutionary search paradigm is powerful; searching in program space produces verifiable, interpretable results. The project's FunSearch-Style Program Discovery (Contribution 7.3) adapts this paradigm using the world model as fast approximate evaluator.

#### vs. DreamerV3 (Hafner et al., 2023)

DreamerV3 trains policies to maximize reward in fixed environments. Key differences: this project discovers scientific insights across self-constructed environments. DreamerV3 takes environments as given; this project constructs them from natural language. DreamerV3's monolithic RSSM has no physics priors; this project's hybrid world model uses a physics backbone for known dynamics with a neural residual for the unknown remainder. DreamerV3 explores to maximize reward; this project explores to understand dynamics via uncertainty-driven and curiosity-driven search. DreamerV3 has no analysis or interpretation layer; this project's entire purpose is producing human-interpretable scientific knowledge.

What is learned: the RSSM architecture is proven and robust, adopted as the V1 backbone. Symlog predictions and free-bit KL balancing stabilize learning across value scales. The approximately 15-step imagination horizon motivates the hybrid physics-neural approach to extend reliable imagination.

#### vs. SciML.jl Ecosystem

SciML provides building blocks (solvers, neural ODE implementations, sensitivity analysis). Key differences: this project provides an end-to-end discovery pipeline from natural language to validated insight. SciML requires Julia expertise, numerical methods knowledge, and user-specified equations; this project accepts natural language and infers the rest. SciML has no world model, no agent-based orchestration, and no insight extraction pipeline.

What is learned: Universal Differential Equations (neural + mechanistic hybrid) is the right paradigm for combining known physics with learned components. Differentiable simulation is essential. ModelingToolkit.jl's symbolic-numeric bridge is the most sophisticated implementation of a capability this project also needs.

#### vs. Traditional Simulation Workflows

Key differences: traditional workflows require deep domain expertise at every step; this project automates problem specification, simulation construction, exploration, and analysis. A researcher can run hundreds of parameter combinations; a world model can dream through millions. Traditional analysis is manual and intuition-dependent; this project's analysis is systematic and exhaustive. Traditional reproducibility is often poor; this project has built-in provenance tracking.

What is learned: progressive fidelity is how experts actually work. Domain knowledge is irreplaceable for validation. Simulation verification and validation (V&V) is a mature discipline with established practices that inform the automated validation pipeline.

#### vs. AlphaFold / GNoME (DeepMind)

AlphaFold and GNoME are single-domain prediction systems with deep domain-specific inductive biases. Key differences: this project targets multi-domain generality using general-purpose world models. AlphaFold and GNoME are prediction systems (input to output); this project is a discovery system (question to understanding). Neither explores; this project explores systematically. AlphaFold will always predict protein structures better because it encodes 50 years of structural biology; this project's value is providing the discovery pipeline for the vast space of problems where no domain-specific equivalent exists.

What is learned: the most successful AI-for-science systems combine strong inductive biases with large-scale data. Validation against established experimental data is essential for credibility.

### 9.3 Unique Positioning

This project occupies a position that no existing system occupies. It is the only proposed system that combines:

1. **Automated simulation construction** from natural language problem descriptions.
2. **World model-based exploration** using hybrid physics-neural dynamics learning.
3. **Symbolic insight extraction** producing human-interpretable equations, phase diagrams, and causal graphs.
4. **Multi-agent verification** with dimensional analysis, conservation law enforcement, ground-truth spot-checks, and adversarial dream debate.

It is not the best at any single component. DreamerV3 has more battle-tested world models. PySR has more sophisticated symbolic regression. SciML.jl has more comprehensive equation solvers. FunSearch has stronger verifiability guarantees. AlphaFold has higher accuracy in its domain. The value is in the integration: the fact that a researcher can describe a problem in plain language and receive, hours later, a validated report containing discovered equations, phase diagrams, and causal explanations, with explicit confidence scores and full reproducibility provenance.

**The research lab analogy.** The closest analog is not a piece of software but a well-organized research laboratory:

| Lab Role | System Equivalent | Function |
|----------|-------------------|----------|
| Principal investigator | Problem Architect Agent | Defines the research question |
| Domain specialist | Domain Classifier + Simulation Builder | Selects physics, builds the apparatus |
| Lab technician | Simulation Validator Agent | Calibrates instruments, runs quality checks |
| Postdoc | World Model Trainer Agent | Develops theoretical framework |
| Graduate students | Explorer Agents (parallel) | Run thousands of experiments |
| Data analyst | Analyst Agent (PySR, SINDy) | Processes data into findings |
| Devil's advocate | Skeptic Agent | Challenges every finding |
| Theorist | Counterfactual Agent | Establishes causal mechanisms |
| Interdisciplinary collaborator | Cross-Domain Analogy Agent | Notices connections to other fields |
| Lab manager | Coordinator Agent | Allocates resources, manages budgets |
| Communications officer | Communication Agent | Writes reports, makes figures |

### 9.4 What This System Is Not

- **Not a replacement for domain expertise.** It augments domain expertise by automating computational infrastructure. The human provides the questions worth asking and the judgment about what constitutes a meaningful finding.
- **Not a physics engine.** It orchestrates existing physics engines (MuJoCo, PhiFlow, JAX-MD, Brax) and learns world models from their output.
- **Not a paper-writing tool.** It produces findings (validated equations, phase diagrams, causal graphs), not manuscripts. The quality standard is scientific correctness, not narrative plausibility.
- **Not a general AI.** It is a specialized system for simulation-driven scientific discovery, constrained to the simulate-explore-discover pipeline.
- **Not ready for safety-critical applications.** V1 is a research system. Discoveries should be treated as hypotheses to be independently verified, not engineering specifications.
- **Not a silver bullet.** Some problems are not simulation-amenable. Some domains have dynamics too complex for current world models. The claim is solving a specific class of problems -- those where simulation-driven exploration can yield scientific insight -- more efficiently than current approaches.

---

## 10. Target Capabilities

### Natural Language Problem Specification

A researcher states what they want to understand in plain language. The system parses this into a formal problem specification: the entities involved, the dynamics to model, the quantities of interest, the success criteria. Ambiguities are resolved through clarifying dialogue.

Example inputs:
- "What traffic light timing minimizes average commute time in a grid city with rush-hour congestion?"
- "How does antibiotic cycling frequency affect resistance evolution in a hospital ICU?"
- "What wing geometry maximizes lift-to-drag ratio at low Reynolds numbers for a 200g drone?"

### Automated Simulation Environment Construction

The system translates the formal specification into executable simulation code, selecting appropriate physics models, numerical methods, resolution, and boundary conditions. Validation runs against analytical solutions, known empirical data, conservation laws, and dimensional analysis.

### World Model Training and Dreaming

The system trains a neural world model on simulation trajectories, learning a latent representation and transition function for fast rollouts. Dreaming is structured and targeted at: regions of high uncertainty (epistemic exploration), regions near phase transitions or bifurcations (critical phenomena), regions maximizing information gain (goal-directed search), and regions maximally different from previously explored states (diversity-driven coverage).

### Causal and Counterfactual Reasoning

The system performs interventional reasoning, constructing causal graphs from experimental manipulation and reasoning counterfactually about what would have happened under different conditions.

### Symbolic Equation Discovery

From numerical data, the system searches for compact symbolic expressions governing the observed dynamics using symbolic regression (PySR), sparse identification of nonlinear dynamics (SINDy), and related techniques. Discovered equations are checked for dimensional consistency, physical plausibility, and out-of-sample predictive accuracy.

### Human-Interpretable Insight Extraction

Raw simulation data is transformed into phase diagrams, bifurcation maps, sensitivity analyses, governing equations with physical interpretations, plain-language mechanism summaries, and visualizations. Every finding is reported with confidence level, supporting evidence, conditions of validity, and known limitations.

### Cross-Domain Analogy Detection

When the system encounters dynamics structurally similar to dynamics in another domain, it flags the analogy -- including shared topological structure, similar bifurcation behavior, analogous symmetry breaking, and equivalent conservation laws. These analogies transfer solution techniques, suggest new hypotheses, and reveal deep connections between superficially unrelated phenomena.

---

## 11. Long-Term Vision

### A Research Partner, Not a Tool

A tool executes instructions. A research partner contributes ideas, challenges assumptions, notices things missed, and brings complementary strengths to a shared investigation. The long-term vision is the latter: a system that can take a vague research question, help sharpen it into a precise investigation, carry out the computational work, and present findings that genuinely advance understanding.

This does not mean replacing human scientists. It means amplifying them. The system handles the computationally intensive, combinatorially explosive tasks. The human provides the questions worth asking, the domain intuition that constrains the search, the judgment about meaningful findings, and the broader scientific context.

### Collaborative Human-AI Scientific Discovery

The most productive mode is interactive. The researcher poses a question; the system builds a simulation, explores it, and reports initial findings. The researcher examines findings, asks follow-up questions, suggests alternative hypotheses, points out overlooked factors. The system incorporates this feedback, refines its investigation, and produces deeper results. This back-and-forth converges on understanding faster than either party could achieve alone.

The system supports this collaboration at every stage. Its outputs are interpretable and auditable. Its reasoning is transparent. Its confidence estimates are calibrated and honest. When it does not know something, it says so.

### Democratizing Simulation-Driven Research

Today, simulation-based scientific discovery is concentrated in groups that can afford the expertise and infrastructure to build, run, and interpret complex simulations. A system that constructs simulations from natural language, trains world models on them, and extracts interpretable insights lowers the barrier to entry. A biologist wondering about gene regulatory network dynamics should not need to become a computational physicist. An urban planner investigating traffic patterns should not need to write a custom agent-based model from scratch.

The goal is not eliminating the need for domain expertise. Deep knowledge will always produce better questions and more nuanced interpretations. The goal is making simulation and exploration infrastructure accessible to anyone with a well-formed question.

### Toward General Scientific Reasoning

The furthest horizon is a system that can engage in genuine scientific reasoning across arbitrary domains -- not by memorizing known results, but by constructing models, running experiments, discovering patterns, and building understanding from the ground up. This requires advances in automated simulation construction, world model architectures, exploration strategies, symbolic reasoning, causal inference, and human-AI interaction. No single component is sufficient; the value lies in their integration.

The path is incremental: start with well-understood domains where ground truth is available, demonstrate that the system can rediscover known results, then progressively tackle problems where the answers are not yet known. Each domain conquered adds to the library of composable dynamics modules, exploration strategies, and cross-domain analogies, making the next domain easier to approach.

---

## 12. References

### World Models
- Hafner, D. et al. (2023). "Mastering Diverse Domains through World Models." arXiv:2301.04104.
- Micheli, V. et al. (2023). "Transformers are Sample-Efficient World Models." ICLR 2023.
- Alonso, E. et al. (2024). "Diffusion for World Modeling: Visual Details Matter in Atari." NeurIPS 2024.
- Bruce, J. et al. (2024). "Genie: Generative Interactive Environments." ICML 2024.
- Yang, M. et al. (2023). "Learning Interactive Real-World Simulators." arXiv:2310.06114.
- Valevski, D. et al. (2024). "Diffusion Models Are Real-Time Game Engines." arXiv:2408.14837.
- NVIDIA (2025). "Cosmos: World Foundation Models for Physical AI." arXiv:2501.03575.

### AI for Scientific Discovery
- Romera-Paredes, B. et al. (2024). "Mathematical Discoveries from Program Search with Large Language Models." Nature 625, 468-475.
- Jumper, J. et al. (2021). "Highly Accurate Protein Structure Prediction with AlphaFold." Nature 596, 583-589.
- Abramson, J. et al. (2024). "Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3." Nature 630, 493-500.
- Merchant, A. et al. (2023). "Scaling Deep Learning for Materials Discovery." Nature 624, 80-85.
- Lu, C. et al. (2024). "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery." arXiv:2408.06292.

### Equation Discovery
- Cranmer, M. (2023). "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl." arXiv:2305.01582.
- Brunton, S. et al. (2016). "Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems." PNAS 113(15), 3932-3937.
- Udrescu, S.-M. & Tegmark, M. (2020). "AI Feynman: A Physics-Inspired Method for Symbolic Regression." Science Advances 6(16), eaay2631.
- Liu, Z. et al. (2024). "KAN: Kolmogorov-Arnold Networks." arXiv:2404.19756.

### Simulation and Transfer
- Kumar, A. et al. (2021). "RMA: Rapid Motor Adaptation for Legged Robots." RSS 2021.
- Hu, Y. et al. (2020). "DiffTaichi: Differentiable Programming for Physical Simulation." ICLR 2020.
- Schoenholz, S. & Cubuk, E. D. (2020). "JAX, M.D.: A Framework for Differentiable Physics." NeurIPS 2020.
- Freeman, C. D. et al. (2021). "Brax: A Differentiable Physics Engine for Large Scale Rigid Body Simulation." NeurIPS 2021 Datasets and Benchmarks.
- Ma, Y. J. et al. (2023). "Eureka: Human-Level Reward Design via Coding Large Language Models." arXiv:2310.12931.

### Causal Discovery
- Zheng, X. et al. (2018). "DAGs with NO TEARS: Continuous Optimization for Structure Learning." NeurIPS 2018.
- Bello, K. et al. (2022). "DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization." NeurIPS 2022.

### Exploration
- Burda, Y. et al. (2019). "Exploration by Random Network Distillation." ICLR 2019.
- Ecoffet, A. et al. (2021). "First Return, Then Explore." Nature 590, 580-586.

### Neural Surrogates and Weather
- Lam, R. et al. (2023). "Learning Skillful Medium-Range Global Weather Forecasting." Science 382(6677), 1416-1421.
- Price, I. et al. (2024). "Probabilistic Weather Forecasting with Machine Learning." Nature 637, 84-90.
- Bodnar, C. et al. (2024). "Aurora: A Foundation Model of the Atmosphere." arXiv:2405.13063.

### Multi-Agent Systems
- Qian, C. et al. (2023). "ChatDev: Communicative Agents for Software Development." arXiv:2307.07924.
- Hong, S. et al. (2023). "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework." arXiv:2308.00352.
- Wu, Q. et al. (2023). "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." arXiv:2308.08155.

### Additional References Cited in Novel Contributions
- Irving, G. et al. (2018). "AI Safety via Debate." arXiv:1805.00899.
- Lakshminarayanan, B. et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS 2017.
- Goodfellow, I. et al. (2014). "Generative Adversarial Nets." NeurIPS 2014.
- Kendall, A. & Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" NeurIPS 2017.
- Depeweg, S. et al. (2018). "Decomposition of Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning." ICML 2018.
- Pathak, D. et al. (2017). "Curiosity-driven Exploration by Self-Predictive Next Feature Prediction." ICML 2017.
- Snoek, J. et al. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms." NeurIPS 2012.
- Settles, B. (2009). "Active Learning Literature Survey." Computer Sciences Technical Report 1648, University of Wisconsin-Madison.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference.* 2nd ed. Cambridge University Press.
- Raissi, M. et al. (2019). "Physics-Informed Neural Networks." Journal of Computational Physics 378, 686-707.
- Gentner, D. (1983). "Structure-Mapping: A Theoretical Framework for Analogy." Cognitive Science 7(2), 155-170.
- Garcia, J. & Fernandez, F. (2015). "A Comprehensive Survey on Safe Reinforcement Learning." JMLR 16, 1437-1480.

---

*This document consolidates the research vision, literature survey, novel contributions, and competitive positioning for the Simulating Anything project. It is a living document that will evolve as the project develops and the research landscape advances.*
