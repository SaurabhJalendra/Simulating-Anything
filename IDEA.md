# Simulating-Anything — Idea Document

## The Problem

Current AI is two disconnected paradigms.

**Large language models** are powerful pattern matchers over text. They cannot run forward simulations of consequences, cannot cleanly distinguish causation from correlation, cannot accumulate skills, cannot learn continuously, and cannot know the boundaries of their own knowledge.

**Reinforcement learning agents** can act in environments and learn from experience but lack abstract reasoning, struggle with sample efficiency, do not transfer across domains, and cannot articulate or compose what they have learned.

Neither, on its own, exhibits intelligence in the full sense — the kind that can perceive, plan with foresight, simulate consequences before acting, accumulate composable skills, recognize what it does not know, form new concepts from experience, and reason both numerically and symbolically about the world.

What is missing is the **architecture** that combines them under a single, principled cognitive loop.

## The Solution

A modular cognitive architecture grounded in Yann LeCun's Autonomous Machine Intelligence (AMI) blueprint, built on a substrate of 1500+ physics simulators, with the following properties:

- **World models as the substrate of experience.** Multiple specialized world models (RSSM, JEPA-family, PDE foundation models) covering different domain classes, with a router that selects the right one for each task.
- **Causal, not merely predictive.** A structural causal model layer on top of the world model enables Pearl-style interventions and counterfactual reasoning — the difference between "rain correlates with wet ground" and "rain causes wet ground."
- **Hierarchical planning with foresight.** Plans decompose from natural-language goals to abstract strategies to primitive actions, with every level checked against world-model rollouts before execution.
- **Symbolic reasoning as a first-class channel.** SINDy, PySR, KAN-SR, and theorem-search extract explicit knowledge (equations, invariants, laws) from world-model rollouts, making the system's reasoning extractable and auditable.
- **A growing skill library, Voyager-style.** Skills are acquired through curiosity-driven play, self-verified, retrievable by natural-language description, and composable — primitive skills compose into mid-level skills compose into strategies.
- **Four-tier memory.** Working (current context), episodic (concrete past episodes), semantic (the LLM-maintained wiki — extracted facts and concepts), procedural (skills).
- **Reflection and self-modeling.** The system tracks its own capabilities, knowledge gaps, and historical failure patterns; uses this self-knowledge to choose what to practice and what to verify.
- **Autonomous concept formation.** A concept-formation engine watches the experience stream, detects regularities, compresses them into named abstractions, and grows the wiki without human-in-the-loop ingestion.
- **Curiosity-driven exploration.** Information-gain and free-energy signals drive what experiments to run next, replacing uniform parameter sweeps.
- **The LLM as scaffold, not core.** A frontier LLM (Claude) serves as a tool the architecture uses for narrow tasks: parsing goals, generating simulator code, structuring reasoning traces, retrieving from the wiki, writing reports. The LLM never drives the main cognitive loop.

The system loops: perceive → update world model → reason → plan → act → observe → learn → reflect.

## Who It's For

- **Researchers** in AI-for-science, world-model methods, neural-symbolic integration, and cognitive architectures, who need a clean modular substrate to investigate individual components in isolation and integration.
- **Scientists** in domains where understanding (extractable equations, causal structure, identified bifurcations) matters more than predictive accuracy alone.
- **Open-source community** — anyone who wants to study, extend, or build on a working AMI-style implementation.

This is not a product for end users. This is a research platform.

## Why This, Why Now

- **The architecture pieces have matured separately and need integration.** RSSM (DreamerV3, Nature 2025), LeWM (Maes/LeCun, March 2026), Voyager (NVIDIA 2023), Sakana AI Scientist v2 (April 2025), Robin (FutureHouse, May 2025), KAN-SR (2025), WSINDy (2021) — each component now has a working open-source implementation. The pieces are ready; the integration is not done.
- **Frontier labs are pursuing this trajectory in closed code.** OpenAI o1/o3, Anthropic agent SDK, DeepMind SIMA. The open-source community needs a credible reference implementation.
- **The simulator substrate is not built.** Most cognitive-architecture research uses toy environments (Atari, Minecraft, MuJoCo). 1500+ physics domains spanning rigid body, fluid, agent-based, chaotic, reaction-diffusion is unique substrate that does not exist elsewhere — and has been built over the past two months on this project.
- **Sutton's "Era of Experience" (2025) is the philosophical case** that intelligence requires environments to learn in, not just text corpora. Our 1500-domain simulator library is exactly that.

## Success Looks Like

In one year, the project demonstrates an agent that:

- Receives a natural-language physics problem it has not seen before.
- Decides which simulator to spawn, parameterizes it, runs it.
- Plans a solution by composing skills from its library and rolling out the world model.
- Reasons symbolically (extracts equations, identifies bifurcations) about what it observes.
- Verifies its answer by running counterfactuals through the causal layer.
- Reports a confidence-calibrated answer with extractable reasoning trace.
- Outperforms both LLM-only baselines and pure-RL baselines on tasks requiring forward simulation and consequence prediction.
- Has accumulated a library of ~100 self-discovered, self-verified skills that compose hierarchically.
- Has autonomously grown its wiki to ~500 concept and entity pages from experience.

Plus: at least one peer-reviewed publication, an open-source codebase that other researchers extend, and a working demonstration that LLM-only systems cannot replicate.

## Non-Goals

The project is not, and will not claim to be:

- **AGI.** No solo project with current hardware can build AGI. The goal is the cleanest open-source AMI-style architecture, demonstrated on a constrained domain.
- **A universal problem solver.** "Solves any kind of problem" is unscientific. The system is bounded by the domains in its simulator library and the modalities it perceives.
- **Competitive with frontier labs on raw capability.** Our edge is architectural clarity and openness, not parameter count or training compute.
- **A product.** This is a research platform.
- **A pure-LLM agent system.** The LLM is strictly subordinate. We are not building "Claude with extra tools."
- **A pure-RL system.** Reinforcement learning is one component, not the paradigm.
- **Autonomous in deployment.** All capabilities are demonstrated under human-in-the-loop research conditions. No autonomous deployment to external systems.

## Open Questions

- **Causal layer formalism.** Pearl's structural causal models, do-calculus, identifiability — which subset is implementable on top of RSSM/LeWM latents at this scale?
- **Skill library growth without bloat.** As skills accumulate, retrieval and composition cost grows. Pruning, abstraction, hierarchical organization — empirical territory.
- **Concept formation grounding.** Compression-based concept discovery (MDL/Solomonoff) is theoretically clean but practically hard. What's the minimum viable version?
- **Self-model fidelity.** How accurate can the system be about its own knowledge gaps? Calibrated uncertainty is hard.
- **Cross-domain transfer.** Skills learned in Lorenz dynamics — do they transfer to Brusselator? What's the right abstraction layer for transfer?
- **Benchmark design.** Existing benchmarks are LLM-shaped (text in, text out). What does a fair benchmark for an embodied cognitive architecture look like?
- **The simulator-to-real gap.** Eventually, agents trained in simulation should transfer to real-world tasks. What's the shortest credible path?

## Phasing

The project executes in four phases:

- **Phase 1 (current → 2 weeks):** ship the discovery-engine paper as Phase 1 leverage. The discovery engine becomes the *first demonstrated capability* of the architecture, proving the substrate works.
- **Phase 2 (weeks 3–6):** scaffold the cognitive architecture. Refactor the codebase to expose the cognitive structure. Write foundational docs (this one, plus VISION.md and ARCHITECTURE.md). Stub all 18 components.
- **Phase 3 (months 2–6):** implement core cognitive components (planner, skill library, causal layer, reflection, concept formation). Build the curriculum. Run integration experiments on physics-reasoning benchmarks.
- **Phase 4 (months 6–12):** integration paper — "AMI-Lite: A Modular Cognitive Architecture for Embodied Reasoning." Multiple ablation studies. Open-source release. Engage research community.

See `docs/VISION.md` for the full philosophical case and `docs/ARCHITECTURE.md` for the technical specification.
