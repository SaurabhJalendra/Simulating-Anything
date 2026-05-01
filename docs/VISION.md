# Simulating-Anything — Vision

> A modular cognitive architecture in the Autonomous Machine Intelligence tradition, demonstrated on physics reasoning. Open-source, research-grade, honest about scope.

## What this document is for

This is the long-form philosophical and strategic case for the project. For the one-page north star, see `../IDEA.md`. For the technical specification of the architecture, see `ARCHITECTURE.md`.

This document answers: *why this project, why now, what we are building, what we are not building, and what intellectual lineage we belong to*.

---

## 1. The Problem with Current AI

In 2026, "AI" effectively means *frontier large language models*. They are remarkable: fluent across hundreds of domains, capable of in-context learning, useful as tools.

But they are not intelligence in the full sense.

A language model cannot:

- **Run forward simulations of consequences.** It can describe what is likely to happen in words it has seen, but it cannot actually run a physical system forward to see what happens. There is no internal world to consult.
- **Distinguish causation from correlation.** It can produce text that sounds causal — "X causes Y" — without any internal representation that would let it reason about interventions: *what would happen if I did X*, holding everything else fixed.
- **Accumulate skills.** Each conversation starts fresh. Even with retrieval-augmented generation, the model itself does not change. There is no equivalent of "I am better at chess this week than last week."
- **Know what it does not know.** Calibrated uncertainty is poor; hallucination is well documented; the model often says confidently incorrect things in domains it was not trained on.
- **Form new concepts from experience.** Concepts are baked in at training time. New concepts emerging from interaction are not represented internally.
- **Plan with foresight.** Chain-of-thought is not planning — it is autoregressive token generation that *resembles* deliberation. It cannot check intermediate steps against a world model, cannot backtrack systematically, cannot estimate the value of different paths through state space.

These are not minor limitations. They are structural consequences of the architecture: a learned function from text to text, frozen at training, without an internal model of the world or of itself.

Reinforcement learning agents have the opposite limitations: they can act in environments, learn from experience, and accumulate competence — but lack abstract reasoning, struggle with sample efficiency, do not transfer across domains, and cannot articulate or compose what they have learned.

**Neither paradigm, on its own, exhibits intelligence in the full sense.** What is missing is the architecture that combines them.

---

## 2. What Intelligence Requires

We follow LeCun's diagnosis (*A Path Towards Autonomous Machine Intelligence*, 2022) and Sutton's argument (*Era of Experience*, 2025): intelligence is not a function from text to text. It is a **continuously running closed-loop process** that perceives, models the world, plans, acts, observes consequences, learns, and reflects.

Five properties such a system must have:

1. **Causal counterfactual simulation.** Not "what is likely next" but "what would happen if I did X instead of Y." This requires a world model you can intervene on — Pearl's third rung of causation, not the first.

2. **Sample-efficient learning.** Humans learn new games from one demonstration. The difference is strong priors (world models, skill libraries) that constrain hypotheses. Sample efficiency is the product of prior strength and search ability.

3. **Continuous self-modification.** Skills, world models, beliefs, goals — all updated by experience. A system frozen at training is not learning.

4. **Awareness of own knowledge boundaries.** Knowing when you don't know. Quantified uncertainty at every layer drives where to spend compute and what to verify.

5. **Multi-horizon goal-directed behavior.** Goals at seconds, minutes, days, years simultaneously. Each horizon constrains the levels below.

These five define our design constraints. Every architectural component in `ARCHITECTURE.md` exists to enable at least one of them.

---

## 3. The Approach: AMI-Lite, on a Physics Substrate

We are building a modular cognitive architecture in the Autonomous Machine Intelligence tradition. The core components are:

- **World model ensemble** — multiple specialized models (RSSM, LeWM, PDE foundation models), with a router that selects the right one per task.
- **Causal graph layer** — Pearl-style structural causal model on top of the world model, enabling interventions and counterfactuals.
- **Hierarchical planner** — natural-language goals decompose to abstract plans to primitive actions, every level checked against world-model rollouts.
- **Symbolic reasoning channel** — SINDy, PySR, KAN-SR extract equations and laws from world-model rollouts. Reasoning is auditable, not black-box.
- **Voyager-style skill library** — skills acquired through curiosity-driven play, self-verified, retrievable by natural language, hierarchically composable.
- **Four-tier memory** — working, episodic, semantic (the wiki), procedural.
- **Reflection and self-model** — the system tracks its own capabilities, gaps, and failure patterns; this drives meta-learning.
- **Autonomous concept formation** — the system grows its own knowledge base from experience, not just human-ingested sources.
- **Curiosity-driven exploration** — information-gain and free-energy signals drive next experiments.
- **LLM as scaffold** — Claude is called for narrow tasks (parse goals, generate code, structure traces, retrieve, write reports). Never drives the main loop.

The substrate is the **1500+ physics simulator library** built over the past months on this project. Most cognitive-architecture research uses toy environments (Atari, Minecraft, MuJoCo); we have a substrate spanning rigid body, fluid dynamics, agent-based, chaotic ODE, reaction-diffusion, MHD, social, biological, climate — that is unique.

---

## 4. Why This, Why Now

Three reasons the timing is right.

**The architectural pieces have matured separately.** RSSM (DreamerV3, *Nature* 2025), LeWM (Maes/LeCun, March 2026), Voyager (NVIDIA 2023), Robin (FutureHouse, May 2025), Sakana AI Scientist v2 (April 2025), KAN-SR (2025), Weak SINDy (Messenger & Bortz 2021), Time-Warp-Attend (Talmon et al, ICLR 2024) — each component now has a working open-source implementation. The pieces exist. The integration does not.

**Frontier labs are pursuing this trajectory in closed code.** OpenAI o1/o3 (reasoning at inference), Anthropic agent SDK (multi-agent orchestration), DeepMind SIMA and Genie (generalist agents in simulators), Meta V-JEPA 2 (self-supervised video world models). All proprietary, all narrowly scoped, none open-source as a complete architecture. The open-source community needs a credible reference implementation.

**The simulator substrate is not built elsewhere.** Cognitive architecture research has always been bottlenecked by environment availability. SOAR, ACT-R, Voyager — each constrained by the worlds they could operate in. Our 1500-domain library, built incrementally over the past two months, is the largest open physics simulator collection of its kind. Discarding it to "restart with a clean cognitive architecture" would be wasteful. The right move is to expose the cognitive layer on top of what we have.

---

## 5. Intellectual Lineage

We stand on:

- **Yann LeCun's AMI position paper** (2022) — the architectural blueprint
- **Karl Friston's free-energy principle** — the unifying mathematical objective for perception, action, learning
- **Judea Pearl's causal calculus** — the formal language for interventions and counterfactuals
- **Richard Sutton's "Era of Experience"** (2025) — the philosophical case that intelligence requires environments
- **Voyager (NVIDIA, 2023)** — the closest existing implementation pattern: skills + curriculum + self-verification, in Minecraft
- **DeepMind's SIMA / AdA / Gato** — generalist agents in simulators
- **DreamerV3 (Hafner et al, *Nature* 2025)** — proves world models work at scale across many tasks
- **The symbolic regression literature** — AI-Feynman, PySR, KAN, ODEFormer — provides the explicit-knowledge channel
- **Anthropic's multi-agent research system** (June 2025) — the orchestrator-worker pattern

We are not original in any individual component. The originality is in the **integration** — assembling these pieces into the cleanest open-source AMI-style architecture, on a physics substrate that no other group has.

---

## 6. What We Are Not Building

Honesty about scope is essential.

**We are not building AGI.** No solo project on consumer hardware can build AGI in 2026. Frontier labs spending billions are not there. Anyone claiming otherwise is selling something.

**We are not building a universal problem solver.** "Solves any kind of problem" is unscientific. The system will be bounded by the domains in its simulator library and the modalities it perceives.

**We are not competitive on scale.** Our architecture is ~15M to ~1B parameters total across world models — orders of magnitude smaller than frontier LLMs. Our edge is **architectural clarity and openness**, not capacity.

**We are not building a product.** No end users. This is a research platform.

**We are not "Claude with tools."** The LLM is strictly subordinate. The cognitive loop runs without it; the LLM is a callable utility.

**We are not a pure-RL system.** Reinforcement learning is one component (skill acquisition, planner training), not the paradigm.

**We will not deploy autonomously.** All capabilities are demonstrated under human-in-the-loop research conditions.

---

## 7. What Success Looks Like

At one year, the project demonstrates an agent that:

- Receives a natural-language physics problem it has not been trained on.
- Decides which simulator from its library to spawn; parameterizes it; runs it.
- Plans a solution by composing skills from its library and rolling out the world model.
- Reasons symbolically — extracts equations, identifies bifurcations, runs counterfactual interventions.
- Verifies its answer by re-running predictions through the causal layer.
- Reports a confidence-calibrated answer with auditable reasoning trace.
- Outperforms LLM-only and pure-RL baselines on tasks requiring forward simulation, consequence prediction, and skill composition.
- Has accumulated ~100 self-discovered, self-verified skills.
- Has autonomously grown its wiki to ~500 concept and entity pages from experience.

Plus: at least one peer-reviewed publication, an open-source codebase that other researchers fork and extend, and a working demonstration that LLM-only systems cannot replicate.

This is achievable. Not AGI — but a credible reference implementation of the LeCun vision, applied to physics reasoning, fully open. That is itself rare and valuable.

---

## 8. The Discovery Engine as Phase 1

The current project is, today, a scientific discovery engine. 1500+ simulators. 277 trained RSSM world models. 316 discovery campaigns. 118 validated bifurcations. SINDy/PySR equation recovery. A working autoresearch loop.

This is not abandoned in the pivot. It becomes the **Phase 1 demonstrated capability** of the broader cognitive architecture.

The discovery engine *is* the system, in narrow form: simulator + world model + reasoning + autoresearch loop. To extend it to a general-purpose cognitive architecture, we add:

- A planner that can use the discovery engine as one of many tools
- A skill library that captures successful discovery strategies as skills
- A reflection layer that tracks which kinds of problems the system handles well
- A concept-formation engine that turns discovery patterns into named abstractions
- A causal layer that distinguishes "this parameter changes that observable" (correlation) from "this parameter *causes* that change" (intervention)

The discovery paper, when it ships, becomes external proof that the substrate works. That is leverage for the bigger project.

---

## 9. The Phasing

| Phase | Duration | Goal |
|---|---|---|
| **Phase 1: Land the discovery engine** | 2 weeks | Execute the 6 architectural changes from `docs/adr/0001-...md`. Ship the discovery paper. |
| **Phase 2: Cognitive scaffolding** | 4 weeks | Refactor codebase to expose cognitive structure. Stub all 18 components. Foundational docs (this one + ARCHITECTURE.md + IDEA.md). |
| **Phase 3: Core cognition** | 4 months | Implement planner, skill library, causal layer, reflection, concept formation, four-tier memory. Build curriculum. Run integration experiments. |
| **Phase 4: Integration & paper** | 6 months | "AMI-Lite: A Modular Cognitive Architecture for Embodied Reasoning." Ablation studies. Open-source release. Community engagement. |

Phase 1 is engineering on the existing system. Phases 2–4 are the new direction.

---

## 10. The Honest Closing

This project is ambitious. It will not, in any plausible timeline, achieve AGI or universal problem-solving. Such claims are unscientific.

But it can credibly produce:

1. The cleanest open-source modular cognitive architecture in 2026
2. A working demonstration that LLM-subordinate cognitive architectures outperform LLM-only systems on tasks requiring forward simulation and consequence prediction
3. A research platform usable by other groups for years
4. Multiple peer-reviewed publications across world models, symbolic reasoning, skill acquisition, and cognitive architectures
5. A unique simulator substrate that exists nowhere else

That is enough. That is, in fact, more than most open-source AI projects achieve in their lifetime.

The work is hard. Many of the components (causal discovery, concept formation, self-modeling) are research-grade hard with no general solutions. We will hit walls. We will document those walls honestly and move forward.

The aim is not to claim victory. The aim is to build the cleanest open implementation of Yann LeCun's vision, demonstrated on a substrate no one else has, with all limitations made public, in a form that the research community can extend.

That is the project.
