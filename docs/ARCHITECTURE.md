# Simulating-Anything — Architecture Specification

> Formal technical specification for the AMI-Lite cognitive architecture. For the philosophical case, see `VISION.md`. For the project north star, see `../IDEA.md`.

## 0. Document scope

This document specifies the 18 components of the cognitive architecture, their interfaces, the data flow between them, the hard problems each must solve, and the mapping to existing project code. It is the canonical reference for what the system *should* be; it is not yet the description of what the system *is*. Components marked **EXISTS**, **PARTIAL**, or **MISSING** indicate current state.

---

## 1. Design Principles

Five constraints govern every architectural decision:

| Principle | What it forces |
|---|---|
| **Causal counterfactual simulation** | World models must support `do(state, action)` interventions, not only `predict(state, action)`. |
| **Sample-efficient learning** | Strong priors (world models + skills) must constrain hypothesis space. Search ability must be high. |
| **Continuous self-modification** | All learned components must be updateable from experience without catastrophic forgetting. |
| **Calibrated uncertainty** | Every output carries a confidence; high uncertainty triggers verification or abstention. |
| **Multi-horizon goals** | Plans operate at seconds-to-years simultaneously; each horizon constrains levels below. |

When a design decision violates one of these, the decision is wrong.

---

## 2. The 18 Components

### 2.1 Perception (multimodal encoder) — PARTIAL

Encodes raw observations into the unified latent space the world model operates on.

**Inputs:** vector states (current), images (planned), text (planned), graphs (planned).
**Outputs:** unified latent embedding `z_t`, plus per-modality uncertainty estimate.
**Interfaces:** `Perception.encode(obs: dict[modality, ndarray]) -> tuple[Tensor, Uncertainty]`.

**Current code:** `src/simulating_anything/world_model/encoder.py`, `src/simulating_anything/world_model/advanced_encoders.py` (CNNEncoder, MLPEncoder, GNNEncoder, CNN3DEncoder, SetEncoder).

**Missing:** unified multimodal interface; per-modality uncertainty; text and image modalities are not exercised in main pipeline.

**Effort to extend:** 2–3 weeks for multimodal unification; longer for full modality coverage.

---

### 2.2 World Model Ensemble — PARTIAL

The substrate of consequence prediction. Multiple specialized models, with a router selecting per-task.

**Inputs:** latent state `z_t`, action `a_t`.
**Outputs:** predicted next-latent `ẑ_{t+1}`, predicted observation `ô_{t+1}` (optional decoder), epistemic uncertainty.
**Critical interface:** *intervention* — `WorldModel.do(z, action) -> z_next` must be a true do-operation, not conditional prediction. This separates causal use from associational use.

**Current code:** `src/simulating_anything/world_model/rssm.py`, `rssm_v2.py`, `ensemble.py`. 277 trained RSSM models on disk.

**Planned additions (per ADR-0001):**
- LeWM A/B implementation (`world_model/lewm.py`, NEW)
- Multi-physics-pretraining backbone (`world_model/pde_fm.py`, NEW)
- Router module that picks which model fits current state (`world_model/router.py`, NEW)

**Hard problem:** ensemble disagreement is a credible epistemic-uncertainty signal but expensive — N forward passes per step. Resource-manager component negotiates depth.

---

### 2.3 Causal Graph Layer — MISSING

Sits on top of world models. Maintains a structural causal model (SCM): which variables cause which, with quantified strengths.

**Inputs:** sequence of (state, action, next-state) triples from world-model rollouts and real experience.
**Outputs:** SCM graph `G = (V, E)` where `V` are observable variables and `E` are directed causal edges with strengths. Plus identification queries: "is the effect of X on Y identifiable from observational data alone?" (Pearl's identifiability conditions).

**Why this is required:** prediction is not understanding. A predictive world model can produce correct outputs from wrong causal structure (correlation alone). To answer "*what would happen if* I changed X?" we need do-calculus, which requires the SCM.

**Implementation sketch:**
- Use intervention experiments — actively perturb variables in simulator, measure effect on others. This is what 1500+ simulators give us that pure-observational ML cannot.
- Standard algorithms: PC, GES, NOTEARS for structure learning. Recent: deep-learning-based causal discovery (NeurIPS 2023+).
- Validate SCM by predicting the effect of held-out interventions.

**Effort:** 6–8 weeks for first credible version on a single domain class. Longer for cross-domain.

**Open questions:** how does the SCM compose across world models? When the router switches to a different world model, do we have one SCM per world model, or one shared SCM?

---

### 2.4 Hierarchical Planner — MISSING

Three-level planning, with every level checked against world-model rollouts.

**Top level:** LLM-as-planner. Receives natural-language goal, generates abstract plan (e.g., "find Hopf bifurcation in this Brusselator system by sweeping the b parameter, fitting the limit-cycle amplitude as a function of b, identifying the threshold by amplitude onset"). Output is a sequence of subgoals.

**Mid level:** option-critic. Decomposes each subgoal into a temporal sequence of options (sub-policies with termination conditions). Reuses skills from the skill library when available.

**Bottom level:** Monte Carlo Tree Search over world-model rollouts. For each option, expands action sequences, evaluates terminal value, backs up. Standard MCTS or AlphaZero-style with value network.

**Critical:** at every level, plans are validated by rolling out the world model. LLM proposes; world model disposes. If world-model rollout under proposed plan diverges sharply from LLM's predicted outcome, the plan is rejected.

**Outputs:** action sequence with associated value estimate and uncertainty.

**Effort:** 4–6 weeks for first version. Longer for hierarchical option discovery.

**Hard problem:** combinatorial explosion. MCTS depth is limited by world-model rollout cost. Resource-manager allocates compute per planning episode.

---

### 2.5 Reasoning Engine — PARTIAL

Three reasoning modes: forward, backward, symbolic.

**Forward reasoning:** "what happens if I do X?" — world-model rollouts. Implementation: wraps the world model under a query API.

**Backward reasoning (inverse planning):** "what action sequence leads to state Y?" — search through action space using world-model rollouts, gradient-based for differentiable models. Implementation: NEW.

**Symbolic reasoning:** "what equation describes this trajectory?" — SINDy, PySR, KAN-SR, ODEFormer. Already partially implemented; ADR-0001 upgrades to WSINDy + ensemble + KAN-SR.

**Critical role:** the symbolic channel makes the system's reasoning *extractable and auditable*. When the system claims "the system undergoes a Hopf bifurcation at b ≈ 2.0", that claim has an associated extracted equation that can be checked.

**Current code:** `src/simulating_anything/analysis/symbolic_regression.py` (PySR), `equation_discovery.py` (PySINDy). Forward reasoning: implicit in `world_model/`. Backward reasoning: not present.

---

### 2.6 Skill Library (Voyager-style) — PARTIAL

Acquired skills are stored, retrievable, composable.

**Skill structure:**
```
Skill {
    description: str           # Natural-language, embedded for retrieval
    precondition: callable     # State predicate; True if skill applicable
    action_sequence: callable  # state -> action_sequence
    expected_outcome: callable # State predicate over predicted next-state
    verification: callable     # Test that proves the skill works
    sub_skills: list[Skill]    # Composition: this skill calls these
    discovered_at: timestamp
    success_rate: float        # Updated by experience
    latent_signature: Tensor   # Encoded for similarity search
}
```

**Operations:**
- `discover(experience)` — propose new skill from successful trajectory
- `retrieve(description)` — find applicable skill by NL description (semantic search)
- `compose(goal)` — assemble skills hierarchically to achieve goal
- `verify(skill)` — re-run skill in simulator to confirm it still works
- `prune(skill)` — remove skills with low success rate or redundant with others

**Current code:** project skills exist as static `.md` files in `.claude/skills/`. These are *Claude Code skills*, not *agent skills* in the cognitive sense. The cognitive skill library is missing.

**Effort:** 4–6 weeks for first version. Longer for hierarchical composition.

---

### 2.7 Memory Hierarchy — PARTIAL

Four tiers, each with different write rules, retrieval mechanisms, decay.

**Working memory:** ~7 items, attention-based, current planning context. **MISSING.** Effort: 1 week.

**Episodic memory:** concrete past trajectories. Stored in full. Retrievable by similarity to current context. Used for case-based reasoning ("when I saw this pattern before, what worked?"). **PARTIAL** — `src/simulating_anything/knowledge/trajectory_store.py` stores trajectories but lacks retrieval-by-similarity API. Effort: 1 week.

**Semantic memory:** extracted facts, concepts, theories. The wiki. **EXISTS** as of this session. Operations: ingest, retrieve, lint. Will grow autonomously when concept formation is built. Effort: see Concept Formation.

**Procedural memory:** the skill library, plus motor-program-equivalent for primitive actions. **PARTIAL** — see Skill Library.

---

### 2.8 Reflection / Self-Model — MISSING

The system maintains explicit beliefs about itself: capabilities, knowledge gaps, historical failure patterns.

**Self-model state:**
```
SelfModel {
    capabilities: dict[task_type, success_rate]
    knowledge_gaps: list[Topic]
    failure_patterns: list[FailureMode]
    uncertainty_calibration: CalibrationCurve
    skills_inventory: list[skill_id]
    domains_familiar: list[domain]
    last_updated: timestamp
}
```

**Reflection loop:** periodically (every N campaigns or on demand), the system queries itself:

- "Where have my predictions been wrong recently?" → updates `failure_patterns`
- "What domains has my world model not been tested on?" → updates `knowledge_gaps`
- "Which skills have low success rate?" → schedules skill refinement
- "Is my uncertainty calibration accurate?" → updates `uncertainty_calibration`

The output of reflection feeds the curiosity module (what to practice) and the planner (when to abstain or verify more).

**Hard problem:** self-model accuracy is itself uncertain. The system cannot know what it doesn't know in full generality. Best we can do: track historical performance, calibrate via held-out test sets, abstain when uncertainty is high.

**Effort:** 4–6 weeks for first credible version.

---

### 2.9 Curiosity / Intrinsic Motivation — PARTIAL

Drives exploration when no extrinsic reward is available.

**Signals (combined):**
- **Empowerment** — measure of how many distinguishable next-states are reachable. High empowerment = informative action.
- **Surprise** — KL divergence between predicted and observed state. High surprise = interesting.
- **Learning progress** — rate of improvement on prediction tasks. High learning progress = productive direction.
- **Free energy** (Friston) — unifies the above. Minimize prediction error AND seek information gain.

**Output:** for any candidate experiment or action, an intrinsic-reward score. Integrated with extrinsic rewards (when present) by the planner.

**Current code:** `src/simulating_anything/exploration/uncertainty_driven.py` (MC-dropout-based). Implements one signal (epistemic uncertainty); needs extension to free-energy framework.

**ADR-0001's EIG-driven sampling** (Change #2) is a special case of this — drives parameter-sweep selection, currently. Will generalize to whole-system exploration.

---

### 2.10 Actuator with Monitoring — PARTIAL

Executes plans. Continuously compares predicted vs actual state. Aborts and replans when prediction error spikes.

**Loop:**
```
for each step in plan:
    predicted_state = world_model.predict(current_state, action)
    actual_state = environment.step(action)
    if KL(predicted, actual) > threshold:
        abort_plan()
        re-plan from actual_state
    else:
        current_state = actual_state
```

**Critical for self-correction during execution.** Without this, the agent commits to plans that go wrong without noticing.

**Current code:** `src/simulating_anything/campaign/manager.py` partially implements — runs sweeps, doesn't monitor prediction error online.

**Effort:** 2 weeks.

---

### 2.11 Concept Formation Engine — MISSING

Watches the experience stream and autonomously detects regularities, compressing them into named concepts that enter semantic memory.

**Algorithm sketch:**
- Cluster latent trajectories by similarity → candidate concepts
- For each cluster, fit a compact symbolic description (e.g., via PySR/KAN-SR over the cluster's latent dynamics)
- If symbolic description is *significantly more compressive* than raw data (MDL/Solomonoff criterion), commit the concept
- Auto-generate wiki page: name, definition, exemplars, related-concepts links
- LLM is called for: name suggestion, NL definition, link suggestions to existing wiki pages

**Why this matters:** without this, the wiki is human-curated only. With this, the system grows its own conceptual vocabulary from experience. This is what closes the "concept" loop in the cognitive architecture.

**Hard problem:** the right compression criterion. Naive MDL produces unhelpful concepts. Active research area.

**Effort:** 6–8 weeks for first useful version.

---

### 2.12 Language LLM (subordinate) — EXISTS

Strictly subordinate. Specific tasks only.

**Permitted uses:**
- Parse natural-language goals → formal task specifications
- Generate simulator code (current `agents/simulation_builder.py`)
- Structure reasoning traces (chain-of-thought prompting at the top of the planner)
- Retrieve from wiki (semantic search over wiki pages)
- Write reports / explanations (current `agents/communicator.py`)
- Suggest concept names and definitions (called by concept-formation engine)

**Forbidden uses:**
- Driving the main cognitive loop
- Acting as the world model
- Replacing the planner (chain-of-thought ≠ planning)
- Acting without world-model verification of outputs

**Implementation:** ClaudeCodeBackend in `src/simulating_anything/agents/base.py`. Already constrained appropriately.

---

### 2.13 Theory of Mind — MISSING (defer)

When interacting with humans or other agents: model their beliefs, intents, knowledge. Predict their actions.

**For solo physics-discovery work: low priority.** For multi-agent scientific collaboration or interaction with humans: essential.

Defer to Phase 4 or beyond.

---

### 2.14 Resource Manager — MISSING

The system has finite compute, memory, time budgets. The resource manager allocates them across cognitive subsystems.

**Decisions it makes:**
- How many MCTS rollouts per planning step?
- How deep to roll out world model?
- When to invoke LLM (expensive) vs pure-compute paths?
- How many ensemble members to use for uncertainty?
- When to prune skill library or memory?

**Without this:** system blows budget on easy problems and starves hard ones. With it: graceful degradation under constraint.

**Effort:** 2 weeks for first version.

---

### 2.15 Goal Stack — MISSING

Multiple horizons simultaneously.

**Stack levels:**
- **Long-term identity** — "be the cleanest open AMI implementation" — set by humans, persistent
- **Project goals** — "demonstrate skill transfer across 3 domain classes" — quarter-scale
- **Campaign goals** — "find the Hopf bifurcation in Brusselator" — hour-scale
- **Tactical goals** — "next action in current plan" — second-scale

Each level constrains levels below. Reflection updates priorities. Conflicts between levels resolved by explicit precedence.

**Effort:** 2 weeks.

---

### 2.16 Counterfactual Reasoning — MISSING

"What would have happened if I had chosen differently?" — runs world model under hypothetical past actions.

**Critical for credit assignment.** Standard temporal-difference learning estimates value of taken actions; counterfactual estimation extends to non-taken actions, dramatically improving sample efficiency.

**Implementation:** uses the same machinery as the causal layer applied to past episodes. Mostly subsumed by causal layer + episodic memory; small dedicated module to coordinate.

**Effort:** overlaps with causal layer; ~1 week additional.

---

### 2.17 Active Inference / Free Energy — MISSING (math glue)

The Karl Friston framework: agent minimizes variational free energy, which decomposes into prediction error (perception accuracy) plus expected information gain (exploration value). Mathematically unifies perception, action, learning under one objective.

**Why include it:** without a unifying mathematical framework, the architecture is a kitchen sink. With it, every component has a principled objective. Curiosity, perception, planning, learning all reduce to free-energy minimization.

**Implementation:** more refactoring than new code. Reformulate existing objectives (encoder loss, world-model prediction loss, exploration reward, planner value) as components of a single free-energy expression.

**Effort:** 4 weeks of focused work.

---

### 2.18 Constitutional / Value Layer — PARTIAL

Hard constraints the system will not violate.

**For Simulating-Anything specifically:**
- "Report only verified discoveries"
- "No fabricated results"
- "Honest about limitations"
- "Human in the loop for any external action"
- "No claims that cannot be supported by extracted reasoning trace"

These are values, encoded as planner constraints and output filters.

**Current state:** implicit in `CLAUDE.md`. Need to be formalized as runtime constraints, not just developer guidance.

**Effort:** 1–2 weeks to formalize.

---

## 3. Data Flow

The system runs in a closed perceive-plan-act-learn loop:

```
1. ENVIRONMENT     -> obs, reward, status
2. PERCEPTION      -> obs -> (z_t, uncertainty)
3. WORLD MODEL ROUTER -> select model based on z_t
4. WORLD MODEL     -> z_t, available actions -> predicted next-states + epistemic uncertainty
5. CAUSAL LAYER    -> SCM-aware queries; counterfactual rollouts
6. CURIOSITY       -> intrinsic-reward signal over candidate actions
7. RESOURCE MANAGER-> allocates compute budget for this step
8. PLANNER         -> goal stack + world model + skills + curiosity -> action sequence
9. SKILL LIBRARY   -> if applicable skill matches current state, planner may delegate
10. ACTUATOR       -> executes action; monitors predicted vs actual
11. ENVIRONMENT    -> next obs (loop closes)
12. EPISODIC MEMORY-> stores trajectory
13. WORLD MODEL    -> updates from new experience
14. SKILL LIBRARY  -> if successful new pattern, propose new skill
15. CONCEPT FORMATION -> if regularity detected across episodes, propose new concept
16. SEMANTIC MEMORY-> updated with new concepts
17. REFLECTION     -> periodic introspection updates self-model
18. SELF-MODEL     -> drives next reflection cycle's priorities
```

The LLM is called as needed across stages 8 (planner top-level), 14 (skill naming), 15 (concept naming), and at the human-interface boundary (parsing goals at start, writing reports at end).

---

## 4. Hard Problems and Honest Limits

| Problem | Status | Our approach |
|---|---|---|
| **Frame problem** (knowing what's relevant) | Unsolved | Heuristic + attention + skill-precondition matching; accept domain-bounded relevance |
| **Symbol grounding** | Unsolved | Embodied (simulator experience) + linguistic (LLM scaffolding); hybrid grounding |
| **Catastrophic forgetting** | Partially solved | EWC, replay buffers, modular networks; per-domain world models reduce interference |
| **OOD generalization** | Open | Strong priors (skill library) + calibrated uncertainty + abstention |
| **Common sense** | Partially solved | LLM exposure + simulator experience; jointly grounded |
| **Goal specification (Goodhart)** | Open | Constitutional layer + multi-objective + human review |
| **Compute scaling for deep planning** | Practical limit | Hierarchical planning + skill caching + resource manager |
| **Concept formation from raw experience** | Active research | MDL/compression + LLM naming; slow incremental growth |
| **Self-model accuracy** | Hard | Calibrated uncertainty + reflection loops + held-out test sets |
| **Causal discovery from observation** | Theoretical limit | Active intervention via simulators (the substrate's killer feature) |
| **Skill library scaling** | Active research | Hierarchical composition + pruning + abstraction |
| **Sim-to-real transfer** | Hard, deferred | Phase 4+; not in current scope |

We will hit walls. We will document them honestly. None will block the broader project from being a credible reference implementation.

---

## 5. Mapping to Existing Code

| Component | Current code | New code needed |
|---|---|---|
| Perception | `world_model/encoder.py`, `advanced_encoders.py` | unified multimodal interface |
| World model | `world_model/rssm.py`, `rssm_v2.py`, `ensemble.py` | `lewm.py`, `pde_fm.py`, `router.py` |
| Causal layer | (none) | `causal/scm.py`, `causal/intervention.py` |
| Planner | (none) | `planning/hierarchical.py`, `planning/mcts.py`, `planning/option_critic.py` |
| Reasoning | `analysis/symbolic_regression.py`, `equation_discovery.py` | `reasoning/forward.py`, `reasoning/backward.py`; promote symbolic to first-class |
| Skill library | `agents/*.py` (LLM agents only) | `skills/library.py`, `skills/voyager_loop.py`, `skills/composition.py` |
| Memory: working | (none) | `memory/working.py` |
| Memory: episodic | `knowledge/trajectory_store.py` | `memory/episodic.py` (similarity retrieval) |
| Memory: semantic | `wiki/` | `memory/semantic.py` (programmatic interface to wiki) |
| Memory: procedural | (skill library) | covered above |
| Reflection | (none) | `reflection/self_model.py`, `reflection/loop.py` |
| Curiosity | `exploration/uncertainty_driven.py` | extend to free-energy |
| Actuator | `campaign/manager.py` | `actuator/monitor.py` (prediction-error tracking) |
| Concept formation | (none) | `concepts/formation.py`, `concepts/compression.py` |
| LLM (subordinate) | `agents/base.py` (ClaudeCodeBackend) | already correct |
| Theory of mind | (none) | deferred to Phase 4+ |
| Resource manager | (none) | `resource/manager.py` |
| Goal stack | (none) | `goals/stack.py` |
| Counterfactual | (none) | overlap with causal; small `counterfactual/` module |
| Active inference | (none) | refactoring under `objective/free_energy.py` |
| Constitutional | implicit in `CLAUDE.md` | `constitutional/constraints.py` |

For the proposed directory layout, see `REFACTOR-PROPOSAL.md`.

---

## 6. Phase Plan

| Phase | Components added |
|---|---|
| **Phase 1 (2 weeks)** | Existing system upgrades per ADR-0001 (WSINDy, EIG sampling, TWA, Well dataset, Robin committee, LeWM A/B). |
| **Phase 2 (4 weeks)** | Refactor directory; stub all 18 components; foundational docs (this one, IDEA.md, VISION.md). |
| **Phase 3 (4 months)** | Implement: planner, skill library, causal layer, reflection, concept formation, four-tier memory. Curriculum design. Integration experiments. |
| **Phase 4 (6 months)** | Active inference unification; ablation studies; paper. |

Phase 1 keeps current discovery work moving. Phases 2–4 build the cognitive layer on top.

---

## 7. References

Primary literature for each component:

- **World models**: Hafner et al, *Nature* 2025 (DreamerV3); Maes/LeCun et al, arXiv:2603.19312 (LeWM); McCabe et al, NeurIPS 2024 (MPP)
- **Causal layer**: Pearl, *Causality* (2009); Schölkopf et al, *Towards Causal Representation Learning* (2021)
- **Hierarchical planning**: Sutton/Precup/Singh, *Between MDPs and Semi-MDPs* (1999); MuZero/AlphaZero (Silver et al, 2018, 2020)
- **Skill library**: Wang et al, *Voyager*, arXiv:2305.16291 (2023)
- **Symbolic reasoning**: Cranmer et al, *PySR* (2023); Liu et al, *KAN*, ICLR 2025; Messenger & Bortz, *WSINDy* (2021)
- **Reflection / self-model**: Park et al, *Generative Agents* (2023); Anthropic agent SDK (2025)
- **Curiosity / free energy**: Friston, *Free Energy Principle* (2009-present); Pathak et al, *Curiosity-Driven Exploration* (2017)
- **Concept formation**: Schmidhuber, *Compression-based Concept Formation* (1990s-present); MDL literature
- **Active inference**: Friston et al, *Active Inference: A Process Theory* (2017); Buckley et al survey (2017)
- **Vision overall**: LeCun, *A Path Towards Autonomous Machine Intelligence* (2022); Sutton, *Era of Experience* (2025)

For project-specific paper ingestions, see `wiki/sources/`.
