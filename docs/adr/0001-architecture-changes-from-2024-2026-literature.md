# ADR-0001: Architecture Changes Based on 2024-2026 Literature Survey

**Date:** 2026-04-29
**Status:** proposed

## Context

Simulating-Anything is at a decision point. The project has 261 hand-crafted physics domains, 277 trained RSSM world models, 316 discovery campaigns, 118 validated bifurcations. CLAUDE.md documents honest limitations:

- SINDy degrades on extrapolation (1.5–3× training range)
- Noise tolerance: viable at 0.1%, degrades >5%
- Brusselator Hopf detection: 9–28% error
- Dream-based discovery: not yet accurate enough
- 1285 template stress tests dilute the universality claim

A 6-stream parallel literature survey (April 2026) covering JEPA/world models, symbolic regression, multi-physics datasets, autoresearch loops, bifurcation detection, and discovery agents revealed concrete tooling and pattern improvements that address each documented limitation. This ADR proposes the architectural changes.

## Decision

Adopt six changes in priority order (sequenced over ~3 weeks):

### Tier 1 — Engineering wins (Week 1, low risk)

**1. Replace vanilla SINDy with Weak-form + Ensemble SINDy.**
Flip `pysindy` configuration in `src/simulating_anything/analysis/equation_discovery.py` to use `WeakPDELibrary` with `EnsembleOptimizer(STLSQ, n_models=200)` and `library_ensemble=True`. Zero migration cost; both ship in the existing `pysindy` dependency. Lifts noise tolerance from 0.1% to 10–20% (Messenger & Bortz, SIAM MMS 2021; Fasel et al, Proc. Roy. Soc. A 2022).

**2. EIG-driven adaptive parameter sampling.**
Modify `src/simulating_anything/campaign/manager.py` to replace `n_sweep_points=30` linspace with sequential Bayesian experimental design: pick next parameter point that maximizes expected reduction in posterior variance over the threshold location. Wrap in fixed-budget keep/discard pool (Karpathy autoresearch pattern). Reuses existing `analysis/dt_invariance.py` (validator) and `exploration/uncertainty_driven.py` (MC-dropout for EIG proxy). 5–10× sample efficiency improvement empirically.

**3. Time-Warp-Attend Hopf oracle.**
Add CNN-based Hopf classifier on top of existing gradient z-score detector in `src/simulating_anything/analysis/bifurcation_detector.py`. Pretrained TWA model (MIT-licensed, github.com/nitzanlab/time-warp-attend) reports 87% accuracy on Brusselator-class systems at σ=0.1 — directly addresses the 9–28% Brusselator Hopf error. Combined with Bury 2021 deep-EWS as second classifier, ensemble agreement filter targets <5% Hopf error.

### Tier 2 — External grounding (Week 2, low risk)

**4. Ingest The Well + PDEBench.**
New `src/simulating_anything/simulation/well_bridge.py` wraps Polymathic AI's HuggingFace-streamed HDF5 datasets as `SimulationEnvironment` subclasses. Start with Rayleigh-Bénard (~500 GB) and MHD (~1 TB) — most relevant to existing reaction-diffusion and Navier-Stokes work. Adopt Multiple Physics Pretraining (MPP) evaluation protocol for paper-grade comparison. This converts the universality argument from "we built domains that look similar" to "our pipeline rediscovers known physics on the same fields the foundation-model community trains on."

### Tier 3 — Architectural bets (Week 3, medium-high risk)

**5. Robin-pattern validation committee.**
Replace the linear 7-stage pipeline at the validation boundary with a specialist committee (Skeptic + Literature-grounding + Replication-planner) orchestrated as Anthropic's documented orchestrator-worker pattern. Keep `simulation_builder` single-agent (Anthropic's documented "wrong domain" for multi-agent). Each candidate bifurcation/scaling-law passes the committee before entering `validated_discoveries.jsonl`. Token cost rises ~10× at validation, but validation is a small fraction of total compute.

**6. LeWM as parallel A/B vs RSSM.**
Add `src/simulating_anything/world_model/lewm.py` wrapping the LeWorldModel architecture (Maes et al, arXiv:2603.19312). 192-dim latent, 15M params, fits the RTX 5090 envelope. Train on 5 specific domains where RSSM dream-MSE is highest. Probe LeWM latents for known governing variables (LeWM paper proves this works on Push-T: agent location MSE 0.052, block angle r=0.999). Run SINDy + PySR on LeWM latents and compare to RSSM-derived equations. Decision gate: if LeWM+SINDy outperforms on the 5 failure domains, write up as paper contribution; otherwise document as honest negative result.

## Consequences

### Positive

- **Documented limitations addressed by named methods**: noise cliff (WSINDy), Hopf error (TWA), template-domain weakness (The Well), polynomial extrapolation (KAN-SR for failure cases)
- **Paper-grade external grounding** via Polymathic AI's published baselines
- **Discovery-quality pattern** (Robin) replaces throughput-only pattern (Sakana v1)
- **Engineering changes (Tier 1) are pure upside** — no architectural risk, all reversible
- **First architectural decision of the project documented** — establishes ADR practice

### Negative

- **LeWM is a research bet** — no published evidence yet that JEPA latents support equation recovery via SINDy/PySR for chaotic dynamical systems
- **Robin committee adds ~10× token cost at validation** — must monitor budget
- **The Well ingestion is 3 days of glue code** — brittleness during initial integration
- **Six concurrent changes is large surface area** — sequence matters, must respect tier ordering

### Risks

- **LeWM's published probing is on robotics tasks** (Push-T, Reacher, OGBench-Cube), not on chaotic ODEs (Lorenz, Brusselator) — physical-quantity factorization may not transfer
- **TWA was trained on supercritical Hopf only** — subcritical and degenerate Hopf bifurcations less reliable
- **Multi-agent committee at validation may converge to consensus** rather than catch errors — must implement Skeptic agent reward correctly (rewarded for kills, not approvals)
- **External datasets gated by HuggingFace availability** — adds upstream dependency

## Alternatives Considered

| Alternative | Rejected because |
|---|---|
| **Switch wholesale from RSSM to LeWM** | RSSM has 277 working models, median dream MSE 0.07. Wholesale switch is research bet, not engineering upgrade. A/B is the correct path. |
| **Adopt V-JEPA 2 (1.2B params)** | Video-prior bias, designed for action understanding not equation recovery. Wrong tool. |
| **Integrate Genie 3** | Generative renderer, not analytical substrate. Closed weights. Wrong tool entirely. |
| **Sakana-style linear pipeline** for the agent layer | Documented 42% experiment failure rate (Beel et al 2025 evaluation). Counter-pattern. |
| **Multi-agent at simulation_builder boundary** | Anthropic explicitly documents this as wrong-domain (shared context, single-author task). |
| **Aurora / GraphCast for external grounding** | Atmospheric foundation models tangential to bifurcation/scaling-law claim. Defer. |
| **JHTDB / BLASTNet** | Token-gated access, 5–280 TB scale; overkill until reviewers ask. |
| **Stay with current SINDy + PySR alone** | Honest limitations (noise cliff, polynomial extrapolation) are not addressable without either WSINDy or KAN-SR. |

## Sources

Primary literature is cataloged in `wiki/sources/`:
- [LeWorldModel](../../wiki/sources/lewm-2026.md) — Maes/LeCun et al, March 2026
- [The Well](../../wiki/sources/the-well-2024.md) — Polymathic AI, NeurIPS 2024
- [Weak SINDy](../../wiki/sources/wsindy-2021.md) — Messenger & Bortz, 2021
- [Time-Warp-Attend](../../wiki/sources/time-warp-attend-2024.md) — Talmon et al, ICLR 2024
- [Robin](../../wiki/sources/robin-2025.md) — FutureHouse, May 2025

Additional papers (not yet ingested into wiki, cited in research):
- DreamerV3 (Hafner et al, Nature 2025)
- V-JEPA 2 (Assran et al, arXiv:2506.09985, June 2025)
- Multiple Physics Pretraining / MPP (McCabe et al, NeurIPS 2024)
- KAN / KAN-SR (Liu et al, ICLR 2025; arXiv:2509.10089)
- ODEFormer (Becker et al, ICLR 2024)
- Bury et al deep EWS (PNAS 2021)
- Sakana AI Scientist v1 / v2 (arXiv:2408.06292, 2504.08066)
- Anthropic multi-agent research system (June 2025)
- FunSearch (Romera-Paredes et al, Nature 2024)
- LLM-SRBench (OpenReview 2025)
