Architecture
============

7-Stage Pipeline
----------------

.. code-block:: text

   Problem Architect (LLM) -> Domain Classifier (rules + LLM fallback)
     -> Simulation Builder (LLM) -> Ground-Truth Simulation (JAX)
     -> Exploration (uncertainty-driven) -> Analysis (PySR + SINDy + ablation)
     -> Communication Agent (LLM) -> Markdown Report

Simulation Layer
~~~~~~~~~~~~~~~~

All simulations inherit from ``SimulationEnvironment`` (``simulation/base.py``):

- ``reset(seed) -> np.ndarray`` -- initialize state
- ``step() -> np.ndarray`` -- advance one timestep
- ``observe() -> np.ndarray`` -- current observable
- ``run(n_steps) -> TrajectoryData`` -- collect trajectory

World Model
~~~~~~~~~~~

**RSSMv1** (``world_model/rssm.py``): 512 GRU deterministic + 32x32
categorical stochastic = 1536 latent dimensions. Supports CNN (spatial)
and MLP (vector) encoders/decoders.

**RSSMv2** (``world_model/rssm_v2.py``): DreamerV4-style enhancements:

- Deeper prior/posterior (2-layer MLP with LayerNorm)
- Mixed stochastic: categorical + continuous Gaussian
- Continue predictor for episode boundaries

**Advanced Encoders** (``world_model/advanced_encoders.py``):

- ``GraphEncoder``: Message-passing GNN for graph-structured data
- ``CNN3DEncoder``: 3D convolutions for volumetric data

Analysis
~~~~~~~~

- **PySR**: Evolutionary symbolic regression via Julia
- **SINDy**: Sparse identification of nonlinear dynamics (PySINDy)
- **Ablation**: Single-factor and pipeline component ablation studies
- **Cross-domain**: Analogy detection across 187 domains

Autonomous Discovery
~~~~~~~~~~~~~~~~~~~~

The campaign manager orchestrates fully autonomous research:

1. ``ResearchPlannerAgent`` decomposes questions into experiments
2. ``SimulationGeneratorAgent`` builds simulations from descriptions
3. ``SimulationValidator`` validates generated code
4. ``HypothesisTester`` validates discovered equations
5. ``CampaignManager`` chains experiments with replanning

Composable Dynamics
~~~~~~~~~~~~~~~~~~~

Build simulations from reusable modules (``simulation/composable.py``):

- Force modules: ``HarmonicForce``, ``CubicForce``, ``PendulumForce``
- Damping: ``LinearDamping``, ``VanDerPolDamping``
- Integration: ``NewtonianDynamics``
- Population: ``LogisticGrowth``, ``PredatorPreyInteraction``
- Epidemic: ``SIRDynamics``
- Chemical: ``BrusselatorKinetics``

Sim-to-Real Transfer
~~~~~~~~~~~~~~~~~~~~

The ``TransferValidator`` (``verification/transfer_validation.py``)
validates discoveries against real experimental data:

- Statistical tests: KS, correlation, Spearman
- Accuracy metrics: R-squared, RMSE, MAPE
- Composite transfer score with confidence levels
