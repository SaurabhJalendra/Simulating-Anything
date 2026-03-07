Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install -e ".[dev]"

For GPU support (required for world model training):

.. code-block:: bash

   # Inside WSL2 Ubuntu
   pip install "jax[cuda12]" equinox optax diffrax

Basic Usage
-----------

Run a discovery pipeline:

.. code-block:: python

   from simulating_anything import Pipeline

   pipeline = Pipeline()
   report = pipeline.run("How do patterns form in reaction-diffusion systems?")

Autonomous Discovery
--------------------

Ask any scientific question and let the system investigate autonomously:

.. code-block:: python

   from simulating_anything import Pipeline

   pipeline = Pipeline()
   campaign = pipeline.discover("How do sand dunes form?", max_steps=20)

Command Line Interface
----------------------

.. code-block:: bash

   # Run the full pipeline
   simulating-anything run "How do pendulums work?"

   # Autonomous discovery campaign
   simulating-anything discover "What causes traffic jams?"

   # List available domains
   simulating-anything domains

   # Run rediscovery experiments
   simulating-anything rediscover projectile

Composable Simulations
----------------------

Build custom simulations by composing reusable dynamics modules:

.. code-block:: python

   from simulating_anything.simulation.composable import (
       HarmonicForce, LinearDamping, NewtonianDynamics,
       ComposedSimulation,
   )

   sim = ComposedSimulation.from_modules(
       modules=[
           HarmonicForce(var="x", accel_var="a_x", param_k="k"),
           LinearDamping(vel_var="v", accel_var="a_x", param_c="c"),
           NewtonianDynamics(pos_var="x", vel_var="v", accel_var="a_x", param_m="m"),
       ],
       defaults={"k": 4.0, "c": 0.2, "m": 1.0, "x_0": 1.0, "v_0": 0.0},
       accel_vars={"a_x"},
   )

   sim.reset()
   for _ in range(1000):
       state = sim.step()
