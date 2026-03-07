Simulation Domains
==================

Simulating Anything includes 192 simulation domains spanning physics,
biology, chemistry, epidemiology, neuroscience, and more.

Core Domains (14 rediscoveries)
-------------------------------

These domains have full PySR/SINDy rediscovery results:

.. list-table::
   :header-rows: 1
   :widths: 25 35 15

   * - Domain
     - Target Equation
     - R-squared
   * - Projectile
     - R = v^2 sin(2theta) / g
     - 0.9999
   * - Lotka-Volterra
     - Equilibrium (gamma/delta, alpha/beta)
     - 1.0
   * - Gray-Scott
     - Turing instability boundary
     - 0.985
   * - SIR Epidemic
     - R0 = beta/gamma
     - 1.0
   * - Double Pendulum
     - T = 2*pi*sqrt(L/g)
     - 0.999993
   * - Harmonic Oscillator
     - omega_0 = sqrt(k/m)
     - 1.0
   * - Lorenz Attractor
     - SINDy ODE recovery
     - 0.99999
   * - Navier-Stokes 2D
     - decay_rate = 4*nu
     - 1.0
   * - Van der Pol
     - Period scaling T(mu)
     - 0.99996
   * - Kuramoto
     - Sync transition r(K)
     - 0.9695
   * - Brusselator
     - b_c = 1 + a^2
     - 0.9960
   * - FitzHugh-Nagumo
     - f-I curve, SINDy ODE
     - 0.99999999
   * - Heat Equation 1D
     - Decay rate = D*k^2
     - 1.0
   * - Logistic Map
     - Feigenbaum, Lyapunov(r=4)=ln(2)
     - --

Extended Domains
----------------

177+ additional domains covering:

**Chaotic ODEs**: Lorenz, Rossler, Chua, Chen, Aizawa, Halvorsen,
Burke-Shaw, Sprott (A-S), Thomas, Dadras, Genesio-Tesi, Lu-Chen,
Shimizu-Morioka, Newton-Leipnik, Wang, Arneodo, Rucklidge, Rabinovich-Fabrikant,
Duffing, Duffing-VdP, Ueda, Lorenz-84, Lorenz-96, Lorenz-Stenflo,
Lorenz-Haken, Sakarya, Qi, WINDMI, Finance, Tigan, Liu, Hadley, Vallis

**Neuroscience**: Hodgkin-Huxley, FitzHugh-Nagumo (ODE, spatial, lattice, ring),
FitzHugh-Rinzel, Hindmarsh-Rose, Morris-Lecar, Izhikevich, Rulkov Map,
Wilson-Cowan, Cable Equation, Theta Neuron, Jansen-Rit, Amari Neural Field

**Ecology**: Lotka-Volterra (2/3/4-species), Rosenzweig-MacArthur,
Competitive LV, Predator-Prey-Mutualist, Predator-Two-Prey, Allee,
Bazykin, Beddington-DeAngelis, Ivlev, Prey Refuge, Predator-Prey-Toxin,
Predator-Prey-Parasite, Seasonal, Harvested Population

**Epidemiology**: SIR, SEIR, SEIRS, SIRD, SIRS, SIS, Network SIS,
SIR-Vaccination, SIR-Stochastic, SIR-Metapopulation, Zombie-SIR, Eco-Epidemic

**PDEs**: Navier-Stokes, Turbulent Flow, Heat Equation, Damped Wave, Burgers,
KdV, Kuramoto-Sivashinsky, Ginzburg-Landau, Cahn-Hilliard, Sine-Gordon,
Swift-Hohenberg, Shallow Water, Gray-Scott (1D/2D), Brusselator-Diffusion,
Schnakenberg, FHN-Spatial, Oregonator-1D, BZ-Spiral, Diffusive-LV, Advection

**Statistical Mechanics**: Ising Model, Boltzmann Gas, Lennard-Jones MD,
Bak-Sneppen SOC, Vicsek Flocking

**Chemistry**: Brusselator, Oregonator, Selkov, Autocatalator, Glycolytic Oscillator

**Lattice/Soliton**: Toda Lattice, FPUT, Spring-Mass Chain, Sine-Gordon

**Quantum**: Quantum Harmonic Oscillator

**Relativity**: Schwarzschild Geodesics

**Discrete Maps**: Logistic, Henon, Ikeda, Standard, Tent, Lozi, Cubic,
Ricker, Tinkerbell, Rulkov, Bouncing Ball, Coupled Map Lattice, Gauss, Circle

**Protein Folding**: HP Lattice Model

**Robotics**: CartPole-Brax (control-oriented)
