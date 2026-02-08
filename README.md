# Supernova Simulation

## Description

This project simulates the core collapse and subsequent explosion of a star using Smoothed Particle Hydrodynamics (SPH).
It models gas dynamics, self-gravity, and thermal pressure to visualize the lifecycle of a supernova event.

The simulation progresses through three distinct phases:

1. **Stable Equilibrium:** A cloud of gas particles maintains hydrostatic equilibrium through self-gravity and pressure.
2. **Core Collapse:** Gravity is artificially increased to induce a rapid implosion, simulating the loss of fusion
   pressure.
3. **Explosion:** Upon reaching nuclear density, a repulsive force and a massive injection of energy drive a shockwave
   outward, leaving behind a dense remnant neutron star.

## Tech Used

* **Python:** Primary programming language.
* **NumPy:** Vectorized array operations for physics calculations.
* **Matplotlib:** Real-time visualization and animation.
* **Numba:** Just-in-Time (JIT) compilation to accelerate calculation loops (gravity, density, pressure).

## Installation

1. Install Python 3.8 or higher.
2. Install the required dependencies:
   ```bash
   pip install numpy matplotlib numba
   ```

## Usage

Run the simulation script directly:

```bash
python simulation.py
```

## Controls
* **SPACE:** Initiate Core Collapse
* **[ :** Decrease simulation speed (steps per frame)
* **] :** Increase simulation speed (steps per frame)
* **R :** Reset the simulation
* **Q :** Quit

## Physics Model
* Hydrodynamics: Uses standard SPH formulations for density, pressure, and viscosity.
* Gravity: N-body gravity calculation with softening to prevent singularities.
* Equation of State: Combines ideal gas law with a degeneracy pressure term that activates at high densities.
* Integration: Symplectic integration for time-stepping.