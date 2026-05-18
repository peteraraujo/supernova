import numpy as np

# Simulation Parameters
G_BASE = np.float32(0.05)
GAMMA = np.float32(5.0 / 3.0)
H = np.float32(0.15)
MASS = np.float32(2.0)
VISC_ALPHA = np.float32(1.0)
VISC_BETA = np.float32(2.0)
GRAV_SOFTENING = np.float32(0.05)
DAMP_FACTOR = np.float32(0.95)
MIN_TEMP = np.float32(0.01)

# Equation of State (Neutron Star)
NUCLEAR_DENSITY = np.float32(25.0)
DEGENERACY_COEFF = np.float32(100.0)
DEGENERACY_EXP = np.float32(3.0)

# Simulation Settings
N_PARTICLES = 1800
DT_MAX = 0.004
CFL_FACTOR = 0.2

# Grid Optimization
GRID_SIZE = 100
GRID_CELL = np.float32(2.0 * H)
GRID_ORIGIN = np.float32(-5.0)

# Supernova Phase Parameters
COLLAPSE_GRAVITY_MULT = np.float32(60.0)
COLLAPSE_COOL_RATE = np.float32(0.96)
COLLAPSE_MIN_DURATION = np.float32(0.05)
SUPERNOVA_INJECT_E = np.float32(10000.0)
REMNANT_COUNT = 250

# Derived Constant
SIGMA = np.float32(10.0 / (7.0 * np.pi * H * H))