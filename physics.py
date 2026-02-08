import numpy as np
from numba import njit, prange
import config as cfg
from kernels import kernel_w, kernel_grad_w

@njit(fastmath=True)
def build_grid(pos, head, next_particle):
    head.fill(-1)
    next_particle.fill(-1)
    n = len(pos)

    for i in range(n):
        cx = int((pos[i, 0] - cfg.GRID_ORIGIN) / cfg.GRID_CELL)
        cy = int((pos[i, 1] - cfg.GRID_ORIGIN) / cfg.GRID_CELL)

        if 0 <= cx < cfg.GRID_SIZE and 0 <= cy < cfg.GRID_SIZE:
            cell_idx = cx + cy * cfg.GRID_SIZE
            next_particle[i] = head[cell_idx]
            head[cell_idx] = i

@njit(parallel=True, fastmath=True)
def compute_gravity(pos, mass_per_particle, G):
    n = len(pos)
    acc_grav = np.zeros((n, 2))
    soft2 = cfg.GRAV_SOFTENING ** 2

    for i in prange(n):
        ax = 0.0
        ay = 0.0
        for j in range(n):
            if i == j:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            r2 = dx * dx + dy * dy
            inv_dist3 = 1.0 / ((r2 + soft2) ** 1.5)
            ax += dx * inv_dist3
            ay += dy * inv_dist3

        acc_grav[i, 0] = G * mass_per_particle * ax
        acc_grav[i, 1] = G * mass_per_particle * ay

    return acc_grav

@njit(parallel=True, fastmath=True)
def compute_density_pressure(pos, u, mass_per_particle, head, next_particle):
    n = len(pos)
    rho = np.zeros(n)
    pressure = np.zeros(n)
    h2_support = (2.0 * cfg.H) ** 2

    for i in prange(n):
        cx_i = int((pos[i, 0] - cfg.GRID_ORIGIN) / cfg.GRID_CELL)
        cy_i = int((pos[i, 1] - cfg.GRID_ORIGIN) / cfg.GRID_CELL)
        rho_val = 0.0

        for off_x in range(-1, 2):
            for off_y in range(-1, 2):
                nx = cx_i + off_x
                ny = cy_i + off_y

                if 0 <= nx < cfg.GRID_SIZE and 0 <= ny < cfg.GRID_SIZE:
                    cell_idx = nx + ny * cfg.GRID_SIZE
                    j = head[cell_idx]
                    while j != -1:
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        r2 = dx * dx + dy * dy
                        if r2 < h2_support:
                            rho_val += mass_per_particle * kernel_w(np.sqrt(r2), cfg.H)
                        j = next_particle[j]

        if rho_val < 1e-5:
            rho_val = 1e-5
        rho[i] = rho_val

        p_ideal = (cfg.GAMMA - 1.0) * rho_val * u[i]
        p_degen = 0.0
        if rho_val > cfg.NUCLEAR_DENSITY:
            excess = (rho_val - cfg.NUCLEAR_DENSITY)
            p_degen = cfg.DEGENERACY_COEFF * (excess ** cfg.DEGENERACY_EXP)

        pressure[i] = p_ideal + p_degen

    return rho, pressure

@njit(parallel=True, fastmath=True)
def compute_sph_forces(pos, vel, rho, pressure, mass_per_particle, head, next_particle):
    n = len(pos)
    acc_sph = np.zeros((n, 2))
    du_dt = np.zeros(n)
    h2_support = (2.0 * cfg.H) ** 2

    for i in prange(n):
        cx_i = int((pos[i, 0] - cfg.GRID_ORIGIN) / cfg.GRID_CELL)
        cy_i = int((pos[i, 1] - cfg.GRID_ORIGIN) / cfg.GRID_CELL)

        for off_x in range(-1, 2):
            for off_y in range(-1, 2):
                nx = cx_i + off_x
                ny = cy_i + off_y

                if 0 <= nx < cfg.GRID_SIZE and 0 <= ny < cfg.GRID_SIZE:
                    cell_idx = nx + ny * cfg.GRID_SIZE
                    j = head[cell_idx]
                    while j != -1:
                        if i == j:
                            j = next_particle[j]
                            continue

                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        r2 = dx * dx + dy * dy

                        if 1e-12 < r2 < h2_support:
                            r = np.sqrt(r2)
                            r_vec = np.array([dx, dy])
                            grad_w = kernel_grad_w(r_vec, r, cfg.H)

                            v_ij = vel[i] - vel[j]
                            v_dot_r = np.dot(v_ij, r_vec)

                            visc_term = 0.0
                            if v_dot_r < 0:
                                mu = cfg.H * v_dot_r / (r2 + 0.01 * cfg.H ** 2)
                                c_i = np.sqrt(cfg.GAMMA * pressure[i] / rho[i])
                                c_j = np.sqrt(cfg.GAMMA * pressure[j] / rho[j])
                                c_sound = 0.5 * (c_i + c_j)
                                rho_bar = 0.5 * (rho[i] + rho[j])
                                visc_term = (-cfg.VISC_ALPHA * c_sound * mu + cfg.VISC_BETA * mu ** 2) / rho_bar

                            p_term = (pressure[i] / (rho[i] ** 2) + pressure[j] / (rho[j] ** 2))
                            total_term = p_term + visc_term
                            force_vec = -mass_per_particle * total_term * grad_w

                            acc_sph[i, 0] += force_vec[0]
                            acc_sph[i, 1] += force_vec[1]
                            du_dt[i] += 0.5 * mass_per_particle * total_term * np.dot(v_ij, grad_w)

                        j = next_particle[j]
    return acc_sph, du_dt

@njit(fastmath=True)
def get_dt(rho, pressure, vel):
    c_s = np.sqrt(cfg.GAMMA * pressure / rho)
    max_signal = np.max(c_s)
    max_vel = np.max(np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2))

    if max_signal + max_vel < 1e-5:
        return cfg.DT_MAX

    dt = cfg.CFL_FACTOR * cfg.H / (max_signal + max_vel)
    return min(dt, cfg.DT_MAX)