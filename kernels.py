import numpy as np
from numba import njit, cuda
from config import SIGMA

# CUDA device function for kernel_w
@cuda.jit(device=True, fastmath=True)
def kernel_w_cuda(r, h):
    q = r / h
    if q >= 2.0:
        return 0.0

    if q <= 1.0:
        val = 1.0 - 1.5 * q ** 2 + 0.75 * q ** 3
    else:
        val = 0.25 * (2.0 - q) ** 3

    return val * SIGMA

# CUDA device function for kernel_grad_w
@cuda.jit(device=True, fastmath=True)
def kernel_grad_w_cuda(dx, dy, r, h):
    q = r / h
    if q >= 2.0 or r < 1e-10:
        return 0.0, 0.0

    if q <= 1.0:
        deriv = -3.0 * q + 2.25 * q ** 2
    else:
        deriv = -0.75 * (2.0 - q) ** 2

    factor = deriv * SIGMA / (h * r)
    return dx * factor, dy * factor

@njit(fastmath=True)
def kernel_w(r, h):
    q = r / h
    if q >= 2.0:
        return 0.0

    if q <= 1.0:
        val = 1.0 - 1.5 * q ** 2 + 0.75 * q ** 3
    else:
        val = 0.25 * (2.0 - q) ** 3

    return val * SIGMA

@njit(fastmath=True)
def kernel_grad_w(r_vec, r, h):
    q = r / h
    if q >= 2.0 or r < 1e-10:
        return np.array([0.0, 0.0])

    if q <= 1.0:
        deriv = -3.0 * q + 2.25 * q ** 2
    else:
        deriv = -0.75 * (2.0 - q) ** 2

    factor = deriv * SIGMA / (h * r)
    return r_vec * factor