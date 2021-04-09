# import numba, tensorflow and numpy, load cython
import functools
from time import time

import jax.numpy as jnp
import numba
import numpy as np
import pyximport
import tensorflow as tf
import torch
from jax import jit

pyximport.install()
from compute_utils import *
import array_benchmark_pythran


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.experimental.set_visible_devices([], 'GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)


def timeit(n=10):
    """
    Decorator to run function n times and print out the total time elapsed.
    """
    def dec(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            t0 = time()
            for i in range(n):
                # func(*args, **kwargs)
                print(func(*args, **kwargs)[-1, -1])
            print("%s iterated %d times\nTime elapsed %.3fs\n" % (
                func.__name__, n, time() - t0))

        return wrapped
    return dec


@tf.function
def compute_tf(m, n):
    x1 = tf.range(0, m - 1, 1) ** 2
    x2 = tf.range(0, n - 1, 1) ** 2
    return x1[:, None] + x2[None, :]


@tf.function
def compute_tf_my(m, n):
    x = tf.reshape(tf.range(m), (-1, 1)) ** 2 + tf.range(n) ** 2
    return x


def compute_torch(m, n):
    x1 = torch.arange(0, m - 1, 1) ** 2
    x2 = torch.arange(0, n - 1, 1) ** 2
    return x1[:, None] + x2[None, :]


def compute_torch_my(m, n):
    x = torch.reshape(torch.arange(1, m - 1), (-1, 1)) ** 2 + torch.arange(n - 1) ** 2
    return x


# @numba.jit(forceobj=True)
def compute_torch_my_int32(m, n):
    j = torch.arange(m, dtype=torch.int32, device="cpu")
    k = torch.arange(n, dtype=torch.int32, device="cpu")
    x = torch.reshape(j, (-1, 1)) ** 2 + k ** 2
    return x


def compute_torch_my_int32_tensor_input(m, n):
    x = torch.reshape(m, (-1, 1)) ** 2 + n ** 2
    return x


compute_tf(tf.constant(10), tf.constant(10))  # trace once
compute_tf_my(tf.constant(10), tf.constant(10))  # trace once
compute_torch(torch.tensor([10]).item(), torch.tensor([10]).item())
compute_torch_my(torch.tensor([10]).item(), torch.tensor([10]).item())


def compute_numpy(m, n):
    x1 = np.linspace(0., m - 1, m) ** 2
    x2 = np.linspace(0., n - 1, n) ** 2
    return x1[:, None] + x2[None, :]


def compute_numpy_range(m, n):
    x = np.power(np.arange(m).reshape(-1, 1), 2) + np.power(np.arange(n), 2)
    return x


def compute_numpy_range_int32(m, n):
    x = np.power(np.arange(m, dtype=np.int32).reshape(-1, 1), 2) + np.power(np.arange(n, dtype=np.int32), 2)
    return x


@numba.njit
def compute_numpy_range_numba(m, n):
    x = np.power(np.arange(m).reshape(-1, 1), 2) + np.power(np.arange(n), 2)
    return x


@numba.njit(fastmath=True)
def compute_numpy_range_numba_fastmath(m, n):
    x = np.power(np.arange(m).reshape(-1, 1), 2) + np.power(np.arange(n), 2)
    return x


@numba.njit
def compute_numba(m, n):
    x = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            x[i, j] = i * i + j * j
    return x


@numba.njit(fastmath=True)
def compute_numba_fastmath(m, n):
    x = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            x[i, j] = i * i + j * j
    return x


@numba.njit(parallel=True, nogil=True)
def compute_numba_parallel_two(m, n):
    x = np.empty((m, n))
    for i in numba.prange(m):
        for j in numba.prange(n):
            x[i, j] = i * i + j * j
    return x


@numba.njit(parallel=True, nogil=True, fastmath=True)
def compute_numba_parallel_fastmath(m, n):
    x = np.empty((m, n))
    for i in numba.prange(m):
        for j in numba.prange(n):
            x[i, j] = i * i + j * j
    return x


def compute_jax_range(m, n):
    x = jnp.power(jnp.arange(m).reshape(-1, 1), 2) + jnp.power(jnp.arange(n), 2)
    return x


def compute_jax_range_arange_input(m, n):
    x = jnp.power(m.reshape(-1, 1), 2) + jnp.power(n, 2)
    return x


compute_numba_parallel_two(10, 10)

compute_numba(10, 10)
compute_numpy_range_numba(10, 10)
compute_numba_fastmath(10, 10)
compute_numba_parallel_fastmath(10, 10)
compute_numpy_range_numba_fastmath(10, 10)

# compute_torch_my_int32_tensor_input_jit = torch.jit.trace(compute_torch_my_int32_tensor_input,
#                                                           (torch.arange(10, dtype=torch.int32),
#                                                            torch.arange(10, dtype=torch.int32)))
m2, n2 = jnp.arange(10), jnp.arange(10)
t = jit(compute_jax_range_arange_input)(m2, n2)
m = 10000
n = 10000
n_loop = 5

# from time import time
# t = 10000
# s = time()
# print(compute_cython_memview_two_c_contigious(t, t)[t-1, t-1])
# print(time() - s)


# timeit(n=n_loop)(compute_numpy)(m, n)
# timeit(n=n_loop)(compute_numpy_range)(m, n)
timeit(n=n_loop)(compute_numpy_range_int32)(m, n)
# t0 = time()
# for i in range(n_loop):
#     # print(func(*args, **kwargs)[-1, -1])
#     m2, n2 = jnp.arange(m), jnp.arange(n)
#     t = jit(compute_jax_range_arange_input)(m2, n2)
# print(f"compute_torch_my_int32_tensor_input_jit {time() - t0}")
# timeit(n=n_loop)(compute_jax_range_arange_input)(jnp.arange(m), jnp.arange(n))
# timeit(n=n_loop)(compute_jax_range)(m, n)

# timeit(n=n_loop)(compute_numpy_range_numba)(m, n)
# timeit(n=n_loop)(compute_numba)(m, n)
timeit(n=n_loop)(compute_numba_parallel_two)(m, n)
# timeit(n=n_loop)(compute_numpy_range_numba_fastmath)(m, n)
# timeit(n=n_loop)(compute_numba_fastmath)(m, n)
# timeit(n=n_loop)(compute_numba_parallel_fastmath)(m, n)
# timeit(n=n_loop)(compute_cython_numpy)(m, n)
# timeit(n=n_loop)(compute_numpy_range_cython)(m, n)
# timeit(n=n_loop)(compute_cython_numpy_no_prange)(m, n)
# timeit(n=n_loop)(compute_cython_numpy_memview)(m, n)
# timeit(n=n_loop)(compute_cython_memview_no_prange)(m, n)
# timeit(n=n_loop)(compute_cython_memview_two)(m, n)
timeit(n=n_loop)(compute_cython_memview_two_c_contigious)(m, n)
# timeit(n=n_loop)(array_benchmark_pythran.compute_benchmark)(m, n)
# timeit(n=n_loop)(compute_cython_memview_two_Py_ssize_t)(m, n)
# timeit(n=n_loop)(compute_cython_memview_one)(m, n)
# timeit(n=n_loop)(compute_cython_memview_no_checks)(m, n)
timeit(n=n_loop)(compute_tf)(tf.constant(m), tf.constant(n))
# timeit(n=n_loop)(compute_tf_my)(tf.constant(m), tf.constant(n))
# timeit(n=n_loop)(compute_torch)(torch.tensor([m]).item(), torch.tensor([n]).item())
# timeit(n=n_loop)(compute_torch_my)(torch.tensor([m]).item(), torch.tensor([n]).item())
timeit(n=n_loop)(compute_torch_my_int32)(torch.tensor([m]).item(), torch.tensor([n]).item())
# timeit(n=n_loop)(compute_torch_my_int32_tensor_input)(torch.arange(m, dtype=torch.int32),
#                                                       torch.arange(n, dtype=torch.int32))
# t0 = time()
# for i in range(n_loop):
#     # print(func(*args, **kwargs)[-1, -1])
#     compute_torch_my_int32_tensor_input_jit(torch.arange(m, dtype=torch.int32), torch.arange(n, dtype=torch.int32))
# print(f"compute_torch_my_int32_tensor_input_jit {time() - t0}")
