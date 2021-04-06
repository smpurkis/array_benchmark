import functools
from time import time
import numpy as np


def timeit(n=10):
    """
    Decorator to run function n times and print out the total time elapsed.
    """

    def dec(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            t0 = time()
            for i in range(n):
                print(func(*args, **kwargs)[-1, -1])
            print("%s iterated %d times\nTime elapsed %.3fs\n" % (
                func.__name__, n, time() - t0))

        return wrapped

    return dec


def compute_normal(m, n):
    x = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            x[i, j] = i * i + j * j
    return x


def compute_numpy_range_int32(m, n):
    x = np.power(np.arange(m, dtype=np.int32).reshape(-1, 1), 2) + np.power(np.arange(n, dtype=np.int32), 2)
    return x


def compute_numpy_range_int64(m, n):
    x = np.power(np.arange(m, dtype=np.int64).reshape(-1, 1), 2) + np.power(np.arange(n, dtype=np.int64), 2)
    return x


m = 15000
n = 15000
n_loop = 5
compute_numpy_range_int32(10, 10)
compute_numpy_range_int64(10, 10)


# timeit(n=n_loop)(compute_normal)(m, n)
timeit(n=n_loop)(compute_numpy_range_int32)(m, n)
timeit(n=n_loop)(compute_numpy_range_int64)(m, n)

