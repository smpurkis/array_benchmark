# cython: infer_types=True
# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
cimport cython
from cython import nogil
from cython.parallel import prange
from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_cython_numpy(int m, int n):
    cdef long [:, :] x = np.empty((m, n), dtype=int)
    cdef int i, j
    for i in prange(m, nogil=True):
        for j in prange(n):
            x[i, j] = i*i + j*j
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_cython_numpy_no_prange(int m, int n):
    cdef long [:, :] x = np.empty((m, n), dtype=int)
    cdef int i, j
    for i in range(m):
        for j in range(n):
            x[i, j] = i*i + j*j
    return x



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_cython_numpy_memview(int m, int n):
    cdef long [:, :] x = np.empty((m, n), dtype=int)
    cdef long [:, :] x_view = x
    cdef int i, j
    for i in prange(m, nogil=True):
        for j in prange(n):
            x_view[i, j] = i*i + j*j
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_cython_memview_no_prange(int m, int n):
    cdef x = cvarray(shape=(m, n), itemsize=sizeof(int), format="i")
    cdef int [:, :] x_view = x
    cdef int i, j
    for i in range(m):
        for j in range(n):
            x_view[i, j] = i*i + j*j
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cvarray compute_cython_memview_two(int m, int n):
    cdef cvarray x = cvarray(shape=(m, n), itemsize=sizeof(int), format="i")
    cdef int [:, :] x_view = x
    cdef int i, j
    for i in prange(m, nogil=True):
        for j in prange(n):
            x_view[i, j] = i*i + j*j
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cvarray compute_cython_memview_two_c_contigious(int m, int n):
    cdef cvarray x = cvarray(shape=(m, n), itemsize=sizeof(int), format="i")
    cdef int [:, ::1] x_view = x
    cdef int i, j
    for i in prange(m, nogil=True):
        for j in prange(n):
            x_view[i, j] = i*i + j*j
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cvarray compute_cython_memview_two_Py_ssize_t(Py_ssize_t m, Py_ssize_t n):
    cdef cvarray x = cvarray(shape=(m, n), itemsize=sizeof(int), format="i")
    cdef int [:, :] x_view = x
    cdef int i, j
    for i in prange(m, nogil=True):
        for j in prange(n):
            x_view[i, j] = i*i + j*j
    return x




@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cvarray compute_cython_memview_one(int m, int n):
    cdef cvarray x = cvarray(shape=(m, n), itemsize=sizeof(int), format="i")
    cdef int [:, :] x_view = x
    cdef int i, j
    for i in prange(m, nogil=True):
        for j in range(n):
            x_view[i, j] = i*i + j*j
    return x

cpdef cvarray compute_cython_memview_no_checks(int m, int n):
    cdef cvarray x = cvarray(shape=(m, n), itemsize=sizeof(int), format="i")
    cdef int [:, :] x_view = x
    cdef int i, j
    for i in prange(m, nogil=True):
        for j in prange(n):
            x_view[i, j] = i*i + j*j
    return x



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray compute_numpy_range_cython(int m, int n):
    cdef np.ndarray x = np.power(np.arange(m).reshape(-1, 1), 2) + np.power(np.arange(n), 2)
    return x