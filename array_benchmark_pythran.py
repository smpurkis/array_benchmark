import numpy as np
# pythran export compute_benchmark(int, int)

def compute_benchmark(m, n):
    arr = np.zeros((m, n), dtype=np.int32)
    for i in range(m):
        for j in range(n):
            arr[i][j] = i * i + j * j
    return arr
