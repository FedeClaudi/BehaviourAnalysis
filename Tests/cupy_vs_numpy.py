import numpy as np
import cupy as cp
import timeit

cp.cuda.Device(0).use()

def numpy_func():
    arr = np.full((1000, 1000), np.random.randint(10)).astype(np.int32)
    mean = np.mean(arr)

def cupy_func():
    arr = cp.full((1000, 1000), cp.random.randint(10)).astype(cp.int32)
    mean = cp.mean(arr)


if __name__ == "__main__":
    n_reps = 100

    nres = timeit.Timer(numpy_func).timeit(number=n_reps)/n_reps * 1000
    print("Numpy took {} ms".format(round(nres, 2)))


    cres = timeit.Timer(cupy_func).timeit(number=n_reps)/n_reps * 1000
    print("Cupy took {} ms".format(round(cres, 2)))

    print("The speed gain was: ", round(nres/cres, 2))


