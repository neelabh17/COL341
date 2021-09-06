import time
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
def createRandomMatrix(size):
    return [[random.randint(0,99) for _ in range(size)] for _ in range(size)]
    # assuming number in 0-99 in the matrix

N = 13
x = [i for i in range(N+1)]
y_numpy = []

# 0-13
for size in tqdm(range(N+1)):

    # cpu calculations
    total_operations = 2*((2**size)**3) 

    mat1_numpy = np.random.rand(2**size, 2**size)
    mat2_numpy = np.random.rand(2**size, 2**size)
    
    numpy_time_start = time.time()

    numpy_out = np.dot(mat1_numpy, mat2_numpy)

    numpy_time = time.time()-numpy_time_start
    gig = 10**9


    numpy_gflops = total_operations/(gig*numpy_time)
    y_numpy.append(numpy_gflops)

print(y_numpy)
plt.ylabel("GFLOPS")
plt.xlabel("N")
plt.plot(x, y_numpy, label = "numpy")
plt.legend()
plt.savefig("compare_numpy.png")
