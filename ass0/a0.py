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
y_cpu = []
y_numpy = []

# 0-13
for size in tqdm(range(N+1)):

    # cpu calculations
    mat1_cpu = createRandomMatrix(2**size)
    mat2_cpu = createRandomMatrix(2**size)
    cpu_out = [[0 for _ in range(2**size)] for _ in range(2**size)]
    cpu_time_start = time.time()
    for i in range(2**size):
        for j in range(2**size):
            for k in range(2**size):
                # 2 operations
                cpu_out[j][i] += mat1_cpu[j][k]*mat2_cpu[k][i]
    cpu_time = time.time()-cpu_time_start
    total_operations = 2*((2**size)**3) 
    gig = 10**9

    cpu_gflops = total_operations/(gig*cpu_time)
    y_cpu.append(cpu_gflops)

    # numpy calculations
    mat1_numpy = np.array(mat1_cpu)
    mat2_numpy = np.array(mat2_cpu)
    
    numpy_time_start = time.time()

    numpy_out = np.dot(mat1_numpy, mat2_numpy)

    numpy_time = time.time()-numpy_time_start

    numpy_gflops = total_operations/(gig*numpy_time)
    y_numpy.append(numpy_gflops)

print(y_cpu)
print(y_numpy)
plt.ylabel("GFLOPS")
plt.xlabel("N")
plt.plot(x,y_cpu, label = f"loop based [max:{max(y_cpu)}]")
plt.plot(x, y_numpy, label = "numpy")
plt.legend()
plt.savefig("compare.png")
