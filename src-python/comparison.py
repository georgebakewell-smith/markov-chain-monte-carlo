import magnetisation as mag
import mcmc
import numpy as np
import matplotlib.pyplot as plt
import time

N = 14
T = 0.1
M = 20000  # number of MCMC steps
K = 1    # number of steps before swapping states
number_trajectories = 100
number_glasses = 100
model = "glass"
uniform_comparison = np.empty((number_glasses, M))
local_comparison = np.empty((number_glasses, M))
dual_comparison = np.empty((number_glasses, M))
dual_coupling_comparison = np.empty((number_glasses, M))

start = time.time()

for i in range(number_glasses):
    print(f"Evaluating model {i + 1}/{number_glasses}")
    [J, h] = mcmc.generate_model(N, model, random=True, seed=None)
    mean_average_uniform, mean_average_local, mean_average_parallel, mean_average_coupling, average_exact_magnetisation = mag.method_comparison(J, h, N, T, M, K, number_trajectories, save_data="no")
    uniform_comparison[i] = np.abs(mean_average_uniform - average_exact_magnetisation)
    local_comparison[i] = np.abs(mean_average_local - average_exact_magnetisation)
    dual_comparison[i] = np.abs(mean_average_parallel - average_exact_magnetisation)
    dual_coupling_comparison[i] = np.abs(mean_average_coupling - average_exact_magnetisation)

end = time.time()
print(f"Overall execution time : {end - start:.4f} seconds")

# Plotting the results
x = np.arange(M)
plt.figure()
plt.plot(x, np.mean(uniform_comparison, axis=0), label='Uniform', color='blue')
plt.plot(x, np.mean(local_comparison, axis=0), label='Local', color='orange')
plt.plot(x, np.mean(dual_comparison, axis=0), label='Dual', color='red')
plt.plot(x, np.mean(dual_coupling_comparison, axis=0), label='Dual MH', color='purple')

plt.xlabel('Step')
plt.ylabel('Average magnetisation difference')
plt.legend(loc='upper right')
plt.grid(True)
plt.xlim(0, M)
plt.ylim(0, 0.3)
plt.title(f"Replica Exchange Simulations: N = {N}, K = {K} (Constant), {number_glasses} Glasses Sampled")  
plt.show()