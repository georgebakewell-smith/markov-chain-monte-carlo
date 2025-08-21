import magnetisation as mag
import mcmc

folder = "./"
N = 16
[J, h] = mcmc.generate_model(N, "glass", random=True, seed=12)  # seed = 12 N=10, is good for spin glass, seed=12 N=14 good for nn
T = 0.1
M = 10000  # number of MCMC steps
number_trajectories = 1000 # number of trajectories to be sampled

Ks = [5]

for K in Ks:
    print(f"Evaluating K = {K}")
    #filename = "dataK" + str(K) + "N14.json"
    filename = "data.json"
    mag.method_comparison(J, h, N, T, M, K, number_trajectories, folder, filename)