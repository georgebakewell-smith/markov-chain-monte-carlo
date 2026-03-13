import numpy as np
import json
from matplotlib import pyplot as plt

def generate_plots(folder, filename):

    # Open and read data
    with open(folder + "/" + filename, "r") as f:
        loaded = json.load(f)

    # load data back to numpy arrays
    N = np.array(loaded["N"])
    M = np.array(loaded["M"])
    T = loaded["T"]
    K = loaded["K"]
    number_trajectories = loaded["number_trajectories"]
    number_glasses = loaded["number_glasses"]
    model = loaded["model"]
    uniform_comparison = np.array(loaded["uniform_comparison"])
    local_comparison = np.array(loaded["local_comparison"])
    dual_comparison = np.array(loaded["dual_comparison"])
    dual_coupling_comparison = np.array(loaded["dual_coupling_comparison"])
    energy_uniform = np.array(loaded["energy_uniform"])
    energy_local = np.array(loaded["energy_local"])
    energy_parallel = np.array(loaded["energy_parallel"])
    energy_coupling = np.array(loaded["energy_coupling"])

    # Plotting the results
    line1 = 'Uniform No Swap'
    line2 = 'Uniform Dumb Swap'
    line3 = 'Alternate Smart Swap'
    line4 = 'Local No Swap'
    x = np.arange(M)
    plt.subplot(2,1,1)
    plt.plot(x, np.mean(uniform_comparison, axis=0), label=line1, color='blue')
    plt.plot(x, np.mean(local_comparison, axis=0), label=line2, color='orange')
    plt.plot(x, np.mean(dual_comparison, axis=0), label=line3, color='red')
    plt.plot(x, np.mean(dual_coupling_comparison, axis=0), label=line4, color='purple')
    plt.xlabel('Step')
    plt.ylabel('Average magnetisation difference')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xlim(0, M)
    plt.ylim(0, 0.5)
    

    plt.subplot(2,1,2)
    plt.plot(x, np.mean(energy_uniform, axis=0), label=line1, color='blue')
    plt.plot(x, np.mean(energy_local, axis=0), label=line2, color='orange')
    plt.plot(x, np.mean(energy_parallel, axis=0), label=line3, color='red')
    plt.plot(x, np.mean(energy_coupling, axis=0), label=line4, color='purple')
    plt.xlabel('Step')
    plt.ylabel('Average energy difference')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xlim(0, M)
    plt.suptitle(f"Replica Exchange Simulations: N = {N}, K = {K} (Constant), T = {T}, {number_glasses} Glasses Sampled")
    plt.ylim(0, 70)
    plt.show()

if __name__ == '__main__':

    filename = "N14compLocalT1.json"
    folder = "meeting_plots"
    generate_plots(folder, filename)