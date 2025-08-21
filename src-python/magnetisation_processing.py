import numpy as np
import json
from matplotlib import pyplot as plt

def generate_plots(folder, filename, output_file):

    # Open and read data
    with open(folder + "/" + filename, "r") as f:
        loaded = json.load(f)

    # load data back to numpy arrays
    N = np.array(loaded["N"])
    N_exact_threshold = np.array(loaded["N_exact_threshold"])
    M = np.array(loaded["M"])
    T = np.array(loaded["T"])
    J = np.array(loaded["J"])
    h = np.array(loaded["h"])
    K = np.array(loaded["K"])
    x = np.array(loaded["x"])
    burn_in = np.array(loaded["burn_in"])
    energies = np.array(loaded["energies"])
    counts = np.array(loaded["counts"])
    number_trajectories = np.array(loaded["number_trajectories"])
    average_exact_magnetisation = np.array(loaded["average_exact_magnetisation"])
    average_exact_energy = np.array(loaded["average_exact_energy"])
    mean_average_uniform = np.array(loaded["mean_average_uniform"])
    mean_average_local = np.array(loaded["mean_average_local"])
    uniform_error = np.array(loaded["uniform_error"])
    local_error = np.array(loaded["local_error"])
    mean_average_parallel = np.array(loaded["mean_average_parallel"])
    mean_average_coupling = np.array(loaded["mean_average_coupling"])
    parallel_error = np.array(loaded["parallel_error"])
    parallel_coupling_error = np.array(loaded["parallel_coupling_error"])
    mean_energy_uniform = np.array(loaded["mean_energy_uniform"])
    mean_energy_local = np.array(loaded["mean_energy_local"])
    mean_energy_parallel = np.array(loaded["mean_energy_parallel"])
    mean_energy_coupling = np.array(loaded["mean_energy_coupling"])
    magnetisation_uniform = np.array(loaded["magnetisation_uniform"])
    magnetisation_local = np.array(loaded["magnetisation_local"])
    magnetisation_parallel = np.array(loaded["magnetisation_parallel"])
    magnetisation_parallel_coupling = np.array(loaded["magnetisation_parallel_coupling"])

    # plot average magnetisation
    transparency = 0.2

    plt.subplot(2, 2, 1)
    plt.plot(x, mean_average_uniform, label='Uniform', color='blue')
    plt.fill_between(x, mean_average_uniform - uniform_error, mean_average_uniform + uniform_error, color='blue', alpha=transparency)
    plt.plot(x, mean_average_local, label='Local', color='orange')
    plt.fill_between(x, mean_average_local - local_error, mean_average_local + local_error, color='orange', alpha=transparency)
    plt.plot(x, mean_average_parallel, label='Dual', color='red')
    plt.fill_between(x, mean_average_parallel - parallel_error, mean_average_parallel + parallel_error, color='red', alpha=transparency)
    plt.plot(x, mean_average_coupling, label='Dual MH', color='purple')
    plt.fill_between(x, mean_average_coupling - parallel_coupling_error, mean_average_coupling + parallel_coupling_error, color='purple', alpha=transparency)
    if N <= N_exact_threshold:
        plt.plot([x[0], x[len(x) - 1]], [average_exact_magnetisation, average_exact_magnetisation], label='Exact', color = 'black', linestyle = 'dashed')
    plt.xlabel('Step')
    plt.ylabel('Average m(s)')
    plt.title('Running average magnetisation after step')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.ylim(-0.75, 0.75)
    plt.xlim(burn_in, M)

    # plot error
    plt.subplot(2, 2, 2)
    plt.plot(x, uniform_error, label='Uniform', color='blue')
    plt.plot(x, local_error, label='Local', color='orange')
    plt.plot(x, parallel_error, label='Dual', color='red')
    plt.plot(x, parallel_coupling_error, label='Dual MH', color='purple')
    plt.xlabel('Step')
    plt.ylabel('Standard deviation')
    plt.title('Magnetisation Error')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.xlim(burn_in, M)
    plt.ylim(0, 0.5)

    #plot energy
    plt.subplot(2, 2, 3)
    plt.plot(x, mean_energy_uniform, label='Uniform', color='blue')
    plt.plot(x, mean_energy_local, label='Local', color='orange')
    plt.plot(x, mean_energy_parallel, label='Dual', color='red')
    plt.plot(x, mean_energy_coupling, label='Dual MH', color='purple')
    if N <= N_exact_threshold:
        plt.plot([x[0], x[len(x) - 1]], [average_exact_energy, average_exact_energy], label='Exact', color = 'black', linestyle = 'dashed')
    plt.xlabel('Step')
    plt.ylabel('Average E(s)')
    plt.ylim(average_exact_energy*1.1, 0)
    plt.title('Instantaneous average energy after step')
    plt.legend(loc='upper right')
    plt.grid(True)

    # plot magnetisation of individual trajectory
    plt.subplot(2, 2, 4)
    plt.plot(x, magnetisation_uniform, label='Uniform', color='blue')
    plt.plot(x, magnetisation_local, label='Local', color='orange')
    plt.plot(x, magnetisation_parallel, label='Dual', color='red')
    plt.plot(x, magnetisation_parallel_coupling, label='Dual MH', color='purple')
    if N <= N_exact_threshold:
        plt.plot([x[0], x[len(x) - 1]], [average_exact_magnetisation, average_exact_magnetisation], label='Exact', color = 'black', linestyle = 'dashed')
    plt.xlabel('Step')
    plt.ylabel('m(s)')
    plt.title('Magnetisation at each step of an individual trajectory')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.ylim(-1, 1)

    plt.savefig(output_file)
    plt.suptitle(f"Replica Exchange Simulations: N = {N}, K = {K} (Constant), {number_trajectories} Trajectories Sampled")
    plt.show()

if __name__ == '__main__':

    filename = "data_seed63N14a1_bad.json"
    folder = "meeting_plots"
    output_file = "magnetisation_fig.pdf"

    generate_plots(folder, filename, output_file)