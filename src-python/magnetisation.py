import numpy as np
import mcmc
import json
import time
import magnetisation_processing as mp

def method_comparison(J, h, N, T, M, K, number_trajectories, folder="./", filename="data.json", save_data = "yes"):

    # Define the parameters
    burn_in = 0 # M//10 # number of steps before running average is computed
    N_exact_threshold = 24 # maximum N to compute partition function exactly
    number_eigenvalues = 5 # number of energy eigenvalues to be displayed

    # Compute exact values
    if N <= N_exact_threshold:
        average_exact_magnetisation, average_exact_energy, energies, counts, energy_lookup, magnetisation_lookup = mcmc.thermodynamic_averages(J, h, N, T, number_eigenvalues)
    else:
        average_exact_magnetisation = np.nan
        average_exact_energy = np.nan
        energies = np.nan
        counts = np.nan

    start = time.time()

    print("About to start simulations...")

    # Run simulations
    average_magnetisation_uniform, energy_uniform, magnetisation_uniform = mcmc.parallel_simulations(N, T, burn_in, M, K, "parallel uniform", number_trajectories, energy_lookup, magnetisation_lookup, -1)
    print("Uniform method complete.")
    average_magnetisation_local, energy_local, magnetisation_local = mcmc.parallel_simulations(N, T, burn_in, M, K, "parallel uniform", number_trajectories, energy_lookup, magnetisation_lookup, 6)
    print("Local method complete.")
    average_parallel, energy_parallel, magnetisation_parallel = mcmc.parallel_simulations(N, T, burn_in, M, K, "parallel local", number_trajectories, energy_lookup, magnetisation_lookup, 6)
    print("Dual method complete.")
    average_parallel_coupling, energy_parallel_coupling, magnetisation_parallel_coupling = mcmc.parallel_simulations(N, T, burn_in, M, K, "parallel alternate", number_trajectories, energy_lookup, magnetisation_lookup, 6)
    print("Dual MH method complete.")
        
    x = range(len(average_magnetisation_uniform[0]))
    
    # Compute mean and error
    mean_average_uniform, uniform_error = mcmc.statistics(average_magnetisation_uniform)
    mean_average_local, local_error = mcmc.statistics(average_magnetisation_local)
    mean_average_parallel, parallel_error = mcmc.statistics(average_parallel)
    mean_average_coupling, parallel_coupling_error = mcmc.statistics(average_parallel_coupling)
    
    mean_energy_uniform = np.mean(energy_uniform, axis=0)
    mean_energy_local = np.mean(energy_local, axis=0) 
    mean_energy_parallel = np.mean(energy_parallel, axis=0)
    mean_energy_coupling = np.mean(energy_parallel_coupling, axis=0)

    if save_data == "yes":
        data_combined = {
            "N": N,
            "N_exact_threshold": N_exact_threshold,
            "M": M,
            "T": T,
            "J": J.tolist(),
            "h": h.tolist(),
            "K": K,
            "x": list(x),
            "burn_in": burn_in,
            "energies": energies.tolist(),
            "counts": counts.tolist(),
            "number_trajectories": number_trajectories,
            "average_exact_magnetisation": average_exact_magnetisation,
            "average_exact_energy": average_exact_energy,
            "mean_average_uniform": mean_average_uniform.tolist(),
            "mean_average_local": mean_average_local.tolist(),
            "uniform_error": uniform_error.tolist(),
            "local_error": local_error.tolist(),
            "mean_average_parallel": mean_average_parallel.tolist(),
            "mean_average_coupling": mean_average_coupling.tolist(),
            "parallel_error": parallel_error.tolist(),
            "parallel_coupling_error": parallel_coupling_error.tolist(),
            "mean_energy_uniform": mean_energy_uniform.tolist(),
            "mean_energy_local": mean_energy_local.tolist(),
            "mean_energy_parallel": mean_energy_parallel.tolist(),
            "mean_energy_coupling": mean_energy_coupling.tolist(),
            "magnetisation_uniform": magnetisation_uniform[0].tolist(),
            "magnetisation_local": magnetisation_local[0].tolist(),
            "magnetisation_parallel": magnetisation_parallel[0].tolist(),
            "magnetisation_parallel_coupling": magnetisation_parallel_coupling[0].tolist()
        }

        # Open and write data
        with open(folder + "/" + filename, "w") as f:
            json.dump(data_combined, f)

    end = time.time()
    print(f"Execution time : {end - start:.4f} seconds")

    return mean_average_uniform, mean_average_local, mean_average_parallel, mean_average_coupling, average_exact_magnetisation, mean_energy_uniform, mean_energy_local, mean_energy_parallel, mean_energy_coupling, average_exact_energy

if __name__ == '__main__':
    # Example usage
    N = 14
    [J, h] = mcmc.generate_model(N, "glass", random=True, seed=None)  # seed = 12 N=10, is good for spin glass, seed=12 N=14 good for nn
    T = 10
    M = 100000  # number of MCMC steps
    K = 1   # number of steps before swapping states
    number_trajectories = 1000  # number of trajectories to be sampled
    
    # Data files
    folder = "meeting_plots"
    filename = "data.json"
    output_file = "magnetisation_fig.pdf"

    method_comparison(J, h, N, T, M, K, number_trajectories, folder, filename)
    mp.generate_plots(folder, filename, output_file)