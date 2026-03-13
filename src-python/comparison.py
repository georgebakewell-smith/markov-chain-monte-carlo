import magnetisation as mag
import mcmc
import numpy as np
import json
import time
import comparison_plotting as cp

def model_comparison(N, T, M, K, number_trajectories, number_glasses, model, folder="./", filename="data.json", save_data = "yes"):
    uniform_comparison = np.empty((number_glasses, M))
    local_comparison = np.empty((number_glasses, M))
    dual_comparison = np.empty((number_glasses, M))
    dual_coupling_comparison = np.empty((number_glasses, M))

    energy_uniform = np.empty((number_glasses, M))
    energy_local = np.empty((number_glasses, M))
    energy_parallel = np.empty((number_glasses, M))
    energy_coupling = np.empty((number_glasses, M))

    start = time.time()

    for i in range(number_glasses):
        print(f"Evaluating model {i + 1}/{number_glasses}")
        [J, h] = mcmc.generate_model(N, model, random=True, seed=None)
        mean_average_uniform, mean_average_local, mean_average_parallel, mean_average_coupling, average_exact_magnetisation, mean_energy_uniform, mean_energy_local, mean_energy_parallel, mean_energy_coupling, average_exact_energy = mag.method_comparison(J, h, N, T, M, K, number_trajectories, save_data="no")
        uniform_comparison[i] = np.abs(mean_average_uniform - average_exact_magnetisation)
        local_comparison[i] = np.abs(mean_average_local - average_exact_magnetisation)
        dual_comparison[i] = np.abs(mean_average_parallel - average_exact_magnetisation)
        dual_coupling_comparison[i] = np.abs(mean_average_coupling - average_exact_magnetisation)
        energy_uniform[i] = np.abs(mean_energy_uniform - average_exact_energy)
        energy_local[i] = np.abs(mean_energy_local - average_exact_energy)
        energy_parallel[i] = np.abs(mean_energy_parallel - average_exact_energy)
        energy_coupling[i] = np.abs(mean_energy_coupling - average_exact_energy)

    if save_data == "yes":
        data_combined = {
            "N": N,
            "M": M,
            "T": T,
            "K": K,
            "number_trajectories": number_trajectories,
            "number_glasses": number_glasses,
            "model": model,
            "uniform_comparison": uniform_comparison.tolist(),
            "local_comparison": local_comparison.tolist(),
            "dual_comparison": dual_comparison.tolist(),
            "dual_coupling_comparison": dual_coupling_comparison.tolist(),
            "energy_uniform": energy_uniform.tolist(),
            "energy_local": energy_local.tolist(),
            "energy_parallel": energy_parallel.tolist(),
            "energy_coupling": energy_coupling.tolist()
        }

        # Open and write data
        with open(folder + "/" + filename, "w") as f:
            json.dump(data_combined, f)

    end = time.time()
    print(f"Overall execution time : {end - start:.4f} seconds")
    return uniform_comparison, local_comparison, dual_comparison, dual_coupling_comparison, energy_uniform, energy_local, energy_parallel, energy_coupling

if __name__ == "__main__":
    N = 12
    T = 10
    M = 20000  # number of MCMC steps
    K = 1   # number of steps before swapping states
    number_trajectories = 100
    number_glasses = 100
    model = "glass"

    # Data files
    folder = "meeting_plots"
    filename = "N14compLocalT1.json"

    model_comparison(N, T, M, K, number_trajectories, number_glasses, model, folder, filename)
    cp.generate_plots(folder, filename)