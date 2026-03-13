import numpy as np
from numba import njit, prange

@njit(parallel = True)
def parallel_simulations(N, T, burn_in, M, K, method, number_trajectories, energy_lookup, magnetisation_lookup, coupling = 0):
    """
            Runs parallel simulations of MCMC trajectories for range of methods
    """

    average_magnetisation = np.empty((number_trajectories, M))
    energy = np.empty((number_trajectories, M))
    magnetisation = np.empty((number_trajectories, M))

    for i in prange(0,number_trajectories):
        av_mag, en, mag = trajectory_simulate(N, T, burn_in, M, K, method, energy_lookup, magnetisation_lookup, coupling)
        average_magnetisation[i,:] = av_mag
        energy[i,:] = en
        magnetisation[i,:] = mag

    return average_magnetisation, energy, magnetisation

@njit
def trajectory_simulate(N, T, burn_in, M, K, method, energy_lookup, magnetisation_lookup, coupling = 0):
    """
            Performs one trajectory for a range of MCMC methods and returns average magnetisation and energy
    """

    # choose initial state uniformly
    state = uniform_proposal(N)

    # compute magnetisation and energy of initial state
    magnetisation = np.empty(M)
    energy = np.empty(M)
    running_average = np.zeros(M)

    energy[0]= energy_lookup[state]
    magnetisation[0] = magnetisation_lookup[state]

    if method == "parallel uniform":
        method_primary, method_auxilliary = "uniform", "uniform"
    elif method == "parallel local":
        method_primary, method_auxilliary = "local", "local"
    elif method == "parallel alternate":
        method_primary, method_auxilliary = "uniform", "local"

    state_auxilliary = state
    counter = 0
    swap_counter = 0
    #energy_parallel_local = np.empty(M)
    #energy_parallel_local[0] = energy_lookup[state_auxilliary]

    for i in range(1, M):
        
        # Obtain proposals
        state_proposal, state_proposal_auxilliary = get_proposals(method_primary, method_auxilliary, N, state, state_auxilliary)
        state, eng = MCMC_step(N, T, state, state_proposal, energy_lookup[state], energy_lookup[state_proposal])
        state_auxilliary, eng_parallel = MCMC_step(N, T, state_auxilliary, state_proposal_auxilliary, energy_lookup[state_auxilliary], energy_lookup[state_proposal_auxilliary])
        
        mag = magnetisation_lookup[state]
        # running average burn in
        if i > burn_in:
            running_average[i] = ((running_average[i-1]*(i - burn_in) + mag)/(i - burn_in + 1))
            energy[i] = ((energy[i-1]*(i - burn_in) + eng)/(i - burn_in + 1))
        elif i == burn_in:
            running_average[i] = mag
            energy[i] = eng
        counter +=1

        if counter == K:
            if coupling == 0:
                tmp = state
                state = state_auxilliary
                state_auxilliary = tmp
            elif coupling == 1:
                # Swap states based on acceptance probability
                delta_E = energy_lookup[state_auxilliary] - energy_lookup[state]
                if np.random.rand() < acceptance_probability(delta_E, T):
                    tmp = state
                    state = state_auxilliary
                    state_auxilliary = tmp
                    swap_counter += 1
            elif coupling == 2:
                delta_E = energy_lookup[state_auxilliary] - energy_lookup[state]
                Z = 1 + np.exp(-abs(delta_E)/T)
                if np.random.rand() < 1/Z:
                    if delta_E <= 0:
                        state = state_auxilliary
                    elif delta_E >= 0:
                        state_auxilliary = state

            elif coupling == 3:
                delta_E = energy_lookup[state_auxilliary] - energy_lookup[state]
                Z = 1 + np.exp(-abs(delta_E)/T)
                old_state = state
                if np.random.rand() < 1 - 1/Z:
                    state = state_auxilliary

                if np.random.rand() < 1 - 1/Z:
                    state_auxilliary = old_state
            elif coupling == 4:
                delta_E = energy_lookup[state_auxilliary] - energy_lookup[state]
                if delta_E < 0:
                    tmp = state
                    state = state_auxilliary
                    state_auxilliary = tmp
                    swap_counter += 1

            elif coupling == 5:
                delta_E = energy_lookup[state_auxilliary] - energy_lookup[state]
                if np.random.rand() < 1 - np.exp(-abs(delta_E/T)):
                    tmp = state
                    state = state_auxilliary
                    state_auxilliary = tmp
                    swap_counter +=1
                

            elif coupling == 6:
                delta_E = energy_lookup[state_auxilliary] - energy_lookup[state] # Ey - Ex
                a = np.exp(-abs(delta_E)/T)
                rng = np.random.rand()
                if rng < a:
                    state_auxilliary = state
                else:
                    state = state_auxilliary
                

            counter = 0
        magnetisation[i] = mag

    return running_average, energy, magnetisation
@njit(inline = "always")
def indicator(x):
    if x==0:
        return 0
    elif x>0:
        return 1
    else:
        return 0


def statistics(array_2D):
    """
        Returns the mean and standard deviation of a 2D array along axis=0
    """
    
    return np.mean(array_2D, axis=0), np.std(array_2D, axis=0)

def generate_model(N, model, random=True, seed = None):
    """
        Generates an Ising model either randomly or from a custom input
    """
        
    J = np.zeros((N, N))
    h = np.zeros(N)

    if seed != None:
            np.random.seed(seed)

    if model == "nn":
        if random==True:
            for i in range(0,N-1):
                J[i][i+1] = np.random.normal()
                J[i+1][i] = J[i][i+1]
                h[i] = np.random.normal()
            h[N-1] = np.random.normal()
            
        else:
            J1D = [-0.99121054, 0.84436089, -0.83043895, 0.95766024, -1.02814718, 1.24969204, 0.81649925, -0.92147578, 1.1139441]
            h1D = [0.47921821, 0.10207621, -0.4780673, -0.39407213, 0.15239487, 0.44938277,  0.91715616, 0.73303354, 0.40145444, 0.55915183]
            for i in range(0,N-1):
                J[i][i+1] = J1D[i]
                #J[i+1][i] = J1D[i]
                h[i] = h1D[i]
            h[N-1] = h1D[N-1]
        
    elif model == "glass":
        for i in range(0, N):
            for j in range(i, N):
                if i != j:
                    J[i][j] = np.random.normal()
                    J[j][i] = J[i][j]
                
            h[i] = np.random.normal()

    np.random.seed(seed=None) # reset seed if used
    
    return [J, h]

@njit(inline = "always")
def get_proposals(method_primary, method_auxilliary, N, state, state_parallel):
    """
        Returns the proposal states for primary and auxilliary methods
    """

    if method_primary == "uniform":
        state_proposal = uniform_proposal(N)
    elif method_primary == "local":
        state_proposal = local_proposal(N, state)

    if method_auxilliary == "uniform":
        state_proposal_auxilliary = uniform_proposal(N)
    elif method_auxilliary == "local":
        state_proposal_auxilliary = local_proposal(N, state_parallel)

    return state_proposal, state_proposal_auxilliary

@njit(inline = "always")
def uniform_proposal(N):
    """
    Returns an integer uniformly from range 0, 2^{-N}
    """

    return np.random.randint(2**N)

@njit(inline = "always")
def local_proposal(N, state):
    """
    Randomly flips one spin and returns decimal representation of new state
    """

    index = np.random.randint(0, N)

    return state ^ (1 << index)

@njit(inline = "always")
def compute_magnetisation(N, state):
    """
        Computes the magnetisation of a state
    """

    total = 0
    for i in range(N):
        bit_index = N - 1 - i
        spin = ((state >> bit_index) & 1) * 2 - 1
        total += spin
    return total / N

@njit(inline="always")
def compute_energy(J, h, N, state):
    """
        Returns energy of input state
    """

    energy = 0.0
    for i in range(N):
        s_i = ((state >> (N - 1 - i)) & 1) * 2 - 1
        for j in range(N):
            s_j = ((state >> (N - 1 - j)) & 1) * 2 - 1
            energy += J[i, j] * s_i * s_j
        energy += h[i] * s_i
    return -energy

@njit(inline = "always")
def acceptance_probability(delta_E, T):
    """
        Computes the acceptance probability for moving to proposed state
    """

    return min(1, np.exp(-delta_E/T))

@njit(inline = "always")
def MCMC_step(N, T, state, proposal, energy_state, energy_proposal):
    """
        Performs one step of MCMC with chosen method, returning the next state
    """

    delta_E = energy_proposal - energy_state

    # decides whether to accept proposal
    if np.random.rand() < acceptance_probability(delta_E, T):
        new_state = proposal
        energy_state = energy_proposal
    else:
        new_state = state

    return new_state, energy_state

@njit
def compute_values_all(J, h, N):

    energies = np.empty(2**N)
    magnetisations = np.empty(2**N)

    for i in range(2**N):
        magnetisations[i] = compute_magnetisation(N, i)
        energies[i] = compute_energy(J, h, N, i)

    return energies, magnetisations

def thermodynamic_averages(J, h, N, T, n):
    """
        Computes the thermodynamic averages exactly and the lowest n energy eigenvalues and their degeneracies
    """

    boltzmann_weight = np.empty(2**N)
    magnetisations = np.empty(2**N)
    unsorted_energies = np.empty(2**N)

    # compute all energies and magnetisations
    unsorted_energies, magnetisations = compute_values_all(J, h, N)
    
    energies, counts = np.unique(unsorted_energies, return_counts=True)
    E_min = energies[0]
    shifted_energies = unsorted_energies - E_min
    boltzmann_weight = np.exp(-shifted_energies/T)
    norm = np.sum(boltzmann_weight)
    average_exact_energy = np.dot(boltzmann_weight, unsorted_energies)/norm
    average_exact_magnetisation = np.dot(boltzmann_weight, magnetisations)/norm

    print("Exact thermodynamic averages complete")
    print(f"Average magnetisation = {average_exact_magnetisation}")
    print(f"Average energy = {average_exact_energy}")
    print("Diagonalisation complete")
    print(f"Lowest {n} energies are : ", energies[:n])
    print("with degeneracies : ", counts[:n])

    return average_exact_magnetisation, average_exact_energy , energies[:n], counts[:n], unsorted_energies, magnetisations