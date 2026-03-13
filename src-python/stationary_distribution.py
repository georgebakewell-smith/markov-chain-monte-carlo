import numpy as np
import mcmc
from numba import njit

def indicator(x):
    if x==0:
        return 0
    elif x>0:
        return 1
    else:
        return 0

def acceptance_probability(delta_E, T):
    """
        Computes the acceptance probability for moving to proposed state
    """

    return min(1, np.exp(-delta_E/T))

def generate_pair(n):

    x = np.floor(n/10)

    return int(x), int(n - 10*x)

@njit(inline = "always")
def accept_1(x, y, delta_E, T, coupling):
    if coupling == 0:
        return 0
    Iy = mcmc.indicator(delta_E)
    Ix = mcmc.indicator(-delta_E)
    Z = np.exp(-abs(delta_E)/T) + Iy + Ix
    if coupling == 6:
        return mcmc.indicator(delta_E)/Z
    if coupling == 5:
        return 0
    if coupling == 7:
        return 1
    if coupling == 8:
        return np.exp(-abs(delta_E)/T)

@njit(inline = "always")
def accept_2(x, y, delta_E, T, coupling):
    if coupling == 0:
        return 0
    Iy = mcmc.indicator(delta_E)
    Ix = mcmc.indicator(-delta_E)
    Z = np.exp(-abs(delta_E)/T) + Iy + Ix
    if coupling == 6:
        return mcmc.indicator(-delta_E)/Z
    if coupling == 5:
        return 0
    if coupling == 7:
        return 0
    if coupling == 8:
        return 1 - np.exp(-abs(delta_E)/T)

@njit(inline = "always")
def accept_x(x, y, delta_E, T, coupling):
    if coupling == 0:
        return 1
    #Iy = mcmc.indicator(delta_E)
    #Ix = mcmc.indicator(-delta_E)
    #Z = np.exp(-abs(delta_E)/T) + Iy + Ix
    if coupling == 6:
        return 0
    if coupling == 5:
        return 1 - np.exp(-abs(delta_E)/T)
    if coupling == 7:
        return 0
    if coupling == 8:
        return 0
    
@njit(inline = "always")    
def accept_s(x, y, delta_E, T, coupling):
    if coupling == 0:
        return 0
    Iy = mcmc.indicator(delta_E)
    Ix = mcmc.indicator(-delta_E)
    Z = np.exp(-abs(delta_E)/T) + Iy + Ix
    if coupling == 6:
        return np.exp(-abs(delta_E)/T)/Z
    if coupling == 5:
        return np.exp(-abs(delta_E)/T)
    if coupling == 7:
        return 0
    if coupling == 8:
        return 0
    
def thermodynamic_averages(J, h, N, T, n):
    """
        Computes the thermodynamic averages exactly and the lowest n energy eigenvalues and their degeneracies
    """

    boltzmann_weight = np.empty(2**N)
    magnetisations = np.empty(2**N)
    unsorted_energies = np.empty(2**N)

    # compute all energies and magnetisations
    unsorted_energies, magnetisations = mcmc.compute_values_all(J, h, N)
    
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

    return unsorted_energies, boltzmann_weight

N = 6
T = 1
n = 1
m = 0
coupling = 8
num_states = 2**N
number_eigenvalues = 5
[J, h] = mcmc.generate_model(N, "glass", random=True, seed=16)
energy_lookup, boltzmann_weight = thermodynamic_averages(J, h, N, T, number_eigenvalues)
stationary_dist_exact = boltzmann_weight/np.sum(boltzmann_weight)

# Create transition matrices for individual proposals
proposal1 = np.ones((num_states,num_states))/N
proposal2 = np.ones((num_states,num_states))/N
accept_prob = np.empty((num_states,num_states))

# Populate acceptance matrix
for i in range(num_states):
    for j in range(num_states):
        delta_E = energy_lookup[j] - energy_lookup[i]
        accept_prob[i][j] = acceptance_probability(delta_E, T)


method1 = proposal1*accept_prob
method2 = proposal2*accept_prob
# Populate diagonals with rejection probability
for i in range(num_states):
    method1[i][i] = method1[i][i] + 1 - np.sum(method1[i])
    method2[i][i] = method2[i][i] + 1 - np.sum(method2[i])

# Right eigenvectors of method1.T  <=> Left eigenvectors of method1
eigenvalues, eigenvectors = np.linalg.eig(method1.T)

# Find index of eigenvalue closest to 1
idx = np.argmin(np.abs(eigenvalues - 1))

leading_eigenvalue = eigenvalues[idx]
leading_left_eigenvector = eigenvectors[:, idx]

# Normalize so it sums to 1 (stationary distribution)
stationary_dist = leading_left_eigenvector / np.sum(leading_left_eigenvector)

coupling_matrix = np.zeros((num_states**2, num_states**2))
for x_old in range(num_states):
    for y_old in range(num_states):
        i = num_states*x_old + y_old
        j1 = num_states*x_old + x_old
        j2 = num_states*y_old + y_old
        jx = num_states*y_old + x_old
        js = i
        delta_E = energy_lookup[y_old] - energy_lookup[x_old]

        coupling_matrix[i][j1] = coupling_matrix[i][j1] + accept_1(x_old, y_old, delta_E, T, coupling)
        coupling_matrix[i][j2] = coupling_matrix[i][j2] + accept_2(x_old, y_old, delta_E, T, coupling)
        coupling_matrix[i][jx] = coupling_matrix[i][jx] + accept_x(x_old, y_old, delta_E, T, coupling)
        coupling_matrix[i][js] = coupling_matrix[i][js] + accept_s(x_old, y_old, delta_E, T, coupling)
stationary_dist2 = np.kron(stationary_dist_exact, stationary_dist_exact)

dual_method = np.kron(np.linalg.matrix_power(method1, n), np.linalg.matrix_power(method2, n))
transition_matrix = coupling_matrix @ dual_method

# Find stationary state of dual transition matrix

# Right eigenvectors of method1.T  <=> Left eigenvectors of method1
eigenvalues_dual, eigenvectors_dual = np.linalg.eig(transition_matrix.T)

# Find index of eigenvalue closest to 1
idx = np.argmin(np.abs(eigenvalues_dual - 1))

leading_eigenvalue_dual = eigenvalues_dual[idx]
leading_left_eigenvector_dual = eigenvectors_dual[:, idx]

# Normalize so it sums to 1 (stationary distribution)
stationary_dist_dual = leading_left_eigenvector_dual / np.sum(leading_left_eigenvector_dual)

flat_state = np.ones(num_states)
identity = np.identity(num_states)
trace_out = np.kron(identity, flat_state)

#print(np.sum(abs(stationary_dist2 @ dual_method - stationary_dist2)))
print(f"residual of dual matrix : {np.sum(abs(stationary_dist_dual @ transition_matrix - stationary_dist_dual))}")
print(f"residual of stationary state : {np.sum(abs(stationary_dist2 - stationary_dist_dual))}")
print(stationary_dist_exact)
print(trace_out @ stationary_dist_dual)

#print(coupling_matrix)