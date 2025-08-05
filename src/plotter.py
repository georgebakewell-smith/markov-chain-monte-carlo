import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def read_simulation_data_linewise(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != '']

    vals = [float(x) for x in lines]
    if len(vals) < 3:
        raise ValueError(f"{filename}: not enough lines (need at least 3)")

    n = int(vals[0])
    exact_energy = vals[1]
    exact_magnetisation = vals[2]
    remaining = vals[3:]

    if len(remaining) == 2 * n:
        est_mag = np.array(remaining[0:n])
        err_mag = np.array(remaining[n:2*n])
        est_energy = None
        err_energy = None
    elif len(remaining) == 4 * n:
        est_mag = np.array(remaining[0:n])
        err_mag = np.array(remaining[n:2*n])
        est_energy = np.array(remaining[2*n:3*n])
        err_energy = np.array(remaining[3*n:4*n])
    else:
        raise ValueError(
            f"{filename}: unexpected number of lines. Expected 3+2*n or 3+4*n, got {len(vals)} total."
        )

    return {
        'n': n,
        'exact_energy': exact_energy,
        'exact_magnetisation': exact_magnetisation,
        'est_mag': est_mag,
        'err_mag': err_mag,
        'est_energy': est_energy,
        'err_energy': err_energy
    }

def plot_simulations(file1='uniform_simulations.csv', file2='local_simulations.csv', savefile=None):
    d1 = read_simulation_data_linewise(file1)
    d2 = read_simulation_data_linewise(file2)

    x1 = np.arange(d1['n'])
    x2 = np.arange(d2['n'])

    fig, (ax_mag, ax_energy) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Magnetisation and Energy with Errors')

    # Magnetisation
    ax_mag.plot(x1, d1['est_mag'], label='Uniform', color='blue')
    ax_mag.fill_between(x1, d1['est_mag'] - d1['err_mag'], d1['est_mag'] + d1['err_mag'], color='blue', alpha=0.3)

    ax_mag.plot(x2, d2['est_mag'], label='Local', color='green')
    ax_mag.fill_between(x2, d2['est_mag'] - d2['err_mag'], d2['est_mag'] + d2['err_mag'], color='green', alpha=0.3)

    if np.isclose(d1['exact_magnetisation'], d2['exact_magnetisation']):
        ax_mag.axhline(d1['exact_magnetisation'], linestyle='--', color='red', label='Exact M')
    else:
        ax_mag.axhline(d1['exact_magnetisation'], linestyle='--', color='red', label='Exact M (Uniform)')
        ax_mag.axhline(d2['exact_magnetisation'], linestyle='--', color='orange', label='Exact M (Local)')

    ax_mag.set_ylabel('Magnetisation')
    ax_mag.legend()
    ax_mag.grid(True)

    # Energy
    if d1['est_energy'] is not None:
        ax_energy.plot(x1, d1['est_energy'], label='Uniform Energy', color='blue')
        ax_energy.fill_between(x1, d1['est_energy'] - d1['err_energy'], d1['est_energy'] + d1['err_energy'], color='blue', alpha=0.3)

    if d2['est_energy'] is not None:
        ax_energy.plot(x2, d2['est_energy'], label='Local Energy', color='green')
        ax_energy.fill_between(x2, d2['est_energy'] - d2['err_energy'], d2['est_energy'] + d2['err_energy'], color='green', alpha=0.3)

    if d1['est_energy'] is not None and d2['est_energy'] is not None:
        if np.isclose(d1['exact_energy'], d2['exact_energy']):
            ax_energy.axhline(d1['exact_energy'], linestyle='--', color='red', label='Exact E')
        else:
            ax_energy.axhline(d1['exact_energy'], linestyle='--', color='red', label='Exact E (Uniform)')
            ax_energy.axhline(d2['exact_energy'], linestyle='--', color='orange', label='Exact E (Local)')

    ax_energy.set_ylabel('Energy')
    ax_energy.set_xlabel('Sample Index')
    ax_energy.legend()
    ax_energy.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if savefile:
        plt.savefig(savefile, dpi=200)
    else:
        plt.show()

if __name__ == '__main__':
    plot_simulations()