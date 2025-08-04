#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_math.h>
#include "../include/mcmc.h"

int main (void)
{
  int n_spins = 14, n_steps = 100, n_trajectories = 1, method = 0;
  double T = 0.1;

  Model *model = mcmc_allocate(n_spins, T, -1);
  double *energy_lookup = mcmc_compute_energies(model, T);

  Data *uniform_simulations = mcmc_run_trajectories(model, n_steps, n_trajectories, energy_lookup, method);

  mcmc_free(model);
  free(uniform_simulations);
  free(energy_lookup);

  return 0;
}