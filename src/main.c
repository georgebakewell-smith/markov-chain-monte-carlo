#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_math.h>
#include "../include/mcmc.h"

int main (void)
{
  int n_spins = 10, n_steps = 5000, n_trajectories = 1000, method1 = 0, method2 = 1;
  double T = 0.1;

  Model *model = mcmc_allocate(n_spins, T, -1);
  double **lookups = mcmc_compute_lookups(model, T);
  double *energy_lookup = lookups[0];
  double *magnetisation_lookup = lookups[1];
  double *exact_values = mcmc_get_exact_values(n_spins, T, energy_lookup, magnetisation_lookup);
  double * average_magnetisation_uniform = malloc(n_steps*sizeof(double));
  double * average_magnetisation_local = malloc(n_steps*sizeof(double));

  // Running simulations
  Data *uniform_simulations = mcmc_run_trajectories(model, n_steps, n_trajectories, energy_lookup, method1);
  Data *local_simulations = mcmc_run_trajectories(model, n_steps, n_trajectories, energy_lookup, method2);

  // Data processing
  mcmc_get_running_magnetisation(average_magnetisation_uniform, uniform_simulations, magnetisation_lookup);
  mcmc_get_running_magnetisation(average_magnetisation_local, local_simulations, magnetisation_lookup);

  mcmc_write_data_to_csv("uniform_simulations.csv", exact_values, average_magnetisation_uniform, n_steps);
  mcmc_write_data_to_csv("local_simulations.csv", exact_values, average_magnetisation_local, n_steps);

  // Print results

  mcmc_free_model(model);
  mcmc_free_data(uniform_simulations);
  mcmc_free_data(local_simulations);
  free(average_magnetisation_uniform);
  free(average_magnetisation_local);
  free(energy_lookup);
  free(magnetisation_lookup);
  free(lookups);

  printf("Model and data structures freed successfully!\n");
  printf("Simulation completed successfully!\n");

  return 0;
}