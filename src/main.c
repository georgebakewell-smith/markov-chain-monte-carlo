#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_math.h>
#include "../include/mcmc.h"

int main (void)
{

    // Default variables for known parameters
    int n_spins = 4;
    int n_steps = 100;
    int n_trajectories = 10;
    double T = 0.1;
    int model_type = 0;

    if(mcmc_load_config(&n_spins, &n_steps, &n_trajectories, &T, &model_type) == 0){
        printf("Configuration loaded: n_spins=%d, n_steps=%d, n_trajectories=%d, T=%.2f, model_type=%d\n", n_spins, n_steps, n_trajectories, T, model_type);
    } else{
        return 1;
    }

    int method1 = 0, method2 = 1;

    Model *model = mcmc_allocate(n_spins, T, model_type, -1);
    //mcmc_print(model);
    double **lookups = mcmc_compute_lookups(model, T);
    double *energy_lookup = lookups[0];
    double *magnetisation_lookup = lookups[1];
    free(lookups);
    double *exact_values = mcmc_get_exact_values(model->n_spins, T, energy_lookup, magnetisation_lookup);
    double *magnetisation_average_uniform = malloc(n_steps*sizeof(double));
    double *magnetisation_average_local = malloc(n_steps*sizeof(double));
    double *uniform_mag_error = malloc(n_steps*sizeof(double));
    double *local_mag_error = malloc(n_steps*sizeof(double));
    double *energy_average_uniform = malloc(n_steps*sizeof(double));
    double *energy_average_local = malloc(n_steps*sizeof(double));
    double *energy_error_uniform = malloc(n_steps*sizeof(double));
    double *energy_error_local = malloc(n_steps*sizeof(double));

    // Running simulations
    Data *uniform_simulations = mcmc_run_trajectories(model, n_steps, n_trajectories, energy_lookup, method1);
    Data *local_simulations = mcmc_run_trajectories(model, n_steps, n_trajectories, energy_lookup, method2);
    mcmc_free_model(model);

    // Data processing
    mcmc_get_averages(magnetisation_average_uniform, uniform_mag_error, energy_average_uniform, energy_error_uniform, uniform_simulations, magnetisation_lookup, energy_lookup);
    mcmc_free_data(uniform_simulations);
    mcmc_get_averages(magnetisation_average_local, local_mag_error, energy_average_local, energy_error_local, local_simulations, magnetisation_lookup, energy_lookup);
    mcmc_free_data(local_simulations); 
    
    free(magnetisation_lookup);
    free(energy_lookup);

    mcmc_write_data_to_csv("uniform_simulations.csv", exact_values, magnetisation_average_uniform, uniform_mag_error, energy_average_uniform, energy_error_uniform, n_steps);
    mcmc_write_data_to_csv("local_simulations.csv", exact_values, magnetisation_average_local, local_mag_error, energy_average_local, energy_error_local, n_steps);

    free(magnetisation_average_uniform);
    free(magnetisation_average_local);
    free(exact_values);
    free(uniform_mag_error);
    free(local_mag_error); 
    free(energy_average_uniform);
    free(energy_average_local);
    free(energy_error_uniform);
    free(energy_error_local);

    printf("Model and data structures freed successfully!\n");
    printf("Simulation completed successfully!\n");

    return 0;
}