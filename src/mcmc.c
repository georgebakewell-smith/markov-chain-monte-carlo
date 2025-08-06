#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_statistics_double.h>
#include <time.h>
#include "../include/mcmc.h"

Data *mcmc_run_trajectories(Model *model, int n_steps, int n_trajectories, double *energy_lookup, int method){

    printf("Running trajectories...\n");

    int n_states, current_state, proposal_state, index;
    n_states = (int)gsl_pow_int(2, model->n_spins);
    // Setup rng environment
    const gsl_rng_type * Type;
    gsl_rng * r;
    Type = gsl_rng_default;
    r = gsl_rng_alloc (Type);
    gsl_rng_set(r, time(NULL));
    
    Data *simulations = malloc(sizeof(Data));
    simulations->stride = n_steps;
    simulations->elements = malloc(n_steps*n_trajectories *sizeof(double));
    simulations->n_blocks = n_trajectories;

    for(int i=0; i<n_trajectories; i++){

        current_state = gsl_rng_uniform_int(r, n_states);
        simulations->elements[i*n_steps] = current_state;

        for(int j=1; j<n_steps; j++){
            if(method==0){
                proposal_state = gsl_rng_uniform_int(r, n_states);

            }else{
                index = gsl_rng_uniform_int(r, model->n_spins);
                proposal_state = current_state ^ (1 << (model->n_spins - 1 - index));
            }
            double delta_E = energy_lookup[proposal_state] - energy_lookup[current_state];

            if(mcmc_acceptance_MH(delta_E, model->T) > gsl_rng_uniform(r)){
                current_state = proposal_state;
            }

            simulations->elements[i*n_steps + j] = current_state;
        }
    }

    return simulations;

}

void mcmc_get_averages(double *magnetisation_average, double *magnetisation_error, double *energy_average, double *energy_error, Data *simulations, double *magnetisation_lookup, double *energy_lookup){
    printf("Calculating running magnetisation...\n");
    int stride = simulations->stride;
    Data *running_magnetisation = malloc(sizeof(Data));
    Data *energies = malloc(sizeof(Data));
    energies->stride = stride;
    energies->elements = malloc(simulations->n_blocks*simulations->stride*sizeof(double));
    energies->n_blocks = simulations->n_blocks;
    running_magnetisation->stride = stride;
    running_magnetisation->elements = malloc(simulations->n_blocks*simulations->stride*sizeof(double));
    running_magnetisation->n_blocks = simulations->n_blocks;

    for(int i=0; i<running_magnetisation->n_blocks; i++){
        running_magnetisation->elements[i*stride] = 0;
        energies->elements[i*stride] = energy_lookup[(int)simulations->elements[i*stride]];
        for(int j=1; j<stride; j++){
            running_magnetisation->elements[i*stride + j] = (running_magnetisation->elements[i*stride + (j-1)]*j + magnetisation_lookup[(int)simulations->elements[i*stride + j]])/(double)(j+1);
            energies->elements[i*stride + j] = energy_lookup[(int)simulations->elements[i*stride + j]];
        }
    }

    for(int i=0; i<stride; i++){
        magnetisation_average[i] = gsl_stats_mean(running_magnetisation->elements + i, stride, running_magnetisation->n_blocks);
        magnetisation_error[i] = gsl_stats_sd_m(running_magnetisation->elements + i, stride, running_magnetisation->n_blocks, magnetisation_average[i]);
        energy_average[i] = gsl_stats_mean(energies->elements + i, stride, energies->n_blocks);
        energy_error[i] = gsl_stats_sd_m(energies->elements + i, stride, energies->n_blocks, energy_average[i]);
    }

    mcmc_free_data(running_magnetisation);
    mcmc_free_data(energies);
}

double mcmc_acceptance_MH(const double delta_E, const double T){

    return GSL_MIN(1.0, exp(-delta_E/T));
}

double** mcmc_compute_lookups(Model *model, double T){
 
    printf("Computing energy lookup table...\n");
    int num_states = (int)gsl_pow_int(2, model->n_spins), s_i, s_j;
    double **lookups = malloc(2 * sizeof(double *));
    lookups[0] = malloc(num_states*sizeof(double));
    lookups[1] = malloc(num_states*sizeof(double));

    for(int state=0; state<num_states; state++){
        lookups[0][state] = 0;
        lookups[1][state] = 0;
        for(int i=0; i<model->n_spins; i++){
            s_i = (state >> (model->n_spins - 1 - i) & 1)*2 -1;
            lookups[1][state] += s_i;
            for(int j=0; j<model->n_spins; j++){
                s_j = (state >> (model->n_spins - 1 - j) & 1)*2 -1;
                lookups[0][state] += model->J[i][j]*s_i*s_j;
            }
            lookups[0][state] += model->h[i]*s_i;
        }
        lookups[0][state] = -lookups[0][state];
        lookups[1][state] /= model->n_spins;
        //printf("State %d: E = %.3f, M = %.3f\n", state, lookups[0][state], lookups[1][state]);
    }
    
    return lookups;
}

Model* mcmc_allocate(int n_spins, double T, int seed){

    // Setup rng environment
    const gsl_rng_type * Type;
    gsl_rng * r;
    gsl_rng_env_setup();
    Type = gsl_rng_default;
    r = gsl_rng_alloc (Type);
    if(seed == -1){
        gsl_rng_set(r, time(NULL));
    }else{
        gsl_rng_set(r, seed);
    }
    
    // Allocate memory
    Model *model = malloc(sizeof(Model));
    model->n_spins = n_spins;
    model->T = T;
    model->h = malloc(n_spins*sizeof(double));
    model->J = malloc(n_spins*sizeof(double *));

    for(int i = 0; i < n_spins; i++){
        model->J[i] = malloc(n_spins*sizeof(double));
        for(int j = 0; j < n_spins; j++){
            *(model->J[i] + j) = gsl_ran_gaussian (r, 1.0);    
        }

        *(model->h + i) = gsl_ran_gaussian (r, 1.0);
    }

    return model;
}

void mcmc_free_model(Model *model){

    free(model->h);
    for(int i = 0; i < model->n_spins; i++){
        free(model->J[i]);
    }
    free(model->J);
    free(model);
}

void mcmc_free_data(Data *data){
    if(data != NULL){
        free(data->elements);
        free(data);
    } else {
        printf("No data to free!\n");
    }
}

void mcmc_print(Model *model){
    /*
         Prints the model for testing
    */
    printf("J = \n");
    for(int i = 0; i < model->n_spins; i++){

        for(int j = 0; j < model->n_spins; j++){
            printf("%.3f\t", *(model->J[i]+j));
        }
        printf("\n");
    }
    printf ("\n");
    printf("h = \n");
    for(int i = 0; i < model->n_spins; i++){
        printf("%.3f\t", *(model->h + i));
    }
    printf ("\n");
}

void mcmc_print_array(double *array, int number_elements){

    for(int i=0; i<number_elements;i++){
        printf("%.2f\n", array[i]);
    }
}

void mcmc_write_data_to_csv(const char *filename, double *exact_values, double *array_mag, double *array_mag_error, double *array_energy, double *array_energy_error, int length) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Failed to open file");
        return;
    }

    // Data parameters
    fprintf(fp, "%d\n", length);
    fprintf(fp, "%f\n", exact_values[0]);
    fprintf(fp, "%f\n", exact_values[1]);

    // Write magnetisation averages
    for (int i = 0; i < length; i++) {
        fprintf(fp, "%f\n", array_mag[i]);
    }
    // Write magnetisation errors
    for (int i = 0; i < length; i++) {
        fprintf(fp, "%f\n", array_mag_error[i]);
    }
    // Write energy averages
    for (int i = 0; i < length; i++) {
        fprintf(fp, "%f\n", array_energy[i]);
    }
    // Write energy errors
    for (int i = 0; i < length; i++) {
        fprintf(fp, "%f\n", array_energy_error[i]);
    }
    fclose(fp);
}

double mcmc_sum(double *array, int stride, int n){
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += array[i * stride];
    }
    return sum;
}
double mcmc_dot(double *a, double *b, int n){
    double dot_product = 0.0;
    for (int i = 0; i < n; i++) {
        dot_product += a[i] * b[i];
    }

    return dot_product;
}

double* mcmc_get_exact_values(int n_spins, double T, double *energy_lookup, double *magnetisation_lookup) {
    double *exact_values = malloc(2 * sizeof(double));
    int n_states = (int)gsl_pow_int(2, n_spins);
    double E_min = gsl_stats_min(energy_lookup, 1, n_states);

    double *boltzmann_weight = malloc(n_states * sizeof(double));
    double *shifted_energy_lookup = malloc(n_states * sizeof(double));
    for (int i = 0; i < n_states; i++) {
        shifted_energy_lookup[i] = energy_lookup[i] - E_min;
        boltzmann_weight[i] = exp(-shifted_energy_lookup[i]/T);
    }
    double norm = mcmc_sum(boltzmann_weight, 1, n_states);
    exact_values[0] = mcmc_dot(energy_lookup, boltzmann_weight, n_states) / norm;
    exact_values[1] = mcmc_dot(magnetisation_lookup, boltzmann_weight, n_states) / norm;

    free(boltzmann_weight);
    free(shifted_energy_lookup);

    printf("Exact values computed: E = %.3f, M = %.3f\n", exact_values[0], exact_values[1]);
    
    return exact_values;
}