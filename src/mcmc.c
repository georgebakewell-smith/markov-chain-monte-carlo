#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <time.h>
#include "../include/mcmc.h"

double* mcmc_compute_energies(Model *model, double T){
 
    printf("Computing energy lookup table\n");
    int num_states = (int)gsl_pow_int(2, model->n_spins), s_i, s_j;
    double *energy = malloc(sizeof(double));
    double *energy_lookup = malloc(num_states*sizeof(double));

    for(int state=0; state<num_states; state++){
        *energy = 0;
        for(int i=0; i<model->n_spins; i++){
            s_i = (state >> (model->n_spins - 1 - i) & 1)*2 -1;
            for(int j=0; j<model->n_spins; j++){
                s_j = (state >> (model->n_spins - 1 - j) & 1)*2 -1;
                *energy += model->J[i][j]*s_i*s_j;
            }
            *energy += model->h[i]*s_i;
        }
        energy_lookup[state] = -*energy;
    }

    free(energy);

    return energy_lookup;
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


    mcmc_print(model);  

    return model;
}

void mcmc_free(Model *model){

    free(model->h);
    for(int i = 0; i < model->n_spins; i++){
        free(model->J[i]);
    }
    free(model->J);
    free(model);
    printf("Free!\n");
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

Data *mcmc_run_trajectories(Model *model, int n_steps, int n_trajectories, double *energy_lookup, int method){

    printf("Running trajectories...\n");

    Data *simulations = malloc(sizeof(Data));
    simulations->stride = n_trajectories;

    return simulations;

}

void mcmc_print_array(double *array, int number_elements){
    for(int i=0; i<number_elements;i++){
        printf("%.2f\n", array[i]);
    }
}