#ifndef MODEL_H
#define MODEL_H

typedef struct {
    double **J;
    double *h;
    double T;
    int n_spins;
} Model;

typedef struct {
    double *element;
    int stride;
} Data;

Model *mcmc_allocate(int n_spins, double T, int seed);
void mcmc_free(Model *model);
void mcmc_print(Model *model);
void mcmc_print_array(double *array, int number_elements);
Data *mcmc_run_trajectories(Model *model, int n_steps, int n_trajectories, double *energy_lookup, int method);
double* mcmc_compute_energies(Model *model, double T);

#endif