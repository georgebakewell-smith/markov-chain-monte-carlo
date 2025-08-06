#ifndef MODEL_H
#define MODEL_H

typedef struct {
    double **J;
    double *h;
    double T;
    int n_spins;
} Model;

typedef struct {
    double *elements;
    int stride;
    int n_blocks;
} Data;

Model *mcmc_allocate(int n_spins, double T, int model_type, int seed);
void mcmc_free_model(Model *model);
void mcmc_free_data(Data *data);
void mcmc_print(Model *model);
void mcmc_print_array(double *array, int number_elements);
Data *mcmc_run_trajectories(Model *model, int n_steps, int n_trajectories, double *energy_lookup, int method);
double** mcmc_compute_lookups(Model *model, double T);
double mcmc_acceptance_MH(const double delta_E, const double T);
void mcmc_get_averages(double *average_magnetisation, double *magnetisation_error, double *energy_average, double *energy_error, Data *simulations, double *magnetisation_lookup, double *energy_lookup);
void mcmc_write_data_to_csv(const char *filename, double *exact_values, double *array_mag, double *array_mag_error, double *array_energy, double *array_energy_error, int length);
double* mcmc_get_exact_values(int n_spins, double T, double *energy_lookup, double *magnetisation_lookup);
double mcmc_sum(double *array, int stride, int n);
double mcmc_dot(double *a, double *b, int n);
int mcmc_load_config(int *n_spins, int *n_steps, int *n_trajectories, double *T, int *model_type);

#endif