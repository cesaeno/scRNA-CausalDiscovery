import numpy as np
import pandas as pd

def scRNA_data_create(number_cells, number_genes, mean_beta=0.06):

    dataset = np.zeros((number_genes, number_cells))

    # Initialization of priors
    raw_betas = np.random.lognormal(mean=2.74, sigma=0.39, size=number_cells)
    scaled_betas = raw_betas * (mean_beta/np.mean(raw_betas))
    k_on_all = 10**(np.random.uniform(np.log10(0.01), np.log10(100), size=number_genes))
    r_all = np.abs(np.random.normal(loc=0.05, scale=0.5, size=number_genes))
    k_off_all = k_on_all / r_all
    fano_factor_all = 10**(np.random.uniform(np.log10(1.001), np.log10(30), size=number_genes))

    for i in range(number_genes):
        activation_probability_i = np.random.beta(k_on_all[i], k_off_all[i])

        # Raw rate of transcription once on
        k_synthetic_i = ((fano_factor_all[i] - 1) * (k_on_all[i] + k_off_all[i]) * (k_on_all[i] + k_off_all[i] + 1)) / \
                        k_off_all[i]
        for j in range(number_cells):
            k_syn_eff_ij = scaled_betas[j] * k_synthetic_i
            x_ij = np.random.poisson(k_syn_eff_ij * activation_probability_i)

            dataset[i, j] = x_ij

    return dataset

def scRNA_data_file_conversion(number_cells, number_genes, file_path):

    dataset = scRNA_data_create(number_cells, number_genes)

    data_frame = pd.DataFrame(dataset)
    return data_frame.to_csv(f"{file_path}.csv", index=False)
