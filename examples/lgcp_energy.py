# Energy for the log Gaussian Cox process
# The code is a "to torch translated" version from
# https://github.com/google-deepmind/annealed_flow_transport

import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_bin_counts(array_in: np.array, num_bins_per_dim: int) -> np.array:
    scaled_array = array_in * num_bins_per_dim
    counts = np.zeros((num_bins_per_dim, num_bins_per_dim))
    for elem in scaled_array:
        flt_row, col_row = np.floor(elem)
        row = int(flt_row)
        col = int(col_row)
        # Deal with the case where the point lies exactly on upper/rightmost edge.
        if row == num_bins_per_dim:
            row -= 1
        if col == num_bins_per_dim:
            col -= 1
        counts[row, col] += 1
    return counts


def get_bin_vals(num_bins: int) -> np.array:
    grid_indices = torch.arange(num_bins, device=device, dtype=torch.float)
    bin_vals = torch.stack(
        (
            grid_indices[:, None].tile(1, num_bins).reshape(-1),
            grid_indices[None, :].tile(num_bins, 1).reshape(-1),
        ),
        -1,
    )
    return bin_vals


def get_latents_from_white(white, const_mean, cholesky_gram):
    latent_function = torch.matmul(cholesky_gram, white) + const_mean
    return latent_function


def get_white_from_latents(latents, const_mean, cholesky_gram):
    white = torch.linalg.solve_triangular(
        cholesky_gram, (latents - const_mean).T, upper=False
    ).T
    return white


def poisson_process_log_likelihood(latent_function, bin_area, flat_bin_counts):
    first_term = latent_function * flat_bin_counts
    second_term = -bin_area * torch.exp(latent_function)
    return torch.sum(first_term + second_term, -1)


def log_gaussian_cox_process():
    num_latents = 1600
    num_grid_per_dim = int(np.sqrt(num_latents))
    file_path = "data/pines.csv"

    def get_pines_points(file_path):
        """Get the pines data points."""
        with open(file_path, mode="rt") as input_file:
            # with open(file_path, "rt") as input_file:
            b = np.genfromtxt(input_file, delimiter=",")
        return b

    bin_counts = torch.tensor(
        get_bin_counts(get_pines_points(file_path), num_grid_per_dim),
        device=device,
        dtype=torch.float,
    )
    flat_bin_counts = torch.reshape(bin_counts, (num_latents,))
    poisson_a = 1.0 / num_latents
    signal_variance = 1.91
    beta = 1.0 / 33

    bin_vals = get_bin_vals(num_grid_per_dim)

    diffs = torch.sum((bin_vals[None, :, :] - bin_vals[:, None, :]) ** 2, -1)
    normalized_distance = torch.sqrt(diffs) / (num_grid_per_dim * beta)
    gram_matrix = signal_variance * torch.exp(-normalized_distance)
    cholesky_gram = torch.linalg.cholesky(gram_matrix)

    half_log_det_gram = torch.sum(torch.log(torch.abs(torch.diag(cholesky_gram))))
    unwhitened_gaussian_log_normalizer = (
        -0.5 * num_latents * np.log(2.0 * np.pi) - half_log_det_gram
    )

    mu_zero = np.log(126.0) - 0.5 * signal_variance

    def unwhitened_posterior_log_density(latents):
        white = get_white_from_latents(latents, mu_zero, cholesky_gram)
        prior_log_density = (
            -0.5 * torch.sum(white**2, -1) + unwhitened_gaussian_log_normalizer
        )
        log_likelihood = poisson_process_log_likelihood(
            latents, poisson_a, flat_bin_counts
        )
        return prior_log_density + log_likelihood

    # use_whitened is False
    posterior_log_density = unwhitened_posterior_log_density

    def target_energy(x):
        return -posterior_log_density(x)

    return target_energy
