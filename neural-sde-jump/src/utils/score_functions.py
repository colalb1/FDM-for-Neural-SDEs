# This code was stolen from https://github.com/issaz/sigker-nsdes/blob/main/src/evaluation/evaluation_functions.py
# and modified to be more readable.

from typing import Optional

import numpy as np
import torch
from scipy.stats import ks_2samp

from src.utils.data_analysis_functions import subtract_initial_point


def get_cross_correlation_matrix(
    paths: torch.Tensor, lags: tuple = (0, 1, 2, 3, 4, 5, 6)
) -> torch.tensor:
    """
    Computes the cross-correlation matrix for a set of paths with specified time lags.

    Args:
        paths (torch.Tensor): Tensor of shape (n_paths, path_length, dim) representing the paths.
        lags (tuple, optional): Tuple of integers representing the time lags to compute cross-correlations. Defaults to (0, 1, 2, 3, 4, 5, 6).

    Returns:
        torch.tensor: Cross-correlation matrix of shape (n_lags, n_lags).
    """
    n_lags = len(lags)

    # Initialize the cross-correlation matrix as an identity matrix
    cross_corr_matrix = torch.eye(n_lags, n_lags)

    # Iterate over pairs of lags to compute cross-correlation
    for i, lag1 in enumerate(lags):
        for j, lag2 in enumerate(lags):
            if i < j:
                if lag1 == 0:
                    # Case where lag1 is zero
                    forward_slice = paths[:, lag2 - 1 :, 0]
                    backward_slice = (
                        paths[:, : -(lag2 - 1), 1] if lag2 > 1 else paths[..., 1]
                    )

                    # Compute average correlation
                    avg_corr = torch.tensor(
                        [
                            np.corrcoef(p, q)[0, 1]
                            for p, q in zip(forward_slice, backward_slice)
                        ]
                    ).mean()
                else:
                    # Case where lag1 is not zero
                    lag1_index = int(lag1 - 1)
                    lag2_index = int(lag2 - 1)
                    forward_shift = lag2_index - lag1_index

                    forward_slice = (
                        paths[:, forward_shift:-lag1_index, 1]
                        if lag1_index != 0
                        else paths[:, forward_shift:, 1]
                    )
                    backward_slice = paths[:, :-lag2_index, 1]

                    # Compute average correlation
                    avg_corr = torch.tensor(
                        [
                            np.corrcoef(p, q)[0, 1]
                            for p, q in zip(forward_slice, backward_slice)
                        ]
                    ).mean()

                # Fill the cross-correlation matrix symmetrically
                cross_corr_matrix[i, j] = avg_corr
                cross_corr_matrix[j, i] = avg_corr

    return cross_corr_matrix


def get_ks_scores(
    real_paths: torch.Tensor,
    generated_paths: torch.Tensor,
    marginals: list[float],
    dim: int = 1,
) -> np.ndarray:
    """
    Computes the Kolmogorov-Smirnov (KS) statistics and p-values for the distributions of real and generated paths
    at specified marginal points.

    Args:
        real_paths (torch.Tensor): Tensor of shape (n_paths, path_length, dim) representing the real paths.
        generated_paths (torch.Tensor): Tensor of the same shape as real_paths representing the generated paths.
        marginals (list[float]): List of fractional indices (between 0 and 1) indicating the points along the path length where the KS test should be performed.
        dim (int, optional): Array of shape (len(marginals), 2) where each row contains the KS statistic and p-value for a specific marginal point. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """
    _, path_length, _ = real_paths.shape  # Extract path length
    ks_scores = np.zeros(
        (len(marginals), 2)
    )  # Initialize the array to store KS statistics and p-values

    # Iterate over each marginal point
    for i, marginal in enumerate(marginals):
        index = int(
            marginal * path_length
        )  # Compute the index corresponding to the marginal point

        # Extract the marginal distributions for real and generated paths at the specified index
        real_marginals = real_paths[:, index, dim]
        generated_marginals = generated_paths[:, index, dim]

        # Perform the KS test
        ks_statistic, ks_p_value = ks_2samp(
            real_marginals, generated_marginals, alternative="two_sided"
        )

        # Store the KS statistic and p-value in the scores array
        ks_scores[i, 0] = ks_statistic
        ks_scores[i, 1] = ks_p_value

    return ks_scores


def generate_ks_results(
    times: torch.Tensor,
    dataloader: torch.utils.data.DataLoader,
    generators: list[callable],
    marginals: list[float],
    n_runs: int,
    dims: int = 1,
    eval_batch_size: int = 128,
) -> np.ndarray:
    """
    Generates KS test results for multiple runs and dimensions, comparing real and generated paths using different generators.

    Args:
        times (torch.Tensor): A time series input or a placeholder required by the generators.
        dataloader (torch.utils.data.DataLoader): Data loader that provides batches of real samples.
        generators (list[callable]): List of generator models used to generate paths.
        marginals (list[float]): List of fractional indices (between 0 and 1) indicating the points along the path length where the KS test should be performed.
        n_runs (int): Number of runs to perform the evaluation.
        dims (int, optional): Number of dimensions to evaluate. Defaults to 1.
        eval_batch_size (int, optional): Batch size for evaluation. Defaults to 128.

    Returns:
        np.ndarray: A numpy array of shape (3, n_runs, dims, len(marginals), 2) containing KS statistics and p-values.
    """

    # Initialize the array to store the total KS scores for all runs, dimensions, and generators
    total_scores = np.zeros((3, n_runs, dims, len(marginals), 2))

    # Loop through each run
    for run_index in range(n_runs):
        with torch.no_grad():
            # Fetch a batch of real samples from the data loader and subtract the initial point
            (real_samples,) = next(iter(dataloader))
            real_samples = subtract_initial_point(real_samples).cpu()

            # Loop through each generator
            for generator_index, generator in enumerate(generators):
                # Generate a batch of samples and subtract the initial point
                generated_samples = subtract_initial_point(
                    generator(times, eval_batch_size)
                ).cpu()

                # Loop through each dimension
                for dim_index in range(dims):
                    # Compute the KS scores using the get_ks_scores function
                    total_scores[generator_index, run_index, dim_index] = get_ks_scores(
                        real_samples, generated_samples, marginals, dim=dim_index + 1
                    )

    return total_scores


def generate_ks_results_nspde(
    grid: torch.Tensor,
    dataloader: torch.utils.data.DataLoader,
    generators: list[callable],
    marginals: list[float],
    n_runs: int,
    dims: int = 1,
    eval_batch_size: int = 128,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Generates KS test results for multiple runs and dimensions, comparing real and generated paths using different generators,
    specifically for neural SPDEs.

    Args:
        grid (torch.Tensor): Tensor representing the spatial-temporal grid.
        dataloader (torch.utils.data.DataLoader): Data loader that provides batches of real samples.
        generators (list[callable]): List of generator models used to generate paths.
        marginals (list[float]): List of fractional indices (between 0 and 1) indicating the points along the path length where the KS test should be performed.
        n_runs (int): Number of runs to perform the evaluation.
        dims (int, optional): Number of dimensions to evaluate. Defaults to 1.
        eval_batch_size (int, optional): Batch size for evaluation. Defaults to 128.
        device (Optional[torch.device], optional): Device to perform computations on. Defaults to None.

    Returns:
        np.ndarray: A numpy array of shape (len(generators), n_runs, dims, len(marginals), 2) containing KS statistics and p-values.
    """
    # Initialize the array to store the total KS scores for all runs, dimensions, and generators
    total_scores = np.zeros((len(generators), n_runs, dims, len(marginals), 2))

    # Loop through each run
    for run_index in range(n_runs):
        with torch.no_grad():
            # Fetch a batch of real samples from the data loader until we get a batch of the correct size
            real_samples = next(iter(dataloader))
            while real_samples.shape[0] != eval_batch_size:
                real_samples = next(iter(dataloader))

            # Prepare initial condition u0 for the generators
            u0 = real_samples[:, 0, :].permute(0, 2, 1).float().to(device)

            # Loop through each generator
            for generator_index, generator in enumerate(generators):
                # Generate a batch of samples
                generated_samples = generator(grid, eval_batch_size, u0).cpu()

                # Loop through each dimension
                for dim_index in range(dims):
                    real_data = real_samples  # Use real_samples directly

                    # Compute the KS scores using the get_ks_scores function
                    total_scores[generator_index, run_index, dim_index] = get_ks_scores(
                        real_data[..., 0],
                        generated_samples[..., 0],
                        marginals,
                        dim=dim_index,
                    )

    return total_scores
