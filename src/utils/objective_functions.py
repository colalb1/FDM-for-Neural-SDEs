import torch


@torch.compile
def unbiased_pairwise_score_estimator(
    generated_paths: torch.Tensor, empirical_paths: torch.Tensor, batch_size: int
) -> torch.tensor:
    """
    Computes an unbiased pairwise score estimator for evaluating the quality of generated paths
    against empirical (real) paths using the RBF kernel with unit kernel bandwidth.

    Args:
        generated_paths (torch.Tensor): A tensor of shape (batch_size, n) representing the generated paths.
        empirical_paths (torch.Tensor): A tensor of shape (batch_size, n) representing the empirical paths.
        batch_size (int): The number of paths in each batch. Must be greater than 1.

    Returns:
        float: The computed unbiased pairwise score that compares the similarity between generated paths and empirical paths.
    """

    # Edge case where the end of the data is reached
    if empirical_paths.shape[0] < batch_size:
        batch_size = empirical_paths.shape[0]
        generated_paths = generated_paths[:batch_size]

    assert batch_size > 1, "Batch size must be greater than 1!"
    assert (
        generated_paths.shape == empirical_paths.shape
    ), f"Generated paths and empirical paths do NOT have the same shape! Generated shape: {generated_paths.shape}. Empirical shape: {empirical_paths.shape}."

    n = generated_paths.shape[1]
    device = generated_paths.device

    # Generate random time indices for generated-to-generated and generated-to-empirical comparisons
    time_indices_generated_to_generated = torch.randint(
        low=0, high=n, size=(2, batch_size, batch_size), device=device
    )
    time_indices_generated_to_empirical = torch.randint(
        low=0, high=n, size=(2, batch_size, batch_size), device=device
    )

    # Precompute the batch index tensor
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)

    # Gather the time step values for generated and empirical paths
    generated_paths_t1 = generated_paths[
        batch_indices, time_indices_generated_to_generated[0]
    ]
    generated_paths_t2 = generated_paths[
        batch_indices, time_indices_generated_to_generated[1]
    ]
    generated_paths_t3 = generated_paths[
        batch_indices, time_indices_generated_to_empirical[0]
    ]
    empirical_paths_t4 = empirical_paths[
        batch_indices, time_indices_generated_to_empirical[1]
    ]

    # Compute pairwise RBF kernel values for generated to generated pairs
    diff_generated_to_generated = generated_paths_t1.unsqueeze(
        2
    ) - generated_paths_t2.unsqueeze(1)
    squared_dist_generated_to_generated = (
        diff_generated_to_generated.pow_(2).sum(dim=-1).mul_(0.5)
    )
    rbf_kernel_generated_to_generated = torch.exp(
        squared_dist_generated_to_generated.neg_()
    )

    # Exclude diagonal elements by creating a mask
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=device).unsqueeze(0)
    sum_rbf_generated_to_generated = rbf_kernel_generated_to_generated.masked_select(
        mask
    ).sum() / (2 * batch_size * (batch_size - 1))

    # Compute pairwise RBF kernel values for generated to empirical pairs
    diff_generated_to_empirical = generated_paths_t3.unsqueeze(
        2
    ) - empirical_paths_t4.unsqueeze(1)
    squared_dist_generated_to_empirical = (
        diff_generated_to_empirical.pow_(2).sum(dim=-1).mul_(0.5)
    )
    rbf_kernel_generated_to_empirical = torch.exp(
        squared_dist_generated_to_empirical.neg_()
    )

    sum_rbf_generated_to_empirical = rbf_kernel_generated_to_empirical.sum().div(
        batch_size**2
    )

    score = sum_rbf_generated_to_generated - sum_rbf_generated_to_empirical

    return score.requires_grad_()
