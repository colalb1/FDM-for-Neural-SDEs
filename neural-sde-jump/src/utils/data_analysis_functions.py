# I stole this from https://github.com/issaz/sigker-nsdes/blob/main/src/utils/helper_functions/data_helper_functions.py
# and modified it to be more readable.

import numpy as np
import torch
import torchsde


def get_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Returns the vector of log-returns.

    Args:
        prices (np.ndarray): Array of prices associated with an asset.

    Returns:
        np.ndarray: Vector of log-returns with prepended 0.
    """
    log_prices = np.log(prices)
    return np.diff(log_prices, prepend=log_prices[0])


def reweighter(num_elements: int, factor: int) -> np.ndarray:
    """
    Takes a number of elements in a vector and splits them progressively via the factor parameter.
    Higher factors mean more recent observations receive more weight.

    Args:
        num_elements (int): Number of elements to reweight.
        factor (int): Factor parameter.

    Returns:
        np.ndarray: Vector of indexes corresponding to reweight.
    """
    indices = np.arange(num_elements)

    if factor <= 1:
        return indices

    split_pct = 1.0 - 1.0 / factor
    res, new, old = [], [], []
    current_index = num_elements // factor
    i = 1

    while current_index > 0:
        old, new = np.split(indices, [int(split_pct * indices.shape[0])])
        res += list(np.repeat(old, i))

        indices = new
        current_index //= factor
        i += 1

    return np.array(res + list(np.repeat(new, i)))


def subtract_initial_point(paths):
    """
    Subtracts the initial point of each path from all subsequent points in that path.

    Args:
        paths (torch.Tensor): Tensor of shape (batch_size, length, dim) representing the paths.

    Returns:
        torch.Tensor: Tensor with the initial points subtracted.
    """
    _, length, dim = paths.size()
    adjusted_paths = paths.clone()
    start_points = torch.transpose(adjusted_paths[:, 0, 1:].unsqueeze(-1), -1, 1)
    adjusted_paths[..., 1:] -= torch.tile(start_points, (1, length, 1))
    return adjusted_paths


def batch_subtract_initial_point(paths):
    """
    Subtracts the initial point of each path from all subsequent points in that path for a batch of paths.

    Args:
        paths (torch.Tensor): Tensor of shape (batch_size, emp_size, length, dim) representing the paths.

    Returns:
        torch.Tensor: Tensor with the initial points subtracted.
    """
    _, _, length, _ = paths.size()
    adjusted_paths = paths.clone()
    start_points = torch.transpose(adjusted_paths[..., 0, 1:].unsqueeze(-1), -1, -2)
    adjusted_paths[..., 1:] -= torch.tile(start_points, (1, 1, length, 1))
    return adjusted_paths


class ConcatDataset(torch.utils.data.Dataset):
    """
    A dataset that concatenates multiple datasets.
    """

    def __init__(self, *datasets):
        """
        Initializes the ConcatDataset.

        Args:
            *datasets: Datasets to concatenate.
        """
        self.datasets = datasets

    def __getitem__(self, index):
        """
        Gets the index-th item from each dataset.

        Args:
            index (int): Index of the item to get.

        Returns:
            tuple: Tuple of items from each dataset.
        """
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        """
        Gets the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return min(len(dataset) for dataset in self.datasets)


def build_path_bank(
    sde, path_length, end_time, dataset_size, output_size, device, **sdeint_kwargs
):
    """
    Builds a bank of paths using a specified SDE model.

    Args:
        sde: SDE model to use.
        path_length (int): Length of the paths.
        end_time (Optional[int]): End time of the paths. Defaults to path_length - 1.
        dataset_size (int): Size of the dataset.
        output_size (int): Output size of the paths.
        device (torch.device): Device to perform computations on.
        **sdeint_kwargs: Additional keyword arguments for the SDE integrator.

    Returns:
        torch.Tensor: Tensor of generated paths.
    """
    initial_value = torch.full(size=(dataset_size, output_size), fill_value=1.0).to(
        device
    )

    if end_time is None:
        end_time = path_length - 1

    time_grid = torch.linspace(0, end_time, path_length, device=device).float()

    try:
        method = sdeint_kwargs.get("sde_method")
        time_scale = sdeint_kwargs.get("sde_dt_scale")
    except KeyError:
        print("sde_int arguments not provided, defaulting")
        if sde.sde_type == "stratonovich":
            method = "euler_heun"
        else:
            method = "euler"

        time_scale = 1.0

    dt = torch.diff(time_grid)[0] * time_scale
    int_kwargs = {"method": method, "dt": dt}

    if method == "reversible_heun":
        int_kwargs["adjoint_method"] = "adjoint_reversible_heun"
        integrator_func = torchsde.sdeint_adjoint
    else:
        integrator_func = torchsde.sdeint

    generated_paths = integrator_func(sde, initial_value, time_grid, **int_kwargs)

    return generated_paths.transpose(0, 1)


def get_scalings(paths, normalization_type="mean_var"):
    """
    Computes scaling factors for normalization.

    Args:
        paths (torch.Tensor): Tensor of paths.
        normalization_type (str, optional): Type of normalization. Defaults to "mean_var".

    Returns:
        tuple: Means and standard deviations or mins and maxs for normalization.
    """
    if normalization_type == "mean_var":
        means = paths[:, -1, :].mean(axis=0)
        stds = paths[:, -1, :].std(axis=0)
        return means, stds
    elif normalization_type == "min_max":
        mins = paths[:, -1, :].min(axis=0)
        maxs = paths[:, -1, :].max(axis=0)
        return mins, maxs
    else:
        return None, None


def normalize(tensor, normalization_type, val1=None, val2=None):
    """
    Normalizes the tensor.

    Args:
        tensor (torch.Tensor): Tensor to normalize.
        normalization_type (str): Type of normalization ("mean_var" or "min_max").
        val1 (Optional[torch.Tensor]): First normalization parameter (mean or min). Defaults to None.
        val2 (Optional[torch.Tensor]): Second normalization parameter (std or max). Defaults to None.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    if normalization_type is None:
        return tensor
    elif normalization_type == "mean_var":
        if val1 is None:
            mean = tensor[:, -1, 1:].mean(axis=0)
            std = tensor[:, -1, 1:].std(axis=0)
        else:
            mean, std = val1, val2

        tensor[..., 1:] = (tensor[..., 1:] - mean) / std
        return tensor
    elif normalization_type == "min_max":
        if val1 is None:
            min_val = tensor[:, -1, 1:].min(axis=0)
            max_val = tensor[:, -1, 1:].max(axis=0)
        else:
            min_val, max_val = val1, val2

        tensor[..., 1:] = (tensor[..., 1:] - min_val) / (max_val - min_val)
        return tensor
    else:
        return "Normalization type does not exist"


def inv_normalize(tensor, normalization_type, val1=None, val2=None):
    """
    Inversely normalizes the tensor.

    Args:
        tensor (torch.Tensor): Tensor to inversely normalize.
        normalization_type (str): Type of normalization ("mean_var" or "min_max").
        val1 (Optional[torch.Tensor]): First normalization parameter (mean or min). Defaults to None.
        val2 (Optional[torch.Tensor]): Second normalization parameter (std or max). Defaults to None.

    Returns:
        torch.Tensor: Inversely normalized tensor.
    """
    if normalization_type is None:
        return tensor
    elif normalization_type == "mean_var":
        if val1 is None:
            mean = tensor[:, -1, 1:].mean(axis=0)
            std = tensor[:, -1, 1:].std(axis=0)
        else:
            mean, std = val1, val2

        tensor[..., 1:] = std * tensor[..., 1:] + mean
        return tensor
    elif normalization_type == "min_max":
        if val1 is None:
            min_val = tensor[:, -1, 1:].min(axis=0)
            max_val = tensor[:, -1, 1:].max(axis=0)
        else:
            min_val, max_val = val1, val2

        tensor[..., 1:] = tensor[..., 1:] * (max_val - min_val) + min_val
        return tensor
    else:
        return "Normalization type does not exist"


def process_generator(
    num_paths: int,
    total_time: float,
    path_length: int,
    drift: float,
    volatility: float,
    initial_value,
    process_type: str = "gbm",
) -> np.ndarray:
    """
    Generates paths for a specified process (Geometric Brownian Motion or Brownian Motion).

    Args:
        num_paths (int): Number of paths to generate.
        total_time (float): Total time.
        path_length (int): Length of each path.
        drift (float): Drift coefficient.
        volatility (float): Volatility coefficient.
        initial_value (float or np.ndarray): Initial values for each path.
        process_type (str, optional): Type of process ("gbm" or "bm"). Defaults to "gbm".

    Returns:
        np.ndarray: Generated paths.
    """
    # Check if the initial value is a float and create an array of initial values
    if isinstance(initial_value, float):
        initial_values = np.full((num_paths, 1), initial_value)
    else:
        assert (
            initial_value.shape[0] == num_paths
        ), "Custom initial points must be provided for each path."
        initial_values = np.expand_dims(initial_value, -1)

    # Create a time grid and compute the time step size
    time_grid = np.linspace(0, total_time, path_length)
    time_step = np.diff(time_grid)[0]

    # Initialize a Brownian motion array
    brownian_motion = np.zeros((num_paths, path_length))
    brownian_motion[:, 1:] = np.sqrt(time_step) * np.random.normal(
        0, 1, size=(num_paths, path_length - 1)
    ).cumsum(axis=1)

    if process_type == "gbm":
        # Generate paths for Geometric Brownian Motion
        return initial_values * np.exp(
            (drift - 0.5 * np.power(volatility, 2)) * time_grid
            + volatility * brownian_motion
        )
    elif process_type == "bm":
        # Generate paths for Brownian Motion
        return initial_values + volatility * brownian_motion + drift * time_grid
