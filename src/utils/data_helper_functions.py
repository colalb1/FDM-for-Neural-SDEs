import torch
import torchcde
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_time_series_data(
    time_series_data: torch.Tensor,
    batch_size: int,
    normalize: bool = True,
    test_size: float = 0.2,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int]:
    """
    Preprocesses time series data for training a neural SDE model and splits it into training and evaluation sets.
    Supports multidimensional time series data.

    Args:
        time_series_data (torch.Tensor): The time series data to preprocess.
        batch_size (int): The size of mini-batches.
        normalize (bool, optional): Whether to normalize the data. Defaults to True.
        test_size (float, optional): The proportion of the data to hold out for evaluation. Defaults to 0.2.

    Returns:
        tuple: A tuple containing two PyTorch DataLoader objects:
            - infinite_train_dataloader (torch.utils.data.DataLoader): The DataLoader for the preprocessed training data.
            - infinite_evaluation_dataloader (torch.utils.data.DataLoader): The DataLoader for the preprocessed evaluation data.
            - data_size (int): The size of the data after preprocessing.
    """
    # Convert the data to a NumPy array
    np_data = time_series_data.detach().numpy()

    # Normalize the data if specified
    if normalize:
        scaler = StandardScaler()
        np_data = scaler.fit_transform(np_data)

    # Split into training and evaluation sets
    train_sequences, eval_sequences = train_test_split(
        np_data, test_size=test_size, random_state=42
    )

    # Convert the sequences to PyTorch tensors
    training_data = torch.as_tensor(train_sequences, dtype=torch.float32)
    evaluation_data = torch.as_tensor(eval_sequences, dtype=torch.float32)

    # Linear interpolation for Neural CDEs
    train_coeffs = torchcde.linear_interpolation_coeffs(training_data)
    eval_coeffs = torchcde.linear_interpolation_coeffs(evaluation_data)

    # Converting linear coefficients to tensor form
    train_dataset = torch.utils.data.TensorDataset(train_coeffs)
    eval_dataset = torch.utils.data.TensorDataset(eval_coeffs)

    # Setting the number of workers for the DataLoader
    num_workers = 4

    # Creating DataLoaders
    # Training tensor
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    infinite_train_dataloader = (
        elem for it in iter(lambda: train_dataloader, None) for elem in it
    )

    # Evaluation tensor
    evaluation_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    infinite_evaluation_dataloader = (
        elem for it in iter(lambda: evaluation_dataloader, None) for elem in it
    )

    # Size of the data after preprocessing
    data_size = train_coeffs.size(-1)

    return infinite_train_dataloader, infinite_evaluation_dataloader, data_size
