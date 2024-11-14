# I stole nearly all of the code in this from https://github.com/issaz/sigker-nsdes/tree/main/src/gan
# which is essentially the same as https://github.com/google-research/torchsde/blob/master/examples/sde_gan.py#L424.
# Figured it would be faster to copy someone's code who is smarter than me and go from there.

import torch
import torch.nn.functional as F


class LipSwish(torch.nn.Module):
    """
    Custom activation function module that applies a scaled SiLU (Sigmoid Linear Unit) activation.
    """

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the scaled SiLU activation function.

        Args:
            input_tensor (torch.Tensor): Input tensor to the activation function.

        Returns:
            torch.Tensor: Output tensor after applying the scaled SiLU activation.
        """
        # Apply the SiLU activation function and scale the result by 0.909
        return 0.909 * F.silu(input_tensor)


class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) with customizable number of layers and activation functions.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        use_tanh: bool,
    ):
        """
        Initializes the MLP.

        Args:
            input_size (int): Size of the input layer.
            output_size (int): Size of the output layer.
            hidden_size (int): Size of the hidden layers.
            num_layers (int): Number of hidden layers.
            use_tanh (bool): Whether to use Tanh activation function at the output layer.
        """
        super().__init__()

        # Create a list to hold the layers of the model
        layers = [torch.nn.Linear(input_size, hidden_size), LipSwish()]

        # Add hidden layers with LipSwish activation functions
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(LipSwish())

        # Add the output layer
        layers.append(torch.nn.Linear(hidden_size, output_size))

        # Optionally add Tanh activation function at the output layer
        if use_tanh:
            layers.append(torch.nn.Tanh())

        # Create a sequential container with the layers
        self._model = torch.nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            input_tensor (torch.Tensor): Input tensor to the MLP.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP.
        """
        return self._model(input_tensor)
