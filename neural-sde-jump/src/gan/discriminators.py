# I stole nearly all of the code in this from https://github.com/issaz/sigker-nsdes/tree/main/src/gan
# which is essentially the same as https://github.com/google-research/torchsde/blob/master/examples/sde_gan.py#L424.
# Figured it would be faster to copy someone's code who is smarter than me and go from there.

import torch
import torchcde

from src.gan.generators import MLP


class DiscriminatorFunc(torch.nn.Module):
    """
    Discriminator function for Controlled Differential Equations (CDEs) with hidden and data components.
    """

    def __init__(
        self, data_size: int, hidden_size: int, mlp_size: int, num_layers: int
    ):
        """
        Initializes the DiscriminatorFunc.

        Args:
            data_size (int): Size of the data input.
            hidden_size (int): Size of the hidden layers.
            mlp_size (int): Size of the MLP layers.
            num_layers (int): Number of MLP layers.
        """
        super().__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size

        # MLP with tanh nonlinearity, important for model performance
        self._module = MLP(
            1 + hidden_size,
            hidden_size * (1 + data_size),
            mlp_size,
            num_layers,
            use_tanh=True,
        )

    def forward(self, time: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the discriminator function.

        Args:
            time (torch.Tensor): Current time step.
            hidden_state (torch.Tensor): Current hidden state.

        Returns:
            torch.Tensor: Output tensor after applying the MLP.
        """
        # Expand time to match the batch size and concatenate with hidden state
        time_expanded = time.expand(hidden_state.size(0), 1)
        time_hidden = torch.cat([time_expanded, hidden_state], dim=1)

        # Apply the MLP and reshape the output
        return self._module(time_hidden).view(
            hidden_state.size(0), self._hidden_size, 1 + self._data_size
        )


class Discriminator(torch.nn.Module):
    """
    Discriminator using a Neural Controlled Differential Equation (Neural CDE).
    """

    def __init__(
        self, data_size: int, hidden_size: int, mlp_size: int, num_layers: int
    ):
        """
        Initializes the Discriminator.

        Args:
            data_size (int): Size of the data input.
            hidden_size (int): Size of the hidden layers.
            mlp_size (int): Size of the MLP layers.
            num_layers (int): Number of MLP layers.
        """
        super().__init__()

        # MLP for initial data to hidden state
        self._initial = MLP(
            1 + data_size, hidden_size, mlp_size, num_layers, use_tanh=False
        )

        # Discriminator function for CDE
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)

        # Linear layer for final readout
        self._readout = torch.nn.Linear(hidden_size, 1)

    def forward(self, data_coeffs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the discriminator score.

        Args:
            data_coeffs (torch.Tensor): Coefficients of the data for interpolation.

        Returns:
            torch.Tensor: Mean discriminator score.
        """
        # data_coeffs has shape (batch_size, time_size, 1 + data_size)
        # The +1 corresponds to time. When solving CDEs, it is most natural to treat time as just another
        # channel: this makes handling irregular data quite easy, when the times may be different between
        # different samples in the batch.

        # Perform linear interpolation on the data coefficients
        interpolated_data = torchcde.LinearInterpolation(data_coeffs)

        # Evaluate the initial data point
        initial_data = interpolated_data.evaluate(interpolated_data.interval[0])

        # Convert initial data to initial hidden state
        initial_hidden_state = self._initial(initial_data)

        # Solve the CDE using the reversible Heun method
        hidden_states = torchcde.cdeint(
            interpolated_data,
            self._func,
            initial_hidden_state,
            interpolated_data.interval,
            method="reversible_heun",
            backend="torchsde",
            dt=1.0,
            adjoint_method="adjoint_reversible_heun",
            adjoint_params=(data_coeffs,) + tuple(self._func.parameters()),
        )

        # Compute the final score using the readout layer
        final_score = self._readout(hidden_states[:, -1])

        # Return the mean score
        return final_score.mean()
