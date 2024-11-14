# I stole nearly all of the code in this from https://github.com/issaz/sigker-nsdes/tree/main/src/gan
# which is essentially the same as https://github.com/google-research/torchsde/blob/master/examples/sde_gan.py#L424.
# Figured it would be faster to copy someone's code who is smarter than me and go from there.

import torch
import torchcde
import torchsde

from gan.mlp import MLP


class GeneratorFunc(torch.nn.Module):
    """
    Generator function for Stochastic Differential Equations (SDEs) with drift and diffusion components.
    """

    sde_type = "stratonovich"
    noise_type = "general"

    def __init__(
        self, noise_size: int, hidden_size: int, mlp_size: int, num_layers: int
    ):
        """
        Initializes the GeneratorFunc.

        Args:
            noise_size (int): Size of the noise input.
            hidden_size (int): Size of the hidden layers.
            mlp_size (int): Size of the MLP layers.
            num_layers (int): Number of MLP layers.
        """
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size

        # Drift and diffusion are MLPs of the same size
        # Note the final tanh nonlinearity to constrain the rate of change of the hidden state
        self._drift = MLP(
            1 + hidden_size, hidden_size, mlp_size, num_layers, use_tanh=True
        )
        self._diffusion = MLP(
            1 + hidden_size,
            hidden_size * noise_size,
            mlp_size,
            num_layers,
            use_tanh=True,
        )

    def f_and_g(self, time: torch.Tensor, hidden_state: torch.Tensor):
        """
        Computes the drift and diffusion components.

        Args:
            time (torch.Tensor): Current time step.
            hidden_state (torch.Tensor): Current hidden state.

        Returns:
            tuple: Drift and diffusion components.
        """
        # Expand time to match the batch size and concatenate with hidden state
        time_expanded = time.expand(hidden_state.size(0), 1)
        time_hidden = torch.cat([time_expanded, hidden_state], dim=1)

        # Compute drift and diffusion
        drift = self._drift(time_hidden)
        diffusion = self._diffusion(time_hidden).view(
            hidden_state.size(0), self._hidden_size, self._noise_size
        )

        return drift, diffusion


class Generator(torch.nn.Module):
    """
    SDE Generator that wraps the GeneratorFunc to compute the SDE.
    """

    def __init__(
        self,
        data_size: int,
        initial_noise_size: int,
        noise_size: int,
        hidden_size: int,
        mlp_size: int,
        num_layers: int,
    ):
        """
        Initializes the Generator.

        Args:
            data_size (int): Size of the data output.
            initial_noise_size (int): Size of the initial noise input.
            noise_size (int): Size of the noise input.
            hidden_size (int): Size of the hidden layers.
            mlp_size (int): Size of the MLP layers.
            num_layers (int): Number of MLP layers.
        """
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size

        # MLP for initial noise to hidden state
        self._initial = MLP(
            initial_noise_size, hidden_size, mlp_size, num_layers, use_tanh=False
        )

        # Generator function for SDE
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers)

        # Linear layer for final readout
        self._readout = torch.nn.Linear(hidden_size, data_size)

    def forward(self, time_steps: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Forward pass to generate data by solving the SDE.

        Args:
            time_steps (torch.Tensor): Time steps to evaluate the SDE at.
            batch_size (int): Batch size for the data generation.

        Returns:
            torch.Tensor: Generated data.
        """
        # Generate initial noise and convert to initial hidden state
        initial_noise = torch.randn(
            batch_size, self._initial_noise_size, device=time_steps.device
        )
        initial_hidden_state = self._initial(initial_noise)

        # Solve the SDE using the reversible Heun method
        hidden_states = torchsde.sdeint_adjoint(
            self._func,
            initial_hidden_state,
            time_steps,
            method="reversible_heun",
            dt=1.0,
            adjoint_method="adjoint_reversible_heun",
        )
        hidden_states = hidden_states.transpose(0, 1)

        # Readout to obtain the data
        data = self._readout(hidden_states)

        return torchcde.linear_interpolation_coeffs(data)
