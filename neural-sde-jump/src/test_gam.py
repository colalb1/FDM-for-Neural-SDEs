import fire  # noqa: F401
import polars as pl
import torch
import torch.optim as optim
from tqdm import tqdm

from gan.generators import Generator
from utils.data_helper_functions import preprocess_time_series_data
from utils.objective_functions import unbiased_pairwise_score_estimator
from utils.plotting_helper_functions import plot_distributions


class RunFDM:
    """A class that implements the Finite Dimensional Matching (FDM) algorithm."""

    def __init__(
        self,
        file_path: str,
        batch_size: int = 32,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,  # For optimizer
        swa_step_start: int = 5000,  # When to start stochastic weight averaging
    ):
        """
        Initialize the RunFDM class with the specified parameters.

        Args:
            file_path (str): The path to the file containing the time series data.
            sequence_length (int): The length of the sequence.
            batch_size (int, optional): The batch size for training. Defaults to 512.
            learning_rate (float, optional): The learning rate for the generator optimizer. Defaults to 2e-4.
            weight_decay (float, optional): The weight decay for the generator optimizer. Defaults to 0.01.
            swa_step_start (int, optional): The step at which to start stochastic weight averaging. Defaults to 5000.
        """

        # CPU is WAY faster than MPS since there is no issue with data transferring
        self.device = "cpu"
        self.file_path = file_path

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.swa_step_start = swa_step_start

        temp_data = pl.read_csv(self.file_path)
        temp_data = self.__ensure_numeric_columns(
            temp_data
        )  # Checking for all numeric columns
        self.data = temp_data.to_numpy()
        self.training_loader, self.evaluation_loader, self.data_size = (
            preprocess_time_series_data(
                time_series_data=torch.Tensor(self.data),
                batch_size=self.batch_size,
            )
        )

        data_length = 64
        self.time_grid = torch.linspace(
            0, data_length - 1, data_length, device=self.device
        )

        # Initializing Generator
        self.__initialize_generator()

    # Generator Initializer
    def __initialize_generator(self):
        """Initialize the generator model."""
        self.generator = Generator(
            data_size=self.data_size,
            initial_noise_size=5,
            noise_size=3,
            hidden_size=16,
            mlp_size=16,
            num_layers=1,
        ).to(self.device)

        self.averaged_generator = optim.swa_utils.AveragedModel(self.generator)
        self.generator_optimizer = optim.Adadelta(
            self.generator.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def __ensure_numeric_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Ensure all columns in the DataFrame are numeric. Raise an error if any column cannot be converted to numeric type.

        Args:
            df (pl.DataFrame): Input DataFrame.

        Returns:
            pl.DataFrame: DataFrame with all numeric columns.
        """
        try:
            df = df.select(pl.all().cast(pl.Float64))
        except Exception as e:
            raise ValueError(
                "All columns must be numeric and convertible to Float64."
            ) from e

        # Verify conversion
        for column in df.columns:
            if not pl.Float64 == df[column].dtype:
                raise ValueError(f"Column {column} could not be converted to Float64.")

        return df

    def finite_dimensional_matching(
        self,
        num_iterations: int,
    ) -> None:
        """
        Finite Dimensional Matching (FDM) algorithm. This is the algorithm of interest.

        Args:
            num_iterations (int): The number of iterations to run the algorithm.
        """
        # Compile the function locally
        __compiled_unbiased_pairwise_score_estimator = torch.compile(
            unbiased_pairwise_score_estimator.__wrapped__
        )
        # Creating range
        trange = tqdm(range(num_iterations), position=0, leave=True)

        for step in trange:
            # Generate a batch of simulated paths
            time_steps = self.time_grid
            simulated_paths = self.generator(
                time_steps=time_steps, batch_size=self.batch_size
            )

            # Randomly sample a batch of empirical paths
            (empirical_paths,) = next(iter(self.training_loader))

            # Reshaping empirical paths to have same size as generated paths (essentially making time_steps copies)
            empirical_paths = empirical_paths.unsqueeze(1).repeat(
                1, time_steps.size(0), 1
            )

            # Compute the score
            score = __compiled_unbiased_pairwise_score_estimator(
                generated_paths=simulated_paths,
                empirical_paths=empirical_paths,
                batch_size=self.batch_size,
            )

            # Update Generator
            generator_loss = -score  # We want to maximize the score
            generator_loss.backward()
            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad(set_to_none=True)

            # Stochastic weighted averaging
            if step > self.swa_step_start:
                self.averaged_generator.update_parameters(self.generator)

            # Update generator
            self.generator.load_state_dict(self.averaged_generator.module.state_dict())

        plot_distributions(
            self.time_grid,
            self.generator,
            self.evaluation_loader,
            self.batch_size,
            (0.1, 0.3, 0.5, 0.7, 0.9),
        )


def main():
    fdm_instance = RunFDM(file_path="data/brent.csv")
    fdm_instance.finite_dimensional_matching(num_iterations=6_000)


if __name__ == "__main__":
    fire.Fire(main)
