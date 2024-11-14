import matplotlib.pyplot as plt
import torch
import torchcde


def plot_distributions(
    time_steps: torch.Tensor,
    generator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_plot_samples: int,
    plot_locs: tuple[float],
) -> None:
    """
    Plot histograms and samples from real and generated distributions.

    Args:
        time_steps (Tensor): The time steps at which to evaluate the samples.
        generator (torch.nn.Module): The generator model.
        dataloader (DataLoader): The dataloader for real samples.
        num_plot_samples (int): The number of samples to plot.
        plot_locs (List[float]): The locations in the sequence at which to plot the histograms.

    Returns:
        None
    """

    # Get real samples
    (real_samples,) = next(iter(dataloader))

    assert num_plot_samples <= real_samples.size(0)

    real_samples = torchcde.LinearInterpolation(real_samples).evaluate(time_steps)
    # real_samples = real_samples[..., 1]

    # Generate samples using the generator
    with torch.no_grad():
        generated_samples = generator(time_steps, real_samples.size(0))
    generated_samples = torchcde.LinearInterpolation(generated_samples).evaluate(
        time_steps
    )
    # generated_samples = generated_samples[..., 1]

    # Plot histograms of the marginal distributions
    for proportion in plot_locs:
        time_index = int(proportion * (real_samples.size(0) - 1))
        real_samples_at_time = real_samples[time_index]
        generated_samples_at_time = generated_samples[time_index]
        _, bins, _ = plt.hist(
            real_samples_at_time.numpy(),
            bins=32,
            alpha=0.7,
            label="Real",
            color="dodgerblue",
            density=True,
        )
        bin_width = bins[1] - bins[0]
        num_bins = int(
            (generated_samples_at_time.max() - generated_samples_at_time.min()).item()
            // bin_width
        )
        plt.hist(
            generated_samples_at_time.numpy(),
            bins=num_bins,
            alpha=0.7,
            label="Generated",
            color="crimson",
            density=True,
        )
        plt.legend()
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Marginal distribution at time {time_index}.")
        plt.tight_layout()
        plt.show()

    # Plot real and generated samples
    real_samples = real_samples[:num_plot_samples]
    generated_samples = generated_samples[:num_plot_samples]

    real_first = True
    generated_first = True
    for real_sample in real_samples:
        kwargs = {"label": "Real"} if real_first else {}
        plt.plot(
            time_steps,
            real_sample,
            color="dodgerblue",
            linewidth=0.5,
            alpha=0.7,
            **kwargs,
        )
        real_first = False
    for generated_sample in generated_samples:
        kwargs = {"label": "Generated"} if generated_first else {}
        plt.plot(
            time_steps,
            generated_sample,
            color="crimson",
            linewidth=0.5,
            alpha=0.7,
            **kwargs,
        )
        generated_first = False
    plt.legend()
    plt.title(f"{num_plot_samples} samples from both real and generated distributions.")
    plt.tight_layout()
    plt.show()
