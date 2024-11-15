# Finite Dimensional Matching for Neural SDEs

This is a work in progress to reproduce [this paper](https://arxiv.org/abs/2410.03973). Said paper is based off of [this paper](https://arxiv.org/abs/2102.03657) whose code is in [this repo](https://github.com/google-research/torchsde/blob/master/examples/sde_gan.py). 

The code from the original paper is proprietary (at the time of writing) and everything that is not in the stated repo was written by me. See the first paper for technical background; I have halted this project indefinitely.

I worked on this for a while because I wanted to learn more about neural nets and how to use [PyTorch](https://pytorch.org/).

Run the system with the `src/test_gam.py` file. It is set up to run on a CPU (not ideal but can be reconfigured for CUDA easily by changing the device).
