import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn


# It's a Gaussian distribution with a fixed mean and a learnable standard deviation
class IsoGaussianModel(nn.Module):
    def __init__(self, size=1):
        """
        :param size: the number of parameters to be learned, defaults to 1 (optional)
        """
        super().__init__()
        self.sigma = nn.Parameter(torch.rand(size), requires_grad=True)

        self.size = size

    def sample(self, mean, n_samples=(200,)):
        """
        "Sample from a Gaussian distribution with mean and standard deviation given by the input parameters."

        :param mean: The mean of the distribution passed from a mean estimator
        :param n_samples: number of samples to draw from the distribution, defaults to 200 (optional)
        :return: Samples from the distribution
        """
        z = torch.randn((*n_samples, self.size))

        return mean.unsqueeze(1) + z * self.sigma

    def forward(self, mean, target, n_samples=(200,)):
        """
        It takes the mean of the distribution, the target value, and the number of samples to draw from the distribution
        It then draws samples from the distribution, and returns the squared error between the target and the closest
        sample

        :param mean: The mean of the distribution passed from a mean estimator
        :param target: the target value
        :param n_samples: number of samples to draw from the distribution, defaults to 200 (optional)
        :return: The minimum squared error between the target and the samples (minMPJPE).
        """
        samples = self.sample(mean, n_samples)
        samples = samples.view(*n_samples, -1, 3)

        return (
            torch.pow(target.unsqueeze(1) - samples, 2).sum(-1).mean(-1).min(1).values
        )

    def log_prob(self, mean, target):
        """
        It takes the mean of the distribution, and the target value, and returns the log probability of the target
        under the distribution

        :param mean: The mean of the distribution passed from a mean estimator
        :param target: the target value
        :return: The log probability of the target under the distribution
        """
        return D.Normal(mean, self.sigma).log_prob(target).sum(-1)
